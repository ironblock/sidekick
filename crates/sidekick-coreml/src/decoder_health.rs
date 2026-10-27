//! A stateful decoder's health, readable from any thread: whether it has
//! failed, how its calls are timed, and how many stalled. Also the per-call
//! watchdog that marks a decoder failed when an ANE prediction hangs.
//!
//! Platform-neutral, so the bookkeeping is tested everywhere; the decoder
//! that feeds it is macOS-only.

use std::collections::VecDeque;
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

/// How long one prediction may run before the decoder is marked failed:
/// 60 s, as #173's runtime used. An ANE request that hangs does so for
/// about 100 s, and can leave every ANE prediction on the machine failing
/// (docs/design/coreml-cache.md, "A hung ANE request").
pub const DEFAULT_CALL_TIMEOUT: Duration = Duration::from_secs(60);

/// A call slower than this many times the warm median counts as a stall:
/// another process's ANE work, or Foundation Models', pushing the active
/// programs past the ANE's budget (docs/design/open-llm.md, "Memory
/// budget").
pub const STALL_FACTOR: f64 = 5.0;

/// Calls after a load that aren't timed for the median: the first runs the
/// programs onto the ANE.
const WARMUP_CALLS: u64 = 2;

/// Timed calls needed before a call can count as a stall.
const MIN_TIMED_CALLS: usize = 8;

/// The median is taken over this many recent calls that weren't stalls.
const MEDIAN_WINDOW: usize = 64;

/// A snapshot of a decoder's health.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct DecoderHealth {
    /// Why the decoder can no longer serve, when it can't: a prediction that
    /// hung. Its thread can't be killed, so recovery is a restart.
    pub failed: Option<String>,
    /// T-token calls through the whole chain since the load.
    pub calls: u64,
    /// Calls slower than [`STALL_FACTOR`] times the warm median.
    pub stalls: u64,
    /// The median of recent calls that weren't stalls, once enough have run.
    pub warm_median_ms: Option<f64>,
    /// The latest stall's duration, and how long ago it ended.
    pub last_stall_ms: Option<f64>,
    pub since_last_stall: Option<Duration>,
    /// The KV state was replaced after a failed call or a non-finite logit.
    pub state_resets: u64,
}

#[derive(Debug, Default)]
struct Inner {
    failed: Option<String>,
    calls: u64,
    stalls: u64,
    window: VecDeque<f64>,
    last_stall: Option<(f64, Instant)>,
    state_resets: u64,
}

impl Inner {
    fn median(&self) -> Option<f64> {
        if self.window.len() < MIN_TIMED_CALLS {
            return None;
        }
        let mut sorted: Vec<f64> = self.window.iter().copied().collect();
        sorted.sort_by(|a, b| a.total_cmp(b));
        Some(sorted[sorted.len() / 2])
    }
}

/// A decoder's shared health record. Cloning shares it: the decoder's
/// thread writes it, and a server reads it for `/health` or to fail queued
/// requests while the decoder's thread is stuck in a hung call.
#[derive(Debug, Clone, Default)]
pub struct DecoderMonitor(Arc<Mutex<Inner>>);

impl DecoderMonitor {
    pub fn new() -> Self {
        Self::default()
    }

    fn lock(&self) -> MutexGuard<'_, Inner> {
        // A panic while holding the lock leaves plain counters behind;
        // they're still worth reading.
        self.0.lock().unwrap_or_else(|e| e.into_inner())
    }

    pub fn snapshot(&self) -> DecoderHealth {
        let inner = self.lock();
        DecoderHealth {
            failed: inner.failed.clone(),
            calls: inner.calls,
            stalls: inner.stalls,
            warm_median_ms: inner.median(),
            last_stall_ms: inner.last_stall.map(|(ms, _)| ms),
            since_last_stall: inner.last_stall.map(|(_, at)| at.elapsed()),
            state_resets: inner.state_resets,
        }
    }

    /// Why the decoder has failed, if it has.
    pub fn failed(&self) -> Option<String> {
        self.lock().failed.clone()
    }

    /// Mark the decoder failed; the first reason sticks.
    pub fn fail(&self, reason: String) {
        let mut inner = self.lock();
        if inner.failed.is_none() {
            inner.failed = Some(reason);
        }
    }

    /// Record one completed call through the chain. Returns whether it was a
    /// stall: slower than [`STALL_FACTOR`] times the warm median. A stall
    /// doesn't enter the median, so contention doesn't raise the bar.
    pub fn record_call(&self, elapsed: Duration) -> bool {
        let ms = elapsed.as_secs_f64() * 1e3;
        let mut inner = self.lock();
        inner.calls += 1;
        if inner.calls <= WARMUP_CALLS {
            return false;
        }
        if inner
            .median()
            .is_some_and(|median| ms > STALL_FACTOR * median)
        {
            inner.stalls += 1;
            inner.last_stall = Some((ms, Instant::now()));
            return true;
        }
        inner.window.push_back(ms);
        if inner.window.len() > MEDIAN_WINDOW {
            inner.window.pop_front();
        }
        false
    }

    pub fn record_state_reset(&self) {
        self.lock().state_resets += 1;
    }
}

#[derive(Debug, Default)]
struct WatchState {
    /// The running call: when it started, and what it is.
    armed: Option<(Instant, String)>,
    stop: bool,
}

#[derive(Debug, Default)]
struct WatchShared {
    state: Mutex<WatchState>,
    wake: Condvar,
}

/// Times each prediction from a thread of its own. A call that runs past
/// the timeout marks the monitor failed: the predicting thread is stuck in
/// Core ML and can't be interrupted, but everyone else learns at once.
#[derive(Debug)]
pub struct Watchdog {
    shared: Arc<WatchShared>,
    thread: Option<JoinHandle<()>>,
}

impl Watchdog {
    /// Start the watchdog's thread; `Err` when the thread can't be spawned.
    pub fn start(timeout: Duration, monitor: DecoderMonitor) -> std::io::Result<Self> {
        let shared = Arc::new(WatchShared::default());
        let watched = Arc::clone(&shared);
        let thread = std::thread::Builder::new()
            .name("sidekick-decoder-watchdog".into())
            .spawn(move || watch(&watched, timeout, &monitor))?;
        Ok(Self {
            shared,
            thread: Some(thread),
        })
    }

    fn lock(&self) -> MutexGuard<'_, WatchState> {
        self.shared.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// A call is starting; `what` names it in the failure.
    pub fn arm(&self, what: String) {
        self.lock().armed = Some((Instant::now(), what));
        self.shared.wake.notify_one();
    }

    /// The call returned.
    pub fn disarm(&self) {
        self.lock().armed = None;
    }
}

impl Drop for Watchdog {
    fn drop(&mut self) {
        self.lock().stop = true;
        self.shared.wake.notify_one();
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

fn watch(shared: &WatchShared, timeout: Duration, monitor: &DecoderMonitor) {
    let mut state = shared.state.lock().unwrap_or_else(|e| e.into_inner());
    loop {
        if state.stop {
            return;
        }
        let Some((started, _)) = &state.armed else {
            state = shared.wake.wait(state).unwrap_or_else(|e| e.into_inner());
            continue;
        };
        let elapsed = started.elapsed();
        if elapsed >= timeout {
            let (_, what) = state.armed.take().expect("armed");
            let reason = format!(
                "{what} ran past {}s, and an ANE prediction can't be cancelled; restart sidekickd to recover",
                timeout.as_secs_f64()
            );
            tracing::error!("decoder prediction hung: {reason}");
            monitor.fail(reason);
            continue;
        }
        state = shared
            .wake
            .wait_timeout(state, timeout - elapsed)
            .unwrap_or_else(|e| e.into_inner())
            .0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ms(n: u64) -> Duration {
        Duration::from_millis(n)
    }

    #[test]
    fn stalls_are_calls_past_five_times_the_warm_median() {
        let m = DecoderMonitor::new();
        // The first calls load the programs: slow, and not timed.
        assert!(!m.record_call(ms(5000)));
        assert!(!m.record_call(ms(900)));
        for _ in 0..MIN_TIMED_CALLS {
            assert!(!m.record_call(ms(54)));
        }
        assert_eq!(m.snapshot().warm_median_ms, Some(54.0));
        assert!(!m.record_call(ms(250)), "4.6x isn't a stall");
        assert!(m.record_call(ms(1200)));
        let h = m.snapshot();
        assert_eq!(
            (h.calls, h.stalls, h.last_stall_ms),
            (2 + MIN_TIMED_CALLS as u64 + 2, 1, Some(1200.0))
        );
        assert!(h.since_last_stall.is_some());
        // Stalls stay out of the median, so a run of them keeps counting.
        for _ in 0..20 {
            assert!(m.record_call(ms(1200)));
        }
        assert_eq!(m.snapshot().warm_median_ms, Some(54.0));
        assert_eq!(m.snapshot().stalls, 21);
    }

    #[test]
    fn no_stall_before_enough_calls_are_timed() {
        let m = DecoderMonitor::new();
        for _ in 0..WARMUP_CALLS {
            m.record_call(ms(1));
        }
        for _ in 0..MIN_TIMED_CALLS - 1 {
            assert!(!m.record_call(ms(1)));
        }
        assert_eq!(m.snapshot().warm_median_ms, None);
        assert!(!m.record_call(ms(1000)));
    }

    #[test]
    fn the_median_follows_recent_calls() {
        let m = DecoderMonitor::new();
        for _ in 0..WARMUP_CALLS + MEDIAN_WINDOW as u64 {
            m.record_call(ms(10));
        }
        // Slower, but never 5x the median at the time: the window moves.
        for _ in 0..MEDIAN_WINDOW {
            assert!(!m.record_call(ms(30)));
        }
        assert_eq!(m.snapshot().warm_median_ms, Some(30.0));
    }

    #[test]
    fn the_first_failure_sticks() {
        let m = DecoderMonitor::new();
        assert_eq!(m.failed(), None);
        m.fail("first".into());
        m.fail("second".into());
        assert_eq!(m.failed().as_deref(), Some("first"));
    }

    #[test]
    fn the_watchdog_fails_a_call_that_runs_past_the_timeout() {
        let m = DecoderMonitor::new();
        // A 2 s timeout: calls here take 10 ms, so even a slow CI runner
        // leaves well over a second's margin.
        let w = Watchdog::start(Duration::from_secs(2), m.clone()).unwrap();
        // Calls that return in time leave it alone.
        for _ in 0..5 {
            w.arm("a quick call".into());
            std::thread::sleep(ms(10));
            w.disarm();
        }
        assert_eq!(m.failed(), None);
        w.arm("chunk 2 at position 48".into());
        let deadline = Instant::now() + Duration::from_secs(10);
        while m.failed().is_none() && Instant::now() < deadline {
            std::thread::sleep(ms(10));
        }
        let reason = m.failed().expect("the watchdog fired");
        assert!(
            reason.contains("chunk 2 at position 48") && reason.contains("past 2s"),
            "{reason}"
        );
        w.disarm();
        drop(w);
    }

    #[test]
    fn dropping_an_idle_watchdog_returns() {
        let w = Watchdog::start(Duration::from_secs(3600), DecoderMonitor::new()).unwrap();
        w.arm("never finishes".into());
        drop(w);
    }
}
