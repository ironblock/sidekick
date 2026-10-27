//! The one ANE chat lane (docs/design/open-llm.md, "Memory budget"):
//! Foundation Models (`apple-fm`) and the open chat models take turns,
//! one request at a time, in arrival order, and one open model is resident
//! at a time.
//!
//! #187 measured why: the Neural Engine's budget is the whole machine's,
//! and Foundation Models' model counts against it. An `apple-fm` request
//! overlapping a generation slowed both about 25×, and two resident
//! decoders pass the budget. Idle programs cost nothing, so the lane
//! schedules rather than evicts: a model stays resident until another open
//! model needs the lane's slot or it has been idle for `model_idle_ttl`.

use crate::decoder::{DecoderProbe, DecoderSpec};
use crate::engine::{Engine, LoadState};
use serde_json::{json, Value};
use sidekick_core::{Availability, ChatBackend, ChatRequest, ChatResponse, DeltaSink, Error, ModelInfo, Result, WorkHold};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::time::{Duration, Instant};
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

/// `Retry-After` on a full queue, in seconds: about one short reply.
pub const BUSY_RETRY_AFTER_SECS: u64 = 5;

/// How long past its deadline [`LanedChat`] waits for a backend that
/// should have timed itself out.
const DEADLINE_GRACE: Duration = Duration::from_secs(1);

/// The default `queue_depth`.
pub const DEFAULT_QUEUE_DEPTH: usize = 4;

/// The default `ane_budget_mib`: 2.75 GiB, under the 2.79 GiB of active
/// programs #187 measured safe on an M1 Max.
pub const DEFAULT_ANE_BUDGET_MIB: u64 = 2816;

/// Reads the process's Neural Engine footprint, in bytes, where the
/// platform reports it.
pub type FootprintReader = Arc<dyn Fn() -> Option<u64> + Send + Sync>;

/// The lane's settings.
#[derive(Clone)]
pub struct LaneOptions {
    /// Requests that may wait while one runs; past it, 503 `backend_busy`.
    pub queue_depth: usize,
    /// How long an open model stays resident after its last request.
    pub idle_ttl: Duration,
    /// A load that would take the process past this many MiB on the
    /// Neural Engine logs a warning; nothing is refused or evicted.
    pub ane_budget_mib: u64,
    pub footprint: FootprintReader,
}

impl Default for LaneOptions {
    fn default() -> Self {
        Self {
            queue_depth: DEFAULT_QUEUE_DEPTH,
            idle_ttl: Duration::from_secs(900),
            ane_budget_mib: DEFAULT_ANE_BUDGET_MIB,
            footprint: Arc::new(|| None),
        }
    }
}

/// The resident open model's decoder thread.
pub(crate) struct Resident {
    pub engine: Engine,
    pub last_used: Instant,
}

/// The lane: a fair queue of one, and the resident open model.
pub struct ChatLane {
    slot: Arc<Semaphore>,
    options: LaneOptions,
    waiting: AtomicUsize,
    /// The model whose request holds the slot.
    running: Mutex<Option<String>>,
    resident: Mutex<Option<Resident>>,
}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

/// A request's turn in the lane; the next request's begins when it drops.
pub struct LanePermit {
    _slot: OwnedSemaphorePermit,
    lane: Arc<ChatLane>,
}

impl Drop for LanePermit {
    fn drop(&mut self) {
        *lock(&self.lane.running) = None;
    }
}

/// Counts a waiting request for as long as it waits.
struct Waiting<'a>(&'a AtomicUsize);

impl Drop for Waiting<'_> {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::SeqCst);
    }
}

impl ChatLane {
    /// A lane, sweeping its idle resident model on a timer when there is a
    /// tokio runtime.
    pub fn new(options: LaneOptions) -> Arc<Self> {
        let lane = Arc::new(Self {
            slot: Arc::new(Semaphore::new(1)),
            options,
            waiting: AtomicUsize::new(0),
            running: Mutex::new(None),
            resident: Mutex::new(None),
        });
        if let Ok(runtime) = tokio::runtime::Handle::try_current() {
            let weak = Arc::downgrade(&lane);
            let every = (lane.options.idle_ttl / 4).clamp(Duration::from_millis(10), Duration::from_secs(30));
            runtime.spawn(async move {
                let mut tick = tokio::time::interval(every);
                tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
                tick.tick().await;
                loop {
                    tick.tick().await;
                    let Some(lane) = weak.upgrade() else { break };
                    lane.unload_idle();
                }
            });
        }
        lane
    }

    pub fn options(&self) -> &LaneOptions {
        &self.options
    }

    /// Wait for the lane, in arrival order, until `deadline`. A request
    /// past the queue's depth is refused at once (`backend_busy`); one
    /// still waiting at its deadline times out, `timeout` being the
    /// request's whole allowance.
    pub async fn acquire(self: &Arc<Self>, model: &str, deadline: tokio::time::Instant, timeout: Duration) -> Result<LanePermit> {
        let slot = match self.slot.clone().try_acquire_owned() {
            Ok(slot) => slot,
            Err(_) => {
                if self.waiting.fetch_add(1, Ordering::SeqCst) >= self.options.queue_depth {
                    self.waiting.fetch_sub(1, Ordering::SeqCst);
                    let running = lock(&self.running).clone().unwrap_or_default();
                    return Err(Error::Busy {
                        message: format!(
                            "`{model}` shares the Neural Engine's chat lane, which is generating for `{running}` with \
                             {} more waiting (queue_depth); retry",
                            self.options.queue_depth
                        ),
                        retry_after_secs: BUSY_RETRY_AFTER_SECS,
                    });
                }
                let _waiting = Waiting(&self.waiting);
                match tokio::time::timeout_at(deadline, self.slot.clone().acquire_owned()).await {
                    Ok(Ok(slot)) => slot,
                    Ok(Err(_)) => return Err(Error::Other("the chat lane closed".into())),
                    Err(_) => return Err(Error::Timeout { secs: timeout.as_secs() }),
                }
            }
        };
        *lock(&self.running) = Some(model.to_string());
        Ok(LanePermit { _slot: slot, lane: self.clone() })
    }

    /// The resident engine for `model`, starting it if needed. Another
    /// model's is stopped first, and its thread waited for, so that its
    /// programs have left the Neural Engine before this one's load is
    /// measured. Call with the lane's permit held, so that no other request
    /// is using the decoder being unloaded.
    pub(crate) async fn engine(&self, _permit: &LanePermit, model: &str, start: impl FnOnce() -> Result<Engine>) -> Result<Engine> {
        let old = {
            let mut resident = lock(&self.resident);
            if let Some(r) = resident.as_mut() {
                if r.engine.model == model {
                    r.last_used = Instant::now();
                    return Ok(r.engine.clone());
                }
            }
            resident.take()
        };
        if let Some(old) = old {
            tracing::info!(model = %old.engine.model, next = %model, "unloading the resident open chat model: one is resident at a time");
            unload(&old.engine).await;
        }
        let engine = start()?;
        *lock(&self.resident) = Some(Resident { engine: engine.clone(), last_used: Instant::now() });
        Ok(engine)
    }

    /// Mark `model` used now, as its request finishes.
    pub(crate) fn touch(&self, model: &str) {
        if let Some(r) = lock(&self.resident).as_mut().filter(|r| r.engine.model == model) {
            r.last_used = Instant::now();
        }
    }

    /// Unload `model` if it is resident: its load failed.
    pub(crate) fn forget(&self, model: &str) {
        let mut resident = lock(&self.resident);
        if resident.as_ref().is_some_and(|r| r.engine.model == model) {
            if let Some(r) = resident.take() {
                r.engine.stop();
            }
        }
    }

    /// The resident model and its engine, if any.
    pub(crate) fn resident(&self) -> Option<(Engine, Instant)> {
        lock(&self.resident).as_ref().map(|r| (r.engine.clone(), r.last_used))
    }

    /// Unload the resident model if it has been idle past the TTL and no
    /// request holds the lane. A model still loading isn't idle: its idle
    /// time starts when its load completes.
    pub fn unload_idle(&self) {
        let Ok(_slot) = self.slot.clone().try_acquire_owned() else { return };
        let mut resident = lock(&self.resident);
        let idle = resident
            .as_ref()
            .is_some_and(|r| !r.engine.loading() && r.last_used.elapsed() >= self.options.idle_ttl);
        if idle {
            let r = resident.take().expect("checked");
            r.engine.stop();
            tracing::info!(model = %r.engine.model, "unloaded idle open chat model");
        }
    }

    /// Unload the resident model now, whatever its idle time, unless a
    /// request holds the lane or it is still loading.
    #[cfg(test)]
    pub(crate) fn unload_idle_now(&self) {
        let Ok(_slot) = self.slot.clone().try_acquire_owned() else { return };
        let mut resident = lock(&self.resident);
        if resident.as_ref().is_some_and(|r| !r.engine.loading()) {
            let r = resident.take().expect("checked");
            r.engine.stop();
        }
    }

    /// The process's Neural Engine footprint, where it is reported.
    pub fn footprint(&self) -> Option<u64> {
        (self.options.footprint)()
    }

    /// Log a warning if loading `estimate` bytes more would take the
    /// process past the budget.
    pub(crate) fn check_budget(&self, model: &str, estimate: u64) {
        let budget = self.options.ane_budget_mib * 1024 * 1024;
        let now = self.footprint().unwrap_or(0);
        if now + estimate > budget {
            tracing::warn!(
                model = %model,
                ane_mib = now / (1 << 20),
                estimate_mib = estimate / (1 << 20),
                budget_mib = self.options.ane_budget_mib,
                "loading this model takes the daemon's Neural Engine programs past ane_budget_mib; once active \
                 programs pass the budget, they reload on nearly every call, about 20 times slower (docs/design/\
                 open-llm.md, \"Memory budget\")"
            );
        }
    }

    /// `/health`'s view of the lane.
    pub fn health(&self) -> Value {
        let resident = lock(&self.resident).as_ref().map(|r| {
            json!({
                "model": r.engine.model,
                "state": r.engine.state_name(),
                "idle_secs": r.last_used.elapsed().as_secs(),
            })
        });
        let footprint = self.footprint();
        let budget = self.options.ane_budget_mib * 1024 * 1024;
        json!({
            "queue_depth": self.options.queue_depth,
            "running": *lock(&self.running),
            "waiting": self.waiting.load(Ordering::SeqCst),
            "resident": resident,
            "idle_ttl_secs": self.options.idle_ttl.as_secs(),
            "ane": {
                // The whole process's Neural Engine programs, as the kernel
                // reports them (`ri_neural_footprint`): every loaded Core
                // ML model's, embedders and classifiers included.
                "process_bytes": footprint,
                "budget_mib": self.options.ane_budget_mib,
                "over_budget": footprint.map(|f| f > budget),
            },
        })
    }
}

/// A chat backend that takes the lane for each request: `apple-fm` when
/// open chat models are installed. Its wait counts against its timeout.
///
/// The inner backend times the request out itself, at the request's
/// `deadline`, so that its own cancel path runs; and the lane's turn rides
/// in the request's `hold`, which the backend keeps alive until its work
/// has really finished (Foundation Models' blocking thread outlives a
/// timeout). So an abandoned `apple-fm` request still holds the lane while
/// it runs, and no open model starts beside it.
pub struct LanedChat {
    inner: Arc<dyn ChatBackend>,
    lane: Arc<ChatLane>,
    timeout: Duration,
}

impl LanedChat {
    pub fn new(inner: Arc<dyn ChatBackend>, lane: Arc<ChatLane>, timeout: Duration) -> Self {
        Self { inner, lane, timeout }
    }

    /// Run `work` on `req` in the lane: its wait and its work bounded
    /// together by the timeout, the deadline passed to the backend.
    async fn laned<F>(&self, mut req: ChatRequest, work: impl FnOnce(ChatRequest) -> F) -> Result<ChatResponse>
    where
        F: std::future::Future<Output = Result<ChatResponse>>,
    {
        let deadline = tokio::time::Instant::now() + self.timeout;
        let permit = self.lane.acquire(self.inner.id(), deadline, self.timeout).await?;
        req.deadline = Some(deadline.into_std());
        req.hold = Some(WorkHold(Arc::new(permit)));
        // A backend that ignores the deadline is still bounded; one that
        // honors it times out first, through its own path.
        tokio::time::timeout_at(deadline + DEADLINE_GRACE, work(req))
            .await
            .unwrap_or(Err(Error::Timeout { secs: self.timeout.as_secs() }))
    }
}

#[async_trait::async_trait]
impl ChatBackend for LanedChat {
    fn id(&self) -> &str {
        self.inner.id()
    }

    fn context_limit(&self) -> Option<usize> {
        self.inner.context_limit()
    }

    async fn model_info(&self) -> Option<ModelInfo> {
        self.inner.model_info().await
    }

    async fn availability(&self) -> Availability {
        self.inner.availability().await
    }

    async fn complete(&self, req: ChatRequest) -> Result<ChatResponse> {
        self.laned(req, |req| self.inner.complete(req)).await
    }

    async fn complete_stream(&self, req: ChatRequest, sink: DeltaSink) -> Result<ChatResponse> {
        self.laned(req, |req| self.inner.complete_stream(req, sink)).await
    }
}

/// How long a model switch waits for the old decoder's thread to exit: a
/// call in progress finishes first (about 54 ms on the 2B build), and a
/// load in progress, which can't be stopped, completes.
const UNLOAD_WAIT: Duration = Duration::from_secs(120);

/// Stop `engine` and wait for its thread to exit, unless its decoder has
/// failed: a thread stuck in a hung call never exits.
async fn unload(engine: &Engine) {
    engine.stop();
    if engine.facts().is_some_and(|f| f.probe.status().failed.is_some()) {
        tracing::warn!(model = %engine.model, "not waiting for a failed decoder's thread to exit");
        return;
    }
    if !engine.exited(UNLOAD_WAIT).await {
        tracing::warn!(
            model = %engine.model,
            "the unloaded model's decoder thread hasn't exited after {}s; loading the next one anyway",
            UNLOAD_WAIT.as_secs()
        );
    }
}

/// What a request learns about its model's engine once it has loaded.
pub(crate) struct Ready {
    pub probe: Arc<dyn DecoderProbe>,
    pub spec: DecoderSpec,
    /// Whether this request waited on (part of) the load, and how long.
    pub waited: Option<Duration>,
}

/// Wait for `engine` to load, up to `timeout`.
pub(crate) async fn ready(engine: &Engine, timeout: Duration) -> std::result::Result<Ready, ReadyError> {
    let started = Instant::now();
    let mut rx = engine.loaded();
    let was_loading = matches!(*rx.borrow(), LoadState::Loading);
    let state = tokio::time::timeout(timeout, rx.wait_for(|s| !matches!(s, LoadState::Loading)))
        .await
        .map_err(|_| ReadyError::Timeout)?
        .map_err(|_| ReadyError::Failed("the decoder thread exited during its load".into()))?
        .clone();
    match state {
        LoadState::Ready(loaded) => Ok(Ready {
            probe: loaded.probe.clone(),
            spec: loaded.spec,
            waited: was_loading.then(|| started.elapsed()),
        }),
        LoadState::Failed(reason) => Err(ReadyError::Failed(reason)),
        LoadState::Loading => unreachable!("waited for it to change"),
    }
}

pub(crate) enum ReadyError {
    Timeout,
    Failed(String),
}
