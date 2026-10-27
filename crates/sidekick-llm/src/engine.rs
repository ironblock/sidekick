//! A loaded open model's decoder thread (docs/design/open-llm.md,
//! "Concurrency and state"). A dedicated thread, not tokio's blocking pool:
//! the decoder's Core ML state isn't documented as thread-safe, and its
//! buffers stay with the thread. It loads the decoder, then runs one job at
//! a time from its channel, and exits (dropping the decoder) when told to
//! stop or when every handle to it is gone; it says when it has exited, so
//! that the next model loads only once this one's programs are released.

use crate::decoder::{Decoder, DecoderProbe, DecoderSpec};
use crate::generate::{generate, Generated, Job};
use crate::prompt::Prompter;
use sidekick_core::Result;
use std::sync::mpsc;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{oneshot, watch};

/// Loads a model's decoder, on its thread.
pub type DecoderLoader = Arc<dyn Fn() -> Result<Box<dyn Decoder>> + Send + Sync>;

/// A loaded decoder's facts.
pub(crate) struct Loaded {
    pub probe: Arc<dyn DecoderProbe>,
    pub spec: DecoderSpec,
    pub ane_bytes: Option<u64>,
    pub load: Duration,
}

#[derive(Clone)]
pub(crate) enum LoadState {
    Loading,
    Ready(Arc<Loaded>),
    Failed(String),
}

pub(crate) enum Message {
    Work(Work),
    /// Drop the decoder and exit, after any job before it.
    Stop,
}

pub(crate) struct Work {
    pub job: Job,
    pub reply: oneshot::Sender<Result<Generated>>,
}

/// A handle to a decoder thread. Clones share the thread.
#[derive(Clone)]
pub(crate) struct Engine {
    pub model: String,
    jobs: mpsc::Sender<Message>,
    loaded: watch::Receiver<LoadState>,
    /// True once the thread has exited and dropped its decoder.
    exited: watch::Receiver<bool>,
}

impl Engine {
    /// Start `model`'s thread, which loads its decoder with `loader`.
    pub fn start(model: &str, loader: DecoderLoader, prompter: Arc<Prompter>, stop_ids: Arc<[u32]>) -> Result<Self> {
        let (jobs, rx) = mpsc::channel::<Message>();
        let (state, loaded) = watch::channel(LoadState::Loading);
        let (exit, exited) = watch::channel(false);
        let name = model.to_string();
        std::thread::Builder::new()
            .name(format!("sidekick-llm-{model}"))
            .spawn(move || {
                // Sent when this closure returns, the decoder dropped.
                struct Exit(watch::Sender<bool>);
                impl Drop for Exit {
                    fn drop(&mut self) {
                        let _ = self.0.send(true);
                    }
                }
                let _exit = Exit(exit);
                let started = Instant::now();
                let mut decoder = match loader() {
                    Ok(d) => d,
                    Err(e) => {
                        tracing::warn!(model = %name, "open chat model failed to load: {e}");
                        let _ = state.send(LoadState::Failed(e.to_string()));
                        return;
                    }
                };
                let load = started.elapsed();
                let loaded = Loaded { probe: decoder.probe(), spec: decoder.spec(), ane_bytes: decoder.ane_bytes(), load };
                tracing::info!(
                    model = %name,
                    load_ms = load.as_millis() as u64,
                    ane_mib = loaded.ane_bytes.map(|b| b / (1 << 20)),
                    "open chat model loaded"
                );
                let _ = state.send(LoadState::Ready(Arc::new(loaded)));
                while let Ok(Message::Work(Work { mut job, reply })) = rx.recv() {
                    let result = generate(decoder.as_mut(), &prompter, &stop_ids, &mut job);
                    let _ = reply.send(result);
                }
                drop(decoder);
                tracing::debug!(model = %name, "decoder thread exits: its model was unloaded");
            })
            .map_err(|e| sidekick_core::Error::Other(format!("can't start a decoder thread: {e}")))?;
        Ok(Self { model: model.to_string(), jobs, loaded, exited })
    }

    /// Tell the thread to drop its decoder and exit, after any job queued.
    pub fn stop(&self) {
        let _ = self.jobs.send(Message::Stop);
    }

    /// Wait up to `limit` for the thread to have exited. A thread stuck in a
    /// hung call never does; the caller goes on without it.
    pub async fn exited(&self, limit: Duration) -> bool {
        let mut rx = self.exited.clone();
        let exited = matches!(tokio::time::timeout(limit, rx.wait_for(|e| *e)).await, Ok(Ok(_)));
        exited
    }

    /// Whether the load is still running.
    pub fn loading(&self) -> bool {
        matches!(*self.loaded.borrow(), LoadState::Loading)
    }


    pub fn loaded(&self) -> watch::Receiver<LoadState> {
        self.loaded.clone()
    }

    /// The load's facts once it is done.
    pub fn facts(&self) -> Option<Arc<Loaded>> {
        match &*self.loaded.borrow() {
            LoadState::Ready(l) => Some(l.clone()),
            _ => None,
        }
    }

    pub fn state_name(&self) -> &'static str {
        match &*self.loaded.borrow() {
            LoadState::Loading => "loading",
            LoadState::Ready(l) if l.probe.status().failed.is_some() => "failed",
            LoadState::Ready(_) => "ready",
            LoadState::Failed(_) => "load_failed",
        }
    }

    /// Queue a job; the thread runs it after any before it.
    pub fn submit(&self, work: Work) -> bool {
        self.jobs.send(Message::Work(work)).is_ok()
    }
}
