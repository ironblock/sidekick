//! Request deadlines that leave model loading out.
//!
//! `request_timeout_secs` bounds a request's work, not the loads it waits
//! on. A model's first load, and each bucket's, compiles a program for the
//! Mac's ANE or GPU: seconds for a small bucket, minutes for a large one
//! (about 4 minutes for a 2,048-token bucket), so a deadline that also
//! covered loading turned a first request into a 504 while the load
//! finished in the background. A request now runs in two phases:
//! 1. loading: the model (the pool loads it unless it's resident), its
//!    inputs prepared, and the buckets they need loaded, bounded only by
//!    `load_timeout_secs`, which is far longer than any measured compile. A
//!    request waiting on another request's load of the same bucket waits
//!    the same way.
//! 2. prediction: bounded by `request_timeout_secs`, from the end of the
//!    loading phase when anything loaded, otherwise from the request's
//!    arrival, as before. A hung prediction still times out.
//!
//! Either timeout abandons the wait, not the work: a load still finishes and
//! becomes resident, so a retry benefits, and an in-flight prediction runs
//! to completion on its blocking thread.

use super::ApiError;
use axum::http::StatusCode;
use sidekick_core::Error;
use std::future::Future;
use std::time::Duration;
use tokio::sync::oneshot;
use tokio::task::JoinHandle;
use tokio::time::Instant;

/// What a request's work is called in its timeout message.
#[derive(Debug, Clone, Copy)]
pub enum Work {
    Classification,
    Embedding,
    Reranking,
}

impl Work {
    fn name(self) -> &'static str {
        match self {
            Self::Classification => "Classification",
            Self::Embedding => "Embedding",
            Self::Reranking => "Reranking",
        }
    }
}

/// One request's deadlines.
pub struct Deadline {
    work: Work,
    model: String,
    started: Instant,
    request: Duration,
    load: Duration,
    /// When loading ended, if the request waited on any.
    loaded_at: Option<Instant>,
    /// What it waited on: "the model", "bucket 1024", …
    loaded: Vec<String>,
}

/// Sent from the blocking task once the loading phase is over: the
/// buckets it waited on a load for.
pub type LoadedTx = oneshot::Sender<Vec<usize>>;

impl Deadline {
    pub fn new(work: Work, model: &str, request: Duration, load: Duration) -> Self {
        Self { work, model: model.to_string(), started: Instant::now(), request, load, loaded_at: None, loaded: Vec::new() }
    }

    /// The model, from `get` (a pool's `get_tracked`), within the load
    /// bound.
    pub async fn model<T>(&mut self, get: impl Future<Output = Result<(T, bool), Error>>) -> Result<T, ApiError> {
        let (model, loaded) =
            tokio::time::timeout_at(self.started + self.load, get).await.map_err(|_| self.load_timeout())??;
        if loaded {
            self.loaded_at = Some(Instant::now());
            self.loaded.push("the model".into());
        }
        Ok(model)
    }

    /// Wait for `task`: its loading phase, which ends when it sends on the
    /// channel `loaded` pairs with, within the load bound, then the rest
    /// within the request timeout. A task that fails before sending ends
    /// the loading phase with its error.
    pub async fn finish<R>(
        &mut self,
        loaded: oneshot::Receiver<Vec<usize>>,
        task: JoinHandle<Result<R, Error>>,
        what: &str,
    ) -> Result<R, ApiError> {
        let load_deadline = self.started + self.load;
        match tokio::time::timeout_at(load_deadline, loaded).await {
            Err(_) => return Err(self.load_timeout()),
            Ok(Ok(buckets)) if !buckets.is_empty() => {
                self.loaded_at = Some(Instant::now());
                self.loaded.push(match buckets.as_slice() {
                    [b] => format!("bucket {b}"),
                    bs => format!("buckets {}", bs.iter().map(usize::to_string).collect::<Vec<_>>().join(", ")),
                });
            }
            // Nothing loaded, or the task ended before its loading phase did.
            Ok(_) => {}
        }
        let deadline = self.loaded_at.unwrap_or(self.started) + self.request;
        tokio::time::timeout_at(deadline, task)
            .await
            .map_err(|_| self.request_timeout())?
            .map_err(|e| ApiError::from(Error::Other(format!("{what} task: {e}"))))?
            .map_err(ApiError::from)
    }

    fn request_timeout(&self) -> ApiError {
        let within = secs(self.request);
        let message = match self.loaded_at {
            None => format!("{} did not complete within {within}", self.work.name()),
            Some(at) => format!(
                "{} did not complete within {within} of model `{}` finishing its load ({}, {}, not counted)",
                self.work.name(),
                self.model,
                self.loaded.join(" and "),
                secs(at - self.started)
            ),
        };
        ApiError::new(StatusCode::GATEWAY_TIMEOUT, "timeout", message)
    }

    fn load_timeout(&self) -> ApiError {
        let mut e = ApiError::new(
            StatusCode::GATEWAY_TIMEOUT,
            "load_timeout",
            format!(
                "Model `{}` was still loading after {} (a bucket's first load compiles it for this Mac). The load \
                 continues in the background; retry later",
                self.model,
                secs(self.load)
            ),
        );
        e.retry_after_secs = Some(LOAD_RETRY_AFTER_SECS);
        e
    }
}

/// `Retry-After` on a load timeout: the load is still running.
const LOAD_RETRY_AFTER_SECS: u64 = 30;

/// `60s`, or `0.2s` below a second.
fn secs(d: Duration) -> String {
    if d >= Duration::from_secs(1) {
        format!("{}s", d.as_secs())
    } else {
        format!("{:.1}s", d.as_secs_f64())
    }
}

/// Send the loading phase's end once; later calls do nothing.
pub fn signal(tx: &mut Option<LoadedTx>, buckets: &[usize]) {
    if let Some(tx) = tx.take() {
        let _ = tx.send(buckets.to_vec());
    }
}
