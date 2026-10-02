//! Lazy-loading, idle-evicting pools of models.
//!
//! Core ML model loads cost 100ms–1s; a pool keeps models resident after
//! first use and drops them after `idle_ttl` without traffic, so a burst of
//! calls pays the load once and an idle daemon holds no weights. One pool
//! per kind of model: [`EmbedderPool`] and [`ClassifierPool`].
//!
//! Each pool sweeps itself on a timer: an expired model is dropped within a
//! quarter of `idle_ttl` (at most 30 s) of expiring, whether or not any
//! request arrives. Without the sweep, expiry only took effect on the
//! pool's next use, so a daemon left alone kept its weights indefinitely.
//! A model in use by a request is freed when that request finishes.

use sidekick_core::{Classifier, Embedder, Error, ModelRegistry, Result};
use std::collections::HashMap;
use std::sync::{Arc, Weak};
use std::time::Duration;
use tokio::sync::Mutex;
// tokio's clock, so tests can pause it; in a running daemon it is the
// system's monotonic clock.
use tokio::time::Instant;

/// Most time between two sweeps of a pool.
const MAX_SWEEP_INTERVAL: Duration = Duration::from_secs(30);

/// Loads one model by id. Runs on a blocking thread.
pub type Loader<T> = Arc<dyn Fn(&str) -> Result<Arc<T>> + Send + Sync>;

struct Entry<T: ?Sized> {
    model: Arc<T>,
    last_used: Instant,
}

pub struct ModelPool<T: ?Sized> {
    /// What the pool holds, for logs ("embedding model", "classifier").
    kind: &'static str,
    idle_ttl: Duration,
    loader: Loader<T>,
    entries: Arc<Mutex<HashMap<String, Entry<T>>>>,
}

type Entries<T> = Mutex<HashMap<String, Entry<T>>>;

/// Drop `entries`' expired models, as `get` and `resident` do.
async fn evict_expired<T: ?Sized>(entries: &Entries<T>, ttl: Duration, kind: &str) {
    let mut entries = entries.lock().await;
    let before = entries.len();
    entries.retain(|_, e| e.last_used.elapsed() < ttl);
    let dropped = before - entries.len();
    if dropped > 0 {
        tracing::info!("unloaded {dropped} idle {kind}{}", if dropped == 1 { "" } else { "s" });
    }
}

/// Sweep `entries` every quarter of `ttl` (at most every 30 s) until the
/// pool is dropped: the task holds the entries weakly, so it never keeps a
/// pool alive. Outside a tokio runtime there is no sweep, and expiry takes
/// effect on the pool's next use.
fn spawn_sweeper<T: ?Sized + Send + Sync + 'static>(entries: Weak<Entries<T>>, ttl: Duration, kind: &'static str) {
    let Ok(runtime) = tokio::runtime::Handle::try_current() else {
        tracing::debug!("no tokio runtime: {kind} pool won't sweep idle models");
        return;
    };
    let every = (ttl / 4).clamp(Duration::from_millis(10), MAX_SWEEP_INTERVAL);
    runtime.spawn(async move {
        let mut tick = tokio::time::interval(every);
        tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        tick.tick().await; // the first tick is immediate
        loop {
            tick.tick().await;
            let Some(entries) = entries.upgrade() else { break };
            evict_expired(&entries, ttl, kind).await;
        }
    });
}

pub type EmbedderPool = ModelPool<dyn Embedder>;
pub type ClassifierPool = ModelPool<dyn Classifier>;

impl EmbedderPool {
    /// The registry's embedding models, loaded by `sidekick_embed`.
    pub fn embedders(registry: Arc<ModelRegistry>, idle_ttl: Duration) -> Self {
        Self::new("embedding model", idle_ttl, move |id| {
            Ok(Arc::from(sidekick_embed::load_embedder(registry.get(id)?)?))
        })
    }
}

impl ClassifierPool {
    /// The registry's classifiers, loaded by `sidekick_embed`.
    pub fn classifiers(registry: Arc<ModelRegistry>, idle_ttl: Duration) -> Self {
        Self::new("classifier", idle_ttl, move |id| {
            Ok(Arc::from(sidekick_embed::load_classifier(registry.classifier(id)?)?))
        })
    }
}

impl<T: ?Sized + Send + Sync + 'static> ModelPool<T> {
    pub fn new(
        kind: &'static str,
        idle_ttl: Duration,
        loader: impl Fn(&str) -> Result<Arc<T>> + Send + Sync + 'static,
    ) -> Self {
        let entries = Arc::new(Mutex::new(HashMap::new()));
        spawn_sweeper(Arc::downgrade(&entries), idle_ttl, kind);
        Self { kind, idle_ttl, loader: Arc::new(loader), entries }
    }

    pub async fn get(&self, id: &str) -> Result<Arc<T>> {
        let ttl = self.idle_ttl;
        {
            let mut entries = self.entries.lock().await;
            entries.retain(|_, e| e.last_used.elapsed() < ttl);
            if let Some(entry) = entries.get_mut(id) {
                entry.last_used = Instant::now();
                return Ok(entry.model.clone());
            }
        }

        // Load with the lock RELEASED: a cold Core ML load takes seconds
        // (~15s measured for a 600MB artifact) and holding the mutex across
        // it would stall every other model's requests and /health. Two
        // concurrent first-hits may both load; the loser's copy is dropped —
        // wasteful but bounded, and far simpler than single-flight tracking.
        //
        // load-and-insert runs in a DETACHED task so that a caller who gives
        // up (the request deadline cancels this future) doesn't cancel the
        // insert with it: the finished load still becomes resident, and the
        // client's retry finds it instead of restarting a doomed load.
        let loader = self.loader.clone();
        let entries = self.entries.clone();
        let kind = self.kind;
        let id = id.to_string();
        let load = tokio::spawn(async move {
            let load_id = id.clone();
            let model = tokio::task::spawn_blocking(move || loader(&load_id))
                .await
                .map_err(|e| Error::Other(format!("load task failed: {e}")))??;
            let mut entries = entries.lock().await;
            let entry = entries
                .entry(id.clone())
                .or_insert_with(|| Entry { model: model.clone(), last_used: Instant::now() });
            entry.last_used = Instant::now();
            tracing::info!(model = %id, "{kind} loaded");
            Ok(entry.model.clone())
        });
        load.await.map_err(|e| Error::Other(format!("load task failed: {e}")))?
    }

    /// Number of currently-resident models (for /health).
    pub async fn resident(&self) -> usize {
        let mut entries = self.entries.lock().await;
        let ttl = self.idle_ttl;
        entries.retain(|_, e| e.last_used.elapsed() < ttl);
        entries.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[tokio::test]
    async fn loads_once_keeps_resident_and_evicts_when_idle() {
        let loads = Arc::new(AtomicUsize::new(0));
        let counter = loads.clone();
        let pool: ModelPool<str> = ModelPool::new("test model", Duration::from_millis(200), move |id| {
            counter.fetch_add(1, Ordering::SeqCst);
            match id {
                "missing" => Err(Error::ModelNotFound(id.into())),
                _ => Ok(Arc::from(id)),
            }
        });
        assert_eq!(&*pool.get("a").await.unwrap(), "a");
        assert_eq!(&*pool.get("a").await.unwrap(), "a");
        assert_eq!(loads.load(Ordering::SeqCst), 1);
        assert_eq!(pool.resident().await, 1);
        assert!(matches!(pool.get("missing").await, Err(Error::ModelNotFound(_))));
        tokio::time::sleep(Duration::from_millis(250)).await;
        assert_eq!(pool.resident().await, 0);
        pool.get("a").await.unwrap();
        assert_eq!(loads.load(Ordering::SeqCst), 3);
    }

    /// A model `get` handed out, and whether the pool has let go of it: the
    /// pool's `Arc` is the only other one once the caller drops theirs.
    fn held(model: &Arc<str>) -> bool {
        Arc::strong_count(model) > 1
    }

    #[tokio::test(start_paused = true)]
    async fn an_idle_model_is_dropped_without_any_further_call() {
        let ttl = Duration::from_secs(900);
        let pool: ModelPool<str> = ModelPool::new("test model", ttl, |id| Ok(Arc::from(id)));
        let model = pool.get("a").await.unwrap();
        assert!(held(&model));
        // Just under the TTL: still resident.
        tokio::time::sleep(ttl - Duration::from_secs(1)).await;
        assert!(held(&model), "dropped before its TTL");
        // Past the TTL plus one sweep interval (ttl/4 capped at 30 s), with
        // no get() or resident() in between: the sweep alone dropped it.
        tokio::time::sleep(Duration::from_secs(1) + MAX_SWEEP_INTERVAL + Duration::from_secs(1)).await;
        assert!(!held(&model), "still resident past its TTL with no traffic");
        assert_eq!(pool.resident().await, 0);
    }

    #[tokio::test(start_paused = true)]
    async fn the_sweep_stops_with_its_pool() {
        let pool: ModelPool<str> = ModelPool::new("test model", Duration::from_secs(1), |id| Ok(Arc::from(id)));
        let entries = Arc::downgrade(&pool.entries);
        drop(pool);
        // The sweeper holds the entries weakly: they're gone with the pool,
        // and its next tick ends the task.
        assert!(entries.upgrade().is_none());
        tokio::time::sleep(Duration::from_secs(5)).await;
    }

    #[test]
    fn a_pool_built_outside_a_runtime_still_works() {
        // No runtime: no sweeper (and no panic); expiry applies on next use.
        let pool: ModelPool<str> = ModelPool::new("test model", Duration::from_secs(1), |id| Ok(Arc::from(id)));
        let rt = tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap();
        assert_eq!(&*rt.block_on(pool.get("a")).unwrap(), "a");
    }
}
