//! Lazy-loading, idle-evicting pools of models.
//!
//! Core ML model loads cost 100ms–1s; a pool keeps models resident after
//! first use and drops them after `idle_ttl` without traffic, so a burst of
//! calls pays the load once and an idle daemon holds no weights. One pool
//! per kind of model: [`EmbedderPool`] and [`ClassifierPool`].

use sidekick_core::{Classifier, Embedder, Error, ModelRegistry, Result};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::Mutex;

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
        Self {
            kind,
            idle_ttl,
            loader: Arc::new(loader),
            entries: Arc::new(Mutex::new(HashMap::new())),
        }
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
}
