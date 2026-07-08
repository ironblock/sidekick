//! Lazy-loading, idle-evicting pool of embedding models.
//!
//! Core ML model loads cost 100ms–1s; the pool keeps models resident after
//! first use and drops them after `idle_ttl` without traffic, so a burst of
//! embedding calls pays the load once and an idle daemon holds no weights.

use sidekick_core::{Embedder, Error, ModelRegistry, Result};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::Mutex;

struct Entry {
    embedder: Arc<dyn Embedder>,
    last_used: Instant,
}

pub struct EmbedderPool {
    registry: ModelRegistry,
    idle_ttl: Duration,
    entries: Arc<Mutex<HashMap<String, Entry>>>,
}

impl EmbedderPool {
    pub fn new(registry: ModelRegistry, idle_ttl: Duration) -> Self {
        Self { registry, idle_ttl, entries: Arc::new(Mutex::new(HashMap::new())) }
    }

    pub fn registry(&self) -> &ModelRegistry {
        &self.registry
    }

    pub async fn get(&self, id: &str) -> Result<Arc<dyn Embedder>> {
        let ttl = self.idle_ttl;
        {
            let mut entries = self.entries.lock().await;
            entries.retain(|_, e| e.last_used.elapsed() < ttl);
            if let Some(entry) = entries.get_mut(id) {
                entry.last_used = Instant::now();
                return Ok(entry.embedder.clone());
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
        let model = self.registry.get(id)?.clone();
        let entries = self.entries.clone();
        let id = id.to_string();
        let load = tokio::spawn(async move {
            let embedder =
                tokio::task::spawn_blocking(move || sidekick_embed::load_embedder(&model))
                    .await
                    .map_err(|e| Error::Other(format!("load task failed: {e}")))??;
            let embedder: Arc<dyn Embedder> = Arc::from(embedder);
            let mut entries = entries.lock().await;
            let entry = entries
                .entry(id.clone())
                .or_insert_with(|| Entry { embedder: embedder.clone(), last_used: Instant::now() });
            entry.last_used = Instant::now();
            tracing::info!(model = %id, "embedding model loaded");
            Ok(entry.embedder.clone())
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
