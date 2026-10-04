use crate::pool::{ClassifierPool, EmbedderPool};
use sidekick_core::{ChatBackend, ModelRegistry};
use sidekick_embed::placement::Placements;
use std::sync::{Arc, Mutex, PoisonError};
use std::time::{Duration, Instant};

#[derive(Clone)]
pub struct AppState {
    pub chat: Arc<dyn ChatBackend>,
    /// Every model's manifest. Listings and request validation read it
    /// without loading anything.
    pub registry: Arc<ModelRegistry>,
    pub embedders: Arc<EmbedderPool>,
    pub classifiers: Arc<ClassifierPool>,
    /// Whether this build can run classifiers (Core ML). When it can't,
    /// they're left out of /v1/models, as chat is, and every classify
    /// request is a 503.
    pub classifiers_supported: bool,
    /// The chat model's provenance id, cached off the request path.
    pub chat_model: Arc<ChatModelCache>,
    pub api_key: Option<Arc<str>>,
    pub started_at: Instant,
    /// Hard cap on one embeddings or classify request (model load +
    /// prediction). Chat enforces the same config value inside the FM
    /// backend.
    pub request_timeout: Duration,
    /// How long an embeddings, classify or rerank request waits on model
    /// and bucket loads, which `request_timeout` doesn't count
    /// (api::deadline).
    pub load_timeout: Duration,
    /// Where Core ML placed each loaded bucket's operations, read in the
    /// background after the bucket's first load. `None` when
    /// `report_compute_plans` is off or the build has no Core ML.
    pub placements: Option<Arc<Placements>>,
}

/// How long a known Foundation Models variant id is reused before a
/// background refresh. It changes only when the OS updates the model.
const CHAT_MODEL_TTL: Duration = Duration::from_secs(600);
/// The same while the backend reports none (Apple Intelligence off, assets
/// still downloading), so the header catches up soon after it's available.
const CHAT_MODEL_RETRY: Duration = Duration::from_secs(30);

/// The Foundation Models variant id that chat's `sidekick-model` header
/// names. Asking the backend is a blocking call into the Swift shim, so
/// requests read this cache and never wait on it: a stale entry is
/// refreshed by a detached task, one at a time.
#[derive(Default)]
pub struct ChatModelCache {
    state: Mutex<CacheState>,
}

#[derive(Default)]
struct CacheState {
    variant_id: Option<String>,
    fetched: Option<Instant>,
    refreshing: bool,
}

impl ChatModelCache {
    pub fn set(&self, variant_id: Option<String>) {
        let mut s = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        s.variant_id = variant_id;
        s.fetched = Some(Instant::now());
        s.refreshing = false;
    }

    /// The cached id, and whether this caller should start a refresh.
    fn read(&self) -> (Option<String>, bool) {
        let mut s = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        let ttl = if s.variant_id.is_some() { CHAT_MODEL_TTL } else { CHAT_MODEL_RETRY };
        let stale = s.fetched.is_none_or(|t| t.elapsed() >= ttl);
        let refresh = stale && !s.refreshing;
        if refresh {
            s.refreshing = true;
        }
        (s.variant_id.clone(), refresh)
    }
}

impl AppState {
    /// The chat model for provenance: the Foundation Models variant id when
    /// the backend has reported one, otherwise the chat model id. Never
    /// waits on the backend.
    pub fn chat_model_id(&self) -> String {
        let (id, refresh) = self.chat_model.read();
        if refresh {
            let state = self.clone();
            tokio::spawn(async move { state.refresh_chat_model().await });
        }
        id.unwrap_or_else(|| self.chat.id().to_string())
    }

    /// Ask the backend for its variant id and cache it.
    pub async fn refresh_chat_model(&self) {
        let info = self.chat.model_info().await;
        self.chat_model.set(info.and_then(|i| i.variant_id));
    }
}
