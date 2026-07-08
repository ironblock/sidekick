use crate::pool::EmbedderPool;
use sidekick_core::ChatBackend;
use std::sync::Arc;
use std::time::{Duration, Instant};

#[derive(Clone)]
pub struct AppState {
    pub chat: Arc<dyn ChatBackend>,
    pub embedders: Arc<EmbedderPool>,
    pub api_key: Option<Arc<str>>,
    pub started_at: Instant,
    /// Hard cap on one embeddings request (model load + prediction). Chat
    /// enforces the same config value inside the FM backend.
    pub request_timeout: Duration,
}
