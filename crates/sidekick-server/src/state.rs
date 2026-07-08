use crate::pool::EmbedderPool;
use sidekick_core::ChatBackend;
use std::sync::Arc;
use std::time::Instant;

#[derive(Clone)]
pub struct AppState {
    pub chat: Arc<dyn ChatBackend>,
    pub embedders: Arc<EmbedderPool>,
    pub api_key: Option<Arc<str>>,
    pub started_at: Instant,
}
