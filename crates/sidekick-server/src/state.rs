use crate::pool::{ClassifierPool, EmbedderPool};
use sidekick_core::{ChatBackend, ModelRegistry};
use std::sync::Arc;
use std::time::{Duration, Instant};

#[derive(Clone)]
pub struct AppState {
    pub chat: Arc<dyn ChatBackend>,
    /// Every model's manifest. Listings and request validation read it
    /// without loading anything.
    pub registry: Arc<ModelRegistry>,
    pub embedders: Arc<EmbedderPool>,
    pub classifiers: Arc<ClassifierPool>,
    pub api_key: Option<Arc<str>>,
    pub started_at: Instant,
    /// Hard cap on one embeddings or classify request (model load +
    /// prediction). Chat enforces the same config value inside the FM
    /// backend.
    pub request_timeout: Duration,
}
