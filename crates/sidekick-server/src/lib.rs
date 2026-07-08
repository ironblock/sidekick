//! sidekickd — an OpenAI-compatible daemon over Apple on-device inference.
//!
//! Endpoints:
//! - `POST /v1/chat/completions` — Apple Foundation Models (streaming and
//!   non-streaming, `response_format: json_schema` via guided generation)
//! - `POST /v1/embeddings` — registry models (Core ML/ANE encoders, static
//!   floor tier), with Matryoshka `dimensions` and base64 support
//! - `GET /v1/models`, `GET /health`
//!
//! The library form exists so integration tests (and embedders of the
//! daemon) can build the router without binding a socket.

pub mod api;
pub mod config;
pub mod pool;
pub mod state;

pub use api::build_router;
pub use config::Config;
pub use pool::EmbedderPool;
pub use state::AppState;

use sidekick_core::{ChatBackend, ModelRegistry};
use std::sync::Arc;
use std::time::Instant;

/// Assemble state from config with the default (Foundation Models) chat
/// backend. Tests inject their own backend via `AppState` directly.
pub fn build_state(config: &Config) -> anyhow::Result<AppState> {
    let registry = ModelRegistry::scan(&config.models_dir())?;
    if registry.is_empty() {
        tracing::warn!(
            dir = %config.models_dir().display(),
            "no embedding models found; /v1/embeddings will 404"
        );
    }
    let chat: Arc<dyn ChatBackend> =
        Arc::new(sidekick_fm::fm_backend(config.session_ttl(), config.request_timeout()));
    Ok(AppState {
        chat,
        embedders: Arc::new(EmbedderPool::new(registry, config.model_idle_ttl())),
        api_key: config.api_key.as_deref().map(Arc::from),
        started_at: Instant::now(),
    })
}
