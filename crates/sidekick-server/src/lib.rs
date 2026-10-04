//! sidekickd — an OpenAI-compatible daemon over Apple on-device inference.
//!
//! Endpoints:
//! - `POST /v1/chat/completions` — Apple Foundation Models (streaming and
//!   non-streaming, `response_format: json_schema` via guided generation)
//! - `POST /v1/embeddings` — registry models (Core ML/ANE encoders, static
//!   floor tier), with Matryoshka `dimensions` and base64 support
//! - `POST /v1/classify` — registry classifiers (Core ML/ANE):
//!   text-classification and zero-shot, vLLM's `/classify` contract
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
pub use pool::{ClassifierPool, EmbedderPool, ModelPool};
pub use state::AppState;

use sidekick_core::{ChatBackend, ModelRegistry, ScanOptions};
use std::sync::Arc;
use std::time::Instant;

/// Assemble state from config with the default (Foundation Models) chat
/// backend. Tests inject their own backend via `AppState` directly.
pub fn build_state(config: &Config) -> anyhow::Result<AppState> {
    let options = ScanOptions {
        compute_units: config.compute_units(),
        ignore_ane_weight_cap: config.ignore_ane_weight_cap,
        ignore_cpu_seq_cap: config.ignore_cpu_seq_cap,
    };
    let registry = Arc::new(ModelRegistry::scan_with(&config.models_dir(), &options)?);
    if registry.is_empty() && registry.classifier_ids().next().is_none() {
        tracing::warn!(
            dir = %config.models_dir().display(),
            "no models found; /v1/embeddings and /v1/classify will 404"
        );
    }
    let chat: Arc<dyn ChatBackend> =
        Arc::new(sidekick_fm::fm_backend(config.session_ttl(), config.request_timeout()));
    Ok(AppState {
        chat,
        embedders: Arc::new(EmbedderPool::embedders(registry.clone(), config.model_idle_ttl())),
        classifiers: Arc::new(ClassifierPool::classifiers(registry.clone(), config.model_idle_ttl())),
        registry,
        classifiers_supported: sidekick_embed::CLASSIFIERS_SUPPORTED,
        chat_model: Default::default(),
        api_key: config.api_key.as_deref().map(Arc::from),
        started_at: Instant::now(),
        request_timeout: config.request_timeout(),
        load_timeout: config.load_timeout(),
        placements: config
            .report_compute_plans
            .then(|| sidekick_embed::placement::enable(config.compute_plan_cache_dir()))
            .flatten(),
    })
}
