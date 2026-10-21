//! sidekickd — an OpenAI-compatible daemon for on-device embeddings,
//! classification and chat: Apple Foundation Models and Core ML on macOS,
//! and ONNX Runtime on the CPU on macOS and Linux.
//!
//! Endpoints:
//! - `POST /v1/chat/completions` — Apple Foundation Models (streaming and
//!   non-streaming, `response_format: json_schema` via guided generation)
//! - `POST /v1/embeddings` — registry models (Core ML/ANE encoders, ONNX
//!   Runtime CPU encoders, static floor tier), with Matryoshka `dimensions`
//!   and base64 support
//! - `POST /v1/classify` — registry classifiers (Core ML/ANE):
//!   text-classification and zero-shot, vLLM's `/classify` contract
//! - `POST /tokenize`, `POST /detokenize` — an embedder's or
//!   text-classification model's tokenizer, in vLLM's shape
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
pub use pool::{ClassifierPool, EmbedderPool, ModelPool, TokenizerPool};
pub use state::AppState;

use sidekick_core::{ChatBackend, EmbeddingBackendKind, ModelRegistry, ScanOptions};
use std::sync::Arc;
use std::time::Instant;

/// Assemble state from config with the default (Foundation Models) chat
/// backend. Tests inject their own backend via `AppState` directly.
pub fn build_state(config: &Config) -> anyhow::Result<AppState> {
    let options = ScanOptions {
        compute_units: config.compute_units(),
        ignore_ane_weight_cap: config.ignore_ane_weight_cap,
        ignore_cpu_seq_cap: config.ignore_cpu_seq_cap,
        ignore_ane_seq_cap: config.ignore_ane_seq_cap,
    };
    let onnx_threads = config.onnx_threads()?;
    let mut registry = ModelRegistry::scan_with(&config.models_dir(), &options)?;
    // ONNX Runtime loads only for an onnx model, and then at startup, so a
    // missing runtime skips the onnx models with its reason in /health
    // right away. Without one, it's never loaded (D40).
    let has_onnx = registry.iter().any(|m| m.manifest.backend == EmbeddingBackendKind::Onnx);
    if has_onnx {
        if let Err(reason) = sidekick_embed::onnx::load_runtime(config.onnx_runtime.as_deref()) {
            registry.skip_backend(EmbeddingBackendKind::Onnx, &reason);
        }
    }
    let registry = Arc::new(registry);
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
        embedders: Arc::new(EmbedderPool::embedders_with(registry.clone(), config.model_idle_ttl(), onnx_threads.clone())),
        classifiers: Arc::new(ClassifierPool::classifiers(registry.clone(), config.model_idle_ttl())),
        tokenizers: Arc::new(TokenizerPool::tokenizers(registry.clone(), config.model_idle_ttl())),
        registry,
        classifiers_supported: sidekick_embed::CLASSIFIERS_SUPPORTED,
        chat_model: Default::default(),
        api_key: config.api_key.as_deref().map(Arc::from),
        started_at: Instant::now(),
        request_timeout: config.request_timeout(),
        load_timeout: config.load_timeout(),
        onnx_threads,
        placements: config
            .report_compute_plans
            .then(|| sidekick_embed::placement::enable(config.compute_plan_cache_dir()))
            .flatten(),
    })
}
