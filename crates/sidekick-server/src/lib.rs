//! sidekickd — an OpenAI-compatible daemon for on-device embeddings,
//! classification and chat: Apple Foundation Models and Core ML on macOS,
//! and ONNX Runtime on the CPU on macOS and Linux.
//!
//! Endpoints:
//! - `POST /v1/chat/completions` — chat models by exact `model` id: Apple
//!   Foundation Models as `apple-fm` (streaming and non-streaming,
//!   `response_format: json_schema` via guided generation)
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
pub mod chat_models;
pub mod config;
pub mod fetch;
pub mod pool;
pub mod state;
pub mod warm;

pub use api::build_router;
pub use config::Config;
pub use pool::{ClassifierPool, EmbedderPool, ModelPool, TokenizerPool};
pub use state::AppState;

use chat_models::ChatModels;
use sidekick_core::{ChatBackend, EmbeddingBackendKind, ModelRegistry, ScanOptions};
use std::sync::Arc;
use std::time::Instant;

/// What the registry scan is told: the daemon config's compute units and
/// caps, and whether this machine has a Neural Engine.
pub fn scan_options(config: &Config) -> ScanOptions {
    ScanOptions {
        compute_units: config.compute_units(),
        ignore_ane_weight_cap: config.ignore_ane_weight_cap,
        ignore_cpu_seq_cap: config.ignore_cpu_seq_cap,
        ignore_ane_seq_cap: config.ignore_ane_seq_cap,
        no_neural_engine: !sidekick_embed::has_neural_engine(),
    }
}

/// The models directory, scanned as the daemon scans it: what it serves,
/// on the compute units and buckets it serves them with.
pub fn scan(config: &Config) -> anyhow::Result<ModelRegistry> {
    Ok(ModelRegistry::scan_with(&config.models_dir(), &scan_options(config))?)
}

/// Assemble state from config, with Foundation Models as the chat model
/// `apple-fm`. Tests inject their own chat models via `AppState` directly.
pub fn build_state(config: &Config) -> anyhow::Result<AppState> {
    let onnx_threads = config.onnx_threads()?;
    let request_timeouts = config.request_timeouts()?;
    let mut registry = scan(config)?;
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
    let mut chat = ChatModels::default();
    chat.insert(chat_models::foundation_models(&request_timeouts, |timeout| {
        Arc::new(sidekick_fm::fm_backend(config.session_ttl(), timeout)) as Arc<dyn ChatBackend>
    }))?;
    Ok(AppState {
        chat: Arc::new(chat),
        embedders: Arc::new(EmbedderPool::embedders_with(registry.clone(), config.model_idle_ttl(), onnx_threads.clone())),
        classifiers: Arc::new(ClassifierPool::classifiers(registry.clone(), config.model_idle_ttl())),
        tokenizers: Arc::new(TokenizerPool::tokenizers(registry.clone(), config.model_idle_ttl())),
        registry,
        classifiers_supported: sidekick_embed::CLASSIFIERS_SUPPORTED,
        api_key: config.api_key.as_deref().map(Arc::from),
        started_at: Instant::now(),
        request_timeouts,
        load_timeout: config.load_timeout(),
        onnx_threads,
        // Started by the daemon once it's serving (`warm::Warmup::start`).
        warmup: Default::default(),
        placements: config
            .report_compute_plans
            .then(|| sidekick_embed::placement::enable(config.compute_plan_cache_dir()))
            .flatten(),
    })
}
