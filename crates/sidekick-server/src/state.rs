use crate::chat_models::ChatModels;
use crate::config::{OnnxThreads, RequestTimeouts};
use crate::pool::{ClassifierPool, EmbedderPool, TokenizerPool};
use sidekick_core::ModelRegistry;
use sidekick_embed::placement::Placements;
use std::sync::Arc;
use std::time::{Duration, Instant};

#[derive(Clone)]
pub struct AppState {
    /// The chat models, by id: `/v1/chat/completions` routes on the
    /// request's `model`, exactly (D53).
    pub chat: Arc<ChatModels>,
    /// Every model's manifest. Listings and request validation read it
    /// without loading anything.
    pub registry: Arc<ModelRegistry>,
    pub embedders: Arc<EmbedderPool>,
    pub classifiers: Arc<ClassifierPool>,
    /// The same models' tokenizers alone, for /tokenize and /detokenize.
    pub tokenizers: Arc<TokenizerPool>,
    /// Whether this build can run classifiers (Core ML). When it can't,
    /// they're left out of /v1/models, as chat is, and every classify
    /// request is a 503.
    pub classifiers_supported: bool,
    pub api_key: Option<Arc<str>>,
    pub started_at: Instant,
    /// Each model's cap on a request's work, past any load it waited on
    /// (api::deadline). A chat backend is built with its model's and
    /// enforces it itself.
    pub request_timeouts: RequestTimeouts,
    /// How long an embeddings, classify or rerank request waits on model
    /// and bucket loads, which `request_timeouts` don't count
    /// (api::deadline).
    pub load_timeout: Duration,
    /// Where Core ML placed each loaded bucket's operations, read in the
    /// background after the bucket's first load. `None` when
    /// `report_compute_plans` is off or the build has no Core ML.
    pub placements: Option<Arc<Placements>>,
    /// Each `onnx` model's inference threads (D40), for /health.
    pub onnx_threads: OnnxThreads,
    /// The startup warm-up of Core ML's compile cache (D52), for /health.
    pub warmup: Arc<crate::warm::Warmup>,
}
