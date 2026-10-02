//! Core traits and types for sidekick.
//!
//! Sidekick runs "very small" asynchronous inference tasks (titles, tags,
//! embeddings, extraction) on Apple Silicon, preferring the Apple Neural
//! Engine and Apple's Foundation Models where available, with graceful
//! degradation elsewhere. This crate holds the backend-neutral vocabulary:
//! chat, embedding and classification traits, availability states, and the on-disk model
//! manifest format. It is deliberately free of any Apple dependency so that
//! consumers (and CI) can build it anywhere.

pub mod chat;
pub mod classify;
pub mod embed;
pub mod error;
pub mod manifest;

pub use chat::{
    ChatBackend, ChatMessage, ChatRequest, ChatResponse, DeltaSink, FinishReason, ModelInfo, Role,
    Usage,
};
pub use classify::{
    activate, ClassifyParams, ClassifyTask, Classifier, PairParams, Prepared, ProblemType, QuestionType,
    Source,
    TruncationSide,
};
pub use embed::{truncate_normalized, EmbedLimits, EmbedPurpose, Embedder, Truncate};
pub use error::{Error, Result};
pub use manifest::{
    calibration_key, AneWeightLimit, ClassifierManifest, ClassifyFormat, ComputeUnits, EmbeddingBackendKind,
    ModelManifest, OptionRendering, ScanOptions, MAX_ANE_PROGRAM_WEIGHT_BYTES,
    ModelRegistry, Pooling, ResolvedClassifier,
};

use serde::{Deserialize, Serialize};

/// Whether a backend can currently serve requests.
///
/// Availability on Apple platforms is a state machine, not a boolean:
/// Apple Intelligence can be toggled, model assets download lazily, and the
/// ANE may be absent. Backends should re-evaluate cheaply per probe.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum Availability {
    Available,
    Unavailable { reason: UnavailableReason },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UnavailableReason {
    /// Hardware can never run this backend (e.g. no Apple Silicon / no ANE).
    DeviceNotEligible,
    /// Apple Intelligence is switched off in system settings.
    AppleIntelligenceNotEnabled,
    /// Model assets are still downloading or compiling; may become available.
    ModelNotReady,
    /// Backend not compiled into this build (non-macOS stub).
    NotSupportedInBuild,
    /// Anything else, with a human-readable explanation.
    Other(String),
}

impl Availability {
    pub fn is_available(&self) -> bool {
        matches!(self, Availability::Available)
    }

    pub fn unavailable(reason: UnavailableReason) -> Self {
        Availability::Unavailable { reason }
    }
}
