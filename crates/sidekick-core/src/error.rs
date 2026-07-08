use crate::UnavailableReason;

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("backend unavailable: {0:?}")]
    Unavailable(UnavailableReason),

    #[error("model not found: {0}")]
    ModelNotFound(String),

    #[error("invalid manifest at {path}: {message}")]
    InvalidManifest { path: String, message: String },

    // No `actual` count: the Foundation Models shim boundary only surfaces
    // the error description, never token counts, so it was always a stub 0
    // that rendered as the self-contradictory "0 tokens > 4096".
    #[error("input exceeds context budget of {limit} tokens")]
    ContextOverflow { limit: usize },

    #[error("guided generation failed: {0}")]
    GuidedGeneration(String),

    #[error("tokenizer error: {0}")]
    Tokenizer(String),

    #[error("inference error: {0}")]
    Inference(String),

    #[error("generation did not complete within {secs}s")]
    Timeout { secs: u64 },

    #[error(transparent)]
    Io(#[from] std::io::Error),

    #[error("{0}")]
    Other(String),
}
