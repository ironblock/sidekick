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

    #[error("input exceeds context budget: {actual} tokens > {limit}")]
    ContextOverflow { actual: usize, limit: usize },

    #[error("guided generation failed: {0}")]
    GuidedGeneration(String),

    #[error("tokenizer error: {0}")]
    Tokenizer(String),

    #[error("inference error: {0}")]
    Inference(String),

    #[error(transparent)]
    Io(#[from] std::io::Error),

    #[error("{0}")]
    Other(String),
}
