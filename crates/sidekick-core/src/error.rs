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

    /// The request didn't fit the model's context window. `actual` is the
    /// token count when the backend reports one (Foundation Models does on
    /// macOS 27; macOS 26 reports neither number, so `limit` is then the
    /// last known context size).
    #[error("{}", overflow_message(.limit, .actual))]
    ContextOverflow { limit: usize, actual: Option<usize> },

    /// The model's safety guardrails rejected the input or refused to answer.
    #[error("content rejected by the model's guardrails: {0}")]
    ContentFiltered(String),

    #[error("rate limited: {message}")]
    RateLimited { message: String, retry_after_secs: Option<u64> },

    /// A failure worth retrying on a fresh session (timeouts, contention).
    #[error("transient backend failure: {0}")]
    Transient(String),

    /// The model's queue is full (HTTP 503 `backend_busy` with
    /// `Retry-After`): an open chat model's, which `apple-fm` shares (D53).
    #[error("busy: {message}")]
    Busy { message: String, retry_after_secs: u64 },

    /// The model was still loading when the request's load timeout passed
    /// (HTTP 504 `load_timeout`); the load continues, so a retry finds it.
    #[error("model `{model}` was still loading after {secs}s")]
    LoadTimeout { model: String, secs: u64 },

    #[error("unsupported language or locale: {0}")]
    UnsupportedLanguage(String),

    /// A `response_format` schema outside the supported subset (a client
    /// error: HTTP 400 `unsupported_schema`); the message names the keyword
    /// and where it is.
    #[error("response_format schema: {0}")]
    UnsupportedSchema(String),

    #[error("guided generation failed: {0}")]
    GuidedGeneration(String),

    /// A request the model can't serve as asked (a client error: HTTP 400).
    #[error("invalid request: {0}")]
    InvalidRequest(String),

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

fn overflow_message(limit: &usize, actual: &Option<usize>) -> String {
    match actual {
        Some(actual) => format!("input of {actual} tokens exceeds context budget of {limit} tokens"),
        None => format!("input exceeds context budget of {limit} tokens"),
    }
}
