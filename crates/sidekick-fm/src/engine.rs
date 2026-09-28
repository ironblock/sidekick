use sidekick_core::{Availability, ModelInfo, Result, UnavailableReason};

/// Options for a single response.
#[derive(Debug, Clone, Default)]
pub struct RespondOptions {
    /// JSON Schema for guided generation. When set, the engine returns the
    /// JSON text of content constrained to the schema.
    pub schema: Option<serde_json::Value>,
    pub temperature: Option<f32>,
    pub max_tokens: Option<u32>,
}

/// Token accounting for one response, as Foundation Models reports it on
/// macOS 27 (`LanguageModelSession.Response.usage`). `input` includes the
/// session's instructions and prior turns; `cached` is the part of `input`
/// served from the session's cache.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Deserialize)]
pub struct EngineUsage {
    pub input: u32,
    pub cached: u32,
    pub output: u32,
    #[serde(default)]
    pub reasoning: u32,
}

/// One response from an engine.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct EngineResponse {
    /// Assistant text, or the JSON text of schema-constrained content.
    pub text: String,
    /// Real token usage when the engine reports it (macOS 27).
    pub usage: Option<EngineUsage>,
    /// Whether generation stopped at `max_tokens`; None when unknown.
    pub truncated: Option<bool>,
}

impl EngineResponse {
    /// A response with text only (no usage or truncation information).
    pub fn text(text: impl Into<String>) -> Self {
        Self { text: text.into(), ..Default::default() }
    }
}

/// A response delivered as cumulative snapshots.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct StreamedResponse {
    /// The final snapshot's text, usage and truncation.
    pub response: EngineResponse,
    /// Generation stopped early because the snapshot callback asked it to.
    pub cancelled: bool,
}

/// A provider of stateful chat sessions. The Foundation Models FFI is the
/// real implementation; tests use mocks.
///
/// `respond` blocks; callers run it on a blocking thread.
pub trait SessionEngine: Send + Sync + 'static {
    type Session: Send + 'static;

    fn availability(&self) -> Availability;

    /// Facts about the model (variant, context size, capabilities), where
    /// the engine can report them.
    fn model_info(&self) -> Option<ModelInfo> {
        None
    }

    /// Create a session primed with system instructions (may be empty).
    fn create(&self, instructions: &str) -> Result<Self::Session>;

    /// Send one user prompt to the session and return the assistant text
    /// (or schema-constrained JSON text when `opts.schema` is set).
    fn respond(
        &self,
        session: &mut Self::Session,
        prompt: &str,
        opts: &RespondOptions,
    ) -> Result<EngineResponse>;

    /// Like [`respond`](Self::respond), but calls `on_snapshot` with the
    /// cumulative text as it is generated; returning false stops generation.
    /// Plain text only (`opts.schema` must be None). A session whose stream
    /// was stopped early must not be used again.
    ///
    /// The default delivers the whole reply as a single snapshot.
    fn respond_stream(
        &self,
        session: &mut Self::Session,
        prompt: &str,
        opts: &RespondOptions,
        on_snapshot: &mut (dyn FnMut(&str) -> bool + Send),
    ) -> Result<StreamedResponse> {
        let response = self.respond(session, prompt, opts)?;
        let cancelled = !on_snapshot(&response.text);
        Ok(StreamedResponse { response, cancelled })
    }
}

/// Engine used when Foundation Models isn't compiled in.
pub struct StubEngine;

impl SessionEngine for StubEngine {
    type Session = ();

    fn availability(&self) -> Availability {
        Availability::unavailable(UnavailableReason::NotSupportedInBuild)
    }

    fn create(&self, _instructions: &str) -> Result<()> {
        Err(sidekick_core::Error::Unavailable(
            UnavailableReason::NotSupportedInBuild,
        ))
    }

    fn respond(&self, _s: &mut (), _prompt: &str, _opts: &RespondOptions) -> Result<EngineResponse> {
        Err(sidekick_core::Error::Unavailable(
            UnavailableReason::NotSupportedInBuild,
        ))
    }
}
