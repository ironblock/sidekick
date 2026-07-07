use sidekick_core::{Availability, Result, UnavailableReason};

/// Options for a single response.
#[derive(Debug, Clone, Default)]
pub struct RespondOptions {
    /// JSON Schema for guided generation. When set, the engine returns the
    /// JSON text of content constrained to the schema.
    pub schema: Option<serde_json::Value>,
    pub temperature: Option<f32>,
    pub max_tokens: Option<u32>,
}

/// A provider of stateful chat sessions. The Foundation Models FFI is the
/// real implementation; tests use mocks.
///
/// `respond` blocks; callers run it on a blocking thread.
pub trait SessionEngine: Send + Sync + 'static {
    type Session: Send + 'static;

    fn availability(&self) -> Availability;

    /// Create a session primed with system instructions (may be empty).
    fn create(&self, instructions: &str) -> Result<Self::Session>;

    /// Send one user prompt to the session and return the assistant text
    /// (or schema-constrained JSON text when `opts.schema` is set).
    fn respond(
        &self,
        session: &mut Self::Session,
        prompt: &str,
        opts: &RespondOptions,
    ) -> Result<String>;
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

    fn respond(&self, _s: &mut (), _prompt: &str, _opts: &RespondOptions) -> Result<String> {
        Err(sidekick_core::Error::Unavailable(
            UnavailableReason::NotSupportedInBuild,
        ))
    }
}
