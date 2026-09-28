use crate::{Availability, Result};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Role {
    System,
    User,
    Assistant,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: Role,
    pub content: String,
}

impl ChatMessage {
    pub fn new(role: Role, content: impl Into<String>) -> Self {
        Self { role, content: content.into() }
    }
}

/// Backend-neutral chat request. The server translates OpenAI wire format
/// into this; a future library API constructs it directly.
#[derive(Debug, Clone, Default)]
pub struct ChatRequest {
    pub messages: Vec<ChatMessage>,
    pub temperature: Option<f32>,
    pub max_tokens: Option<u32>,
    /// JSON Schema for constrained output. When set, backends that support
    /// guided generation (Foundation Models) must return valid JSON matching
    /// the schema; backends that don't should fall back to prompt-based JSON
    /// coaxing and report so via `ChatResponse::constrained`.
    pub schema: Option<serde_json::Value>,
    /// Stop sequences (OpenAI semantics): the reply ends before the earliest
    /// occurrence of any of them, which is not included.
    pub stop: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FinishReason {
    Stop,
    Length,
    ContentFilter,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    /// Part of `prompt_tokens` served from a cache (e.g. a reused session),
    /// when the backend reports real usage; None when usage is estimated.
    pub cached_tokens: Option<u32>,
    /// Part of `completion_tokens` spent on reasoning, when reported.
    pub reasoning_tokens: Option<u32>,
}

/// Facts about the model behind a chat backend, where it can report them.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelInfo {
    /// Human-readable model name (Foundation Models on macOS 27: e.g.
    /// "AFM 3 Core"). May be localized; use `variant_id` in logic.
    pub variant: Option<String>,
    /// Stable model identifier (Foundation Models: "core3",
    /// "core_advanced3", or "other").
    pub variant_id: Option<String>,
    /// Combined input+output token budget.
    pub context_size: Option<usize>,
    /// What the *model* supports (e.g. "guided_generation", "tool_calling",
    /// "vision", "reasoning") — not necessarily what the daemon exposes.
    pub capabilities: Option<Vec<String>>,
}

#[derive(Debug, Clone)]
pub struct ChatResponse {
    pub content: String,
    pub finish: FinishReason,
    pub usage: Usage,
    /// True when the output was produced under real constrained decoding
    /// (as opposed to best-effort prompting).
    pub constrained: bool,
}

/// A generation backend. Implementations: Foundation Models (macOS 26+),
/// future Core ML LLM tier, mock (tests).
///
/// Multi-turn state is the caller's concern: requests carry full history,
/// OpenAI-style. Backends may internally cache sessions keyed on history
/// prefixes (see sidekick-server's session cache), but must produce correct
/// results for any history from a cold start.
#[async_trait::async_trait]
pub trait ChatBackend: Send + Sync {
    /// Stable identifier, used as the OpenAI `model` name (e.g. "apple-fm").
    fn id(&self) -> &str;

    /// Combined input+output token budget, if the backend has a hard one
    /// (Foundation Models on-device: 4096 for AFM 3 Core).
    fn context_limit(&self) -> Option<usize> {
        None
    }

    /// Facts about the underlying model, if the backend can report them.
    async fn model_info(&self) -> Option<ModelInfo> {
        None
    }

    async fn availability(&self) -> Availability;

    async fn complete(&self, req: ChatRequest) -> Result<ChatResponse>;
}
