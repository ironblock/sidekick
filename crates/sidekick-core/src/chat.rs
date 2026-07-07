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
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FinishReason {
    Stop,
    Length,
    ContentFilter,
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
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
    /// (Foundation Models on-device: 4096).
    fn context_limit(&self) -> Option<usize> {
        None
    }

    async fn availability(&self) -> Availability;

    async fn complete(&self, req: ChatRequest) -> Result<ChatResponse>;
}
