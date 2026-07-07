//! OpenAI wire types — just the subset sidekickd speaks.

use serde::{Deserialize, Serialize};
use serde_json::Value;

// ---------- chat completions ----------

#[derive(Debug, Deserialize)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<WireMessage>,
    #[serde(default)]
    pub temperature: Option<f32>,
    /// Legacy name, still what most clients send.
    #[serde(default)]
    pub max_tokens: Option<u32>,
    #[serde(default)]
    pub max_completion_tokens: Option<u32>,
    #[serde(default)]
    pub stream: Option<bool>,
    #[serde(default)]
    pub stream_options: Option<StreamOptions>,
    #[serde(default)]
    pub response_format: Option<ResponseFormat>,
}

#[derive(Debug, Deserialize)]
pub struct StreamOptions {
    #[serde(default)]
    pub include_usage: bool,
}

#[derive(Debug, Deserialize)]
pub struct WireMessage {
    pub role: String,
    #[serde(default)]
    pub content: WireContent,
}

/// OpenAI content is either a plain string or an array of typed parts.
/// We accept both and flatten text parts; non-text parts are rejected
/// upstream (this daemon is text-only).
#[derive(Debug, Default, Deserialize)]
#[serde(untagged)]
pub enum WireContent {
    Text(String),
    Parts(Vec<ContentPart>),
    #[default]
    Null,
}

#[derive(Debug, Deserialize)]
pub struct ContentPart {
    #[serde(rename = "type")]
    pub kind: String,
    #[serde(default)]
    pub text: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct ResponseFormat {
    #[serde(rename = "type")]
    pub kind: String,
    #[serde(default)]
    pub json_schema: Option<JsonSchemaFormat>,
}

#[derive(Debug, Deserialize)]
pub struct JsonSchemaFormat {
    #[serde(default)]
    pub name: Option<String>,
    pub schema: Value,
}

#[derive(Debug, Serialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub choices: Vec<Choice>,
    pub usage: WireUsage,
}

#[derive(Debug, Serialize)]
pub struct Choice {
    pub index: u32,
    pub message: AssistantMessage,
    pub finish_reason: &'static str,
}

#[derive(Debug, Serialize)]
pub struct AssistantMessage {
    pub role: &'static str,
    pub content: String,
}

#[derive(Debug, Serialize)]
pub struct WireUsage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

// Streaming chunk shapes.

#[derive(Debug, Serialize)]
pub struct ChatCompletionChunk {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChunkChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<WireUsage>,
}

#[derive(Debug, Serialize)]
pub struct ChunkChoice {
    pub index: u32,
    pub delta: Delta,
    pub finish_reason: Option<&'static str>,
}

#[derive(Debug, Default, Serialize)]
pub struct Delta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
}

// ---------- embeddings ----------

#[derive(Debug, Deserialize)]
pub struct EmbeddingsRequest {
    pub model: String,
    pub input: EmbeddingsInput,
    #[serde(default)]
    pub dimensions: Option<usize>,
    #[serde(default)]
    pub encoding_format: Option<String>,
    /// Non-standard extension (Cohere-style): "query" applies the model's
    /// query prefix instead of the document prefix.
    #[serde(default)]
    pub input_type: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum EmbeddingsInput {
    One(String),
    Many(Vec<String>),
}

#[derive(Debug, Serialize)]
pub struct EmbeddingsResponse {
    pub object: &'static str,
    pub data: Vec<EmbeddingObject>,
    pub model: String,
    pub usage: WireUsage,
}

#[derive(Debug, Serialize)]
pub struct EmbeddingObject {
    pub object: &'static str,
    pub index: usize,
    pub embedding: EmbeddingPayload,
}

#[derive(Debug, Serialize)]
#[serde(untagged)]
pub enum EmbeddingPayload {
    Floats(Vec<f32>),
    Base64(String),
}

// ---------- models ----------

#[derive(Debug, Serialize)]
pub struct ModelList {
    pub object: &'static str,
    pub data: Vec<ModelObject>,
}

#[derive(Debug, Serialize)]
pub struct ModelObject {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub owned_by: &'static str,
}

pub fn now_unix() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}
