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
    #[serde(default)]
    pub stop: Option<StopSequences>,
    // Fields this daemon can't honor. Parsed only so a request that depends
    // on them is rejected instead of silently answered without them (D22).
    #[serde(default)]
    pub n: Option<u32>,
    #[serde(default)]
    pub tools: Option<Vec<Value>>,
    #[serde(default)]
    pub functions: Option<Vec<Value>>,
    #[serde(default)]
    pub tool_choice: Option<Value>,
    #[serde(default)]
    pub function_call: Option<Value>,
    #[serde(default)]
    pub logprobs: Option<bool>,
    #[serde(default)]
    pub top_logprobs: Option<u32>,
}

/// OpenAI's `stop`: one string or up to four.
#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum StopSequences {
    One(String),
    Many(Vec<String>),
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
    /// Non-standard extension: true when the output was produced under real
    /// constrained decoding (response_format json_schema), false for plain
    /// or best-effort-prompted output (json_object). OpenAI clients ignore
    /// unknown top-level fields.
    pub constrained: bool,
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
    /// Standard OpenAI detail; present when the backend reports real usage.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_tokens_details: Option<PromptTokensDetails>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub completion_tokens_details: Option<CompletionTokensDetails>,
}

#[derive(Debug, Serialize)]
pub struct PromptTokensDetails {
    pub cached_tokens: u32,
}

#[derive(Debug, Serialize)]
pub struct CompletionTokensDetails {
    pub reasoning_tokens: u32,
}

impl WireUsage {
    /// Plain counts, no details.
    pub fn new(prompt_tokens: u32, completion_tokens: u32) -> Self {
        Self {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
            prompt_tokens_details: None,
            completion_tokens_details: None,
        }
    }
}

impl From<sidekick_core::Usage> for WireUsage {
    fn from(u: sidekick_core::Usage) -> Self {
        Self {
            prompt_tokens_details: u.cached_tokens.map(|cached_tokens| PromptTokensDetails { cached_tokens }),
            completion_tokens_details: u
                .reasoning_tokens
                .map(|reasoning_tokens| CompletionTokensDetails { reasoning_tokens }),
            ..Self::new(u.prompt_tokens, u.completion_tokens)
        }
    }
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
    /// Non-standard extension; set on the finish chunk only. See
    /// [`ChatCompletionResponse::constrained`].
    #[serde(skip_serializing_if = "Option::is_none")]
    pub constrained: Option<bool>,
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

// ---------- classify ----------

/// vLLM's `/classify` request (SGLang's `/v1/classify`), plus sidekick's
/// extensions (docs/design/classify.md). Enumerated fields are strings and
/// `input` is raw JSON so that a bad value gets a 400 naming the field.
#[derive(Debug, Deserialize)]
pub struct ClassifyRequest {
    pub model: String,
    #[serde(default)]
    pub input: Option<Value>,
    #[serde(default)]
    pub use_activation: Option<bool>,
    #[serde(default)]
    pub truncate_prompt_tokens: Option<i64>,
    #[serde(default)]
    pub truncation_side: Option<String>,
    #[serde(default)]
    pub add_special_tokens: Option<bool>,
    /// vLLM's chat-form input; not supported.
    #[serde(default)]
    pub messages: Option<Value>,
    #[serde(flatten)]
    pub pooling: PoolingFields,
    // Extensions.
    #[serde(default)]
    pub candidate_labels: Option<Vec<String>>,
    #[serde(default)]
    pub calibration: Option<String>,
    #[serde(default)]
    pub question_type: Option<String>,
    #[serde(default)]
    pub instructions: Option<String>,
    /// gliner2 format: score each label independently (sigmoid `probs`).
    /// The name Hugging Face's zero-shot pipeline uses.
    #[serde(default)]
    pub multi_label: Option<bool>,
}

/// vLLM's pooling fields that classify and rerank share
/// (vllm/entrypoints/pooling/base/protocol.py), checked by
/// `super::pooling::check`.
#[derive(Debug, Default, Deserialize)]
pub struct PoolingFields {
    #[serde(default)]
    pub request_id: Option<Value>,
    #[serde(default)]
    pub priority: Option<Value>,
    #[serde(default)]
    pub padding: Option<Value>,
    #[serde(default)]
    pub cache_salt: Option<Value>,
    #[serde(default)]
    pub mm_processor_kwargs: Option<Value>,
    /// Removed from vLLM, which rejects it in any form, `null` included.
    #[serde(default, deserialize_with = "present")]
    pub normalize: Option<Value>,
    /// vLLM rejects the removed pooling tasks `score` and `encode`.
    #[serde(default)]
    pub task: Option<Value>,
}

/// `Some` whenever the field is present, `null` included (serde's `Option`
/// reads `null` as absent).
fn present<'de, D: serde::Deserializer<'de>>(d: D) -> Result<Option<Value>, D::Error> {
    Value::deserialize(d).map(Some)
}

#[derive(Debug, Serialize)]
pub struct ClassifyResponse {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub data: Vec<ClassifyData>,
    pub usage: WireUsage,
}

#[derive(Debug, Serialize)]
pub struct ClassifyData {
    pub index: usize,
    pub label: String,
    pub probs: Vec<f32>,
    pub num_classes: usize,
}

// ---------- rerank ----------

/// vLLM's `RerankRequest` (the Jina shape), which `/v1/rerank`, `/rerank`
/// and (with Cohere's defaults) `/v2/rerank` read. docs/design/rerank.md.
#[derive(Debug, Deserialize)]
pub struct RerankRequest {
    pub model: String,
    #[serde(default)]
    pub query: Option<Value>,
    #[serde(default)]
    pub documents: Option<Value>,
    #[serde(default)]
    pub top_n: Option<Value>,
    #[serde(default)]
    pub use_activation: Option<bool>,
    #[serde(default)]
    pub truncate_prompt_tokens: Option<i64>,
    #[serde(default)]
    pub truncation_side: Option<String>,
    #[serde(default)]
    pub max_tokens_per_query: Option<i64>,
    #[serde(default)]
    pub max_tokens_per_doc: Option<i64>,
    /// Extension (Jina; Cohere v1).
    #[serde(default)]
    pub return_documents: Option<bool>,
    /// vLLM feeds these to a chat template; cross-encoders have none.
    #[serde(default)]
    pub instruction: Option<Value>,
    #[serde(default)]
    pub chat_template_kwargs: Option<Value>,
    #[serde(flatten)]
    pub pooling: PoolingFields,
}

#[derive(Debug, Serialize)]
pub struct RerankResponse {
    pub id: String,
    pub model: String,
    pub usage: RerankUsage,
    pub results: Vec<RerankResult>,
    /// `/v2/rerank` only: Cohere's.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub meta: Option<CohereMeta>,
}

#[derive(Debug, Serialize)]
pub struct RerankUsage {
    pub prompt_tokens: u32,
    pub total_tokens: u32,
}

#[derive(Debug, Serialize)]
pub struct RerankResult {
    pub index: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub document: Option<RerankDocument>,
    pub relevance_score: f32,
}

#[derive(Debug, Serialize)]
pub struct RerankDocument {
    pub text: String,
}

/// Cohere's v2 `meta`.
#[derive(Debug, Serialize)]
pub struct CohereMeta {
    pub api_version: CohereApiVersion,
    pub billed_units: CohereBilledUnits,
}

#[derive(Debug, Serialize)]
pub struct CohereApiVersion {
    pub version: &'static str,
}

#[derive(Debug, Serialize)]
pub struct CohereBilledUnits {
    pub input_tokens: u32,
}

impl CohereMeta {
    pub fn v2(input_tokens: u32) -> Self {
        Self {
            api_version: CohereApiVersion { version: "2" },
            billed_units: CohereBilledUnits { input_tokens },
        }
    }
}

// ---------- Cohere v2 embed ----------

/// Cohere's `/v2/embed` request, as vLLM serves it.
#[derive(Debug, Deserialize)]
pub struct CohereEmbedRequest {
    pub model: String,
    #[serde(default)]
    pub texts: Option<Vec<String>>,
    #[serde(default)]
    pub images: Option<Value>,
    #[serde(default)]
    pub inputs: Option<Value>,
    #[serde(default)]
    pub input_type: Option<String>,
    #[serde(default)]
    pub embedding_types: Option<Vec<String>>,
    #[serde(default)]
    pub output_dimension: Option<usize>,
    #[serde(default)]
    pub truncate: Option<String>,
    #[serde(default)]
    pub max_tokens: Option<i64>,
    #[serde(default)]
    pub priority: Option<Value>,
}

#[derive(Debug, Serialize)]
pub struct CohereEmbedResponse {
    pub id: String,
    pub embeddings: CohereEmbeddings,
    pub texts: Vec<String>,
    pub meta: CohereMeta,
    pub response_type: &'static str,
}

/// Embeddings by type; only the requested types are present.
#[derive(Debug, Default, Serialize)]
pub struct CohereEmbeddings {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub float: Option<Vec<Vec<f32>>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub base64: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub binary: Option<Vec<Vec<i32>>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ubinary: Option<Vec<Vec<u8>>>,
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
    /// Extension: what the model does, in Hugging Face's pipeline
    /// vocabulary (`text-generation`, `feature-extraction`,
    /// `text-classification`, `zero-shot-classification`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub task: Option<&'static str>,
    /// Extension, text-classification: the labels in `probs` order.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub labels: Option<Vec<String>>,
    /// Extension, zero-shot: most `candidate_labels` per request.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_labels: Option<usize>,
    /// Extension, classifiers: most inputs per request.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_batch: Option<usize>,
    /// Extension, classifiers: the non-standard request fields the model
    /// accepts.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extensions: Option<Vec<&'static str>>,
    /// Extension, classifiers: the extension fields every request must
    /// send, a subset of `extensions`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub required: Option<Vec<&'static str>>,
    /// Extension, embedders and classifiers: the compute units the model is
    /// loaded with (`cpu` for a static model), as `sidekick-compute-units`
    /// reports them.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub compute_units: Option<&'static str>,
    /// Extension, classifiers: the temperatures `calibration: "model"`
    /// applies, keyed `"<question_type>:<label count bucket>"`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub calibration: Option<std::collections::BTreeMap<String, f32>>,
}

impl ModelObject {
    pub fn new(id: String, created: u64, task: &'static str) -> Self {
        Self {
            id,
            object: "model",
            created,
            owned_by: "sidekick",
            task: Some(task),
            labels: None,
            max_labels: None,
            max_batch: None,
            extensions: None,
            required: None,
            compute_units: None,
            calibration: None,
        }
    }
}

pub fn now_unix() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}
