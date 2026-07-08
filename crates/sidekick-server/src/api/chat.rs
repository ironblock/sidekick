use super::wire::*;
use super::ApiError;
use crate::state::AppState;
use axum::extract::State;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::Json;
use futures::stream;
use sidekick_core::{ChatMessage, ChatRequest, FinishReason, Role};

pub async fn chat_completions(
    State(state): State<AppState>,
    Json(req): Json<ChatCompletionRequest>,
) -> Result<Response, ApiError> {
    if req.model != state.chat.id() {
        return Err(ApiError::model_not_found(&req.model));
    }

    let core_req = to_core_request(&req)?;
    let stream_requested = req.stream.unwrap_or(false);
    let include_usage = req
        .stream_options
        .as_ref()
        .map(|o| o.include_usage)
        .unwrap_or(false);

    let response = state.chat.complete(core_req).await?;

    let usage = WireUsage {
        prompt_tokens: response.usage.prompt_tokens,
        completion_tokens: response.usage.completion_tokens,
        total_tokens: response.usage.prompt_tokens + response.usage.completion_tokens,
    };
    let finish = match response.finish {
        FinishReason::Stop => "stop",
        FinishReason::Length => "length",
        FinishReason::ContentFilter => "content_filter",
    };
    let id = format!("chatcmpl-{}", uuid::Uuid::new_v4().simple());
    let created = now_unix();
    let model = req.model.clone();

    if !stream_requested {
        return Ok(Json(ChatCompletionResponse {
            id,
            object: "chat.completion",
            created,
            model,
            choices: vec![Choice {
                index: 0,
                message: AssistantMessage { role: "assistant", content: response.content },
                finish_reason: finish,
            }],
            usage,
        })
        .into_response());
    }

    // Foundation Models responses for sidekick-sized tasks complete in well
    // under a second, so v1 "streams" by emitting the finished completion as
    // one delta. Wire-compatible with every OpenAI streaming client; real
    // token streaming is a shim upgrade later.
    let chunk = |choices: Vec<ChunkChoice>, usage: Option<WireUsage>| ChatCompletionChunk {
        id: id.clone(),
        object: "chat.completion.chunk",
        created,
        model: model.clone(),
        choices,
        usage,
    };

    let mut events = vec![
        chunk(
            vec![ChunkChoice {
                index: 0,
                delta: Delta { role: Some("assistant"), content: None },
                finish_reason: None,
            }],
            None,
        ),
        chunk(
            vec![ChunkChoice {
                index: 0,
                delta: Delta { role: None, content: Some(response.content) },
                finish_reason: None,
            }],
            None,
        ),
        chunk(
            vec![ChunkChoice {
                index: 0,
                delta: Delta::default(),
                finish_reason: Some(finish),
            }],
            None,
        ),
    ];
    if include_usage {
        events.push(chunk(vec![], Some(usage)));
    }

    let stream = stream::iter(
        events
            .into_iter()
            .map(|c| Event::default().json_data(c))
            .chain(std::iter::once(Ok(Event::default().data("[DONE]")))),
    );
    Ok(Sse::new(stream).keep_alive(KeepAlive::default()).into_response())
}

fn to_core_request(req: &ChatCompletionRequest) -> Result<ChatRequest, ApiError> {
    if req.messages.is_empty() {
        return Err(ApiError::invalid("`messages` must not be empty"));
    }
    let mut messages = Vec::with_capacity(req.messages.len());
    for m in &req.messages {
        let role = match m.role.as_str() {
            "system" | "developer" => Role::System,
            "user" => Role::User,
            "assistant" => Role::Assistant,
            other => {
                return Err(ApiError::invalid(format!(
                    "unsupported message role `{other}` (this daemon is text-only)"
                )))
            }
        };
        let content = flatten_content(&m.content)?;
        messages.push(ChatMessage::new(role, content));
    }

    let schema = match &req.response_format {
        None => None,
        Some(f) if f.kind == "text" => None,
        Some(f) if f.kind == "json_schema" => Some(
            f.json_schema
                .as_ref()
                .ok_or_else(|| ApiError::invalid("response_format.json_schema is required"))?
                .schema
                .clone(),
        ),
        // json_object has no schema to constrain against; nudge via prompt.
        Some(f) if f.kind == "json_object" => {
            if let Some(last) = messages.iter_mut().rev().find(|m| m.role == Role::User) {
                last.content
                    .push_str("\n\nRespond with a single valid JSON object and nothing else.");
            }
            None
        }
        Some(f) => {
            return Err(ApiError::invalid(format!(
                "unsupported response_format type `{}`",
                f.kind
            )))
        }
    };

    Ok(ChatRequest {
        messages,
        temperature: req.temperature,
        max_tokens: req.max_completion_tokens.or(req.max_tokens),
        schema,
    })
}

fn flatten_content(content: &WireContent) -> Result<String, ApiError> {
    match content {
        WireContent::Text(s) => Ok(s.clone()),
        WireContent::Null => Ok(String::new()),
        WireContent::Parts(parts) => {
            let mut out = String::new();
            for p in parts {
                if p.kind != "text" {
                    return Err(ApiError::invalid(format!(
                        "unsupported content part type `{}` (text only)",
                        p.kind
                    )));
                }
                if let Some(t) = &p.text {
                    out.push_str(t);
                }
            }
            Ok(out)
        }
    }
}
