use super::wire::*;
use super::ApiError;
use crate::state::AppState;
use axum::extract::State;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::Json;
use sidekick_core::{
    ChatBackend, ChatMessage, ChatRequest, ChatResponse, DeltaSink, Error, FinishReason, Role,
};
use std::collections::VecDeque;
use std::sync::Arc;
use tokio::sync::mpsc::UnboundedReceiver;
use tokio::task::JoinHandle;

pub async fn chat_completions(
    State(state): State<AppState>,
    Json(req): Json<ChatCompletionRequest>,
) -> Result<Response, ApiError> {
    if req.model != state.chat.id() {
        return Err(ApiError::model_not_found(&req.model));
    }

    let core_req = to_core_request(&req)?;
    let meta = StreamMeta {
        id: format!("chatcmpl-{}", uuid::Uuid::new_v4().simple()),
        created: now_unix(),
        model: req.model.clone(),
        include_usage: req.stream_options.as_ref().map(|o| o.include_usage).unwrap_or(false),
    };

    if req.stream.unwrap_or(false) {
        return stream_completion(state.chat.clone(), core_req, meta).await;
    }

    let response = state.chat.complete(core_req).await?;
    Ok(Json(ChatCompletionResponse {
        id: meta.id,
        object: "chat.completion",
        created: meta.created,
        model: meta.model,
        choices: vec![Choice {
            index: 0,
            message: AssistantMessage { role: "assistant", content: response.content },
            finish_reason: finish_str(response.finish),
        }],
        usage: WireUsage::from(response.usage),
        constrained: response.constrained,
    })
    .into_response())
}

fn finish_str(finish: FinishReason) -> &'static str {
    match finish {
        FinishReason::Stop => "stop",
        FinishReason::Length => "length",
        FinishReason::ContentFilter => "content_filter",
    }
}

struct StreamMeta {
    id: String,
    created: u64,
    model: String,
    include_usage: bool,
}

type SseItem = Result<Event, axum::Error>;

impl StreamMeta {
    fn chunk(&self, choices: Vec<ChunkChoice>, usage: Option<WireUsage>, constrained: Option<bool>) -> SseItem {
        Event::default().json_data(ChatCompletionChunk {
            id: self.id.clone(),
            object: "chat.completion.chunk",
            created: self.created,
            model: self.model.clone(),
            choices,
            usage,
            constrained,
        })
    }

    fn role(&self) -> SseItem {
        self.chunk(
            vec![ChunkChoice {
                index: 0,
                delta: Delta { role: Some("assistant"), content: None },
                finish_reason: None,
            }],
            None,
            None,
        )
    }

    fn content(&self, text: String) -> SseItem {
        self.chunk(
            vec![ChunkChoice {
                index: 0,
                delta: Delta { role: None, content: Some(text) },
                finish_reason: None,
            }],
            None,
            None,
        )
    }

    fn finish(&self, reason: &'static str, constrained: bool) -> SseItem {
        self.chunk(
            vec![ChunkChoice { index: 0, delta: Delta::default(), finish_reason: Some(reason) }],
            None,
            Some(constrained),
        )
    }

    /// The events that close a stream once the backend is done.
    fn closing(&self, result: Result<ChatResponse, Error>) -> Vec<SseItem> {
        match result {
            Ok(response) => {
                let mut events = vec![self.finish(finish_str(response.finish), response.constrained)];
                if self.include_usage {
                    events.push(self.chunk(vec![], Some(WireUsage::from(response.usage)), None));
                }
                events.push(Ok(Event::default().data("[DONE]")));
                events
            }
            // The model's guardrails stopped it mid-reply: an OpenAI
            // content_filter finish, not an error.
            Err(Error::ContentFiltered(_)) => vec![
                self.finish("content_filter", false),
                Ok(Event::default().data("[DONE]")),
            ],
            Err(e) => {
                let error = ApiError::from(e);
                vec![Event::default().json_data(error.body())]
            }
        }
    }
}

/// Stream a completion as OpenAI server-sent events.
///
/// The response is committed only once there is something to send — the
/// first delta, or the backend finishing — so a failure before any text
/// keeps its real HTTP status (503, 400, 429, 504). After that, deltas are
/// forwarded as they arrive; a failure mid-stream becomes an
/// `{"error": ...}` event. When the client disconnects, the stream (and with
/// it the channel receiver) is dropped, the backend's sink starts returning
/// false, and generation stops.
async fn stream_completion(
    chat: Arc<dyn ChatBackend>,
    req: ChatRequest,
    meta: StreamMeta,
) -> Result<Response, ApiError> {
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<String>();
    // Unbounded: the sink is called on the backend's generation thread,
    // which must never block on a slow client.
    let sink: DeltaSink = Box::new(move |delta: &str| tx.send(delta.to_string()).is_ok());
    let mut task: JoinHandle<Result<ChatResponse, Error>> =
        tokio::spawn(async move { chat.complete_stream(req, sink).await });

    let first = tokio::select! {
        biased;
        Some(delta) = rx.recv() => Start::Delta(delta),
        joined = &mut task => Start::Finished(joined_result(joined)),
    };

    let mut queue: VecDeque<SseItem> = VecDeque::from([meta.role()]);
    let (task, finished) = match first {
        Start::Delta(delta) => {
            queue.push_back(meta.content(delta));
            (Some(task), None)
        }
        Start::Finished(Err(e)) if rx.is_empty() => return Err(e.into()),
        Start::Finished(result) => (None, Some(result)),
    };

    let state = SseState { meta, queue, rx, task, finished, done: false };
    let events = futures::stream::unfold(state, |mut st| async move {
        loop {
            if let Some(event) = st.queue.pop_front() {
                return Some((event, st));
            }
            if st.done {
                return None;
            }
            match st.rx.recv().await {
                Some(delta) => st.queue.push_back(st.meta.content(delta)),
                // The sender lives in the backend's sink, dropped when the
                // backend finishes: all deltas are in, collect the result.
                None => {
                    let result = match (st.finished.take(), st.task.take()) {
                        (Some(result), _) => result,
                        (None, Some(task)) => joined_result(task.await),
                        (None, None) => Err(Error::Other("stream finished twice".into())),
                    };
                    st.queue.extend(st.meta.closing(result));
                    st.done = true;
                }
            }
        }
    });
    Ok(Sse::new(events).keep_alive(KeepAlive::default()).into_response())
}

enum Start {
    Delta(String),
    Finished(Result<ChatResponse, Error>),
}

struct SseState {
    meta: StreamMeta,
    queue: VecDeque<SseItem>,
    rx: UnboundedReceiver<String>,
    task: Option<JoinHandle<Result<ChatResponse, Error>>>,
    finished: Option<Result<ChatResponse, Error>>,
    done: bool,
}

fn joined_result(
    joined: Result<Result<ChatResponse, Error>, tokio::task::JoinError>,
) -> Result<ChatResponse, Error> {
    joined.unwrap_or_else(|e| Err(Error::Other(format!("chat task failed: {e}"))))
}

/// Reject parameters this daemon can't honor, rather than answering as if
/// they weren't there (D22). Harmless forms that clients send by default —
/// `n: 1`, `tool_choice: "auto"`/`"none"` without tools, `logprobs: false` —
/// are accepted. Everything else unknown (`seed`, `top_p`, penalties,
/// `user`, …) is ignored on purpose.
fn reject_unsupported(req: &ChatCompletionRequest) -> Result<(), ApiError> {
    if req.n.is_some_and(|n| n > 1) {
        return Err(ApiError::invalid("`n` > 1 is not supported; send separate requests"));
    }
    if req.tools.as_ref().is_some_and(|t| !t.is_empty())
        || req.functions.as_ref().is_some_and(|f| !f.is_empty())
    {
        return Err(ApiError::invalid("tool calling is not supported (this daemon is text-only)"));
    }
    let forces_a_tool = |choice: &Option<serde_json::Value>| match choice {
        None | Some(serde_json::Value::Null) => false,
        Some(serde_json::Value::String(s)) => s != "auto" && s != "none",
        Some(_) => true,
    };
    if forces_a_tool(&req.tool_choice) || forces_a_tool(&req.function_call) {
        return Err(ApiError::invalid("tool calling is not supported (this daemon is text-only)"));
    }
    if req.logprobs == Some(true) || req.top_logprobs.is_some_and(|n| n > 0) {
        return Err(ApiError::invalid("logprobs are not supported"));
    }
    Ok(())
}

fn to_core_request(req: &ChatCompletionRequest) -> Result<ChatRequest, ApiError> {
    if req.messages.is_empty() {
        return Err(ApiError::invalid("`messages` must not be empty"));
    }
    reject_unsupported(req)?;
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

    // Some(0) would collide with the FFI's 0-means-unset sentinel and be
    // silently treated as "no limit"; OpenAI 400s it too.
    let max_tokens = req.max_completion_tokens.or(req.max_tokens);
    if max_tokens == Some(0) {
        return Err(ApiError::invalid(
            "max_tokens (or max_completion_tokens) must be at least 1",
        ));
    }

    let stop = match &req.stop {
        None => Vec::new(),
        Some(StopSequences::One(s)) => vec![s.clone()],
        Some(StopSequences::Many(list)) => list.clone(),
    };
    if stop.len() > 4 {
        return Err(ApiError::invalid("`stop` accepts at most 4 sequences"));
    }
    if stop.iter().any(String::is_empty) {
        return Err(ApiError::invalid("`stop` sequences must not be empty"));
    }
    // Cutting schema-constrained output short would break the schema.
    if !stop.is_empty() && schema.is_some() {
        return Err(ApiError::invalid(
            "`stop` can't be combined with response_format json_schema",
        ));
    }

    Ok(ChatRequest { messages, temperature: req.temperature, max_tokens, schema, stop })
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
