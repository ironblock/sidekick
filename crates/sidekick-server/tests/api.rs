//! End-to-end tests of the OpenAI wire surface, using a mock chat backend
//! and a real on-disk static embedding model.

use axum::body::Body;
use axum::http::{Request, StatusCode};
use http_body_util::BodyExt;
use serde_json::{json, Value};
use sidekick_core::{
    Availability, ChatBackend, ChatRequest, ChatResponse, FinishReason, ModelRegistry, Result,
    Usage,
};
use sidekick_server::{build_router, AppState, EmbedderPool};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tower::ServiceExt;

struct MockChat {
    available: bool,
}

#[async_trait::async_trait]
impl ChatBackend for MockChat {
    fn id(&self) -> &str {
        "apple-fm"
    }

    fn context_limit(&self) -> Option<usize> {
        Some(4096)
    }

    async fn availability(&self) -> Availability {
        if self.available {
            Availability::Available
        } else {
            Availability::unavailable(
                sidekick_core::UnavailableReason::AppleIntelligenceNotEnabled,
            )
        }
    }

    async fn complete(&self, req: ChatRequest) -> Result<ChatResponse> {
        if !self.available {
            return Err(sidekick_core::Error::Unavailable(
                sidekick_core::UnavailableReason::AppleIntelligenceNotEnabled,
            ));
        }
        let content = if let Some(schema) = &req.schema {
            format!("{{\"schema_props\": {}}}", schema["properties"].to_string().len())
        } else {
            format!("echo: {}", req.messages.last().unwrap().content)
        };
        Ok(ChatResponse {
            content,
            finish: FinishReason::Stop,
            usage: Usage { prompt_tokens: 10, completion_tokens: 5 },
            constrained: req.schema.is_some(),
        })
    }
}

/// Write the same static-model fixture used in sidekick-embed's unit tests.
fn write_embedding_fixture(dir: &std::path::Path) {
    let model_dir = dir.join("test-static");
    std::fs::create_dir_all(&model_dir).unwrap();
    std::fs::write(
        model_dir.join("manifest.toml"),
        r#"
id = "test-static"
backend = "static"
artifact = "model.safetensors"
tokenizer = "tokenizer.json"
dims = 4
matryoshka = [4, 2]
max_seq_len = 512
"#,
    )
    .unwrap();
    let tokenizer_json = json!({
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": [],
        "normalizer": {"type": "Lowercase"},
        "pre_tokenizer": {"type": "Whitespace"},
        "post_processor": null,
        "decoder": null,
        "model": {
            "type": "WordLevel",
            "vocab": {"hello": 0, "world": 1, "[UNK]": 2},
            "unk_token": "[UNK]"
        }
    });
    std::fs::write(model_dir.join("tokenizer.json"), tokenizer_json.to_string()).unwrap();
    let rows: [[f32; 4]; 3] = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 2.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
    ];
    let bytes: Vec<u8> = rows.iter().flatten().flat_map(|f| f.to_le_bytes()).collect();
    let view =
        safetensors_view(&bytes, vec![3, 4]);
    let data = safetensors::serialize([("embeddings", view)], &None).unwrap();
    std::fs::write(model_dir.join("model.safetensors"), data).unwrap();
}

fn safetensors_view(bytes: &[u8], shape: Vec<usize>) -> safetensors::tensor::TensorView<'_> {
    safetensors::tensor::TensorView::new(safetensors::Dtype::F32, shape, bytes).unwrap()
}

fn test_state(chat_available: bool, api_key: Option<&str>) -> AppState {
    let dir = std::env::temp_dir().join(format!(
        "sk-server-test-{}-{}",
        std::process::id(),
        Instant::now().elapsed().as_nanos()
    ));
    write_embedding_fixture(&dir);
    let registry = ModelRegistry::scan(&dir).unwrap();
    AppState {
        chat: Arc::new(MockChat { available: chat_available }),
        embedders: Arc::new(EmbedderPool::new(registry, Duration::from_secs(60))),
        api_key: api_key.map(Arc::from),
        started_at: Instant::now(),
    }
}

async fn call(state: AppState, req: Request<Body>) -> (StatusCode, Value) {
    let response = build_router(state).oneshot(req).await.unwrap();
    let status = response.status();
    let bytes = response.into_body().collect().await.unwrap().to_bytes();
    let text = String::from_utf8_lossy(&bytes).to_string();
    let value = serde_json::from_str(&text).unwrap_or(Value::String(text));
    (status, value)
}

fn post_json(uri: &str, body: Value) -> Request<Body> {
    Request::post(uri)
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap()
}

#[tokio::test(flavor = "multi_thread")]
async fn chat_completion_round_trip() {
    let (status, body) = call(
        test_state(true, None),
        post_json(
            "/v1/chat/completions",
            json!({
                "model": "apple-fm",
                "messages": [
                    {"role": "system", "content": "you title things"},
                    {"role": "user", "content": "hello"}
                ]
            }),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["object"], "chat.completion");
    assert_eq!(body["choices"][0]["message"]["content"], "echo: hello");
    assert_eq!(body["choices"][0]["finish_reason"], "stop");
    assert_eq!(body["usage"]["total_tokens"], 15);
}

#[tokio::test(flavor = "multi_thread")]
async fn chat_supports_content_parts_and_json_schema() {
    let (status, body) = call(
        test_state(true, None),
        post_json(
            "/v1/chat/completions",
            json!({
                "model": "apple-fm",
                "messages": [
                    {"role": "user", "content": [{"type": "text", "text": "extract this"}]}
                ],
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "title",
                        "schema": {"type": "object", "properties": {"title": {"type": "string"}}}
                    }
                }
            }),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let content = body["choices"][0]["message"]["content"].as_str().unwrap();
    assert!(content.starts_with("{\"schema_props\""), "schema reached backend: {content}");
}

#[tokio::test(flavor = "multi_thread")]
async fn chat_streaming_emits_sse_and_done() {
    let response = build_router(test_state(true, None))
        .oneshot(post_json(
            "/v1/chat/completions",
            json!({
                "model": "apple-fm",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": true
            }),
        ))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let content_type = response.headers()["content-type"].to_str().unwrap().to_string();
    assert!(content_type.starts_with("text/event-stream"), "{content_type}");
    let bytes = response.into_body().collect().await.unwrap().to_bytes();
    let text = String::from_utf8_lossy(&bytes);
    assert!(text.contains("chat.completion.chunk"));
    assert!(text.contains("\"content\":\"echo: hi\""));
    assert!(text.contains("\"finish_reason\":\"stop\""));
    assert!(text.trim_end().ends_with("data: [DONE]"));
}

#[tokio::test(flavor = "multi_thread")]
async fn chat_unavailable_maps_to_503_with_reason() {
    let (status, body) = call(
        test_state(false, None),
        post_json(
            "/v1/chat/completions",
            json!({"model": "apple-fm", "messages": [{"role": "user", "content": "hi"}]}),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(body["error"]["code"], "backend_unavailable");
    assert!(body["error"]["message"]
        .as_str()
        .unwrap()
        .contains("Apple Intelligence"));
}

#[tokio::test(flavor = "multi_thread")]
async fn unknown_model_is_404() {
    let (status, body) = call(
        test_state(true, None),
        post_json(
            "/v1/chat/completions",
            json!({"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]}),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    assert_eq!(body["error"]["code"], "model_not_found");
}

#[tokio::test(flavor = "multi_thread")]
async fn embeddings_round_trip_with_dimensions_and_base64() {
    let state = test_state(true, None);

    let (status, body) = call(
        state.clone(),
        post_json(
            "/v1/embeddings",
            json!({"model": "test-static", "input": ["hello world", "hello"]}),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["data"].as_array().unwrap().len(), 2);
    let v0 = body["data"][0]["embedding"].as_array().unwrap();
    assert_eq!(v0.len(), 4);
    let norm: f64 = v0.iter().map(|x| x.as_f64().unwrap().powi(2)).sum::<f64>().sqrt();
    assert!((norm - 1.0).abs() < 1e-5);

    // Matryoshka truncation via `dimensions`.
    let (status, body) = call(
        state.clone(),
        post_json(
            "/v1/embeddings",
            json!({"model": "test-static", "input": "hello", "dimensions": 2}),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["data"][0]["embedding"].as_array().unwrap().len(), 2);

    // Unsupported dimensions rejected.
    let (status, _) = call(
        state.clone(),
        post_json(
            "/v1/embeddings",
            json!({"model": "test-static", "input": "hello", "dimensions": 3}),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);

    // base64 encoding: 4 f32 = 16 bytes -> 24 base64 chars.
    let (status, body) = call(
        state,
        post_json(
            "/v1/embeddings",
            json!({"model": "test-static", "input": "hello", "encoding_format": "base64"}),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let b64 = body["data"][0]["embedding"].as_str().unwrap();
    assert_eq!(b64.len(), 24);
}

#[tokio::test(flavor = "multi_thread")]
async fn models_and_health_report_state() {
    let (status, body) = call(
        test_state(true, None),
        Request::get("/v1/models").body(Body::empty()).unwrap(),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let ids: Vec<&str> = body["data"]
        .as_array()
        .unwrap()
        .iter()
        .map(|m| m["id"].as_str().unwrap())
        .collect();
    assert!(ids.contains(&"apple-fm"));
    assert!(ids.contains(&"test-static"));

    let (status, body) = call(
        test_state(false, None),
        Request::get("/health").body(Body::empty()).unwrap(),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["chat"]["availability"]["state"], "unavailable");
    assert_eq!(body["chat"]["context_limit"], 4096);
}

#[tokio::test(flavor = "multi_thread")]
async fn api_key_enforced_on_v1_but_not_health() {
    let state = test_state(true, Some("secret"));

    let (status, _) = call(
        state.clone(),
        Request::get("/v1/models").body(Body::empty()).unwrap(),
    )
    .await;
    assert_eq!(status, StatusCode::UNAUTHORIZED);

    let (status, _) = call(
        state.clone(),
        Request::get("/v1/models")
            .header("authorization", "Bearer secret")
            .body(Body::empty())
            .unwrap(),
    )
    .await;
    assert_eq!(status, StatusCode::OK);

    let (status, _) = call(
        state,
        Request::get("/health").body(Body::empty()).unwrap(),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "/health stays open for probes");
}
