//! End-to-end tests of the OpenAI wire surface, using a mock chat backend
//! and a real on-disk static embedding model.

use axum::body::Body;
use axum::http::{Request, StatusCode};
use http_body_util::BodyExt;
use serde_json::{json, Value};
use sidekick_core::{
    Availability, ChatBackend, ChatRequest, ChatResponse, FinishReason, ModelInfo, ModelRegistry,
    Result, Usage,
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

    async fn model_info(&self) -> Option<ModelInfo> {
        self.available.then(|| ModelInfo {
            variant: Some("AFM 3 Core".into()),
            variant_id: Some("core3".into()),
            context_size: Some(4096),
            capabilities: Some(vec!["guided_generation".into(), "tool_calling".into()]),
        })
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
        // A last message of the form `error:<kind>` makes the mock fail with
        // that error, to exercise the HTTP error mapping.
        if let Some(kind) = req.messages.last().unwrap().content.strip_prefix("error:") {
            use sidekick_core::Error;
            return Err(match kind {
                "overflow" => Error::ContextOverflow { limit: 4096, actual: Some(4459) },
                "overflow_unknown" => Error::ContextOverflow { limit: 4096, actual: None },
                "content_filter" => Error::ContentFiltered("guardrailViolation".into()),
                "rate_limited" => Error::RateLimited {
                    message: "slow down".into(),
                    retry_after_secs: Some(7),
                },
                "transient" => Error::Transient("concurrentRequests".into()),
                "language" => Error::UnsupportedLanguage("xx".into()),
                "guided" => Error::GuidedGeneration("unsupported JSON schema type: null".into()),
                "not_ready" => Error::Unavailable(sidekick_core::UnavailableReason::ModelNotReady),
                other => Error::Inference(other.into()),
            });
        }
        let content = if let Some(schema) = &req.schema {
            format!("{{\"schema_props\": {}}}", schema["properties"].to_string().len())
        } else if !req.stop.is_empty() {
            format!("stops: {}", req.stop.join("|"))
        } else {
            format!("echo: {}", req.messages.last().unwrap().content)
        };
        Ok(ChatResponse {
            content,
            finish: FinishReason::Stop,
            usage: Usage {
                prompt_tokens: 10,
                completion_tokens: 5,
                cached_tokens: Some(3),
                reasoning_tokens: None,
            },
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
    // A process-wide counter, not a timestamp: `Instant::now().elapsed()` is
    // ~0ns and collided across concurrently-running tests, letting one test
    // scan another's half-written fixture (observed as a ~1-in-5 flake).
    static FIXTURE_SEQ: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let dir = std::env::temp_dir().join(format!(
        "sk-server-test-{}-{}",
        std::process::id(),
        FIXTURE_SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    ));
    write_embedding_fixture(&dir);
    let registry = ModelRegistry::scan(&dir).unwrap();
    AppState {
        chat: Arc::new(MockChat { available: chat_available }),
        embedders: Arc::new(EmbedderPool::new(registry, Duration::from_secs(60))),
        api_key: api_key.map(Arc::from),
        started_at: Instant::now(),
        request_timeout: Duration::from_secs(60),
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
    assert_eq!(body["usage"]["prompt_tokens_details"]["cached_tokens"], 3);
    assert!(
        body["usage"].get("completion_tokens_details").is_none(),
        "no reasoning detail when none was reported"
    );
    assert_eq!(body["constrained"], false, "extension field present and honest");
}

#[tokio::test(flavor = "multi_thread")]
async fn chat_rejects_zero_max_tokens() {
    let (status, body) = call(
        test_state(true, None),
        post_json(
            "/v1/chat/completions",
            json!({
                "model": "apple-fm",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 0
            }),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(body["error"]["message"].as_str().unwrap().contains("max_tokens"));
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
    assert!(body["chat"]["variant"].is_null(), "no model info while unavailable");
    assert_eq!(body["chat"]["fm_sdk"], sidekick_fm::FM_SDK);

    let (_, body) = call(
        test_state(true, None),
        Request::get("/health").body(Body::empty()).unwrap(),
    )
    .await;
    assert_eq!(body["chat"]["variant"], "AFM 3 Core");
    assert_eq!(body["chat"]["variant_id"], "core3");
    assert_eq!(body["chat"]["model_capabilities"], json!(["guided_generation", "tool_calling"]));
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

#[tokio::test(flavor = "multi_thread")]
async fn chat_errors_map_to_openai_statuses() {
    for (kind, status, code) in [
        ("overflow", StatusCode::BAD_REQUEST, "context_length_exceeded"),
        ("overflow_unknown", StatusCode::BAD_REQUEST, "context_length_exceeded"),
        ("content_filter", StatusCode::BAD_REQUEST, "content_filter"),
        ("rate_limited", StatusCode::TOO_MANY_REQUESTS, "rate_limit_exceeded"),
        ("transient", StatusCode::SERVICE_UNAVAILABLE, "backend_busy"),
        ("language", StatusCode::BAD_REQUEST, "unsupported_language"),
        ("guided", StatusCode::BAD_REQUEST, "unsupported_schema"),
        ("not_ready", StatusCode::SERVICE_UNAVAILABLE, "backend_unavailable"),
        ("anything else", StatusCode::INTERNAL_SERVER_ERROR, "internal_error"),
    ] {
        let (got, body) = call(
            test_state(true, None),
            post_json(
                "/v1/chat/completions",
                json!({"model": "apple-fm", "messages": [{"role": "user", "content": format!("error:{kind}")}]}),
            ),
        )
        .await;
        assert_eq!(got, status, "{kind}: {body}");
        assert_eq!(body["error"]["code"], code, "{kind}: {body}");
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn overflow_reports_real_counts_when_known() {
    let (_, body) = call(
        test_state(true, None),
        post_json(
            "/v1/chat/completions",
            json!({"model": "apple-fm", "messages": [{"role": "user", "content": "error:overflow"}]}),
        ),
    )
    .await;
    let message = body["error"]["message"].as_str().unwrap();
    assert!(message.contains("4459") && message.contains("4096"), "{message}");
}

#[tokio::test(flavor = "multi_thread")]
async fn rate_limit_sets_retry_after_and_type() {
    let response = build_router(test_state(true, None))
        .oneshot(post_json(
            "/v1/chat/completions",
            json!({"model": "apple-fm", "messages": [{"role": "user", "content": "error:rate_limited"}]}),
        ))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
    assert_eq!(response.headers()["retry-after"], "7");
    let bytes = response.into_body().collect().await.unwrap().to_bytes();
    let body: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(body["error"]["type"], "rate_limit_error");
}

#[tokio::test(flavor = "multi_thread")]
async fn stop_is_accepted_as_string_or_list_and_validated() {
    let chat = |extra: Value| {
        let mut body = json!({"model": "apple-fm", "messages": [{"role": "user", "content": "count"}]});
        body.as_object_mut().unwrap().extend(extra.as_object().unwrap().clone());
        post_json("/v1/chat/completions", body)
    };
    let (status, body) = call(test_state(true, None), chat(json!({"stop": "5"}))).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["choices"][0]["message"]["content"], "stops: 5", "reaches the backend");

    let (status, body) = call(test_state(true, None), chat(json!({"stop": ["a", "b"]}))).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["choices"][0]["message"]["content"], "stops: a|b");

    let (status, _) = call(test_state(true, None), chat(json!({"stop": null}))).await;
    assert_eq!(status, StatusCode::OK, "null means no stop sequences");

    for bad in [
        json!({"stop": ["1", "2", "3", "4", "5"]}),
        json!({"stop": ""}),
        json!({"stop": ["ok", ""]}),
        json!({"stop": "x", "response_format": {"type": "json_schema",
            "json_schema": {"name": "t", "schema": {"type": "object"}}}}),
    ] {
        let (status, body) = call(test_state(true, None), chat(bad.clone())).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{bad}: {body}");
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn parameters_that_cannot_be_honored_are_rejected() {
    let chat = |extra: Value| {
        let mut body = json!({"model": "apple-fm", "messages": [{"role": "user", "content": "hi"}]});
        body.as_object_mut().unwrap().extend(extra.as_object().unwrap().clone());
        post_json("/v1/chat/completions", body)
    };
    // What SDKs commonly send by default must keep working.
    for ok in [
        json!({"n": 1}),
        json!({"tools": []}),
        json!({"tool_choice": "auto"}),
        json!({"tool_choice": "none"}),
        json!({"tool_choice": null}),
        json!({"function_call": "none"}),
        json!({"logprobs": false}),
        json!({"top_logprobs": 0}),
        json!({"seed": 7, "top_p": 0.9, "presence_penalty": 0, "user": "x"}),
    ] {
        let (status, body) = call(test_state(true, None), chat(ok.clone())).await;
        assert_eq!(status, StatusCode::OK, "{ok}: {body}");
    }
    for bad in [
        json!({"n": 2}),
        json!({"tools": [{"type": "function", "function": {"name": "f"}}]}),
        json!({"functions": [{"name": "f"}]}),
        json!({"tool_choice": "required"}),
        json!({"tool_choice": {"type": "function", "function": {"name": "f"}}}),
        json!({"function_call": {"name": "f"}}),
        json!({"logprobs": true}),
        json!({"top_logprobs": 3}),
    ] {
        let (status, body) = call(test_state(true, None), chat(bad.clone())).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{bad}: {body}");
    }
}
