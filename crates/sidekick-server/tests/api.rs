//! End-to-end tests of the OpenAI wire surface, using a mock chat backend
//! and a real on-disk static embedding model.

mod common;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use common::*;
use http_body_util::BodyExt;
use serde_json::{json, Value};
use sidekick_server::build_router;
use std::sync::Arc;
use tower::ServiceExt;

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
    // Core ML models list each bucket's compile-cache state (D52); nothing
    // has loaded them, and without a ledger nothing records them warm.
    let model = |id: &str| body["data"].as_array().unwrap().iter().find(|m| m["id"] == id).unwrap().clone();
    assert!(model("test-static").get("compile_cache").is_none(), "a static model has no compile cache");
    let sentiment = model("sentiment");
    if sidekick_embed::CLASSIFIERS_SUPPORTED {
        let buckets = sentiment["compile_cache"].as_object().expect("a Core ML classifier's buckets");
        assert!(!buckets.is_empty());
        assert!(buckets.values().all(|s| s == "cold"), "{buckets:?}");
    } else {
        assert!(sentiment.get("compile_cache").is_none(), "a build without Core ML can't say");
    }

    let (status, body) = call(
        test_state(false, None),
        Request::get("/health").body(Body::empty()).unwrap(),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let fm = &body["chat"]["models"]["apple-fm"];
    assert_eq!(fm["availability"], json!({"state": "unavailable", "reason": "apple_intelligence_not_enabled"}));
    assert_eq!(fm["context_limit"], 4096);
    assert!(fm["variant"].is_null(), "no model info while unavailable");
    assert_eq!(fm["fm_sdk"], sidekick_fm::FM_SDK);
    // No startup warm-up in a state built for tests.
    assert_eq!(body["warmup"]["state"], "off");

    let (_, body) = call(
        test_state(true, None),
        Request::get("/health").body(Body::empty()).unwrap(),
    )
    .await;
    let fm = &body["chat"]["models"]["apple-fm"];
    assert_eq!(fm["variant"], "AFM 3 Core");
    assert_eq!(fm["variant_id"], "core3");
    assert_eq!(fm["model_capabilities"], json!(["guided_generation", "tool_calling"]));
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

/// POST a streaming chat request; returns status, content type, and the
/// SSE `data:` payloads in order.
async fn stream_chat(content: &str, include_usage: bool) -> (StatusCode, String, Vec<String>) {
    let response = build_router(test_state(true, None))
        .oneshot(post_json(
            "/v1/chat/completions",
            json!({
                "model": "apple-fm",
                "messages": [{"role": "user", "content": content}],
                "stream": true,
                "stream_options": {"include_usage": include_usage}
            }),
        ))
        .await
        .unwrap();
    let status = response.status();
    let content_type = response.headers()["content-type"].to_str().unwrap().to_string();
    let bytes = response.into_body().collect().await.unwrap().to_bytes();
    let data = String::from_utf8_lossy(&bytes)
        .lines()
        .filter_map(|l| l.strip_prefix("data: ").map(str::to_string))
        .collect();
    (status, content_type, data)
}

#[tokio::test(flavor = "multi_thread")]
async fn stream_forwards_each_delta_then_finish_usage_and_done() {
    let (status, _, data) = stream_chat("stream:Hel|lo| world", true).await;
    assert_eq!(status, StatusCode::OK);
    let chunks: Vec<Value> = data[..data.len() - 1].iter().map(|d| serde_json::from_str(d).unwrap()).collect();
    assert_eq!(chunks[0]["choices"][0]["delta"]["role"], "assistant");
    let contents: Vec<&str> = chunks
        .iter()
        .filter_map(|c| c["choices"][0]["delta"]["content"].as_str())
        .collect();
    assert_eq!(contents, vec!["Hel", "lo", " world"], "one chunk per delta, in order");
    let finish = chunks.iter().find(|c| !c["choices"][0]["finish_reason"].is_null()).unwrap();
    assert_eq!(finish["choices"][0]["finish_reason"], "stop");
    let usage = chunks.last().unwrap();
    assert_eq!(usage["usage"]["total_tokens"], 13);
    assert_eq!(usage["usage"]["prompt_tokens_details"]["cached_tokens"], 4);
    assert_eq!(data.last().unwrap(), "[DONE]");
}

#[tokio::test(flavor = "multi_thread")]
async fn stream_failure_before_any_text_keeps_its_status() {
    let (status, content_type, _) = stream_chat("error:rate_limited", false).await;
    assert_eq!(status, StatusCode::TOO_MANY_REQUESTS);
    assert!(content_type.starts_with("application/json"), "{content_type}");
    let (status, _, _) = stream_chat("error:overflow", false).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
}

#[tokio::test(flavor = "multi_thread")]
async fn stream_failure_after_text_is_an_error_event() {
    let (status, _, data) = stream_chat("stream:partial|FAIL", false).await;
    assert_eq!(status, StatusCode::OK, "already committed");
    assert!(data.iter().any(|d| d.contains("\"content\":\"partial\"")));
    let last: Value = serde_json::from_str(data.last().unwrap()).unwrap();
    assert_eq!(last["error"]["code"], "internal_error");
    assert!(!data.iter().any(|d| d == "[DONE]"), "a failed stream doesn't claim completion");
}

#[tokio::test(flavor = "multi_thread")]
async fn stream_guardrail_stop_is_a_content_filter_finish() {
    let (_, _, data) = stream_chat("stream:partial|FILTER", false).await;
    let finish: Value = serde_json::from_str(&data[data.len() - 2]).unwrap();
    assert_eq!(finish["choices"][0]["finish_reason"], "content_filter");
    assert_eq!(data.last().unwrap(), "[DONE]");
}

/// The test state serving `apple-fm` and a second chat model, `test-llm`.
fn two_chat_models() -> sidekick_server::AppState {
    let mut state = test_state(true, None);
    state.chat = chat_of([mock_fm(MockChat::fm(true)), mock_open_model("test-llm")]);
    state
}

fn ask(model: &str) -> Request<Body> {
    post_json("/v1/chat/completions", json!({"model": model, "messages": [{"role": "user", "content": "hi"}]}))
}

#[tokio::test(flavor = "multi_thread")]
async fn chat_routes_by_the_exact_model_id() {
    for stream in [false, true] {
        for (model, reply, provenance, units) in
            [("apple-fm", "echo: hi", "core3", None), ("test-llm", "test-llm: hi", "test-llm", Some("cpu_and_ne"))]
        {
            let state = two_chat_models();
            state.chat.get(model).unwrap().refresh_provenance().await;
            let request = post_json(
                "/v1/chat/completions",
                json!({"model": model, "stream": stream, "messages": [{"role": "user", "content": "hi"}]}),
            );
            let (status, headers, body) = call_with_headers(state, request).await;
            assert_eq!(status, StatusCode::OK, "{model}: {body}");
            if stream {
                assert!(body.as_str().unwrap().contains(&format!("\"content\":\"{reply}\"")), "{model}: {body}");
            } else {
                assert_eq!(body["choices"][0]["message"]["content"], reply, "{model} answered");
                assert_eq!(body["model"], model);
            }
            assert_eq!(headers["sidekick-model"], provenance);
            assert_eq!(headers.get("sidekick-compute-units").map(|v| v.to_str().unwrap()), units);
        }
    }
    // No default, alias or near match: the id must be exact.
    for model in ["", "Apple-FM", "apple-fm ", "apple", "test-llm@main", "gpt-4o"] {
        let (status, body) = call(two_chat_models(), ask(model)).await;
        assert_eq!(status, StatusCode::NOT_FOUND, "{model:?}: {body}");
        assert_eq!(
            body,
            json!({"error": {
                "message": format!("The model `{model}` does not exist or is not loaded"),
                "type": "invalid_request_error",
                "code": "model_not_found",
            }}),
            "OpenAI's shape"
        );
    }
    let no_model = post_json("/v1/chat/completions", json!({"messages": [{"role": "user", "content": "hi"}]}));
    let (status, body) = call(two_chat_models(), no_model).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "`model` is required: {body}");
}

#[tokio::test(flavor = "multi_thread")]
async fn another_routes_model_is_a_400_naming_its_route() {
    let others = [("test-static", "/v1/embeddings"), ("sentiment", "/v1/classify"), ("reranker", "/v1/rerank")];
    for (model, route) in others {
        let (status, body) = call(two_chat_models(), ask(model)).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{model}: {body}");
        assert_eq!(body["error"]["code"], "invalid_request_error");
        let message = body["error"]["message"].as_str().unwrap();
        assert!(message.contains(&format!("use {route}")), "{model}: {message}");
    }
    // And a chat model sent elsewhere names the chat route.
    let (status, body) =
        call(two_chat_models(), post_json("/v1/embeddings", json!({"model": "test-llm", "input": "hi"}))).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    assert!(body["error"]["message"].as_str().unwrap().contains("use /v1/chat/completions"), "{body}");
    let tokenize = post_json("/tokenize", json!({"model": "test-llm", "prompt": "hi"}));
    let (status, body) = call(two_chat_models(), tokenize).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
}

#[tokio::test(flavor = "multi_thread")]
async fn models_lists_each_chat_model_with_what_it_honors() {
    let (status, body) = call(two_chat_models(), Request::get("/v1/models").body(Body::empty()).unwrap()).await;
    assert_eq!(status, StatusCode::OK);
    let model = |id: &str| body["data"].as_array().unwrap().iter().find(|m| m["id"] == id).unwrap().clone();
    let mut fm = model("apple-fm");
    fm.as_object_mut().unwrap().remove("created");
    assert_eq!(
        fm,
        json!({
            "id": "apple-fm",
            "object": "model",
            "owned_by": "sidekick",
            "task": "text-generation",
            "backend": "foundation_models",
            "context_length": 4096,
            "honored_fields": ["max_completion_tokens", "max_tokens", "response_format", "stop", "temperature"],
            "guided_streaming": false,
            "guardrails": true,
        })
    );
    let open = model("test-llm");
    assert_eq!(open["task"], "text-generation");
    assert_eq!(open["backend"], "coreml");
    assert_eq!(open["guided_streaming"], true);
    assert_eq!(open["guardrails"], false);
    assert_eq!(open["compute_units"], "cpu_and_ne");
    assert_eq!(open["graded_os"], "macOS 27");
}

#[tokio::test(flavor = "multi_thread")]
async fn health_reports_each_chat_models_availability_and_timeout() {
    let mut state = two_chat_models();
    state.chat = chat_of([mock_fm(MockChat::fm(false)), mock_open_model("test-llm")]);
    let (_, body) = call(state, Request::get("/health").body(Body::empty()).unwrap()).await;
    let models = body["chat"]["models"].as_object().unwrap();
    assert_eq!(models.keys().collect::<Vec<_>>(), ["apple-fm", "test-llm"]);
    let fm = &models["apple-fm"];
    assert_eq!(fm["backend"], "foundation_models");
    assert_eq!(fm["availability"]["state"], "unavailable");
    assert_eq!(fm["availability"]["reason"], "apple_intelligence_not_enabled");
    assert_eq!(fm["request_timeout_secs"], 60);
    assert_eq!(fm["request_timeout_source"], "config");
    let open = &models["test-llm"];
    assert_eq!(open["availability"], json!({"state": "available"}));
    assert_eq!(open["request_timeout_secs"], 120);
    assert_eq!(open["request_timeout_source"], "manifest");
    assert!(open.get("fm_sdk").is_none(), "Foundation Models' facts are its own");
    assert_eq!(body["request_timeout_overrides"], json!({"default_secs": 60, "applied": {}, "unmatched": []}));
}

#[tokio::test(flavor = "multi_thread")]
async fn a_guardrail_refusal_says_whether_output_was_guided() {
    let refused = |extra: Value| {
        let mut body = json!({"model": "apple-fm", "messages": [{"role": "user", "content": "error:content_filter"}]});
        body.as_object_mut().unwrap().extend(extra.as_object().unwrap().clone());
        post_json("/v1/chat/completions", body)
    };
    let schema = json!({"response_format": {"type": "json_schema",
        "json_schema": {"name": "t", "schema": {"type": "object", "properties": {"t": {"type": "string"}}}}}});
    for stream in [false, true] {
        let mut guided = schema.clone();
        guided["stream"] = json!(stream);
        let (status, body) = call(test_state(true, None), refused(guided)).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "guided output arrives whole, so a refusal keeps its status");
        assert_eq!(body["error"]["code"], "content_filter");
        assert_eq!(body["error"]["param"], "response_format");
        let message = body["error"]["message"].as_str().unwrap();
        let guided = "`apple-fm`'s guardrails refused this request, whose output was guided";
        assert!(message.starts_with(guided), "{message}");
        assert!(message.contains("guardrailViolation"), "the backend's reason: {message}");
        assert!(message.contains("a retry without response_format"), "{message}");

        let (status, body) = call(test_state(true, None), refused(json!({"stream": stream}))).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(body["error"]["code"], "content_filter");
        assert!(body["error"].get("param").is_none(), "{body}");
        let message = body["error"]["message"].as_str().unwrap();
        assert!(message.contains("whose output was plain text: guardrailViolation"), "{message}");
    }
    // Mid-stream, it stays a content_filter finish, not guided.
    let (_, _, data) = stream_chat("stream:partial|FILTER", false).await;
    let finish: Value = serde_json::from_str(&data[data.len() - 2]).unwrap();
    assert_eq!(finish["choices"][0]["finish_reason"], "content_filter");
    assert_eq!(finish["constrained"], false);
}

/// A Foundation Models engine whose every reply takes `delay`.
struct SlowEngine {
    delay: std::time::Duration,
}

impl sidekick_fm::SessionEngine for SlowEngine {
    type Session = ();

    fn availability(&self) -> sidekick_core::Availability {
        sidekick_core::Availability::Available
    }

    fn create(&self, _instructions: &str) -> sidekick_core::Result<()> {
        Ok(())
    }

    fn respond(
        &self,
        _session: &mut (),
        prompt: &str,
        _opts: &sidekick_fm::RespondOptions,
    ) -> sidekick_core::Result<sidekick_fm::EngineResponse> {
        std::thread::sleep(self.delay);
        Ok(sidekick_fm::EngineResponse::text(format!("slow: {prompt}")))
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn a_per_model_request_timeout_reaches_the_chat_backend() {
    use std::time::Duration;
    // A reply takes 3 s: inside the global 10 s, past apple-fm's own 1 s.
    let state_for = |config: &str| {
        let config: sidekick_server::Config = toml::from_str(config).unwrap();
        let timeouts = config.request_timeouts().unwrap();
        let fm = sidekick_server::chat_models::foundation_models(&timeouts, |timeout| {
            let engine = SlowEngine { delay: Duration::from_secs(3) };
            Arc::new(sidekick_fm::SessionChatBackend::new(engine, Duration::from_secs(60), timeout))
        });
        let mut state = test_state(true, None);
        state.chat = chat_of([fm]);
        state.request_timeouts = timeouts;
        state
    };
    let overridden = "request_timeout_secs = 10\n\
                      [models.\"apple-fm\"]\nrequest_timeout_secs = 1\n\
                      [models.nope]\nrequest_timeout_secs = 5\n";
    for stream in [false, true] {
        let request = post_json(
            "/v1/chat/completions",
            json!({"model": "apple-fm", "stream": stream, "messages": [{"role": "user", "content": "hi"}]}),
        );
        let started = std::time::Instant::now();
        let (status, body) = call(state_for(overridden), request).await;
        assert_eq!(status, StatusCode::GATEWAY_TIMEOUT, "stream {stream}: {body}");
        assert_eq!(body["error"]["code"], "timeout");
        assert_eq!(body["error"]["message"], "Generation did not complete within 1s");
        assert!(started.elapsed() < Duration::from_secs(3), "the model's 1 s, not the reply's 3 s");
    }
    let (status, body) = call(state_for("request_timeout_secs = 10\n"), ask("apple-fm")).await;
    assert_eq!(status, StatusCode::OK, "the global timeout without an override: {body}");
    assert_eq!(body["choices"][0]["message"]["content"], "slow: hi");

    let (_, body) = call(state_for(overridden), Request::get("/health").body(Body::empty()).unwrap()).await;
    let fm = &body["chat"]["models"]["apple-fm"];
    assert_eq!(fm["request_timeout_secs"], 1);
    assert_eq!(fm["request_timeout_source"], "config (model)");
    assert_eq!(
        body["request_timeout_overrides"],
        json!({"default_secs": 10, "applied": {"apple-fm": 1}, "unmatched": ["nope"]})
    );
}
