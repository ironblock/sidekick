//! An open chat model on the chat route (D53), over a fake decoder: the
//! wire contract (usage, streaming, provenance, the 400s and the busy 503)
//! and what `/v1/models` and `/health` report. The Core ML decoder itself
//! is tested in sidekick-coreml, and the loop in sidekick-llm.

mod common;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use common::*;
use http_body_util::BodyExt;
use serde_json::{json, Value};
use sidekick_core::chat_manifest::{ChatSection, SamplingSection};
use sidekick_core::Result;
use sidekick_llm::{ChatLane, LanedChat, Decoder, DecoderLoader, DecoderProbe, DecoderSpec, LaneOptions, OpenModel, OpenModelSpec, Prompter};
use sidekick_server::chat_models::{self, ChatModels};
use sidekick_server::config::RequestTimeouts;
use sidekick_server::{build_router, AppState};
use std::sync::Arc;
use std::time::Duration;
use tower::ServiceExt;

/// Words, the template's markers, and the special tokens.
const VOCAB: &[&str] = &["[UNK]", "<s>", "</s>", "<|im_start|>", "<|im_end|>", "user", "assistant", "hello", "world", "system"];

fn id(word: &str) -> u32 {
    VOCAB.iter().position(|w| *w == word).unwrap() as u32
}

/// A word-level tokenizer: decoding joins words with spaces.
fn tokenizer() -> tokenizers::Tokenizer {
    let vocab: serde_json::Map<String, Value> = VOCAB.iter().enumerate().map(|(i, w)| (w.to_string(), json!(i))).collect();
    let added: Vec<Value> = ["<s>", "</s>", "<|im_start|>", "<|im_end|>"]
        .iter()
        .map(|t| json!({"id": id(t), "content": t, "single_word": false, "lstrip": false, "rstrip": false,
                        "normalized": false, "special": true}))
        .collect();
    let json = json!({
        "version": "1.0", "truncation": null, "padding": null, "added_tokens": added, "normalizer": null,
        "pre_tokenizer": {"type": "Whitespace"}, "post_processor": null, "decoder": null,
        "model": {"type": "WordLevel", "vocab": vocab, "unk_token": "[UNK]"}
    });
    tokenizers::Tokenizer::from_bytes(serde_json::to_vec(&json).unwrap()).unwrap()
}

const TEMPLATE: &str = "{{ bos_token }}{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{% endfor %}\
                        {% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}";

/// Replies "hello world", then `</s>`, a call every `delay`.
struct Fake {
    delay: Duration,
    prompt: usize,
    logits: Vec<f32>,
}

impl Decoder for Fake {
    fn spec(&self) -> DecoderSpec {
        DecoderSpec { tokens_per_call: 16, context_length: 48, vocab_size: VOCAB.len() }
    }
    fn prefill(&mut self, tokens: &[u32]) -> Result<&[f32]> {
        self.prompt = tokens.len();
        Ok(self.at(0))
    }
    fn step(&mut self, _: u32, pos: usize) -> Result<&[f32]> {
        Ok(self.at(pos + 1 - self.prompt))
    }
    fn probe(&self) -> Arc<dyn DecoderProbe> {
        Arc::new(sidekick_llm::decoder::NoProbe)
    }
}

impl Fake {
    fn at(&mut self, k: usize) -> &[f32] {
        std::thread::sleep(self.delay);
        self.logits.fill(0.0);
        let next = ["hello", "world"].get(k).copied().unwrap_or("</s>");
        self.logits[id(next) as usize] = 1.0;
        &self.logits
    }
}

/// The test state with `tiny`, an open model, beside `apple-fm` in one lane.
fn state(queue_depth: usize, delay: Duration) -> AppState {
    let mut state = test_state(true, None);
    let lane = ChatLane::new(LaneOptions { queue_depth, ..Default::default() });
    let section = ChatSection {
        template: "chat_template.jinja".into(),
        template_kwargs: Default::default(),
        bos_token: Some("<s>".into()),
        eos_token: Some("</s>".into()),
        stop_token_ids: vec![id("</s>")],
    };
    let spec = OpenModelSpec {
        id: "tiny".into(),
        context_length: 48,
        prompter: Arc::new(Prompter::new(tokenizer(), TEMPLATE, &section).unwrap()),
        stop_token_ids: section.stop_token_ids.clone(),
        sampling: SamplingSection { temperature: 0.0, top_p: 1.0 },
        estimated_ane_bytes: 1 << 20,
    };
    let loader: DecoderLoader =
        Arc::new(move || Ok(Box::new(Fake { delay, prompt: 0, logits: vec![0.0; VOCAB.len()] }) as Box<dyn Decoder>));
    let timeout = (Duration::from_secs(30), "manifest");
    let model = OpenModel::new(spec, lane.clone(), loader, timeout.0, Duration::from_secs(30));
    let mut chat = ChatModels::default();
    // `apple-fm` in the lane, as build_state puts it when there are open
    // models.
    let fm_lane = lane.clone();
    chat.insert(chat_models::foundation_models(&RequestTimeouts::default(), move |t| {
        Arc::new(LanedChat::new(Arc::new(MockChat::fm(true)), fm_lane, t)) as Arc<dyn sidekick_core::ChatBackend>
    }))
    .unwrap();
    chat.insert(chat_models::open_model(Arc::new(model), Some("abc123"), timeout)).unwrap();
    chat.set_lane(lane);
    state.chat = Arc::new(chat);
    state
}

fn ask(body: Value) -> Request<Body> {
    let mut body = body;
    body["model"] = json!("tiny");
    if body.get("messages").is_none() {
        body["messages"] = json!([{"role": "user", "content": "hello"}]);
    }
    post_json("/v1/chat/completions", body)
}

#[tokio::test(flavor = "multi_thread")]
async fn answers_with_exact_usage_and_provenance() {
    let (status, headers, body) = call_with_headers(state(4, Duration::ZERO), ask(json!({}))).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["choices"][0]["message"]["content"], "hello world");
    assert_eq!(body["choices"][0]["finish_reason"], "stop");
    // `<s><|im_start|>user hello<|im_end|><|im_start|>assistant`: 7 tokens;
    // `hello world </s>`: 3, the stop token counted.
    assert_eq!(body["usage"]["prompt_tokens"], 7);
    assert_eq!(body["usage"]["completion_tokens"], 3);
    assert_eq!(body["usage"]["total_tokens"], 10);
    assert_eq!(body["constrained"], false);
    assert_eq!(headers["sidekick-model"], "tiny@abc123");
    assert_eq!(headers["sidekick-compute-units"], "cpu_and_ne");

    let (_, body) = call(state(4, Duration::ZERO), ask(json!({"max_tokens": 1}))).await;
    assert_eq!((body["choices"][0]["message"]["content"].as_str(), body["choices"][0]["finish_reason"].as_str()), (Some("hello"), Some("length")));
    let (_, body) = call(state(4, Duration::ZERO), ask(json!({"stop": [" wor"]}))).await;
    assert_eq!((body["choices"][0]["message"]["content"].as_str(), body["choices"][0]["finish_reason"].as_str()), (Some("hello"), Some("stop")));
}

#[tokio::test(flavor = "multi_thread")]
async fn streams_the_routes_events() {
    let response = build_router(state(4, Duration::ZERO))
        .oneshot(ask(json!({"stream": true, "stream_options": {"include_usage": true}})))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let bytes = response.into_body().collect().await.unwrap().to_bytes();
    let data: Vec<String> =
        String::from_utf8_lossy(&bytes).lines().filter_map(|l| l.strip_prefix("data: ").map(str::to_string)).collect();
    assert_eq!(data.last().unwrap(), "[DONE]");
    let chunks: Vec<Value> = data[..data.len() - 1].iter().map(|d| serde_json::from_str(d).unwrap()).collect();
    assert_eq!(chunks[0]["choices"][0]["delta"]["role"], "assistant");
    let text: String = chunks.iter().filter_map(|c| c["choices"][0]["delta"]["content"].as_str()).collect();
    assert_eq!(text, "hello world");
    let finish = &chunks[chunks.len() - 2];
    assert_eq!((finish["choices"][0]["finish_reason"].as_str(), finish["constrained"].as_bool()), (Some("stop"), Some(false)));
    assert_eq!(chunks.last().unwrap()["usage"]["completion_tokens"], 3);
}

#[tokio::test(flavor = "multi_thread")]
async fn refuses_what_it_cant_honor() {
    // Prompt (7) plus max_tokens past the context (48).
    let (status, body) = call(state(4, Duration::ZERO), ask(json!({"max_tokens": 42}))).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(body["error"]["code"], "context_length_exceeded", "{body}");
    let (status, _) = call(state(4, Duration::ZERO), ask(json!({"max_tokens": 41}))).await;
    assert_eq!(status, StatusCode::OK, "exactly the context");
    let long = vec!["hello"; 60].join(" ");
    let (status, body) = call(state(4, Duration::ZERO), ask(json!({"messages": [{"role": "user", "content": long}]}))).await;
    assert_eq!((status, body["error"]["code"].as_str()), (StatusCode::BAD_REQUEST, Some("context_length_exceeded")));
    // Guided output waits for llguidance (#186).
    let schema = json!({"type": "json_schema", "json_schema": {"name": "t", "schema": {"type": "object", "properties": {"a": {"type": "string"}}}}});
    let (status, body) = call(state(4, Duration::ZERO), ask(json!({"response_format": schema}))).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(body["error"]["message"].as_str().unwrap().contains("json_schema"), "{body}");
}

#[tokio::test(flavor = "multi_thread")]
async fn a_full_lane_is_busy_for_open_models_and_apple_fm_alike() {
    let state = state(0, Duration::from_millis(400));
    let router = build_router(state.clone());
    let running = tokio::spawn(router.clone().oneshot(ask(json!({}))));
    let lane = state.chat.lane().unwrap().clone();
    let started = std::time::Instant::now();
    while lane.health()["running"].is_null() {
        assert!(started.elapsed() < Duration::from_secs(10));
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    for model in ["tiny", "apple-fm"] {
        let request = post_json("/v1/chat/completions", json!({"model": model, "messages": [{"role": "user", "content": "x"}]}));
        let response = router.clone().oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE, "{model}");
        assert_eq!(response.headers()["retry-after"], "5");
        let body: Value = serde_json::from_slice(&response.into_body().collect().await.unwrap().to_bytes()).unwrap();
        assert_eq!(body["error"]["code"], "backend_busy");
    }
    assert_eq!(running.await.unwrap().unwrap().status(), StatusCode::OK);
}

#[tokio::test(flavor = "multi_thread")]
async fn models_and_health_report_the_open_model() {
    let state = state(3, Duration::ZERO);
    let (_, models) = call(state.clone(), Request::get("/v1/models").body(Body::empty()).unwrap()).await;
    let tiny = models["data"].as_array().unwrap().iter().find(|m| m["id"] == "tiny").expect("listed");
    assert_eq!(tiny["task"], "text-generation");
    assert_eq!((tiny["backend"].as_str(), tiny["compute_units"].as_str()), (Some("coreml"), Some("cpu_and_ne")));
    assert_eq!(tiny["context_length"], 48);
    assert_eq!(tiny["guardrails"], false);
    assert!(tiny["honored_fields"].as_array().unwrap().contains(&json!("seed")));

    let (status, _) = call(state.clone(), ask(json!({}))).await;
    assert_eq!(status, StatusCode::OK);
    let (_, health) = call(state, Request::get("/health").body(Body::empty()).unwrap()).await;
    let tiny = &health["chat"]["models"]["tiny"];
    assert_eq!(tiny["backend"], "coreml");
    assert_eq!((tiny["state"].as_str(), tiny["resident"].as_bool()), (Some("ready"), Some(true)));
    assert_eq!((tiny["queue_depth"].as_u64(), tiny["requests"].as_u64()), (Some(3), Some(1)));
    assert_eq!(tiny["ane"]["estimated_bytes"], 1 << 20);
    assert_eq!(tiny["request_timeout_secs"], 30);
    let lane = &health["chat"]["lane"];
    assert_eq!(lane["resident"]["model"], "tiny");
    assert_eq!(lane["ane"]["budget_mib"], 2816);
}
