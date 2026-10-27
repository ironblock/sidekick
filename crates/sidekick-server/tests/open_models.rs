//! An open chat model on the chat route (D53), over a fake decoder: the
//! wire contract (usage, streaming, provenance, guided output, the 400s and
//! the busy 503) and what `/v1/models` and `/health` report. The Core ML decoder itself
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
    fn step(&mut self, tokens: &[u32], pos: usize) -> Result<&[f32]> {
        Ok(self.at(pos + tokens.len() - self.prompt))
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

/// A byte-level tokenizer (one token a byte, GPT-2's byte map) with the
/// template's special tokens from 256: llguidance reads it, as it reads a
/// real model's.
fn byte_tokenizer() -> tokenizers::Tokenizer {
    let mut printable: Vec<u32> = (b'!' as u32..=b'~' as u32).chain(0xA1..=0xAC).chain(0xAE..=0xFF).collect();
    let mut chars = printable.clone();
    let mut n = 0;
    for b in 0..256u32 {
        if !printable.contains(&b) {
            printable.push(b);
            chars.push(256 + n);
            n += 1;
        }
    }
    let mut vocab = serde_json::Map::new();
    for (b, c) in printable.iter().zip(&chars) {
        vocab.insert(char::from_u32(*c).unwrap().to_string(), json!(b));
    }
    let specials = ["<s>", "</s>", "<|im_start|>", "<|im_end|>"];
    let added: Vec<Value> = specials
        .iter()
        .enumerate()
        .map(|(i, t)| json!({"id": 256 + i, "content": t, "single_word": false, "lstrip": false, "rstrip": false,
                            "normalized": false, "special": true}))
        .collect();
    for (i, t) in specials.iter().enumerate() {
        vocab.insert(t.to_string(), json!(256 + i));
    }
    let json = json!({
        "version": "1.0", "truncation": null, "padding": null, "added_tokens": added, "normalizer": null,
        "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true, "use_regex": false},
        "post_processor": null,
        "decoder": {"type": "ByteLevel", "add_prefix_space": true, "trim_offsets": true, "use_regex": true},
        "model": {"type": "BPE", "dropout": null, "unk_token": null, "continuing_subword_prefix": null,
                  "end_of_word_suffix": null, "fuse_unk": false, "byte_fallback": false, "ignore_merges": false,
                  "vocab": vocab, "merges": []}
    });
    tokenizers::Tokenizer::from_bytes(serde_json::to_vec(&json).unwrap()).unwrap()
}

/// A decoder over the byte tokenizer whose every row ranks the same bytes
/// first: it closes strings, arrays and objects as soon as a grammar allows,
/// else writes a comma, `1`, `b`, `x`, in that order.
struct Prefers {
    logits: Vec<f32>,
}

impl Prefers {
    fn new() -> Self {
        let mut logits: Vec<f32> = (0..260).map(|i| -(i as f32)).collect();
        for (rank, b) in "\"}],1bx".bytes().enumerate() {
            logits[b as usize] = 1000.0 - rank as f32;
        }
        Self { logits }
    }
}

impl Decoder for Prefers {
    fn spec(&self) -> DecoderSpec {
        DecoderSpec { tokens_per_call: 16, context_length: 1024, vocab_size: 260 }
    }
    fn prefill(&mut self, _: &[u32]) -> Result<&[f32]> {
        Ok(&self.logits)
    }
    fn step(&mut self, _: &[u32], _: usize) -> Result<&[f32]> {
        Ok(&self.logits)
    }
    fn probe(&self) -> Arc<dyn DecoderProbe> {
        Arc::new(sidekick_llm::decoder::NoProbe)
    }
}

fn section(stop: u32) -> ChatSection {
    ChatSection {
        template: "chat_template.jinja".into(),
        template_kwargs: Default::default(),
        bos_token: Some("<s>".into()),
        eos_token: Some("</s>".into()),
        stop_token_ids: vec![stop],
    }
}

/// The test state with two open models beside `apple-fm` in one lane:
/// `tiny`, over a word-level tokenizer, and `bytes`, over a byte-level one,
/// for guided output.
fn state(queue_depth: usize, delay: Duration) -> AppState {
    let mut state = test_state(true, None);
    let lane = ChatLane::new(LaneOptions { queue_depth, ..Default::default() });
    let section_tiny = section(id("</s>"));
    let spec = OpenModelSpec {
        id: "tiny".into(),
        context_length: 48,
        prompter: Arc::new(Prompter::new(tokenizer(), TEMPLATE, &section_tiny).unwrap()),
        stop_token_ids: section_tiny.stop_token_ids.clone(),
        vocab_size: VOCAB.len(),
        schema_in_prompt: true,
        sampling: SamplingSection { temperature: 0.0, top_p: 1.0 },
        estimated_ane_bytes: 1 << 20,
        queue_depth,
    };
    let loader: DecoderLoader =
        Arc::new(move || Ok(Box::new(Fake { delay, prompt: 0, logits: vec![0.0; VOCAB.len()] }) as Box<dyn Decoder>));
    let timeout = (Duration::from_secs(30), "manifest");
    let model = OpenModel::new(spec, lane.clone(), loader, timeout.0, Duration::from_secs(30));
    let section_bytes = section(257);
    let spec = OpenModelSpec {
        id: "bytes".into(),
        context_length: 1024,
        prompter: Arc::new(Prompter::new(byte_tokenizer(), TEMPLATE, &section_bytes).unwrap()),
        stop_token_ids: section_bytes.stop_token_ids.clone(),
        vocab_size: 260,
        schema_in_prompt: true,
        sampling: SamplingSection { temperature: 0.0, top_p: 1.0 },
        estimated_ane_bytes: 1 << 20,
        queue_depth,
    };
    let loader: DecoderLoader = Arc::new(|| Ok(Box::new(Prefers::new()) as Box<dyn Decoder>));
    let bytes = OpenModel::new(spec, lane.clone(), loader, timeout.0, Duration::from_secs(30));
    let mut chat = ChatModels::default();
    // `apple-fm` in the lane, as build_state puts it when there are open
    // models.
    let fm_lane = lane.clone();
    chat.insert(chat_models::foundation_models(&RequestTimeouts::default(), move |t| {
        Arc::new(LanedChat::new(Arc::new(MockChat::fm(true)), fm_lane, t)) as Arc<dyn sidekick_core::ChatBackend>
    }))
    .unwrap();
    chat.insert(chat_models::open_model(Arc::new(model), Some("abc123"), timeout)).unwrap();
    chat.insert(chat_models::open_model(Arc::new(bytes), Some("abc123"), timeout)).unwrap();
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
}

fn guided(schema: Value, extra: Value) -> Request<Body> {
    let mut body = json!({
        "model": "bytes",
        "messages": [{"role": "user", "content": "hello"}],
        "response_format": {"type": "json_schema", "json_schema": {"name": "t", "schema": schema}}
    });
    for (k, v) in extra.as_object().unwrap() {
        body[k] = v.clone();
    }
    post_json("/v1/chat/completions", body)
}

fn record() -> Value {
    json!({
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "n": {"type": ["integer", "null"]},
            "tags": {"type": "array", "items": {"enum": ["red", "blue"]}, "maxItems": 2},
            "ok": {"type": "boolean"}
        },
        "required": ["name", "n", "tags"]
    })
}

#[tokio::test(flavor = "multi_thread")]
async fn guided_output_is_valid_json_in_schema_order() {
    let (status, body) = call(state(4, Duration::ZERO), guided(record(), json!({}))).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let content = body["choices"][0]["message"]["content"].as_str().unwrap();
    assert_eq!(content, r#"{"name": "", "n": 1, "tags": ["blue"]}"#);
    let schema = sidekick_core::JsonSchema::parse(&record()).unwrap();
    assert_eq!(schema.validate(&serde_json::from_str(content).unwrap()), Ok(()));
    assert_eq!((body["choices"][0]["finish_reason"].as_str(), body["constrained"].as_bool()), (Some("stop"), Some(true)));
    // Every token counted, forced or sampled; the JSON's end is the reply's.
    assert_eq!(body["usage"]["completion_tokens"], content.len());
    // The schema is written into the prompt (one token a byte here), as a
    // system message: the guided prompt is longer by at least its text.
    let (_, plain) = call(state(4, Duration::ZERO), post_json("/v1/chat/completions", json!({
        "model": "bytes", "messages": [{"role": "user", "content": "hello"}], "max_tokens": 1
    })))
    .await;
    let schema = sidekick_core::JsonSchema::parse(&record()).unwrap();
    let instruction = sidekick_llm::guided::schema_instruction(&schema);
    let grew = body["usage"]["prompt_tokens"].as_u64().unwrap() - plain["usage"]["prompt_tokens"].as_u64().unwrap();
    assert!(grew as usize >= instruction.len(), "{grew} < {}", instruction.len());

    // Cut by max_tokens: `length`, and the JSON is incomplete.
    let (status, body) = call(state(4, Duration::ZERO), guided(record(), json!({"max_tokens": 10}))).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["choices"][0]["message"]["content"], r#"{"name": ""#);
    assert_eq!((body["choices"][0]["finish_reason"].as_str(), body["constrained"].as_bool()), (Some("length"), Some(true)));
}

#[tokio::test(flavor = "multi_thread")]
async fn guided_output_streams_as_it_is_generated() {
    let response = build_router(state(4, Duration::ZERO))
        .oneshot(guided(record(), json!({"stream": true, "stream_options": {"include_usage": true}})))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let bytes = response.into_body().collect().await.unwrap().to_bytes();
    let data: Vec<String> =
        String::from_utf8_lossy(&bytes).lines().filter_map(|l| l.strip_prefix("data: ").map(str::to_string)).collect();
    let chunks: Vec<Value> = data[..data.len() - 1].iter().map(|d| serde_json::from_str(d).unwrap()).collect();
    let deltas: Vec<&str> = chunks.iter().filter_map(|c| c["choices"][0]["delta"]["content"].as_str()).collect();
    assert!(deltas.len() > 5, "sent as generated, not whole: {deltas:?}");
    assert_eq!(deltas.concat(), r#"{"name": "", "n": 1, "tags": ["blue"]}"#);
    let finish = &chunks[chunks.len() - 2];
    assert_eq!((finish["choices"][0]["finish_reason"].as_str(), finish["constrained"].as_bool()), (Some("stop"), Some(true)));
}

#[tokio::test(flavor = "multi_thread")]
async fn guided_requests_are_refused_as_on_apple_fm() {
    // The shared schema layer's 400, before the model is involved.
    let schema = json!({"type": "object", "properties": {"sku": {"type": "string", "pattern": "^[A-Z]+$"}}});
    let (status, body) = call(state(4, Duration::ZERO), guided(schema, json!({}))).await;
    assert_eq!((status, body["error"]["code"].as_str()), (StatusCode::BAD_REQUEST, Some("unsupported_schema")), "{body}");
    assert!(body["error"]["message"].as_str().unwrap().contains("/properties/sku"), "{body}");
    // `stop` and a schema don't mix, on any model.
    let (status, body) = call(state(4, Duration::ZERO), guided(record(), json!({"stop": ["}"]}))).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    // A model whose tokenizer llguidance can't read says so.
    let body = json!({
        "model": "tiny",
        "messages": [{"role": "user", "content": "hello"}],
        "response_format": {"type": "json_schema", "json_schema": {"name": "t", "schema": record()}}
    });
    let (status, body) = call(state(4, Duration::ZERO), post_json("/v1/chat/completions", body)).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(body["error"]["message"].as_str().unwrap().contains("can't guide"), "{body}");
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
async fn a_client_that_leaves_while_queued_gives_up_its_place() {
    let state = state(1, Duration::from_millis(300));
    let router = build_router(state.clone());
    let lane = state.chat.lane().unwrap().clone();
    let running = tokio::spawn(router.clone().oneshot(ask(json!({}))));
    let started = std::time::Instant::now();
    while lane.health()["running"].is_null() {
        assert!(started.elapsed() < Duration::from_secs(10));
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    // A streamed request queues behind it; its client gives up before the
    // first delta.
    let queued = router.clone().oneshot(ask(json!({"stream": true})));
    assert!(tokio::time::timeout(Duration::from_millis(100), queued).await.is_err());
    // Its place in the queue is given back at once, not when its turn comes.
    let started = std::time::Instant::now();
    while lane.health()["waiting"] != 0 {
        assert!(started.elapsed() < Duration::from_secs(10), "{}", lane.health());
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    assert_eq!(running.await.unwrap().unwrap().status(), StatusCode::OK);
    // And it never ran.
    let (_, health) = call(state, Request::get("/health").body(Body::empty()).unwrap()).await;
    assert_eq!(health["chat"]["models"]["tiny"]["requests"], 1);
}

#[tokio::test(flavor = "multi_thread")]
async fn temperature_is_checked_at_the_route() {
    for t in [-0.1, 2.5] {
        let (status, body) = call(state(4, Duration::ZERO), ask(json!({"temperature": t}))).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{t}");
        assert!(body["error"]["message"].as_str().unwrap().contains("temperature"), "{body}");
    }
    let (status, _) = call(state(4, Duration::ZERO), ask(json!({"temperature": 2.0}))).await;
    assert_eq!(status, StatusCode::OK);
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
    // `tiny`'s word-level tokenizer can't be guided; `bytes`' can.
    assert!(!tiny["honored_fields"].as_array().unwrap().contains(&json!("response_format")));
    assert_eq!(tiny["guided_streaming"], false);
    let bytes = models["data"].as_array().unwrap().iter().find(|m| m["id"] == "bytes").expect("listed");
    assert!(bytes["honored_fields"].as_array().unwrap().contains(&json!("response_format")));
    assert_eq!(bytes["guided_streaming"], true);

    let (status, _) = call(state.clone(), ask(json!({}))).await;
    assert_eq!(status, StatusCode::OK);
    let (_, health) = call(state, Request::get("/health").body(Body::empty()).unwrap()).await;
    let tiny = &health["chat"]["models"]["tiny"];
    assert_eq!(tiny["backend"], "coreml");
    assert_eq!((tiny["state"].as_str(), tiny["resident"].as_bool()), (Some("ready"), Some(true)));
    assert_eq!((tiny["queue_depth"].as_u64(), tiny["lane_queue_depth"].as_u64(), tiny["requests"].as_u64()), (Some(3), Some(3), Some(1)));
    assert_eq!(tiny["ane"]["estimated_bytes"], 1 << 20);
    assert_eq!((tiny["guided"]["available"].as_bool(), tiny["schema_in_prompt"].as_bool()), (Some(false), Some(true)));
    assert_eq!(tiny["request_timeout_secs"], 30);
    let lane = &health["chat"]["lane"];
    assert_eq!(lane["resident"]["model"], "tiny");
    assert_eq!(lane["ane"]["budget_mib"], 2816);
}
