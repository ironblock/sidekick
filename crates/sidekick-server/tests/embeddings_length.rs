//! How long an input `/v1/embeddings` takes, and what happens past it.
//! `/v1/models` lists each embedder's `max_seq_len`, and a Core ML
//! model's `buckets`, after the registry's CPU cap (D33). The request's
//! `truncate` (Cohere's values, as on `/v2/embed` but in any case, or TEI's
//! boolean) picks what happens to a longer input: `END`, the default, keeps
//! its start, `START` its end, and `NONE` refuses the whole request with a
//! 400 naming the input.
//!
//! Over the static test model, and, on macOS with `--features coreml`, a
//! real Core ML model (sidekick-embed's tiny Qwen3 fixture) on the CPU.
//! The onnx backend's are in onnx_routes.rs.

mod common;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use common::*;
use serde_json::{json, Value};
use sidekick_core::ModelRegistry;
use sidekick_server::{AppState, EmbedderPool};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

/// A Core ML embedder's manifest: the tiny Qwen3 fixture's, with `units`
/// and its buckets.
fn coreml_manifest(id: &str, units: &str, buckets: &[usize]) -> String {
    format!(
        "id = \"{id}\"\nbackend = \"coreml\"\nartifact = \"model_{{seq}}.mlmodelc\"\ntokenizer = \"tokenizer.json\"\n\
         dims = 32\npooling = \"none\"\nbuckets = {buckets:?}\nmax_seq_len = {}\ncompute_units = \"{units}\"\n\n\
         [io]\ninput_ids = \"input_ids\"\nattention_mask = \"attention_mask\"\noutput = \"embedding\"\n",
        buckets.last().unwrap()
    )
}

/// The test state over a models directory with the static test model and
/// `embedders`: (id, manifest). Returns the directory, for its models'
/// other files.
fn state_with(tag: &str, embedders: &[(&str, String)]) -> (AppState, PathBuf) {
    let dir = std::env::temp_dir().join(format!("sk-embeddings-length-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    write_embedding_fixture(&dir);
    for (id, manifest) in embedders {
        std::fs::create_dir_all(dir.join(id)).unwrap();
        std::fs::write(dir.join(id).join("manifest.toml"), manifest).unwrap();
    }
    let registry = Arc::new(ModelRegistry::scan(&dir).unwrap());
    assert!(registry.skipped().is_empty(), "{:?}", registry.skipped().first().map(|s| &s.reason));
    let mut state = test_state(true, None);
    state.embedders = Arc::new(EmbedderPool::embedders(registry.clone(), Duration::from_secs(60)));
    state.registry = registry;
    (state, dir)
}

async fn models(state: AppState) -> Vec<Value> {
    let (status, body) = call(state, Request::get("/v1/models").body(Body::empty()).unwrap()).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    body["data"].as_array().unwrap().clone()
}

fn entry<'a>(models: &'a [Value], id: &str) -> &'a Value {
    models.iter().find(|m| m["id"] == id).unwrap_or_else(|| panic!("{id} not listed"))
}

async fn embed(state: &AppState, body: Value) -> (StatusCode, Value) {
    call(state.clone(), post_json("/v1/embeddings", body)).await
}

#[tokio::test(flavor = "multi_thread")]
async fn models_list_each_embedders_effective_length() {
    // Listed from manifests: nothing loads, so the artifacts needn't exist.
    let (state, dir) = state_with(
        "models",
        &[
            ("coreml-ane", coreml_manifest("coreml-ane", "cpu_and_ne", &[128, 512])),
            // Served on the CPU past 1,024 tokens: capped (D33).
            ("coreml-capped", coreml_manifest("coreml-capped", "cpu_only", &[128, 1024, 2048])),
        ],
    );
    let models = models(state).await;
    let ane = entry(&models, "coreml-ane");
    assert_eq!((&ane["max_seq_len"], &ane["buckets"]), (&json!(512), &json!([128, 512])));
    let capped = entry(&models, "coreml-capped");
    assert_eq!((&capped["max_seq_len"], &capped["buckets"]), (&json!(1024), &json!([128, 1024])));
    assert_eq!(capped["seq_cap"]["manifest_max_seq_len"], 2048);
    // A static model has a limit and no buckets.
    let fixed = entry(&models, "test-static");
    assert_eq!(fixed["max_seq_len"], 512);
    assert!(fixed.get("buckets").is_none(), "{fixed}");
    // The chat model's entry is unchanged.
    let chat = entry(&models, "apple-fm");
    assert!(chat.get("max_seq_len").is_none() && chat.get("buckets").is_none(), "{chat}");
    std::fs::remove_dir_all(&dir).unwrap();
}

#[tokio::test(flavor = "multi_thread")]
async fn truncate_picks_what_happens_past_the_limit() {
    let (state, dir) = state_with("truncate", &[]);
    // 600 tokens, over test-static's 512.
    let long = "hello ".repeat(600);
    let request = |truncate: Option<Value>, input: Value| {
        let mut body = json!({"model": "test-static", "input": input});
        if let Some(t) = truncate {
            body["truncate"] = t;
        }
        body
    };
    let batch = json!(["hello world", long, "world"]);

    // Without `truncate`, an over-length input is truncated, as before the
    // field existed: the static model's rows are hello [1, 0, 0, 0] and
    // world [0, 2, 0, 0], so its 512 hellos embed as one.
    let (status, default) = embed(&state, request(None, batch.clone())).await;
    assert_eq!(status, StatusCode::OK, "{default}");
    let (a, b) = (1.0 / 5f64.sqrt(), 2.0 / 5f64.sqrt());
    for (i, want) in [[a, b, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]].iter().enumerate() {
        let got: Vec<f64> = default["data"][i]["embedding"].as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect();
        assert!(got.iter().zip(want).all(|(g, w)| (g - w).abs() < 1e-6), "input {i}: {got:?}");
    }
    // `END`, and TEI's `true`, are the default.
    for end in [json!("END"), json!("end"), json!(true)] {
        assert_eq!(embed(&state, request(Some(end), batch.clone())).await, (StatusCode::OK, default.clone()));
    }
    assert_eq!(embed(&state, request(Some(json!("START")), batch.clone())).await.0, StatusCode::OK);

    // `NONE`, in any case, or TEI's `false`: one over-length input refuses
    // the whole request, naming it.
    for none in [json!("none"), json!(false)] {
        let (status, body) = embed(&state, request(Some(none), batch.clone())).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    }
    let (status, body) = embed(&state, request(Some(json!("NONE")), batch.clone())).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    assert_eq!(body["error"]["type"], "invalid_request_error");
    assert_eq!(
        body["error"]["message"],
        "input 1: 600 tokens, over model `test-static`'s limit of 512 tokens, and truncate is NONE \
         (send END or START to truncate it)"
    );
    let (status, body) = embed(&state, request(Some(json!("NONE")), json!(long))).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    assert!(body["error"]["message"].as_str().unwrap().starts_with("input 0: 600 tokens"), "{body}");

    // Within the limit, `NONE` changes nothing.
    let short = json!(["hello world", "world"]);
    let (status, body) = embed(&state, request(Some(json!("NONE")), short.clone())).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body, embed(&state, request(None, short)).await.1);

    // Anything else is a 400; `null` is no value.
    for bad in [json!("MIDDLE"), json!(""), json!(1)] {
        let (status, body) = embed(&state, request(Some(bad.clone()), json!("hello"))).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{bad}: {body}");
        assert!(body["error"]["message"].as_str().unwrap().starts_with("unsupported truncate"), "{body}");
    }
    assert_eq!(embed(&state, request(Some(Value::Null), json!("hello"))).await.0, StatusCode::OK);

    // /v2/embed keeps Cohere's spelling, and names a request's max_tokens
    // as the limit it set.
    let v2 = |truncate: &str| json!({"model": "test-static", "texts": ["hello world hello"], "truncate": truncate, "max_tokens": 2});
    let (status, body) = call(state.clone(), post_json("/v2/embed", v2("none"))).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    let (status, body) = call(state.clone(), post_json("/v2/embed", v2("NONE"))).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    assert!(body["error"]["message"].as_str().unwrap().starts_with("input 0: 3 tokens, over max_tokens 2,"), "{body}");
    std::fs::remove_dir_all(&dir).unwrap();
}

/// The tiny Qwen3 embedder from sidekick-embed's fixtures, served by the
/// daemon on the CPU, in buckets 16 and 32.
#[cfg(all(target_os = "macos", feature = "coreml"))]
#[tokio::test(flavor = "multi_thread")]
async fn a_core_ml_model_lists_its_buckets_and_refuses_past_the_largest() {
    let fixtures = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../sidekick-embed/tests/fixtures/tiny-qwen3-embed-chunked");
    let (state, dir) = state_with("coreml", &[("tiny-qwen3-embed", coreml_manifest("tiny-qwen3-embed", "cpu_only", &[16, 32]))]);
    let model = dir.join("tiny-qwen3-embed");
    let vocab: serde_json::Map<String, Value> = ["[UNK]", "a", "b"].iter().enumerate().map(|(i, w)| (w.to_string(), i.into())).collect();
    let tokenizer = json!({
        "version": "1.0", "truncation": null, "padding": null, "added_tokens": [], "normalizer": null,
        "pre_tokenizer": {"type": "Whitespace"}, "post_processor": null, "decoder": null,
        "model": {"type": "WordLevel", "vocab": vocab, "unk_token": "[UNK]"}
    });
    std::fs::write(model.join("tokenizer.json"), tokenizer.to_string()).unwrap();
    for bucket in [16, 32] {
        let name = format!("model_{bucket}.mlmodelc");
        std::os::unix::fs::symlink(fixtures.join(&name), model.join(&name)).unwrap();
    }

    let models = models(state.clone()).await;
    let m = entry(&models, "tiny-qwen3-embed");
    assert_eq!((&m["max_seq_len"], &m["buckets"]), (&json!(32), &json!([16, 32])));

    let long = "a ".repeat(40);
    let (status, body) =
        embed(&state, json!({"model": "tiny-qwen3-embed", "input": ["a b", long], "truncate": "NONE"})).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    let message = body["error"]["message"].as_str().unwrap();
    assert!(message.starts_with("input 1: 40 tokens, over model `tiny-qwen3-embed`'s limit of 32 tokens"), "{message}");

    let (status, headers, body) = call_with_headers(
        state.clone(),
        post_json("/v1/embeddings", json!({"model": "tiny-qwen3-embed", "input": ["a b", long]})),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(headers["sidekick-buckets"], "16,32", "truncated to the largest bucket");
    std::fs::remove_dir_all(&dir).unwrap();
}
