//! The onnx backend (D40) behind the daemon's routes, on a real ONNX
//! Runtime with sidekick-embed's tiny fixture graph: /v1/embeddings and
//! /v2/embed serve it, /v1/models and the provenance headers say it runs
//! on the CPU with no buckets, and /health reports the runtime and each
//! model's threads.
//!
//! Needs `--features onnx` and ORT_DYLIB_PATH; skips without them unless
//! SIDEKICK_REQUIRE_ONNX=1.
#![cfg(feature = "onnx")]

mod common;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use common::*;
use serde_json::{json, Value};
use sidekick_core::ModelRegistry;
use sidekick_server::config::OnnxThreads;
use sidekick_server::{AppState, EmbedderPool};
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

fn have_runtime() -> bool {
    if std::env::var_os("ORT_DYLIB_PATH").is_some() {
        return true;
    }
    assert!(std::env::var("SIDEKICK_REQUIRE_ONNX").as_deref() != Ok("1"), "SIDEKICK_REQUIRE_ONNX=1 but ORT_DYLIB_PATH isn't set");
    false
}

/// The test state with one onnx model, `tiny`, over the fixture's
/// token-level graph pooled at its first token, with 2 threads.
fn onnx_state() -> AppState {
    let fixtures = Path::new(env!("CARGO_MANIFEST_DIR")).join("../sidekick-embed/tests/fixtures/tiny-onnx");
    let dir = std::env::temp_dir().join(format!("sk-onnx-routes-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let model = dir.join("tiny");
    std::fs::create_dir_all(&model).unwrap();
    std::fs::write(
        model.join("manifest.toml"),
        "id = \"tiny\"\nbackend = \"onnx\"\nartifact = \"hidden.onnx\"\ntokenizer = \"tokenizer.json\"\ndims = 8\n\
         pooling = \"cls\"\nmax_seq_len = 16\n\n[io]\ninput_ids = \"input_ids\"\nattention_mask = \"attention_mask\"\n\
         token_type_ids = \"token_type_ids\"\nposition_ids = \"position_ids\"\noutput = \"last_hidden_state\"\n",
    )
    .unwrap();
    let words = ["[UNK]", "[CLS]", "[SEP]", "a", "b", "c", "d", "e", "f", "g", "h", "i"];
    let vocab: serde_json::Map<String, Value> = words.iter().enumerate().map(|(i, w)| (w.to_string(), i.into())).collect();
    let tokenizer = json!({
        "version": "1.0", "truncation": null, "padding": null, "added_tokens": [], "normalizer": null,
        "pre_tokenizer": {"type": "WhitespaceSplit"}, "post_processor": null, "decoder": null,
        "model": {"type": "WordLevel", "vocab": vocab, "unk_token": "[UNK]"}
    });
    std::fs::write(model.join("tokenizer.json"), tokenizer.to_string()).unwrap();
    std::os::unix::fs::symlink(fixtures.join("hidden.onnx"), model.join("hidden.onnx")).unwrap();
    let registry = Arc::new(ModelRegistry::scan(&dir).unwrap());
    assert!(registry.skipped().is_empty(), "{:?}", registry.skipped().first().map(|s| &s.reason));
    let threads = OnnxThreads { default: 3, by_model: [("tiny".into(), 2), ("ghost".into(), 1)].into() };
    let mut state = test_state(true, None);
    state.embedders = Arc::new(EmbedderPool::embedders_with(registry.clone(), Duration::from_secs(60), threads.clone()));
    state.registry = registry;
    state.onnx_threads = threads;
    state
}

#[tokio::test(flavor = "multi_thread")]
async fn the_routes_serve_an_onnx_model_on_the_cpu() {
    if !have_runtime() {
        return;
    }
    let state = onnx_state();
    let (status, headers, body) =
        call_with_headers(state.clone(), post_json("/v1/embeddings", json!({"model": "tiny", "input": ["[CLS] b c [SEP]", "[CLS] d [SEP]"]}))).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let first: Vec<f64> = body["data"][0]["embedding"].as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect();
    let norm: f64 = first.iter().map(|x| x * x).sum::<f64>().sqrt();
    assert!((norm - 1.0).abs() < 1e-5 && first.len() == 8);
    assert_eq!(headers["sidekick-compute-units"], "cpu");
    assert!(headers.get("sidekick-buckets").is_none(), "one graph: no buckets");

    let (status, _, body) = call_with_headers(
        state.clone(),
        post_json("/v2/embed", json!({"model": "tiny", "texts": ["[CLS] b c [SEP]"], "input_type": "search_document"})),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let v2: Vec<f64> = body["embeddings"]["float"][0].as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect();
    assert!(v2.iter().zip(&first).all(|(a, b)| (a - b).abs() < 1e-6), "both routes, one model");

    let (_, models) = call(state.clone(), Request::get("/v1/models").body(Body::empty()).unwrap()).await;
    let tiny = models["data"].as_array().unwrap().iter().find(|m| m["id"] == "tiny").unwrap().clone();
    assert_eq!((tiny["backend"].as_str(), tiny["compute_units"].as_str()), (Some("onnx"), Some("cpu")));
    assert!(tiny.get("placement").is_none() && tiny.get("compute_units_source").is_none());

    let (_, health) = call(state, Request::get("/health").body(Body::empty()).unwrap()).await;
    let onnx = &health["onnx"];
    assert_eq!(onnx["supported"], true);
    assert_eq!(onnx["runtime"]["state"], "loaded");
    assert_eq!(onnx["runtime"]["version"], onnx["pinned_version"]);
    assert_eq!(onnx["runtime"]["source"], "ORT_DYLIB_PATH");
    assert!(onnx["runtime"]["file"].as_str().unwrap().starts_with("libonnxruntime"));
    assert_eq!(onnx["default_intra_op_threads"], 3);
    let detected = sidekick_embed::onnx::detected_cores();
    assert_eq!(onnx["detected"], json!({"source": detected.source, "cores": detected.cores}));
    assert_eq!(onnx["models"], json!({"tiny": {"intra_op_threads": 2, "quantized": false}}));
    assert_eq!(onnx["unmatched_thread_overrides"], json!(["ghost"]));
}
