//! The daemon config's `onnx_runtime` names the ONNX Runtime library, ahead
//! of ORT_DYLIB_PATH (D40): a daemon with an onnx model loads it at startup
//! from there, and /health names the config as its source.
//!
//! Needs `--features onnx` and ONNX Runtime at ORT_DYLIB_PATH (used here as
//! the configured path); skips without them unless SIDEKICK_REQUIRE_ONNX=1.
#![cfg(feature = "onnx")]

use axum::body::Body;
use axum::http::Request;
use http_body_util::BodyExt;
use sidekick_server::{build_router, build_state, Config};
use std::path::{Path, PathBuf};
use tower::ServiceExt;

#[tokio::test]
async fn the_configured_runtime_wins_and_loads_at_startup() {
    let Some(library) = std::env::var_os("ORT_DYLIB_PATH").map(PathBuf::from) else {
        assert!(std::env::var("SIDEKICK_REQUIRE_ONNX").as_deref() != Ok("1"), "SIDEKICK_REQUIRE_ONNX=1 but ORT_DYLIB_PATH isn't set");
        return;
    };
    let fixtures = Path::new(env!("CARGO_MANIFEST_DIR")).join("../sidekick-embed/tests/fixtures/tiny-onnx");
    let dir = std::env::temp_dir().join(format!("sk-onnx-config-{}", std::process::id()));
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
    for f in ["hidden.onnx", "tokenizer.json"] {
        std::fs::copy(fixtures.join(f), model.join(f)).unwrap();
    }
    let config = Config { models_dir: Some(dir.clone()), onnx_runtime: Some(library), ..Config::default() };
    let state = build_state(&config).unwrap();
    // Loaded at startup, before any request.
    let runtime = sidekick_embed::onnx::runtime_state().expect("loaded at startup").unwrap();
    assert_eq!(runtime.source.name(), "config", "the config wins over ORT_DYLIB_PATH");
    let response = build_router(state).oneshot(Request::get("/health").body(Body::empty()).unwrap()).await.unwrap();
    let health: serde_json::Value = serde_json::from_slice(&response.into_body().collect().await.unwrap().to_bytes()).unwrap();
    assert_eq!(health["onnx"]["runtime"]["state"], "loaded");
    assert_eq!(health["onnx"]["runtime"]["source"], "config");
    assert_eq!(health["embeddings"]["models"], serde_json::json!(["tiny"]));
    std::fs::remove_dir_all(&dir).unwrap();
}
