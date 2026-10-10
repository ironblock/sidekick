//! ONNX Runtime loads only for an onnx model (D40): a daemon whose models
//! directory has none never loads it, in any build, and /health says so.

use axum::body::Body;
use axum::http::Request;
use http_body_util::BodyExt;
use sidekick_server::{build_router, build_state, Config};
use tower::ServiceExt;

#[tokio::test]
async fn without_an_onnx_model_the_runtime_never_loads() {
    let dir = std::env::temp_dir().join(format!("sk-onnx-lazy-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let config = Config { models_dir: Some(dir.clone()), ..Config::default() };
    let state = build_state(&config).unwrap();
    assert!(sidekick_embed::onnx::runtime_state().is_none(), "nothing dlopened");
    let response = build_router(state).oneshot(Request::get("/health").body(Body::empty()).unwrap()).await.unwrap();
    let health: serde_json::Value = serde_json::from_slice(&response.into_body().collect().await.unwrap().to_bytes()).unwrap();
    assert_eq!(health["onnx"]["runtime"], serde_json::json!({"state": "not loaded", "reason": "no onnx models"}));
    assert_eq!(health["onnx"]["supported"], sidekick_embed::onnx::SUPPORTED);
    std::fs::remove_dir_all(&dir).unwrap();
}
