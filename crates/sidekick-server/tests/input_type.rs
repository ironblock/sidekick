//! `input_type` and the prompts it picks (D9): `document` (the default)
//! and `query` put the manifest's prompts before each input, and `none`
//! puts nothing, on /v1/embeddings, /v2/embed and /tokenize alike.
//! /v1/models lists each embedder's prompts. Over static test models, whose
//! vectors are worked out by hand: an existing request's vectors are pinned
//! to what they were before `none` existed.

mod common;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use common::*;
use serde_json::{json, Value};
use sidekick_core::ModelRegistry;
use sidekick_server::{AppState, EmbedderPool, TokenizerPool};
use std::sync::Arc;
use std::time::Duration;

/// The test state with test-static (no prompts) and `prompted`, the same
/// table with a one-token query prompt and a two-token document prompt.
fn state(tag: &str) -> (AppState, std::path::PathBuf) {
    let dir = std::env::temp_dir().join(format!("sk-input-type-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    write_embedding_fixture(&dir);
    let prompted = dir.join("prompted");
    std::fs::create_dir_all(&prompted).unwrap();
    for file in ["model.safetensors", "tokenizer.json"] {
        std::fs::copy(dir.join("test-static").join(file), prompted.join(file)).unwrap();
    }
    std::fs::write(
        prompted.join("manifest.toml"),
        "id = \"prompted\"\nbackend = \"static\"\nartifact = \"model.safetensors\"\ntokenizer = \"tokenizer.json\"\n\
         dims = 4\nmax_seq_len = 512\n\n[prefixes]\nquery = \"world \"\ndocument = \"hello world \"\n",
    )
    .unwrap();
    let registry = Arc::new(ModelRegistry::scan(&dir).unwrap());
    let mut state = test_state(true, None);
    state.embedders = Arc::new(EmbedderPool::embedders(registry.clone(), Duration::from_secs(60)));
    state.tokenizers = Arc::new(TokenizerPool::tokenizers(registry.clone(), Duration::from_secs(60)));
    state.registry = registry;
    (state, dir)
}

async fn post(state: &AppState, route: &str, body: Value) -> (StatusCode, Value) {
    call(state.clone(), post_json(route, body)).await
}

/// /v1/embeddings' vector for `text` as `input_type`.
async fn embed(state: &AppState, model: &str, text: &str, input_type: Option<&str>) -> Vec<f64> {
    let mut body = json!({"model": model, "input": text});
    if let Some(t) = input_type {
        body["input_type"] = json!(t);
    }
    let (status, body) = post(state, "/v1/embeddings", body).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    body["data"][0]["embedding"].as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect()
}

fn close(a: &[f64], b: &[f64]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| (x - y).abs() < 1e-6)
}

#[tokio::test(flavor = "multi_thread")]
async fn existing_requests_keep_their_prompts_and_none_drops_them() {
    let (state, dir) = state("embed");
    // The table's rows: hello [1, 0, 0, 0], world [0, 2, 0, 0]. A static
    // vector is the normalized mean of its tokens' rows.
    let h = 0.5f64.sqrt();
    let (a, b) = (1.0 / 5f64.sqrt(), 2.0 / 5f64.sqrt());
    // Pinned: as before `none` existed. Default and `document`:
    // "hello world hello"; `query`: "world hello".
    for input_type in [None, Some("document"), Some("passage")] {
        assert!(close(&embed(&state, "prompted", "hello", input_type).await, &[h, h, 0.0, 0.0]), "{input_type:?}");
    }
    assert!(close(&embed(&state, "prompted", "hello", Some("query")).await, &[a, b, 0.0, 0.0]));
    // `none`: the text as sent.
    assert!(close(&embed(&state, "prompted", "hello", Some("none")).await, &[1.0, 0.0, 0.0, 0.0]));
    assert_eq!(
        embed(&state, "prompted", "hello world hello", Some("none")).await,
        embed(&state, "prompted", "hello", None).await,
        "a client's own prompt, sent with none, is the default's"
    );
    // A model with no prompts: `none` is the default.
    assert_eq!(
        embed(&state, "test-static", "hello world", Some("none")).await,
        embed(&state, "test-static", "hello world", None).await
    );

    // /v2/embed: Cohere's names as before, and `none`.
    let v2 = |input_type: &str| json!({"model": "prompted", "texts": ["hello"], "input_type": input_type});
    let floats = |body: &Value| -> Vec<f64> {
        body["embeddings"]["float"][0].as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect()
    };
    for (input_type, want) in [("search_document", [h, h, 0.0, 0.0]), ("search_query", [a, b, 0.0, 0.0]), ("none", [1.0, 0.0, 0.0, 0.0])] {
        let (status, body) = post(&state, "/v2/embed", v2(input_type)).await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert!(close(&floats(&body), &want), "{input_type}: {body}");
    }
    // Cohere's other names stay refused: no manifest declares their prompts.
    for input_type in ["classification", "clustering"] {
        let (status, body) = post(&state, "/v2/embed", v2(input_type)).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
        assert!(body["error"]["message"].as_str().unwrap().starts_with(&format!("unsupported input_type `{input_type}`")));
    }
    // /v1/embeddings refuses what it refused.
    let (status, body) = post(&state, "/v1/embeddings", json!({"model": "prompted", "input": "hi", "input_type": "search_query"})).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    assert!(body["error"]["message"].as_str().unwrap().contains("`none`"), "{body}");
    std::fs::remove_dir_all(&dir).unwrap();
}

#[tokio::test(flavor = "multi_thread")]
async fn tokenize_counts_and_fits_as_the_route_embeds() {
    let (state, dir) = state("tokenize");
    let count = |input_type: &'static str, text: String| {
        let state = state.clone();
        async move {
            let (status, body) =
                post(&state, "/tokenize", json!({"model": "prompted", "prompt": text, "input_type": input_type})).await;
            assert_eq!(status, StatusCode::OK, "{body}");
            (body["count"].as_u64().unwrap() as usize, body["fits"].as_bool().unwrap())
        }
    };
    assert_eq!(count("document", "hello".into()).await, (3, true));
    assert_eq!(count("search_query", "hello".into()).await, (2, true));
    assert_eq!(count("none", "hello".into()).await, (1, true));
    // 511 words fit with no prompt, not with the document prompt's two.
    let text = "hello ".repeat(511);
    assert_eq!(count("none", text.clone()).await, (511, true));
    assert_eq!(count("document", text.clone()).await, (513, false));
    for (input_type, ok) in [("none", true), ("document", false)] {
        let body = json!({"model": "prompted", "input": text, "input_type": input_type});
        assert_eq!(post(&state, "/v1/embeddings", body).await.0 == StatusCode::OK, ok, "{input_type}");
    }
    std::fs::remove_dir_all(&dir).unwrap();
}

#[tokio::test(flavor = "multi_thread")]
async fn models_list_each_embedders_prompts() {
    let (state, dir) = state("models");
    let (status, body) = call(state.clone(), Request::get("/v1/models").body(Body::empty()).unwrap()).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let entry = |id: &str| body["data"].as_array().unwrap().iter().find(|m| m["id"] == id).unwrap().clone();
    assert_eq!(entry("prompted")["prompts"], json!({"query": "world ", "document": "hello world "}));
    assert_eq!(entry("test-static")["prompts"], json!({"query": "", "document": ""}));
    // Only embedders have prompts.
    assert!(entry("apple-fm").get("prompts").is_none());
    std::fs::remove_dir_all(&dir).unwrap();
}
