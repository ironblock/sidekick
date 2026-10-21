//! `prompt_name` and the manifest's `[prompts]` (D45): a request names one
//! of the model's prompts, by its exact name, in place of an `input_type`
//! prompt, on /v1/embeddings, /v2/embed and /tokenize alike. Naming both is
//! a 400, and so is a name the model doesn't have. /v1/models lists each
//! embedder's named prompts. Over static test models, whose vectors are
//! worked out by hand: a manifest's `[prompts]` changes no request that
//! doesn't name one.

mod common;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use common::*;
use serde_json::{json, Value};
use sidekick_core::ModelRegistry;
use sidekick_server::{AppState, EmbedderPool, TokenizerPool};
use std::sync::Arc;
use std::time::Duration;

/// The test state with test-static (no prompts) and `named`, the same
/// table with input_type prefixes and two named prompts.
fn state(tag: &str) -> (AppState, std::path::PathBuf) {
    let dir = std::env::temp_dir().join(format!("sk-named-prompts-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    write_embedding_fixture(&dir);
    let named = dir.join("named");
    std::fs::create_dir_all(&named).unwrap();
    for file in ["model.safetensors", "tokenizer.json"] {
        std::fs::copy(dir.join("test-static").join(file), named.join(file)).unwrap();
    }
    std::fs::write(
        named.join("manifest.toml"),
        "id = \"named\"\nbackend = \"static\"\nartifact = \"model.safetensors\"\ntokenizer = \"tokenizer.json\"\n\
         dims = 4\nmax_seq_len = 512\n\n[prefixes]\nquery = \"world \"\ndocument = \"hello world \"\n\n\
         [prompts]\nSTS = \"world world \"\ndocument = \"hello world \"\n",
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

/// /v1/embeddings' vector for `body`'s one input.
async fn embed(state: &AppState, body: Value) -> Vec<f64> {
    let (status, body) = post(state, "/v1/embeddings", body).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    body["data"][0]["embedding"].as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect()
}

fn floats(body: &Value) -> Vec<f64> {
    body["embeddings"]["float"][0].as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect()
}

fn close(a: &[f64], b: &[f64]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| (x - y).abs() < 1e-6)
}

fn message(body: &Value) -> &str {
    body["error"]["message"].as_str().unwrap()
}

#[tokio::test(flavor = "multi_thread")]
async fn a_named_prompt_replaces_the_input_type_prompt() {
    let (state, dir) = state("embed");
    // The table's rows: hello [1, 0, 0, 0], world [0, 2, 0, 0]. A static
    // vector is the normalized mean of its tokens' rows.
    let h = 0.5f64.sqrt();
    let (a, b) = (1.0 / 5f64.sqrt(), 2.0 / 5f64.sqrt());
    let (c, d) = (1.0 / 17f64.sqrt(), 4.0 / 17f64.sqrt());
    // Pinned: requests without prompt_name embed as they did before
    // [prompts] existed. Default and `document`: "hello world hello";
    // `query`: "world hello"; `none`: "hello".
    for (input_type, want) in [
        (None, [h, h, 0.0, 0.0]),
        (Some("document"), [h, h, 0.0, 0.0]),
        (Some("query"), [a, b, 0.0, 0.0]),
        (Some("none"), [1.0, 0.0, 0.0, 0.0]),
    ] {
        let mut body = json!({"model": "named", "input": "hello"});
        if let Some(t) = input_type {
            body["input_type"] = json!(t);
        }
        assert!(close(&embed(&state, body).await, &want), "{input_type:?}");
    }
    // `STS`: "world world hello".
    let sts = embed(&state, json!({"model": "named", "input": "hello", "prompt_name": "STS"})).await;
    assert!(close(&sts, &[c, d, 0.0, 0.0]), "{sts:?}");
    // The checkpoint's `document` prompt, by name, is the default's.
    assert_eq!(
        embed(&state, json!({"model": "named", "input": "hello", "prompt_name": "document"})).await,
        embed(&state, json!({"model": "named", "input": "hello"})).await,
    );
    // /v2/embed, without input_type.
    let (status, body) = post(&state, "/v2/embed", json!({"model": "named", "texts": ["hello"], "prompt_name": "STS"})).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert!(close(&floats(&body), &[c, d, 0.0, 0.0]), "{body}");
    std::fs::remove_dir_all(&dir).unwrap();
}

#[tokio::test(flavor = "multi_thread")]
async fn a_name_with_an_input_type_or_not_the_models_is_refused() {
    let (state, dir) = state("refused");
    // Both: any input_type, `none` included, with a name.
    for input_type in ["document", "query", "none"] {
        let body = json!({"model": "named", "input": "hello", "input_type": input_type, "prompt_name": "STS"});
        let (status, body) = post(&state, "/v1/embeddings", body).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
        assert!(message(&body).starts_with("send input_type or prompt_name, not both"), "{body}");
    }
    let body = json!({"model": "named", "texts": ["hello"], "input_type": "search_query", "prompt_name": "STS"});
    assert_eq!(post(&state, "/v2/embed", body).await.0, StatusCode::BAD_REQUEST);
    let body = json!({"model": "named", "prompt": "hello", "input_type": "none", "prompt_name": "STS"});
    assert_eq!(post(&state, "/tokenize", body).await.0, StatusCode::BAD_REQUEST);
    // A name the model doesn't have, case included, lists the ones it has.
    for route in ["/v1/embeddings", "/v2/embed", "/tokenize"] {
        let body = json!({"model": "named", "input": "hello", "texts": ["hello"], "prompt": "hello", "prompt_name": "sts"});
        let (status, body) = post(&state, route, body).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{route}: {body}");
        assert_eq!(message(&body), "model `named` has no prompt named `sts` (its prompts: `STS`, `document`)", "{route}");
    }
    // A model without named prompts says so.
    let (status, body) = post(&state, "/v1/embeddings", json!({"model": "test-static", "input": "hello", "prompt_name": "STS"})).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    assert!(message(&body).starts_with("model `test-static` has no named prompts"), "{body}");
    std::fs::remove_dir_all(&dir).unwrap();
}

#[tokio::test(flavor = "multi_thread")]
async fn start_truncation_keeps_a_named_prompt_whole() {
    let (state, dir) = state("start");
    let start = |extra: Value, text: &str| {
        let mut body = json!({"model": "named", "texts": [text], "truncate": "START", "max_tokens": 3});
        body.as_object_mut().unwrap().extend(extra.as_object().unwrap().clone());
        let state = state.clone();
        async move {
            let (status, body) = post(&state, "/v2/embed", body).await;
            assert_eq!(status, StatusCode::OK, "{body}");
            floats(&body)
        }
    };
    // `STS` is sidekick's prompt, so START keeps it and the text's last
    // token: "world world hello".
    let (c, d) = (1.0 / 17f64.sqrt(), 4.0 / 17f64.sqrt());
    let named = start(json!({"prompt_name": "STS"}), "hello hello hello hello").await;
    assert!(close(&named, &[c, d, 0.0, 0.0]), "{named:?}");
    // The same prompt typed into the text under `none` is text, and START
    // cuts it: "hello hello hello".
    let typed = start(json!({"input_type": "none"}), "world world hello hello hello hello").await;
    assert!(close(&typed, &[1.0, 0.0, 0.0, 0.0]), "{typed:?}");
    std::fs::remove_dir_all(&dir).unwrap();
}

#[tokio::test(flavor = "multi_thread")]
async fn tokenize_counts_and_fits_with_the_named_prompt() {
    let (state, dir) = state("tokenize");
    let count = |prompt_name: &'static str, text: String| {
        let state = state.clone();
        async move {
            let (status, body) =
                post(&state, "/tokenize", json!({"model": "named", "prompt": text, "prompt_name": prompt_name})).await;
            assert_eq!(status, StatusCode::OK, "{body}");
            (body["count"].as_u64().unwrap() as usize, body["fits"].as_bool().unwrap())
        }
    };
    assert_eq!(count("STS", "hello".into()).await, (3, true));
    // 510 words fit after `STS`'s two tokens; 511 don't, and the route
    // agrees.
    assert_eq!(count("STS", "hello ".repeat(510)).await, (512, true));
    assert_eq!(count("STS", "hello ".repeat(511)).await, (513, false));
    for (words, ok) in [(510, true), (511, false)] {
        let body = json!({"model": "named", "input": "hello ".repeat(words), "prompt_name": "STS"});
        assert_eq!(post(&state, "/v1/embeddings", body).await.0 == StatusCode::OK, ok, "{words}");
    }
    std::fs::remove_dir_all(&dir).unwrap();
}

#[tokio::test(flavor = "multi_thread")]
async fn models_list_each_embedders_named_prompts() {
    let (state, dir) = state("models");
    let (status, body) = call(state.clone(), Request::get("/v1/models").body(Body::empty()).unwrap()).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let entry = |id: &str| body["data"].as_array().unwrap().iter().find(|m| m["id"] == id).unwrap().clone();
    assert_eq!(entry("named")["named_prompts"], json!({"STS": "world world ", "document": "hello world "}));
    assert_eq!(entry("named")["prompts"], json!({"query": "world ", "document": "hello world "}));
    assert_eq!(entry("test-static")["named_prompts"], json!({}));
    assert!(entry("apple-fm").get("named_prompts").is_none());
    std::fs::remove_dir_all(&dir).unwrap();
}
