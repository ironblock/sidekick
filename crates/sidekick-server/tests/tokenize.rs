//! `/tokenize` and `/detokenize` (D44): vLLM's shape over an embedder's or
//! a text-classification model's tokenizer. The contract that matters: an
//! embedder's `count`, with the default options, is the number
//! /v1/embeddings compares with the limit under `truncate: NONE` (prompt
//! prefix and special tokens included), and `max_model_len` is that
//! limit. So a text of `max_model_len` tokens embeds, and one past it is
//! refused with /tokenize's count in the message.
//!
//! Over static test models (always), sidekick-embed's tiny ONNX graph
//! (`--features onnx` with ORT_DYLIB_PATH), and its tiny Core ML model (on
//! macOS with `--features coreml`).

mod common;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use common::*;
use serde_json::{json, Value};
use sidekick_core::ModelRegistry;
use sidekick_server::{AppState, EmbedderPool, TokenizerPool};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

/// A word-level tokenizer.json over `words`, with `[CLS] … [SEP]` around
/// each input when `specials` (ids 1 and 2, which must be in `words`), and
/// truncating at `truncate_at` tokens on its own when given, as
/// LFM2.5-Embedding's does at 512.
fn tokenizer_json(words: &[&str], specials: bool, truncate_at: Option<usize>) -> Value {
    let vocab: serde_json::Map<String, Value> = words.iter().enumerate().map(|(i, w)| (w.to_string(), i.into())).collect();
    let special = |id: usize, content: &str| {
        json!({"id": id, "content": content, "single_word": false, "lstrip": false, "rstrip": false,
               "normalized": false, "special": true})
    };
    let (added, post) = if specials {
        assert_eq!((words[1], words[2]), ("[CLS]", "[SEP]"));
        let post = json!({
            "type": "TemplateProcessing",
            "single": [{"SpecialToken": {"id": "[CLS]", "type_id": 0}}, {"Sequence": {"id": "A", "type_id": 0}},
                       {"SpecialToken": {"id": "[SEP]", "type_id": 0}}],
            "pair": [{"SpecialToken": {"id": "[CLS]", "type_id": 0}}, {"Sequence": {"id": "A", "type_id": 0}},
                     {"SpecialToken": {"id": "[SEP]", "type_id": 0}}, {"Sequence": {"id": "B", "type_id": 1}},
                     {"SpecialToken": {"id": "[SEP]", "type_id": 1}}],
            "special_tokens": {
                "[CLS]": {"id": "[CLS]", "ids": [1], "tokens": ["[CLS]"]},
                "[SEP]": {"id": "[SEP]", "ids": [2], "tokens": ["[SEP]"]}
            }
        });
        (json!([special(1, "[CLS]"), special(2, "[SEP]")]), post)
    } else {
        (json!([]), Value::Null)
    };
    let truncation = truncate_at.map_or(Value::Null, |max_length| {
        json!({"direction": "Right", "max_length": max_length, "strategy": "LongestFirst", "stride": 0})
    });
    json!({
        "version": "1.0", "truncation": truncation, "padding": null, "added_tokens": added, "normalizer": null,
        "pre_tokenizer": {"type": "WhitespaceSplit"}, "post_processor": post,
        "decoder": {"type": "WordPiece", "prefix": "##", "cleanup": false},
        "model": {"type": "WordLevel", "vocab": vocab, "unk_token": "[UNK]"}
    })
}

/// A models directory with the static test model (`test-static`: no
/// prefixes, no special tokens, 512 tokens), and `extra`: (id, manifest,
/// tokenizer.json) models whose artifact is test-static's (static ones) or
/// linked in by the caller.
fn models_dir(tag: &str, extra: &[(&str, String, Value)]) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("sk-tokenize-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    write_embedding_fixture(&dir);
    for (id, manifest, tokenizer) in extra {
        let model = dir.join(id);
        std::fs::create_dir_all(&model).unwrap();
        std::fs::write(model.join("manifest.toml"), manifest).unwrap();
        std::fs::write(model.join("tokenizer.json"), tokenizer.to_string()).unwrap();
        if manifest.contains("backend = \"static\"") {
            std::fs::copy(dir.join("test-static/model.safetensors"), model.join("model.safetensors")).unwrap();
        }
    }
    dir
}

/// The test state over `dir`'s models.
fn state_over(dir: &Path) -> AppState {
    let registry = Arc::new(ModelRegistry::scan(dir).unwrap());
    assert!(registry.skipped().is_empty(), "{:?}", registry.skipped().first().map(|s| &s.reason));
    let mut state = test_state(true, None);
    state.embedders = Arc::new(EmbedderPool::embedders(registry.clone(), Duration::from_secs(60)));
    state.tokenizers = Arc::new(TokenizerPool::tokenizers(registry.clone(), Duration::from_secs(60)));
    state.registry = registry;
    state
}

/// A static model over test-static's 4-row table, with `prefixes` (a TOML
/// table body) and `max_seq_len`.
fn static_manifest(id: &str, max_seq_len: usize, prefixes: &str) -> String {
    format!(
        "id = \"{id}\"\nbackend = \"static\"\nartifact = \"model.safetensors\"\ntokenizer = \"tokenizer.json\"\n\
         dims = 4\nmax_seq_len = {max_seq_len}\n\n[prefixes]\n{prefixes}\n"
    )
}

async fn post(state: &AppState, route: &str, body: Value) -> (StatusCode, Value) {
    call(state.clone(), post_json(route, body)).await
}

/// /tokenize's count and max_model_len for `text`, as `input_type`.
async fn count(state: &AppState, model: &str, text: &str, input_type: Option<&str>) -> (usize, usize) {
    let mut body = json!({"model": model, "prompt": text});
    if let Some(t) = input_type {
        body["input_type"] = json!(t);
    }
    let (status, body) = post(state, "/tokenize", body).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let count = body["count"].as_u64().unwrap() as usize;
    assert_eq!(count, body["tokens"].as_array().unwrap().len());
    (count, body["max_model_len"].as_u64().unwrap() as usize)
}

/// The contract, for `model` as `input_type`: /tokenize counts a text of
/// exactly `max_model_len` tokens, which /v1/embeddings embeds under
/// `truncate: NONE`, and a longer one, which it refuses naming the same
/// count. `word` is one token. Returns `max_model_len`.
async fn agrees_with_embeddings(state: &AppState, model: &str, word: &str, input_type: Option<&str>) -> usize {
    let (fixed, limit) = count(state, model, "", input_type).await;
    let text = |words: usize| vec![word; words].join(" ");
    let mut request = json!({"model": model, "truncate": "NONE"});
    if let Some(t) = input_type {
        request["input_type"] = json!(t);
    }

    let at_limit = text(limit - fixed);
    assert_eq!(count(state, model, &at_limit, input_type).await, (limit, limit), "{model} {input_type:?}");
    request["input"] = json!(at_limit);
    let (status, body) = post(state, "/v1/embeddings", request.clone()).await;
    assert_eq!(status, StatusCode::OK, "{model} {input_type:?}, at the limit: {body}");

    // Well past the limit: counted whole, not truncated.
    let over = text(limit - fixed + 7);
    let (n, _) = count(state, model, &over, input_type).await;
    assert_eq!(n, limit + 7);
    request["input"] = json!(over);
    let (status, body) = post(state, "/v1/embeddings", request).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    let message = body["error"]["message"].as_str().unwrap();
    let want = format!("input 0: {n} tokens, over model `{model}`'s limit of {limit} tokens");
    assert!(message.starts_with(&want), "{message}\nwanted: {want}");
    limit
}

/// The `max_seq_len` /v1/models lists for `model`.
async fn listed_max_seq_len(state: &AppState, model: &str) -> usize {
    let (status, body) = call(state.clone(), Request::get("/v1/models").body(Body::empty()).unwrap()).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let entry = body["data"].as_array().unwrap().iter().find(|m| m["id"] == model).unwrap();
    entry["max_seq_len"].as_u64().unwrap() as usize
}

#[tokio::test(flavor = "multi_thread")]
async fn a_static_models_count_is_the_one_embeddings_checks() {
    let words = ["hello", "world", "[UNK]"];
    let dir = models_dir(
        "static",
        &[
            // Prefixes of one and two tokens, which count.
            (
                "prefixed",
                static_manifest("prefixed", 32, "query = \"world \"\ndocument = \"hello world \""),
                tokenizer_json(&words, false, None),
            ),
            // Truncates at 8 tokens on its own, under a 512 max_seq_len.
            ("truncating", static_manifest("truncating", 512, ""), tokenizer_json(&words, false, Some(8))),
        ],
    );
    let state = state_over(&dir);

    assert_eq!(agrees_with_embeddings(&state, "test-static", "hello", None).await, 512);
    assert_eq!(listed_max_seq_len(&state, "test-static").await, 512);

    // The default input_type is /v1/embeddings': the document prefix.
    for input_type in [None, Some("document"), Some("passage"), Some("query")] {
        assert_eq!(agrees_with_embeddings(&state, "prefixed", "hello", input_type).await, 32);
    }
    assert_eq!(count(&state, "prefixed", "hello", None).await.0, 3);
    assert_eq!(count(&state, "prefixed", "hello", Some("document")).await.0, 3);
    assert_eq!(count(&state, "prefixed", "hello", Some("query")).await.0, 2);
    // A static model's inputs carry no special tokens, so there are none
    // to leave out; the prefix still counts.
    let (status, body) = post(&state, "/tokenize", json!({"model": "prefixed", "prompt": "hello", "add_special_tokens": false})).await;
    assert_eq!((status, body["count"].clone()), (StatusCode::OK, json!(3)), "{body}");

    // The tokenizer's own limit is the one /v1/embeddings applies, and
    // /tokenize counts past it, where the tokenizer as shipped would stop.
    assert_eq!(agrees_with_embeddings(&state, "truncating", "hello", None).await, 8);
    assert_eq!(count(&state, "truncating", &"hello ".repeat(100), None).await, (100, 8));
    std::fs::remove_dir_all(&dir).unwrap();
}

#[tokio::test(flavor = "multi_thread")]
async fn tokenize_shapes_and_refusals() {
    let state = test_state(true, None);
    // vLLM's shape: ids, and with return_token_strs their vocabulary entries.
    let (status, headers, body) = call_with_headers(
        state.clone(),
        post_json("/tokenize", json!({"model": "test-static", "prompt": "Hello world zebra", "return_token_strs": true})),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body, json!({"count": 3, "max_model_len": 512, "tokens": [0, 1, 2], "token_strs": ["hello", "world", "[UNK]"]}));
    assert_eq!(headers["sidekick-model"], "test-static");
    let (status, body) = post(&state, "/tokenize", json!({"model": "test-static", "prompt": "hello"})).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert!(body.get("token_strs").is_none(), "{body}");

    // /detokenize: ids back to text; an id outside the vocabulary is a 400.
    let (status, body) = post(&state, "/detokenize", json!({"model": "test-static", "tokens": [0, 1]})).await;
    assert_eq!((status, body), (StatusCode::OK, json!({"prompt": "hello world"})));
    let (status, body) = post(&state, "/detokenize", json!({"model": "test-static", "tokens": [0, 3]})).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    assert_eq!(body["error"]["message"], "token id 3 isn't in the model's vocabulary of 3 tokens");
    let (status, _) = post(&state, "/detokenize", json!({"model": "test-static", "tokens": [-1]})).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);

    let refused = |body: Value, route: &'static str| {
        let state = state.clone();
        async move {
            let (status, body) = post(&state, route, body).await;
            assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
            assert_eq!(body["error"]["type"], "invalid_request_error");
            body["error"]["message"].as_str().unwrap().to_string()
        }
    };
    // Encoders have no chat template.
    let m = refused(json!({"model": "test-static", "messages": [{"role": "user", "content": "hi"}]}), "/tokenize").await;
    assert!(m.starts_with("`messages` isn't supported"), "{m}");
    let m = refused(json!({"model": "test-static"}), "/tokenize").await;
    assert_eq!(m, "`prompt` is required");
    let m = refused(json!({"model": "test-static", "prompt": "hi", "input_type": "search_query"}), "/tokenize").await;
    assert!(m.starts_with("unsupported input_type `search_query`"), "{m}");
    // The chat model has no token ids to give.
    for route in ["/tokenize", "/detokenize"] {
        let m = refused(json!({"model": "apple-fm", "prompt": "hi", "tokens": [1]}), route).await;
        assert!(m.contains("token counts but not token ids"), "{m}");
    }
    // Rerankers and zero-shot formats aren't one text: refused from the
    // manifest, before any tokenizer loads.
    let m = refused(json!({"model": "reranker", "prompt": "hi"}), "/tokenize").await;
    assert!(m.starts_with("a text-ranking model's input is a (query, document) pair"), "{m}");
    for (model, format) in [("decider", "laya"), ("schema-decider", "gliner2"), ("fev-decider", "fev"), ("nli-decider", "nli")] {
        let m = refused(json!({"model": model, "prompt": "hi"}), "/tokenize").await;
        assert!(m.starts_with(&format!("a zero-shot model in the {format} format")), "{model}: {m}");
    }
    let (status, body) = post(&state, "/tokenize", json!({"model": "ghost", "prompt": "hi"})).await;
    assert_eq!((status, &body["error"]["code"]), (StatusCode::NOT_FOUND, &json!("model_not_found")));
}

#[tokio::test(flavor = "multi_thread")]
async fn a_text_classifier_counts_with_its_special_tokens() {
    let state = test_state(true, None);
    // The sentiment test model, 64 tokens, given a tokenizer.json.
    let dir = state.registry.classifier("sentiment").unwrap().dir.clone();
    let words = ["[UNK]", "[CLS]", "[SEP]", "good", "bad"];
    std::fs::write(dir.join("tokenizer.json"), tokenizer_json(&words, true, None).to_string()).unwrap();

    let (status, headers, body) = call_with_headers(
        state.clone(),
        post_json("/tokenize", json!({"model": "sentiment", "prompt": "good bad", "return_token_strs": true})),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(
        body,
        json!({"count": 4, "max_model_len": 64, "tokens": [1, 3, 4, 2], "token_strs": ["[CLS]", "good", "bad", "[SEP]"]})
    );
    assert_eq!(headers["sidekick-model"], "sentiment@abc123");
    let (status, body) = post(&state, "/tokenize", json!({"model": "sentiment", "prompt": "good bad", "add_special_tokens": false})).await;
    assert_eq!((status, &body["tokens"]), (StatusCode::OK, &json!([3, 4])), "{body}");
    // Past the limit: counted whole.
    let (status, body) = post(&state, "/tokenize", json!({"model": "sentiment", "prompt": "good ".repeat(100)})).await;
    assert_eq!((status, &body["count"]), (StatusCode::OK, &json!(102)), "{body}");
    // A classifier has no prompt prefix to pick.
    let (status, body) = post(&state, "/tokenize", json!({"model": "sentiment", "prompt": "good", "input_type": "query"})).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    // Special tokens are kept on the way back.
    let (status, body) = post(&state, "/detokenize", json!({"model": "sentiment", "tokens": [1, 3, 4, 2]})).await;
    assert_eq!((status, body), (StatusCode::OK, json!({"prompt": "[CLS] good bad [SEP]"})));
}

/// sidekick-embed's tiny ONNX graph with `[CLS] … [SEP]` and prompt
/// prefixes, 16 tokens.
#[cfg(feature = "onnx")]
#[tokio::test(flavor = "multi_thread")]
async fn an_onnx_models_count_includes_its_special_tokens_and_prefix() {
    if std::env::var_os("ORT_DYLIB_PATH").is_none() {
        assert!(std::env::var("SIDEKICK_REQUIRE_ONNX").as_deref() != Ok("1"), "SIDEKICK_REQUIRE_ONNX=1 but ORT_DYLIB_PATH isn't set");
        return;
    }
    let words = ["[UNK]", "[CLS]", "[SEP]", "a", "b", "c", "d", "e", "f", "g", "h", "i"];
    let manifest = "id = \"tiny\"\nbackend = \"onnx\"\nartifact = \"hidden.onnx\"\ntokenizer = \"tokenizer.json\"\ndims = 8\n\
                    pooling = \"cls\"\nmax_seq_len = 16\n\n[io]\ninput_ids = \"input_ids\"\nattention_mask = \"attention_mask\"\n\
                    token_type_ids = \"token_type_ids\"\nposition_ids = \"position_ids\"\noutput = \"last_hidden_state\"\n\n\
                    [prefixes]\nquery = \"a \"\ndocument = \"b c \"\n";
    let dir = models_dir("onnx", &[("tiny", manifest.into(), tokenizer_json(&words, true, None))]);
    let fixtures = Path::new(env!("CARGO_MANIFEST_DIR")).join("../sidekick-embed/tests/fixtures/tiny-onnx");
    std::os::unix::fs::symlink(fixtures.join("hidden.onnx"), dir.join("tiny/hidden.onnx")).unwrap();
    let state = state_over(&dir);

    for input_type in [None, Some("query")] {
        assert_eq!(agrees_with_embeddings(&state, "tiny", "d", input_type).await, 16);
    }
    assert_eq!(listed_max_seq_len(&state, "tiny").await, 16);
    // [CLS] b c d [SEP], and without the special tokens b c d.
    let (status, body) = post(&state, "/tokenize", json!({"model": "tiny", "prompt": "d", "return_token_strs": true})).await;
    assert_eq!((status, &body["token_strs"]), (StatusCode::OK, &json!(["[CLS]", "b", "c", "d", "[SEP]"])), "{body}");
    let (_, body) = post(&state, "/tokenize", json!({"model": "tiny", "prompt": "d", "add_special_tokens": false})).await;
    assert_eq!(body["tokens"], json!([4, 5, 6]));
    std::fs::remove_dir_all(&dir).unwrap();
}

/// sidekick-embed's tiny Qwen3 Core ML model on the CPU, in buckets 16 and
/// 32, with a one-token query prefix.
#[cfg(all(target_os = "macos", feature = "coreml"))]
#[tokio::test(flavor = "multi_thread")]
async fn a_core_ml_models_count_is_the_one_embeddings_checks() {
    let manifest = "id = \"tiny-qwen3-embed\"\nbackend = \"coreml\"\nartifact = \"model_{seq}.mlmodelc\"\n\
                    tokenizer = \"tokenizer.json\"\ndims = 32\npooling = \"none\"\nbuckets = [16, 32]\nmax_seq_len = 32\n\
                    compute_units = \"cpu_only\"\n\n[io]\ninput_ids = \"input_ids\"\nattention_mask = \"attention_mask\"\n\
                    output = \"embedding\"\n\n[prefixes]\nquery = \"b \"\n";
    let dir = models_dir("coreml", &[("tiny-qwen3-embed", manifest.into(), tokenizer_json(&["[UNK]", "a", "b"], false, None))]);
    let fixtures = Path::new(env!("CARGO_MANIFEST_DIR")).join("../sidekick-embed/tests/fixtures/tiny-qwen3-embed-chunked");
    for bucket in [16, 32] {
        let name = format!("model_{bucket}.mlmodelc");
        std::os::unix::fs::symlink(fixtures.join(&name), dir.join("tiny-qwen3-embed").join(&name)).unwrap();
    }
    let state = state_over(&dir);
    for input_type in [None, Some("query")] {
        assert_eq!(agrees_with_embeddings(&state, "tiny-qwen3-embed", "a", input_type).await, 32);
    }
    assert_eq!(listed_max_seq_len(&state, "tiny-qwen3-embed").await, 32);
    std::fs::remove_dir_all(&dir).unwrap();
}
