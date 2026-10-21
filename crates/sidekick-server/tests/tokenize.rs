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

/// /tokenize's answer for `text` as `input_type`: (count, max_model_len,
/// fits).
async fn tokenize(state: &AppState, model: &str, text: &str, input_type: Option<&str>) -> (usize, usize, bool) {
    let mut body = json!({"model": model, "prompt": text});
    if let Some(t) = input_type {
        body["input_type"] = json!(t);
    }
    let (status, body) = post(state, "/tokenize", body).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let count = body["count"].as_u64().unwrap() as usize;
    assert_eq!(count, body["tokens"].as_array().unwrap().len());
    (count, body["max_model_len"].as_u64().unwrap() as usize, body["fits"].as_bool().unwrap())
}

/// /v1/embeddings' verdict on `text` under `truncate: NONE`: `None` when
/// it embeds it, else its 400's message.
async fn refusal(state: &AppState, model: &str, text: &str, input_type: Option<&str>) -> Option<String> {
    let mut request = json!({"model": model, "input": text, "truncate": "NONE"});
    if let Some(t) = input_type {
        request["input_type"] = json!(t);
    }
    let (status, body) = post(state, "/v1/embeddings", request).await;
    match status {
        StatusCode::OK => None,
        StatusCode::BAD_REQUEST => Some(body["error"]["message"].as_str().unwrap().to_string()),
        other => panic!("{model}: {other}: {body}"),
    }
}

/// /tokenize on `text`, checked against /v1/embeddings: `fits` is its
/// verdict, and a refusal for length names /tokenize's count and
/// max_model_len. Returns (count, max_model_len, fits).
async fn agrees(state: &AppState, model: &str, text: &str, input_type: Option<&str>) -> (usize, usize, bool) {
    let (n, limit, fits) = tokenize(state, model, text, input_type).await;
    let refused = refusal(state, model, text, input_type).await;
    assert_eq!(fits, refused.is_none(), "{model} {input_type:?}, {} bytes: {refused:?}", text.len());
    if let Some(m) = refused.filter(|m| m.contains(" tokens, over")) {
        let want = format!("input 0: {n} tokens, over model `{model}`'s limit of {limit} tokens");
        assert!(m.starts_with(&want), "{m}\nwanted: {want}");
    }
    (n, limit, fits)
}

/// The contract, for `model` as `input_type`, `word` being one token: a
/// text of exactly `max_model_len` tokens fits and embeds; a longer one is
/// counted whole and refused, naming that count. And `fits`, not the
/// count, decides: a long run without spaces is one token but too many
/// bytes, as is a long run of spaces, which is none. Returns
/// `max_model_len`.
async fn agrees_with_embeddings(state: &AppState, model: &str, word: &str, input_type: Option<&str>) -> usize {
    let (fixed, limit, _) = agrees(state, model, "", input_type).await;
    let text = |words: usize| vec![word; words].join(" ");
    assert_eq!(agrees(state, model, &text(limit - fixed), input_type).await, (limit, limit, true), "{model} {input_type:?}");
    let (n, _, fits) = agrees(state, model, &text(limit - fixed + 7), input_type).await;
    assert_eq!((n, fits), (limit + 7, false), "counted whole");
    // As Thai, Lao or Khmer prose is to a WordPiece vocabulary: one
    // unknown token per run, but past the byte bound.
    let run = "x".repeat(limit * 64 + 1);
    assert_eq!(agrees(state, model, &run, input_type).await, (fixed + 1, limit, false));
    let blank = " ".repeat(limit * 64 + 1);
    assert_eq!(agrees(state, model, &blank, input_type).await, (fixed, limit, false));
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
    // A static model embeds an empty text (no special tokens, no prefix).
    assert_eq!(agrees(&state, "test-static", "", None).await, (0, 512, true));

    // The default input_type is /v1/embeddings': the document prefix.
    for input_type in [None, Some("document"), Some("passage"), Some("query"), Some("none")] {
        assert_eq!(agrees_with_embeddings(&state, "prefixed", "hello", input_type).await, 32);
    }
    assert_eq!(tokenize(&state, "prefixed", "hello", None).await.0, 3);
    assert_eq!(tokenize(&state, "prefixed", "hello", Some("query")).await.0, 2);
    // /v2/embed's names map as /v2/embed maps them.
    assert_eq!(tokenize(&state, "prefixed", "hello", Some("search_document")).await.0, 3);
    assert_eq!(tokenize(&state, "prefixed", "hello", Some("search_query")).await.0, 2);
    // A static model's inputs carry no special tokens, so there are none
    // to leave out; the prefix still counts.
    let (status, body) = post(&state, "/tokenize", json!({"model": "prefixed", "prompt": "hello", "add_special_tokens": false})).await;
    assert_eq!((status, body["count"].clone()), (StatusCode::OK, json!(3)), "{body}");

    // The tokenizer's own limit is the one /v1/embeddings applies, and
    // /tokenize counts past it, where the tokenizer as shipped would stop.
    assert_eq!(agrees_with_embeddings(&state, "truncating", "hello", None).await, 8);
    assert_eq!(tokenize(&state, "truncating", &"hello ".repeat(100), None).await, (100, 8, false));
    std::fs::remove_dir_all(&dir).unwrap();
}

#[tokio::test(flavor = "multi_thread")]
async fn fits_follows_a_v2_embed_max_tokens() {
    let state = test_state(true, None);
    // /v2/embed's NONE measures against max_tokens, and so does `fits`.
    for (text, fits) in [("hello world", true), ("hello world hello", false)] {
        let body = json!({"model": "test-static", "prompt": text, "max_tokens": 2});
        let (status, body) = post(&state, "/tokenize", body).await;
        assert_eq!((status, &body["fits"], &body["max_model_len"]), (StatusCode::OK, &json!(fits), &json!(512)), "{body}");
        let v2 = json!({"model": "test-static", "texts": [text], "truncate": "NONE", "max_tokens": 2});
        let (status, body) = post(&state, "/v2/embed", v2).await;
        assert_eq!(status == StatusCode::OK, fits, "{body}");
    }
    // Validated as /v2/embed validates it.
    for bad in [json!(0), json!(513)] {
        let (status, body) = post(&state, "/tokenize", json!({"model": "test-static", "prompt": "hi", "max_tokens": bad})).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn tokenize_shapes_and_refusals() {
    let state = test_state(true, None);
    // vLLM's shape: ids, and with return_token_strs their vocabulary
    // entries, and the `fits` extension.
    let (status, headers, body) = call_with_headers(
        state.clone(),
        post_json("/tokenize", json!({"model": "test-static", "prompt": "Hello world zebra", "return_token_strs": true})),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(
        body,
        json!({"count": 3, "max_model_len": 512, "tokens": [0, 1, 2], "token_strs": ["hello", "world", "[UNK]"], "fits": true})
    );
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
    let m = refused(json!({"model": "test-static", "prompt": "hi", "input_type": "classification"}), "/tokenize").await;
    assert!(m.starts_with("unsupported input_type `classification`"), "{m}");
    // The chat model has no token ids to give.
    for route in ["/tokenize", "/detokenize"] {
        let m = refused(json!({"model": "apple-fm", "prompt": "hi", "tokens": [1]}), route).await;
        assert!(m.contains("token counts but not token ids"), "{m}");
    }
    // Rerankers and zero-shot formats aren't one text: refused from the
    // manifest, before any tokenizer loads.
    let m = refused(json!({"model": "reranker", "prompt": "hi"}), "/tokenize").await;
    assert!(m.starts_with("a text-ranking model's input is a (query, document) pair"), "{m}");
    let formats = [
        ("decider", "laya"),
        ("schema-decider", "gliner2"),
        ("fev-decider", "fev"),
        ("jev-decider", "agentjev"),
        ("nli-decider", "nli"),
    ];
    for (model, format) in formats {
        let m = refused(json!({"model": model, "prompt": "hi"}), "/tokenize").await;
        assert!(m.starts_with(&format!("a zero-shot model in the {format} format")), "{model}: {m}");
    }
    let (status, body) = post(&state, "/tokenize", json!({"model": "ghost", "prompt": "hi"})).await;
    assert_eq!((status, &body["error"]["code"]), (StatusCode::NOT_FOUND, &json!("model_not_found")));
}

#[tokio::test(flavor = "multi_thread")]
async fn both_routes_need_the_api_key() {
    let state = test_state(true, Some("sekrit"));
    for (route, body) in [
        ("/tokenize", json!({"model": "test-static", "prompt": "hello"})),
        ("/detokenize", json!({"model": "test-static", "tokens": [0]})),
    ] {
        let (status, reply) = post(&state, route, body.clone()).await;
        assert_eq!((status, &reply["error"]["code"]), (StatusCode::UNAUTHORIZED, &json!("invalid_api_key")), "{route}");
        let mut request = post_json(route, body);
        request.headers_mut().insert("authorization", "Bearer sekrit".parse().unwrap());
        let (status, reply) = call(state.clone(), request).await;
        assert_eq!(status, StatusCode::OK, "{route}: {reply}");
    }
}

/// A model whose backend this build lacks is a 503, as on /v1/embeddings.
#[cfg(not(feature = "onnx"))]
#[tokio::test(flavor = "multi_thread")]
async fn a_backend_not_built_in_is_a_503() {
    let manifest = "id = \"tiny\"\nbackend = \"onnx\"\nartifact = \"hidden.onnx\"\ntokenizer = \"tokenizer.json\"\ndims = 8\n\
                    pooling = \"cls\"\nmax_seq_len = 16\n\n[io]\ninput_ids = \"input_ids\"\nattention_mask = \"attention_mask\"\n\
                    token_type_ids = \"token_type_ids\"\nposition_ids = \"position_ids\"\noutput = \"last_hidden_state\"\n";
    let dir = models_dir("no-onnx", &[("tiny", manifest.into(), tokenizer_json(&["[UNK]"], false, None))]);
    let fixtures = Path::new(env!("CARGO_MANIFEST_DIR")).join("../sidekick-embed/tests/fixtures/tiny-onnx");
    std::os::unix::fs::symlink(fixtures.join("hidden.onnx"), dir.join("tiny/hidden.onnx")).unwrap();
    let state = state_over(&dir);
    for (route, body) in [
        ("/tokenize", json!({"model": "tiny", "prompt": "a"})),
        ("/detokenize", json!({"model": "tiny", "tokens": [0]})),
        ("/v1/embeddings", json!({"model": "tiny", "input": "a"})),
    ] {
        let (status, reply) = post(&state, route, body).await;
        assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE, "{route}: {reply}");
    }
    std::fs::remove_dir_all(&dir).unwrap();
}

#[tokio::test(flavor = "multi_thread")]
async fn a_text_classifier_counts_and_fits_as_classify_checks() {
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
        json!({"count": 4, "max_model_len": 64, "tokens": [1, 3, 4, 2], "token_strs": ["[CLS]", "good", "bad", "[SEP]"],
               "fits": true})
    );
    assert_eq!(headers["sidekick-model"], "sentiment@abc123");
    let (status, body) = post(&state, "/tokenize", json!({"model": "sentiment", "prompt": "good bad", "add_special_tokens": false})).await;
    assert_eq!((status, &body["tokens"], &body["fits"]), (StatusCode::OK, &json!([3, 4]), &json!(true)), "{body}");
    // At the limit, past it (counted whole), and past /v1/classify's 16
    // bytes per token: one token, but refused.
    for (text, count, fits) in [
        ("good ".repeat(62), 64, true),
        ("good ".repeat(63), 65, false),
        ("good ".repeat(100), 102, false),
        ("x".repeat(64 * 16 + 1), 3, false),
        (String::new(), 2, true),
    ] {
        let (status, body) = post(&state, "/tokenize", json!({"model": "sentiment", "prompt": text})).await;
        assert_eq!((status, &body["count"], &body["fits"]), (StatusCode::OK, &json!(count), &json!(fits)), "{body}");
    }
    // A classifier has no prompt prefix or max_tokens.
    for extra in [json!({"input_type": "query"}), json!({"max_tokens": 8})] {
        let mut body = json!({"model": "sentiment", "prompt": "good"});
        body.as_object_mut().unwrap().extend(extra.as_object().unwrap().clone());
        let (status, body) = post(&state, "/tokenize", body).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    }
    // Special tokens are kept on the way back.
    let (status, body) = post(&state, "/detokenize", json!({"model": "sentiment", "tokens": [1, 3, 4, 2]})).await;
    assert_eq!((status, body), (StatusCode::OK, json!({"prompt": "[CLS] good bad [SEP]"})));
}

/// sidekick-embed's tiny ONNX graph with `[CLS] … [SEP]` and prompt
/// prefixes, 16 tokens; and again with a tokenizer that truncates at 10 on
/// its own, as LFM2.5-Embedding's does at 512.
#[cfg(feature = "onnx")]
#[tokio::test(flavor = "multi_thread")]
async fn an_onnx_models_count_includes_its_special_tokens_and_prefix() {
    if std::env::var_os("ORT_DYLIB_PATH").is_none() {
        assert!(std::env::var("SIDEKICK_REQUIRE_ONNX").as_deref() != Ok("1"), "SIDEKICK_REQUIRE_ONNX=1 but ORT_DYLIB_PATH isn't set");
        return;
    }
    let words = ["[UNK]", "[CLS]", "[SEP]", "a", "b", "c", "d", "e", "f", "g", "h", "i"];
    let manifest = |id: &str| {
        format!(
            "id = \"{id}\"\nbackend = \"onnx\"\nartifact = \"hidden.onnx\"\ntokenizer = \"tokenizer.json\"\ndims = 8\n\
             pooling = \"cls\"\nmax_seq_len = 16\n\n[io]\ninput_ids = \"input_ids\"\nattention_mask = \"attention_mask\"\n\
             token_type_ids = \"token_type_ids\"\nposition_ids = \"position_ids\"\noutput = \"last_hidden_state\"\n\n\
             [prefixes]\nquery = \"a \"\ndocument = \"b c \"\n"
        )
    };
    let dir = models_dir(
        "onnx",
        &[
            ("tiny", manifest("tiny"), tokenizer_json(&words, true, None)),
            ("tiny-truncating", manifest("tiny-truncating"), tokenizer_json(&words, true, Some(10))),
        ],
    );
    let fixtures = Path::new(env!("CARGO_MANIFEST_DIR")).join("../sidekick-embed/tests/fixtures/tiny-onnx");
    for id in ["tiny", "tiny-truncating"] {
        std::os::unix::fs::symlink(fixtures.join("hidden.onnx"), dir.join(id).join("hidden.onnx")).unwrap();
    }
    let state = state_over(&dir);

    for input_type in [None, Some("query"), Some("none")] {
        assert_eq!(agrees_with_embeddings(&state, "tiny", "d", input_type).await, 16);
        assert_eq!(agrees_with_embeddings(&state, "tiny-truncating", "d", input_type).await, 10);
    }
    // `none`: the text as sent, special tokens still added. The document
    // prompt sent as text gives the default's vector.
    assert_eq!(tokenize(&state, "tiny", "d", Some("none")).await.0, 3);
    let embedding = |input: &str, input_type: &str| {
        let state = state.clone();
        let body = json!({"model": "tiny", "input": input, "input_type": input_type});
        async move {
            let (status, body) = post(&state, "/v1/embeddings", body).await;
            assert_eq!(status, StatusCode::OK, "{body}");
            body["data"][0]["embedding"].clone()
        }
    };
    assert_eq!(embedding("b c d", "none").await, embedding("d", "document").await);
    assert_ne!(embedding("d", "none").await, embedding("d", "document").await);
    assert_eq!(listed_max_seq_len(&state, "tiny").await, 16);
    // /v1/models lists the manifest's; /tokenize the tokenizer's, which
    // /v1/embeddings applies (D44).
    assert_eq!(listed_max_seq_len(&state, "tiny-truncating").await, 16);
    // [CLS] b c d [SEP], and without the special tokens b c d.
    let (status, body) = post(&state, "/tokenize", json!({"model": "tiny", "prompt": "d", "return_token_strs": true})).await;
    assert_eq!((status, &body["token_strs"]), (StatusCode::OK, &json!(["[CLS]", "b", "c", "d", "[SEP]"])), "{body}");
    let (_, body) = post(&state, "/tokenize", json!({"model": "tiny", "prompt": "d", "add_special_tokens": false})).await;
    assert_eq!(body["tokens"], json!([4, 5, 6]));
    std::fs::remove_dir_all(&dir).unwrap();
}

/// sidekick-embed's tiny Qwen3 Core ML model on the CPU, in buckets 16 and
/// 32, with a one-token query prefix; and with `[CLS] … [SEP]` and a
/// tokenizer that truncates at 20 on its own.
#[cfg(all(target_os = "macos", feature = "coreml"))]
#[tokio::test(flavor = "multi_thread")]
async fn a_core_ml_models_count_is_the_one_embeddings_checks() {
    let manifest = |id: &str| {
        format!(
            "id = \"{id}\"\nbackend = \"coreml\"\nartifact = \"model_{{seq}}.mlmodelc\"\n\
             tokenizer = \"tokenizer.json\"\ndims = 32\npooling = \"none\"\nbuckets = [16, 32]\nmax_seq_len = 32\n\
             compute_units = \"cpu_only\"\n\n[io]\ninput_ids = \"input_ids\"\nattention_mask = \"attention_mask\"\n\
             output = \"embedding\"\n\n[prefixes]\nquery = \"b \"\n"
        )
    };
    let dir = models_dir(
        "coreml",
        &[
            ("tiny-qwen3-embed", manifest("tiny-qwen3-embed"), tokenizer_json(&["[UNK]", "a", "b"], false, None)),
            ("tiny-truncating", manifest("tiny-truncating"), tokenizer_json(&["[UNK]", "[CLS]", "[SEP]"], true, Some(20))),
        ],
    );
    let fixtures = Path::new(env!("CARGO_MANIFEST_DIR")).join("../sidekick-embed/tests/fixtures/tiny-qwen3-embed-chunked");
    for id in ["tiny-qwen3-embed", "tiny-truncating"] {
        for bucket in [16, 32] {
            let name = format!("model_{bucket}.mlmodelc");
            std::os::unix::fs::symlink(fixtures.join(&name), dir.join(id).join(&name)).unwrap();
        }
    }
    let state = state_over(&dir);
    for input_type in [None, Some("query"), Some("none")] {
        assert_eq!(agrees_with_embeddings(&state, "tiny-qwen3-embed", "a", input_type).await, 32);
    }
    assert_eq!(agrees_with_embeddings(&state, "tiny-truncating", "x", None).await, 20);
    assert_eq!(listed_max_seq_len(&state, "tiny-qwen3-embed").await, 32);
    // No special tokens and no document prefix: an empty text has no
    // tokens, which a Core ML model can't run. Its query prefix is one.
    assert_eq!(agrees(&state, "tiny-qwen3-embed", "", None).await, (0, 32, false));
    assert_eq!(agrees(&state, "tiny-qwen3-embed", "", Some("query")).await, (1, 32, true));
    assert_eq!(agrees(&state, "tiny-qwen3-embed", "", Some("none")).await, (0, 32, false));
    std::fs::remove_dir_all(&dir).unwrap();
}

/// A Core ML embedder's manifest over a word-level tokenizer: `units`,
/// `buckets`, and any further top-level `extra` lines.
#[cfg(all(target_os = "macos", feature = "coreml"))]
fn coreml_manifest(id: &str, units: &str, buckets: &[usize], extra: &str) -> String {
    format!(
        "id = \"{id}\"\nbackend = \"coreml\"\nartifact = \"model_{{seq}}.mlmodelc\"\ntokenizer = \"tokenizer.json\"\n\
         dims = 32\npooling = \"none\"\nbuckets = {buckets:?}\nmax_seq_len = {}\ncompute_units = \"{units}\"\n{extra}\n\
         [io]\ninput_ids = \"input_ids\"\nattention_mask = \"attention_mask\"\noutput = \"embedding\"\n",
        buckets.last().unwrap()
    )
}

/// The ANE cap (D48) is the limit /tokenize reports: one artifact with a
/// 4,096-token bucket fits 3,000 tokens on the GPU and not on the ANE,
/// where it is capped at 2,048. /tokenize reads only the tokenizer, so the
/// artifacts needn't exist (but the build needs the Core ML backend).
#[cfg(all(target_os = "macos", feature = "coreml"))]
#[tokio::test(flavor = "multi_thread")]
async fn the_ane_cap_is_the_limit_tokenize_reports() {
    let words = ["[UNK]", "a", "b"];
    let buckets = [128, 2048, 4096];
    let dir = models_dir(
        "ane-cap",
        &[
            ("long-ane", coreml_manifest("long-ane", "cpu_and_ne", &buckets, ""), tokenizer_json(&words, false, None)),
            ("long-gpu", coreml_manifest("long-gpu", "cpu_and_gpu", &buckets, ""), tokenizer_json(&words, false, None)),
        ],
    );
    let state = state_over(&dir);
    let text = |n: usize| vec!["a"; n].join(" ");
    assert_eq!(tokenize(&state, "long-ane", &text(3000), None).await, (3000, 2048, false));
    assert_eq!(tokenize(&state, "long-ane", &text(2048), None).await, (2048, 2048, true));
    assert_eq!(tokenize(&state, "long-gpu", &text(3000), None).await, (3000, 4096, true));
    assert_eq!(listed_max_seq_len(&state, "long-ane").await, 2048);
    assert_eq!(listed_max_seq_len(&state, "long-gpu").await, 4096);
    std::fs::remove_dir_all(&dir).unwrap();
}

/// sidekick-embed's tiny Qwen3 Core ML model, in buckets 16 and 32, with a
/// manifest ANE cap of 16 tokens (D48): on the ANE an input over 16 tokens
/// is a 400 naming the limit of 16, and /tokenize says it doesn't fit; on
/// the GPU the same artifact embeds it, and /tokenize says it fits.
#[cfg(all(target_os = "macos", feature = "coreml"))]
#[tokio::test(flavor = "multi_thread")]
async fn on_the_ane_an_input_past_the_ane_cap_is_refused_and_fits_on_the_gpu() {
    let words = ["[UNK]", "a", "b"];
    let cap = "ane_max_seq_len = 16\n";
    let dir = models_dir(
        "coreml-ane-cap",
        &[
            ("tiny-ane", coreml_manifest("tiny-ane", "cpu_and_ne", &[16, 32], cap), tokenizer_json(&words, false, None)),
            ("tiny-gpu", coreml_manifest("tiny-gpu", "cpu_and_gpu", &[16, 32], cap), tokenizer_json(&words, false, None)),
        ],
    );
    let fixtures = Path::new(env!("CARGO_MANIFEST_DIR")).join("../sidekick-embed/tests/fixtures/tiny-qwen3-embed-chunked");
    for id in ["tiny-ane", "tiny-gpu"] {
        for bucket in [16, 32] {
            let name = format!("model_{bucket}.mlmodelc");
            std::os::unix::fs::symlink(fixtures.join(&name), dir.join(id).join(&name)).unwrap();
        }
    }
    let state = state_over(&dir);
    assert_eq!(listed_max_seq_len(&state, "tiny-ane").await, 16);
    assert_eq!(listed_max_seq_len(&state, "tiny-gpu").await, 32);
    // /tokenize's `fits` and /v1/embeddings agree at every length, and a
    // refusal names the effective limit.
    assert_eq!(agrees_with_embeddings(&state, "tiny-ane", "a", None).await, 16);
    assert_eq!(agrees_with_embeddings(&state, "tiny-gpu", "a", None).await, 32);
    let twenty = vec!["a"; 20].join(" ");
    assert_eq!(agrees(&state, "tiny-ane", &twenty, None).await, (20, 16, false));
    assert_eq!(agrees(&state, "tiny-gpu", &twenty, None).await, (20, 32, true));
    let message = refusal(&state, "tiny-ane", &twenty, None).await.unwrap();
    assert!(message.starts_with("input 0: 20 tokens, over model `tiny-ane`'s limit of 16 tokens"), "{message}");
    std::fs::remove_dir_all(&dir).unwrap();
}
