//! The rerank routes (`/v1/rerank`, `/rerank`, `/v2/rerank`) and Cohere's
//! `/v2/embed`, end to end with mock rerankers behind real `classifier.toml`
//! manifests and the real static embedder (docs/design/rerank.md, D29).

mod common;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use common::*;
use serde_json::{json, Value};
use sidekick_core::TruncationSide;

const DOCS: [&str; 4] = [
    "the weather is mild",
    "the cat sat on the mat",
    "a cat and a mat",
    "stocks rose sharply",
];

fn body(route: &str, extra: Value) -> Request<Body> {
    let mut b = json!({"model": "reranker", "query": "cat on mat", "documents": DOCS});
    b.as_object_mut().unwrap().extend(extra.as_object().unwrap().clone());
    post_json(route, b)
}

async fn ok(route: &str, extra: Value) -> Value {
    let (status, value) = call(test_state(true, None), body(route, extra)).await;
    assert_eq!(status, StatusCode::OK, "{value}");
    value
}

async fn bad(route: &str, extra: Value, needle: &str) {
    let (status, value) = call(test_state(true, None), body(route, extra.clone())).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{extra} → {value}");
    let message = value["error"]["message"].as_str().unwrap();
    assert!(message.contains(needle), "{extra}: `{message}` lacks `{needle}`");
}

fn indices(v: &Value) -> Vec<u64> {
    v["results"].as_array().unwrap().iter().map(|r| r["index"].as_u64().unwrap()).collect()
}

#[tokio::test(flavor = "multi_thread")]
async fn v1_rerank_is_vllms_shape_sorted_with_provenance() {
    for route in ["/v1/rerank", "/rerank"] {
        let (status, headers, v) = call_with_headers(test_state(true, None), body(route, json!({}))).await;
        assert_eq!(status, StatusCode::OK, "{v}");
        assert!(v["id"].as_str().unwrap().starts_with("score-"), "vLLM's prefix: {v}");
        assert_eq!(v["model"], "reranker");
        // Shared words with "cat on mat": doc 1 has 3, doc 2 has 2, 0 and 3 none.
        assert_eq!(indices(&v), vec![1, 2, 0, 3], "highest first; ties keep request order");
        let top = &v["results"][0];
        assert_eq!(top["relevance_score"], 3.0, "regression: the raw logit");
        assert_eq!(top["document"], json!({"text": DOCS[1]}));
        assert!(v.get("meta").is_none(), "Cohere's meta is /v2 only");
        let tokens: usize = DOCS.iter().map(|d| "cat on mat|".len() + d.len()).sum();
        assert_eq!(v["usage"], json!({"prompt_tokens": tokens, "total_tokens": tokens}));
        assert_eq!(headers["sidekick-model"], "reranker@r1");
        assert_eq!(headers["sidekick-compute-units"], "cpu_and_ne");
        // One per pair, in request order, not the sorted results' order.
        assert_eq!(headers["sidekick-buckets"], "64,64,64,64");
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn top_n_return_documents_and_activation() {
    let v = ok("/v1/rerank", json!({"top_n": 2})).await;
    assert_eq!(indices(&v), vec![1, 2]);
    // 0 (vLLM's default) and more than the documents both return all.
    assert_eq!(indices(&ok("/v1/rerank", json!({"top_n": 0})).await).len(), 4);
    assert_eq!(indices(&ok("/v1/rerank", json!({"top_n": 9})).await).len(), 4);
    let v = ok("/v1/rerank", json!({"return_documents": false})).await;
    assert!(v["results"][0].get("document").is_none());

    // The single-output default: sigmoid; `use_activation: false`, the logit.
    let v = ok("/v1/rerank", json!({"model": "sigmoid-reranker"})).await;
    let s = v["results"][0]["relevance_score"].as_f64().unwrap();
    assert!((s - 1.0 / (1.0 + (-3.0f64).exp())).abs() < 1e-6, "{s}");
    let v = ok("/v1/rerank", json!({"model": "sigmoid-reranker", "use_activation": false})).await;
    assert_eq!(v["results"][0]["relevance_score"], 3.0);

    // One document may be sent bare, as vLLM allows.
    let v = ok("/v1/rerank", json!({"documents": "a cat"})).await;
    assert_eq!(indices(&v), vec![0]);
}

#[tokio::test(flavor = "multi_thread")]
async fn v2_rerank_is_the_superset_both_clients_parse() {
    let (status, headers, v) = call_with_headers(test_state(true, None), body("/v2/rerank", json!({"top_n": 3}))).await;
    assert_eq!(status, StatusCode::OK, "{v}");
    // Cohere's fields...
    assert!(v["id"].is_string());
    assert_eq!(indices(&v), vec![1, 2, 0]);
    assert!(v["results"][0]["relevance_score"].is_number());
    assert_eq!(v["meta"]["api_version"], json!({"version": "2"}));
    assert_eq!(v["meta"]["billed_units"]["input_tokens"], v["usage"]["prompt_tokens"]);
    // ...and vLLM's required ones.
    assert_eq!(v["model"], "reranker");
    assert!(v["usage"]["total_tokens"].is_number());
    assert_eq!(v["results"][0]["document"]["text"], DOCS[1]);
    assert_eq!(headers["sidekick-model"], "reranker@r1");

    // Cohere v2 has no return_documents and vLLM requires `document`: /v2
    // ignores the field.
    let v = ok("/v2/rerank", json!({"return_documents": false})).await;
    assert!(v["results"].as_array().unwrap().iter().all(|r| r["document"]["text"].is_string()), "{v}");
}

#[tokio::test(flavor = "multi_thread")]
async fn results_sort_by_the_raw_logit_where_the_sigmoid_saturates() {
    // Logits 18 and 20 both round to a sigmoid of 1.0 in f32; sorting by
    // the activated score would tie them and keep request order.
    let query: Vec<String> = (0..20).map(|i| format!("w{i}")).collect();
    let docs = [query[..18].join(" "), query.join(" ")];
    let v = ok("/v1/rerank", json!({"model": "sigmoid-reranker", "query": query.join(" "), "documents": docs})).await;
    assert_eq!(v["results"][0]["relevance_score"], v["results"][1]["relevance_score"], "saturated: {v}");
    assert_eq!(indices(&v), vec![1, 0], "the logit 20 document first");
}

#[tokio::test(flavor = "multi_thread")]
async fn truncation_follows_each_routes_contract() {
    let p = test_state_full(true, None);
    let run = |route: &'static str, extra: Value| call(p.state.clone(), body(route, extra));

    // vLLM: nothing unless asked; -1 is the model's maximum.
    assert_eq!(run("/v1/rerank", json!({})).await.0, StatusCode::OK);
    assert_eq!(run("/v1/rerank", json!({"truncate_prompt_tokens": -1, "truncation_side": "left",
                                         "max_tokens_per_query": 5, "max_tokens_per_doc": 0})).await.0, StatusCode::OK);
    // Cohere: documents truncated, the query kept, max_tokens_per_doc 4096
    // capped at the model's maximum.
    assert_eq!(run("/v2/rerank", json!({})).await.0, StatusCode::OK);
    // A client asking /v2 for vLLM's truncation gets it.
    assert_eq!(run("/v2/rerank", json!({"truncate_prompt_tokens": 32})).await.0, StatusCode::OK);
    let pairs = p.pairs.lock().unwrap().clone();
    let first = |i: usize| pairs[i * DOCS.len()].clone();
    assert_eq!((first(0).truncate_prompt_tokens, first(0).keep_query), (None, false));
    assert_eq!(first(0).max_tokens_per_doc, None);
    assert_eq!(first(1).truncate_prompt_tokens, Some(64));
    assert_eq!(first(1).truncation_side, TruncationSide::Left);
    assert_eq!((first(1).max_tokens_per_query, first(1).max_tokens_per_doc), (Some(5), None));
    assert_eq!((first(2).keep_query, first(2).max_tokens_per_doc), (true, Some(64)));
    assert_eq!((first(3).keep_query, first(3).truncate_prompt_tokens), (false, Some(32)));
}

#[tokio::test(flavor = "multi_thread")]
async fn requests_are_checked_in_full_before_anything_runs() {
    for (extra, needle) in [
        (json!({"documents": []}), "must not be empty"),
        (json!({"documents": ["a", "b", "c", "d", "e"]}), "maximum of 4"),
        (json!({"documents": [{"text": "a"}]}), "document objects"),
        (json!({"query": null}), "`query` is required"),
        (json!({"query": {"content": []}}), "must be a string"),
        (json!({"top_n": -1}), "`top_n` must be a non-negative integer"),
        (json!({"truncate_prompt_tokens": 65}), "truncate_prompt_tokens"),
        (json!({"max_tokens_per_doc": 64}), "max_tokens_per_doc must be"),
        (json!({"max_tokens_per_query": -1}), "max_tokens_per_query must be"),
        (json!({"truncation_side": "middle"}), "truncation_side"),
        (json!({"instruction": "judge relevance"}), "chat template"),
        (json!({"chat_template_kwargs": {"a": 1}}), "chat template"),
        (json!({"priority": 2}), "priority scheduling"),
        (json!({"normalize": true}), "`normalize` was removed"),
    ] {
        bad("/v1/rerank", extra, needle).await;
    }
    // A document that fails `prepare` stops the batch before any pair runs.
    let p = test_state_full(true, None);
    let (status, v) = call(p.state, body("/v1/rerank", json!({"documents": ["cat", "reject this"]}))).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{v}");
    assert!(v["error"]["message"].as_str().unwrap().contains("document 1: rejected by prepare"));
    assert_eq!(p.runs.load(std::sync::atomic::Ordering::SeqCst), 0);
    // Bad model output is a 500.
    for doc in ["nan here", "fail now"] {
        let (status, _) = call(test_state(true, None), body("/v1/rerank", json!({"documents": [doc]}))).await;
        assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR, "{doc}");
    }
    // request_id sets the id, as for classify.
    let v = ok("/v1/rerank", json!({"request_id": "r-7"})).await;
    assert_eq!(v["id"], "score-r-7");
}

#[tokio::test(flavor = "multi_thread")]
async fn rerankers_serve_the_rerank_routes_only() {
    let (status, v) = call(test_state(true, None), post_json("/v1/classify", json!({"model": "reranker", "input": "a"}))).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    let m = v["error"]["message"].as_str().unwrap();
    assert!(m.contains("text-ranking") && m.contains("/v1/rerank"), "{m}");
    for model in ["sentiment", "test-static", "apple-fm"] {
        let (status, v) = call(test_state(true, None), body("/v1/rerank", json!({"model": model}))).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{model}: {v}");
    }
    let (status, _) = call(test_state(true, None), body("/v1/rerank", json!({"model": "nope"}))).await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    let (status, _) = call(test_state(true, None), post_json("/v1/embeddings", json!({"model": "reranker", "input": "a"}))).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);

    let (_, models) = call(test_state(true, None), Request::get("/v1/models").body(Body::empty()).unwrap()).await;
    let r = models["data"].as_array().unwrap().iter().find(|m| m["id"] == "reranker").unwrap().clone();
    assert_eq!(r["task"], "text-ranking");
    assert_eq!(r["max_batch"], 4);
    assert_eq!(r["extensions"], json!(["return_documents"]));
    assert!(r.get("labels").is_none() && r.get("max_labels").is_none());
}

#[tokio::test(flavor = "multi_thread")]
async fn new_routes_require_the_api_key() {
    for (route, b) in [
        ("/v1/rerank", json!({"model": "reranker", "query": "a", "documents": ["b"]})),
        ("/rerank", json!({"model": "reranker", "query": "a", "documents": ["b"]})),
        ("/v2/rerank", json!({"model": "reranker", "query": "a", "documents": ["b"]})),
        ("/v2/embed", json!({"model": "test-static", "texts": ["hello"]})),
    ] {
        let (status, _) = call(test_state(true, Some("secret")), post_json(route, b.clone())).await;
        assert_eq!(status, StatusCode::UNAUTHORIZED, "{route}");
        let req = Request::post(route)
            .header("content-type", "application/json")
            .header("authorization", "Bearer secret")
            .body(Body::from(b.to_string()))
            .unwrap();
        let (status, v) = call(test_state(true, Some("secret")), req).await;
        assert_eq!(status, StatusCode::OK, "{route}: {v}");
    }
}

// ---------- /v2/embed ----------

fn embed(extra: Value) -> Request<Body> {
    let mut b = json!({"model": "test-static", "texts": ["hello world", "world"]});
    b.as_object_mut().unwrap().extend(extra.as_object().unwrap().clone());
    post_json("/v2/embed", b)
}

#[tokio::test(flavor = "multi_thread")]
async fn v2_embed_is_cohere_shaped_by_type() {
    let (status, headers, v) = call_with_headers(
        test_state(true, None),
        embed(json!({"input_type": "search_document", "embedding_types": ["float", "base64"]})),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{v}");
    assert!(v["id"].as_str().unwrap().starts_with("embd-"));
    assert_eq!(v["response_type"], "embeddings_by_type");
    assert_eq!(v["texts"], json!(["hello world", "world"]));
    assert_eq!(v["meta"]["api_version"]["version"], "2");
    assert!(v["embeddings"].get("binary").is_none(), "only the requested types");
    // The fixture: hello → e0, world → 2·e1, so "hello world" → (1, 2, 0, 0)/√5.
    let f = &v["embeddings"]["float"];
    let inv = 1.0 / 5f64.sqrt();
    assert!((f[0][0].as_f64().unwrap() - inv).abs() < 1e-6);
    assert!((f[0][1].as_f64().unwrap() - 2.0 * inv).abs() < 1e-6);
    assert_eq!(f[1], json!([0.0, 1.0, 0.0, 0.0]));
    // base64: the same floats, little-endian f32.
    use base64::Engine as _;
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(v["embeddings"]["base64"][1].as_str().unwrap())
        .unwrap();
    let floats: Vec<f32> = bytes.chunks(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect();
    assert_eq!(floats, vec![0.0, 1.0, 0.0, 0.0]);
    assert_eq!(headers["sidekick-model"], "test-static");
    assert_eq!(headers["sidekick-compute-units"], "cpu");
    assert!(headers.get("sidekick-buckets").is_none(), "a static embedder has no buckets");

    // Bits pack 8 dimensions per byte (packing itself: embed_v2's unit
    // test); this fixture has 4.
    for t in ["binary", "ubinary"] {
        let (status, v) = call(test_state(true, None), embed(json!({"embedding_types": [t]}))).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{t}");
        assert!(v["error"]["message"].as_str().unwrap().contains("multiple of 8"));
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn v2_embed_rejects_what_it_cannot_honor() {
    for (extra, needle) in [
        (json!({"input_type": "classification"}), "unsupported input_type"),
        (json!({"embedding_types": ["int8"]}), "calibration ranges"),
        (json!({"embedding_types": ["floats"]}), "unknown embedding type"),
        (json!({"embedding_types": []}), "must not be empty"),
        (json!({"texts": []}), "`texts` is required"),
        (json!({"images": ["data:,"]}), "text only"),
        (json!({"truncate": "MIDDLE"}), "unsupported truncate"),
        (json!({"output_dimension": 3}), "supports output_dimension [4, 2]"),
        (json!({"max_tokens": 0}), "max_tokens must be positive"),
        (json!({"max_tokens": 513}), "exceeds model `test-static`'s maximum of 512"),
        (json!({"priority": 1}), "priority scheduling"),
        (json!({"priority": 0.0}), "must be an integer"),
    ] {
        let (status, v) = call(test_state(true, None), embed(extra.clone())).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{extra}: {v}");
        let m = v["error"]["message"].as_str().unwrap();
        assert!(m.contains(needle), "{extra}: `{m}` lacks `{needle}`");
    }
    let (status, v) = call(test_state(true, None), post_json("/v2/embed", json!({"model": "reranker", "texts": ["a"]}))).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(v["error"]["message"].as_str().unwrap().contains("text-ranking"));
}

#[tokio::test(flavor = "multi_thread")]
async fn v2_embed_truncation_and_dimensions() {
    // The static fixture's max_seq_len is 512 tokens.
    let long = vec!["hello"; 600].join(" ") + " world";
    let request = |extra: Value| {
        let mut b = json!({"model": "test-static", "texts": [long.clone()]});
        b.as_object_mut().unwrap().extend(extra.as_object().unwrap().clone());
        post_json("/v2/embed", b)
    };
    // END (default) keeps the start: all "hello" → e0.
    let (_, v) = call(test_state(true, None), request(json!({}))).await;
    assert_eq!(v["embeddings"]["float"][0], json!([1.0, 0.0, 0.0, 0.0]));
    // START keeps the end, where "world" is.
    let (_, v) = call(test_state(true, None), request(json!({"truncate": "START", "max_tokens": 2}))).await;
    let f = &v["embeddings"]["float"][0];
    assert!(f[1].as_f64().unwrap() > 0.8, "{f}");
    // NONE: an over-long input is a 400.
    let (status, v) = call(test_state(true, None), request(json!({"truncate": "NONE"}))).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{v}");
    assert!(v["error"]["message"].as_str().unwrap().contains("truncate is NONE"));
    // max_tokens with END.
    let (_, v) = call(test_state(true, None), post_json("/v2/embed", json!({"model": "test-static", "texts": ["world hello"], "max_tokens": 1}))).await;
    assert_eq!(v["embeddings"]["float"][0], json!([0.0, 1.0, 0.0, 0.0]));
    // output_dimension: the Matryoshka 2.
    let (_, v) = call(test_state(true, None), post_json("/v2/embed", json!({"model": "test-static", "texts": ["world"], "output_dimension": 2}))).await;
    assert_eq!(v["embeddings"]["float"][0], json!([0.0, 1.0]));
}
