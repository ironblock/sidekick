//! `POST /v1/classify` end to end with mock classifiers behind real
//! `classifier.toml` manifests, plus what the endpoint changed elsewhere:
//! task-aware listings, wrong-route errors, JSON errors on every route and
//! provenance headers.

mod common;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use common::*;
use serde_json::{json, Value};
use sidekick_core::{activate, ProblemType, TruncationSide};
use std::time::Duration;

fn classify(body: Value) -> Request<Body> {
    post_json("/v1/classify", body)
}

async fn classify_ok(body: Value) -> Value {
    let (status, value) = call(test_state(true, None), classify(body)).await;
    assert_eq!(status, StatusCode::OK, "{value}");
    value
}

/// A 400 in the ApiError shape whose message contains `needle`.
async fn classify_400(body: Value, needle: &str) {
    let (status, value) = call(test_state(true, None), classify(body.clone())).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body} → {value}");
    assert_eq!(value["error"]["type"], "invalid_request_error", "{value}");
    let message = value["error"]["message"].as_str().unwrap();
    assert!(message.contains(needle), "{body}: `{message}` lacks `{needle}`");
}

fn probs(v: &Value) -> Vec<f32> {
    v["probs"].as_array().unwrap().iter().map(|p| p.as_f64().unwrap() as f32).collect()
}

#[tokio::test(flavor = "multi_thread")]
async fn text_classification_round_trip_with_provenance() {
    let (status, headers, body) = call_with_headers(
        test_state(true, None),
        classify(json!({"model": "sentiment", "input": "a good film", "user": "u-1"})),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert!(body["id"].as_str().unwrap().starts_with("classify-"));
    assert_eq!(body["object"], "list");
    assert_eq!(body["model"], "sentiment");
    assert!(body["created"].as_u64().unwrap() > 0);
    let data = body["data"].as_array().unwrap();
    assert_eq!(data.len(), 1);
    assert_eq!(data[0]["index"], 0);
    assert_eq!(data[0]["label"], "5 stars");
    assert_eq!(data[0]["num_classes"], 5);
    let p = probs(&data[0]);
    assert_eq!(p, activate(ProblemType::SingleLabel, &[0.0, 1.0, 2.0, 3.0, 4.0], None));
    assert!((p.iter().sum::<f32>() - 1.0).abs() < 1e-6);
    // The mock's tokens are the input's bytes.
    assert_eq!(body["usage"], json!({"prompt_tokens": 11, "completion_tokens": 0, "total_tokens": 11}));

    assert_eq!(headers["sidekick-version"], env!("CARGO_PKG_VERSION"));
    assert_eq!(headers["sidekick-model"], "sentiment@abc123");
    assert_eq!(headers["sidekick-compute-units"], "cpu_and_ne");
}

#[tokio::test(flavor = "multi_thread")]
async fn batches_keep_order_and_use_activation_false_returns_logits() {
    let body = classify_ok(json!({"model": "sentiment", "input": ["bad", "good"], "use_activation": false})).await;
    let data = body["data"].as_array().unwrap();
    assert_eq!((data[0]["index"].clone(), data[1]["index"].clone()), (json!(0), json!(1)));
    assert_eq!(data[0]["label"], "1 star");
    assert_eq!(probs(&data[0]), vec![5.0, 4.0, 3.0, 2.0, 1.0]);
    assert_eq!(data[1]["label"], "5 stars");
    assert_eq!(body["usage"]["prompt_tokens"], 7);
}

#[tokio::test(flavor = "multi_thread")]
async fn input_and_vllm_fields_are_honored_or_rejected() {
    // max_batch is the model's: 4 for sentiment.
    classify_400(json!({"model": "sentiment", "input": ["a", "b", "c", "d", "e"]}), "maximum of 4").await;
    classify_ok(json!({"model": "sentiment", "input": ["a", "b", "c", "d"]})).await;
    classify_400(json!({"model": "sentiment", "input": []}), "must not be empty").await;
    classify_400(json!({"model": "sentiment"}), "`input` is required").await;
    classify_400(json!({"model": "sentiment", "input": [1, 2, 3]}), "token-id input").await;
    classify_400(json!({"model": "sentiment", "input": 7}), "string or an array of strings").await;
    classify_400(
        json!({"model": "sentiment", "messages": [{"role": "user", "content": "hi"}]}),
        "messages",
    )
    .await;
    classify_400(json!({"model": "sentiment", "input": "a", "add_special_tokens": false}), "add_special_tokens").await;
    classify_ok(json!({"model": "sentiment", "input": "a", "add_special_tokens": true})).await;
    // Fields vLLM's classify request doesn't define are ignored (D22).
    classify_ok(json!({"model": "sentiment", "input": "a", "encoding_format": "float", "dimensions": 3})).await;
}

#[tokio::test(flavor = "multi_thread")]
async fn vllm_pooling_fields_are_honored_or_rejected() {
    let ok = |extra: Value| {
        let mut body = json!({"model": "sentiment", "input": "a"});
        body.as_object_mut().unwrap().extend(extra.as_object().unwrap().clone());
        classify_ok(body)
    };
    let bad = |extra: Value, needle: &'static str| {
        let mut body = json!({"model": "sentiment", "input": "a"});
        body.as_object_mut().unwrap().extend(extra.as_object().unwrap().clone());
        classify_400(body, needle)
    };

    // request_id becomes the response id, as in vLLM; X-Request-Id wins.
    let body = ok(json!({"request_id": "abc-1"})).await;
    assert_eq!(body["id"], "classify-abc-1");
    let req = Request::post("/v1/classify")
        .header("content-type", "application/json")
        .header("x-request-id", "from-header")
        .body(Body::from(json!({"model": "sentiment", "input": "a", "request_id": "abc-1"}).to_string()))
        .unwrap();
    let (status, body) = call(test_state(true, None), req).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["id"], "classify-from-header");
    bad(json!({"request_id": 7}), "`request_id` must be a string").await;

    // The forms that change nothing are accepted.
    for harmless in [
        json!({"priority": 0}),
        json!({"priority": null}),
        json!({"padding": "do_not_pad"}),
        json!({"padding": null}),
        json!({"cache_salt": "c2FsdA-random"}),
        json!({"cache_salt": null}),
        json!({"mm_processor_kwargs": {}}),
        json!({"mm_processor_kwargs": null}),
        json!({"task": "classify"}),
        json!({"user": "someone"}),
    ] {
        ok(harmless).await;
    }
    // Everything else is a 400.
    bad(json!({"priority": 1}), "priority scheduling").await;
    bad(json!({"priority": -5}), "priority scheduling").await;
    bad(json!({"priority": "high"}), "must be an integer").await;
    bad(json!({"padding": "max_length"}), "`padding: max_length` isn't supported").await;
    bad(json!({"padding": "longest"}), "`max_length` or `do_not_pad`").await;
    for salt in [json!(""), json!("a/b"), json!("x".repeat(129)), json!(5)] {
        bad(json!({"cache_salt": salt}), "cache_salt").await;
    }
    bad(json!({"mm_processor_kwargs": {"size": 3}}), "text only").await;
    // vLLM rejects `normalize` in any form, and the removed pooling tasks.
    for v in [json!(true), json!(false), Value::Null] {
        bad(json!({"normalize": v}), "`normalize` was removed; use `use_activation` instead").await;
    }
    bad(json!({"task": "score"}), "`score` task was removed").await;
    bad(json!({"task": "encode"}), "`encode` task was removed").await;
}

#[tokio::test(flavor = "multi_thread")]
async fn truncation_reaches_the_classifier_as_vllm_defines_it() {
    let (state, seen) = test_state_with(true, None);
    let run = |body: Value| call(state.clone(), classify(body));

    let (status, body) = run(json!({"model": "sentiment", "input": "a good film", "truncate_prompt_tokens": 4})).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["usage"]["prompt_tokens"], 4);
    // -1 is the model's maximum.
    let (status, _) = run(json!({"model": "sentiment", "input": "x", "truncate_prompt_tokens": -1, "truncation_side": "left"})).await;
    assert_eq!(status, StatusCode::OK);
    let seen = seen.lock().unwrap().clone();
    assert_eq!(seen[0].truncate_prompt_tokens, Some(4));
    assert_eq!(seen[0].truncation_side, TruncationSide::Right);
    assert_eq!(seen[1].truncate_prompt_tokens, Some(64));
    assert_eq!(seen[1].truncation_side, TruncationSide::Left);

    for n in [0, -2, 65] {
        classify_400(json!({"model": "sentiment", "input": "a", "truncate_prompt_tokens": n}), "truncate_prompt_tokens").await;
    }
    classify_400(json!({"model": "sentiment", "input": "a", "truncation_side": "middle"}), "truncation_side").await;
}

#[tokio::test(flavor = "multi_thread")]
async fn extensions_a_fixed_label_model_does_not_take_are_400s() {
    for (field, value) in [
        ("candidate_labels", json!(["a", "b"])),
        ("question_type", json!("choice")),
        ("instructions", json!("x")),
    ] {
        let mut body = json!({"model": "sentiment", "input": "a"});
        body[field] = value;
        classify_400(body, &format!("`{field}` isn't supported by model `sentiment` (a text-classification model)")).await;
    }
    // Calibration is an extension of models that declare temperatures,
    // and /v1/models lists it for exactly those: any value is a 400 here.
    for c in ["model", "none", "platt"] {
        classify_400(
            json!({"model": "sentiment", "input": "a", "calibration": c}),
            "`calibration` isn't supported by model `sentiment`",
        )
        .await;
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn zero_shot_follows_the_request_labels() {
    let body = classify_ok(json!({
        "model": "decider",
        "input": ["it is about sports", "cooking tonight"],
        "candidate_labels": ["politics", "sports", "cooking"],
        "question_type": "choice",
        "instructions": "What is the topic?",
    }))
    .await;
    let data = body["data"].as_array().unwrap();
    assert_eq!(data[0]["label"], "sports");
    assert_eq!(data[0]["num_classes"], 3);
    assert_eq!(probs(&data[0]), activate(ProblemType::SingleLabel, &[0.0, 3.0, 0.0], None));
    assert_eq!(data[1]["label"], "cooking");
}

#[tokio::test(flavor = "multi_thread")]
async fn calibration_applies_the_manifest_temperature_for_the_question() {
    let request = |calibration: &str| {
        json!({
            "model": "decider", "input": "sports", "candidate_labels": ["politics", "sports", "cooking"],
            "question_type": "choice", "calibration": calibration,
        })
    };
    let raw = classify_ok(request("none")).await;
    let calibrated = classify_ok(request("model")).await;
    // "choice:3-5" = 2.0 flattens the distribution.
    assert_eq!(probs(&calibrated["data"][0]), activate(ProblemType::SingleLabel, &[0.0, 3.0, 0.0], Some(2.0)));
    assert!(probs(&calibrated["data"][0])[1] < probs(&raw["data"][0])[1]);
    // No temperature for 2-option choice questions.
    classify_400(
        json!({"model": "decider", "input": "x", "candidate_labels": ["a", "b"], "question_type": "choice", "calibration": "model"}),
        "no calibration temperature for choice questions with 2 labels",
    )
    .await;
    classify_ok(json!({
        "model": "decider", "input": "x", "candidate_labels": ["false", "true: it holds"],
        "question_type": "noul", "calibration": "model",
    }))
    .await;
    // Raw logits need no temperature, so none is missing.
    let raw = classify_ok(json!({
        "model": "decider", "input": "b", "candidate_labels": ["a", "b"], "question_type": "choice",
        "calibration": "model", "use_activation": false,
    }))
    .await;
    assert_eq!(probs(&raw["data"][0]), vec![0.0, 3.0]);
    classify_400(json!({"model": "decider", "input": "x", "candidate_labels": ["a", "b"],
                        "question_type": "choice", "calibration": "platt"}), "unsupported calibration").await;
}

#[tokio::test(flavor = "multi_thread")]
async fn a_request_is_checked_in_full_before_anything_runs() {
    // Manifest-level errors come before the model loads.
    let (state, seen) = test_state_with(true, None);
    let (status, _) = call(
        state.clone(),
        classify(json!({"model": "decider", "input": "x", "candidate_labels": ["a", "b"],
                        "question_type": "choice", "calibration": "model"})),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    let (status, _) = call(
        state.clone(),
        classify(json!({"model": "decider", "input": "x", "candidate_labels": ["true", "false"], "question_type": "noul"})),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    let (_, health) = call(state.clone(), Request::get("/health").body(Body::empty()).unwrap()).await;
    assert_eq!(health["classifiers"]["resident"], 0, "nothing loaded for requests the manifest rejects");
    assert!(seen.lock().unwrap().is_empty());

    // An input that fails `prepare` stops the batch before any input runs.
    let (state, _, runs) = test_state_probe(true, None);
    let (status, body) = call(state, classify(json!({"model": "sentiment", "input": ["good", "reject me", "bad"]}))).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    assert!(body["error"]["message"].as_str().unwrap().contains("input 1: rejected by prepare"), "{body}");
    assert_eq!(runs.load(std::sync::atomic::Ordering::SeqCst), 0, "no input ran");
}

#[tokio::test(flavor = "multi_thread")]
async fn zero_shot_requests_are_validated_against_the_format() {
    let base = || json!({"model": "decider", "input": "x", "candidate_labels": ["a", "b"], "question_type": "choice"});
    let with = |field: &str, value: Value| {
        let mut b = base();
        if value.is_null() {
            b.as_object_mut().unwrap().remove(field);
        } else {
            b[field] = value;
        }
        b
    };
    classify_ok(base()).await;
    classify_400(with("candidate_labels", Value::Null), "needs `candidate_labels`").await;
    classify_400(with("question_type", Value::Null), "needs `question_type`").await;
    classify_400(with("question_type", json!("rank")), "unsupported question_type `rank`").await;
    classify_400(with("candidate_labels", json!(["a"])), "at least 2").await;
    classify_400(with("candidate_labels", json!(["a", "b", "c", "d", "e"])), "maximum of 4").await;
    classify_400(with("candidate_labels", json!(["a", "b", "a"])), "duplicate candidate label `a`").await;
    classify_400(with("truncate_prompt_tokens", json!(8)), "laya format").await;
    classify_400(with("truncation_side", json!("left")), "laya format").await;
    classify_ok(with("truncation_side", json!("right"))).await;
    classify_400(with("input", json!(["a", "b", "c"])), "maximum of 2").await;
    // noul labels are validated by the format's own renderer.
    let noul = |labels: Value| json!({"model": "decider", "input": "x", "candidate_labels": labels, "question_type": "noul"});
    classify_400(noul(json!(["true", "false"])), "`false` then `true`").await;
    classify_ok(noul(json!(["false: no", "true"]))).await;
}

#[tokio::test(flavor = "multi_thread")]
async fn gliner2_multi_label_switches_probs_to_per_label_sigmoids() {
    let request = |multi: Option<bool>| {
        let mut b = json!({
            "model": "schema-decider",
            "input": "a refund for the damaged parcel",
            "candidate_labels": ["refund", "damaged", "cancel"],
            "instructions": "intent",
        });
        if let Some(m) = multi {
            b["multi_label"] = json!(m);
        }
        b
    };
    // The mock scores 3.0 for each label the input contains, 0.0 otherwise.
    let logits = [3.0, 3.0, 0.0];
    for multi in [None, Some(false)] {
        let body = classify_ok(request(multi)).await;
        let d = &body["data"][0];
        assert_eq!(probs(d), activate(ProblemType::SingleLabel, &logits, None), "{multi:?}");
        assert!((probs(d).iter().sum::<f32>() - 1.0).abs() < 1e-6);
    }
    let body = classify_ok(request(Some(true))).await;
    let d = &body["data"][0];
    assert_eq!(probs(d), activate(ProblemType::MultiLabel, &logits, None));
    // One label, the argmax, as in vLLM's response; clients threshold probs.
    assert_eq!(d["label"], "refund");
    assert_eq!(d["num_classes"], 3);
}

#[tokio::test(flavor = "multi_thread")]
async fn gliner2_requests_are_validated_against_the_format() {
    let base = || json!({"model": "schema-decider", "input": "x", "candidate_labels": ["a", "b"]});
    let with = |field: &str, value: Value| {
        let mut b = base();
        b[field] = value;
        b
    };
    classify_ok(base()).await;
    classify_ok(with("instructions", json!("intent: what the customer wants"))).await;
    classify_ok(with("multi_label", json!(true))).await;
    classify_400(with("question_type", json!("choice")), "gliner2 format").await;
    classify_400(with("calibration", json!("model")), "gliner2 format").await;
    classify_400(with("truncate_prompt_tokens", json!(8)), "gliner2 format").await;
    classify_400(with("truncation_side", json!("left")), "gliner2 format").await;
    classify_400(with("candidate_labels", json!(["a", "b", "c", "d", "e"])), "maximum of 4").await;
    // multi_label is the gliner2 format's alone.
    for model in [json!({"model": "sentiment", "input": "x"}),
                  json!({"model": "decider", "input": "x", "candidate_labels": ["a", "b"], "question_type": "choice"})] {
        let mut b = model.clone();
        b["multi_label"] = json!(true);
        classify_400(b, "`multi_label` isn't supported").await;
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn julia_rendering_is_checked_before_the_model_loads() {
    let p = test_state_full(true, None);
    let julia = |q: &str, labels: Value, instructions: Option<&str>| {
        let mut b = json!({"model": "julia", "input": "x", "candidate_labels": labels, "question_type": q});
        if let Some(i) = instructions {
            b["instructions"] = json!(i);
        }
        b
    };
    let cases = [
        (julia("choice", json!(["a", "b"]), None), "`instructions` is required"),
        (julia("noul", json!(["false: wet", "true"]), Some("Is it dry?")), "both `false` and `true`, or neither"),
        (julia("choice", json!(["b", "b: "]), Some("Which?")), "render as the same option `b`"),
        (julia("choice", json!(["a", ": "]), Some("Which?")), "renders as an empty option"),
    ];
    for (body, needle) in cases {
        let (status, value) = call(p.state.clone(), classify(body.clone())).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{body}: {value}");
        let message = value["error"]["message"].as_str().unwrap();
        assert!(message.contains(needle), "{body}: {message}");
    }
    assert_eq!(p.runs.load(std::sync::atomic::Ordering::Relaxed), 0, "nothing ran");

    let (status, value) = call(p.state.clone(), classify(julia("noul", json!(["false", "true"]), Some("Is it dry?")))).await;
    assert_eq!(status, StatusCode::OK, "{value}");
    assert_eq!(probs(&value["data"][0]).len(), 2);
}

#[tokio::test(flavor = "multi_thread")]
async fn models_are_sent_to_the_route_for_their_task() {
    classify_400(json!({"model": "test-static", "input": "a"}), "model `test-static` is a feature-extraction model").await;
    classify_400(json!({"model": "apple-fm", "input": "a"}), "text-generation").await;
    let (status, body) = call(
        test_state(true, None),
        post_json("/v1/embeddings", json!({"model": "decider", "input": "a"})),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    let message = body["error"]["message"].as_str().unwrap();
    assert!(message.contains("zero-shot-classification") && message.contains("/v1/classify"), "{message}");
    let (status, body) = call(
        test_state(true, None),
        post_json("/v1/chat/completions", json!({"model": "sentiment", "messages": [{"role": "user", "content": "hi"}]})),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(body["error"]["message"].as_str().unwrap().contains("text-classification"));

    let (status, body) = call(test_state(true, None), classify(json!({"model": "nope", "input": "a"}))).await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    assert_eq!(body["error"]["code"], "model_not_found");
}

#[tokio::test(flavor = "multi_thread")]
async fn bad_model_output_and_failures_are_500s() {
    for input in ["nan here", "fail now"] {
        let (status, body) = call(test_state(true, None), classify(json!({"model": "sentiment", "input": input}))).await;
        assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR, "{input}: {body}");
        assert_eq!(body["error"]["type"], "server_error");
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn malformed_json_is_an_api_error_400_on_every_route() {
    for route in ["/v1/classify", "/v1/embeddings", "/v1/chat/completions"] {
        let cases = [
            ("application/json", "{not json"),
            // Well-formed, but a required field is missing (axum's 422).
            ("application/json", "{}"),
            // Wrong type.
            ("application/json", r#"{"model": 7}"#),
            // No JSON content type (axum's 415).
            ("text/plain", r#"{"model": "x"}"#),
        ];
        for (content_type, body) in cases {
            let req = Request::post(route)
                .header("content-type", content_type)
                .body(Body::from(body))
                .unwrap();
            let (status, value) = call(test_state(true, None), req).await;
            assert_eq!(status, StatusCode::BAD_REQUEST, "{route} {body}: {value}");
            assert_eq!(value["error"]["type"], "invalid_request_error", "{route} {body}: {value}");
            assert!(value["error"]["message"].as_str().unwrap().starts_with("Invalid JSON body"), "{value}");
        }
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn listings_are_task_aware() {
    let (status, body) = call(test_state(true, None), Request::get("/v1/models").body(Body::empty()).unwrap()).await;
    assert_eq!(status, StatusCode::OK);
    let model = |id: &str| -> Value {
        body["data"].as_array().unwrap().iter().find(|m| m["id"] == id).cloned().unwrap_or_else(|| panic!("{id} not listed"))
    };
    assert_eq!(model("apple-fm")["task"], "text-generation");
    let e = model("test-static");
    assert_eq!(e["task"], "feature-extraction");
    assert!(e.get("labels").is_none() && e.get("max_batch").is_none());

    let s = model("sentiment");
    assert_eq!(s["task"], "text-classification");
    assert_eq!(s["labels"], json!(["1 star", "2 stars", "3 stars", "4 stars", "5 stars"]));
    assert_eq!(s["max_batch"], 4);
    assert_eq!(s["extensions"], json!([]));
    assert_eq!(s["required"], json!([]));
    assert!(s.get("max_labels").is_none() && s.get("calibration").is_none());

    let d = model("decider");
    assert_eq!(d["task"], "zero-shot-classification");
    assert_eq!(d["max_labels"], 4);
    assert_eq!(d["extensions"], json!(["candidate_labels", "calibration", "question_type", "instructions"]));
    assert_eq!(d["required"], json!(["candidate_labels", "question_type"]));
    // No default instructions: every request sends them.
    assert_eq!(model("julia")["required"], json!(["candidate_labels", "question_type", "instructions"]));
    assert_eq!(model("reranker")["required"], json!([]));
    assert_eq!(d["calibration"], json!({"choice:3-5": 2.0, "noul:2": 0.5}));
    assert!(d.get("labels").is_none());

    let g = model("schema-decider");
    assert_eq!(g["task"], "zero-shot-classification");
    assert_eq!(g["extensions"], json!(["candidate_labels", "instructions", "multi_label"]));
    assert_eq!(g["required"], json!(["candidate_labels"]));
    assert!(g.get("calibration").is_none());

    let (_, health) = call(test_state(true, None), Request::get("/health").body(Body::empty()).unwrap()).await;
    assert_eq!(
        health["classifiers"]["models"],
        json!(["decider", "julia", "reranker", "schema-decider", "sentiment", "sigmoid-reranker"])
    );
    assert_eq!(health["classifiers"]["resident"], 0);
    assert_eq!(health["embeddings"]["models"], json!(["test-static"]));
    let skipped = health["skipped_models"].as_array().unwrap();
    assert_eq!(skipped.len(), 1, "{skipped:?}");
    // Relative to the models directory: /health is unauthenticated.
    assert_eq!(skipped[0]["path"], "broken/classifier.toml");
}

#[tokio::test(flavor = "multi_thread")]
async fn resident_classifiers_are_counted() {
    let state = test_state(true, None);
    let (status, _) = call(state.clone(), classify(json!({"model": "sentiment", "input": "a"}))).await;
    assert_eq!(status, StatusCode::OK);
    let (_, health) = call(state, Request::get("/health").body(Body::empty()).unwrap()).await;
    assert_eq!(health["classifiers"]["resident"], 1);
    assert_eq!(health["embeddings"]["resident"], 0);
}

#[tokio::test(flavor = "multi_thread")]
async fn embeddings_and_chat_carry_provenance_headers() {
    let (status, headers, _) = call_with_headers(
        test_state(true, None),
        post_json("/v1/embeddings", json!({"model": "test-static", "input": "hello"})),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(headers["sidekick-version"], env!("CARGO_PKG_VERSION"));
    assert_eq!(headers["sidekick-model"], "test-static");
    assert_eq!(headers["sidekick-compute-units"], "cpu");

    for stream in [false, true] {
        let state = test_state(true, None);
        state.refresh_chat_model().await;
        let (status, headers, _) = call_with_headers(
            state,
            post_json(
                "/v1/chat/completions",
                json!({"model": "apple-fm", "stream": stream, "messages": [{"role": "user", "content": "hi"}]}),
            ),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(headers["sidekick-model"], "core3", "the Foundation Models variant id");
        assert_eq!(headers["sidekick-version"], env!("CARGO_PKG_VERSION"));
        assert!(headers.get("sidekick-compute-units").is_none());
    }
    // An error carries no provenance.
    let (_, headers, _) = call_with_headers(
        test_state(false, None),
        post_json("/v1/chat/completions", json!({"model": "apple-fm", "messages": [{"role": "user", "content": "hi"}]})),
    )
    .await;
    assert!(headers.get("sidekick-model").is_none(), "no provenance on an error");
}

#[tokio::test(flavor = "multi_thread")]
async fn a_slow_model_info_never_delays_chat() {
    // The shim takes a minute to report its variant: requests don't wait,
    // name the chat id until the variant is known, and start one refresh.
    let mut state = test_state(true, None);
    state.chat = std::sync::Arc::new(MockChat { available: true, info_delay: Duration::from_secs(60) });
    for stream in [false, true] {
        let request = post_json(
            "/v1/chat/completions",
            json!({"model": "apple-fm", "stream": stream, "messages": [{"role": "user", "content": "hi"}]}),
        );
        let (status, headers, _) = tokio::time::timeout(Duration::from_secs(5), call_with_headers(state.clone(), request))
            .await
            .expect("chat waited on model_info");
        assert_eq!(status, StatusCode::OK);
        assert_eq!(headers["sidekick-model"], "apple-fm");
    }
    // Once known, the variant is what the header names.
    state.chat = std::sync::Arc::new(MockChat { available: true, info_delay: Duration::ZERO });
    state.refresh_chat_model().await;
    assert_eq!(state.chat_model_id(), "core3");
}

#[tokio::test(flavor = "multi_thread")]
async fn builds_without_core_ml_hide_classifiers() {
    let mut state = test_state(true, None);
    state.classifiers_supported = false;
    let (_, models) = call(state.clone(), Request::get("/v1/models").body(Body::empty()).unwrap()).await;
    let ids: Vec<&str> = models["data"].as_array().unwrap().iter().map(|m| m["id"].as_str().unwrap()).collect();
    assert_eq!(ids, vec!["apple-fm", "test-static"]);
    let (status, body) = call(state.clone(), classify(json!({"model": "sentiment", "input": "a"}))).await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE, "{body}");
    let (_, health) = call(state, Request::get("/health").body(Body::empty()).unwrap()).await;
    assert_eq!(health["classifiers"]["supported"], false);
}
