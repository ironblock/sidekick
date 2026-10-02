//! `CoremlClassifier` on the fev format with real Core ML: the tiny causal
//! pointer model in tests/fixtures/tiny-fev (built by
//! tools/make_classifier_test_models.py), fed `marker_pos` and `decide_pos`.
//! Predictions match the fp32 torch model in every bucket, pads change
//! nothing, and a position outside the real tokens is refused.
//!
//! macOS with `--features coreml` only; runs on the CPU, so it needs no ANE.
#![cfg(all(target_os = "macos", feature = "coreml"))]

use serde::Deserialize;
use sidekick_core::manifest::ModelRegistry;
use sidekick_core::{Classifier, ClassifyParams, Prepared, QuestionType};
use sidekick_coreml::ComputeUnits;
use sidekick_embed::CoremlClassifier;
use std::path::{Path, PathBuf};

const MANIFEST: &str = r#"
id = "tiny-fev"
task = "zero-shot-classification"
artifact = "model_{seq}.mlmodelc"
tokenizer = "tokenizer.json"
buckets = [16, 32]
max_seq_len = 32
max_batch = 4

[classify]
format = "fev"
max_labels = 4

[classify.fev]
state_max_len = 12
delimiters = { state = "<|r0|>", question = "<|r1|>", option = "<|r2|>", option_end = "<|r3|>", decide = "<|r4|>" }

[classify.io]
input_ids = "input_ids"
attention_mask = "attention_mask"
marker_pos = "marker_pos"
decide_pos = "decide_pos"
output = "logits"
"#;

/// A word-level tokenizer whose ids 0-4 are the delimiters the fixture's
/// rows use, within the model's 32-entry vocabulary.
fn tokenizer() -> String {
    let mut vocab = serde_json::Map::new();
    let delimiters = ["<|r0|>", "<|r1|>", "<|r2|>", "<|r3|>", "<|r4|>"];
    let words = ["[UNK]", "no", "yes", "a", "b", "c", "d", "e", "f", "<", ">", "¦"];
    for (i, w) in delimiters.iter().chain(&words).enumerate() {
        vocab.insert(w.to_string(), i.into());
    }
    let added: Vec<_> = delimiters
        .iter()
        .enumerate()
        .map(|(i, d)| serde_json::json!({"id": i, "content": d, "single_word": false, "lstrip": false,
                                        "rstrip": false, "normalized": false, "special": true}))
        .collect();
    serde_json::json!({
        "version": "1.0", "truncation": null, "padding": null, "added_tokens": added,
        "normalizer": null, "pre_tokenizer": {"type": "Whitespace"}, "post_processor": null,
        "decoder": null, "model": {"type": "WordLevel", "vocab": vocab, "unk_token": "[UNK]"}
    })
    .to_string()
}

fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny-fev")
}

fn model_dir(tag: &str, manifest: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("sk-tiny-fev-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let model = dir.join("tiny-fev");
    std::fs::create_dir_all(&model).unwrap();
    std::fs::write(model.join("classifier.toml"), manifest).unwrap();
    std::fs::write(model.join("tokenizer.json"), tokenizer()).unwrap();
    for bucket in [16, 32] {
        let name = format!("model_{bucket}.mlmodelc");
        std::os::unix::fs::symlink(fixtures().join(&name), model.join(&name)).unwrap();
    }
    dir
}

fn load(dir: &Path) -> sidekick_core::Result<CoremlClassifier> {
    let registry = ModelRegistry::scan(dir).unwrap();
    assert!(registry.skipped().is_empty(), "{:?}", registry.skipped().first().map(|s| &s.reason));
    CoremlClassifier::load_with(registry.classifier("tiny-fev").unwrap(), ComputeUnits::CpuOnly)
}

#[derive(Deserialize)]
struct Expected {
    cases: Vec<ExpectedCase>,
}

#[derive(Deserialize)]
struct ExpectedCase {
    ids: Vec<i32>,
    markers: Vec<i32>,
    decide: i32,
    logits: Vec<f32>,
}

fn prepared(case: &ExpectedCase, bucket: usize) -> Prepared {
    Prepared {
        ids: case.ids.clone(),
        type_ids: vec![],
        markers: case.markers.clone(),
        qtype: None,
        decide_pos: Some(case.decide), seg: vec![], position_ids: vec![],
        bucket,
    }
}

#[test]
fn fev_predictions_match_torch_in_every_bucket_and_ignore_pads() {
    let dir = model_dir("ok", MANIFEST);
    let clf = load(&dir).unwrap();
    let expected: Expected =
        serde_json::from_str(&std::fs::read_to_string(fixtures().join("expected.json")).unwrap()).unwrap();
    for case in &expected.cases {
        let bucket = if case.ids.len() <= 16 { 16 } else { 32 };
        let p = prepared(case, bucket);
        // Its own bucket, the larger one, and random pad ids: the model is
        // causal, so pads after the row never reach a real token.
        let mut runs = vec![clf.run(&p).unwrap(), clf.run_in(&p, 32, &[]).unwrap()];
        runs.push(clf.run_in(&p, bucket, &[7; 32]).unwrap());
        for logits in runs {
            assert_eq!(logits.len(), case.markers.len());
            for (got, want) in logits.iter().zip(&case.logits) {
                // fp16 compute: tolerate its rounding, not a wrong slot.
                assert!((got - want).abs() < 2e-2, "{logits:?} vs {:?}", case.logits);
            }
        }
    }

    // The product path end to end: prepare with the tokenizer, then run.
    let params = ClassifyParams {
        candidate_labels: vec!["false".into(), "true: c d".into()],
        question_type: Some(QuestionType::Noul),
        instructions: Some("a b".into()),
        ..Default::default()
    };
    let logits = clf.classify("e f e f", &params).unwrap();
    assert_eq!(logits.len(), 2);
    assert!(logits.iter().all(|x| x.is_finite()));
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn fev_positions_outside_the_real_tokens_are_refused() {
    let dir = model_dir("positions", MANIFEST);
    let clf = load(&dir).unwrap();
    let expected: Expected =
        serde_json::from_str(&std::fs::read_to_string(fixtures().join("expected.json")).unwrap()).unwrap();
    let case = &expected.cases[0];
    let used = case.ids.len() as i32;
    for (markers, decide) in [
        (case.markers.clone(), used),     // decide in the padding
        (case.markers.clone(), -1),       // negative
        (vec![case.markers[0], used + 2], case.decide), // an option end in the padding
    ] {
        let p = Prepared { markers, decide_pos: Some(decide), ..prepared(case, 16) };
        let e = clf.run(&p).unwrap_err();
        assert!(e.to_string().contains("real tokens"), "{e}");
    }
    let p = Prepared { decide_pos: None, ..prepared(case, 16) };
    assert!(clf.run(&p).unwrap_err().to_string().contains("decide position"));
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn fev_decide_pos_must_be_an_input_of_the_artifact() {
    let dir = model_dir("bad-io", &MANIFEST.replace("decide_pos = \"decide_pos\"", "decide_pos = \"decide\""));
    let e = load(&dir).err().expect("the load fails");
    assert!(e.to_string().contains("no int32 multi-array input `decide`"), "{e}");
    std::fs::remove_dir_all(&dir).unwrap();
}
