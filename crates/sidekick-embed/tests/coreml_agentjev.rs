//! `CoremlClassifier` on the agentjev format with real Core ML: the tiny tree
//! model in tests/fixtures/tiny-agentjev (built by
//! tools/make_classifier_test_models.py with sidekick_convert's own tree
//! mask), fed `seg`, `position_ids` and `cand_end`. Predictions match the
//! fp32 torch model in every bucket, pads change nothing (no real token
//! attends them: the mask needs attention_mask 1 and a segment >= 0 on
//! every key), the graph reads seg and position_ids, and inputs that don't
//! describe the real tokens are refused.
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
id = "tiny-agentjev"
task = "zero-shot-classification"
artifact = "model_{seq}.mlmodelc"
tokenizer = "tokenizer.json"
buckets = [16, 32]
max_seq_len = 32
max_batch = 4

[classify]
format = "agentjev"
max_labels = 4

[classify.io]
input_ids = "input_ids"
attention_mask = "attention_mask"
seg = "seg"
position_ids = "position_ids"
marker_pos = "cand_end"
output = "logits"
"#;

/// A word-level tokenizer within the model's 32-entry vocabulary, with
/// AgentJev's tags as words.
fn tokenizer() -> String {
    let mut vocab = serde_json::Map::new();
    let words = ["[UNK]", "[STATE]", "[QUESTION]", "[CANDIDATE]", "a", "b", "c", "d", "e", "f", "FALSE", "TRUE"];
    for (i, w) in words.iter().enumerate() {
        vocab.insert(w.to_string(), i.into());
    }
    serde_json::json!({
        "version": "1.0", "truncation": null, "padding": null, "added_tokens": [],
        "normalizer": null, "pre_tokenizer": {"type": "Whitespace"}, "post_processor": null,
        "decoder": null, "model": {"type": "WordLevel", "vocab": vocab, "unk_token": "[UNK]"}
    })
    .to_string()
}

fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny-agentjev")
}

fn model_dir(tag: &str, manifest: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("sk-tiny-agentjev-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let model = dir.join("tiny-agentjev");
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
    CoremlClassifier::load_with(registry.classifier("tiny-agentjev").unwrap(), ComputeUnits::CpuOnly)
}

#[derive(Deserialize)]
struct Expected {
    cases: Vec<ExpectedCase>,
}

#[derive(Deserialize)]
struct ExpectedCase {
    ids: Vec<i32>,
    seg: Vec<i32>,
    position_ids: Vec<i32>,
    markers: Vec<i32>,
    logits: Vec<f32>,
}

fn expected() -> Expected {
    serde_json::from_str(&std::fs::read_to_string(fixtures().join("expected.json")).unwrap()).unwrap()
}

fn prepared(case: &ExpectedCase, bucket: usize) -> Prepared {
    Prepared {
        ids: case.ids.clone(),
        type_ids: vec![],
        markers: case.markers.clone(),
        qtype: None,
        decide_pos: None,
        seg: case.seg.clone(),
        position_ids: case.position_ids.clone(),
        bucket,
    }
}

#[test]
fn agentjev_predictions_match_torch_in_every_bucket_and_ignore_pads() {
    let dir = model_dir("ok", MANIFEST);
    let clf = load(&dir).unwrap();
    for case in &expected().cases {
        let bucket = if case.ids.len() <= 16 { 16 } else { 32 };
        let p = prepared(case, bucket);
        // Its own bucket, the larger one, and random pad ids, which no real
        // token attends.
        let pads: Vec<i32> = (0..32).map(|i| (i * 7 + 3) % 32).collect();
        let runs = [clf.run(&p).unwrap(), clf.run_in(&p, 32, &[]).unwrap(), clf.run_in(&p, bucket, &pads).unwrap()];
        for logits in runs {
            assert_eq!(logits.len(), case.markers.len());
            for (got, want) in logits.iter().zip(&case.logits) {
                // fp16 compute: tolerate its rounding, not a wrong slot.
                assert!((got - want).abs() < 2e-2, "{logits:?} vs {:?}", case.logits);
            }
        }
    }

    // The product path end to end: prepare the tree with the tokenizer, then run.
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
fn agentjev_reads_its_tree_inputs() {
    // The graph must use seg and position_ids: corrupting either moves the
    // logits. (Pads never reach a real token through either path: the mask
    // also requires attention_mask 1 on every key.)
    let dir = model_dir("tree", MANIFEST);
    let clf = load(&dir).unwrap();
    let cases = expected().cases;
    let case = &cases[1];
    let moved = |p: &Prepared| {
        let got = clf.run(p).unwrap();
        got.iter().zip(&case.logits).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max)
    };
    // Every branch merged into the prefix: siblings now see each other.
    let merged = Prepared { seg: vec![0; case.ids.len()], ..prepared(case, 16) };
    assert!(moved(&merged) > 0.1, "seg ignored: {}", moved(&merged));
    // Positions that run on through the branches instead of restarting.
    let running = Prepared { position_ids: (0..case.ids.len() as i32).collect(), ..prepared(case, 16) };
    assert!(moved(&running) > 0.1, "position_ids ignored: {}", moved(&running));
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn agentjev_inputs_outside_the_real_tokens_are_refused() {
    let dir = model_dir("positions", MANIFEST);
    let clf = load(&dir).unwrap();
    let cases = expected().cases;
    let case = &cases[0];
    let used = case.ids.len() as i32;
    // A candidate end in the padding, or negative.
    for markers in [vec![case.markers[0], used], vec![-1, case.markers[1]]] {
        let p = Prepared { markers, ..prepared(case, 16) };
        let e = clf.run(&p).unwrap_err();
        assert!(e.to_string().contains("real tokens"), "{e}");
    }
    // Segments or positions that don't cover exactly the real tokens.
    let short = |v: &Vec<i32>| v[..v.len() - 1].to_vec();
    for p in [
        Prepared { seg: short(&case.seg), ..prepared(case, 16) },
        Prepared { position_ids: short(&case.position_ids), ..prepared(case, 16) },
        Prepared { seg: vec![], ..prepared(case, 16) },
    ] {
        let e = clf.run(&p).unwrap_err();
        assert!(e.to_string().contains("segments"), "{e}");
    }
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn agentjev_named_inputs_must_exist_in_the_artifact() {
    for (tag, from, to, missing) in [
        ("bad-seg", "seg = \"seg\"", "seg = \"segments\"", "segments"),
        ("bad-pos", "position_ids = \"position_ids\"", "position_ids = \"positions\"", "positions"),
        ("bad-ends", "marker_pos = \"cand_end\"", "marker_pos = \"ends\"", "ends"),
    ] {
        let dir = model_dir(tag, &MANIFEST.replace(from, to));
        let e = load(&dir).err().expect("the load fails");
        assert!(e.to_string().contains(&format!("no int32 multi-array input `{missing}`")), "{tag}: {e}");
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
