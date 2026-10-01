//! `CoremlClassifier` on real Core ML, with the tiny artifacts in
//! tests/fixtures (built by tools/make_classifier_test_models.py): a
//! laya-format model, a pair-input reranker and a gliner2-format model with
//! per-token logits. Every bucket's interface is
//! checked when the classifier loads, and predictions match the fp32 torch
//! model they came from.
//!
//! macOS with `--features coreml` only; runs on the CPU, so it needs no ANE.
#![cfg(all(target_os = "macos", feature = "coreml"))]

use serde::Deserialize;
use sidekick_core::manifest::ModelRegistry;
use sidekick_core::{Classifier, ClassifyParams, PairParams, Prepared, QuestionType};
use sidekick_coreml::ComputeUnits;
use sidekick_embed::CoremlClassifier;
use std::path::{Path, PathBuf};

fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny-laya")
}

fn reranker_fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny-reranker")
}

const MANIFEST: &str = r#"
id = "tiny-laya"
task = "zero-shot-classification"
artifact = "model_{seq}.mlmodelc"
tokenizer = "tokenizer.json"
buckets = [16, 32]
max_seq_len = 32
max_batch = 4

[classify]
format = "laya"
max_labels = 4

[classify.laya]
head_max_len = 12
default_instructions = { choice = "which", score = "which", noul = "which" }

[classify.io]
input_ids = "input_ids"
attention_mask = "attention_mask"
marker_pos = "marker_pos"
qtype = "qtype"
output = "logits"
"#;

/// A word-level tokenizer over the fixture model's 32-entry vocabulary.
fn tokenizer() -> String {
    let mut vocab = serde_json::Map::new();
    let words = ["[UNK]", "[CLS]", "[SEP]", "[MASK]", "[PAD]", "choice", "score", "noul", "question", ":",
                 "which", "a", "b", "c", "d", "e", "f", "g", "h"];
    for (i, w) in words.iter().enumerate() {
        vocab.insert(w.to_string(), i.into());
    }
    let special = |id: u32, content: &str| {
        serde_json::json!({"id": id, "content": content, "single_word": false, "lstrip": false,
                           "rstrip": false, "normalized": false, "special": true})
    };
    serde_json::json!({
        "version": "1.0", "truncation": null, "padding": null,
        "added_tokens": [special(0, "[UNK]"), special(1, "[CLS]"), special(2, "[SEP]"), special(3, "[MASK]"), special(4, "[PAD]")],
        "normalizer": {"type": "Lowercase"},
        "pre_tokenizer": {"type": "Whitespace"},
        "post_processor": {
            "type": "TemplateProcessing",
            "single": [{"SpecialToken": {"id": "[CLS]", "type_id": 0}}, {"Sequence": {"id": "A", "type_id": 0}}, {"SpecialToken": {"id": "[SEP]", "type_id": 0}}],
            "pair": [{"SpecialToken": {"id": "[CLS]", "type_id": 0}}, {"Sequence": {"id": "A", "type_id": 0}}, {"SpecialToken": {"id": "[SEP]", "type_id": 0}}, {"Sequence": {"id": "B", "type_id": 1}}, {"SpecialToken": {"id": "[SEP]", "type_id": 1}}],
            "special_tokens": {
                "[CLS]": {"id": "[CLS]", "ids": [1], "tokens": ["[CLS]"]},
                "[SEP]": {"id": "[SEP]", "ids": [2], "tokens": ["[SEP]"]}
            }
        },
        "decoder": null,
        "model": {"type": "WordLevel", "vocab": vocab, "unk_token": "[UNK]"}
    })
    .to_string()
}

/// A model directory whose buckets link to the named fixture artifacts
/// (`None`: that bucket's artifact is missing).
fn model_dir(tag: &str, manifest: &str, bucket_16: Option<&str>, bucket_32: Option<&str>) -> PathBuf {
    model_dir_in(&fixtures(), tag, manifest, bucket_16, bucket_32)
}

fn model_dir_in(
    fixtures: &Path,
    tag: &str,
    manifest: &str,
    bucket_16: Option<&str>,
    bucket_32: Option<&str>,
) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("sk-tiny-laya-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let model = dir.join("tiny-laya");
    std::fs::create_dir_all(&model).unwrap();
    std::fs::write(model.join("classifier.toml"), manifest).unwrap();
    std::fs::write(model.join("tokenizer.json"), tokenizer()).unwrap();
    for (bucket, artifact) in [(16, bucket_16), (32, bucket_32)] {
        if let Some(a) = artifact {
            std::os::unix::fs::symlink(fixtures.join(a), model.join(format!("model_{bucket}.mlmodelc"))).unwrap();
        }
    }
    dir
}

fn load(dir: &Path) -> sidekick_core::Result<CoremlClassifier> {
    load_id(dir, "tiny-laya")
}

fn load_id(dir: &Path, id: &str) -> sidekick_core::Result<CoremlClassifier> {
    let registry = ModelRegistry::scan(dir).unwrap();
    assert!(registry.skipped().is_empty(), "{:?}", registry.skipped());
    CoremlClassifier::load_with(registry.classifier(id).unwrap(), ComputeUnits::CpuOnly)
}

#[derive(Deserialize)]
struct Expected {
    cases: Vec<ExpectedCase>,
}

#[derive(Deserialize)]
struct ExpectedCase {
    ids: Vec<i32>,
    markers: Vec<i32>,
    qtype: i32,
    logits: Vec<f32>,
}

#[test]
fn predictions_match_torch_in_every_bucket() {
    let dir = model_dir("ok", MANIFEST, Some("model_16.mlmodelc"), Some("model_32.mlmodelc"));
    let clf = load(&dir).unwrap();
    let expected: Expected =
        serde_json::from_str(&std::fs::read_to_string(fixtures().join("expected.json")).unwrap()).unwrap();
    for case in &expected.cases {
        let bucket = if case.ids.len() <= 16 { 16 } else { 32 };
        let prepared = Prepared {
            ids: case.ids.clone(),
            type_ids: vec![],
            markers: case.markers.clone(),
            qtype: Some(case.qtype),
            bucket,
        };
        // Its own bucket, the larger one, and random pad ids: all the same.
        let mut runs = vec![clf.run(&prepared).unwrap(), clf.run_in(&prepared, 32, &[]).unwrap()];
        runs.push(clf.run_in(&prepared, bucket, &[7; 32]).unwrap());
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
        candidate_labels: vec!["a".into(), "b c".into(), "d".into()],
        question_type: Some(QuestionType::Choice),
        ..Default::default()
    };
    let logits = clf.classify("e f g h", &params).unwrap();
    assert_eq!(logits.len(), 3);
    assert!(logits.iter().all(|x| x.is_finite()));
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn the_manifest_compute_units_reach_core_ml() {
    // Absent, the default; otherwise each value the manifest can name. The
    // units are read back from the loaded model's Core ML configuration.
    for (line, want) in [
        ("", ComputeUnits::CpuAndNeuralEngine),
        ("compute_units = \"cpu_and_gpu\"\n", ComputeUnits::CpuAndGpu),
        ("compute_units = \"cpu_only\"\n", ComputeUnits::CpuOnly),
        ("compute_units = \"all\"\n", ComputeUnits::All),
    ] {
        let manifest = MANIFEST.replace("max_batch = 4\n", &format!("max_batch = 4\n{line}"));
        let dir = model_dir(&format!("units-{}", want.name()), &manifest, Some("model_16.mlmodelc"), Some("model_32.mlmodelc"));
        let registry = ModelRegistry::scan(&dir).unwrap();
        let model = registry.classifier("tiny-laya").unwrap();
        assert_eq!(model.manifest.compute_units, want);
        let clf = CoremlClassifier::load(model).unwrap();
        assert_eq!(clf.compute_units().unwrap(), want, "{line:?}");
        // And it serves there.
        let params = ClassifyParams {
            candidate_labels: vec!["a".into(), "b".into()],
            question_type: Some(QuestionType::Choice),
            ..Default::default()
        };
        assert!(clf.classify("c d", &params).unwrap().iter().all(|x| x.is_finite()));
        std::fs::remove_dir_all(&dir).unwrap();
    }
    // An explicit preference overrides the manifest's.
    let manifest = MANIFEST.replace("max_batch = 4\n", "max_batch = 4\ncompute_units = \"cpu_and_gpu\"\n");
    let dir = model_dir("units-override", &manifest, Some("model_16.mlmodelc"), Some("model_32.mlmodelc"));
    assert_eq!(load(&dir).unwrap().compute_units().unwrap(), ComputeUnits::CpuOnly);
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn every_bucket_is_checked_at_load() {
    // Before the check covered every bucket, each of these loaded: only the
    // smallest bucket was loaded eagerly, and the larger one failed on the
    // first request long enough to need it.
    struct Case {
        tag: &'static str,
        manifest: String,
        bucket_16: Option<&'static str>,
        bucket_32: Option<&'static str>,
        want: &'static [&'static str],
    }
    let case = |tag, manifest: &str, bucket_16, bucket_32, want| Case {
        tag,
        manifest: manifest.to_string(),
        bucket_16,
        bucket_32,
        want,
    };
    let cases = [
        case("k5", MANIFEST, Some("model_16.mlmodelc"), Some("k5_32.mlmodelc"),
             &["model_32.mlmodelc", "`marker_pos` is [1, 5], expected [1, 4]"]),
        case("wrong-bucket", MANIFEST, Some("model_16.mlmodelc"), Some("model_16.mlmodelc"),
             &["model_32.mlmodelc", "`input_ids` is [1, 16], expected [1, 32]"]),
        case("missing", MANIFEST, Some("model_16.mlmodelc"), None,
             &["model_32.mlmodelc", "can't read the artifact"]),
        // The manifest disagrees with every bucket: the first one is named.
        case("max-labels", &MANIFEST.replace("max_labels = 4", "max_labels = 5"),
             Some("model_16.mlmodelc"), Some("model_32.mlmodelc"),
             &["model_16.mlmodelc", "expected [1, 5]"]),
    ];
    for c in cases {
        let dir = model_dir(c.tag, &c.manifest, c.bucket_16, c.bucket_32);
        let err = match load(&dir) {
            Ok(_) => panic!("{}: loaded", c.tag),
            Err(e) => e.to_string(),
        };
        for w in c.want {
            assert!(err.contains(w), "{}: `{err}` lacks `{w}`", c.tag);
        }
        // Core ML's own load error names the full path; sidekick's don't.
        assert!(!err.contains(dir.to_str().unwrap()) || c.tag == "missing", "{}: {err}", c.tag);
        std::fs::remove_dir_all(&dir).unwrap();
    }
}

const RERANKER: &str = r#"
id = "tiny-reranker"
task = "text-ranking"
artifact = "model_{seq}.mlmodelc"
tokenizer = "tokenizer.json"
buckets = [16, 32]
max_seq_len = 32
problem_type = "regression"

[classify]
labels = ["score"]

[classify.io]
input_ids = "input_ids"
attention_mask = "attention_mask"
token_type_ids = "token_type_ids"
output = "logits"
"#;

#[derive(Deserialize)]
struct ExpectedPairs {
    cases: Vec<ExpectedPair>,
}

#[derive(Deserialize)]
struct ExpectedPair {
    ids: Vec<i32>,
    type_ids: Vec<i32>,
    score: f32,
}

#[test]
fn a_reranker_feeds_segment_ids_and_matches_torch() {
    let dir = model_dir_in(&reranker_fixtures(), "rerank", RERANKER, Some("model_16.mlmodelc"), Some("model_32.mlmodelc"));
    let clf = load_id(&dir, "tiny-reranker").unwrap();
    let expected: ExpectedPairs =
        serde_json::from_str(&std::fs::read_to_string(reranker_fixtures().join("expected.json")).unwrap()).unwrap();
    for case in &expected.cases {
        let bucket = if case.ids.len() <= 16 { 16 } else { 32 };
        let prepared = Prepared { ids: case.ids.clone(), type_ids: case.type_ids.clone(), markers: vec![], qtype: None, bucket };
        for score in [clf.run(&prepared).unwrap(), clf.run_in(&prepared, 32, &[5; 32]).unwrap()] {
            assert_eq!(score.len(), 1);
            assert!((score[0] - case.score).abs() < 2e-2, "{score:?} vs {}", case.score);
        }
        // Segment ids matter: all-zero ones score differently.
        let flat = Prepared { type_ids: vec![], ..prepared.clone() };
        assert!((clf.run(&flat).unwrap()[0] - case.score).abs() > 1e-3);
    }
    // The product path: the tokenizer pairs the texts with segment ids.
    let p = clf.prepare_pair("a b", "c d e", &PairParams::default()).unwrap();
    assert_eq!(p.type_ids, vec![0, 0, 0, 0, 1, 1, 1, 1]);
    assert_eq!(clf.run(&p).unwrap().len(), 1);
    std::fs::remove_dir_all(&dir).unwrap();

    // An artifact input the manifest doesn't name fails the load.
    let unnamed = RERANKER.replace("token_type_ids = \"token_type_ids\"\n", "");
    let dir = model_dir_in(&reranker_fixtures(), "rerank-unnamed", &unnamed, Some("model_16.mlmodelc"), Some("model_32.mlmodelc"));
    let registry = ModelRegistry::scan(&dir).unwrap();
    let err = CoremlClassifier::load_with(registry.classifier("tiny-reranker").unwrap(), ComputeUnits::CpuOnly)
        .err()
        .unwrap()
        .to_string();
    // The artifact check runs before the tokenizer's (segment ids with no
    // input), which would refuse this model too.
    assert!(err.contains("takes input `token_type_ids`, which the manifest's [classify.io] doesn't name"), "{err}");
    std::fs::remove_dir_all(&dir).unwrap();
}

fn gliner2_fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny-gliner2")
}

const GLINER2: &str = r#"
id = "tiny-gliner2"
task = "zero-shot-classification"
artifact = "model_{seq}.mlmodelc"
tokenizer = "tokenizer.json"
buckets = [16, 32]
max_seq_len = 32
max_batch = 4

[classify]
format = "gliner2"
max_labels = 4

[classify.gliner2]
default_instructions = "label"

[classify.io]
input_ids = "input_ids"
attention_mask = "attention_mask"
output = "logits"
"#;

/// A word-level tokenizer with GLiNER2's markers, over ids the fixture
/// model's 32-entry vocabulary covers (the ids in its expected.json).
fn gliner2_tokenizer() -> String {
    let mut vocab = serde_json::Map::new();
    let words = ["[UNK]", "[P]", "[L]", "[SEP_TEXT]", "[DESCRIPTION]", "(", ")", ".", ":", "label",
                 "a", "b", "c", "d", "e", "f", "g", "h"];
    for (i, w) in words.iter().enumerate() {
        vocab.insert(w.to_string(), i.into());
    }
    let special = |id: u32, content: &str| {
        serde_json::json!({"id": id, "content": content, "single_word": false, "lstrip": false,
                           "rstrip": false, "normalized": false, "special": true})
    };
    serde_json::json!({
        "version": "1.0", "truncation": null, "padding": null,
        "added_tokens": [special(0, "[UNK]"), special(1, "[P]"), special(2, "[L]"), special(3, "[SEP_TEXT]"), special(4, "[DESCRIPTION]")],
        "normalizer": null,
        "pre_tokenizer": {"type": "Whitespace"},
        "post_processor": null,
        "decoder": null,
        "model": {"type": "WordLevel", "vocab": vocab, "unk_token": "[UNK]"}
    })
    .to_string()
}

fn gliner2_dir(tag: &str, bucket_16: &str, bucket_32: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("sk-tiny-gliner2-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let model = dir.join("tiny-gliner2");
    std::fs::create_dir_all(&model).unwrap();
    std::fs::write(model.join("classifier.toml"), GLINER2).unwrap();
    std::fs::write(model.join("tokenizer.json"), gliner2_tokenizer()).unwrap();
    for (bucket, artifact) in [(16, bucket_16), (32, bucket_32)] {
        std::os::unix::fs::symlink(gliner2_fixtures().join(artifact), model.join(format!("model_{bucket}.mlmodelc"))).unwrap();
    }
    dir
}

#[derive(Deserialize)]
struct ExpectedMarkers {
    cases: Vec<ExpectedMarkerCase>,
}

#[derive(Deserialize)]
struct ExpectedMarkerCase {
    ids: Vec<i32>,
    markers: Vec<i32>,
    logits: Vec<f32>,
}

#[test]
fn gliner2_reads_per_token_logits_at_the_markers_in_every_bucket() {
    let dir = gliner2_dir("ok", "model_16.mlmodelc", "model_32.mlmodelc");
    let clf = load_id(&dir, "tiny-gliner2").unwrap();
    let expected: ExpectedMarkers =
        serde_json::from_str(&std::fs::read_to_string(gliner2_fixtures().join("expected.json")).unwrap()).unwrap();
    for case in &expected.cases {
        let bucket = if case.ids.len() <= 16 { 16 } else { 32 };
        let prepared = Prepared { ids: case.ids.clone(), type_ids: vec![], markers: case.markers.clone(), qtype: None, bucket };
        // Its own bucket, the larger one, and random pad ids: all the same.
        let mut runs = vec![clf.run(&prepared).unwrap(), clf.run_in(&prepared, 32, &[9; 32]).unwrap()];
        runs.push(clf.run_in(&prepared, bucket, &[7; 32]).unwrap());
        for logits in runs {
            assert_eq!(logits.len(), case.markers.len());
            for (got, want) in logits.iter().zip(&case.logits) {
                assert!((got - want).abs() < 2e-2, "{logits:?} vs {:?}", case.logits);
            }
        }
        // A marker outside the real tokens would read a pad: refused.
        for bad in [case.ids.len() as i32, -1] {
            let mut p = prepared.clone();
            p.markers[0] = bad;
            let err = clf.run(&p).unwrap_err().to_string();
            assert!(err.contains("aren't all positions"), "{err}");
        }
    }

    // The product path: the input builder lays the schema out, and the
    // classifier reads its [L] positions.
    let params = ClassifyParams { candidate_labels: vec!["a".into(), "b: c d".into(), "e".into()], ..Default::default() };
    let prepared = clf.prepare("F G H", &params).unwrap();
    assert_eq!(prepared.markers.len(), 3);
    assert!(prepared.markers.iter().all(|&m| prepared.ids[m as usize] == 2), "{prepared:?}");
    let logits = clf.run(&prepared).unwrap();
    assert_eq!(logits.len(), 3);
    assert!(logits.iter().all(|x| x.is_finite()));
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn gliner2_output_must_be_one_logit_per_token() {
    let dir = gliner2_dir("short", "model_16.mlmodelc", "short_32.mlmodelc");
    let err = match load_id(&dir, "tiny-gliner2") {
        Ok(_) => panic!("loaded"),
        Err(e) => e.to_string(),
    };
    assert!(err.contains("model_32.mlmodelc") && err.contains("is [1, 16], expected [1, 32] (one logit per token)"), "{err}");
    std::fs::remove_dir_all(&dir).unwrap();
}
