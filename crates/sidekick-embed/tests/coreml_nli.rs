//! The nli format (D41) on real Core ML, with the tiny cross-encoders in
//! tests/fixtures/tiny-nli (built by tools/make_classifier_test_models.py):
//! a BERT-style 3-way model that takes segment ids and a RoBERTa-style
//! 2-way model that doesn't. Each label's pair is laid out as the
//! tokenizer pairs text, runs as its own prediction, and the activated
//! values are the scores Hugging Face's zero-shot pipeline computes from
//! the fp32 torch logits, with and without `multi_label`.
//!
//! macOS with `--features coreml` only; runs on the CPU, so it needs no ANE.
#![cfg(all(target_os = "macos", feature = "coreml"))]

use serde::Deserialize;
use sidekick_core::manifest::ModelRegistry;
use sidekick_core::{activate, Classifier, ClassifyParams, ProblemType};
use sidekick_coreml::ComputeUnits;
use sidekick_embed::CoremlClassifier;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny-nli")
}

const BERT: &str = r#"
id = "tiny-nli"
task = "zero-shot-classification"
artifact = "model_{seq}.mlmodelc"
tokenizer = "tokenizer.json"
buckets = [16, 32]
max_seq_len = 32
max_batch = 4

[classify]
format = "nli"
max_labels = 4

[classify.nli]
entailment = 2
contradiction = 0
n_classes = 3
hypothesis_template = "which {}"

[classify.io]
input_ids = "input_ids"
attention_mask = "attention_mask"
token_type_ids = "token_type_ids"
output = "logits"
"#;

/// RoBERTa-style: entailment vs not_entailment, no segment ids.
fn roberta_manifest() -> String {
    BERT.replace("entailment = 2\ncontradiction = 0\nn_classes = 3", "entailment = 0\ncontradiction = 1\nn_classes = 2")
        .replace("token_type_ids = \"token_type_ids\"\n", "")
}

/// A word-level tokenizer over the fixture's vocabulary: BERT's pair
/// template (`[CLS] A [SEP] B [SEP]`, segments 0 then 1), or RoBERTa's
/// (`<s> A </s></s> B </s>`).
fn tokenizer(roberta: bool) -> String {
    let (cls, sep) = if roberta { ("<s>", "</s>") } else { ("[CLS]", "[SEP]") };
    let mut vocab = serde_json::Map::new();
    let words = ["[UNK]", cls, sep, "[MASK]", "[PAD]", "choice", "score", "noul", "question", ":", "which", "a", "b",
                 "c", "d", "e", "f", "g", "h"];
    for (i, w) in words.iter().enumerate() {
        vocab.insert(w.to_string(), i.into());
    }
    let special = |id: u32, content: &str| {
        serde_json::json!({"id": id, "content": content, "single_word": false, "lstrip": false,
                           "rstrip": false, "normalized": false, "special": true})
    };
    let post_processor = if roberta {
        serde_json::json!({"type": "RobertaProcessing", "sep": [sep, 2], "cls": [cls, 1],
                           "trim_offsets": true, "add_prefix_space": false})
    } else {
        serde_json::json!({
            "type": "TemplateProcessing",
            "single": [{"SpecialToken": {"id": cls, "type_id": 0}}, {"Sequence": {"id": "A", "type_id": 0}}, {"SpecialToken": {"id": sep, "type_id": 0}}],
            "pair": [{"SpecialToken": {"id": cls, "type_id": 0}}, {"Sequence": {"id": "A", "type_id": 0}}, {"SpecialToken": {"id": sep, "type_id": 0}},
                     {"Sequence": {"id": "B", "type_id": 1}}, {"SpecialToken": {"id": sep, "type_id": 1}}],
            "special_tokens": {
                cls: {"id": cls, "ids": [1], "tokens": [cls]},
                sep: {"id": sep, "ids": [2], "tokens": [sep]}
            }
        })
    };
    serde_json::json!({
        "version": "1.0", "truncation": null, "padding": null,
        "added_tokens": [special(0, "[UNK]"), special(1, cls), special(2, sep)],
        "normalizer": null,
        "pre_tokenizer": {"type": "Whitespace"},
        "post_processor": post_processor,
        "decoder": null,
        "model": {"type": "WordLevel", "vocab": vocab, "unk_token": "[UNK]"}
    })
    .to_string()
}

/// A model directory over the fixture's `<prefix>_16` and `<prefix>_32`.
fn model_dir(tag: &str, manifest: &str, prefix: &str, roberta: bool) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("sk-tiny-nli-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let model = dir.join("tiny-nli");
    std::fs::create_dir_all(&model).unwrap();
    std::fs::write(model.join("classifier.toml"), manifest).unwrap();
    std::fs::write(model.join("tokenizer.json"), tokenizer(roberta)).unwrap();
    for bucket in [16, 32] {
        std::os::unix::fs::symlink(fixtures().join(format!("{prefix}_{bucket}.mlmodelc")), model.join(format!("model_{bucket}.mlmodelc")))
            .unwrap();
    }
    dir
}

fn load(dir: &Path) -> sidekick_core::Result<CoremlClassifier> {
    let registry = ModelRegistry::scan(dir).unwrap();
    assert!(registry.skipped().is_empty(), "{:?}", registry.skipped());
    CoremlClassifier::load_with(registry.classifier("tiny-nli").unwrap(), ComputeUnits::CpuOnly)
}

#[derive(Deserialize)]
struct Model {
    cases: Vec<Case>,
}

#[derive(Deserialize)]
struct Case {
    text: String,
    labels: Vec<String>,
    multi_label: bool,
    pairs: Vec<Pair>,
    scores: Vec<f32>,
}

#[derive(Deserialize)]
struct Pair {
    ids: Vec<i32>,
    type_ids: Vec<i32>,
    logits: Vec<f32>,
}

fn close(a: &[f32], b: &[f32], tol: f32) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| (x - y).abs() < tol)
}

#[test]
fn nli_scores_match_huggingfaces_pipeline_on_bert_and_roberta_pairs() {
    let expected: BTreeMap<String, Model> =
        serde_json::from_str(&std::fs::read_to_string(fixtures().join("expected.json")).unwrap()).unwrap();
    for (name, manifest, roberta, indices) in
        [("bert", BERT.to_string(), false, (2, 0)), ("roberta", roberta_manifest(), true, (0, 1))]
    {
        let dir = model_dir(name, &manifest, name, roberta);
        let clf = load(&dir).unwrap();
        for case in &expected[name].cases {
            let params = ClassifyParams {
                candidate_labels: case.labels.clone(),
                multi_label: case.multi_label,
                ..Default::default()
            };
            let prepared = clf.prepare(&case.text, &params).unwrap();
            let nli = prepared.nli.as_ref().unwrap();
            // The tokenizer's pairs, label by label, as the generator laid
            // them out; segment ids only for the model that takes them.
            for (ours, theirs) in nli.pairs.iter().zip(&case.pairs) {
                assert_eq!(ours.ids, theirs.ids, "{name} {}", case.text);
                assert_eq!(ours.type_ids, theirs.type_ids, "{name} {}", case.text);
                let at = if ours.ids.len() <= 16 { 16 } else { 32 };
                assert_eq!(ours.bucket, at);
                // Each pair's class logits against fp32 torch, in its own
                // bucket and with random pad ids in the larger one.
                for logits in [clf.run_nli_pair(ours, at, &[]).unwrap(), clf.run_nli_pair(ours, 32, &[7; 32]).unwrap()] {
                    assert!(close(&logits, &theirs.logits, 2e-2), "{name}: {logits:?} vs {:?}", theirs.logits);
                }
            }
            // The values, activated as the server does, are the pipeline's
            // scores: a softmax across labels, or each label alone (asked,
            // or the only one).
            let values = clf.run(&prepared).unwrap();
            // The pairs' class logits (`nli_probs`, D47) are torch's, and
            // the values `run` returns are derived from exactly them.
            let classes = clf.run_nli(&prepared).unwrap();
            assert_eq!(classes.len(), case.labels.len());
            for (logits, theirs) in classes.iter().zip(&case.pairs) {
                assert!(close(logits, &theirs.logits, 2e-2), "{name}: {logits:?} vs {:?}", theirs.logits);
                let (ours, torch) = (activate(ProblemType::SingleLabel, logits, None), activate(ProblemType::SingleLabel, &theirs.logits, None));
                assert!(close(&ours, &torch, 1e-2) && (ours.iter().sum::<f32>() - 1.0).abs() < 1e-5, "{name}: {ours:?}");
            }
            let (entailment, contradiction) = indices;
            let derived: Vec<f32> =
                classes.iter().map(|l| sidekick_core::nli_value(l, entailment, contradiction, nli.per_label)).collect();
            assert_eq!(derived, values, "{name}");
            let problem = if nli.per_label { ProblemType::MultiLabel } else { ProblemType::SingleLabel };
            assert_eq!(nli.per_label, case.multi_label || case.labels.len() == 1);
            let probs = activate(problem, &values, None);
            assert!(close(&probs, &case.scores, 1e-2), "{name} {}: {probs:?} vs {:?}", case.text, case.scores);
            // Every pair in a larger bucket: the same values.
            if prepared.bucket < 32 {
                assert!(close(&clf.run_in(&prepared, 32, &[]).unwrap(), &values, 1e-2), "{name}");
            }
        }
        std::fs::remove_dir_all(&dir).unwrap();
    }
}

#[test]
fn every_bucket_must_output_the_manifests_classes() {
    // The 2-way artifacts under a manifest that says 3 classes.
    let wrong = roberta_manifest().replace("n_classes = 2", "n_classes = 3");
    let dir = model_dir("classes", &wrong, "roberta", true);
    let err = load(&dir).err().unwrap().to_string();
    assert!(err.contains("expected [1, 3]") && err.contains("n_classes"), "{err}");
    std::fs::remove_dir_all(&dir).unwrap();
    // The BERT artifacts take segment ids the manifest must name.
    let unnamed = BERT.replace("token_type_ids = \"token_type_ids\"\n", "");
    let dir = model_dir("unnamed", &unnamed, "bert", false);
    let err = load(&dir).err().unwrap().to_string();
    assert!(err.contains("takes input `token_type_ids`"), "{err}");
    std::fs::remove_dir_all(&dir).unwrap();
}
