//! Chunked buckets (D37) with real Core ML: the tiny agentjev tree model in
//! tests/fixtures/tiny-agentjev-chunked, split into two programs per bucket
//! by sidekick_convert's own chunking (tools/make_classifier_test_models.py),
//! with the same model unchunked beside it. The chain matches the fp32 torch
//! model in every bucket and the unchunked program bit for bit, hands
//! `hidden_out` to the next chunk without copying it itself, and a chain whose chunks
//! don't fit together is refused at load.
//!
//! macOS with `--features coreml` only; runs on the CPU, so it needs no ANE.
#![cfg(all(target_os = "macos", feature = "coreml"))]

use serde::Deserialize;
use sidekick_core::manifest::ModelRegistry;
use sidekick_core::Prepared;
use sidekick_coreml::{ComputeUnits, CoremlChain, Int32Input};
use sidekick_embed::CoremlClassifier;
use std::path::{Path, PathBuf};

const MANIFEST: &str = r#"
id = "tiny-agentjev"
task = "zero-shot-classification"
artifact = "model_{seq}.{chunk}.mlmodelc"
tokenizer = "tokenizer.json"
buckets = [16, 32]
max_seq_len = 32
max_batch = 4

[chunking]
chunks = 2

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

fn unchunked() -> String {
    MANIFEST.replace("model_{seq}.{chunk}.mlmodelc", "model_{seq}.mlmodelc").replace("[chunking]\nchunks = 2\n", "")
}

fn tokenizer() -> String {
    let mut vocab = serde_json::Map::new();
    for (i, w) in ["[UNK]", "[STATE]", "[QUESTION]", "[CANDIDATE]", "a", "b"].iter().enumerate() {
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
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny-agentjev-chunked")
}

/// A models directory with the manifest and `links`: (installed name,
/// fixture name).
fn model_dir(tag: &str, manifest: &str, links: &[(String, String)]) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("sk-tiny-chunked-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let model = dir.join("tiny-agentjev");
    std::fs::create_dir_all(&model).unwrap();
    std::fs::write(model.join("classifier.toml"), manifest).unwrap();
    std::fs::write(model.join("tokenizer.json"), tokenizer()).unwrap();
    for (name, fixture) in links {
        std::os::unix::fs::symlink(fixtures().join(fixture), model.join(name)).unwrap();
    }
    dir
}

/// Every fixture program under its own name.
fn all_programs() -> Vec<(String, String)> {
    let mut links = Vec::new();
    for b in [16, 32] {
        for name in [format!("model_{b}.0.mlmodelc"), format!("model_{b}.1.mlmodelc"), format!("model_{b}.mlmodelc")] {
            links.push((name.clone(), name));
        }
    }
    links
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

fn expected() -> Vec<ExpectedCase> {
    let text = std::fs::read_to_string(fixtures().join("expected.json")).unwrap();
    serde_json::from_str::<Expected>(&text).unwrap().cases
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
fn a_chain_matches_torch_and_the_unchunked_program_bit_for_bit() {
    let dir = model_dir("ok", MANIFEST, &all_programs());
    let chained = load(&dir).unwrap();
    let whole_dir = model_dir("whole", &unchunked(), &all_programs());
    let whole = load(&whole_dir).unwrap();
    for case in &expected() {
        let bucket = if case.ids.len() <= 16 { 16 } else { 32 };
        for b in [bucket, 32] {
            let got = chained.run_in(&prepared(case, bucket), b, &[]).unwrap();
            let want = whole.run_in(&prepared(case, bucket), b, &[]).unwrap();
            assert_eq!(got, want, "bucket {b}: the chain must be the unchunked program exactly");
            for (g, w) in got.iter().zip(&case.logits) {
                assert!((g - w).abs() < 2e-2, "bucket {b}: {got:?} vs torch {:?}", case.logits);
            }
        }
    }
    std::fs::remove_dir_all(&dir).unwrap();
    std::fs::remove_dir_all(&whole_dir).unwrap();
}

/// sidekick hands chunk 0's `hidden_out` feature value to chunk 1 as its
/// `hidden_in` without copying it: the buffer chunk 0 produced is the one
/// chunk 1 is given. Whether Core ML copies it internally isn't observable
/// here; the boundary's measured cost (D37) includes that if it does.
#[test]
fn sidekick_hands_the_next_chunk_the_produced_buffer() {
    let case = &expected()[0];
    let paths: Vec<PathBuf> = (0..2).map(|c| fixtures().join(format!("model_16.{c}.mlmodelc"))).collect();
    for units in [ComputeUnits::CpuOnly, ComputeUnits::CpuAndGpu] {
        let chain = CoremlChain::load(&paths, units).unwrap();
        assert_eq!(chain.len(), 2);
        let pad = |v: &[i32], fill: i32, n: usize| {
            let mut out = v.to_vec();
            out.resize(n, fill);
            out
        };
        let inputs = [
            Int32Input { name: "input_ids", shape: vec![1, 16], data: pad(&case.ids, 0, 16) },
            Int32Input { name: "attention_mask", shape: vec![1, 16], data: pad(&vec![1; case.ids.len()], 0, 16) },
            Int32Input { name: "seg", shape: vec![1, 16], data: pad(&case.seg, -1, 16) },
            Int32Input { name: "position_ids", shape: vec![1, 16], data: pad(&case.position_ids, 0, 16) },
            Int32Input { name: "cand_end", shape: vec![1, 4], data: pad(&case.markers, -1, 4) },
        ];
        let (out, handoffs) = chain.predict_int32_tracing_handoffs(&inputs, "logits").unwrap();
        assert_eq!(handoffs.len(), 1, "one boundary");
        let (produced, given) = handoffs[0];
        assert_ne!(produced, 0);
        assert_eq!(produced, given, "{units:?}: chunk 1 was given chunk 0's own output buffer");
        assert_eq!(out.data, chain.predict_int32(&inputs, "logits").unwrap().data);
    }
}

#[test]
fn a_chain_whose_chunks_dont_fit_together_is_refused() {
    let link = |pairs: &[(&str, &str)]| -> Vec<(String, String)> {
        pairs.iter().map(|(a, b)| (a.to_string(), b.to_string())).collect()
    };
    let ok32 = [("model_32.0.mlmodelc", "model_32.0.mlmodelc"), ("model_32.1.mlmodelc", "model_32.1.mlmodelc")];
    // (tag, programs installed, the error's substring)
    type Case<'a> = (&'a str, Vec<(String, String)>, &'a str);
    let cases: Vec<Case> = vec![
        // The chunks in the wrong order.
        ("swapped", link(&[("model_16.0.mlmodelc", "model_16.1.mlmodelc"), ("model_16.1.mlmodelc", "model_16.0.mlmodelc")]),
         "the first chunk takes `hidden_in`"),
        // A first chunk that is the whole model: no hidden_out to hand on.
        ("whole-first", link(&[("model_16.0.mlmodelc", "model_16.mlmodelc"), ("model_16.1.mlmodelc", "model_16.1.mlmodelc")]),
         "must output `hidden_out`"),
        // A second chunk from another bucket.
        ("other-bucket", link(&[("model_16.0.mlmodelc", "model_16.0.mlmodelc"), ("model_16.1.mlmodelc", "model_32.1.mlmodelc")]),
         "model_16.1.mlmodelc: input"),
        // A chunk missing.
        ("missing", link(&[("model_16.0.mlmodelc", "model_16.0.mlmodelc")]), "model_16.1.mlmodelc: can't read the artifact"),
    ];
    for (tag, mut links, want) in cases {
        links.extend(link(&ok32));
        let dir = model_dir(tag, MANIFEST, &links);
        let e = load(&dir).err().unwrap_or_else(|| panic!("{tag}: loaded"));
        assert!(e.to_string().contains(want), "{tag}: {e}");
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
