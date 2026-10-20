//! A chunked embedder (D37) with real Core ML: the tiny plain-Qwen3
//! last-token embedder in tests/fixtures/tiny-qwen3-embed-chunked, split into
//! two programs per bucket by sidekick_convert's own chunking
//! (tools/make_classifier_test_models.py), with the same model unchunked
//! beside it. Through the embedder's own load and run, the chain matches the
//! unchunked program bit for bit and the fp32 torch model, and a chain whose
//! chunks don't fit together is refused at load. The unchunked model also
//! shows truncate `NONE` refusing an over-length batch before any bucket
//! loads.
//!
//! macOS with `--features coreml` only; runs on the CPU and the GPU, so it
//! needs no ANE.
#![cfg(all(target_os = "macos", feature = "coreml"))]

use serde::Deserialize;
use sidekick_core::manifest::ModelRegistry;
use sidekick_coreml::ComputeUnits;
use sidekick_embed::CoremlEmbedder;
use std::path::{Path, PathBuf};

const MANIFEST: &str = r#"
id = "tiny-qwen3-embed"
backend = "coreml"
artifact = "model_{seq}.{chunk}.mlmodelc"
tokenizer = "tokenizer.json"
dims = 32
pooling = "none"
buckets = [16, 32]
max_seq_len = 32

[chunking]
chunks = 2

[io]
input_ids = "input_ids"
attention_mask = "attention_mask"
output = "embedding"
"#;

fn unchunked() -> String {
    MANIFEST.replace("model_{seq}.{chunk}.mlmodelc", "model_{seq}.mlmodelc").replace("[chunking]\nchunks = 2\n", "")
}

/// A word-level tokenizer; the tests run token ids directly.
fn tokenizer() -> String {
    let mut vocab = serde_json::Map::new();
    for (i, w) in ["[UNK]", "a", "b"].iter().enumerate() {
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
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny-qwen3-embed-chunked")
}

/// A models directory with the manifest and `links`: (installed name,
/// fixture name).
fn model_dir(tag: &str, manifest: &str, links: &[(String, String)]) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("sk-tiny-chunked-embed-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let model = dir.join("tiny-qwen3-embed");
    std::fs::create_dir_all(&model).unwrap();
    std::fs::write(model.join("manifest.toml"), manifest).unwrap();
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

fn load(dir: &Path, units: ComputeUnits) -> sidekick_core::Result<CoremlEmbedder> {
    let registry = ModelRegistry::scan(dir).unwrap();
    assert!(registry.skipped().is_empty(), "{:?}", registry.skipped().first().map(|s| &s.reason));
    CoremlEmbedder::load_with(registry.get("tiny-qwen3-embed").unwrap(), units)
}

#[derive(Deserialize)]
struct Expected {
    cases: Vec<ExpectedCase>,
}

#[derive(Deserialize)]
struct ExpectedCase {
    ids: Vec<i32>,
    embedding: Vec<f32>,
}

fn expected() -> Vec<ExpectedCase> {
    let text = std::fs::read_to_string(fixtures().join("expected.json")).unwrap();
    serde_json::from_str::<Expected>(&text).unwrap().cases
}

#[test]
fn a_chained_embedder_matches_torch_and_the_unchunked_program_bit_for_bit() {
    let dir = model_dir("ok", MANIFEST, &all_programs());
    let whole_dir = model_dir("whole", &unchunked(), &all_programs());
    for units in [ComputeUnits::CpuOnly, ComputeUnits::CpuAndGpu] {
        let chained = load(&dir, units).unwrap();
        let whole = load(&whole_dir, units).unwrap();
        for case in &expected() {
            // the embedder unit-normalizes; so does the reference here
            let norm = case.embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
            let want_torch: Vec<f32> = case.embedding.iter().map(|x| x / norm).collect();
            let bucket = if case.ids.len() <= 16 { 16 } else { 32 };
            for b in [bucket, 32] {
                let got = chained.run(&case.ids, b).unwrap();
                let want = whole.run(&case.ids, b).unwrap();
                assert_eq!(got, want, "{units:?} bucket {b}: the chain must be the unchunked program exactly");
                let cosine: f32 = got.iter().zip(&want_torch).map(|(g, w)| g * w).sum();
                assert!(cosine > 0.999, "{units:?} bucket {b}: cosine {cosine} to torch");
            }
        }
    }
    std::fs::remove_dir_all(&dir).unwrap();
    std::fs::remove_dir_all(&whole_dir).unwrap();
}

#[test]
fn a_chained_embedder_whose_chunks_dont_fit_together_is_refused() {
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
        let e = load(&dir, ComputeUnits::CpuOnly).err().unwrap_or_else(|| panic!("{tag}: loaded"));
        assert!(e.to_string().contains(want), "{tag}: {e}");
        std::fs::remove_dir_all(&dir).unwrap();
    }
}

#[test]
fn truncate_none_refuses_a_batch_before_any_bucket_loads() {
    use sidekick_core::{EmbedLimits, EmbedPurpose, Embedder, Truncate};
    let dir = model_dir("none", &unchunked(), &all_programs());
    let model = load(&dir, ComputeUnits::CpuOnly).unwrap();
    let long = "a ".repeat(40);
    let batch = ["a b", long.as_str()];
    let none = EmbedLimits { truncate: Truncate::Reject, max_tokens: None };
    let e = model.embed_with(&batch, EmbedPurpose::Document, none).unwrap_err();
    assert!(e.to_string().contains("input 1: 40 tokens, over model `tiny-qwen3-embed`'s limit of 32 tokens"), "{e}");
    // Every input is prepared before any runs: nothing loaded, so nothing ran.
    assert!(model.resident_buckets().is_empty(), "{:?}", model.resident_buckets());
    // The default truncates it to the largest bucket instead.
    let (vectors, buckets) = model.embed_bucketed(&batch, EmbedPurpose::Document, EmbedLimits::default()).unwrap();
    assert_eq!((vectors.len(), buckets), (2, Some(vec![16, 32])));
    std::fs::remove_dir_all(&dir).unwrap();
}
