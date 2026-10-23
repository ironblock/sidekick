//! Warming a bucket on real Core ML (D52), with a tiny artifact from
//! tests/fixtures: a warm-up loads the bucket through the same loader a
//! request uses, under the same key, reports it warm, and keeps nothing
//! resident; a request then serves from it; and a bucket a request has
//! loaded isn't loaded again by a warm-up. No ledger is kept here
//! (`warm::enable` is the daemon's), so nothing is written outside the
//! temporary directory.
//!
//! macOS with `--features coreml` only.
#![cfg(all(target_os = "macos", feature = "coreml"))]

use sidekick_core::manifest::ModelRegistry;
use sidekick_core::{EmbedLimits, EmbedPurpose, Embedder};
use sidekick_coreml::ComputeUnits;
use sidekick_embed::warm::{self, Key, Label, State, Warmed};
use sidekick_embed::CoremlEmbedder;
use std::path::{Path, PathBuf};

fn tokenizer() -> String {
    let mut vocab = serde_json::Map::new();
    for (i, w) in ["[UNK]", "a", "b", "c", "d"].iter().enumerate() {
        vocab.insert(w.to_string(), i.into());
    }
    serde_json::json!({
        "version": "1.0", "truncation": null, "padding": null, "added_tokens": [],
        "normalizer": null, "pre_tokenizer": {"type": "Whitespace"}, "post_processor": null,
        "decoder": null, "model": {"type": "WordLevel", "vocab": vocab, "unk_token": "[UNK]"}
    })
    .to_string()
}

/// A models directory with tiny-gliner2's 16-token program as a
/// 16-dimensional embedder on `units`.
fn models_dir(tag: &str, units: ComputeUnits) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("sk-warm-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let model = dir.join("tiny");
    std::fs::create_dir_all(&model).unwrap();
    let manifest = format!(
        "id = \"tiny\"\nbackend = \"coreml\"\nartifact = \"model_{{seq}}.mlmodelc\"\ntokenizer = \"tokenizer.json\"\n\
         dims = 16\npooling = \"none\"\nbuckets = [16]\nmax_seq_len = 16\ncompute_units = \"{}\"\n\
         [io]\ninput_ids = \"input_ids\"\nattention_mask = \"attention_mask\"\noutput = \"logits\"\n",
        units.name()
    );
    std::fs::write(model.join("manifest.toml"), manifest).unwrap();
    std::fs::write(model.join("tokenizer.json"), tokenizer()).unwrap();
    let fixture = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny-gliner2/model_16.mlmodelc");
    std::os::unix::fs::symlink(fixture, model.join("model_16.mlmodelc")).unwrap();
    dir
}

#[test]
fn a_warmed_bucket_is_warm_holds_nothing_and_serves() {
    let units = ComputeUnits::CpuAndGpu;
    let dir = models_dir("warmed", units);
    let registry = ModelRegistry::scan(&dir).unwrap();
    let model = registry.get("tiny").unwrap();
    let key = Key::bucket(&model.dir, &model.manifest.artifact, 16, model.manifest.chunks(), units);
    let label = Label { model: "tiny", bucket: 16 };
    assert_eq!(warm::status(&key).unwrap().state, State::Cold);

    let warmed = warm::warm(&key, label).unwrap();
    assert!(matches!(warmed, Warmed::Loaded { recorded: false, .. }), "{warmed:?}");
    let status = warm::status(&key).unwrap();
    assert_eq!((status.state, status.loaded), (State::Warm, true));
    assert!(status.load_ms.is_some());
    assert_eq!(warm::warm(&key, label).unwrap(), Warmed::AlreadyLoaded, "loaded once per process");

    // The embedder a request gets resolves the same key, and serves.
    let embedder = CoremlEmbedder::load(model).unwrap();
    assert_eq!(embedder.resident_buckets(), Vec::<usize>::new(), "the warm-up kept nothing resident");
    let (v, buckets) = embedder.embed_bucketed(&["a b"], EmbedPurpose::Document, EmbedLimits::default()).unwrap();
    assert_eq!((v[0].len(), buckets), (16, Some(vec![16])));
    assert_eq!(embedder.resident_buckets(), [16]);
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn a_bucket_a_request_loaded_is_warm_and_not_warmed_again() {
    let units = ComputeUnits::CpuOnly;
    let dir = models_dir("requested", units);
    let registry = ModelRegistry::scan(&dir).unwrap();
    let model = registry.get("tiny").unwrap();
    let embedder = CoremlEmbedder::load(model).unwrap();
    embedder.embed(&["a"], EmbedPurpose::Query).unwrap();
    let key = Key::bucket(&model.dir, &model.manifest.artifact, 16, model.manifest.chunks(), units);
    let status = warm::status(&key).unwrap();
    assert_eq!((status.state, status.loaded), (State::Warm, true));
    assert_eq!(warm::warm(&key, Label { model: "tiny", bucket: 16 }).unwrap(), Warmed::AlreadyLoaded);
    // The same program on other compute units is another cache entry.
    let ane = Key { units: ComputeUnits::CpuAndNeuralEngine, ..key };
    assert_eq!(warm::status(&ane).unwrap().state, State::Cold);
    std::fs::remove_dir_all(&dir).unwrap();
}
