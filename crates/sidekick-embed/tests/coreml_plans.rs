//! Reading a loaded bucket's Core ML compute plan in the background (D35),
//! with the tiny artifacts in tests/fixtures.
//!
//! Its own test binary, so its own process: turning plan reporting on is
//! process-wide, and from then on every bucket any test loads gets its plan
//! read in the background. A test that deletes its models directory while
//! that read is in flight makes Core ML abort the process.
//!
//! macOS with `--features coreml` only.
#![cfg(all(target_os = "macos", feature = "coreml"))]

use sidekick_core::manifest::ModelRegistry;
use sidekick_core::{EmbedLimits, EmbedPurpose, Embedder};
use sidekick_coreml::ComputeUnits;
use sidekick_embed::CoremlEmbedder;
use std::path::{Path, PathBuf};

fn fixtures(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures").join(name)
}

/// A word-level tokenizer over the tiny models' vocabulary, without special
/// tokens.
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

/// A models directory holding one model: `file` with `body`, its tokenizer,
/// and `links` (artifact name in the model dir, fixture path).
fn models_dir(tag: &str, id: &str, file: &str, body: &str, links: &[(&str, PathBuf)]) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("sk-plans-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let model = dir.join(id);
    std::fs::create_dir_all(&model).unwrap();
    std::fs::write(model.join(file), body).unwrap();
    std::fs::write(model.join("tokenizer.json"), tokenizer()).unwrap();
    for (name, target) in links {
        std::os::unix::fs::symlink(target, model.join(name)).unwrap();
    }
    dir
}

#[test]
fn a_loaded_buckets_compute_plan_is_read_in_the_background_and_cached() {
    use sidekick_embed::placement::{self, Placement};
    // Not ANE-dependent: the plan is read for the CPU, so it's the same on
    // any Mac.
    let manifest = "id = \"plan-embedder\"\nbackend = \"coreml\"\nartifact = \"model_{seq}.mlmodelc\"\n\
                    tokenizer = \"tokenizer.json\"\ndims = 16\npooling = \"none\"\nbuckets = [16]\nmax_seq_len = 16\n\
                    compute_units = \"cpu_only\"\n\
                    [io]\ninput_ids = \"input_ids\"\nattention_mask = \"attention_mask\"\noutput = \"logits\"\n";
    let dir = models_dir(
        "plan",
        "plan-embedder",
        "manifest.toml",
        manifest,
        &[("model_16.mlmodelc", fixtures("tiny-gliner2").join("model_16.mlmodelc"))],
    );
    let cache = dir.join("plans");
    let service = placement::enable(Some(cache.clone())).expect("a Core ML build");
    let registry = ModelRegistry::scan(&dir).unwrap();
    let model = registry.get("plan-embedder").unwrap();
    let path = model.dir.join("model_16.mlmodelc");
    let embedder = CoremlEmbedder::load(model).unwrap();
    // Loading the model keeps no bucket resident, so it reads no plan: a
    // bucket's plan is read after the bucket's first load, here by a request.
    std::thread::sleep(std::time::Duration::from_millis(200));
    assert_eq!(service.get(&path, ComputeUnits::CpuOnly), None, "no plan before the bucket loads");
    embedder.embed_bucketed(&["a b"], EmbedPurpose::Document, EmbedLimits::default()).unwrap();
    let until = std::time::Instant::now() + std::time::Duration::from_secs(120);
    let counts = loop {
        match service.get(&path, ComputeUnits::CpuOnly) {
            Some(Placement::Ready(counts)) => break counts,
            Some(Placement::Failed(e)) => panic!("{e}"),
            _ if std::time::Instant::now() < until => std::thread::sleep(std::time::Duration::from_millis(20)),
            other => panic!("no plan: {other:?}"),
        }
    };
    assert!(counts.cpu > 0 && counts.ane == 0 && counts.gpu == 0, "{counts:?}");
    assert_eq!(std::fs::read_dir(&cache).unwrap().count(), 1, "one cached plan");
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn a_plan_of_an_artifact_that_is_gone_is_an_error_not_an_abort() {
    // Core ML aborts the process when asked to plan a compiled model that
    // isn't there (a model uninstalled after its load); the read checks first.
    let dir = std::env::temp_dir().join(format!("sk-plans-gone-{}", std::process::id()));
    for path in [dir.join("model_16.mlmodelc"), dir.join("empty.mlmodelc")] {
        let _ = std::fs::create_dir_all(dir.join("empty.mlmodelc"));
        let err = sidekick_coreml::compute_plan(&path, ComputeUnits::CpuOnly).unwrap_err();
        assert!(err.to_string().contains("is gone"), "{err}");
    }
    std::fs::remove_dir_all(&dir).unwrap();
}
