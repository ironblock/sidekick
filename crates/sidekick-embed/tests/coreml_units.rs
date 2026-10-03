//! A manifest's `compute_units` on real Core ML, with the tiny artifacts in
//! tests/fixtures (built by tools/make_classifier_test_models.py):
//! - an embedder loads with its manifest's compute units, as the classifier
//!   does (tests/coreml_classifier.rs), read back from Core ML;
//! - D27's shape guard refuses a multi-shape artifact on macOS 27 whatever
//!   the compute units, through `CoremlModel::load` and through a classifier
//!   whose manifest asks for the GPU.
//!
//! macOS with `--features coreml` only.
#![cfg(all(target_os = "macos", feature = "coreml"))]

use sidekick_core::manifest::ModelRegistry;
use sidekick_core::{EmbedLimits, EmbedPurpose, Embedder};
use sidekick_coreml::{input_shapes, load_verdict, ComputeUnits, CoremlModel, ShapeVerdict};
use sidekick_embed::{CoremlClassifier, CoremlEmbedder};
use std::path::{Path, PathBuf};

const ALL_UNITS: [ComputeUnits; 4] =
    [ComputeUnits::CpuAndNeuralEngine, ComputeUnits::CpuAndGpu, ComputeUnits::CpuOnly, ComputeUnits::All];

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
    let dir = std::env::temp_dir().join(format!("sk-units-{tag}-{}", std::process::id()));
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
fn an_embedder_loads_with_its_manifest_compute_units() {
    // tiny-gliner2's 16-token bucket outputs one value per token, [1, 16]:
    // as an embedder with `pooling = "none"` it is a 16-dimensional one.
    for (line, want) in [("", ComputeUnits::CpuAndNeuralEngine), ("compute_units = \"cpu_and_gpu\"\n", ComputeUnits::CpuAndGpu)] {
        let manifest = format!(
            "id = \"tiny-embedder\"\nbackend = \"coreml\"\nartifact = \"model_{{seq}}.mlmodelc\"\n\
             tokenizer = \"tokenizer.json\"\ndims = 16\npooling = \"none\"\nbuckets = [16]\nmax_seq_len = 16\n{line}\
             [io]\ninput_ids = \"input_ids\"\nattention_mask = \"attention_mask\"\noutput = \"logits\"\n"
        );
        let dir = models_dir(
            &format!("embedder-{}", want.name()),
            "tiny-embedder",
            "manifest.toml",
            &manifest,
            &[("model_16.mlmodelc", fixtures("tiny-gliner2").join("model_16.mlmodelc"))],
        );
        let registry = ModelRegistry::scan(&dir).unwrap();
        assert!(registry.skipped().is_empty(), "{:?}", registry.skipped().first().map(|s| &s.reason));
        let model = registry.get("tiny-embedder").unwrap();
        assert_eq!(model.manifest.compute_units_name(), want.name());
        let embedder = CoremlEmbedder::load(model).unwrap();
        assert_eq!(embedder.compute_units().unwrap(), want, "{line:?}");
        // And it serves there, reporting the bucket each input ran in.
        let (v, buckets) =
            embedder.embed_bucketed(&["a b c", "d"], EmbedPurpose::Document, EmbedLimits::default()).unwrap();
        assert_eq!((v[0].len(), buckets), (16, Some(vec![16, 16])));
        assert!(v[0].iter().all(|x| x.is_finite()));
        std::fs::remove_dir_all(&dir).unwrap();
    }
}

#[test]
fn d27_judges_a_multi_shape_artifact_the_same_under_every_compute_unit() {
    let path = fixtures("tiny-multishape").join("model.mlmodelc");
    // The verdict comes from the model description, which doesn't depend on
    // the compute units: on macOS 27 it refuses this layout, earlier it warns.
    let verdict = load_verdict(&input_shapes(&path).unwrap());
    assert!(!matches!(verdict, ShapeVerdict::Static), "the fixture must have several shapes: {verdict:?}");
    let refuse = matches!(verdict, ShapeVerdict::Refuse(_));
    for units in ALL_UNITS {
        match CoremlModel::load(&path, units) {
            Err(e) => {
                assert!(refuse, "{units:?}: refused, but the verdict was {verdict:?}: {e}");
                assert!(e.to_string().contains("several enumerated shapes"), "{units:?}: {e}");
            }
            Ok(_) => assert!(!refuse, "{units:?}: loaded a model the verdict refuses"),
        }
    }

    // Through a classifier whose manifest asks for the GPU: on macOS 27 the
    // load fails on the guard before any prediction.
    let manifest = "id = \"multi\"\ntask = \"text-classification\"\nartifact = \"model.mlmodelc\"\n\
                    tokenizer = \"tokenizer.json\"\nbuckets = [16, 32]\nmax_seq_len = 32\n\
                    compute_units = \"cpu_and_gpu\"\n\n[classify]\nlabels = [\"score\"]\n\n\
                    [classify.io]\ninput_ids = \"input_ids\"\nattention_mask = \"attention_mask\"\noutput = \"logits\"\n";
    let dir = models_dir("multishape", "multi", "classifier.toml", manifest, &[("model.mlmodelc", path)]);
    let registry = ModelRegistry::scan(&dir).unwrap();
    let model = registry.classifier("multi").unwrap();
    assert_eq!(model.manifest.compute_units, ComputeUnits::CpuAndGpu);
    match CoremlClassifier::load(model) {
        Err(e) if refuse => assert!(e.to_string().contains("several enumerated shapes"), "{e}"),
        Err(e) => panic!("before macOS 27 the guard only warns, so the load should succeed: {e}"),
        Ok(_) => assert!(!refuse, "loaded a model the verdict refuses"),
    }
    std::fs::remove_dir_all(&dir).unwrap();
}
