//! The onnx embedding backend (D40) on a real ONNX Runtime, with the tiny
//! graphs in tests/fixtures/tiny-onnx (built by
//! tools/make_onnx_test_models.py, expected vectors from numpy):
//! - a token-level graph pooled in Rust (mean, cls, last_token), its input
//!   types followed (int64 ids, int32 positions), token types fed zeros;
//! - a pooled graph (`pooling = "none"`) with int32 inputs;
//! - the product path from text, and the registry's scan-time check on
//!   real ONNX files.
//!
//! Needs `--features onnx` and ONNX Runtime at ORT_DYLIB_PATH. Without it
//! the tests skip, unless SIDEKICK_REQUIRE_ONNX=1 (CI) makes that a failure.
#![cfg(feature = "onnx")]

use serde::Deserialize;
use sidekick_core::manifest::ModelRegistry;
use sidekick_core::{EmbedLimits, EmbedPurpose, Embedder};
use sidekick_embed::onnx::{load_runtime, OnnxEmbedder, RuntimeSource, PINNED_VERSION};
use std::path::{Path, PathBuf};

fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny-onnx")
}

/// Whether ONNX Runtime is here to test with.
fn have_runtime() -> bool {
    if std::env::var_os("ORT_DYLIB_PATH").is_some() {
        return true;
    }
    assert!(
        std::env::var("SIDEKICK_REQUIRE_ONNX").as_deref() != Ok("1"),
        "SIDEKICK_REQUIRE_ONNX=1 but ORT_DYLIB_PATH isn't set"
    );
    eprintln!("skipping: set ORT_DYLIB_PATH to ONNX Runtime {PINNED_VERSION} to run the onnx backend's tests");
    false
}

/// A word-level tokenizer over the fixture's 12 ids, splitting on spaces
/// only, with no special tokens of its own.
#[test]
fn an_input_with_no_tokens_is_a_bad_request() {
    use sidekick_core::{EmbedPurpose, Embedder, Error};
    if !have_runtime() {
        return;
    }
    // This tokenizer adds no special tokens, so "" tokenizes to nothing.
    let dir = models_dir("empty", "pooled.onnx", "none", POOLED_IO);
    let model = load(&dir, 1);
    let err = model.embed(&[""], EmbedPurpose::Document).unwrap_err();
    assert!(matches!(err, Error::InvalidRequest(_)), "{err:?}");
    std::fs::remove_dir_all(&dir).unwrap();
}

fn tokenizer() -> String {
    let words = ["[UNK]", "[CLS]", "[SEP]", "a", "b", "c", "d", "e", "f", "g", "h", "i"];
    let vocab: serde_json::Map<String, serde_json::Value> =
        words.iter().enumerate().map(|(i, w)| (w.to_string(), i.into())).collect();
    serde_json::json!({
        "version": "1.0", "truncation": null, "padding": null, "added_tokens": [],
        "normalizer": null, "pre_tokenizer": {"type": "WhitespaceSplit"}, "post_processor": null,
        "decoder": null, "model": {"type": "WordLevel", "vocab": vocab, "unk_token": "[UNK]"}
    })
    .to_string()
}

const HIDDEN_IO: &str = "input_ids = \"input_ids\"\nattention_mask = \"attention_mask\"\n\
                         token_type_ids = \"token_type_ids\"\nposition_ids = \"position_ids\"\n\
                         output = \"last_hidden_state\"\n";
const POOLED_IO: &str = "input_ids = \"input_ids\"\nattention_mask = \"attention_mask\"\noutput = \"embedding\"\n";

/// A models directory holding one onnx model over `graph`.
fn models_dir(tag: &str, graph: &str, pooling: &str, io: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("sk-onnx-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let model = dir.join("tiny");
    std::fs::create_dir_all(&model).unwrap();
    std::fs::write(
        model.join("manifest.toml"),
        format!(
            "id = \"tiny\"\nbackend = \"onnx\"\nartifact = \"{graph}\"\ntokenizer = \"tokenizer.json\"\ndims = 8\n\
             pooling = \"{pooling}\"\nmax_seq_len = 16\n\n[io]\n{io}"
        ),
    )
    .unwrap();
    std::fs::write(model.join("tokenizer.json"), tokenizer()).unwrap();
    std::os::unix::fs::symlink(fixtures().join(graph), model.join(graph)).unwrap();
    dir
}

fn load(dir: &Path, threads: usize) -> OnnxEmbedder {
    let registry = ModelRegistry::scan(dir).unwrap();
    assert!(registry.skipped().is_empty(), "{:?}", registry.skipped().first().map(|s| &s.reason));
    OnnxEmbedder::load(registry.get("tiny").unwrap(), threads).unwrap()
}

#[derive(Deserialize)]
struct Expected {
    cases: Vec<Case>,
}

#[derive(Deserialize)]
struct Case {
    ids: Vec<i32>,
    mean: Vec<f32>,
    cls: Vec<f32>,
    last_token: Vec<f32>,
    pooled: Vec<f32>,
}

fn expected() -> Expected {
    serde_json::from_str(&std::fs::read_to_string(fixtures().join("expected.json")).unwrap()).unwrap()
}

fn max_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max)
}

#[test]
fn the_runtime_loads_and_reports_its_version() {
    if !have_runtime() {
        return;
    }
    let rt = load_runtime(None).unwrap();
    assert_eq!(rt.version, PINNED_VERSION, "the pinned runtime");
    assert_eq!(rt.source, RuntimeSource::Env);
    assert!(rt.warning.is_none());
    assert!(rt.library.is_file());
}

#[test]
fn a_token_level_graph_is_pooled_in_rust_as_numpy_pools_it() {
    if !have_runtime() {
        return;
    }
    for pooling in ["mean", "cls", "last_token"] {
        let dir = models_dir(&format!("hidden-{pooling}"), "hidden.onnx", pooling, HIDDEN_IO);
        let model = load(&dir, 1);
        for case in &expected().cases {
            let want = match pooling {
                "mean" => &case.mean,
                "cls" => &case.cls,
                _ => &case.last_token,
            };
            let got = model.run(&case.ids).unwrap();
            assert!(max_diff(&got, want) < 1e-5, "{pooling} {:?}: {got:?} vs {want:?}", case.ids);
        }
        std::fs::remove_dir_all(&dir).unwrap();
    }
}

#[test]
fn a_pooled_graph_with_int32_inputs_is_taken_as_it_is() {
    if !have_runtime() {
        return;
    }
    let dir = models_dir("pooled", "pooled.onnx", "none", POOLED_IO);
    let model = load(&dir, 1);
    for case in &expected().cases {
        let got = model.run(&case.ids).unwrap();
        assert!(max_diff(&got, &case.pooled) < 1e-5, "{:?}: {got:?} vs {:?}", case.ids, case.pooled);
    }
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn the_product_path_tokenizes_then_runs_each_input_alone() {
    if !have_runtime() {
        return;
    }
    let dir = models_dir("product", "hidden.onnx", "cls", HIDDEN_IO);
    let model = load(&dir, 2);
    assert_eq!(model.intra_op_threads(), 2);
    let cases = expected().cases;
    // "[CLS] b c [SEP]" is ids 1 4 5 2, the first case.
    let (vectors, buckets) = model
        .embed_bucketed(&["[CLS] b c [SEP]", "[CLS] d [SEP]"], EmbedPurpose::Document, EmbedLimits::default())
        .unwrap();
    assert!(max_diff(&vectors[0], &cases[0].cls) < 1e-5);
    assert!(max_diff(&vectors[1], &cases[1].cls) < 1e-5);
    assert_eq!(buckets, None, "one graph: no buckets to report");
    // max_seq_len (16) truncates, keeping the final token.
    let long = format!("[CLS] {} [SEP]", vec!["a"; 30].join(" "));
    assert_eq!(model.prepare(&long, EmbedPurpose::Document, EmbedLimits::default()).unwrap().len(), 16);
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn the_scan_time_check_reads_real_onnx_files() {
    // No runtime needed: the registry reads the graph's declarations.
    let dir = models_dir("scan-ok", "hidden.onnx", "cls", HIDDEN_IO);
    assert!(ModelRegistry::scan(&dir).unwrap().skipped().is_empty());
    std::fs::remove_dir_all(&dir).unwrap();
    // The graph takes token_type_ids and position_ids, which this [io] omits.
    let dir = models_dir("scan-missing", "hidden.onnx", "cls", POOLED_IO.replace("embedding", "last_hidden_state").as_str());
    let reg = ModelRegistry::scan(&dir).unwrap();
    assert!(reg.skipped()[0].reason.contains("takes `token_type_ids`"), "{}", reg.skipped()[0].reason);
    std::fs::remove_dir_all(&dir).unwrap();
    // A pooled graph read as token-level.
    let dir = models_dir("scan-rank", "pooled.onnx", "mean", POOLED_IO);
    let reg = ModelRegistry::scan(&dir).unwrap();
    assert!(reg.skipped()[0].reason.contains("has rank 2"), "{}", reg.skipped()[0].reason);
    std::fs::remove_dir_all(&dir).unwrap();
}
