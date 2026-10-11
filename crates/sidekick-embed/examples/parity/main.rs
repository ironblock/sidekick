//! Parity suite: every compute path sidekick can run a model on, graded
//! against the model as published, on inputs chosen to hit the failure
//! modes seen on real models (fixtures/parity/corpus.toml).
//!
//! For each Core ML model in the registry that has a reference (written by
//! tools/parity_reference.py for embedders, and by the classifier reference
//! generator for classifiers), the suite:
//!
//! 1. reads the compute plan of every bucket before anything predicts, and
//!    fails a model the ANE would reject (such artifacts can abort the
//!    process at predict time);
//! 2. runs the corpus through sidekick's product path on `.cpuOnly`,
//!    `.cpuAndGPU` and `.cpuAndNeuralEngine`, each in its own process. For
//!    embedders that is prefix, tokenizer, truncation, bucketing, padding,
//!    prediction, pooling and normalization; for classifiers, the input
//!    builder (laya's format included), bucketing, padding and prediction;
//! 3. enforces the hard gates in fixtures/parity/expectations.toml on every
//!    path (finite output, token ids equal to the reference pipeline's,
//!    bucket and pad invariance, repeatability, determinism across
//!    processes) and this chip's accuracy floors, if recorded;
//! 4. reports, for embedders, worst-case cosine vs the fp32 reference,
//!    similarity drift, per-tag worst cases, Matryoshka dims and ONNX
//!    oracles; for classifiers, raw Δp and Δlogit, argmax flips where the
//!    reference's top-2 margin is at least 0.05, calibrated Δp and gold
//!    accuracy (classify_grade.rs); and the worst cases side by side across
//!    paths.
//!
//! An `onnx` model (D40) is graded on ONNX Runtime's CPU instead, as the
//! `onnx-cpu` path: natively when the suite is built with `--features
//! onnx` (and ORT_DYLIB_PATH names the runtime), and from results written
//! by tools/onnx_parity.py through `--external`, as a cross-check. A build
//! with `--features onnx` alone runs on any OS, Linux included: it grades
//! onnx models and lists Core ML models as skipped.
//!
//! ```sh
//! cargo run --release -p sidekick-embed --features coreml[,onnx] --example parity -- \
//!     [--models-dir DIR]... [--refs DIR] [--model ID]... [--paths cpu,gpu,ane] \
//!     [--external FILE]... [--json report.json] [--suggest-floors]
//! ```
//!
//! Exit status: 0 pass, 1 a gate or floor failed, 2 usage or setup error.
//! The pure logic (metrics, grading, reference and expectation parsing) is
//! unit-tested on every platform by `cargo test`.

// The live runner needs Core ML (macOS, the coreml feature) or ONNX
// Runtime (the onnx feature, any OS). The grading code is exercised by unit
// tests everywhere; without Core ML its classifier half has no runner.
#![cfg_attr(any(test, not(all(target_os = "macos", feature = "coreml"))), allow(dead_code))]

mod classify_grade;
mod classify_reference;
mod expect;
mod fixtures;
mod grade;
mod metrics;
mod reference;
#[cfg(any(all(target_os = "macos", feature = "coreml"), feature = "onnx"))]
mod run;

#[cfg(any(all(target_os = "macos", feature = "coreml"), feature = "onnx"))]
fn main() {
    std::process::exit(run::main());
}

#[cfg(not(any(all(target_os = "macos", feature = "coreml"), feature = "onnx")))]
fn main() {
    eprintln!(
        "the parity suite runs Core ML models (on macOS, --features coreml) or ONNX models (--features onnx): \
         build it with one"
    );
    std::process::exit(2);
}
