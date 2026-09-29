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
//! ```sh
//! cargo run --release -p sidekick-embed --features coreml --example parity -- \
//!     [--models-dir DIR]... [--refs DIR] [--model ID]... [--paths cpu,gpu,ane] \
//!     [--json report.json] [--suggest-floors]
//! ```
//!
//! Exit status: 0 pass, 1 a gate or floor failed, 2 usage or setup error.
//! The pure logic (metrics, grading, reference and expectation parsing) is
//! unit-tested on every platform by `cargo test`.

// The live runner exists only on macOS with the coreml feature; elsewhere,
// and under the test harness, the grading code is exercised by unit tests.
#![cfg_attr(
    any(test, not(all(target_os = "macos", feature = "coreml"))),
    allow(dead_code)
)]

mod classify_grade;
mod classify_reference;
mod expect;
mod grade;
mod metrics;
mod reference;
#[cfg(all(target_os = "macos", feature = "coreml"))]
mod run;

#[cfg(all(target_os = "macos", feature = "coreml"))]
fn main() {
    std::process::exit(run::main());
}

#[cfg(not(all(target_os = "macos", feature = "coreml")))]
fn main() {
    eprintln!("the parity suite runs Core ML models: build it on macOS with --features coreml");
    std::process::exit(2);
}
