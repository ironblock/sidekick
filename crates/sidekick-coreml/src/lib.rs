//! Minimal safe wrapper over Core ML.
//!
//! Scope is deliberately tiny: load a compiled model with a chosen compute
//! unit preference, run batch-1 predictions with int32/float inputs, read
//! float outputs. Everything sidekick needs for encoder models on the ANE,
//! and nothing else.
//!
//! On non-macOS targets this crate compiles to an empty stub so that the
//! workspace builds and tests everywhere; `sidekick-embed` gates its Core ML
//! backend on `target_os = "macos"` accordingly.

#[cfg(target_os = "macos")]
mod model;
#[cfg(target_os = "macos")]
pub use model::{CoremlModel, OutputTensor};

/// Compute-unit preference. `CpuAndNeuralEngine` is sidekick's default: it
/// keeps background work off the GPU entirely, which is the point of the
/// project. Use `All` only when measuring.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ComputeUnits {
    All,
    #[default]
    CpuAndNeuralEngine,
    CpuAndGpu,
    CpuOnly,
}

/// A named int32 tensor input (shape is row-major, batch dim included).
#[derive(Debug)]
pub struct Int32Input<'a> {
    pub name: &'a str,
    pub shape: Vec<usize>,
    pub data: Vec<i32>,
}
