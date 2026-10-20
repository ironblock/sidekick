//! ONNX Runtime on the CPU for sidekick (D40): what `sidekick-embed`'s
//! `onnx` backend runs on, as `sidekick-coreml` is what its Core ML
//! backends run on.
//!
//! - [`load_runtime`]: find `libonnxruntime` and load it, once per
//!   process: from the daemon config's `onnx_runtime`, else
//!   `ORT_DYLIB_PATH`, else beside the executable. Nothing links at build
//!   time; the daemon calls this at startup only when it has an `onnx`
//!   model.
//! - [`OnnxModel`] (feature `runtime`): one graph in a session, fed the
//!   integer types its inputs declare.
//! - [`avx512`]: on an Intel Mac, turn on AVX-512 state for the thread
//!   that loads the runtime, so ONNX Runtime picks its AVX-512 kernels.
//! - [`default_threads`] and [`detected_cores`]: the inference-thread rule,
//!   the machine's fastest physical cores capped at 8.
//!
//! Without the `runtime` feature the crate is the pinned version and the
//! thread rule only, `ort` isn't in the build, and [`load_runtime`] reports
//! that the build has no ONNX Runtime. The graph checks the registry runs
//! at scan (interface, quantization, external data) need no runtime, and
//! live in `sidekick_core::onnx_graph`.

pub mod avx512;
mod runtime;
#[cfg(feature = "runtime")]
mod session;
mod threads;

pub use runtime::{load_runtime, runtime_state, RuntimeInfo, RuntimeSource};
#[cfg(feature = "runtime")]
pub use session::{MemoryOptions, OnnxModel, SHRINK_FROM_TOKENS};
pub use threads::{
    default_intra_op_threads, default_threads, detected_cores, DetectedCores, FALLBACK_INTRA_OP_THREADS,
    MAX_DEFAULT_INTRA_OP_THREADS,
};

/// The ONNX Runtime version sidekick is built and tested against on this
/// target: 1.23.2 on Intel macOS, the last Microsoft built for it, and
/// 1.28.3 everywhere else. Another version from 1.23 on loads, with a
/// warning; older ones are refused.
#[cfg(all(target_os = "macos", target_arch = "x86_64"))]
pub const PINNED_VERSION: &str = "1.23.2";
#[cfg(not(all(target_os = "macos", target_arch = "x86_64")))]
pub const PINNED_VERSION: &str = "1.28.3";

/// Whether this build has ONNX Runtime (the `runtime` feature).
pub const RUNTIME: bool = cfg!(feature = "runtime");
