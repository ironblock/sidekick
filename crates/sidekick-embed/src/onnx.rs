//! The ONNX Runtime CPU backend for embeddings (D40).
//!
//! This build has no ONNX Runtime: every `onnx` manifest is skipped with
//! [`runtime`]'s reason.

/// The ONNX Runtime this process can use, or why it can't use one. A
/// daemon skips every `onnx` manifest with the reason when it can't.
pub fn runtime() -> Result<RuntimeInfo, String> {
    Err("this build has no ONNX Runtime support (built without the `onnx` feature)".into())
}

/// The loaded ONNX Runtime.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuntimeInfo {
    /// The version the library reports.
    pub version: String,
    /// Where it was loaded from.
    pub library: std::path::PathBuf,
}
