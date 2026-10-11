//! Finding and loading `libonnxruntime`, once per process (D40).

use std::path::Path;

/// The loaded ONNX Runtime.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuntimeInfo {
    /// The version the library reports.
    pub version: String,
    /// Where it was loaded from.
    pub library: std::path::PathBuf,
    /// Who named it.
    pub source: RuntimeSource,
    /// Set when the loaded version isn't [`PINNED_VERSION`](crate::PINNED_VERSION)'s minor.
    pub warning: Option<String>,
}

/// Who named the runtime library, in the order sidekick looks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuntimeSource {
    /// The daemon config's `onnx_runtime`.
    Config,
    /// The `ORT_DYLIB_PATH` environment variable.
    Env,
    /// Beside the executable, where release archives put it.
    BesideExecutable,
}

impl RuntimeSource {
    /// As `/health` reports it.
    pub fn name(self) -> &'static str {
        match self {
            Self::Config => "config",
            Self::Env => "ORT_DYLIB_PATH",
            Self::BesideExecutable => "beside sidekickd",
        }
    }
}

static RUNTIME: std::sync::OnceLock<Result<RuntimeInfo, String>> = std::sync::OnceLock::new();

/// Load ONNX Runtime, once per process: from `configured` (the daemon
/// config's `onnx_runtime`), else `ORT_DYLIB_PATH`, else beside the
/// executable. There's no search of system library paths: naming the file
/// is the opt-in. Later calls return the first call's result, whatever
/// they pass.
pub fn load_runtime(configured: Option<&Path>) -> Result<RuntimeInfo, String> {
    RUNTIME.get_or_init(|| load(configured)).clone()
}

/// What [`load_runtime`] gave, or `None` if this process never loaded it
/// (no `onnx` model).
pub fn runtime_state() -> Option<Result<RuntimeInfo, String>> {
    RUNTIME.get().cloned()
}

#[cfg(not(feature = "runtime"))]
fn load(_: Option<&Path>) -> Result<RuntimeInfo, String> {
    Err("this build has no ONNX Runtime support (built without the `onnx` feature)".into())
}

#[cfg(feature = "runtime")]
use loader::load;

#[cfg(feature = "runtime")]
mod loader {
    use super::{RuntimeInfo, RuntimeSource};
    use crate::PINNED_VERSION;
    use std::path::{Path, PathBuf};

    /// The runtime library's file name on this platform.
    #[cfg(target_os = "macos")]
    const LIBRARY: &str = "libonnxruntime.dylib";
    #[cfg(not(target_os = "macos"))]
    const LIBRARY: &str = "libonnxruntime.so";

    /// The configured library, else `ORT_DYLIB_PATH`'s, else the one beside
    /// the executable; loaded, and what loaded or why nothing did.
    pub(super) fn load(configured: Option<&Path>) -> Result<RuntimeInfo, String> {
        let (library, source) = match (configured, std::env::var_os("ORT_DYLIB_PATH")) {
            (Some(p), _) => (p.to_path_buf(), RuntimeSource::Config),
            (None, Some(p)) => (PathBuf::from(p), RuntimeSource::Env),
            (None, None) => {
                let exe = std::env::current_exe().map_err(|e| format!("can't find the executable: {e}"))?;
                let beside = exe.parent().map(|d| d.join(LIBRARY)).unwrap_or_else(|| PathBuf::from(LIBRARY));
                if !beside.is_file() {
                    // The executable's name, not its path: /health reports this.
                    let name = exe.file_name().map_or("the executable".into(), |n| n.to_string_lossy().into_owned());
                    return Err(format!(
                        "ONNX Runtime ({LIBRARY}) isn't beside {name}, and neither `onnx_runtime` in the config nor \
                         ORT_DYLIB_PATH names one; install ONNX Runtime {PINNED_VERSION} there, or name it"
                    ));
                }
                (beside, RuntimeSource::BesideExecutable)
            }
        };
        let version = library_version(&library)?;
        ort::init_from(&library)
            .map_err(|e| format!("can't load ONNX Runtime: {e}"))?
            .with_name("sidekick")
            .commit();
        let pinned_minor = PINNED_VERSION.rsplit_once('.').map_or(PINNED_VERSION, |(m, _)| m);
        let warning = (!version.starts_with(&format!("{pinned_minor}.")))
            .then(|| format!("ONNX Runtime {version} loaded; sidekick is tested with {PINNED_VERSION}"));
        if let Some(w) = &warning {
            tracing::warn!("{w}");
        }
        tracing::info!(version = %version, library = %library.display(), "ONNX Runtime loaded");
        Ok(RuntimeInfo { version, library, source, warning })
    }

    /// The version string the library reports, read before ort loads it, so
    /// a library that isn't ONNX Runtime fails with a reason, not a panic.
    fn library_version(path: &Path) -> Result<String, String> {
        type ApiBase = unsafe extern "C" fn() -> *const ort::sys::OrtApiBase;
        // SAFETY: loading a library runs its initializers; the path is the
        // operator's choice of ONNX Runtime.
        let lib = unsafe { libloading::Library::new(path) }
            .map_err(|e| format!("can't load ONNX Runtime from {}: {e}", path.display()))?;
        // SAFETY: OrtGetApiBase has this signature in every ONNX Runtime.
        let get: libloading::Symbol<ApiBase> = unsafe { lib.get(b"OrtGetApiBase") }
            .map_err(|_| format!("{} isn't ONNX Runtime (no OrtGetApiBase)", path.display()))?;
        // SAFETY: the API base and its version string are static in the library.
        let version = unsafe {
            let base = get();
            if base.is_null() {
                return Err(format!("{}: OrtGetApiBase returned nothing", path.display()));
            }
            std::ffi::CStr::from_ptr(((*base).GetVersionString)()).to_string_lossy().into_owned()
        };
        // ort keeps its own handle; this one closes when it drops.
        Ok(version)
    }
}
