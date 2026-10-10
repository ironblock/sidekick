//! The ONNX Runtime CPU backend for embeddings (D40).
//!
//! An `onnx` model is one graph with dynamic shapes, run by ONNX Runtime
//! on the CPU. Each input runs alone, at its own length, so an embedding
//! never depends on what else is in the request. The tokenizing
//! (`embed_input`) and the pooling and normalization (`pooling`) are the
//! Core ML embedder's: the two backends differ only in the forward pass.
//!
//! ONNX Runtime isn't linked at build time. `libonnxruntime` loads when
//! the process first asks for [`runtime`], from `ORT_DYLIB_PATH` or beside
//! the executable. A build without the `onnx` feature has no runtime, and
//! the daemon skips every `onnx` manifest with [`runtime`]'s reason.

/// The ONNX Runtime version sidekick is built and tested against. A
/// runtime of another 1.x version at least as new loads with a warning.
pub const PINNED_VERSION: &str = "1.28.3";

/// Whether this build has the ONNX Runtime backend (the `onnx` feature).
pub const SUPPORTED: bool = cfg!(feature = "onnx");

/// Inference threads per model when the operator doesn't say:
/// [`DEFAULT_INTRA_OP_THREADS`], or one less than the machine's logical
/// CPUs if that's fewer, and at least 1.
pub const DEFAULT_INTRA_OP_THREADS: usize = 4;

/// The default intra-op threads on this machine.
pub fn default_intra_op_threads() -> usize {
    let cpus = std::thread::available_parallelism().map_or(1, |n| n.get());
    DEFAULT_INTRA_OP_THREADS.min(cpus.saturating_sub(1)).max(1)
}

/// The loaded ONNX Runtime.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuntimeInfo {
    /// The version the library reports.
    pub version: String,
    /// Where it was loaded from.
    pub library: std::path::PathBuf,
    /// Whether `ORT_DYLIB_PATH` named it (else it sat beside the executable).
    pub from_env: bool,
    /// Set when the loaded version isn't [`PINNED_VERSION`].
    pub warning: Option<String>,
}

#[cfg(not(feature = "onnx"))]
pub fn runtime() -> Result<RuntimeInfo, String> {
    Err("this build has no ONNX Runtime support (built without the `onnx` feature)".into())
}

#[cfg(feature = "onnx")]
pub use backend::{runtime, OnnxEmbedder};

#[cfg(feature = "onnx")]
mod backend {
    use super::{RuntimeInfo, PINNED_VERSION};
    use crate::embed_input::EmbedInput;
    use crate::pooling::{normalize_in_place, pool};
    use ort::session::builder::GraphOptimizationLevel;
    use ort::session::Session;
    use ort::value::{DynValue, Tensor, TensorElementType};
    use sidekick_core::manifest::ResolvedModel;
    use sidekick_core::{EmbedLimits, EmbedPurpose, Embedder, Error, Pooling, Result};
    use std::path::{Path, PathBuf};
    use std::sync::{Mutex, OnceLock, PoisonError};

    /// The runtime library's file name on this platform.
    #[cfg(target_os = "macos")]
    const LIBRARY: &str = "libonnxruntime.dylib";
    #[cfg(not(target_os = "macos"))]
    const LIBRARY: &str = "libonnxruntime.so";

    /// Load ONNX Runtime once per process, from `ORT_DYLIB_PATH` when set,
    /// else beside the executable, and report what loaded or why nothing did.
    pub fn runtime() -> std::result::Result<RuntimeInfo, String> {
        static RUNTIME: OnceLock<std::result::Result<RuntimeInfo, String>> = OnceLock::new();
        RUNTIME.get_or_init(load_runtime).clone()
    }

    fn load_runtime() -> std::result::Result<RuntimeInfo, String> {
        let from_env = std::env::var_os("ORT_DYLIB_PATH").is_some();
        let library = match std::env::var_os("ORT_DYLIB_PATH") {
            Some(p) => PathBuf::from(p),
            None => {
                let exe = std::env::current_exe().map_err(|e| format!("can't find the executable: {e}"))?;
                let beside = exe.parent().map(|d| d.join(LIBRARY)).unwrap_or_else(|| PathBuf::from(LIBRARY));
                if !beside.is_file() {
                    return Err(format!(
                        "ONNX Runtime ({LIBRARY}) isn't beside {} and ORT_DYLIB_PATH isn't set; install ONNX \
                         Runtime {PINNED_VERSION} there, or point ORT_DYLIB_PATH at it",
                        exe.display()
                    ));
                }
                beside
            }
        };
        let version = library_version(&library)?;
        ort::init_from(&library)
            .map_err(|e| format!("can't load ONNX Runtime: {e}"))?
            .with_name("sidekick")
            .commit();
        let pinned_minor = PINNED_VERSION.rsplit_once('.').map_or(PINNED_VERSION, |(m, _)| m);
        let warning = (!version.starts_with(&format!("{pinned_minor}."))).then(|| {
            format!("ONNX Runtime {version} loaded; sidekick is tested with {PINNED_VERSION}")
        });
        if let Some(w) = &warning {
            tracing::warn!("{w}");
        }
        tracing::info!(version = %version, library = %library.display(), "ONNX Runtime loaded");
        Ok(RuntimeInfo { version, library, from_env, warning })
    }

    /// The version string the library reports, read before ort loads it, so
    /// a library that isn't ONNX Runtime fails with a reason, not a panic.
    fn library_version(path: &Path) -> std::result::Result<String, String> {
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

    /// One `[io]` input: its name and the integer type the graph declares.
    struct Input {
        name: String,
        int64: bool,
    }

    /// An embedding model on ONNX Runtime's CPU.
    pub struct OnnxEmbedder {
        id: String,
        dims: usize,
        matryoshka: Vec<usize>,
        pooling: Pooling,
        max_seq_len: usize,
        input: EmbedInput,
        /// `run` needs the session exclusively: concurrent requests to one
        /// model take turns, each using all its intra-op threads.
        session: Mutex<Session>,
        input_ids: Input,
        attention_mask: Option<Input>,
        token_type_ids: Option<Input>,
        position_ids: Option<Input>,
        output: String,
        intra_op_threads: usize,
    }

    impl OnnxEmbedder {
        /// Load with `intra_op_threads` inference threads.
        pub fn load(model: &ResolvedModel, intra_op_threads: usize) -> Result<Self> {
            runtime().map_err(Error::Inference)?;
            let m = &model.manifest;
            let fail = |what: &str, e: ort::Error| Error::Inference(format!("model `{}`: {what}: {e}", m.id));
            let path = model.dir.join(&m.artifact);
            let session = Session::builder()
                .map_err(|e| fail("session", e))?
                .with_intra_threads(intra_op_threads)
                .map_err(|e| fail("intra-op threads", e.into()))?
                .with_inter_threads(1)
                .map_err(|e| fail("inter-op threads", e.into()))?
                .with_optimization_level(GraphOptimizationLevel::Level3)
                .map_err(|e| fail("optimization level", e.into()))?
                .commit_from_file(&path)
                .map_err(|e| fail(&format!("can't load {}", m.artifact), e))?;
            let declared = |name: &str| -> Result<Input> {
                let outlet = session.inputs().iter().find(|i| i.name() == name).ok_or_else(|| {
                    Error::Inference(format!("model `{}`: the graph has no input `{name}`", m.id))
                })?;
                let int64 = match outlet.dtype().tensor_type() {
                    Some(TensorElementType::Int64) => true,
                    Some(TensorElementType::Int32) => false,
                    other => {
                        return Err(Error::Inference(format!(
                            "model `{}`: input `{name}` must be int32 or int64, not {other:?}",
                            m.id
                        )))
                    }
                };
                Ok(Input { name: name.to_string(), int64 })
            };
            let io = &m.io;
            Ok(Self {
                id: m.id.clone(),
                dims: m.dims,
                matryoshka: m.matryoshka.clone(),
                pooling: m.pooling,
                max_seq_len: m.max_seq_len,
                input: EmbedInput::load(model)?,
                input_ids: declared(&io.input_ids)?,
                attention_mask: io.attention_mask.as_deref().map(declared).transpose()?,
                token_type_ids: io.token_type_ids.as_deref().map(declared).transpose()?,
                position_ids: io.position_ids.as_deref().map(declared).transpose()?,
                output: io.output.clone(),
                session: Mutex::new(session),
                intra_op_threads,
            })
        }

        /// The inference threads this model's session uses.
        pub fn intra_op_threads(&self) -> usize {
            self.intra_op_threads
        }

        /// Prefix, tokenize and truncate one input, as the daemon does.
        /// Public for the parity suite.
        #[doc(hidden)]
        pub fn prepare(&self, text: &str, purpose: EmbedPurpose, limits: EmbedLimits) -> Result<Vec<i32>> {
            self.input.ids(text, purpose, limits, self.max_seq_len)
        }

        /// Run `ids` (one input, no padding), pool, and unit-normalize.
        #[doc(hidden)]
        pub fn run(&self, ids: &[i32]) -> Result<Vec<f32>> {
            self.run_batch(&[ids], ids.len()).map(|mut v| v.remove(0))
        }

        /// Run inputs padded to `len` with id 0 as one batch: the length
        /// and batch-invariance measurements (D40) compare this with `run`.
        #[doc(hidden)]
        pub fn run_batch(&self, batch: &[&[i32]], len: usize) -> Result<Vec<Vec<f32>>> {
            let fail = |e: ort::Error| Error::Inference(format!("model `{}`: {e}", self.id));
            let n = batch.len();
            let mut ids = Vec::with_capacity(n * len);
            let mut mask = Vec::with_capacity(n * len);
            for row in batch {
                if row.len() > len || row.is_empty() {
                    return Err(Error::Inference(format!("{} tokens can't run at length {len}", row.len())));
                }
                ids.extend(row.iter().map(|&t| i64::from(t)));
                ids.resize(ids.len() + len - row.len(), 0);
                mask.extend(std::iter::repeat_n(1i64, row.len()));
                mask.resize(mask.len() + len - row.len(), 0);
            }
            let positions: Vec<i64> = (0..n).flat_map(|_| 0..len as i64).collect();
            let shape = [n, len];
            let tensor = |input: &Input, data: &[i64]| -> std::result::Result<DynValue, ort::Error> {
                Ok(if input.int64 {
                    Tensor::from_array((shape, data.to_vec()))?.into_dyn()
                } else {
                    Tensor::from_array((shape, data.iter().map(|&v| v as i32).collect::<Vec<_>>()))?.into_dyn()
                })
            };
            let mut inputs: Vec<(String, DynValue)> = vec![(self.input_ids.name.clone(), tensor(&self.input_ids, &ids).map_err(fail)?)];
            if let Some(i) = &self.attention_mask {
                inputs.push((i.name.clone(), tensor(i, &mask).map_err(fail)?));
            }
            if let Some(i) = &self.token_type_ids {
                inputs.push((i.name.clone(), tensor(i, &vec![0; n * len]).map_err(fail)?));
            }
            if let Some(i) = &self.position_ids {
                inputs.push((i.name.clone(), tensor(i, &positions).map_err(fail)?));
            }
            let mut session = self.session.lock().unwrap_or_else(PoisonError::into_inner);
            let outputs = session.run(inputs).map_err(fail)?;
            let (out_shape, data) = outputs[self.output.as_str()].try_extract_tensor::<f32>().map_err(fail)?;
            let out_shape: Vec<i64> = out_shape.iter().copied().collect();
            let want: Vec<i64> = match self.pooling {
                Pooling::None => vec![n as i64, self.dims as i64],
                _ => vec![n as i64, len as i64, self.dims as i64],
            };
            if out_shape != want {
                return Err(Error::Inference(format!(
                    "model `{}`: output `{}` has shape {out_shape:?}, expected {want:?}",
                    self.id, self.output
                )));
            }
            let per_row = data.len() / n;
            let mask_u32: Vec<u32> = mask.iter().map(|&m| m as u32).collect();
            Ok((0..n)
                .map(|r| {
                    let mut v = pool(&data[r * per_row..(r + 1) * per_row], self.dims, &mask_u32[r * len..(r + 1) * len], self.pooling);
                    normalize_in_place(&mut v);
                    v
                })
                .collect())
        }
    }

    impl Embedder for OnnxEmbedder {
        fn id(&self) -> &str {
            &self.id
        }

        fn dims(&self) -> usize {
            self.dims
        }

        fn matryoshka_dims(&self) -> &[usize] {
            &self.matryoshka
        }

        fn embed(&self, texts: &[&str], purpose: EmbedPurpose) -> Result<Vec<Vec<f32>>> {
            self.embed_with(texts, purpose, EmbedLimits::default())
        }

        fn embed_with(&self, texts: &[&str], purpose: EmbedPurpose, limits: EmbedLimits) -> Result<Vec<Vec<f32>>> {
            // Prepare every text first, so a rejected one costs no inference.
            let prepared = texts.iter().map(|t| self.prepare(t, purpose, limits)).collect::<Result<Vec<_>>>()?;
            prepared.iter().map(|ids| self.run(ids)).collect()
        }
    }
}
