//! The ONNX Runtime CPU backend for embeddings (D40).
//!
//! An `onnx` model is one graph with dynamic shapes, run by ONNX Runtime
//! on the CPU. Each input runs alone, at its own length, so an embedding
//! never depends on what else is in the request. The tokenizing
//! (`embed_input`) and the pooling and normalization (`pooling`) are the
//! Core ML embedder's: the two backends differ only in the forward pass.
//!
//! ONNX Runtime isn't linked at build time. `libonnxruntime` loads the
//! first time the process calls [`load_runtime`]: the daemon does that at
//! startup when its models directory has an `onnx` model, and never
//! otherwise. It comes from the configured path, else `ORT_DYLIB_PATH`,
//! else beside the executable. A build without the `onnx` feature has no
//! runtime, and the daemon skips every `onnx` manifest with the reason.

/// The ONNX Runtime version sidekick is built and tested against on this
/// target: 1.23.2 on Intel macOS, the last Microsoft built for it, and
/// 1.28.3 everywhere else. Another version from 1.23 on loads, with a
/// warning; older ones are refused.
#[cfg(all(target_os = "macos", target_arch = "x86_64"))]
pub const PINNED_VERSION: &str = "1.23.2";
#[cfg(not(all(target_os = "macos", target_arch = "x86_64")))]
pub const PINNED_VERSION: &str = "1.28.3";

/// Whether this build has the ONNX Runtime backend (the `onnx` feature).
pub const SUPPORTED: bool = cfg!(feature = "onnx");

/// Inference threads per model when the machine's fastest cores can't be
/// detected: [`FALLBACK_INTRA_OP_THREADS`], or one less than its logical
/// CPUs if that's fewer, and at least 1.
pub const FALLBACK_INTRA_OP_THREADS: usize = 4;

/// An upper bound on the detected default, for machines with many cores.
/// Unset: the default is every fast physical core (D40).
pub const MAX_DEFAULT_INTRA_OP_THREADS: Option<usize> = None;

/// The machine's fastest tier of physical cores, and how they were found.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DetectedCores {
    /// `hw.perflevel0.physicalcpu` (Apple silicon's P-cores),
    /// `hw.physicalcpu` (an Intel Mac), `linux cpu_core` (a hybrid Intel
    /// CPU's P-cores), `linux topology` (physical cores), or `fallback`.
    pub source: &'static str,
    pub cores: usize,
}

/// The default intra-op threads on this machine (D40): one per physical core
/// of its fastest tier, so ONNX Runtime's threads don't share a core with
/// each other (SMT) or wait on an efficiency core. Never the logical CPU
/// count: on a 12-core, 24-thread Xeon, 24 threads ran 1.4-2.9x slower than
/// 8. If nothing can be detected, the fallback rule.
pub fn default_intra_op_threads() -> usize {
    detected_cores().cores
}

/// What [`default_intra_op_threads`] is based on, detected once.
pub fn detected_cores() -> DetectedCores {
    static DETECTED: std::sync::OnceLock<DetectedCores> = std::sync::OnceLock::new();
    *DETECTED.get_or_init(|| {
        let detected = detect().unwrap_or_else(|| {
            let cpus = std::thread::available_parallelism().map_or(1, |n| n.get());
            DetectedCores { source: "fallback", cores: FALLBACK_INTRA_OP_THREADS.min(cpus.saturating_sub(1)).max(1) }
        });
        let cores = MAX_DEFAULT_INTRA_OP_THREADS.map_or(detected.cores, |cap| detected.cores.min(cap));
        DetectedCores { cores, ..detected }
    })
}

#[cfg(target_os = "macos")]
fn detect() -> Option<DetectedCores> {
    let sysctl = |name: &str| -> Option<usize> {
        let out = std::process::Command::new("/usr/sbin/sysctl").args(["-n", name]).output().ok()?;
        if !out.status.success() {
            return None;
        }
        String::from_utf8_lossy(&out.stdout).trim().parse().ok().filter(|&n: &usize| n > 0)
    };
    // Apple silicon reports its performance cores as perflevel0; an Intel
    // Mac has no perflevels.
    match sysctl("hw.perflevel0.physicalcpu") {
        Some(cores) => Some(DetectedCores { source: "hw.perflevel0.physicalcpu", cores }),
        None => sysctl("hw.physicalcpu").map(|cores| DetectedCores { source: "hw.physicalcpu", cores }),
    }
}

#[cfg(target_os = "linux")]
fn detect() -> Option<DetectedCores> {
    linux::detect(std::path::Path::new("/"))
}

#[cfg(not(any(target_os = "macos", target_os = "linux")))]
fn detect() -> Option<DetectedCores> {
    None
}

/// Physical cores from sysfs, under `root` (`/` on a running system), so
/// tests can give it fixture trees.
#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
mod linux {
    use super::DetectedCores;
    use std::collections::BTreeSet;
    use std::path::Path;

    /// A hybrid Intel CPU's P-cores (`/sys/devices/cpu_core/cpus`), else
    /// every physical core; each physical core counted once, however many
    /// SMT threads it has.
    pub fn detect(root: &Path) -> Option<DetectedCores> {
        let cpu = root.join("sys/devices/system/cpu");
        if let Ok(list) = std::fs::read_to_string(root.join("sys/devices/cpu_core/cpus")) {
            if let Some(cores) = parse_cpu_list(list.trim()).and_then(|cpus| physical_cores(&cpu, &cpus)) {
                return Some(DetectedCores { source: "linux cpu_core", cores });
            }
        }
        let cpus: Vec<usize> = std::fs::read_dir(&cpu)
            .ok()?
            .filter_map(|e| e.ok()?.file_name().to_str()?.strip_prefix("cpu")?.parse().ok())
            .collect();
        physical_cores(&cpu, &cpus).map(|cores| DetectedCores { source: "linux topology", cores })
    }

    /// Unique (package, core) pairs among `cpus`; `None` if any CPU's
    /// topology can't be read.
    fn physical_cores(cpu: &Path, cpus: &[usize]) -> Option<usize> {
        let read = |n: usize, file: &str| -> Option<i64> {
            std::fs::read_to_string(cpu.join(format!("cpu{n}/topology/{file}"))).ok()?.trim().parse().ok()
        };
        let mut cores = BTreeSet::new();
        for &n in cpus {
            cores.insert((read(n, "physical_package_id")?, read(n, "core_id")?));
        }
        (!cores.is_empty()).then_some(cores.len())
    }

    /// A kernel CPU list: `0-7,16-23`.
    pub fn parse_cpu_list(list: &str) -> Option<Vec<usize>> {
        let mut cpus = Vec::new();
        for part in list.split(',').filter(|p| !p.is_empty()) {
            match part.split_once('-') {
                Some((a, b)) => cpus.extend(a.parse::<usize>().ok()?..=b.parse::<usize>().ok()?),
                None => cpus.push(part.parse().ok()?),
            }
        }
        Some(cpus)
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        /// A sysfs tree: each (cpu, package, core) gets its topology files.
        fn tree(tag: &str, cpus: &[(usize, i64, i64)], cpu_core: Option<&str>) -> std::path::PathBuf {
            let root = std::env::temp_dir().join(format!("sk-sysfs-{tag}-{}", std::process::id()));
            let _ = std::fs::remove_dir_all(&root);
            for &(n, package, core) in cpus {
                let t = root.join(format!("sys/devices/system/cpu/cpu{n}/topology"));
                std::fs::create_dir_all(&t).unwrap();
                std::fs::write(t.join("physical_package_id"), format!("{package}\n")).unwrap();
                std::fs::write(t.join("core_id"), format!("{core}\n")).unwrap();
            }
            // Entries that aren't CPUs.
            std::fs::create_dir_all(root.join("sys/devices/system/cpu/cpufreq")).unwrap();
            if let Some(list) = cpu_core {
                std::fs::create_dir_all(root.join("sys/devices/cpu_core")).unwrap();
                std::fs::write(root.join("sys/devices/cpu_core/cpus"), format!("{list}\n")).unwrap();
            }
            root
        }

        #[test]
        fn smt_siblings_count_once() {
            // 12 cores, 2 threads each: cpu n and n+12 share core n.
            let cpus: Vec<_> = (0..24).map(|n| (n, 0, (n % 12) as i64)).collect();
            let root = tree("smt", &cpus, None);
            assert_eq!(detect(&root), Some(DetectedCores { source: "linux topology", cores: 12 }));
            std::fs::remove_dir_all(&root).unwrap();
        }

        #[test]
        fn two_packages_count_their_cores_apart() {
            let cpus: Vec<_> = (0..8).map(|n| (n, (n / 4) as i64, (n % 4) as i64)).collect();
            let root = tree("packages", &cpus, None);
            assert_eq!(detect(&root).unwrap().cores, 8, "core ids repeat across packages");
            std::fs::remove_dir_all(&root).unwrap();
        }

        #[test]
        fn a_hybrid_cpu_counts_its_p_cores() {
            // 6 P-cores with SMT (cpus 0-11, two per core) and 8 E-cores
            // (cpus 12-19).
            let mut cpus: Vec<_> = (0..12).map(|n| (n, 0, (n / 2) as i64)).collect();
            cpus.extend((12..20).map(|n| (n, 0, 100 + n as i64)));
            let root = tree("hybrid", &cpus, Some("0-11"));
            assert_eq!(detect(&root), Some(DetectedCores { source: "linux cpu_core", cores: 6 }));
            std::fs::remove_dir_all(&root).unwrap();
        }

        #[test]
        fn missing_files_detect_nothing() {
            let root = tree("missing", &[], None);
            assert_eq!(detect(&root), None);
            // A CPU without its topology files.
            std::fs::create_dir_all(root.join("sys/devices/system/cpu/cpu0")).unwrap();
            assert_eq!(detect(&root), None);
            // An unreadable cpu_core list falls back to the topology.
            let root2 = tree("bad-list", &[(0, 0, 0), (1, 0, 1)], Some("x-y"));
            assert_eq!(detect(&root2), Some(DetectedCores { source: "linux topology", cores: 2 }));
            for r in [root, root2] {
                std::fs::remove_dir_all(&r).unwrap();
            }
        }

        #[test]
        fn cpu_lists_parse() {
            assert_eq!(parse_cpu_list("0-3,8,10-11"), Some(vec![0, 1, 2, 3, 8, 10, 11]));
            assert_eq!(parse_cpu_list(""), Some(vec![]));
            assert_eq!(parse_cpu_list("a"), None);
        }
    }
}

/// The loaded ONNX Runtime.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuntimeInfo {
    /// The version the library reports.
    pub version: String,
    /// Where it was loaded from.
    pub library: std::path::PathBuf,
    /// Who named it.
    pub source: RuntimeSource,
    /// Set when the loaded version isn't [`PINNED_VERSION`]'s minor.
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
pub fn load_runtime(configured: Option<&std::path::Path>) -> Result<RuntimeInfo, String> {
    RUNTIME.get_or_init(|| load(configured)).clone()
}

/// What [`load_runtime`] gave, or `None` if this process never loaded it
/// (no `onnx` model).
pub fn runtime_state() -> Option<Result<RuntimeInfo, String>> {
    RUNTIME.get().cloned()
}

#[cfg(not(feature = "onnx"))]
fn load(_: Option<&std::path::Path>) -> Result<RuntimeInfo, String> {
    Err("this build has no ONNX Runtime support (built without the `onnx` feature)".into())
}

#[cfg(feature = "onnx")]
use backend::load;
#[cfg(feature = "onnx")]
pub use backend::OnnxEmbedder;

#[cfg(feature = "onnx")]
mod backend {
    use super::{load_runtime, RuntimeInfo, RuntimeSource, PINNED_VERSION};
    use crate::embed_input::EmbedInput;
    use crate::pooling::{normalize_in_place, pool};
    use ort::session::builder::GraphOptimizationLevel;
    use ort::session::Session;
    use ort::value::{DynValue, Tensor, TensorElementType};
    use sidekick_core::manifest::ResolvedModel;
    use sidekick_core::{EmbedLimits, EmbedPurpose, Embedder, Error, Pooling, Result};
    use std::path::{Path, PathBuf};
    use std::sync::{Mutex, PoisonError};

    /// The runtime library's file name on this platform.
    #[cfg(target_os = "macos")]
    const LIBRARY: &str = "libonnxruntime.dylib";
    #[cfg(not(target_os = "macos"))]
    const LIBRARY: &str = "libonnxruntime.so";

    /// The configured library, else `ORT_DYLIB_PATH`'s, else the one beside
    /// the executable; loaded, and what loaded or why nothing did.
    pub(super) fn load(configured: Option<&Path>) -> std::result::Result<RuntimeInfo, String> {
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
        let warning = (!version.starts_with(&format!("{pinned_minor}."))).then(|| {
            format!("ONNX Runtime {version} loaded; sidekick is tested with {PINNED_VERSION}")
        });
        if let Some(w) = &warning {
            tracing::warn!("{w}");
        }
        tracing::info!(version = %version, library = %library.display(), "ONNX Runtime loaded");
        Ok(RuntimeInfo { version, library, source, warning })
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
            // The daemon loaded it at startup; tools load it here.
            load_runtime(None).map_err(Error::Inference)?;
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
            // Checked at scan, but the file may have changed since: an error
            // naming what the graph does return, not a panic.
            let output = outputs.get(self.output.as_str()).ok_or_else(|| {
                let names: Vec<&str> = outputs.keys().collect();
                Error::Inference(format!("model `{}`: the graph returned no `{}` (its outputs: {names:?})", self.id, self.output))
            })?;
            let (out_shape, data) = output.try_extract_tensor::<f32>().map_err(fail)?;
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
