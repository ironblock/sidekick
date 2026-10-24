//! The live suite (macOS, `--features coreml`): a parent that reads compute
//! plans and grades, and workers that each run one model on one compute path
//! in their own process. An Objective-C exception inside Core ML (on
//! macOS 27 a flexible-shape artifact aborts at predict under `.cpuOnly`)
//! or a stuck ANE compile then costs one cell of the report, not the whole
//! run.

#[cfg(all(target_os = "macos", feature = "coreml"))]
mod classify;

use crate::expect::{Expectations, Path3};
use crate::grade::{check_external, fmt, grade, suggest_floor, CaseResult, Check, PathGrade, WorkerResult};
use crate::metrics::cosine;
use crate::reference::{corpus_sha256, corpus_sha256_of, long_corpus_reaches, sha256_hex, Corpus, Reference};
#[cfg(all(target_os = "macos", feature = "coreml"))]
use crate::reference::Case;
use serde::Serialize;
use sidekick_core::manifest::{CpuSeqLimit, ModelRegistry, ResolvedClassifier, ResolvedModel};
use sidekick_core::{EmbedPurpose, Embedder, EmbeddingBackendKind};
#[cfg(all(target_os = "macos", feature = "coreml"))]
use sidekick_coreml::{compute_plan, ComputeUnits, PlanSummary};
#[cfg(all(target_os = "macos", feature = "coreml"))]
use sidekick_embed::CoremlEmbedder;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

const USAGE: &str = "\
usage: parity [options]
  --models-dir DIR     a models directory to scan (repeatable; default: the
                       daemon's, ~/Library/Application Support/sidekick/models)
  --refs DIR           references as DIR/<model id>/reference.* (default:
                       <model dir>/parity/), and reference-long.* for the
                       long-input corpus, graded for a model whose
                       max_seq_len reaches it
  --model ID           only this model (repeatable)
  --paths LIST         compute paths, comma-separated: cpu,gpu,ane (Core ML)
                       and onnx-cpu (default: all this build can run;
                       onnx-cpu alone without Core ML)
  --fixtures DIR       the repository's fixtures/ directory (corpora,
                       expectations); default: $SIDEKICK_FIXTURES, else
                       ./fixtures, else the checkout this was built from
  --expectations FILE  default: parity/expectations.toml in the fixtures
  --json FILE          also write the full report as JSON
  --suggest-floors     print [[floor]] entries for this chip from this run
  --allow-unverified-plans
                       grade a model whose compute plan Core ML can't produce
                       (a warning, not a failure), provided its ANE output
                       isn't bit-identical to CPU_ONLY
  --timeout SECS       per worker (default 1200)
  --ignore-ane-seq-cap grade a model's ane path past its ANE length cap
                       (2,048 tokens, or its manifest's ane_max_seq_len;
                       D48). By default the ane path loads no bucket past
                       it, reads no compute plan for one, and reports the
                       cases longer than it as capped, not graded
  --ignore-cpu-seq-cap grade a model's cpu path past the CPU length cap
                       (1,024 tokens, D33), for measurement. By default the
                       cpu path loads no bucket past it, as the runtime
                       serves a cpu_only model, and reports the cases
                       longer than it as capped, not graded; a manifest's
                       cpu_seq_limit = \"ignore\" lifts it, as it does for
                       the runtime
  --onnx-threads N     intra-op threads for onnx-cpu workers (default:
                       sidekick's default for the machine, D40)
  --external FILE      grade results written outside the suite, a worker's
                       JSON (model, path, cases, repeat_bitwise, load_ms)
                       for path onnx-cpu, as that model's onnx-cpu path
                       (repeatable; D40)
  --bucket-invariance MODE
                       classifiers: which larger buckets each case re-runs in
                       for bucket invariance. full: every larger bucket.
                       sampled: the next larger bucket for every case, every
                       larger bucket for the adversarial cases and every 10th
                       case. auto (default): sampled when the largest bucket
                       is over 512 tokens, else full";

/// The suite measures every compute path itself (its ANE plan check reports
/// a model Core ML won't place on the ANE), so it loads models past the ANE
/// weight cap too. It applies the length caps itself, each to its own path
/// ([`PathCaps`]): the CPU's (D33) to the cpu path and the ANE's (D48) to
/// the ane path, so the scan doesn't.
const MEASURE_ALL: sidekick_core::ScanOptions = sidekick_core::ScanOptions {
    compute_units: std::collections::BTreeMap::new(),
    ignore_ane_weight_cap: true,
    ignore_cpu_seq_cap: true,
    ignore_ane_seq_cap: true,
    no_neural_engine: false,
};

/// The ANE length cap the ane path runs under (D48), as the runtime serves
/// a model on the ANE: its worker loads no bucket past it, the suite reads
/// no compute plan for one, and the cases longer than it are reported as
/// capped rather than graded. `None` when the model is within it, or under
/// `--ignore-ane-seq-cap` (`ignore`).
pub fn ane_path_cap(ignore: bool, max_seq_len: usize, ane_max_seq_len: Option<usize>) -> Option<usize> {
    let cap = sidekick_core::ane_seq_cap(ane_max_seq_len);
    (!ignore && max_seq_len > cap).then_some(cap)
}

/// The CPU length cap the cpu path runs under (D33), as the runtime serves
/// a model with `cpu_only`: [`sidekick_core::MAX_CPU_INVARIANT_SEQ`], past
/// which Core ML's CPU sums attention in a length-dependent order, so a
/// model's buckets differ slightly. Its worker loads no bucket past it, and
/// the cases longer than it are reported as capped rather than graded.
/// `None` when the model is within it, when its manifest sets
/// `cpu_seq_limit = "ignore"` (the runtime serves every bucket then), or
/// under `--ignore-cpu-seq-cap` (`ignore`).
pub fn cpu_path_cap(ignore: bool, max_seq_len: usize, limit: CpuSeqLimit) -> Option<usize> {
    let cap = sidekick_core::MAX_CPU_INVARIANT_SEQ;
    (!ignore && limit == CpuSeqLimit::Enforce && max_seq_len > cap).then_some(cap)
}

/// The length cap each compute path runs under, as the runtime serves a
/// model there: the cpu path D33's ([`cpu_path_cap`]), the ane path D48's
/// ([`ane_path_cap`]); the gpu and onnx-cpu paths none. The suite grades
/// what sidekick serves on each path.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PathCaps {
    pub cpu: Option<usize>,
    pub ane: Option<usize>,
}

impl PathCaps {
    fn new(o: &Options, max_seq_len: usize, cpu_seq_limit: CpuSeqLimit, ane_max_seq_len: Option<usize>) -> Self {
        Self {
            cpu: cpu_path_cap(o.ignore_cpu_cap, max_seq_len, cpu_seq_limit),
            ane: ane_path_cap(o.ignore_ane_cap, max_seq_len, ane_max_seq_len),
        }
    }

    /// The cap `path` runs under, if any.
    pub fn of(&self, path: Path3) -> Option<usize> {
        match path {
            Path3::Cpu => self.cpu,
            Path3::Ane => self.ane,
            _ => None,
        }
    }

    /// Each capped path with its cap.
    pub fn iter(&self) -> impl Iterator<Item = (Path3, usize)> {
        [(Path3::Cpu, self.cpu), (Path3::Ane, self.ane)].into_iter().filter_map(|(p, c)| c.map(|c| (p, c)))
    }
}

/// A path's cap as the report names it: "the CPU cap of 1024 tokens (D33)".
pub fn cap_name(path: Path3, cap: usize) -> String {
    match path {
        Path3::Cpu => format!("the CPU cap of {cap} tokens (D33)"),
        _ => format!("the ANE cap of {cap} tokens (D48)"),
    }
}

/// Pairs of `a`'s cases and `b`'s with the same id, in `b`'s order: the
/// cases two paths both ran, when a capped path (D33, D48) ran fewer.
pub fn same_cases<'a, T>(a: &'a [T], b: &'a [T], id: impl Fn(&T) -> &str) -> Vec<(&'a T, &'a T)> {
    let by_id: std::collections::HashMap<&str, &T> = a.iter().map(|c| (id(c), c)).collect();
    b.iter().filter_map(|y| by_id.get(id(y)).map(|x| (*x, y))).collect()
}

/// Why a Core ML path or model can't run in this build.
#[cfg_attr(all(target_os = "macos", feature = "coreml"), allow(dead_code))]
const CORE_ML_BUILD: &str = "Core ML runs in a parity build on macOS with --features coreml";

/// Cases re-run in a second ANE process to check determinism across loads.
const DETERMINISM_CASES: usize = 8;

pub fn main() -> i32 {
    let args: Vec<String> = std::env::args().skip(1).collect();
    #[cfg(all(target_os = "macos", feature = "coreml"))]
    if args.first().map(String::as_str) == Some("--plan") {
        return match plan_child(&args[1..]) {
            Ok(()) => 0,
            Err(e) => {
                eprintln!("plan error: {e}");
                3
            }
        };
    }
    if args.first().map(String::as_str) == Some("--worker") {
        return match worker(&args[1..]) {
            Ok(()) => 0,
            Err(e) => {
                eprintln!("worker error: {e}");
                3
            }
        };
    }
    match parent(&args) {
        Ok(true) => 0,
        Ok(false) => 1,
        Err(e) => {
            eprintln!("parity: {e}");
            2
        }
    }
}

/// The Core ML compute units for a path; onnx-cpu isn't one.
#[cfg(all(target_os = "macos", feature = "coreml"))]
fn units(path: Path3) -> Result<ComputeUnits, String> {
    match path {
        Path3::Cpu => Ok(ComputeUnits::CpuOnly),
        Path3::Gpu => Ok(ComputeUnits::CpuAndGpu),
        Path3::Ane => Ok(ComputeUnits::CpuAndNeuralEngine),
        Path3::OnnxCpu => Err("a Core ML model runs on cpu, gpu or ane, not onnx-cpu".into()),
    }
}

fn reference_dir(model: &ResolvedModel, refs: Option<&Path>) -> PathBuf {
    match refs {
        Some(r) => r.join(&model.manifest.id),
        None => model.dir.join("parity"),
    }
}

fn load_model(models_dir: &Path, id: &str) -> Result<ResolvedModel, String> {
    let reg = ModelRegistry::scan_with(models_dir, &MEASURE_ALL).map_err(|e| e.to_string())?;
    reg.get(id).cloned().map_err(|e| e.to_string())
}

// ---- compute plans ---------------------------------------------------------

#[cfg(all(target_os = "macos", feature = "coreml"))]
/// `--plan <artifact> <out.json>`: one compute plan, read in its own
/// process so that a crash inside Core ML can't take the suite down.
fn plan_child(args: &[String]) -> Result<(), String> {
    let [artifact, out] = args else {
        return Err("bad plan arguments".into());
    };
    let plan = compute_plan(Path::new(artifact), ComputeUnits::CpuAndNeuralEngine)
        .map_err(|e| e.to_string())?;
    let json = serde_json::json!({
        "ane": plan.ane,
        "cpu": plan.cpu,
        "gpu": plan.gpu,
        "unassigned": plan.unassigned,
        "off_ane_ops": plan.off_ane_ops,
    });
    std::fs::write(out, json.to_string()).map_err(|e| e.to_string())
}

#[cfg(all(target_os = "macos", feature = "coreml"))]
/// Where Core ML caches compiled bundles for this executable.
fn bundle_cache() -> String {
    let exe = std::env::current_exe()
        .ok()
        .and_then(|p| p.file_name().map(|n| n.to_string_lossy().into_owned()))
        .unwrap_or_else(|| "parity".into());
    format!("~/Library/Caches/{exe}/com.apple.e5rt.e5bundlecache")
}

#[cfg(all(target_os = "macos", feature = "coreml"))]
/// One plan read in a child process. `Ok(None)`: Core ML produced no plan
/// (every operation unassigned, or "internal failure").
fn read_plan_once(
    artifact: &Path,
    timeout: Duration,
    scratch: &Path,
) -> Result<Option<PlanSummary>, String> {
    static N: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let n = N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let out = scratch.join(format!("plan-{n}.json"));
    let args: Vec<std::ffi::OsString> = vec!["--plan".into(), artifact.into(), out.clone().into()];
    match run_child(&args, &out, &out.with_extension("log"), timeout) {
        Ok(()) => {}
        Err(e) if e.contains("internal failure") => return Ok(None),
        Err(e) => return Err(e),
    }
    let v: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&out).map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?;
    let n = |k: &str| v[k].as_u64().unwrap_or(0) as usize;
    let plan = PlanSummary {
        ane: n("ane"),
        cpu: n("cpu"),
        gpu: n("gpu"),
        unassigned: n("unassigned"),
        off_ane_ops: serde_json::from_value(v["off_ane_ops"].clone()).unwrap_or_default(),
    };
    Ok((plan.assigned() > 0).then_some(plan))
}

#[cfg(all(target_os = "macos", feature = "coreml"))]
/// The compute plan for `.cpuAndNeuralEngine`, and whether it had to be
/// read from a copy.
///
/// Core ML caches compiled bundles per executable, keyed by artifact path.
/// A broken cache entry makes every plan for that path come back empty or
/// fail with "internal failure", deterministically, while the artifact is
/// fine: a copy at another path reads normally. So an unavailable plan is
/// read once more from an APFS clone at a fixed path (reused across runs,
/// so the cache doesn't grow). `Ok(None)` means it stayed unavailable.
fn read_plan(
    artifact: &Path,
    label: &str,
    timeout: Duration,
    scratch: &Path,
) -> Result<Option<(PlanSummary, bool)>, String> {
    if let Some(plan) = read_plan_once(artifact, timeout, scratch)? {
        return Ok(Some((plan, false)));
    }
    let clones = std::env::temp_dir().join("sidekick-parity-plans");
    std::fs::create_dir_all(&clones).map_err(|e| e.to_string())?;
    let clone = clones.join(format!("{label}.mlmodelc"));
    let _ = std::fs::remove_dir_all(&clone);
    let cloned = Command::new("cp")
        .arg("-Rc")
        .arg(artifact)
        .arg(&clone)
        .status()
        .map_err(|e| e.to_string())?;
    if !cloned.success() {
        return Ok(None);
    }
    Ok(read_plan_once(&clone, timeout, scratch)?.map(|p| (p, true)))
}

/// Run this binary with `args` as a child, logging to `log`, with a
/// timeout. `Err` describes a crash, a non-zero exit or a timeout.
fn run_child(
    args: &[std::ffi::OsString],
    out: &Path,
    log: &Path,
    timeout: Duration,
) -> Result<(), String> {
    let _ = std::fs::remove_file(out);
    let exe = std::env::current_exe().map_err(|e| e.to_string())?;
    let logf = std::fs::File::create(log).map_err(|e| e.to_string())?;
    let mut child = Command::new(exe)
        .args(args.iter())
        .stdout(Stdio::null())
        .stderr(logf)
        .spawn()
        .map_err(|e| e.to_string())?;
    let start = Instant::now();
    let status = loop {
        if let Some(s) = child.try_wait().map_err(|e| e.to_string())? {
            break s;
        }
        if start.elapsed() > timeout {
            let _ = child.kill();
            let _ = child.wait();
            return Err(format!("timed out after {}s", timeout.as_secs()));
        }
        std::thread::sleep(Duration::from_millis(100));
    };
    if status.success() {
        return Ok(());
    }
    use std::os::unix::process::ExitStatusExt;
    let lines: Vec<String> = std::fs::read_to_string(log)
        .unwrap_or_default()
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(String::from)
        .collect();
    let tail = lines[lines.len().saturating_sub(3)..].join(" | ");
    let how = match status.signal() {
        Some(sig) => format!("killed by signal {sig}"),
        None => format!("exit {}", status.code().unwrap_or(-1)),
    };
    Err(format!("{how}: {tail}"))
}

// ---- worker ----------------------------------------------------------------

/// A tiny deterministic generator for pad ids (no rand dependency).
struct XorShift(u64);

impl XorShift {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
}

/// One input's vector through the product path, or, when the product path
/// refuses the output as degenerate (all zeros or a non-finite value), the
/// raw output that it refused. The case is then graded as unusable output
/// (F) instead of the whole path failing. Any other error stands.
fn product_or_raw(
    embedded: sidekick_core::Result<Vec<Vec<f32>>>,
    raw: impl FnOnce() -> sidekick_core::Result<Vec<f32>>,
) -> Result<Vec<f32>, String> {
    match embedded {
        Ok(mut v) => Ok(v.remove(0)),
        // only the backends' refusal of a degenerate output: any other
        // failure fails the path
        Err(sidekick_core::Error::Inference(m)) if m.contains("which isn't an embedding") => match raw() {
            Ok(r) if sidekick_embed::degenerate(&r).is_some() => Ok(r),
            _ => Err(m),
        },
        Err(err) => Err(err.to_string()),
    }
}

fn bitwise_eq(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}

fn finite_cos(a: &[f32], b: &[f32]) -> Option<f64> {
    if a.iter().chain(b).all(|x| x.is_finite()) {
        cosine(a, b)
    } else {
        None
    }
}

/// The onnx-cpu worker (D40): the product path on ONNX Runtime's CPU, in the
/// Core ML worker's result shape, so `tools/onnx_parity.py`'s files and
/// this grade alike. Each input runs alone and unpadded, as the daemon runs
/// it, so there are no bucket or pad comparisons (n = 0).
#[cfg(feature = "onnx")]
fn onnx_worker(
    model: &ResolvedModel,
    reference: &Reference,
    id: &str,
    out: &str,
    limit: Option<usize>,
    threads: Option<usize>,
) -> Result<(), String> {
    use sidekick_core::EmbedLimits;
    use sidekick_embed::onnx::{default_intra_op_threads, OnnxEmbedder};
    let e = |e: sidekick_core::Error| e.to_string();
    let t0 = Instant::now();
    let emb = OnnxEmbedder::load(model, threads.unwrap_or_else(default_intra_op_threads)).map_err(e)?;
    let load_ms = t0.elapsed().as_secs_f64() * 1e3;
    let cases = &reference.cases[..limit.unwrap_or(reference.cases.len()).min(reference.cases.len())];
    let purpose = |p: &str| if p == "query" { EmbedPurpose::Query } else { EmbedPurpose::Document };
    let mut results = Vec::with_capacity(cases.len());
    let mut raw = Vec::with_capacity(cases.len());
    for case in cases {
        let ids = emb.prepare(&case.text, purpose(&case.purpose), EmbedLimits::default()).map_err(e)?;
        let t = Instant::now();
        let v = product_or_raw(emb.embed(&[case.text.as_str()], purpose(&case.purpose)), || emb.run(&ids))?;
        let ms = t.elapsed().as_secs_f64() * 1e3;
        let ids_match = ids == case.ids;
        let model_only = if ids_match { None } else { Some(emb.run(&case.ids).map_err(e)?) };
        let finite = v.iter().all(|x| x.is_finite());
        results.push(CaseResult {
            id: case.id.clone(),
            ids_match,
            bucket: ids.len(),
            vector: v.iter().map(|&x| if x.is_finite() { x } else { 0.0 }).collect(),
            finite,
            model_only,
            bucket_invariance: Check::default(),
            pad_invariance: Check::default(),
            ms,
        });
        raw.push(v);
    }
    let mut repeat_bitwise = true;
    for (case, first) in cases.iter().zip(&raw).take(5) {
        let ids = emb.prepare(&case.text, purpose(&case.purpose), EmbedLimits::default()).map_err(e)?;
        let again = product_or_raw(emb.embed(&[case.text.as_str()], purpose(&case.purpose)), || emb.run(&ids))?;
        repeat_bitwise &= bitwise_eq(first, &again);
    }
    let result = WorkerResult {
        model: id.to_string(),
        path: Path3::OnnxCpu.name().into(),
        cases: results,
        repeat_bitwise,
        load_ms,
    };
    std::fs::write(out, serde_json::to_vec(&result).map_err(|e| e.to_string())?).map_err(|e| e.to_string())
}

#[cfg(not(feature = "onnx"))]
fn onnx_worker(_: &ResolvedModel, _: &Reference, _: &str, _: &str, _: Option<usize>, _: Option<usize>) -> Result<(), String> {
    Err("this parity build has no onnx backend: build it with --features onnx".into())
}

/// `--worker <models-dir> <model-id> <refs-dir|-> <path> <out.json> [limit]`
fn worker(args: &[String]) -> Result<(), String> {
    let [models_dir, id, refs, path, out, rest @ ..] = args else {
        return Err("bad worker arguments".into());
    };
    // The optional rest: a case limit, `--bucket-invariance=MODE`,
    // `--corpus=long` (an embedder's long-input reference),
    // `--onnx-threads=N`, `--ignore-ane-seq-cap` and `--ignore-cpu-seq-cap`.
    let mut limit: Option<usize> = None;
    let mut bucket_mode = BucketInvariance::Auto;
    let mut corpus = Corpus::Main;
    let mut onnx_threads: Option<usize> = None;
    let mut ignore_ane_cap = false;
    let mut ignore_cpu_cap = false;
    for a in rest {
        if a == "--ignore-ane-seq-cap" {
            ignore_ane_cap = true;
        } else if a == "--ignore-cpu-seq-cap" {
            ignore_cpu_cap = true;
        } else if let Some(m) = a.strip_prefix("--bucket-invariance=") {
            bucket_mode = BucketInvariance::parse(m).ok_or("bad bucket-invariance mode")?;
        } else if let Some(c) = a.strip_prefix("--corpus=") {
            corpus = Corpus::parse(c).ok_or("bad corpus")?;
        } else if let Some(n) = a.strip_prefix("--onnx-threads=") {
            onnx_threads = Some(n.parse().map_err(|_| "bad onnx-threads")?);
        } else {
            limit = Some(a.parse().map_err(|_| "bad limit")?);
        }
    }
    let path = Path3::parse(path).ok_or("bad path")?;
    let refs = (refs != "-").then(|| PathBuf::from(refs));
    let registry = ModelRegistry::scan_with(Path::new(models_dir), &MEASURE_ALL).map_err(|e| e.to_string())?;
    if let Ok(model) = registry.classifier(id) {
        #[cfg(all(target_os = "macos", feature = "coreml"))]
        {
            let m = &model.manifest;
            let cap = match path {
                Path3::Cpu => cpu_path_cap(ignore_cpu_cap, m.max_seq_len, m.cpu_seq_limit),
                Path3::Ane => ane_path_cap(ignore_ane_cap, m.max_seq_len, m.ane_max_seq_len),
                _ => None,
            };
            return classify::worker(model, refs.as_deref(), path, out, limit, bucket_mode, cap);
        }
        #[cfg(not(all(target_os = "macos", feature = "coreml")))]
        return Err(format!("{} is a Core ML classifier ({}): {CORE_ML_BUILD}", model.manifest.id, bucket_mode.name()));
    }
    let model = load_model(Path::new(models_dir), id)?;
    let reference = Reference::load_corpus(&reference_dir(&model, refs.as_deref()), corpus)?;
    if model.manifest.backend == EmbeddingBackendKind::Onnx {
        if path != Path3::OnnxCpu {
            return Err(format!("an onnx model runs on onnx-cpu, not {}", path.name()));
        }
        return onnx_worker(&model, &reference, id, out, limit, onnx_threads);
    }
    let m = &model.manifest;
    let cap = match path {
        Path3::Cpu => cpu_path_cap(ignore_cpu_cap, m.max_seq_len, m.cpu_seq_limit.unwrap_or_default()),
        Path3::Ane => ane_path_cap(ignore_ane_cap, m.max_seq_len, m.ane_max_seq_len),
        _ => None,
    };
    coreml_worker(&model, &reference, id, path, out, limit, cap)
}

#[cfg(not(all(target_os = "macos", feature = "coreml")))]
fn coreml_worker(
    model: &ResolvedModel,
    _: &Reference,
    _: &str,
    _: Path3,
    _: &str,
    _: Option<usize>,
    _: Option<usize>,
) -> Result<(), String> {
    Err(format!("{} is a Core ML model: {CORE_ML_BUILD}", model.manifest.id))
}

/// `model` as the runtime serves it on `path` under a length cap of `cap`
/// tokens (D33 on the CPU, D48 on the ANE): the buckets past it dropped,
/// and the largest kept one its `max_seq_len`.
#[cfg(all(target_os = "macos", feature = "coreml"))]
fn capped_model(model: &ResolvedModel, path: Path3, cap: usize) -> Result<ResolvedModel, String> {
    let mut capped = model.clone();
    capped.manifest.buckets.retain(|&b| b <= cap);
    capped.manifest.max_seq_len =
        *capped.manifest.buckets.last().ok_or(format!("every bucket is past {}", cap_name(path, cap)))?;
    Ok(capped)
}

/// A Core ML embedder on one compute path. With `cap` (a path capped where
/// the runtime caps it: the cpu path by D33, the ane path by D48), only the
/// buckets up to it load, and only the cases whose reference ids fit it run.
#[cfg(all(target_os = "macos", feature = "coreml"))]
fn coreml_worker(
    model: &ResolvedModel,
    reference: &Reference,
    id: &str,
    path: Path3,
    out: &str,
    limit: Option<usize>,
    cap: Option<usize>,
) -> Result<(), String> {
    let vocab = tokenizers::Tokenizer::from_file(model.tokenizer_path())
        .map_err(|e| e.to_string())?
        .get_vocab_size(true) as u64;

    let capped;
    let model = match cap {
        Some(cap) => {
            capped = capped_model(model, path, cap)?;
            &capped
        }
        None => model,
    };
    let t0 = Instant::now();
    let emb = CoremlEmbedder::load_with(model, units(path)?).map_err(|e| e.to_string())?;
    let load_ms = t0.elapsed().as_secs_f64() * 1e3;
    let buckets = emb.buckets().to_vec();
    let cases: Vec<&Case> = reference.cases[..limit
        .unwrap_or(reference.cases.len())
        .min(reference.cases.len())]
        .iter()
        .filter(|c| cap.is_none_or(|cap| c.ids.len() <= cap))
        .collect();
    let full = limit.is_none();
    let mut rng = XorShift(0x5eed_1234_abcd_0001);
    let e = |e: sidekick_core::Error| e.to_string();

    let mut results = Vec::with_capacity(cases.len());
    let mut raw = Vec::with_capacity(cases.len());
    for &case in &cases {
        let purpose = if case.purpose == "query" {
            EmbedPurpose::Query
        } else {
            EmbedPurpose::Document
        };
        let prepared = emb.prepare(&case.text, purpose).map_err(e)?;
        let t = Instant::now();
        let v = product_or_raw(emb.embed(&[case.text.as_str()], purpose), || {
            emb.run(&prepared.ids, prepared.bucket)
        })?;
        let ms = t.elapsed().as_secs_f64() * 1e3;

        let ids_match = prepared.ids == case.ids;
        let model_only = if ids_match {
            None
        } else {
            let b = *buckets
                .iter()
                .find(|&&b| b >= case.ids.len())
                .ok_or("reference ids exceed buckets")?;
            Some(emb.run(&case.ids, b).map_err(e)?)
        };

        let mut bucket_invariance = Check::default();
        let mut pad_invariance = Check::default();
        if full {
            for &b in buckets.iter().filter(|&&b| b > prepared.bucket) {
                bucket_invariance.add(finite_cos(&v, &emb.run(&prepared.ids, b).map_err(e)?));
            }
            let pads = prepared.bucket - prepared.ids.len();
            if pads > 0 {
                let pad_ids: Vec<i32> = (0..pads).map(|_| (rng.next() % vocab) as i32).collect();
                let w = emb
                    .run_padded(&prepared.ids, prepared.bucket, &pad_ids)
                    .map_err(e)?;
                pad_invariance.add(finite_cos(&v, &w));
            }
        }

        let finite = v.iter().all(|x| x.is_finite());
        results.push(CaseResult {
            id: case.id.clone(),
            ids_match,
            bucket: prepared.bucket,
            vector: v
                .iter()
                .map(|&x| if x.is_finite() { x } else { 0.0 })
                .collect(),
            finite,
            model_only,
            bucket_invariance,
            pad_invariance,
            ms,
        });
        raw.push(v);
    }

    // After every bucket has run: the first cases again, bit for bit.
    let mut repeat_bitwise = true;
    for (&case, first) in cases.iter().zip(&raw).take(5) {
        let purpose = if case.purpose == "query" {
            EmbedPurpose::Query
        } else {
            EmbedPurpose::Document
        };
        let prepared = emb.prepare(&case.text, purpose).map_err(e)?;
        let again = product_or_raw(emb.embed(&[case.text.as_str()], purpose), || {
            emb.run(&prepared.ids, prepared.bucket)
        })?;
        repeat_bitwise &= bitwise_eq(first, &again);
    }

    let result = WorkerResult {
        model: id.to_string(),
        path: path.name().into(),
        cases: results,
        repeat_bitwise,
        load_ms,
    };
    std::fs::write(out, serde_json::to_vec(&result).map_err(|e| e.to_string())?)
        .map_err(|e| e.to_string())
}

// ---- parent ----------------------------------------------------------------

struct Options {
    models_dirs: Vec<PathBuf>,
    refs: Option<PathBuf>,
    models: Vec<String>,
    paths: Vec<Path3>,
    /// Whether `--paths` named them, rather than the default: a named path
    /// a model can't run is reported as skipped.
    paths_named: bool,
    /// `--expectations`; `parity/expectations.toml` in the fixtures by default.
    expectations: Option<PathBuf>,
    /// `--fixtures`.
    fixtures: Option<PathBuf>,
    json: Option<PathBuf>,
    suggest: bool,
    allow_unverified: bool,
    timeout: Duration,
    pub bucket_invariance: BucketInvariance,
    /// Results written outside the suite, graded as their own path
    /// (`--external`; D40).
    externals: Vec<PathBuf>,
    /// `--onnx-threads`: onnx-cpu workers' intra-op threads.
    onnx_threads: Option<usize>,
    /// `--ignore-ane-seq-cap`: grade the ane path past the ANE length cap
    /// (D48).
    pub ignore_ane_cap: bool,
    /// `--ignore-cpu-seq-cap`: grade the cpu path past the CPU length cap
    /// (D33).
    pub ignore_cpu_cap: bool,
}

/// Which larger buckets a classifier case re-runs in for bucket invariance
/// (`--bucket-invariance`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BucketInvariance {
    Auto,
    Full,
    Sampled,
}

impl BucketInvariance {
    pub fn name(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Full => "full",
            Self::Sampled => "sampled",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        [Self::Auto, Self::Full, Self::Sampled].into_iter().find(|m| m.name() == s)
    }
}

fn parse(args: &[String]) -> Result<Options, String> {
    let mut o = Options {
        models_dirs: Vec::new(),
        refs: None,
        models: Vec::new(),
        paths: default_paths(),
        paths_named: false,
        expectations: None,
        fixtures: None,
        json: None,
        suggest: false,
        allow_unverified: false,
        timeout: Duration::from_secs(1200),
        bucket_invariance: BucketInvariance::Auto,
        externals: Vec::new(),
        onnx_threads: None,
        ignore_ane_cap: false,
        ignore_cpu_cap: false,
    };
    let mut it = args.iter();
    while let Some(a) = it.next() {
        let mut val = || {
            it.next()
                .cloned()
                .ok_or(format!("{a} needs a value\n{USAGE}"))
        };
        match a.as_str() {
            "--models-dir" => o.models_dirs.push(val()?.into()),
            "--refs" => o.refs = Some(val()?.into()),
            "--model" => o.models.push(val()?),
            "--paths" => {
                o.paths = val()?
                    .split(',')
                    .map(|p| match Path3::parse(p.trim()) {
                        Some(Path3::OnnxCpu) => Ok(Path3::OnnxCpu),
                        Some(path) if !CORE_ML => Err(format!("{} is a Core ML path: {CORE_ML_BUILD}", path.name())),
                        Some(path) => Ok(path),
                        None => Err(format!("unknown path {p:?}")),
                    })
                    .collect::<Result<_, _>>()?;
                o.paths_named = true;
            }
            "--external" => o.externals.push(val()?.into()),
            "--onnx-threads" => {
                let n: usize = val()?.parse().map_err(|_| "bad --onnx-threads")?;
                if n == 0 {
                    return Err("--onnx-threads must be at least 1".into());
                }
                o.onnx_threads = Some(n);
            }
            "--expectations" => o.expectations = Some(val()?.into()),
            "--fixtures" => o.fixtures = Some(val()?.into()),
            "--json" => o.json = Some(val()?.into()),
            "--suggest-floors" => o.suggest = true,
            "--allow-unverified-plans" => o.allow_unverified = true,
            "--ignore-ane-seq-cap" => o.ignore_ane_cap = true,
            "--ignore-cpu-seq-cap" => o.ignore_cpu_cap = true,
            "--timeout" => {
                o.timeout = Duration::from_secs(val()?.parse().map_err(|_| "bad --timeout")?)
            }
            "--bucket-invariance" => {
                let v = val()?;
                o.bucket_invariance =
                    BucketInvariance::parse(&v).ok_or(format!("unknown --bucket-invariance {v:?}"))?
            }
            "-h" | "--help" => return Err(USAGE.into()),
            other => return Err(format!("unknown argument {other:?}\n{USAGE}")),
        }
    }
    if o.models_dirs.is_empty() {
        let home = std::env::var("HOME").map_err(|_| "HOME is not set")?;
        o.models_dirs
            .push(Path::new(&home).join("Library/Application Support/sidekick/models"));
    }
    Ok(o)
}

/// Whether this build runs Core ML paths.
const CORE_ML: bool = cfg!(all(target_os = "macos", feature = "coreml"));

/// The paths run without `--paths`: Core ML's three and onnx-cpu; onnx-cpu
/// alone in a build without Core ML. Each model runs those of them it can
/// ([`model_paths`]).
fn default_paths() -> Vec<Path3> {
    if CORE_ML {
        Path3::ALL.into_iter().chain([Path3::OnnxCpu]).collect()
    } else {
        vec![Path3::OnnxCpu]
    }
}

/// Whether this Mac has a Neural Engine (an Intel Mac doesn't): without
/// one, there's no ane path, and no ANE eligibility to gate on.
fn has_ane() -> bool {
    static ANE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    #[cfg(all(target_os = "macos", feature = "coreml"))]
    return *ANE.get_or_init(sidekick_coreml::has_neural_engine);
    #[cfg(not(all(target_os = "macos", feature = "coreml")))]
    return *ANE.get_or_init(|| false);
}

/// What a model runs on, which decides its paths.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Runtime {
    /// A Core ML embedder or classifier: cpu, gpu, and ane where this Mac
    /// has a Neural Engine (`ane`).
    CoreMl { ane: bool },
    /// An onnx model (D40): onnx-cpu, in a worker when this build has the
    /// backend (`native`), and from any `--external` results either way.
    Onnx { native: bool },
}

/// Of the paths asked for, those a model's workers run, and those it
/// can't, with why. An onnx model's onnx-cpu in a build without the
/// backend is neither: only `--external` results grade it.
fn model_paths(asked: &[Path3], runtime: Runtime) -> (Vec<Path3>, Vec<(Path3, &'static str)>) {
    let (mut run, mut skip) = (Vec::new(), Vec::new());
    for &path in asked {
        match (runtime, path) {
            (Runtime::CoreMl { .. }, Path3::OnnxCpu) => skip.push((path, "a Core ML model doesn't run on ONNX Runtime")),
            (Runtime::CoreMl { ane: false }, Path3::Ane) => skip.push((path, NO_ANE)),
            (Runtime::CoreMl { .. }, _) => run.push(path),
            (Runtime::Onnx { native: true }, Path3::OnnxCpu) => run.push(path),
            (Runtime::Onnx { native: false }, Path3::OnnxCpu) => {}
            (Runtime::Onnx { .. }, _) => skip.push((path, "an onnx model runs on onnx-cpu, not Core ML")),
        }
    }
    (run, skip)
}

/// Why a Core ML model's ane path is skipped on an Intel Mac.
const NO_ANE: &str = "this Mac has no Neural Engine";

/// A model's worker paths, printing each `--paths` named that it can't run,
/// and, named or not, an ane path skipped for want of a Neural Engine.
fn paths_for(o: &Options, runtime: Runtime) -> Vec<Path3> {
    let (run, skip) = model_paths(&o.paths, runtime);
    for (path, why) in skip {
        if o.paths_named || why == NO_ANE {
            println!("  {}: skipped ({why})", path.name());
        }
    }
    run
}

/// Whether a model's compute plans gate its run: only where it's served on
/// the ANE (`cpu_and_ne` or `all`) and this Mac has one. On an Intel Mac
/// every plan is off the ANE, and Core ML runs `cpu_and_ne` on the CPU, so
/// a plan there says nothing about whether the model runs.
#[cfg_attr(not(all(target_os = "macos", feature = "coreml")), allow(dead_code))]
fn plans_gate(served_on_ane: bool, has_ane: bool) -> bool {
    served_on_ane && has_ane
}

/// The chip, as floors name it, and the OS. On macOS, the CPU's brand
/// string (`Apple M1 Max`, or an Intel Mac's `Intel(R) Xeon(R) W-3235 …`)
/// and `macOS <version> (<build>)`; on Linux, /proc/cpuinfo's model name and
/// os-release's PRETTY_NAME with the kernel.
fn machine() -> (String, String) {
    if cfg!(target_os = "macos") {
        let os = format!("macOS {} ({})", sh("sw_vers", &["-productVersion"]), sh("sw_vers", &["-buildVersion"]));
        return (sh("sysctl", &["-n", "machdep.cpu.brand_string"]), os);
    }
    let cpuinfo = std::fs::read_to_string("/proc/cpuinfo").unwrap_or_default();
    let chip = cpuinfo
        .lines()
        .find_map(|l| l.strip_prefix("model name")?.split_once(':').map(|(_, v)| v.trim().to_string()))
        .unwrap_or_else(|| sh("uname", &["-m"]));
    let release = std::fs::read_to_string("/etc/os-release").unwrap_or_default();
    let name = release
        .lines()
        .find_map(|l| l.strip_prefix("PRETTY_NAME=").map(|v| v.trim_matches('"').to_string()))
        .unwrap_or_else(|| "Linux".into());
    (chip, format!("{name}, kernel {}", sh("uname", &["-r"])))
}

fn sh(cmd: &str, args: &[&str]) -> String {
    Command::new(cmd)
        .args(args)
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_default()
}

/// The full sha256 of a file, read in blocks (an external weights file can
/// be gigabytes).
fn file_sha256(path: &Path) -> Result<String, String> {
    use sha2::{Digest, Sha256};
    use std::io::Read;
    let mut f = std::fs::File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut h = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = f.read(&mut buf).map_err(|e| format!("{}: {e}", path.display()))?;
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
    }
    Ok(h.finalize().iter().map(|b| format!("{b:02x}")).collect())
}

/// The files an onnx model's grade depends on: its graph, the external
/// weights files the graph names, and its tokenizer, each by the path its
/// manifest gives it relative to the model directory (the ledger's paths),
/// sorted by path.
fn onnx_files(model: &ResolvedModel) -> Result<Vec<FileHash>, String> {
    let m = &model.manifest;
    let interface = sidekick_core::onnx_graph::read_interface(&model.artifact_path())?;
    let graph_dir = m.artifact.rsplit_once('/').map(|(d, _)| format!("{d}/")).unwrap_or_default();
    let mut paths = vec![m.artifact.clone(), m.tokenizer.clone()];
    for location in interface.external_data {
        let p = format!("{graph_dir}{location}");
        if !paths.contains(&p) {
            paths.push(p);
        }
    }
    paths.sort();
    paths
        .into_iter()
        .map(|rel| Ok(FileHash { sha256: file_sha256(&model.dir.join(&rel))?, path: rel }))
        .collect()
}

fn file_sha(path: &Path) -> Option<String> {
    std::fs::read(path)
        .ok()
        .map(|b| sha256_hex(&b)[..12].to_string())
}

#[derive(Serialize)]
struct BucketInfo {
    bucket: usize,
    model_mil: Option<String>,
    weights: Option<String>,
    /// `None` when the plan wasn't read (`not_read`).
    ane_ops: Option<usize>,
    assigned_ops: Option<usize>,
    /// The plan's verdict; `None` when it wasn't read (`not_read`).
    plan: Option<Result<(), String>>,
    /// Why the plan wasn't read: the bucket is past the ANE length cap of a
    /// model served on the ANE (D48), so the ANE never serves it, and
    /// reading its plan would compile it for the ANE.
    #[serde(skip_serializing_if = "Option::is_none")]
    not_read: Option<String>,
}

/// One file of the build a report graded, by its full sha256.
#[derive(Serialize, Debug, PartialEq)]
struct FileHash {
    /// Relative to the model directory, as the checksum ledger records it.
    path: String,
    sha256: String,
}

#[derive(Serialize)]
struct ModelReport {
    id: String,
    source: String,
    buckets: Vec<BucketInfo>,
    /// An onnx model's graph, its external weights files and its tokenizer,
    /// by full sha256: what `buckets` records of a Core ML build's programs,
    /// so the checksum ledger (`tools/ledger.py add --reports`, D50) can tie
    /// the grades to these exact files. Absent for a Core ML model.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    files: Vec<FileHash>,
    paths: Vec<PathGrade>,
    /// Paths whose worker failed to produce results, with why.
    crashed: Vec<(String, String)>,
    determinism: Option<Check>,
    failures: Vec<String>,
    regressions: Vec<String>,
    warnings: Vec<String>,
    /// The long-input corpus's grades, for a model whose max_seq_len
    /// reaches it and that has its reference. Absent otherwise, so other
    /// models' reports are unchanged by it.
    #[serde(skip_serializing_if = "Option::is_none")]
    long: Option<LongReport>,
}

/// The long-input corpus (fixtures/parity/corpus_long.toml), graded apart
/// from the main corpus: its own reference, cases and per-path grades. Its
/// gate failures are also the model's, prefixed `long corpus:`.
#[derive(Serialize)]
struct LongReport {
    corpus: &'static str,
    /// Each case, with the token count the reference pipeline fed it, in
    /// reference order (each path's `per_case` follows it).
    cases: Vec<LongCase>,
    /// The corpus's cases past the model's max_seq_len, left out.
    skipped: Vec<String>,
    paths: Vec<PathGrade>,
}

#[derive(Serialize)]
struct LongCase {
    id: String,
    tokens: usize,
}

#[derive(Serialize)]
struct Report {
    chip: String,
    macos: String,
    models: Vec<ModelReport>,
    #[cfg(all(target_os = "macos", feature = "coreml"))]
    classifiers: Vec<classify::ClassifierReport>,
}

/// What reading every bucket's compute plan found.
struct Plans {
    /// No bucket's plan failed its verdict.
    ok: bool,
    /// Buckets without a readable plan, allowed by --allow-unverified-plans.
    unverified: Vec<usize>,
    buckets: Vec<BucketInfo>,
}

#[cfg(all(target_os = "macos", feature = "coreml"))]
/// A chained bucket's plan (D37): its chunks' counts summed, read from a
/// copy if any chunk's was; `None` if any chunk has none.
fn read_chain_plan(
    artifacts: &[PathBuf],
    label: &str,
    timeout: Duration,
    scratch: &Path,
) -> Result<Option<(PlanSummary, bool)>, String> {
    if let [one] = artifacts {
        return read_plan(one, label, timeout, scratch);
    }
    let mut sum = PlanSummary { ane: 0, cpu: 0, gpu: 0, unassigned: 0, off_ane_ops: Default::default() };
    let mut copied = false;
    for (i, art) in artifacts.iter().enumerate() {
        let Some((p, c)) = read_plan(art, &format!("{label}.{i}"), timeout, scratch)? else { return Ok(None) };
        sum.ane += p.ane;
        sum.cpu += p.cpu;
        sum.gpu += p.gpu;
        sum.unassigned += p.unassigned;
        for (op, n) in p.off_ane_ops {
            *sum.off_ane_ops.entry(op).or_default() += n;
        }
        copied |= c;
    }
    Ok(Some((sum, copied)))
}

#[cfg(all(target_os = "macos", feature = "coreml"))]
/// The files' hashes, one per chunk, joined with `+`.
fn chain_sha(artifacts: &[PathBuf], file: &str) -> Option<String> {
    let shas: Option<Vec<String>> = artifacts.iter().map(|a| file_sha(&a.join(file))).collect();
    shas.map(|s| s.join("+"))
}

#[cfg(all(target_os = "macos", feature = "coreml"))]
/// Read and print every bucket's compute plan, before anything predicts.
/// The plan is a gate only for a model served on the ANE (`units`
/// `cpu_and_ne` or `all`), as the converters gate the served path: there an
/// ineligible plan means the model doesn't run where it's served. A model
/// served elsewhere (`cpu_and_gpu`, `cpu_only`) has its plan reported, an
/// ineligible or unreadable one as a warning, and every path still runs,
/// the ANE path reporting what Core ML actually did. (A multi-shape
/// artifact, which could abort a prediction, is refused at load by D27's
/// guard, whatever the plan.) For a model served on the ANE, a bucket past
/// `ane_cap` (D48) has no plan read: the runtime never serves it there, and
/// reading one compiles it for the ANE, which took half an hour for a
/// 4,096-token one.
#[allow(clippy::too_many_arguments)]
fn check_plans(
    id: &str,
    buckets: &[usize],
    artifacts_for: impl Fn(usize) -> Vec<PathBuf>,
    units: ComputeUnits,
    ane_cap: Option<usize>,
    o: &Options,
    scratch: &Path,
    failures: &mut Vec<String>,
    warnings: &mut Vec<String>,
) -> Plans {
    let read = |arts: &[PathBuf], label: &str| read_chain_plan(arts, label, o.timeout, scratch);
    check_plans_with(id, buckets, artifacts_for, units, ane_cap, o, failures, warnings, read)
}

#[cfg(all(target_os = "macos", feature = "coreml"))]
/// [`check_plans`], reading each bucket's plan with `read` (its artifacts
/// and a label), so a test can see which buckets' plans are read.
#[allow(clippy::too_many_arguments)]
fn check_plans_with(
    id: &str,
    buckets: &[usize],
    artifacts_for: impl Fn(usize) -> Vec<PathBuf>,
    units: ComputeUnits,
    ane_cap: Option<usize>,
    o: &Options,
    failures: &mut Vec<String>,
    warnings: &mut Vec<String>,
    mut read: impl FnMut(&[PathBuf], &str) -> Result<Option<(PlanSummary, bool)>, String>,
) -> Plans {
    let ane_here = has_ane();
    let served_on_ane = matches!(units, ComputeUnits::CpuAndNeuralEngine | ComputeUnits::All);
    let gated = plans_gate(served_on_ane, ane_here);
    let mut plans = Plans { ok: true, unverified: Vec::new(), buckets: Vec::new() };
    for &b in buckets {
        let arts = artifacts_for(b);
        if let Some(cap) = ane_cap.filter(|&cap| served_on_ane && b > cap) {
            let why = format!("not read (past the ANE cap of {cap} tokens, D48)");
            println!(
                "  bucket {b:>4}: plan {why}; the ane path doesn't run it  model.mil {} weights {}",
                chain_sha(&arts, "model.mil").unwrap_or("-".into()),
                chain_sha(&arts, "weights/weight.bin").unwrap_or("-".into()),
            );
            plans.buckets.push(BucketInfo {
                bucket: b,
                model_mil: chain_sha(&arts, "model.mil"),
                weights: chain_sha(&arts, "weights/weight.bin"),
                ane_ops: None,
                assigned_ops: None,
                plan: None,
                not_read: Some(why),
            });
            continue;
        }
        let plan = read(&arts, &format!("{id}-{b}"));
        let (ane, assigned, verdict) = match &plan {
            Ok(Some((p, _))) => (p.ane, p.assigned(), p.verdict()),
            Ok(None) => (0, 0, Ok(())),
            Err(e) => (0, 0, Err(format!("could not read the compute plan: {e}"))),
        };
        println!(
            "  bucket {b:>4}: {}  model.mil {} weights {}",
            match (&plan, &verdict) {
                (Ok(None), _) => "NO COMPUTE PLAN: eligibility unverified".to_string(),
                (Ok(Some((_, true))), Ok(())) => {
                    format!("ANE ops {ane}/{assigned} eligible (read from a copy)")
                }
                (_, Ok(())) => format!("ANE ops {ane}/{assigned} eligible"),
                (_, Err(e)) if gated => format!("ANE ops {ane}/{assigned} NOT ELIGIBLE: {e}"),
                (_, Err(_)) if !ane_here => {
                    format!("ANE ops {ane}/{assigned} (no Neural Engine on this Mac: reported, not gated)")
                }
                (_, Err(e)) => format!(
                    "ANE ops {ane}/{assigned} not ANE-eligible ({e}); served on {}, so reported, not gated",
                    units.name()
                ),
            },
            chain_sha(&arts, "model.mil").unwrap_or("-".into()),
            chain_sha(&arts, "weights/weight.bin").unwrap_or("-".into()),
        );
        if matches!(plan, Ok(Some((_, true)))) {
            warnings.push(format!(
                "bucket {b}: its compute plan read only from a copy of the artifact; \
                 Core ML's cache entry for its path looks broken ({}, safe to delete)",
                bundle_cache()
            ));
        }
        if matches!(plan, Ok(None)) {
            let why = format!(
                "bucket {b}: no compute plan, from its path or a fresh copy (every operation \
                 unassigned, or Core ML \"internal failure\"). Core ML's bundle cache for \
                 this executable may hold broken entries: {} (safe to delete). Check the \
                 bucket with ane_check, or pass --allow-unverified-plans",
                bundle_cache()
            );
            if o.allow_unverified || !gated {
                warnings.push(why);
                plans.unverified.push(b);
            } else {
                failures.push(why);
            }
        }
        match &verdict {
            Err(e) if gated => {
                plans.ok = false;
                failures.push(format!("bucket {b}: compute plan: {e}"));
            }
            Err(_) if !ane_here => {}
            Err(e) => warnings.push(format!(
                "bucket {b}: not ANE-eligible ({e}); the model is served on {}, so its plan isn't gated",
                units.name()
            )),
            Ok(()) => {}
        }
        plans.buckets.push(BucketInfo {
            bucket: b,
            model_mil: chain_sha(&arts, "model.mil"),
            weights: chain_sha(&arts, "weights/weight.bin"),
            ane_ops: Some(ane),
            assigned_ops: Some(assigned),
            plan: Some(verdict),
            not_read: None,
        });
    }
    plans
}

/// Run a worker in its own process, with a timeout, on the main corpus.
fn spawn_worker<T: serde::de::DeserializeOwned>(
    id: &str,
    models_dir: &Path,
    path: Path3,
    limit: Option<usize>,
    scratch: &Path,
    o: &Options,
) -> Result<T, String> {
    spawn_corpus_worker(id, models_dir, path, limit, Corpus::Main, scratch, o)
}

/// Run a worker in its own process, with a timeout, on `corpus`'s
/// reference.
fn spawn_corpus_worker<T: serde::de::DeserializeOwned>(
    id: &str,
    models_dir: &Path,
    path: Path3,
    limit: Option<usize>,
    corpus: Corpus,
    scratch: &Path,
    o: &Options,
) -> Result<T, String> {
    let (refs, timeout) = (o.refs.as_deref(), o.timeout);
    let tag = format!(
        "{id}-{}-{}{}",
        path.name(),
        limit.map_or("full".into(), |l| l.to_string()),
        if corpus == Corpus::Main { String::new() } else { format!("-{}", corpus.name()) }
    );
    let out = scratch.join(format!("{tag}.json"));
    let log = scratch.join(format!("{tag}.log"));
    let mut args: Vec<std::ffi::OsString> = vec![
        "--worker".into(),
        models_dir.into(),
        id.into(),
        refs.map_or("-".into(), |r| r.as_os_str().to_owned()),
        path.name().into(),
        out.clone().into(),
    ];
    if let Some(l) = limit {
        args.push(l.to_string().into());
    }
    args.push(format!("--bucket-invariance={}", o.bucket_invariance.name()).into());
    if corpus != Corpus::Main {
        args.push(format!("--corpus={}", corpus.name()).into());
    }
    if let Some(n) = o.onnx_threads {
        args.push(format!("--onnx-threads={n}").into());
    }
    if o.ignore_ane_cap {
        args.push("--ignore-ane-seq-cap".into());
    }
    if o.ignore_cpu_cap {
        args.push("--ignore-cpu-seq-cap".into());
    }
    run_child(&args, &out, &log, timeout)?;
    let bytes = std::fs::read(&out).map_err(|e| e.to_string())?;
    serde_json::from_slice(&bytes).map_err(|e| e.to_string())
}

fn parent(args: &[String]) -> Result<bool, String> {
    let o = parse(args)?;
    // Before anything reads a fixture; workers inherit it.
    crate::fixtures::init(o.fixtures.as_deref())?;
    let expectations = Expectations::load(&o.expectations.clone().unwrap_or_else(Expectations::default_path))?;
    let (chip, macos) = machine();
    let corpus = corpus_sha256()?;
    let scratch = std::env::temp_dir().join(format!("sidekick-parity-{}", std::process::id()));
    std::fs::create_dir_all(&scratch).map_err(|e| e.to_string())?;

    // Results written outside the suite (`--external`), by model id.
    let mut externals: Vec<WorkerResult> = Vec::new();
    for file in &o.externals {
        let text = std::fs::read_to_string(file).map_err(|e| format!("{}: {e}", file.display()))?;
        let r: WorkerResult = serde_json::from_str(&text).map_err(|e| format!("{}: {e}", file.display()))?;
        externals.push(r);
    }

    let mut models: Vec<(PathBuf, ResolvedModel)> = Vec::new();
    // Core ML models (and classifiers) this build can't run.
    let mut skipped_core_ml: Vec<String> = Vec::new();
    for dir in &o.models_dirs {
        let reg = ModelRegistry::scan_with(dir, &MEASURE_ALL).map_err(|e| e.to_string())?;
        for m in reg.iter() {
            let wanted = o.models.is_empty() || o.models.contains(&m.manifest.id);
            match m.manifest.backend {
                EmbeddingBackendKind::Coreml if wanted && !CORE_ML => skipped_core_ml.push(m.manifest.id.clone()),
                EmbeddingBackendKind::Coreml | EmbeddingBackendKind::Onnx if wanted => models.push((dir.clone(), m.clone())),
                _ => {}
            }
        }
    }
    let mut classifiers: Vec<(PathBuf, ResolvedClassifier)> = Vec::new();
    for dir in &o.models_dirs {
        let reg = ModelRegistry::scan_with(dir, &MEASURE_ALL).map_err(|e| e.to_string())?;
        for c in reg.classifiers() {
            if o.models.is_empty() || o.models.contains(&c.manifest.id) {
                // Classifiers are Core ML models.
                if CORE_ML {
                    classifiers.push((dir.clone(), c.clone()));
                } else {
                    skipped_core_ml.push(c.manifest.id.clone());
                }
            }
        }
    }
    for r in &externals {
        if !models.iter().any(|(_, m)| m.manifest.id == r.model) {
            return Err(format!("--external results for model {}, which no --models-dir has", r.model));
        }
    }
    for want in &o.models {
        if !models.iter().any(|(_, m)| &m.manifest.id == want)
            && !classifiers.iter().any(|(_, c)| &c.manifest.id == want)
            && !skipped_core_ml.contains(want)
        {
            return Err(format!("model {want} not found in any --models-dir"));
        }
    }
    let has_floors = expectations.floor.iter().any(|f| f.chip == chip);
    println!(
        "parity suite: {chip}, {macos}, {} model(s)",
        models.len() + classifiers.len()
    );
    if !skipped_core_ml.is_empty() {
        println!("skipped {} Core ML model(s) ({}): {CORE_ML_BUILD}", skipped_core_ml.len(), skipped_core_ml.join(", "));
    }
    if !has_floors {
        println!("no floors recorded for this chip: enforcing gates only, reporting accuracy");
    }

    let mut report = Report {
        chip: chip.clone(),
        macos,
        models: Vec::new(),
        #[cfg(all(target_os = "macos", feature = "coreml"))]
        classifiers: Vec::new(),
    };
    let mut ok = true;
    let mut graded = 0usize;
    // Models none of `--paths` runs on.
    let mut unrunnable = 0usize;
    let mut suggestions = Vec::new();
    for (models_dir, model) in &models {
        let id = &model.manifest.id;
        let refs_dir = reference_dir(model, o.refs.as_deref());
        println!("\n=== {id}");
        // An onnx model runs on ONNX Runtime's CPU (D40): natively, in a
        // build with the onnx feature, and from any --external results,
        // graded as a cross-check.
        let onnx = model.manifest.backend == EmbeddingBackendKind::Onnx;
        let native_onnx = onnx && sidekick_embed::onnx::SUPPORTED;
        let external: Vec<&WorkerResult> = externals.iter().filter(|r| &r.model == id).collect();
        if onnx && !native_onnx && external.is_empty() {
            println!("  SKIPPED: an onnx model needs a parity build with --features onnx, or --external results");
            continue;
        }
        let mut worker_paths = paths_for(&o, if onnx { Runtime::Onnx { native: native_onnx } } else { Runtime::CoreMl { ane: has_ane() } });
        // The length caps (D33 on the cpu path, D48 on the ane path): a
        // capped path runs the buckets and cases within its cap, or nothing
        // when no bucket is. An onnx model runs on neither path.
        let m = &model.manifest;
        let caps = if onnx {
            PathCaps::default()
        } else {
            PathCaps::new(&o, m.max_seq_len, m.cpu_seq_limit.unwrap_or_default(), m.ane_max_seq_len)
        };
        for (path, cap) in caps.iter() {
            if !m.buckets.iter().any(|&b| b <= cap) && worker_paths.contains(&path) {
                println!("  {}: skipped (every bucket is past {})", path.name(), cap_name(path, cap));
                worker_paths.retain(|&p| p != path);
            }
        }
        if worker_paths.is_empty() && external.is_empty() {
            println!("  SKIPPED: none of --paths runs on this model");
            unrunnable += 1;
            continue;
        }
        let reference = match Reference::load(&refs_dir) {
            Ok(r) => r,
            Err(e) if !refs_dir.join("reference.json").exists() && o.models.is_empty() => {
                println!(
                    "  SKIPPED: no reference ({e}); generate one with tools/parity_reference.py"
                );
                continue;
            }
            Err(e) => {
                println!("  FAIL: unusable reference: {e}");
                ok = false;
                continue;
            }
        };
        graded += 1;
        let tok_sha = std::fs::read(model.tokenizer_path())
            .map(|b| sha256_hex(&b))
            .unwrap_or_default();
        let mut mr = ModelReport {
            id: id.clone(),
            source: reference.source_label(),
            buckets: Vec::new(),
            files: Vec::new(),
            paths: Vec::new(),
            crashed: Vec::new(),
            determinism: None,
            failures: Vec::new(),
            regressions: Vec::new(),
            warnings: Vec::new(),
            long: None,
        };
        if let Some(why) = reference.stale(&model.manifest, &corpus, &tok_sha) {
            println!("  FAIL: {why}");
            mr.failures.push(why);
            report.models.push(mr);
            ok = false;
            continue;
        }
        let version = |p: &str| {
            reference
                .versions
                .get(p)
                .cloned()
                .flatten()
                .unwrap_or("?".into())
        };
        println!(
            "  reference: {} ({} pooling; prompts {}), {} cases, oracles {}; torch {}, sentence-transformers {}",
            reference.source_label(),
            reference.pooling,
            reference.prompt_source.values().cloned().collect::<Vec<_>>().join("/"),
            reference.cases.len(),
            reference.oracles.join(", "),
            version("torch"),
            version("sentence-transformers"),
        );

        // Compute plans first, on every bucket, before anything predicts:
        // an artifact the ANE rejects can abort the process at predict. An
        // onnx model has none.
        let no_plans = || Plans { ok: true, unverified: Vec::new(), buckets: Vec::new() };
        #[cfg(all(target_os = "macos", feature = "coreml"))]
        let plans = if onnx {
            no_plans()
        } else {
            check_plans(
                id,
                &model.manifest.buckets,
                |b| model.artifact_paths_for_bucket(b),
                model.manifest.compute_units.unwrap_or_default(),
                caps.ane,
                &o,
                &scratch,
                &mut mr.failures,
                &mut mr.warnings,
            )
        };
        // Without Core ML, only onnx models get here.
        #[cfg(not(all(target_os = "macos", feature = "coreml")))]
        let plans = no_plans();
        mr.buckets = plans.buckets;
        if onnx {
            match onnx_files(model) {
                Ok(files) => {
                    for f in &files {
                        println!("  {}: sha256 {}", f.path, f.sha256);
                    }
                    mr.files = files;
                }
                Err(e) => mr.warnings.push(format!("couldn't hash the model's files: {e}")),
            }
        }
        let (plan_ok, unverified) = (plans.ok, plans.unverified);
        if !plan_ok {
            println!("  FAIL: not ANE-eligible; skipping predictions (such an artifact can abort at predict)");
            ok = false;
            report.models.push(mr);
            continue;
        }

        // What a capped path is graded against: the cases within its cap.
        let capped_refs = capped_references(&caps, &reference);
        for (path, cap, (sub, _)) in &capped_refs {
            if !sub.has_core_case() && worker_paths.contains(path) {
                println!("  {}: skipped (no case within {})", path.name(), cap_name(*path, *cap));
                worker_paths.retain(|p| p != path);
            }
        }

        let mut results: Vec<(Path3, WorkerResult)> = Vec::new();
        for &path in &worker_paths {
            match spawn_worker::<WorkerResult>(id, models_dir, path, None, &scratch, &o) {
                Ok(r) => results.push((path, r)),
                Err(e) => {
                    println!("  {}: CRASHED: {e}", path.name());
                    mr.crashed.push((path.name().into(), e.clone()));
                    mr.failures
                        .push(format!("{}: worker failed: {e}", path.name()));
                }
            }
        }

        let natives = results.len();
        for r in &external {
            match check_external(&reference, r) {
                Ok(()) => results.push((Path3::OnnxCpu, (*r).clone())),
                Err(e) => mr.failures.push(format!("onnx-cpu: unusable --external results: {e}")),
            }
        }

        // Determinism across processes: a second, independent ANE load.
        if let Some((_, first)) = results.iter().find(|(p, _)| *p == Path3::Ane) {
            match spawn_worker::<WorkerResult>(id, models_dir, Path3::Ane, Some(DETERMINISM_CASES), &scratch, &o) {
                Ok(second) => {
                    let mut c = Check::default();
                    for (a, b) in first.cases.iter().zip(&second.cases) {
                        c.add(if a.finite && b.finite {
                            cosine(&a.vector, &b.vector)
                        } else {
                            None
                        });
                    }
                    if !c.min.is_some_and(|m| m >= expectations.gates.determinism) {
                        mr.failures.push(format!(
                            "ANE output differs between processes: {} < {}",
                            fmt(c.min),
                            expectations.gates.determinism
                        ));
                    }
                    mr.determinism = Some(c);
                }
                Err(e) => mr
                    .failures
                    .push(format!("ane (second process): worker failed: {e}")),
            }
        }

        let grades: Vec<PathGrade> = results
            .iter()
            .map(|(p, r)| {
                let floor = expectations.floor(&chip, id, *p);
                let (graded_on, capped) = path_reference(*p, &reference, &capped_refs);
                let mut g = grade(
                    graded_on,
                    r,
                    *p,
                    &expectations.gates,
                    &expectations.grades,
                    floor,
                    &model.manifest.matryoshka,
                );
                g.capped = capped;
                g
            })
            .enumerate()
            // Results written outside the suite are a cross-check, named apart.
            .map(|(i, mut g)| {
                if i >= natives {
                    g.path = format!("{}:external", g.path);
                }
                g
            })
            .collect();

        // A bucket graded without a compute plan must show it ran somewhere
        // other than the CPU: bit-identical ANE and CPU_ONLY output means
        // the "ANE" path fell back to the CPU.
        // Paired by case: a capped path (D33, D48) runs fewer cases.
        let cpu = results.iter().find(|(p, _)| *p == Path3::Cpu);
        let ane = results.iter().find(|(p, _)| *p == Path3::Ane);
        let both = |c: &WorkerResult, a: &WorkerResult| -> Vec<(CaseResult, CaseResult)> {
            same_cases(&c.cases, &a.cases, |x: &CaseResult| x.id.as_str())
                .into_iter()
                .map(|(x, y)| (x.clone(), y.clone()))
                .collect()
        };
        for &b in &unverified {
            let (Some((_, c)), Some((_, a))) = (cpu, ane) else {
                mr.failures.push(format!(
                    "bucket {b}: without a compute plan, grading needs both the cpu and ane paths \
                     to rule out a CPU fallback"
                ));
                continue;
            };
            let in_bucket: Vec<(CaseResult, CaseResult)> =
                both(c, a).into_iter().filter(|(_, y)| y.bucket == b).collect();
            if in_bucket.is_empty() {
                mr.failures.push(format!(
                    "bucket {b}: no compute plan and no case to check it with"
                ));
            } else if in_bucket
                .iter()
                .all(|(x, y)| bitwise_eq(&y.vector, &x.vector))
            {
                mr.failures.push(format!(
                    "bucket {b}: no compute plan, and its ANE output is bit-identical to CPU_ONLY: \
                     it runs on the CPU"
                ));
            }
        }

        // A full runtime fallback to the CPU would make the ANE path look
        // perfect. Only a complete fallback is visible this way.
        if let (Some((_, c)), Some((_, a))) = (cpu, ane) {
            let pairs = both(c, a);
            if pairs.iter().all(|(x, y)| bitwise_eq(&x.vector, &y.vector)) {
                mr.warnings.push("ANE output is bit-identical to CPU_ONLY: the model may be running entirely on the CPU".into());
            }
            let med = |mut ms: Vec<f64>| {
                ms.sort_by(f64::total_cmp);
                ms.get(ms.len() / 2).copied().unwrap_or(f64::NAN)
            };
            let ratio = med(pairs.iter().map(|(x, _)| x.ms).collect()) / med(pairs.iter().map(|(_, y)| y.ms).collect());
            if ratio < 1.1 {
                mr.warnings.push(format!(
                    "CPU_ONLY/ANE median latency ratio {ratio:.2}x < 1.1x: possible runtime fallback (or a loaded machine)"
                ));
            }
        }

        print_grades(&reference, &grades, &capped_refs, mr.determinism.as_ref());
        for g in &grades {
            mr.failures
                .extend(g.failures.iter().map(|f| format!("{}: {f}", g.path)));
            mr.regressions
                .extend(g.regressions.iter().map(|f| format!("{}: {f}", g.path)));
            if o.suggest {
                if let Some(s) = suggest_floor(&chip, id, g) {
                    suggestions.push(s);
                }
            }
        }
        let main_longest = reference.cases.iter().map(|c| c.ids.len()).max().unwrap_or(0);
        mr.long = grade_long(model, models_dir, &worker_paths, caps, main_longest, &tok_sha, &expectations, &o, &scratch, &mut mr);
        for f in &mr.failures {
            println!("  FAIL: {f}");
        }
        for f in &mr.regressions {
            println!("  REGRESSION: {f}");
        }
        for w in &mr.warnings {
            println!("  warning: {w}");
        }
        if mr.failures.is_empty() && mr.regressions.is_empty() {
            println!("  PASS");
        } else {
            ok = false;
        }
        mr.paths = grades;
        report.models.push(mr);
    }

    #[cfg(all(target_os = "macos", feature = "coreml"))]
    for (models_dir, model) in &classifiers {
        unrunnable += usize::from(model_paths(&o.paths, Runtime::CoreMl { ane: has_ane() }).0.is_empty());
        if let Some((mr, pass)) = classify::grade_model(model, models_dir, &o, &expectations, &scratch) {
            graded += 1;
            ok &= pass;
            report.classifiers.push(mr);
        }
    }

    if o.suggest {
        println!("\n# floors for {chip}, from this run:\n");
        for s in &suggestions {
            println!("{s}");
        }
    }
    if let Some(json) = &o.json {
        std::fs::write(
            json,
            serde_json::to_vec_pretty(&report).map_err(|e| e.to_string())?,
        )
        .map_err(|e| format!("{}: {e}", json.display()))?;
    }
    if graded == 0 {
        if models.is_empty() && classifiers.is_empty() && !skipped_core_ml.is_empty() {
            println!("\nFAIL: nothing was graded (every model needs Core ML)");
        } else if unrunnable > 0 {
            let paths: Vec<&str> = o.paths.iter().map(|p| p.name()).collect();
            println!("\nFAIL: nothing was graded (no model runs on --paths {})", paths.join(","));
        } else {
            println!("\nFAIL: nothing was graded (no model had a reference)");
        }
        ok = false;
    }
    let _ = std::fs::remove_dir_all(&scratch);
    println!("\n{}", if ok { "parity: PASS" } else { "parity: FAIL" });
    Ok(ok)
}

/// Whether a model without a long-input reference fails, rather than
/// being warned: when it was named (`--model`, as a missing main reference
/// fails then too) and its main corpus stops short of its max_seq_len, so
/// nothing else grades its longest inputs (an onnx build whose limit is
/// above the main corpus's 700-token over-length cases). A model whose
/// main corpus already reaches its limit (EmbeddingGemma-2's 2,048, which
/// the over-length cases fill) is only warned.
fn long_reference_required(named: bool, max_seq_len: usize, main_longest: usize) -> bool {
    named && main_longest < max_seq_len
}

/// The long-input corpus for one model, on the paths its main corpus ran
/// in workers: `None` (and nothing printed) when the model's max_seq_len
/// doesn't reach the corpus. A missing long reference fails the model when
/// [`long_reference_required`], and is a warning otherwise. Its gate
/// failures join the model's, prefixed `long corpus:`. No floors apply to
/// it yet.
#[allow(clippy::too_many_arguments)]
fn grade_long(
    model: &ResolvedModel,
    models_dir: &Path,
    paths: &[Path3],
    caps: PathCaps,
    main_longest: usize,
    tok_sha: &str,
    expectations: &Expectations,
    o: &Options,
    scratch: &Path,
    mr: &mut ModelReport,
) -> Option<LongReport> {
    let corpus = Corpus::Long;
    match long_corpus_reaches(model.manifest.max_seq_len) {
        Ok(true) => {}
        Ok(false) => return None,
        Err(e) => {
            mr.failures.push(format!("long corpus: {e}"));
            return None;
        }
    }
    let file = corpus.file();
    if paths.is_empty() {
        let why = format!(
            "long corpus: max_seq_len {} reaches fixtures/parity/{file}, but it wasn't graded: \
             no worker ran this model (--external results cover the main corpus only)",
            model.manifest.max_seq_len
        );
        mr.warnings.push(why);
        return None;
    }
    let dir = reference_dir(model, o.refs.as_deref());
    if !dir.join(format!("{}.json", corpus.stem())).exists() {
        let why = format!(
            "long corpus: max_seq_len {} reaches fixtures/parity/{file} (the main corpus stops at {main_longest} \
             tokens), but there's no {}.json; generate one with tools/parity_reference.py --corpus {}",
            model.manifest.max_seq_len,
            corpus.stem(),
            corpus.name()
        );
        let named = o.models.contains(&model.manifest.id);
        if long_reference_required(named, model.manifest.max_seq_len, main_longest) {
            mr.failures.push(why);
        } else {
            mr.warnings.push(why);
        }
        return None;
    }
    let fail = |mr: &mut ModelReport, why: String| {
        mr.failures.push(format!("long corpus: {why}"));
        None
    };
    let reference = match Reference::load_corpus(&dir, corpus) {
        Ok(r) => r,
        Err(e) => return fail(mr, format!("unusable reference: {e}")),
    };
    let sha = match corpus_sha256_of(corpus) {
        Ok(s) => s,
        Err(e) => return fail(mr, e),
    };
    if let Some(why) = reference.stale(&model.manifest, &sha, tok_sha) {
        return fail(mr, why);
    }
    let tokens: Vec<usize> = reference.cases.iter().map(|c| c.ids.len()).collect();
    println!(
        "\n  long corpus (fixtures/parity/{file}): {} cases of {}-{} tokens{}",
        reference.cases.len(),
        tokens.iter().min().unwrap_or(&0),
        tokens.iter().max().unwrap_or(&0),
        if reference.skipped.is_empty() {
            String::new()
        } else {
            format!("; past max_seq_len, skipped: {}", reference.skipped.join(", "))
        }
    );
    let id = &model.manifest.id;
    let capped_refs = capped_references(&caps, &reference);
    let mut grades = Vec::new();
    for &path in paths {
        let (graded_on, capped) = path_reference(path, &reference, &capped_refs);
        if let Some(cap) = caps.of(path).filter(|_| !graded_on.has_core_case()) {
            println!("  {}: skipped (no case within {})", path.name(), cap_name(path, cap));
            continue;
        }
        match spawn_corpus_worker::<WorkerResult>(id, models_dir, path, None, corpus, scratch, o) {
            Ok(r) => {
                let mut g = grade(
                    graded_on,
                    &r,
                    path,
                    &expectations.gates,
                    &expectations.grades,
                    None,
                    &model.manifest.matryoshka,
                );
                g.capped = capped;
                grades.push(g);
            }
            Err(e) => {
                println!("  {}: CRASHED: {e}", path.name());
                mr.failures.push(format!("long corpus: {}: worker failed: {e}", path.name()));
            }
        }
    }
    print_grades(&reference, &grades, &capped_refs, None);
    for g in &grades {
        mr.failures.extend(g.failures.iter().map(|f| format!("long corpus: {}: {f}", g.path)));
    }
    Some(LongReport {
        corpus: file,
        cases: reference.cases.iter().map(|c| LongCase { id: c.id.clone(), tokens: c.ids.len() }).collect(),
        skipped: reference.skipped.clone(),
        paths: grades,
    })
}

/// A capped path, its cap, and what it is graded against: the reference's
/// cases within the cap, and the ids of the rest.
type CappedReference = (Path3, usize, (Reference, Vec<String>));

/// The reference each capped path is graded against (D33, D48).
fn capped_references(caps: &PathCaps, reference: &Reference) -> Vec<CappedReference> {
    caps.iter().map(|(path, cap)| (path, cap, reference.within(cap))).collect()
}

/// The reference a path is graded against, and the cases it leaves out:
/// for a capped path (the cpu path by D33, the ane path by D48), only the
/// cases within its cap.
fn path_reference<'a>(
    path: Path3,
    reference: &'a Reference,
    capped: &'a [CappedReference],
) -> (&'a Reference, Vec<String>) {
    match capped.iter().find(|(p, _, _)| *p == path) {
        Some((_, _, (sub, left_out))) => (sub, left_out.clone()),
        None => (reference, Vec::new()),
    }
}

/// Print the grades: those over the whole reference in one table, and
/// each capped path (D33, D48) in its own, over the cases it ran, after a
/// line naming the cases it didn't.
fn print_grades(reference: &Reference, grades: &[PathGrade], capped_refs: &[CappedReference], determinism: Option<&Check>) {
    let (capped, full): (Vec<PathGrade>, Vec<PathGrade>) = grades.iter().cloned().partition(|g| !g.capped.is_empty());
    print_model(reference, &full, determinism);
    for (path, cap, (sub, _)) in capped_refs {
        let mine: Vec<PathGrade> = capped.iter().filter(|g| g.path == path.name()).cloned().collect();
        for g in &mine {
            println!(
                "  {}: capped at {}; {} case(s) past it not run: {}",
                g.path,
                cap_name(*path, *cap),
                g.capped.len(),
                g.capped.join(", ")
            );
        }
        print_model(sub, &mine, None);
    }
}

fn print_model(reference: &Reference, grades: &[PathGrade], determinism: Option<&Check>) {
    if grades.is_empty() {
        return;
    }
    // Wide enough for every path's name (onnx-cpu:external is the longest).
    let w = grades.iter().map(|g| g.path.len()).max().unwrap_or(4).max(4);
    // A corpus without stress cases (the long one) has no stress column.
    let stressed = reference.cases.iter().any(|c| c.tags.iter().any(|t| t == crate::grade::STRESS_TAG));
    println!(
        "  {:<w$} {:<5} {:>10} {:<24} {:>10} {:>9} {:>9} {:>8} {:>6} {:>10} {:>10} {:>8}",
        "path",
        "grade",
        "worst",
        "(case)",
        "stress",
        "mean",
        "drift",
        "bias",
        "flips",
        "buckets",
        "pads",
        "ms/case"
    );
    for g in grades {
        println!(
            "  {:<w$} {:<5} {:>10} {:<24} {:>10} {:>9.6} {:>9} {:>+8.4} {:>6} {:>10} {:>10} {:>8.1}",
            g.path,
            g.letter,
            fmt(g.worst),
            format!("({})", g.worst_case),
            if stressed { fmt(g.stress) } else { "-".into() },
            g.mean,
            g.drift.map_or("n/a".into(), |d| format!("{d:.5}")),
            g.drift_bias,
            g.rank_flips,
            if g.bucket_invariance.n == 0 {
                "-".into()
            } else {
                fmt(g.bucket_invariance.min)
            },
            if g.pad_invariance.n == 0 {
                "-".into()
            } else {
                fmt(g.pad_invariance.min)
            },
            g.median_ms,
        );
    }
    if let Some(d) = determinism {
        println!(
            "  ANE across two processes: {} over {} cases",
            fmt(d.min),
            d.n
        );
    }

    // Worst case per tag, per path.
    let cw = (w + 1).max(10);
    println!(
        "  worst by tag:  {}",
        grades
            .iter()
            .map(|g| format!("{:>cw$}", g.path))
            .collect::<String>()
    );
    for tag in grades[0].by_tag.keys() {
        let cells: String = grades
            .iter()
            .map(|g| {
                format!(
                    "{:>cw$}",
                    fmt(g.by_tag[tag]).get(..8).unwrap_or("?").to_string()
                )
            })
            .collect();
        println!("    {tag:<14}{cells}");
    }

    // Matryoshka and oracles.
    for g in grades {
        if !g.matryoshka.is_empty() {
            let m: Vec<String> = g
                .matryoshka
                .iter()
                .map(|(d, w)| format!("{d}: {}", fmt(*w)))
                .collect();
            println!("  {} matryoshka worst: {}", g.path, m.join(", "));
        }
    }
    let torch = &reference.vectors["torch"];
    for (name, oracle) in reference.vectors.iter().filter(|(n, _)| *n != "torch") {
        let scores: Vec<Option<f64>> = oracle
            .iter()
            .zip(torch)
            .map(|(o, t)| cosine(o, t))
            .collect();
        let worst = crate::metrics::summarize(&scores);
        let vs: Vec<String> = grades
            .iter()
            .map(|g| format!("{} {}", g.path, fmt(g.vs_oracles[name])))
            .collect();
        println!(
            "  {name} (report only): vs torch {} ({}); sidekick vs it: {}",
            fmt(worst.worst),
            reference.cases[worst.worst_at].id,
            vs.join(", ")
        );
    }

    // The lowest cases, side by side across paths: how to tell a conversion
    // defect (every path low) from fp16 numerics (GPU and ANE low) from an
    // ANE lowering bug (only the ANE low). See docs/MODELS.md.
    let n = reference.cases.len();
    let mut order: Vec<usize> = (0..n).collect();
    let low = |i: usize| {
        grades
            .iter()
            .map(|g| g.per_case[i].unwrap_or(f64::NEG_INFINITY))
            .fold(f64::INFINITY, f64::min)
    };
    order.sort_by(|&a, &b| low(a).total_cmp(&low(b)));
    println!("  lowest cases:");
    for &i in order.iter().take(6) {
        let c = &reference.cases[i];
        let cells: Vec<String> = grades
            .iter()
            .map(|g| format!("{} {}", g.path, fmt(g.per_case[i])))
            .collect();
        println!(
            "    {:<26} {:>4} tok  {}  [{}]",
            c.id,
            c.ids.len(),
            cells.join("  "),
            c.tags.join(",")
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_onnx_models_files_are_hashed_by_their_manifest_paths() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny-onnx");
        let reg = ModelRegistry::scan_with(&dir, &MEASURE_ALL).unwrap();
        let model = reg.get("tiny-onnx").unwrap();
        let files = onnx_files(model).unwrap();
        let paths: Vec<&str> = files.iter().map(|f| f.path.as_str()).collect();
        assert_eq!(paths, ["../hidden.onnx", "tokenizer.json"]);
        for f in &files {
            assert_eq!(f.sha256, sha256_hex(&std::fs::read(model.dir.join(&f.path)).unwrap()), "{}", f.path);
        }
    }

    const ONNX: Runtime = Runtime::Onnx { native: true };
    const CORE_ML_ANE: Runtime = Runtime::CoreMl { ane: true };

    #[test]
    fn by_default_each_model_runs_its_own_backends_paths() {
        let paths = default_paths();
        let (core_ml, skipped) = model_paths(&paths, CORE_ML_ANE);
        let (onnx, _) = model_paths(&paths, ONNX);
        if CORE_ML {
            // Both backends in one build: a Core ML embedder or classifier
            // never gets onnx-cpu, nor an onnx model a Core ML path.
            assert_eq!(paths, [Path3::Cpu, Path3::Gpu, Path3::Ane, Path3::OnnxCpu]);
            assert_eq!(core_ml, Path3::ALL);
            assert_eq!(skipped.iter().map(|s| s.0).collect::<Vec<_>>(), [Path3::OnnxCpu]);
        } else {
            assert_eq!(paths, [Path3::OnnxCpu]);
            assert!(core_ml.is_empty());
        }
        assert_eq!(onnx, [Path3::OnnxCpu]);
    }

    #[test]
    fn a_path_a_model_cant_run_is_skipped_with_why() {
        let (run, skip) = model_paths(&[Path3::OnnxCpu, Path3::Ane], CORE_ML_ANE);
        assert_eq!(run, [Path3::Ane]);
        assert_eq!(skip, [(Path3::OnnxCpu, "a Core ML model doesn't run on ONNX Runtime")]);
        let (run, skip) = model_paths(&[Path3::Cpu, Path3::OnnxCpu], ONNX);
        assert_eq!(run, [Path3::OnnxCpu]);
        assert_eq!(skip, [(Path3::Cpu, "an onnx model runs on onnx-cpu, not Core ML")]);
        // Nothing it can run: no worker at all.
        assert!(model_paths(&[Path3::OnnxCpu], CORE_ML_ANE).0.is_empty());
        // Without the backend, onnx-cpu comes from --external results only:
        // no worker, and not a skip.
        assert_eq!(model_paths(&[Path3::OnnxCpu], Runtime::Onnx { native: false }), (vec![], vec![]));
    }

    #[test]
    fn a_missing_long_reference_fails_a_named_model_the_main_corpus_stops_short_of() {
        // Qwen3-Embedding-0.6B's onnx build: 4,096 tokens, main corpus to 700.
        assert!(long_reference_required(true, 4096, 700));
        // The same, not named: every model with a reference is graded, and
        // one without a long reference is warned, as one without a main
        // reference is skipped.
        assert!(!long_reference_required(false, 4096, 700));
        // EmbeddingGemma-2: its over-length cases already fill its 2,048.
        assert!(!long_reference_required(true, 2048, 2048));
    }

    #[test]
    fn the_ane_path_runs_under_the_ane_cap() {
        use sidekick_core::DEFAULT_ANE_MAX_SEQ;
        // Past the default cap: capped at it.
        assert_eq!(ane_path_cap(false, 4096, None), Some(DEFAULT_ANE_MAX_SEQ));
        // Within it, or at it: nothing to cap.
        assert_eq!(ane_path_cap(false, 2048, None), None);
        assert_eq!(ane_path_cap(false, 512, None), None);
        // The manifest's own cap, either way.
        assert_eq!(ane_path_cap(false, 2048, Some(1024)), Some(1024));
        assert_eq!(ane_path_cap(false, 4096, Some(4096)), None);
        // --ignore-ane-seq-cap grades every bucket.
        assert_eq!(ane_path_cap(true, 4096, None), None);
    }

    #[test]
    fn the_cpu_path_runs_under_the_cpu_cap() {
        use sidekick_core::MAX_CPU_INVARIANT_SEQ;
        // Past 1,024 tokens: capped there, as the runtime serves cpu_only.
        assert_eq!(cpu_path_cap(false, 2048, CpuSeqLimit::Enforce), Some(MAX_CPU_INVARIANT_SEQ));
        // Within it, or at it: nothing to cap.
        assert_eq!(cpu_path_cap(false, 1024, CpuSeqLimit::Enforce), None);
        assert_eq!(cpu_path_cap(false, 512, CpuSeqLimit::Enforce), None);
        // A manifest that lifts the cap is served, and graded, uncapped.
        assert_eq!(cpu_path_cap(false, 2048, CpuSeqLimit::Ignore), None);
        // --ignore-cpu-seq-cap grades every bucket, for measurement.
        assert_eq!(cpu_path_cap(true, 2048, CpuSeqLimit::Enforce), None);
    }

    #[test]
    fn each_path_has_its_own_cap() {
        let o = parse(&["--models-dir".into(), "unused".into()]).unwrap();
        // Qwen3-Embedding-0.6B's Core ML build, 2,048 tokens: the cpu path
        // capped at 1,024, the ane path within its cap.
        let caps = PathCaps::new(&o, 2048, CpuSeqLimit::Enforce, None);
        assert_eq!((caps.of(Path3::Cpu), caps.of(Path3::Gpu), caps.of(Path3::Ane)), (Some(1024), None, None));
        assert_eq!(caps.iter().collect::<Vec<_>>(), [(Path3::Cpu, 1024)]);
        // 4,096 tokens: both capped; the gpu path never is.
        let caps = PathCaps::new(&o, 4096, CpuSeqLimit::Enforce, None);
        assert_eq!(caps.iter().collect::<Vec<_>>(), [(Path3::Cpu, 1024), (Path3::Ane, 2048)]);
        assert_eq!(caps.of(Path3::OnnxCpu), None);
        // Each flag lifts its own path's cap only.
        let o = parse(&["--models-dir".into(), "unused".into(), "--ignore-cpu-seq-cap".into()]).unwrap();
        assert_eq!(PathCaps::new(&o, 4096, CpuSeqLimit::Enforce, None).iter().collect::<Vec<_>>(), [(Path3::Ane, 2048)]);
        assert_eq!(cap_name(Path3::Cpu, 1024), "the CPU cap of 1024 tokens (D33)");
        assert_eq!(cap_name(Path3::Ane, 2048), "the ANE cap of 2048 tokens (D48)");
    }

    /// D48: a bucket past the ANE cap of a model served on the ANE has no
    /// compute plan read (reading one compiles it for the ANE); served on
    /// the GPU, every bucket's plan is read.
    #[cfg(all(target_os = "macos", feature = "coreml"))]
    #[test]
    fn no_plan_is_read_for_a_bucket_past_the_ane_cap_of_an_ane_served_model() {
        let o = parse(&["--models-dir".into(), "unused".into()]).unwrap();
        let read_for = |units: ComputeUnits| {
            let mut read: Vec<String> = Vec::new();
            let plans = check_plans_with(
                "m",
                &[128, 2048, 4096],
                |_| Vec::new(),
                units,
                Some(2048),
                &o,
                &mut Vec::new(),
                &mut Vec::new(),
                |_: &[PathBuf], label: &str| {
                    read.push(label.to_string());
                    let plan = PlanSummary { ane: 9, cpu: 1, gpu: 0, unassigned: 0, off_ane_ops: Default::default() };
                    Ok(Some((plan, false)))
                },
            );
            (read, plans.buckets)
        };
        for units in [ComputeUnits::CpuAndNeuralEngine, ComputeUnits::All] {
            let (read, buckets) = read_for(units);
            assert_eq!(read, ["m-128", "m-2048"], "{units:?}");
            let capped = &buckets[2];
            assert_eq!((capped.ane_ops, capped.assigned_ops, &capped.plan), (None, None, &None));
            assert_eq!(capped.not_read.as_deref(), Some("not read (past the ANE cap of 2048 tokens, D48)"));
            assert_eq!((buckets[0].ane_ops, &buckets[0].not_read), (Some(9), &None));
        }
        let (read, buckets) = read_for(ComputeUnits::CpuAndGpu);
        assert_eq!(read, ["m-128", "m-2048", "m-4096"]);
        assert!(buckets.iter().all(|b| b.not_read.is_none() && b.plan.is_some()));
    }

    #[test]
    fn a_capped_path_is_compared_on_the_cases_both_paths_ran() {
        let cpu = ["a", "b", "c", "d"];
        let ane = ["a", "c"];
        let pairs = same_cases(&cpu, &ane, |s: &&str| *s);
        assert_eq!(pairs, [(&"a", &"a"), (&"c", &"c")]);
    }

    #[test]
    fn without_a_neural_engine_a_core_ml_model_runs_cpu_and_gpu_ungated() {
        // An Intel Mac: ane is skipped, cpu and gpu run.
        let (run, skip) = model_paths(&[Path3::Cpu, Path3::Gpu, Path3::Ane], Runtime::CoreMl { ane: false });
        assert_eq!(run, [Path3::Cpu, Path3::Gpu]);
        assert_eq!(skip, [(Path3::Ane, NO_ANE)]);
        // Plans gate only a model served on the ANE, on a Mac that has one.
        assert!(plans_gate(true, true));
        assert!(!plans_gate(true, false), "no Neural Engine: no ANE eligibility to gate on");
        assert!(!plans_gate(false, true));
    }
}
