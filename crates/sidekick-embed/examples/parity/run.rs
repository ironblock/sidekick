//! The live suite (macOS, `--features coreml`): a parent that reads compute
//! plans and grades, and workers that each run one model on one compute path
//! in their own process. An Objective-C exception inside Core ML (on
//! macOS 27 a flexible-shape artifact aborts at predict under `.cpuOnly`)
//! or a stuck ANE compile then costs one cell of the report, not the whole
//! run.

mod classify;

use crate::expect::{Expectations, Path3};
use crate::grade::{fmt, grade, suggest_floor, CaseResult, Check, PathGrade, WorkerResult};
use crate::metrics::cosine;
use crate::reference::{corpus_sha256, sha256_hex, Reference};
use serde::Serialize;
use sidekick_core::manifest::{ModelRegistry, ResolvedClassifier, ResolvedModel};
use sidekick_core::{EmbedPurpose, Embedder, EmbeddingBackendKind};
use sidekick_coreml::{compute_plan, ComputeUnits, PlanSummary};
use sidekick_embed::CoremlEmbedder;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

const USAGE: &str = "\
usage: parity [options]
  --models-dir DIR     a models directory to scan (repeatable; default: the
                       daemon's, ~/Library/Application Support/sidekick/models)
  --refs DIR           references as DIR/<model id>/reference.* (default:
                       <model dir>/parity/)
  --model ID           only this model (repeatable)
  --paths LIST         compute paths, comma-separated: cpu,gpu,ane (default all)
  --expectations FILE  default: fixtures/parity/expectations.toml
  --json FILE          also write the full report as JSON
  --suggest-floors     print [[floor]] entries for this chip from this run
  --allow-unverified-plans
                       grade a model whose compute plan Core ML can't produce
                       (a warning, not a failure), provided its ANE output
                       isn't bit-identical to CPU_ONLY
  --timeout SECS       per worker (default 1200)";

/// The suite measures every compute path itself (its ANE plan check reports
/// a model Core ML won't place on the ANE), so it loads models past the ANE
/// weight cap too.
const MEASURE_ALL: sidekick_core::ScanOptions = sidekick_core::ScanOptions { ignore_ane_weight_cap: true };

/// Cases re-run in a second ANE process to check determinism across loads.
const DETERMINISM_CASES: usize = 8;

pub fn main() -> i32 {
    let args: Vec<String> = std::env::args().skip(1).collect();
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

fn units(path: Path3) -> ComputeUnits {
    match path {
        Path3::Cpu => ComputeUnits::CpuOnly,
        Path3::Gpu => ComputeUnits::CpuAndGpu,
        Path3::Ane => ComputeUnits::CpuAndNeuralEngine,
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

/// Where Core ML caches compiled bundles for this executable.
fn bundle_cache() -> String {
    let exe = std::env::current_exe()
        .ok()
        .and_then(|p| p.file_name().map(|n| n.to_string_lossy().into_owned()))
        .unwrap_or_else(|| "parity".into());
    format!("~/Library/Caches/{exe}/com.apple.e5rt.e5bundlecache")
}

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

/// `--worker <models-dir> <model-id> <refs-dir|-> <path> <out.json> [limit]`
fn worker(args: &[String]) -> Result<(), String> {
    let [models_dir, id, refs, path, out, rest @ ..] = args else {
        return Err("bad worker arguments".into());
    };
    let limit: Option<usize> = rest
        .first()
        .map(|s| s.parse().map_err(|_| "bad limit"))
        .transpose()?;
    let path = Path3::parse(path).ok_or("bad path")?;
    let refs = (refs != "-").then(|| PathBuf::from(refs));
    let registry = ModelRegistry::scan_with(Path::new(models_dir), &MEASURE_ALL).map_err(|e| e.to_string())?;
    if let Ok(model) = registry.classifier(id) {
        return classify::worker(model, refs.as_deref(), path, out, limit);
    }
    let model = load_model(Path::new(models_dir), id)?;
    let reference = Reference::load(&reference_dir(&model, refs.as_deref()))?;
    let vocab = tokenizers::Tokenizer::from_file(model.tokenizer_path())
        .map_err(|e| e.to_string())?
        .get_vocab_size(true) as u64;

    let t0 = Instant::now();
    let emb = CoremlEmbedder::load_with(&model, units(path)).map_err(|e| e.to_string())?;
    let load_ms = t0.elapsed().as_secs_f64() * 1e3;
    let buckets = emb.buckets().to_vec();
    let cases = &reference.cases[..limit
        .unwrap_or(reference.cases.len())
        .min(reference.cases.len())];
    let full = limit.is_none();
    let mut rng = XorShift(0x5eed_1234_abcd_0001);
    let e = |e: sidekick_core::Error| e.to_string();

    let mut results = Vec::with_capacity(cases.len());
    let mut raw = Vec::with_capacity(cases.len());
    for case in cases {
        let purpose = if case.purpose == "query" {
            EmbedPurpose::Query
        } else {
            EmbedPurpose::Document
        };
        let prepared = emb.prepare(&case.text, purpose).map_err(e)?;
        let t = Instant::now();
        let v = emb
            .embed(&[case.text.as_str()], purpose)
            .map_err(e)?
            .remove(0);
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
    for (case, first) in cases.iter().zip(&raw).take(5) {
        let purpose = if case.purpose == "query" {
            EmbedPurpose::Query
        } else {
            EmbedPurpose::Document
        };
        let again = emb
            .embed(&[case.text.as_str()], purpose)
            .map_err(e)?
            .remove(0);
        repeat_bitwise &= bitwise_eq(first, &again);
    }

    let result = WorkerResult {
        model: id.clone(),
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
    expectations: PathBuf,
    json: Option<PathBuf>,
    suggest: bool,
    allow_unverified: bool,
    timeout: Duration,
}

fn parse(args: &[String]) -> Result<Options, String> {
    let mut o = Options {
        models_dirs: Vec::new(),
        refs: None,
        models: Vec::new(),
        paths: Path3::ALL.to_vec(),
        expectations: Expectations::default_path(),
        json: None,
        suggest: false,
        allow_unverified: false,
        timeout: Duration::from_secs(1200),
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
                    .map(|p| Path3::parse(p.trim()).ok_or(format!("unknown path {p:?}")))
                    .collect::<Result<_, _>>()?
            }
            "--expectations" => o.expectations = val()?.into(),
            "--json" => o.json = Some(val()?.into()),
            "--suggest-floors" => o.suggest = true,
            "--allow-unverified-plans" => o.allow_unverified = true,
            "--timeout" => {
                o.timeout = Duration::from_secs(val()?.parse().map_err(|_| "bad --timeout")?)
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

fn sh(cmd: &str, args: &[&str]) -> String {
    Command::new(cmd)
        .args(args)
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_default()
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
    ane_ops: usize,
    assigned_ops: usize,
    plan: Result<(), String>,
}

#[derive(Serialize)]
struct ModelReport {
    id: String,
    source: String,
    buckets: Vec<BucketInfo>,
    paths: Vec<PathGrade>,
    /// Paths whose worker failed to produce results, with why.
    crashed: Vec<(String, String)>,
    determinism: Option<Check>,
    failures: Vec<String>,
    regressions: Vec<String>,
    warnings: Vec<String>,
}

#[derive(Serialize)]
struct Report {
    chip: String,
    macos: String,
    models: Vec<ModelReport>,
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

/// Read and print every bucket's compute plan, before anything predicts:
/// an artifact the ANE rejects can abort the process at predict.
fn check_plans(
    id: &str,
    buckets: &[usize],
    artifact_for: impl Fn(usize) -> PathBuf,
    o: &Options,
    scratch: &Path,
    failures: &mut Vec<String>,
    warnings: &mut Vec<String>,
) -> Plans {
    let mut plans = Plans { ok: true, unverified: Vec::new(), buckets: Vec::new() };
    for &b in buckets {
        let art = artifact_for(b);
        let plan = read_plan(&art, &format!("{id}-{b}"), o.timeout, scratch);
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
                (_, Err(e)) => format!("ANE ops {ane}/{assigned} NOT ELIGIBLE: {e}"),
            },
            file_sha(&art.join("model.mil")).unwrap_or("-".into()),
            file_sha(&art.join("weights/weight.bin")).unwrap_or("-".into()),
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
            if o.allow_unverified {
                warnings.push(why);
                plans.unverified.push(b);
            } else {
                failures.push(why);
            }
        }
        if let Err(e) = &verdict {
            plans.ok = false;
            failures.push(format!("bucket {b}: compute plan: {e}"));
        }
        plans.buckets.push(BucketInfo {
            bucket: b,
            model_mil: file_sha(&art.join("model.mil")),
            weights: file_sha(&art.join("weights/weight.bin")),
            ane_ops: ane,
            assigned_ops: assigned,
            plan: verdict,
        });
    }
    plans
}

/// Run a worker in its own process, with a timeout.
fn spawn_worker<T: serde::de::DeserializeOwned>(
    id: &str,
    models_dir: &Path,
    refs: Option<&Path>,
    path: Path3,
    limit: Option<usize>,
    timeout: Duration,
    scratch: &Path,
) -> Result<T, String> {
    let tag = format!(
        "{id}-{}-{}",
        path.name(),
        limit.map_or("full".into(), |l| l.to_string())
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
    run_child(&args, &out, &log, timeout)?;
    let bytes = std::fs::read(&out).map_err(|e| e.to_string())?;
    serde_json::from_slice(&bytes).map_err(|e| e.to_string())
}

fn parent(args: &[String]) -> Result<bool, String> {
    let o = parse(args)?;
    let expectations = Expectations::load(&o.expectations)?;
    let chip = sh("sysctl", &["-n", "machdep.cpu.brand_string"]);
    let macos = format!(
        "{} ({})",
        sh("sw_vers", &["-productVersion"]),
        sh("sw_vers", &["-buildVersion"])
    );
    let corpus = corpus_sha256();
    let scratch = std::env::temp_dir().join(format!("sidekick-parity-{}", std::process::id()));
    std::fs::create_dir_all(&scratch).map_err(|e| e.to_string())?;

    let mut models: Vec<(PathBuf, ResolvedModel)> = Vec::new();
    for dir in &o.models_dirs {
        let reg = ModelRegistry::scan_with(dir, &MEASURE_ALL).map_err(|e| e.to_string())?;
        for m in reg.iter() {
            if m.manifest.backend == EmbeddingBackendKind::Coreml
                && (o.models.is_empty() || o.models.contains(&m.manifest.id))
            {
                models.push((dir.clone(), m.clone()));
            }
        }
    }
    let mut classifiers: Vec<(PathBuf, ResolvedClassifier)> = Vec::new();
    for dir in &o.models_dirs {
        let reg = ModelRegistry::scan_with(dir, &MEASURE_ALL).map_err(|e| e.to_string())?;
        for c in reg.classifiers() {
            if o.models.is_empty() || o.models.contains(&c.manifest.id) {
                classifiers.push((dir.clone(), c.clone()));
            }
        }
    }
    for want in &o.models {
        if !models.iter().any(|(_, m)| &m.manifest.id == want)
            && !classifiers.iter().any(|(_, c)| &c.manifest.id == want)
        {
            return Err(format!("model {want} not found in any --models-dir"));
        }
    }
    let has_floors = expectations.floor.iter().any(|f| f.chip == chip);
    println!(
        "parity suite: {chip}, macOS {macos}, {} model(s)",
        models.len() + classifiers.len()
    );
    if !has_floors {
        println!("no floors recorded for this chip: enforcing gates only, reporting accuracy");
    }

    let mut report = Report {
        chip: chip.clone(),
        macos,
        models: Vec::new(),
        classifiers: Vec::new(),
    };
    let mut ok = true;
    let mut graded = 0usize;
    let mut suggestions = Vec::new();
    for (models_dir, model) in &models {
        let id = &model.manifest.id;
        let refs_dir = reference_dir(model, o.refs.as_deref());
        println!("\n=== {id}");
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
            paths: Vec::new(),
            crashed: Vec::new(),
            determinism: None,
            failures: Vec::new(),
            regressions: Vec::new(),
            warnings: Vec::new(),
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
        // an artifact the ANE rejects can abort the process at predict.
        let plans = check_plans(
            id,
            &model.manifest.buckets,
            |b| model.artifact_path_for_bucket(b),
            &o,
            &scratch,
            &mut mr.failures,
            &mut mr.warnings,
        );
        mr.buckets = plans.buckets;
        let (plan_ok, unverified) = (plans.ok, plans.unverified);
        if !plan_ok {
            println!("  FAIL: not ANE-eligible; skipping predictions (such an artifact can abort at predict)");
            ok = false;
            report.models.push(mr);
            continue;
        }

        let mut results: Vec<(Path3, WorkerResult)> = Vec::new();
        for &path in &o.paths {
            match spawn_worker::<WorkerResult>(
                id,
                models_dir,
                o.refs.as_deref(),
                path,
                None,
                o.timeout,
                &scratch,
            ) {
                Ok(r) => results.push((path, r)),
                Err(e) => {
                    println!("  {}: CRASHED: {e}", path.name());
                    mr.crashed.push((path.name().into(), e.clone()));
                    mr.failures
                        .push(format!("{}: worker failed: {e}", path.name()));
                }
            }
        }

        // Determinism across processes: a second, independent ANE load.
        if let Some((_, first)) = results.iter().find(|(p, _)| *p == Path3::Ane) {
            match spawn_worker::<WorkerResult>(
                id,
                models_dir,
                o.refs.as_deref(),
                Path3::Ane,
                Some(DETERMINISM_CASES),
                o.timeout,
                &scratch,
            ) {
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
                grade(
                    &reference,
                    r,
                    *p,
                    &expectations.gates,
                    floor,
                    &model.manifest.matryoshka,
                )
            })
            .collect();

        // A bucket graded without a compute plan must show it ran somewhere
        // other than the CPU: bit-identical ANE and CPU_ONLY output means
        // the "ANE" path fell back to the CPU.
        let cpu = results.iter().find(|(p, _)| *p == Path3::Cpu);
        let ane = results.iter().find(|(p, _)| *p == Path3::Ane);
        for &b in &unverified {
            let (Some((_, c)), Some((_, a))) = (cpu, ane) else {
                mr.failures.push(format!(
                    "bucket {b}: without a compute plan, grading needs both the cpu and ane paths \
                     to rule out a CPU fallback"
                ));
                continue;
            };
            let in_bucket: Vec<usize> = (0..a.cases.len())
                .filter(|&i| a.cases[i].bucket == b)
                .collect();
            if in_bucket.is_empty() {
                mr.failures.push(format!(
                    "bucket {b}: no compute plan and no case to check it with"
                ));
            } else if in_bucket
                .iter()
                .all(|&i| bitwise_eq(&a.cases[i].vector, &c.cases[i].vector))
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
            if c.cases
                .iter()
                .zip(&a.cases)
                .all(|(x, y)| bitwise_eq(&x.vector, &y.vector))
            {
                mr.warnings.push("ANE output is bit-identical to CPU_ONLY: the model may be running entirely on the CPU".into());
            }
            let med = |r: &WorkerResult| {
                let mut ms: Vec<f64> = r.cases.iter().map(|c| c.ms).collect();
                ms.sort_by(f64::total_cmp);
                ms[ms.len() / 2]
            };
            let ratio = med(c) / med(a);
            if ratio < 1.1 {
                mr.warnings.push(format!(
                    "CPU_ONLY/ANE median latency ratio {ratio:.2}x < 1.1x: possible runtime fallback (or a loaded machine)"
                ));
            }
        }

        print_model(&reference, &grades, &mr);
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

    for (models_dir, model) in &classifiers {
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
        println!("\nFAIL: nothing was graded (no model had a reference)");
        ok = false;
    }
    let _ = std::fs::remove_dir_all(&scratch);
    println!("\n{}", if ok { "parity: PASS" } else { "parity: FAIL" });
    Ok(ok)
}

fn print_model(reference: &Reference, grades: &[PathGrade], mr: &ModelReport) {
    if grades.is_empty() {
        return;
    }
    println!(
        "  {:<4} {:<5} {:>10} {:<24} {:>10} {:>9} {:>9} {:>8} {:>6} {:>10} {:>10} {:>8}",
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
            "  {:<4} {:<5} {:>10} {:<24} {:>10} {:>9.6} {:>9} {:>+8.4} {:>6} {:>10} {:>10} {:>8.1}",
            g.path,
            g.letter,
            fmt(g.worst),
            format!("({})", g.worst_case),
            fmt(g.stress),
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
    if let Some(d) = &mr.determinism {
        println!(
            "  ANE across two processes: {} over {} cases",
            fmt(d.min),
            d.n
        );
    }

    // Worst case per tag, per path.
    println!(
        "  worst by tag:  {}",
        grades
            .iter()
            .map(|g| format!("{:>10}", g.path))
            .collect::<String>()
    );
    for tag in grades[0].by_tag.keys() {
        let cells: String = grades
            .iter()
            .map(|g| {
                format!(
                    "{:>10}",
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
