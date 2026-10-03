//! Where Core ML places each loaded bucket's operations: its compute plan,
//! read off the request path and reported by `/v1/models` and `/health`.
//!
//! A manifest's `compute_units` is a preference. Core ML decides per
//! operation which device runs it, and an operation the ANE can't run falls
//! back to the CPU or the GPU without any error. The parity suite gates
//! that before a model ships; this reports it for the models a daemon
//! actually serves, on the machine and macOS build it runs on.
//!
//! Reading a plan compiles the model for its compute units, as a load
//! does, so it is never on the request path:
//! - a bucket's plan is read after its first load, never for buckets no
//!   request has used;
//! - one background thread reads every plan, one at a time, in the order
//!   the buckets loaded;
//! - a plan read is cached on disk, keyed by the artifact (its program's
//!   content, its weights' sizes and times), the compute units and the
//!   macOS build, so a restart doesn't read it again;
//! - a listing only looks results up, and never waits on a read.
//!
//! The service is off unless the daemon turns it on ([`enable`]), so tools
//! that load models (the parity suite, tests) never read plans by accident.

use sidekick_core::{ComputeUnits, Result};
use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::sync::mpsc::{self, Sender};
use std::sync::{Arc, Mutex, OnceLock, PoisonError};

/// How many operations a plan puts on each device.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct OpCounts {
    pub ane: usize,
    pub gpu: usize,
    pub cpu: usize,
    /// Operations with no device (constants and other bookkeeping).
    pub unassigned: usize,
    /// Operator names off the ANE, with counts.
    pub off_ane_ops: BTreeMap<String, usize>,
}

/// One bucket's plan, as far as the service knows it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Placement {
    /// Queued or being read.
    Pending,
    Ready(Arc<OpCounts>),
    /// The read failed. The next load of the bucket tries again.
    Failed(String),
}

/// Reads a compiled model's plan: Core ML's in the daemon, a stand-in in
/// tests.
pub type PlanReader = Arc<dyn Fn(&Path, ComputeUnits) -> Result<OpCounts> + Send + Sync>;

type Key = (PathBuf, ComputeUnits);

/// The plan-reading service. One per process, from [`enable`]; tests build
/// their own with [`Placements::new`].
pub struct Placements {
    known: Mutex<HashMap<Key, Placement>>,
    queue: Mutex<Sender<Key>>,
}

impl Placements {
    /// A service whose one worker thread reads plans with `reader`, caching
    /// them under `cache_dir` when it's set.
    pub fn new(reader: PlanReader, cache_dir: Option<PathBuf>) -> Arc<Self> {
        let (tx, rx) = mpsc::channel::<Key>();
        let service = Arc::new(Self { known: Mutex::default(), queue: Mutex::new(tx) });
        let weak = Arc::downgrade(&service);
        let build = macos_build();
        let spawned = std::thread::Builder::new().name("sidekick-compute-plans".into()).spawn(move || {
            for (path, units) in rx {
                let placement = match read(&reader, cache_dir.as_deref(), &build, &path, units) {
                    Ok(counts) => Placement::Ready(Arc::new(counts)),
                    Err(e) => {
                        tracing::warn!("compute plan of {}: {e}", path.display());
                        Placement::Failed(e.to_string())
                    }
                };
                let Some(service) = weak.upgrade() else { break };
                service.lock().insert((path, units), placement);
            }
        });
        if let Err(e) = spawned {
            tracing::warn!("no compute-plan thread, so no placement reports: {e}");
        }
        service
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, HashMap<Key, Placement>> {
        self.known.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Note that `path` was just loaded with `units`: queue its plan unless
    /// it is known or already queued. A failed read is queued again.
    pub fn loaded(&self, path: &Path, units: ComputeUnits) {
        let key = (path.to_path_buf(), units);
        {
            let mut known = self.lock();
            if matches!(known.get(&key), Some(Placement::Pending | Placement::Ready(_))) {
                return;
            }
            known.insert(key.clone(), Placement::Pending);
        }
        let sent = self.queue.lock().unwrap_or_else(PoisonError::into_inner).send(key.clone());
        if sent.is_err() {
            self.lock().insert(key, Placement::Failed("the compute-plan thread isn't running".into()));
        }
    }

    /// What is known of `path`'s plan under `units`; `None` before its
    /// first load.
    pub fn get(&self, path: &Path, units: ComputeUnits) -> Option<Placement> {
        self.lock().get(&(path.to_path_buf(), units)).cloned()
    }

    /// Each of a model's buckets that has a plan state, by bucket. A
    /// chunked bucket's (D37) is its chunks' together: their counts summed
    /// once every chunk is read, failed if any read failed, pending until
    /// then.
    pub fn for_model(
        &self,
        dir: &Path,
        artifact: &str,
        buckets: &[usize],
        chunks: usize,
        units: ComputeUnits,
    ) -> BTreeMap<usize, Placement> {
        buckets
            .iter()
            .filter_map(|&b| {
                let states: Vec<Option<Placement>> = sidekick_core::artifact_files(artifact, b, chunks)
                    .iter()
                    .map(|f| self.get(&dir.join(f), units))
                    .collect();
                combine(states).map(|p| (b, p))
            })
            .collect()
    }
}

/// One bucket's state from its programs' states, in chain order; `None`
/// when none of them has been loaded.
fn combine(states: Vec<Option<Placement>>) -> Option<Placement> {
    if states.len() == 1 {
        return states.into_iter().next().flatten();
    }
    if states.iter().all(Option::is_none) {
        return None;
    }
    let mut sum = OpCounts::default();
    let mut pending = false;
    for (i, state) in states.into_iter().enumerate() {
        match state {
            Some(Placement::Ready(c)) => {
                sum.ane += c.ane;
                sum.gpu += c.gpu;
                sum.cpu += c.cpu;
                sum.unassigned += c.unassigned;
                for (op, n) in &c.off_ane_ops {
                    *sum.off_ane_ops.entry(op.clone()).or_default() += n;
                }
            }
            Some(Placement::Failed(e)) => return Some(Placement::Failed(format!("chunk {i}: {e}"))),
            Some(Placement::Pending) | None => pending = true,
        }
    }
    Some(if pending { Placement::Pending } else { Placement::Ready(Arc::new(sum)) })
}

static SERVICE: OnceLock<Arc<Placements>> = OnceLock::new();

/// Turn plan reporting on for this process, caching plans under
/// `cache_dir`. Returns the service, or `None` in a build without Core ML.
/// Later calls return the first call's service.
pub fn enable(cache_dir: Option<PathBuf>) -> Option<Arc<Placements>> {
    let reader = core_ml_reader()?;
    Some(SERVICE.get_or_init(|| Placements::new(reader, cache_dir)).clone())
}

/// Called by the Core ML backends after a bucket's first load.
#[cfg_attr(not(all(feature = "coreml", target_os = "macos")), allow(dead_code))]
pub(crate) fn loaded(path: &Path, units: ComputeUnits) {
    if let Some(service) = SERVICE.get() {
        service.loaded(path, units);
    }
}

#[cfg(all(feature = "coreml", target_os = "macos"))]
fn core_ml_reader() -> Option<PlanReader> {
    Some(Arc::new(|path: &Path, units: ComputeUnits| {
        let plan = sidekick_coreml::compute_plan(path, units)?;
        if plan.assigned() == 0 {
            // A known-good artifact reads this way when Core ML's bundle
            // cache entry for its path is broken; don't cache it.
            return Err(sidekick_core::Error::Inference(
                "Core ML assigned no operation to any device (its cache entry for this path may be \
                 broken)"
                    .into(),
            ));
        }
        Ok(OpCounts { ane: plan.ane, gpu: plan.gpu, cpu: plan.cpu, unassigned: plan.unassigned, off_ane_ops: plan.off_ane_ops })
    }))
}

#[cfg(not(all(feature = "coreml", target_os = "macos")))]
fn core_ml_reader() -> Option<PlanReader> {
    None
}

/// A plan, from the disk cache when it has it, otherwise read and cached.
fn read(reader: &PlanReader, cache_dir: Option<&Path>, build: &str, path: &Path, units: ComputeUnits) -> Result<OpCounts> {
    let cached = cache_dir.and_then(|dir| match content_key(path, units, build) {
        Ok(key) => Some(dir.join(format!("{key}.json"))),
        Err(e) => {
            tracing::debug!("not caching the plan of {}: {e}", path.display());
            None
        }
    });
    if let Some(counts) = cached.as_deref().and_then(load_cached) {
        return Ok(counts);
    }
    let started = std::time::Instant::now();
    let counts = reader(path, units)?;
    tracing::info!(
        "compute plan of {} ({}): {} ANE, {} GPU, {} CPU operations, read in {:.1?}",
        path.display(),
        units.name(),
        counts.ane,
        counts.gpu,
        counts.cpu,
        started.elapsed()
    );
    if let Some(file) = &cached {
        if let Err(e) = store(file, &counts) {
            tracing::debug!("couldn't cache the plan at {}: {e}", file.display());
        }
    }
    Ok(counts)
}

/// Files at most this size are hashed whole; larger ones (the weights) by
/// length and modification time. Hashing a large bucket's weights would
/// cost about as long as reading its plan again from Core ML's warm cache.
const HASHED_FILE_MAX: u64 = 1 << 20;

/// sha256 over what decides a plan: the artifact's files in path order
/// (each one's relative path and length, then its bytes, or for a large
/// file its modification time), the compute units, and the macOS build. A
/// rebuilt artifact gets a new key; one copied elsewhere, if its weights
/// keep their times, keeps its key.
fn content_key(path: &Path, units: ComputeUnits, build: &str) -> std::io::Result<String> {
    use sha2::{Digest, Sha256};
    let mut files = Vec::new();
    collect_files(path, path, &mut files)?;
    files.sort();
    let mut hash = Sha256::new();
    hash.update(format!("sidekick compute plan v1\0{}\0{build}\0", units.name()));
    for rel in files {
        let full = path.join(&rel);
        let meta = std::fs::metadata(&full)?;
        hash.update(rel.to_string_lossy().as_bytes());
        hash.update(b"\0");
        hash.update(meta.len().to_le_bytes());
        if meta.len() <= HASHED_FILE_MAX {
            hash.update(std::fs::read(&full)?);
        } else {
            let modified = meta.modified()?.duration_since(std::time::UNIX_EPOCH).unwrap_or_default();
            hash.update(modified.as_nanos().to_le_bytes());
        }
    }
    Ok(hash.finalize().iter().map(|b| format!("{b:02x}")).collect())
}

fn collect_files(root: &Path, dir: &Path, out: &mut Vec<PathBuf>) -> std::io::Result<()> {
    if dir.is_file() {
        out.push(dir.strip_prefix(root).unwrap_or(dir).to_path_buf());
        return Ok(());
    }
    for entry in std::fs::read_dir(dir)? {
        collect_files(root, &entry?.path(), out)?;
    }
    Ok(())
}

fn load_cached(file: &Path) -> Option<OpCounts> {
    let value: serde_json::Value = serde_json::from_slice(&std::fs::read(file).ok()?).ok()?;
    let n = |k: &str| value.get(k)?.as_u64().map(|n| n as usize);
    let off_ane_ops = value
        .get("off_ane_ops")?
        .as_object()?
        .iter()
        .map(|(op, n)| Some((op.clone(), n.as_u64()? as usize)))
        .collect::<Option<_>>()?;
    Some(OpCounts { ane: n("ane")?, gpu: n("gpu")?, cpu: n("cpu")?, unassigned: n("unassigned")?, off_ane_ops })
}

/// Write via a temporary file and a rename, so a reader never sees half a
/// file.
fn store(file: &Path, counts: &OpCounts) -> std::io::Result<()> {
    let dir = file.parent().expect("a file in the cache directory");
    std::fs::create_dir_all(dir)?;
    let body = serde_json::json!({
        "ane": counts.ane,
        "gpu": counts.gpu,
        "cpu": counts.cpu,
        "unassigned": counts.unassigned,
        "off_ane_ops": counts.off_ane_ops,
    });
    let tmp = file.with_extension(format!("tmp{}", std::process::id()));
    std::fs::write(&tmp, body.to_string())?;
    std::fs::rename(&tmp, file)
}

/// The machine a plan was read on: what a recorded plan is compared with.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Machine {
    /// `sysctl machdep.cpu.brand_string`: `Apple M1 Max`.
    pub chip: String,
    /// `25A354`.
    pub macos_build: String,
}

/// This machine, read once. A field that can't be read is empty, so no
/// recorded plan matches it.
pub fn this_machine() -> &'static Machine {
    static MACHINE: OnceLock<Machine> = OnceLock::new();
    MACHINE.get_or_init(|| Machine { chip: chip(), macos_build: macos_build() })
}

fn chip() -> String {
    std::process::Command::new("/usr/sbin/sysctl")
        .args(["-n", "machdep.cpu.brand_string"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_default()
}

/// The macOS build (`25A123`): a new build can change the compiler and so
/// the plan. Empty when it can't be read, which only weakens the key.
fn macos_build() -> String {
    std::fs::read_to_string("/System/Library/CoreServices/SystemVersion.plist")
        .ok()
        .and_then(|plist| {
            let at = plist.find("<key>ProductBuildVersion</key>")?;
            let rest = &plist[at..];
            let start = rest.find("<string>")? + "<string>".len();
            let end = rest[start..].find("</string>")?;
            Some(rest[start..start + end].trim().to_string())
        })
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::{Duration, Instant};

    fn tmp(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("sk-placement-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// A fake artifact directory with one weight file.
    fn artifact(dir: &Path, name: &str, weights: &[u8]) -> PathBuf {
        let path = dir.join(name);
        std::fs::create_dir_all(path.join("weights")).unwrap();
        std::fs::write(path.join("model.mil"), b"program").unwrap();
        std::fs::write(path.join("weights/weight.bin"), weights).unwrap();
        path
    }

    fn wait(service: &Placements, path: &Path, units: ComputeUnits) -> Placement {
        let until = Instant::now() + Duration::from_secs(10);
        loop {
            match service.get(path, units) {
                Some(Placement::Pending) | None if Instant::now() < until => std::thread::sleep(Duration::from_millis(5)),
                other => return other.expect("a placement"),
            }
        }
    }

    /// A reader that counts its calls and reports `ane` operations.
    fn counting(calls: &Arc<AtomicUsize>, ane: usize) -> PlanReader {
        let calls = calls.clone();
        Arc::new(move |_: &Path, _| {
            calls.fetch_add(1, Ordering::SeqCst);
            Ok(OpCounts { ane, cpu: 2, off_ane_ops: BTreeMap::from([("gather".into(), 2)]), ..Default::default() })
        })
    }

    #[test]
    fn a_loaded_bucket_is_read_once_and_reported() {
        let dir = tmp("once");
        let path = artifact(&dir, "model_16.mlmodelc", b"w");
        let calls = Arc::new(AtomicUsize::new(0));
        let service = Placements::new(counting(&calls, 40), None);
        assert_eq!(service.get(&path, ComputeUnits::CpuAndNeuralEngine), None, "nothing before a load");
        service.loaded(&path, ComputeUnits::CpuAndNeuralEngine);
        let Placement::Ready(counts) = wait(&service, &path, ComputeUnits::CpuAndNeuralEngine) else { panic!() };
        assert_eq!((counts.ane, counts.cpu, counts.off_ane_ops["gather"]), (40, 2, 2));
        // Loaded again (after an eviction): already known, not read again.
        service.loaded(&path, ComputeUnits::CpuAndNeuralEngine);
        std::thread::sleep(Duration::from_millis(50));
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        // Other compute units are another plan.
        assert_eq!(service.get(&path, ComputeUnits::CpuAndGpu), None);
        let listed = service.for_model(&dir, "model_{seq}.mlmodelc", &[16, 32], 1, ComputeUnits::CpuAndNeuralEngine);
        assert_eq!(listed.keys().copied().collect::<Vec<_>>(), vec![16], "only the loaded bucket");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn a_chunked_bucket_is_its_chunks_together() {
        let dir = tmp("chunks");
        let units = ComputeUnits::CpuAndNeuralEngine;
        let c0 = artifact(&dir, "model_16.0.mlmodelc", b"w0");
        let c1 = artifact(&dir, "model_16.1.mlmodelc", b"w1");
        // chunk 0 reads 30 ANE operations, chunk 1 40
        let reader: PlanReader = Arc::new(|path: &Path, _| {
            let ane = if path.ends_with("model_16.0.mlmodelc") { 30 } else { 40 };
            Ok(OpCounts { ane, cpu: 1, off_ane_ops: BTreeMap::from([("gather".into(), 1)]), ..Default::default() })
        });
        let service = Placements::new(reader, None);
        let listed = |s: &Placements| s.for_model(&dir, "model_{seq}.{chunk}.mlmodelc", &[16], 2, units).remove(&16);
        assert_eq!(listed(&service), None, "nothing before a load");
        // One chunk read, the other not loaded yet: pending.
        service.loaded(&c0, units);
        wait(&service, &c0, units);
        assert_eq!(listed(&service), Some(Placement::Pending));
        // Both read: summed.
        service.loaded(&c1, units);
        wait(&service, &c1, units);
        let Some(Placement::Ready(sum)) = listed(&service) else { panic!("{:?}", listed(&service)) };
        assert_eq!((sum.ane, sum.cpu, sum.off_ane_ops["gather"]), (70, 2, 2));
        // A chunk whose read failed fails the bucket, naming the chunk.
        let failing = Placements::new(
            Arc::new(|path: &Path, _| match path.ends_with("model_16.1.mlmodelc") {
                true => Err(sidekick_core::Error::Inference("no plan".into())),
                false => Ok(OpCounts::default()),
            }),
            None,
        );
        failing.loaded(&c0, units);
        failing.loaded(&c1, units);
        wait(&failing, &c0, units);
        wait(&failing, &c1, units);
        let Some(Placement::Failed(why)) = listed(&failing) else { panic!() };
        assert!(why.starts_with("chunk 1:"), "{why}");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn plans_are_read_one_at_a_time() {
        let dir = tmp("serial");
        let active = Arc::new(AtomicUsize::new(0));
        let most = Arc::new(AtomicUsize::new(0));
        let (a, m) = (active.clone(), most.clone());
        let reader: PlanReader = Arc::new(move |_: &Path, _| {
            let now = a.fetch_add(1, Ordering::SeqCst) + 1;
            m.fetch_max(now, Ordering::SeqCst);
            std::thread::sleep(Duration::from_millis(20));
            a.fetch_sub(1, Ordering::SeqCst);
            Ok(OpCounts::default())
        });
        let service = Placements::new(reader, None);
        let paths: Vec<PathBuf> = (0..4).map(|i| artifact(&dir, &format!("model_{i}.mlmodelc"), b"w")).collect();
        let threads: Vec<_> = paths
            .iter()
            .cloned()
            .map(|p| {
                let s = service.clone();
                std::thread::spawn(move || s.loaded(&p, ComputeUnits::CpuAndNeuralEngine))
            })
            .collect();
        threads.into_iter().for_each(|t| t.join().unwrap());
        for p in &paths {
            assert!(matches!(wait(&service, p, ComputeUnits::CpuAndNeuralEngine), Placement::Ready(_)));
        }
        assert_eq!(most.load(Ordering::SeqCst), 1, "one plan read in flight");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn a_failed_read_is_reported_and_retried_on_the_next_load() {
        let dir = tmp("fail");
        let path = artifact(&dir, "model_16.mlmodelc", b"w");
        let calls = Arc::new(AtomicUsize::new(0));
        let c = calls.clone();
        let reader: PlanReader = Arc::new(move |_: &Path, _| match c.fetch_add(1, Ordering::SeqCst) {
            0 => Err(sidekick_core::Error::Inference("no plan".into())),
            _ => Ok(OpCounts { ane: 1, ..Default::default() }),
        });
        let service = Placements::new(reader, Some(dir.join("cache")));
        service.loaded(&path, ComputeUnits::CpuAndNeuralEngine);
        let failed = wait(&service, &path, ComputeUnits::CpuAndNeuralEngine);
        assert!(matches!(&failed, Placement::Failed(e) if e.contains("no plan")), "{failed:?}");
        assert!(!dir.join("cache").exists(), "a failure isn't cached");
        service.loaded(&path, ComputeUnits::CpuAndNeuralEngine);
        assert!(matches!(wait(&service, &path, ComputeUnits::CpuAndNeuralEngine), Placement::Ready(_)));
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn the_disk_cache_is_keyed_by_content_units_and_build() {
        let dir = tmp("cache");
        let cache = dir.join("cache");
        let a = artifact(&dir, "a.mlmodelc", b"weights");
        let copy = artifact(&dir, "copy.mlmodelc", b"weights");
        let other = artifact(&dir, "other.mlmodelc", b"other weights");
        let calls = Arc::new(AtomicUsize::new(0));
        let reader = counting(&calls, 7);
        let ne = ComputeUnits::CpuAndNeuralEngine;
        let counts = read(&reader, Some(&cache), "25A1", &a, ne).unwrap();
        assert_eq!(counts.ane, 7);
        // The same content at another path, as a restart would see it: from the cache.
        assert_eq!(read(&reader, Some(&cache), "25A1", &copy, ne).unwrap(), counts);
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        // Other weights, other units or another macOS build: read again.
        read(&reader, Some(&cache), "25A1", &other, ne).unwrap();
        read(&reader, Some(&cache), "25A1", &a, ComputeUnits::CpuAndGpu).unwrap();
        read(&reader, Some(&cache), "25B2", &a, ne).unwrap();
        assert_eq!(calls.load(Ordering::SeqCst), 4);
        // A damaged entry is read past, and rewritten.
        for entry in std::fs::read_dir(&cache).unwrap() {
            std::fs::write(entry.unwrap().path(), b"{").unwrap();
        }
        assert_eq!(read(&reader, Some(&cache), "25A1", &a, ne).unwrap(), counts);
        assert_eq!(calls.load(Ordering::SeqCst), 5);
        assert_eq!(read(&reader, Some(&cache), "25A1", &a, ne).unwrap(), counts);
        assert_eq!(calls.load(Ordering::SeqCst), 5);

        // Large weights count by length and time: rewritten, they're new.
        let big = artifact(&dir, "big.mlmodelc", &vec![1u8; HASHED_FILE_MAX as usize + 1]);
        let key = content_key(&big, ne, "25A1").unwrap();
        assert_eq!(content_key(&big, ne, "25A1").unwrap(), key);
        let weights = std::fs::File::options().write(true).open(big.join("weights/weight.bin")).unwrap();
        weights.set_modified(std::time::SystemTime::UNIX_EPOCH + Duration::from_secs(1)).unwrap();
        assert_ne!(content_key(&big, ne, "25A1").unwrap(), key);
        // The program is read whole.
        let key = content_key(&big, ne, "25A1").unwrap();
        std::fs::write(big.join("model.mil"), b"pRogram").unwrap();
        assert_ne!(content_key(&big, ne, "25A1").unwrap(), key);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn this_machine_is_read() {
        if cfg!(target_os = "macos") {
            let m = this_machine();
            assert!(!m.macos_build.is_empty() && m.macos_build.chars().all(|c| c.is_ascii_alphanumeric()), "{m:?}");
            assert!(!m.chip.is_empty(), "{m:?}");
        }
    }
}
