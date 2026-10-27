//! Core ML's compile cache, from sidekick's side: whether each bucket is
//! warm, and loading buckets before the requests that need them (D52).
//!
//! The first time a process loads a Core ML program on a compute path, Core
//! ML specializes it for the device: on the ANE a full compile, seconds for
//! a small model and minutes for a long bucket of a large one. It caches the
//! result per executable name, home (`~/Library/Caches/<executable>`),
//! macOS build, compute units and program (its path and its weights file),
//! and on the ANE also per code-signing identifier, the compiled program
//! itself living in the ANE compiler's system cache; later loads take
//! cache-hit time. Only the same executable can count on the same key, so
//! warming happens in sidekickd itself: in a child process of the daemon,
//! in `sidekickd warm` and in `sidekickd fetch`.
//!
//! This module:
//! - routes every bucket load in the process through one place ([`Loads`]),
//!   one load of a bucket at a time;
//! - knows each bucket's state ([`State`]): cold, warming, warm or failed;
//! - with a ledger ([`enable`]), remembers which buckets this executable has
//!   loaded under this home and macOS build, so a restarted daemon reports
//!   them warm and warms the cold ones first, and serializes loads of one
//!   bucket across processes with a lock file: a request that needs a
//!   bucket the warm-up process is compiling waits for that compile, then
//!   loads from the cache, rather than compiling it a second time.
//!
//! The ledger is sidekick's record, not Core ML's cache, which can be
//! deleted or purged without the ledger knowing. So it only orders the work
//! and seeds the reported state: a warm-up still loads every bucket it is
//! given, and only a load confirms one is warm.

use sidekick_core::{ComputeUnits, Result};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::time::{Instant, SystemTime};

/// A bucket's state in Core ML's compile cache, as far as sidekick knows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum State {
    /// Neither loaded by this process nor in the ledger: its first load
    /// compiles it.
    Cold,
    /// Being loaded now.
    Warming,
    /// Loaded by this process, or recorded in the ledger as loaded by this
    /// executable under this home and macOS build.
    Warm,
    /// The last load failed. The next one tries again.
    Failed,
}

impl State {
    /// What `/v1/models` and `/health` report.
    pub fn name(self) -> &'static str {
        match self {
            State::Cold => "cold",
            State::Warming => "warming",
            State::Warm => "warm",
            State::Failed => "failed",
        }
    }
}

/// One bucket's [`State`], and what it rests on.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Status {
    pub state: State,
    /// Whether this process has loaded the bucket. A warm bucket it hasn't
    /// loaded is warm by the ledger's record only.
    pub loaded: bool,
    /// How long this process's last load of the bucket took, in ms.
    pub load_ms: Option<u64>,
    /// How long the load the ledger recorded took, in ms: a compile, when
    /// Core ML's cache was cold then.
    pub recorded_ms: Option<u64>,
    /// The last load's error, when it failed.
    pub error: Option<String>,
}

/// What a bucket is: its programs, in chain order (D37), and the compute
/// units they load with. Core ML caches each program per path and compute
/// path, so this is what the cache is keyed by, as far as sidekick can see.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Key {
    pub paths: Vec<PathBuf>,
    pub units: ComputeUnits,
}

impl Key {
    /// `bucket`'s programs in the model directory `dir`, as the backends
    /// resolve them.
    pub fn bucket(dir: &Path, artifact: &str, bucket: usize, chunks: usize, units: ComputeUnits) -> Self {
        let paths = sidekick_core::artifact_files(artifact, bucket, chunks).iter().map(|f| dir.join(f)).collect();
        Self { paths, units }
    }
}

/// Which model and bucket a load is for, for the ledger and the logs.
#[derive(Debug, Clone, Copy)]
pub struct Label<'a> {
    pub model: &'a str,
    pub bucket: usize,
}

/// What a warm-up did.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Warmed {
    /// It loaded the bucket, in `load_ms`. `recorded`: the ledger already
    /// had it, so the load most likely hit Core ML's cache.
    Loaded { load_ms: u64, recorded: bool },
    /// This process had already loaded it: a request, or an earlier warm-up.
    AlreadyLoaded,
    /// Another process loaded it while this one waited for its lock file,
    /// and recorded it: no need to load it again.
    LoadedElsewhere,
}

/// What this process last did with a bucket.
#[derive(Debug, Clone, Default)]
enum Progress {
    #[default]
    Never,
    Loading,
    Loaded(u64),
    Failed(String),
}

#[derive(Default)]
struct Slot {
    /// Held for the whole of a load of the bucket.
    held: Mutex<()>,
    progress: Mutex<Progress>,
}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

/// Every bucket load in a process, keyed by [`Key`]: one load of a key at
/// a time, and each key's [`State`]. The daemon has one, for Core ML
/// programs; tests build their own over any `T`.
pub struct Loads<T> {
    slots: Mutex<HashMap<Key, Arc<Slot>>>,
    ledger: Mutex<Option<Ledger>>,
    _loads: std::marker::PhantomData<fn() -> T>,
}

impl<T> Default for Loads<T> {
    fn default() -> Self {
        Self { slots: Mutex::default(), ledger: Mutex::new(None), _loads: std::marker::PhantomData }
    }
}

impl<T> Loads<T> {
    /// Record loads in `ledger` from now on, and read states from it.
    pub fn set_ledger(&self, ledger: Ledger) {
        *lock(&self.ledger) = Some(ledger);
    }

    pub fn has_ledger(&self) -> bool {
        lock(&self.ledger).is_some()
    }

    fn slot(&self, key: &Key) -> Arc<Slot> {
        lock(&self.slots).entry(key.clone()).or_default().clone()
    }

    /// Load `key` for a request, with `load`. Waits for any load of it
    /// already running: in this process, or, with a ledger, in another
    /// sidekickd process (the daemon's warm-up), whose compile it then
    /// finds in the cache. Loads of other keys don't wait.
    pub fn load(&self, key: &Key, label: Label<'_>, load: impl FnOnce() -> Result<T>) -> Result<T> {
        self.load_as(key, label, load)
    }

    /// [`load`](Self::load) of something other than `T`, under the same
    /// keys, ledger and lock files: an open chat model's decoder (D53),
    /// whose chunk programs are its one bucket.
    pub fn load_as<U>(&self, key: &Key, label: Label<'_>, load: impl FnOnce() -> Result<U>) -> Result<U> {
        let slot = self.slot(key);
        let _held = lock(&slot.held);
        let (result, _) = self.run(&slot, key, label, false, load);
        result.expect("a load that may not skip runs")
    }

    /// Load `key` so that Core ML's cache holds it, unless this process
    /// has already loaded it, then release it: a warm-up keeps nothing
    /// resident. Waits for a load of it already running, here or in
    /// another process; if that one recorded the bucket meanwhile, it isn't
    /// loaded again.
    pub fn warm(&self, key: &Key, label: Label<'_>, load: impl FnOnce() -> Result<T>) -> Result<Warmed> {
        self.warm_as(key, label, load)
    }

    /// [`warm`](Self::warm) with a loader of something other than `T`, as
    /// [`load_as`](Self::load_as).
    pub fn warm_as<U>(&self, key: &Key, label: Label<'_>, load: impl FnOnce() -> Result<U>) -> Result<Warmed> {
        let slot = self.slot(key);
        let _held = lock(&slot.held);
        if matches!(*lock(&slot.progress), Progress::Loaded(_)) {
            return Ok(Warmed::AlreadyLoaded);
        }
        let recorded = self.recorded(key).is_some();
        let (result, load_ms) = self.run(&slot, key, label, !recorded, load);
        match result {
            None => Ok(Warmed::LoadedElsewhere),
            Some(loaded) => {
                drop(loaded?);
                Ok(Warmed::Loaded { load_ms, recorded })
            }
        }
    }

    /// One load of `key`, with its slot held, and how long it took: across
    /// processes too, when there is a ledger. `None` when `skip_if_recorded`
    /// and another process recorded the bucket while this one waited for it.
    fn run<U>(
        &self,
        slot: &Slot,
        key: &Key,
        label: Label<'_>,
        skip_if_recorded: bool,
        load: impl FnOnce() -> Result<U>,
    ) -> (Option<Result<U>>, u64) {
        let before = std::mem::replace(&mut *lock(&slot.progress), Progress::Loading);
        // Another sidekickd process loading the same bucket holds this lock
        // file: wait for its compile instead of starting a second one.
        let lock_path = lock(&self.ledger).as_mut().and_then(|l| l.lock_path(key));
        let _file_lock = lock_path.and_then(|p| match lock_file(&p) {
            Ok(file) => Some(file),
            Err(e) => {
                tracing::debug!("no lock file at {}: {e}", p.display());
                None
            }
        });
        if skip_if_recorded && self.recorded(key).is_some() {
            *lock(&slot.progress) = before;
            return (None, 0);
        }
        let started = Instant::now();
        let result = load();
        let ms = started.elapsed().as_millis() as u64;
        match &result {
            Ok(_) => {
                *lock(&slot.progress) = Progress::Loaded(ms);
                tracing::debug!(model = label.model, bucket = label.bucket, units = key.units.name(), "loaded in {ms} ms");
                if let Some(ledger) = lock(&self.ledger).as_mut() {
                    ledger.record(key, label, ms);
                }
            }
            Err(e) => *lock(&slot.progress) = Progress::Failed(e.to_string()),
        }
        (Some(result), ms)
    }

    fn recorded(&self, key: &Key) -> Option<u64> {
        lock(&self.ledger).as_mut().and_then(|l| l.recorded(key))
    }

    /// Record that a strict check refused `key` (an open chat model's
    /// decoder whose programs weren't confirmed on the Neural Engine), so
    /// that every process loading this build refuses it too.
    pub fn refuse(&self, key: &Key, label: Label<'_>, reason: &str) {
        if let Some(ledger) = lock(&self.ledger).as_mut() {
            ledger.refuse(key, label, reason);
        }
    }

    /// Why a strict check refused `key`, if one did.
    pub fn refusal(&self, key: &Key) -> Option<String> {
        lock(&self.ledger).as_mut().and_then(|l| l.refusal(key))
    }

    /// `key`'s state: this process's last load of it, else the ledger's
    /// record, else cold.
    pub fn status(&self, key: &Key) -> Status {
        let progress = lock(&self.slots).get(key).map(|s| lock(&s.progress).clone()).unwrap_or_default();
        let recorded_ms = self.recorded(key);
        let mut status = Status { state: State::Cold, loaded: false, load_ms: None, recorded_ms, error: None };
        match progress {
            Progress::Never => {
                if recorded_ms.is_some() {
                    status.state = State::Warm;
                }
            }
            Progress::Loading => status.state = State::Warming,
            Progress::Loaded(ms) => {
                status.state = State::Warm;
                status.loaded = true;
                status.load_ms = Some(ms);
            }
            Progress::Failed(e) => {
                status.state = State::Failed;
                status.error = Some(e);
            }
        }
        status
    }
}

/// Open `path` and take an exclusive lock on it, waiting for any other
/// holder. The lock goes with the file (or the process).
fn lock_file(path: &Path) -> std::io::Result<std::fs::File> {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let file = std::fs::OpenOptions::new().create(true).truncate(false).write(true).open(path)?;
    file.lock()?;
    Ok(file)
}

/// sidekick's record of the buckets this executable has loaded, under this
/// home and macOS build: a JSON file beside Core ML's cache, with a lock
/// file per bucket in a directory next to it.
///
/// An entry's key is a hash of what Core ML's cache was measured to depend
/// on (D52): the executable's file name (a copy of sidekickd at another
/// path hits the same cache; one under another name doesn't) and its
/// code-signing identifier, the home directory Core ML caches under, the
/// macOS build, the compute units, and each program's path, with the
/// lengths, inodes and birth times of its files, so that a bucket fetched
/// or converted again counts as cold. A key
/// that over-counts only makes a warm bucket look cold, which a warm-up's
/// load then corrects at cache-hit cost; one that under-counted would
/// report a cold bucket warm.
pub struct Ledger {
    file: PathBuf,
    locks: PathBuf,
    /// The executable, home and macOS build, hashed into every key.
    context: String,
    build: String,
    entries: serde_json::Map<String, serde_json::Value>,
    /// The file's modification time when last read: another process
    /// writing it is read again.
    read_at: Option<SystemTime>,
    fingerprints: HashMap<PathBuf, Option<String>>,
}

/// The ledger's file name, in its directory.
pub const LEDGER_FILE: &str = "coreml-warm.json";

impl Ledger {
    /// The ledger in `dir`, for this executable (`exe`) loading under
    /// `home` on macOS build `build`.
    pub fn open(dir: &Path, exe: &Path, home: &Path, build: &str) -> Self {
        let name = exe.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
        let mut ledger = Self {
            file: dir.join(LEDGER_FILE),
            locks: dir.join("coreml-warm-locks"),
            context: format!("sidekick warm v1\0{name}\0{}\0{build}\0", home.display()),
            build: build.to_string(),
            entries: serde_json::Map::new(),
            read_at: None,
            fingerprints: HashMap::new(),
        };
        ledger.refresh();
        ledger
    }

    /// Key entries also by the executable's code-signing identifier, which
    /// Core ML's ANE entries depend on: an executable re-signed under
    /// another identifier (any rebuild, unless it signs with a fixed one)
    /// finds its ANE buckets cold.
    pub fn signed_as(mut self, identifier: Option<&str>) -> Self {
        self.context.push_str(identifier.unwrap_or(""));
        self.context.push('\0');
        self
    }

    fn modified(&self) -> Option<SystemTime> {
        std::fs::metadata(&self.file).and_then(|m| m.modified()).ok()
    }

    /// Read the file again if it changed since it was last read.
    fn refresh(&mut self) {
        let modified = self.modified();
        if modified.is_some() && modified == self.read_at {
            return;
        }
        self.entries = std::fs::read(&self.file)
            .ok()
            .and_then(|b| serde_json::from_slice::<serde_json::Value>(&b).ok())
            .and_then(|v| v.get("entries")?.as_object().cloned())
            .unwrap_or_default();
        self.read_at = modified;
    }

    /// A program's files' paths, lengths, inodes and birth times, read once
    /// per process; `None` when it can't be read.
    fn fingerprint(&mut self, program: &Path) -> Option<String> {
        if let Some(f) = self.fingerprints.get(program) {
            return f.clone();
        }
        let f = fingerprint(program).ok();
        self.fingerprints.insert(program.to_path_buf(), f.clone());
        f
    }

    fn hash(&mut self, key: &Key) -> Option<String> {
        use sha2::{Digest, Sha256};
        let mut hash = Sha256::new();
        hash.update(self.context.as_bytes());
        hash.update(key.units.name());
        for path in &key.paths {
            hash.update(b"\0");
            hash.update(path.to_string_lossy().as_bytes());
            hash.update(b"\0");
            hash.update(self.fingerprint(path)?);
        }
        Some(hash.finalize().iter().map(|b| format!("{b:02x}")).collect())
    }

    /// The load time the ledger recorded for `key`, if it has it.
    fn recorded(&mut self, key: &Key) -> Option<u64> {
        let hash = self.hash(key)?;
        self.refresh();
        self.entries.get(&hash)?.get("load_ms")?.as_u64()
    }

    /// Record that `key` loaded in `load_ms`, unless it's recorded already:
    /// the first record keeps the load that filled the cache. Entries from
    /// other macOS builds are dropped, since Core ML's cache is per build.
    fn record(&mut self, key: &Key, label: Label<'_>, load_ms: u64) {
        let Some(hash) = self.hash(key) else { return };
        self.refresh();
        // A load replaces a refusal (a strict check another process made),
        // never another load's record.
        if self.entries.get(&hash).is_some_and(|e| e.get("load_ms").is_some()) {
            return;
        }
        let build = self.build.clone();
        self.entries.retain(|_, e| e.get("macos_build").and_then(|b| b.as_str()) == Some(build.as_str()));
        let at = SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).unwrap_or_default().as_secs();
        self.entries.insert(
            hash,
            serde_json::json!({
                "model": label.model,
                "bucket": label.bucket,
                "compute_units": key.units.name(),
                "macos_build": self.build,
                "load_ms": load_ms,
                "recorded_at": at,
            }),
        );
        if let Err(e) = self.write() {
            tracing::debug!("couldn't write {}: {e}", self.file.display());
        }
    }

    /// Write via a temporary file and a rename, so a reader never sees
    /// half a file. Two processes recording at once can lose one record,
    /// which only makes its bucket look cold until it loads again.
    fn write(&mut self) -> std::io::Result<()> {
        let dir = self.file.parent().expect("a file in a directory");
        std::fs::create_dir_all(dir)?;
        let body = serde_json::json!({ "version": 1, "entries": self.entries });
        let tmp = self.file.with_extension(format!("tmp{}", std::process::id()));
        std::fs::write(&tmp, serde_json::to_vec_pretty(&body)?)?;
        std::fs::rename(&tmp, &self.file)?;
        self.read_at = self.modified();
        Ok(())
    }

    /// Record a strict check's refusal of `key`.
    fn refuse(&mut self, key: &Key, label: Label<'_>, reason: &str) {
        let Some(hash) = self.hash(key) else { return };
        self.refresh();
        let at = SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).unwrap_or_default().as_secs();
        self.entries.insert(
            hash,
            serde_json::json!({
                "model": label.model,
                "bucket": label.bucket,
                "compute_units": key.units.name(),
                "macos_build": self.build,
                "refused": reason,
                "recorded_at": at,
            }),
        );
        if let Err(e) = self.write() {
            tracing::debug!("couldn't write {}: {e}", self.file.display());
        }
    }

    /// The refusal recorded for `key`, if any.
    fn refusal(&mut self, key: &Key) -> Option<String> {
        let hash = self.hash(key)?;
        self.refresh();
        Some(self.entries.get(&hash)?.get("refused")?.as_str()?.to_string())
    }

    fn lock_path(&mut self, key: &Key) -> Option<PathBuf> {
        let hash = self.hash(key)?;
        Some(self.locks.join(format!("{hash}.lock")))
    }
}

/// Each file of a program (a directory) or file, in path order: relative
/// path, length, inode and birth time.
fn fingerprint(program: &Path) -> std::io::Result<String> {
    fn walk(root: &Path, at: &Path, out: &mut Vec<String>) -> std::io::Result<()> {
        let meta = std::fs::metadata(at)?;
        if meta.is_dir() {
            for entry in std::fs::read_dir(at)? {
                walk(root, &entry?.path(), out)?;
            }
            return Ok(());
        }
        use std::os::unix::fs::MetadataExt;
        // Core ML's ANE entries know a weights file by its inode and birth
        // time, not its content or modification time (D52).
        let born = meta.created().ok().and_then(|t| t.duration_since(SystemTime::UNIX_EPOCH).ok()).unwrap_or_default();
        let rel = at.strip_prefix(root).unwrap_or(at);
        out.push(format!("{}\0{}\0{}\0{}", rel.display(), meta.len(), meta.ino(), born.as_nanos()));
        Ok(())
    }
    let mut files = Vec::new();
    walk(program, program, &mut files)?;
    files.sort();
    Ok(files.join("\n"))
}

#[cfg(all(feature = "coreml", target_os = "macos"))]
mod coreml {
    use super::*;
    use sidekick_coreml::CoremlChain;
    use std::sync::OnceLock;

    /// The process's Core ML loads.
    pub(crate) fn loads() -> &'static Loads<CoremlChain> {
        static LOADS: OnceLock<Loads<CoremlChain>> = OnceLock::new();
        LOADS.get_or_init(Loads::default)
    }

    /// Load a bucket's programs for a request, through [`loads`].
    pub(crate) fn load_chain(key: &Key, label: Label<'_>) -> Result<Arc<CoremlChain>> {
        loads().load(key, label, || CoremlChain::load(&key.paths, key.units)).map(Arc::new)
    }

    /// Keep a ledger from now on, beside Core ML's cache for this process:
    /// `<home>/Library/Caches/sidekick`, the home being the one Core ML
    /// caches under (`CFFIXED_USER_HOME` moves both). Returns its
    /// directory. Off unless called, so tools and tests that load models
    /// never write one.
    pub fn enable() -> Option<PathBuf> {
        let loads = loads();
        let home = sidekick_coreml::cache_home();
        let dir = home.join("Library").join("Caches").join("sidekick");
        if !loads.has_ledger() {
            let exe = std::env::current_exe().ok()?;
            let build = crate::placement::this_machine().macos_build.clone();
            let ledger = Ledger::open(&dir, &exe, &home, &build);
            loads.set_ledger(ledger.signed_as(sidekick_coreml::signing_identifier().as_deref()));
        }
        Some(dir)
    }

    /// Warm one bucket: see [`Loads::warm`].
    pub fn warm(key: &Key, label: Label<'_>) -> Result<Warmed> {
        loads().warm(key, label, || CoremlChain::load(&key.paths, key.units))
    }

    /// Load `key` with `load`, through the process's loads, ledger and
    /// lock files: see [`Loads::load_as`].
    pub fn load_with<U>(key: &Key, label: Label<'_>, load: impl FnOnce() -> Result<U>) -> Result<U> {
        loads().load_as(key, label, load)
    }

    /// Warm `key` with `load`: see [`Loads::warm_as`].
    pub fn warm_with<U>(key: &Key, label: Label<'_>, load: impl FnOnce() -> Result<U>) -> Result<Warmed> {
        loads().warm_as(key, label, load)
    }

    /// See [`Loads::refuse`].
    pub fn refuse(key: &Key, label: Label<'_>, reason: &str) {
        loads().refuse(key, label, reason)
    }

    /// See [`Loads::refusal`].
    pub fn refusal(key: &Key) -> Option<String> {
        loads().refusal(key)
    }

    /// One bucket's state.
    pub fn status(key: &Key) -> Option<Status> {
        Some(loads().status(key))
    }
}

#[cfg(all(feature = "coreml", target_os = "macos"))]
pub(crate) use coreml::load_chain;
#[cfg(all(feature = "coreml", target_os = "macos"))]
pub use coreml::{enable, load_with, refusal, refuse, status, warm, warm_with};

/// Without Core ML there is no cache to keep.
#[cfg(not(all(feature = "coreml", target_os = "macos")))]
pub fn enable() -> Option<PathBuf> {
    None
}

/// Without Core ML no bucket can be warmed.
#[cfg(not(all(feature = "coreml", target_os = "macos")))]
pub fn warm(key: &Key, label: Label<'_>) -> Result<Warmed> {
    let _ = (key, label);
    Err(sidekick_core::Error::Unavailable(sidekick_core::UnavailableReason::NotSupportedInBuild))
}

/// Without Core ML there is no ledger: `load` runs as it is.
#[cfg(not(all(feature = "coreml", target_os = "macos")))]
pub fn load_with<U>(key: &Key, label: Label<'_>, load: impl FnOnce() -> Result<U>) -> Result<U> {
    let _ = (key, label);
    load()
}

/// Without Core ML nothing can be warmed.
#[cfg(not(all(feature = "coreml", target_os = "macos")))]
pub fn warm_with<U>(key: &Key, label: Label<'_>, load: impl FnOnce() -> Result<U>) -> Result<Warmed> {
    let _ = (key, label, load);
    Err(sidekick_core::Error::Unavailable(sidekick_core::UnavailableReason::NotSupportedInBuild))
}

/// Without Core ML there is no ledger to record a refusal in.
#[cfg(not(all(feature = "coreml", target_os = "macos")))]
pub fn refuse(key: &Key, label: Label<'_>, reason: &str) {
    let _ = (key, label, reason);
}

/// Without Core ML nothing was refused.
#[cfg(not(all(feature = "coreml", target_os = "macos")))]
pub fn refusal(key: &Key) -> Option<String> {
    let _ = key;
    None
}

/// Without Core ML a bucket has no state.
#[cfg(not(all(feature = "coreml", target_os = "macos")))]
pub fn status(key: &Key) -> Option<Status> {
    let _ = key;
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use sidekick_core::Error;
    use std::sync::mpsc;
    use std::time::Duration;

    fn tmp(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("sk-warm-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// A bucket directory with one program file in it.
    fn program(dir: &Path, name: &str) -> Key {
        let path = dir.join(name);
        std::fs::create_dir_all(&path).unwrap();
        std::fs::write(path.join("model.mil"), name).unwrap();
        Key { paths: vec![path], units: ComputeUnits::CpuAndNeuralEngine }
    }

    const LABEL: Label<'static> = Label { model: "m", bucket: 128 };

    fn wait_for(what: &str, mut done: impl FnMut() -> bool) {
        let start = Instant::now();
        while !done() {
            assert!(start.elapsed() < Duration::from_secs(10), "timed out waiting for {what}");
            std::thread::sleep(Duration::from_millis(2));
        }
    }

    #[test]
    fn a_request_for_a_bucket_being_warmed_waits_for_that_load() {
        let loads: Arc<Loads<String>> = Arc::default();
        let key = Key { paths: vec!["/m/model_128.mlmodelc".into()], units: ComputeUnits::CpuAndNeuralEngine };
        let order = Arc::new(Mutex::new(Vec::new()));
        let (release, gate) = mpsc::channel::<()>();
        let warm = {
            let (loads, key, order) = (loads.clone(), key.clone(), order.clone());
            std::thread::spawn(move || {
                loads.warm(&key, LABEL, || {
                    gate.recv().unwrap();
                    lock(&order).push("warm-up");
                    Ok("compiled".to_string())
                })
            })
        };
        wait_for("the warm-up to start", || loads.status(&key).state == State::Warming);
        let request = {
            let (loads, key, order) = (loads.clone(), key.clone(), order.clone());
            std::thread::spawn(move || {
                loads.load(&key, LABEL, || {
                    lock(&order).push("request");
                    Ok("cache hit".to_string())
                })
            })
        };
        std::thread::sleep(Duration::from_millis(30));
        assert!(lock(&order).is_empty(), "the request loaded during the warm-up's load");
        release.send(()).unwrap();
        assert!(matches!(warm.join().unwrap().unwrap(), Warmed::Loaded { recorded: false, .. }));
        assert_eq!(request.join().unwrap().unwrap(), "cache hit");
        assert_eq!(*lock(&order), ["warm-up", "request"], "one after the other");
        let status = loads.status(&key);
        assert_eq!((status.state, status.loaded), (State::Warm, true));
    }

    #[test]
    fn a_loaded_bucket_isnt_warmed_again() {
        let loads: Loads<String> = Loads::default();
        let key = Key { paths: vec!["/m/a".into()], units: ComputeUnits::CpuAndGpu };
        assert_eq!(loads.status(&key).state, State::Cold);
        let warmed = loads.warm(&key, LABEL, || Ok("a".into())).unwrap();
        assert!(matches!(warmed, Warmed::Loaded { recorded: false, .. }), "{warmed:?}");
        assert_eq!(loads.warm(&key, LABEL, || panic!("loaded twice")).unwrap(), Warmed::AlreadyLoaded);
        // A bucket a request loaded is warm too, and a warm-up skips it.
        let other = Key { paths: vec!["/m/b".into()], units: ComputeUnits::CpuAndGpu };
        loads.load(&other, LABEL, || Ok("b".into())).unwrap();
        assert_eq!(loads.status(&other).state, State::Warm);
        assert_eq!(loads.warm(&other, LABEL, || panic!("loaded twice")).unwrap(), Warmed::AlreadyLoaded);
        // The same paths on other compute units are another key.
        let gpu = Key { units: ComputeUnits::CpuAndNeuralEngine, ..other };
        assert_eq!(loads.status(&gpu).state, State::Cold);
    }

    #[test]
    fn a_failed_load_reports_failed_and_the_next_one_retries() {
        let loads: Loads<String> = Loads::default();
        let key = Key { paths: vec!["/m/x".into()], units: ComputeUnits::CpuOnly };
        let err = loads.warm(&key, LABEL, || Err(Error::Inference("no ANE today".into()))).unwrap_err();
        assert!(err.to_string().contains("no ANE today"));
        let status = loads.status(&key);
        assert_eq!(status.state, State::Failed);
        assert!(status.error.unwrap().contains("no ANE today"));
        assert_eq!(loads.load(&key, LABEL, || Ok("x".into())).unwrap(), "x");
        assert_eq!(loads.status(&key).state, State::Warm);
    }

    #[test]
    fn loads_of_different_buckets_dont_wait_for_each_other() {
        let loads: Arc<Loads<String>> = Arc::default();
        let slow = Key { paths: vec!["/m/slow".into()], units: ComputeUnits::CpuAndNeuralEngine };
        let fast = Key { paths: vec!["/m/fast".into()], units: ComputeUnits::CpuAndNeuralEngine };
        let (release, gate) = mpsc::channel::<()>();
        let warm = {
            let (loads, slow) = (loads.clone(), slow.clone());
            std::thread::spawn(move || {
                loads.warm(&slow, LABEL, || {
                    gate.recv().unwrap();
                    Ok("slow".to_string())
                })
            })
        };
        wait_for("the warm-up to start", || loads.status(&slow).state == State::Warming);
        // Answers while the other bucket is still loading.
        assert_eq!(loads.load(&fast, LABEL, || Ok("fast".into())).unwrap(), "fast");
        assert_eq!(loads.status(&slow).state, State::Warming);
        release.send(()).unwrap();
        warm.join().unwrap().unwrap();
    }

    #[test]
    fn the_ledger_survives_a_restart_and_is_keyed_by_build_executable_and_program() {
        let dir = tmp("ledger");
        let cache = dir.join("cache");
        let key = program(&dir, "model_128.mlmodelc");
        let other = program(&dir, "model_256.mlmodelc");
        let (exe, home) = (Path::new("/bin/sidekickd"), Path::new("/homes/a"));
        let first: Loads<String> = Loads::default();
        first.set_ledger(Ledger::open(&cache, exe, home, "26A1"));
        assert_eq!(first.status(&key).state, State::Cold);
        first.warm(&key, LABEL, || Ok("compiled".into())).unwrap();

        // A new process: warm by the record, not loaded; the other bucket is cold.
        let restarted: Loads<String> = Loads::default();
        restarted.set_ledger(Ledger::open(&cache, exe, home, "26A1"));
        let status = restarted.status(&key);
        assert_eq!((status.state, status.loaded, status.load_ms), (State::Warm, false, None));
        assert!(status.recorded_ms.is_some());
        assert_eq!(restarted.status(&other).state, State::Cold);
        // Recorded is not loaded: a warm-up still loads it, and says it was recorded.
        let warmed = restarted.warm(&key, LABEL, || Ok("cache hit".into())).unwrap();
        assert!(matches!(warmed, Warmed::Loaded { recorded: true, .. }), "{warmed:?}");

        // The same executable name at another path shares Core ML's cache.
        let moved: Loads<String> = Loads::default();
        moved.set_ledger(Ledger::open(&cache, Path::new("/opt/sidekickd"), home, "26A1"));
        assert_eq!(moved.status(&key).state, State::Warm);
        // Another macOS build, executable name or home: cold.
        for (exe, home, build) in [(exe, home, "26B2"), (Path::new("/bin/other"), home, "26A1"), (exe, Path::new("/tmp/h"), "26A1")] {
            let loads: Loads<String> = Loads::default();
            loads.set_ledger(Ledger::open(&cache, exe, home, build));
            assert_eq!(loads.status(&key).state, State::Cold, "{exe:?} {home:?} {build}");
        }
        // Another signing identifier (a rebuilt sidekickd): cold.
        let resigned: Loads<String> = Loads::default();
        resigned.set_ledger(Ledger::open(&cache, exe, home, "26A1").signed_as(Some("sidekickd-0123")));
        assert_eq!(resigned.status(&key).state, State::Cold);
        let reopened = || {
            let loads: Loads<String> = Loads::default();
            loads.set_ledger(Ledger::open(&cache, exe, home, "26A1"));
            loads.status(&key).state
        };
        // A new modification time alone: Core ML still hits, so still warm.
        let file = key.paths[0].join("model.mil");
        std::fs::File::options()
            .write(true)
            .open(&file)
            .unwrap()
            .set_modified(SystemTime::now() + Duration::from_secs(60))
            .unwrap();
        assert_eq!(reopened(), State::Warm, "a modification time isn't part of the key");
        // The same bytes in a new file (a new inode and birth time), as a
        // fetch or a copy back puts them: cold.
        let bytes = std::fs::read(&file).unwrap();
        std::fs::remove_file(&file).unwrap();
        std::fs::write(&file, bytes).unwrap();
        assert_eq!(reopened(), State::Cold, "a new inode is a new entry");

        // Recording on a new build drops the old build's entries.
        let upgraded: Loads<String> = Loads::default();
        upgraded.set_ledger(Ledger::open(&cache, exe, home, "26B2"));
        upgraded.warm(&other, LABEL, || Ok("compiled".into())).unwrap();
        let file: serde_json::Value = serde_json::from_slice(&std::fs::read(cache.join(LEDGER_FILE)).unwrap()).unwrap();
        let entries = file["entries"].as_object().unwrap();
        assert_eq!(entries.len(), 1, "{entries:?}");
        assert_eq!(entries.values().next().unwrap()["macos_build"], "26B2");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_strict_refusal_binds_every_process_until_a_load_replaces_it() {
        let dir = tmp("refusal");
        let cache = dir.join("cache");
        let key = program(&dir, "decoder.0.mlmodelc");
        let (exe, home) = (Path::new("/bin/sidekickd"), Path::new("/homes/a"));
        let child: Loads<String> = Loads::default();
        child.set_ledger(Ledger::open(&cache, exe, home, "26A1"));
        let daemon: Loads<String> = Loads::default();
        daemon.set_ledger(Ledger::open(&cache, exe, home, "26A1"));
        assert_eq!(daemon.refusal(&key), None);
        std::thread::sleep(Duration::from_millis(5));
        child.refuse(&key, LABEL, "its programs aren't confirmed on the Neural Engine");
        assert_eq!(daemon.refusal(&key).as_deref(), Some("its programs aren't confirmed on the Neural Engine"));
        assert_eq!(daemon.status(&key).state, State::Cold, "a refusal isn't a load");
        // Another macOS build doesn't see it, nor does another build of the
        // program (its files' fingerprints differ).
        let other: Loads<String> = Loads::default();
        other.set_ledger(Ledger::open(&cache, exe, home, "26A2"));
        assert_eq!(other.refusal(&key), None);
        // A load that succeeds replaces it.
        std::thread::sleep(Duration::from_millis(5));
        child.load(&key, LABEL, || Ok("loaded".into())).unwrap();
        assert_eq!(daemon.refusal(&key), None);
        assert_eq!(daemon.status(&key).state, State::Warm);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_ledger_sees_another_processs_records() {
        let dir = tmp("shared");
        let cache = dir.join("cache");
        let key = program(&dir, "model_128.mlmodelc");
        let (exe, home) = (Path::new("/bin/sidekickd"), Path::new("/homes/a"));
        let daemon: Loads<String> = Loads::default();
        daemon.set_ledger(Ledger::open(&cache, exe, home, "26A1"));
        assert_eq!(daemon.status(&key).state, State::Cold);
        // `sidekickd warm`, in another process, records it.
        std::thread::sleep(Duration::from_millis(5));
        let cli: Loads<String> = Loads::default();
        cli.set_ledger(Ledger::open(&cache, exe, home, "26A1"));
        cli.warm(&key, LABEL, || Ok("compiled".into())).unwrap();
        assert_eq!(daemon.status(&key).state, State::Warm, "the daemon reads the changed file");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_load_waits_for_another_processs_load_of_the_same_bucket() {
        let dir = tmp("flock");
        let cache = dir.join("cache");
        let key = program(&dir, "model_128.mlmodelc");
        let (exe, home) = (Path::new("/bin/sidekickd"), Path::new("/homes/a"));
        // Two ledgers stand for two processes: each opens its own lock file,
        // and flock(2) locks conflict between open files.
        let (a, b): (Arc<Loads<String>>, Arc<Loads<String>>) = (Arc::default(), Arc::default());
        a.set_ledger(Ledger::open(&cache, exe, home, "26A1"));
        b.set_ledger(Ledger::open(&cache, exe, home, "26A1"));
        let (release, gate) = mpsc::channel::<()>();
        let (started_tx, started) = mpsc::channel::<()>();
        let order = Arc::new(Mutex::new(Vec::new()));
        let first = {
            let (a, key, order) = (a.clone(), key.clone(), order.clone());
            std::thread::spawn(move || {
                a.warm(&key, LABEL, || {
                    started_tx.send(()).unwrap();
                    gate.recv().unwrap();
                    lock(&order).push("a");
                    Ok("a".into())
                })
            })
        };
        // Loading, so holding the lock file.
        started.recv().unwrap();
        let second = {
            let (b, key, order) = (b.clone(), key.clone(), order.clone());
            std::thread::spawn(move || {
                b.warm(&key, LABEL, || {
                    lock(&order).push("b");
                    Ok("b".into())
                })
            })
        };
        std::thread::sleep(Duration::from_millis(50));
        assert!(lock(&order).is_empty(), "the second process loaded while the first held the lock");
        assert_eq!(b.status(&key).state, State::Warming);
        release.send(()).unwrap();
        first.join().unwrap().unwrap();
        // The first recorded it while the second waited: not loaded again.
        assert_eq!(second.join().unwrap().unwrap(), Warmed::LoadedElsewhere);
        assert_eq!(*lock(&order), ["a"]);
        assert_eq!(b.status(&key).state, State::Warm);
        // A request in the second process still loads it, after the first.
        assert_eq!(b.load(&key, LABEL, || Ok("b".into())).unwrap(), "b");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
