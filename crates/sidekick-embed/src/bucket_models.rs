//! Per-bucket Core ML models, loaded lazily and kept resident: the cache
//! behind every Core ML backend. Each sequence-length bucket is its own
//! static-shape artifact (D15); without a `{seq}` placeholder every bucket
//! resolves to the same path and shares one entry. A chunked model's bucket
//! is a chain of programs, its `{chunk}` placeholder numbering them (D37),
//! loaded and kept together.

use sidekick_core::Result;
use sidekick_coreml::{ComputeUnits, CoremlChain};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, PoisonError, TryLockError};

/// One bucket's slot. Its own lock serializes loads of that bucket only.
type Slot = Arc<Mutex<Option<Arc<CoremlChain>>>>;

pub(crate) struct BucketModels {
    dir: PathBuf,
    artifact: String,
    chunks: usize,
    units: ComputeUnits,
    slots: Mutex<BTreeMap<PathBuf, Slot>>,
}

impl BucketModels {
    pub fn new(dir: &Path, artifact: &str, chunks: usize, units: ComputeUnits) -> Self {
        Self {
            dir: dir.to_path_buf(),
            artifact: artifact.to_string(),
            chunks,
            units,
            slots: Mutex::new(BTreeMap::new()),
        }
    }

    /// The artifact file name for `bucket`, relative to the model directory
    /// (its first chunk's, for a chunked model).
    pub fn artifact_name(&self, bucket: usize) -> String {
        self.artifact_names(bucket).swap_remove(0)
    }

    pub fn path(&self, bucket: usize) -> PathBuf {
        self.dir.join(self.artifact_name(bucket))
    }

    /// Every program of `bucket`, in chain order, relative to the model
    /// directory.
    pub fn artifact_names(&self, bucket: usize) -> Vec<String> {
        sidekick_core::artifact_files(&self.artifact, bucket, self.chunks)
    }

    pub fn paths(&self, bucket: usize) -> Vec<PathBuf> {
        self.artifact_names(bucket).iter().map(|n| self.dir.join(n)).collect()
    }

    /// The compute units every bucket loads with.
    pub fn units(&self) -> ComputeUnits {
        self.units
    }

    /// The buckets loaded and resident, smallest first (one being loaded
    /// right now isn't yet).
    pub fn resident(&self, buckets: &[usize]) -> Vec<usize> {
        let slots = self.slots.lock().unwrap_or_else(PoisonError::into_inner);
        buckets
            .iter()
            .copied()
            .filter(|&b| slots.get(&self.paths(b)[0]).is_some_and(|s| s.try_lock().is_ok_and(|m| m.is_some())))
            .collect()
    }

    /// The programs for `bucket`, loading them on first use. Concurrent
    /// first uses of one bucket load it once. Only that bucket's slot is
    /// locked during the load, so the other buckets keep serving, and a load
    /// that fails leaves the slot empty for the next call to retry.
    pub fn get(&self, bucket: usize) -> Result<Arc<CoremlChain>> {
        Ok(self.get_tracked(bucket)?.0)
    }

    /// Load `bucket` unless it's resident, without running it. True when
    /// this call waited on a load: its own, or another caller's of the same
    /// bucket. The daemon loads a request's buckets this way before its
    /// deadline starts, since a first load compiles the bucket and can take
    /// minutes.
    pub fn ensure(&self, bucket: usize) -> Result<bool> {
        Ok(self.get_tracked(bucket)?.1)
    }

    /// [`get`](Self::get), and whether it waited on a load.
    fn get_tracked(&self, bucket: usize) -> Result<(Arc<CoremlChain>, bool)> {
        let paths = self.paths(bucket);
        let slot = self
            .slots
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .entry(paths[0].clone())
            .or_default()
            .clone();
        // A held slot is a load in progress: waiting on it is waiting on a load.
        let (mut slot, waited) = match slot.try_lock() {
            Ok(slot) => (slot, false),
            Err(TryLockError::Poisoned(e)) => (e.into_inner(), false),
            Err(TryLockError::WouldBlock) => (slot.lock().unwrap_or_else(PoisonError::into_inner), true),
        };
        if let Some(model) = slot.as_ref() {
            return Ok((model.clone(), waited));
        }
        let model = Arc::new(CoremlChain::load(&paths, self.units)?);
        *slot = Some(model.clone());
        for path in &paths {
            crate::placement::loaded(path, self.units);
        }
        Ok((model, true))
    }
}
