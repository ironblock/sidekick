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
use std::sync::{Arc, Mutex, PoisonError};

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

    /// The programs for `bucket`, loading them on first use. Concurrent
    /// first uses of one bucket load it once. Only that bucket's slot is
    /// locked during the load, so the other buckets keep serving, and a load
    /// that fails leaves the slot empty for the next call to retry.
    pub fn get(&self, bucket: usize) -> Result<Arc<CoremlChain>> {
        let paths = self.paths(bucket);
        let slot = self
            .slots
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .entry(paths[0].clone())
            .or_default()
            .clone();
        let mut slot = slot.lock().unwrap_or_else(PoisonError::into_inner);
        if let Some(model) = slot.as_ref() {
            return Ok(model.clone());
        }
        let model = Arc::new(CoremlChain::load(&paths, self.units)?);
        *slot = Some(model.clone());
        for path in &paths {
            crate::placement::loaded(path, self.units);
        }
        Ok(model)
    }
}
