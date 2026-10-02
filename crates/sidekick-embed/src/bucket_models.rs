//! Per-bucket Core ML models, loaded lazily and kept resident: the cache
//! behind every Core ML backend. Each sequence-length bucket is its own
//! static-shape artifact (D15); without a `{seq}` placeholder every bucket
//! resolves to the same path and shares one entry.

use sidekick_core::Result;
use sidekick_coreml::{ComputeUnits, CoremlModel};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, PoisonError};

/// One artifact's slot. Its own lock serializes loads of that artifact only.
type Slot = Arc<Mutex<Option<Arc<CoremlModel>>>>;

pub(crate) struct BucketModels {
    dir: PathBuf,
    artifact: String,
    units: ComputeUnits,
    slots: Mutex<BTreeMap<PathBuf, Slot>>,
}

impl BucketModels {
    pub fn new(dir: &Path, artifact: &str, units: ComputeUnits) -> Self {
        Self {
            dir: dir.to_path_buf(),
            artifact: artifact.to_string(),
            units,
            slots: Mutex::new(BTreeMap::new()),
        }
    }

    /// The artifact file name for `bucket`, relative to the model directory.
    pub fn artifact_name(&self, bucket: usize) -> String {
        self.artifact.replace("{seq}", &bucket.to_string())
    }

    pub fn path(&self, bucket: usize) -> PathBuf {
        self.dir.join(self.artifact_name(bucket))
    }

    /// The model for `bucket`, loading it on first use. Concurrent first uses
    /// of one bucket load it once. Only that bucket's slot is locked during
    /// the load, so the other buckets keep serving, and a load that fails
    /// leaves the slot empty for the next call to retry.
    pub fn get(&self, bucket: usize) -> Result<Arc<CoremlModel>> {
        let path = self.path(bucket);
        let slot = self
            .slots
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .entry(path.clone())
            .or_default()
            .clone();
        let mut slot = slot.lock().unwrap_or_else(PoisonError::into_inner);
        if let Some(model) = slot.as_ref() {
            return Ok(model.clone());
        }
        let model = Arc::new(CoremlModel::load(&path, self.units)?);
        *slot = Some(model.clone());
        crate::placement::loaded(&path, self.units);
        Ok(model)
    }
}
