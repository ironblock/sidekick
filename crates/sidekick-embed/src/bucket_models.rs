//! Per-bucket Core ML models, loaded lazily and kept resident: the cache
//! behind every Core ML backend. Each sequence-length bucket is its own
//! static-shape artifact (D15); without a `{seq}` placeholder every bucket
//! resolves to the same path and shares one entry.

use sidekick_core::Result;
use sidekick_coreml::{ComputeUnits, CoremlModel};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

/// Checks a freshly loaded bucket model before it's cached, e.g. that its
/// input shapes match the manifest.
pub(crate) type LoadCheck = Box<dyn Fn(&CoremlModel, &Path) -> Result<()> + Send + Sync>;

pub(crate) struct BucketModels {
    dir: PathBuf,
    artifact: String,
    units: ComputeUnits,
    check: Option<LoadCheck>,
    models: Mutex<BTreeMap<PathBuf, Arc<CoremlModel>>>,
}

impl BucketModels {
    pub fn new(dir: &Path, artifact: &str, units: ComputeUnits) -> Self {
        Self {
            dir: dir.to_path_buf(),
            artifact: artifact.to_string(),
            units,
            check: None,
            models: Mutex::new(BTreeMap::new()),
        }
    }

    pub fn with_check(mut self, check: LoadCheck) -> Self {
        self.check = Some(check);
        self
    }

    pub fn path(&self, bucket: usize) -> PathBuf {
        self.dir.join(self.artifact.replace("{seq}", &bucket.to_string()))
    }

    /// The model for `bucket`, loading it on first use. The lock is held
    /// across the load so concurrent first uses don't load twice.
    pub fn get(&self, bucket: usize) -> Result<Arc<CoremlModel>> {
        let path = self.path(bucket);
        let mut models = self.models.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(m) = models.get(&path) {
            return Ok(m.clone());
        }
        let model = CoremlModel::load(&path, self.units)?;
        if let Some(check) = &self.check {
            check(&model, &path)?;
        }
        let model = Arc::new(model);
        models.insert(path, model.clone());
        Ok(model)
    }
}
