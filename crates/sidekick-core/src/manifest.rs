//! On-disk model registry.
//!
//! A models directory contains one subdirectory per model, each with a
//! `manifest.toml` describing the artifact, its tokenizer, and how to run it:
//!
//! ```text
//! ~/.local/share/sidekick/models/
//!   embeddinggemma-300m/
//!     manifest.toml
//!     model.mlpackage/            # or model.mlmodelc, model.safetensors
//!     tokenizer.json
//! ```

use crate::{Error, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EmbeddingBackendKind {
    /// Core ML encoder, intended for the ANE (macOS only).
    Coreml,
    /// Static token-embedding lookup (model2vec-style); runs anywhere on CPU.
    Static,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Pooling {
    #[default]
    Mean,
    Cls,
    /// Model output is already pooled; take it as-is.
    None,
}

/// `manifest.toml` for an embedding model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelManifest {
    /// Stable id; exposed as the OpenAI `model` name.
    pub id: String,
    pub backend: EmbeddingBackendKind,
    /// Artifact path relative to the manifest's directory. For the coreml
    /// backend this may contain a `{seq}` placeholder resolved against each
    /// bucket (`model_{seq}.mlmodelc` → `model_128.mlmodelc`, …): one static
    /// artifact per bucket. Hardware verification showed a single
    /// enumerated-shapes artifact is rejected by the ANE/CPU (Espresso) path
    /// at plan time and falls back to CPU entirely; per-bucket static shapes
    /// are what actually keep the encoder on the ANE.
    pub artifact: String,
    /// tokenizer.json path relative to the manifest's directory.
    pub tokenizer: String,
    /// Native output dimensionality.
    pub dims: usize,
    /// Matryoshka truncation dims (largest first). Empty = unsupported.
    #[serde(default)]
    pub matryoshka: Vec<usize>,
    #[serde(default)]
    pub pooling: Pooling,
    /// Enumerated sequence-length buckets baked into the Core ML artifact.
    /// Inputs are padded to the smallest bucket that fits. Required for the
    /// coreml backend (flexible shapes push work off the ANE).
    #[serde(default)]
    pub buckets: Vec<usize>,
    /// Hard cap on input tokens; longer inputs are truncated.
    pub max_seq_len: usize,
    /// Feature names in the Core ML model.
    #[serde(default)]
    pub io: CoremlIoNames,
    /// Prompt prefixes some models require (e.g. EmbeddingGemma).
    #[serde(default)]
    pub prefixes: Prefixes,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoremlIoNames {
    pub input_ids: String,
    pub attention_mask: Option<String>,
    pub output: String,
}

impl Default for CoremlIoNames {
    fn default() -> Self {
        Self {
            input_ids: "input_ids".into(),
            attention_mask: Some("attention_mask".into()),
            output: "embeddings".into(),
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Prefixes {
    #[serde(default)]
    pub query: String,
    #[serde(default)]
    pub document: String,
}

/// A manifest resolved against its directory.
#[derive(Debug, Clone)]
pub struct ResolvedModel {
    pub manifest: ModelManifest,
    pub dir: PathBuf,
}

impl ResolvedModel {
    pub fn artifact_path(&self) -> PathBuf {
        self.dir.join(&self.manifest.artifact)
    }
    /// Artifact path for one sequence-length bucket: resolves a `{seq}`
    /// placeholder if present, otherwise the shared artifact path.
    pub fn artifact_path_for_bucket(&self, bucket: usize) -> PathBuf {
        self.dir
            .join(self.manifest.artifact.replace("{seq}", &bucket.to_string()))
    }
    pub fn tokenizer_path(&self) -> PathBuf {
        self.dir.join(&self.manifest.tokenizer)
    }
}

/// Scans a models directory for `*/manifest.toml`.
#[derive(Debug, Default)]
pub struct ModelRegistry {
    models: BTreeMap<String, ResolvedModel>,
}

impl ModelRegistry {
    pub fn scan(models_dir: &Path) -> Result<Self> {
        let mut models = BTreeMap::new();
        if !models_dir.exists() {
            return Ok(Self { models });
        }
        for entry in std::fs::read_dir(models_dir)? {
            let dir = entry?.path();
            let manifest_path = dir.join("manifest.toml");
            if !manifest_path.is_file() {
                continue;
            }
            let raw = std::fs::read_to_string(&manifest_path)?;
            let manifest: ModelManifest =
                toml::from_str(&raw).map_err(|e| Error::InvalidManifest {
                    path: manifest_path.display().to_string(),
                    message: e.to_string(),
                })?;
            Self::validate(&manifest, &manifest_path)?;
            let id = manifest.id.clone();
            if models
                .insert(id.clone(), ResolvedModel { manifest, dir })
                .is_some()
            {
                return Err(Error::InvalidManifest {
                    path: manifest_path.display().to_string(),
                    message: format!("duplicate model id `{id}`"),
                });
            }
        }
        Ok(Self { models })
    }

    fn validate(m: &ModelManifest, path: &Path) -> Result<()> {
        let fail = |message: String| {
            Err(Error::InvalidManifest { path: path.display().to_string(), message })
        };
        if m.id.is_empty() {
            return fail("empty model id".into());
        }
        if m.dims == 0 {
            return fail("dims must be > 0".into());
        }
        if let Some(&first) = m.matryoshka.first() {
            if first != m.dims {
                return fail(format!(
                    "matryoshka must start at native dims {} (got {first})",
                    m.dims
                ));
            }
            if m.matryoshka.windows(2).any(|w| w[0] <= w[1]) {
                return fail("matryoshka dims must be strictly decreasing".into());
            }
        }
        if m.backend == EmbeddingBackendKind::Coreml {
            if m.buckets.is_empty() {
                return fail("coreml backend requires sequence-length `buckets`".into());
            }
            if m.buckets.windows(2).any(|w| w[0] >= w[1]) {
                return fail("buckets must be strictly increasing".into());
            }
            if *m.buckets.last().unwrap() != m.max_seq_len {
                return fail("largest bucket must equal max_seq_len".into());
            }
        }
        if m.backend != EmbeddingBackendKind::Coreml && m.artifact.contains("{seq}") {
            return fail("`{seq}` artifact placeholder is only valid for the coreml backend".into());
        }
        Ok(())
    }

    pub fn get(&self, id: &str) -> Result<&ResolvedModel> {
        self.models
            .get(id)
            .ok_or_else(|| Error::ModelNotFound(id.to_string()))
    }

    pub fn ids(&self) -> impl Iterator<Item = &str> {
        self.models.keys().map(|s| s.as_str())
    }

    pub fn iter(&self) -> impl Iterator<Item = &ResolvedModel> {
        self.models.values()
    }

    pub fn is_empty(&self) -> bool {
        self.models.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_manifest(dir: &Path, name: &str, body: &str) {
        let d = dir.join(name);
        std::fs::create_dir_all(&d).unwrap();
        std::fs::write(d.join("manifest.toml"), body).unwrap();
    }

    #[test]
    fn scans_and_validates() {
        let tmp = std::env::temp_dir().join(format!("sk-registry-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        write_manifest(
            &tmp,
            "gemma",
            r#"
id = "embeddinggemma-300m"
backend = "coreml"
artifact = "model.mlpackage"
tokenizer = "tokenizer.json"
dims = 768
matryoshka = [768, 512, 256, 128]
buckets = [128, 256, 512]
max_seq_len = 512

[prefixes]
query = "task: search result | query: "
document = "title: none | text: "
"#,
        );
        write_manifest(
            &tmp,
            "floor",
            r#"
id = "static-minilm"
backend = "static"
artifact = "model.safetensors"
tokenizer = "tokenizer.json"
dims = 256
max_seq_len = 512
"#,
        );
        let reg = ModelRegistry::scan(&tmp).unwrap();
        assert_eq!(reg.ids().collect::<Vec<_>>(), vec!["embeddinggemma-300m", "static-minilm"]);
        let g = reg.get("embeddinggemma-300m").unwrap();
        assert_eq!(g.manifest.matryoshka, vec![768, 512, 256, 128]);
        assert_eq!(g.manifest.io.input_ids, "input_ids");
        assert!(g.artifact_path().ends_with("gemma/model.mlpackage"));
        std::fs::remove_dir_all(&tmp).unwrap();
    }

    #[test]
    fn rejects_bad_matryoshka_and_buckets() {
        let tmp = std::env::temp_dir().join(format!("sk-registry-bad-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        write_manifest(
            &tmp,
            "bad",
            r#"
id = "bad"
backend = "coreml"
artifact = "m"
tokenizer = "t"
dims = 768
matryoshka = [512, 256]
buckets = [128, 256, 512]
max_seq_len = 512
"#,
        );
        assert!(matches!(
            ModelRegistry::scan(&tmp),
            Err(Error::InvalidManifest { .. })
        ));
        std::fs::remove_dir_all(&tmp).unwrap();
    }

    #[test]
    fn seq_placeholder_resolves_per_bucket() {
        let tmp = std::env::temp_dir().join(format!("sk-registry-seq-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        write_manifest(
            &tmp,
            "bge",
            r#"
id = "bge-small-en-v1.5"
backend = "coreml"
artifact = "model_{seq}.mlmodelc"
tokenizer = "tokenizer.json"
dims = 384
pooling = "none"
buckets = [128, 256, 512]
max_seq_len = 512
"#,
        );
        let reg = ModelRegistry::scan(&tmp).unwrap();
        let m = reg.get("bge-small-en-v1.5").unwrap();
        assert!(m.artifact_path_for_bucket(256).ends_with("bge/model_256.mlmodelc"));
        std::fs::remove_dir_all(&tmp).unwrap();
    }

    #[test]
    fn rejects_seq_placeholder_on_static_backend() {
        let tmp = std::env::temp_dir().join(format!("sk-registry-seq-bad-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        write_manifest(
            &tmp,
            "bad",
            r#"
id = "bad"
backend = "static"
artifact = "model_{seq}.safetensors"
tokenizer = "t"
dims = 256
max_seq_len = 512
"#,
        );
        assert!(matches!(
            ModelRegistry::scan(&tmp),
            Err(Error::InvalidManifest { .. })
        ));
        std::fs::remove_dir_all(&tmp).unwrap();
    }

    #[test]
    fn missing_dir_is_empty_registry() {
        let reg = ModelRegistry::scan(Path::new("/nonexistent/sidekick-models")).unwrap();
        assert!(reg.is_empty());
    }
}
