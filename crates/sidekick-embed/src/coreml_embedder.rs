//! Core ML encoder embedder (ANE-targeted).
//!
//! Pipeline: tokenize (HF `tokenizers`) → pad to the smallest sequence-length
//! bucket that fits → int32 `input_ids`/`attention_mask` prediction →
//! pool per manifest → unit-normalize.
//!
//! Each bucket maps to its own static-shape artifact when the manifest uses
//! a `{seq}` placeholder; bucket models load lazily on first use and stay
//! resident. (A single enumerated-shapes artifact is also supported, but
//! hardware verification showed E5RT rejects flexible shapes at ANE plan
//! time and silently runs the whole encoder on CPU — prefer per-bucket
//! static artifacts.)

use crate::pooling::{mean_pool, normalize_in_place};
use sidekick_core::manifest::ResolvedModel;
use sidekick_core::{EmbedPurpose, Embedder, Error, Pooling, Result};
use sidekick_coreml::{ComputeUnits, CoremlModel, Int32Input};
use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};
use tokenizers::Tokenizer;

pub struct CoremlEmbedder {
    id: String,
    dims: usize,
    matryoshka: Vec<usize>,
    resolved: ResolvedModel,
    /// Lazily loaded per-bucket models. Without a `{seq}` placeholder every
    /// bucket resolves to the same path and shares one entry.
    models: Mutex<BTreeMap<std::path::PathBuf, Arc<CoremlModel>>>,
    tokenizer: Tokenizer,
    buckets: Vec<usize>,
    pooling: Pooling,
    input_ids_name: String,
    attention_mask_name: Option<String>,
    output_name: String,
    prefix_query: String,
    prefix_document: String,
}

impl CoremlEmbedder {
    pub fn load(model: &ResolvedModel) -> Result<Self> {
        let m = &model.manifest;
        let tokenizer = Tokenizer::from_file(model.tokenizer_path())
            .map_err(|e| Error::Tokenizer(e.to_string()))?;
        let embedder = Self {
            id: m.id.clone(),
            dims: m.dims,
            matryoshka: m.matryoshka.clone(),
            resolved: model.clone(),
            models: Mutex::new(BTreeMap::new()),
            tokenizer,
            buckets: m.buckets.clone(),
            pooling: m.pooling,
            input_ids_name: m.io.input_ids.clone(),
            attention_mask_name: m.io.attention_mask.clone(),
            output_name: m.io.output.clone(),
            prefix_query: m.prefixes.query.clone(),
            prefix_document: m.prefixes.document.clone(),
        };
        // Load the smallest bucket eagerly so a broken artifact fails at
        // load time (matching the pool's load-error surface), not on the
        // first request.
        let smallest = *embedder.buckets.first().expect("validated non-empty");
        embedder.model_for_bucket(smallest)?;
        Ok(embedder)
    }

    fn model_for_bucket(&self, bucket: usize) -> Result<Arc<CoremlModel>> {
        let path = self.resolved.artifact_path_for_bucket(bucket);
        let mut models = self.models.lock().unwrap();
        if let Some(m) = models.get(&path) {
            return Ok(m.clone());
        }
        let model = Arc::new(CoremlModel::load(&path, ComputeUnits::CpuAndNeuralEngine)?);
        models.insert(path, model.clone());
        Ok(model)
    }

    fn embed_one(&self, text: &str) -> Result<Vec<f32>> {
        let max = *self.buckets.last().expect("validated non-empty");
        let text = crate::byte_cap(text, max);
        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| Error::Tokenizer(e.to_string()))?;
        let ids: Vec<i32> = encoding
            .get_ids()
            .iter()
            .take(max)
            .map(|&u| u as i32)
            .collect();
        let used = ids.len();
        let bucket = *self
            .buckets
            .iter()
            .find(|&&b| b >= used)
            .unwrap_or(&max);

        let mut input_ids = ids;
        input_ids.resize(bucket, 0);
        let mut mask = vec![1i32; used];
        mask.resize(bucket, 0);

        let mut inputs = vec![Int32Input {
            name: &self.input_ids_name,
            shape: vec![1, bucket],
            data: input_ids,
        }];
        if let Some(mask_name) = &self.attention_mask_name {
            inputs.push(Int32Input {
                name: mask_name,
                shape: vec![1, bucket],
                data: mask.clone(),
            });
        }

        let model = self.model_for_bucket(bucket)?;
        let out = model.predict_int32(&inputs, &self.output_name)?;
        let n: usize = out.shape.iter().product();

        let mut vector = match self.pooling {
            Pooling::None => {
                if n != self.dims {
                    return Err(Error::Inference(format!(
                        "pooled output shape {:?} != dims {}",
                        out.shape, self.dims
                    )));
                }
                out.data
            }
            Pooling::Mean | Pooling::Cls => {
                if n != bucket * self.dims {
                    return Err(Error::Inference(format!(
                        "hidden-state output shape {:?} != [1, {bucket}, {}]",
                        out.shape, self.dims
                    )));
                }
                match self.pooling {
                    Pooling::Cls => out.data[..self.dims].to_vec(),
                    _ => {
                        let mask_u32: Vec<u32> = mask.iter().map(|&m| m as u32).collect();
                        mean_pool(&out.data, self.dims, &mask_u32)
                    }
                }
            }
        };
        normalize_in_place(&mut vector);
        Ok(vector)
    }
}

impl Embedder for CoremlEmbedder {
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
        let prefix = match purpose {
            EmbedPurpose::Query => &self.prefix_query,
            EmbedPurpose::Document => &self.prefix_document,
        };
        texts
            .iter()
            .map(|t| {
                if prefix.is_empty() {
                    self.embed_one(t)
                } else {
                    self.embed_one(&format!("{prefix}{t}"))
                }
            })
            .collect()
    }
}
