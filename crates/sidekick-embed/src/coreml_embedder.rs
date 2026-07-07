//! Core ML encoder embedder (ANE-targeted).
//!
//! Pipeline: tokenize (HF `tokenizers`) → pad to the smallest enumerated
//! shape bucket that fits → int32 `input_ids`/`attention_mask` prediction →
//! pool per manifest → unit-normalize.

use crate::pooling::{mean_pool, normalize_in_place};
use sidekick_core::manifest::ResolvedModel;
use sidekick_core::{EmbedPurpose, Embedder, Error, Pooling, Result};
use sidekick_coreml::{ComputeUnits, CoremlModel, Int32Input};
use tokenizers::Tokenizer;

pub struct CoremlEmbedder {
    id: String,
    dims: usize,
    matryoshka: Vec<usize>,
    model: CoremlModel,
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
        let coreml = CoremlModel::load(&model.artifact_path(), ComputeUnits::CpuAndNeuralEngine)?;
        Ok(Self {
            id: m.id.clone(),
            dims: m.dims,
            matryoshka: m.matryoshka.clone(),
            model: coreml,
            tokenizer,
            buckets: m.buckets.clone(),
            pooling: m.pooling,
            input_ids_name: m.io.input_ids.clone(),
            attention_mask_name: m.io.attention_mask.clone(),
            output_name: m.io.output.clone(),
            prefix_query: m.prefixes.query.clone(),
            prefix_document: m.prefixes.document.clone(),
        })
    }

    fn embed_one(&self, text: &str) -> Result<Vec<f32>> {
        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| Error::Tokenizer(e.to_string()))?;
        let max = *self.buckets.last().expect("validated non-empty");
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

        let out = self.model.predict_int32(&inputs, &self.output_name)?;
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
