//! Static token-embedding lookup (model2vec-style).
//!
//! Artifact format: a safetensors file containing a single `embeddings`
//! tensor of shape `[vocab, dims]` (f32 or f16), paired with a HuggingFace
//! `tokenizer.json`. Embedding a text = tokenize, average the token rows,
//! normalize. No context, no attention — a floor tier that works on any
//! hardware in microseconds.

use crate::pooling::normalize_in_place;
use half::f16;
use sidekick_core::manifest::ResolvedModel;
use sidekick_core::{EmbedPurpose, Embedder, Error, Result};
use tokenizers::Tokenizer;

pub struct StaticEmbedder {
    id: String,
    dims: usize,
    matryoshka: Vec<usize>,
    /// `[vocab * dims]`, row-major.
    table: Vec<f32>,
    tokenizer: Tokenizer,
    prefix_query: String,
    prefix_document: String,
    max_seq_len: usize,
}

impl StaticEmbedder {
    pub fn load(model: &ResolvedModel) -> Result<Self> {
        let m = &model.manifest;
        let raw = std::fs::read(model.artifact_path())?;
        let st = safetensors::SafeTensors::deserialize(&raw)
            .map_err(|e| Error::Inference(format!("safetensors: {e}")))?;
        let tensor = st
            .tensor("embeddings")
            .map_err(|e| Error::Inference(format!("missing `embeddings` tensor: {e}")))?;
        let shape = tensor.shape();
        if shape.len() != 2 || shape[1] != m.dims {
            return Err(Error::Inference(format!(
                "expected embeddings shape [vocab, {}], got {shape:?}",
                m.dims
            )));
        }
        let table = match tensor.dtype() {
            safetensors::Dtype::F32 => tensor
                .data()
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect(),
            safetensors::Dtype::F16 => tensor
                .data()
                .chunks_exact(2)
                .map(|b| f16::from_le_bytes([b[0], b[1]]).to_f32())
                .collect(),
            other => {
                return Err(Error::Inference(format!(
                    "unsupported embeddings dtype {other:?}"
                )))
            }
        };
        let tokenizer = Tokenizer::from_file(model.tokenizer_path())
            .map_err(|e| Error::Tokenizer(e.to_string()))?;
        Ok(Self {
            id: m.id.clone(),
            dims: m.dims,
            matryoshka: m.matryoshka.clone(),
            table,
            tokenizer,
            prefix_query: m.prefixes.query.clone(),
            prefix_document: m.prefixes.document.clone(),
            max_seq_len: m.max_seq_len,
        })
    }

    fn embed_one(&self, text: &str) -> Result<Vec<f32>> {
        let encoding = self
            .tokenizer
            .encode(text, false)
            .map_err(|e| Error::Tokenizer(e.to_string()))?;
        let vocab = self.table.len() / self.dims;
        let mut out = vec![0.0f32; self.dims];
        let mut count = 0usize;
        for &id in encoding.get_ids().iter().take(self.max_seq_len) {
            let id = id as usize;
            if id >= vocab {
                continue;
            }
            let row = &self.table[id * self.dims..(id + 1) * self.dims];
            for (o, x) in out.iter_mut().zip(row) {
                *o += x;
            }
            count += 1;
        }
        if count > 0 {
            let inv = 1.0 / count as f32;
            for o in &mut out {
                *o *= inv;
            }
        }
        normalize_in_place(&mut out);
        Ok(out)
    }
}

impl Embedder for StaticEmbedder {
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

#[cfg(test)]
mod tests {
    use super::*;
    use sidekick_core::manifest::{EmbeddingBackendKind, ModelManifest, Pooling, ResolvedModel};
    use std::path::PathBuf;

    /// Build a tiny WordLevel tokenizer + embedding table fixture on disk.
    fn fixture(dir: &PathBuf) -> ResolvedModel {
        std::fs::create_dir_all(dir).unwrap();
        let tokenizer_json = serde_json::json!({
            "version": "1.0",
            "truncation": null,
            "padding": null,
            "added_tokens": [],
            "normalizer": {"type": "Lowercase"},
            "pre_tokenizer": {"type": "Whitespace"},
            "post_processor": null,
            "decoder": null,
            "model": {
                "type": "WordLevel",
                "vocab": {"hello": 0, "world": 1, "[UNK]": 2},
                "unk_token": "[UNK]"
            }
        });
        std::fs::write(dir.join("tokenizer.json"), tokenizer_json.to_string()).unwrap();

        // vocab=3, dims=4. hello -> e0, world -> e1 (orthogonal), unk -> 0.
        let rows: [[f32; 4]; 3] = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 2.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ];
        let bytes: Vec<u8> = rows
            .iter()
            .flatten()
            .flat_map(|f| f.to_le_bytes())
            .collect();
        let view =
            safetensors::tensor::TensorView::new(safetensors::Dtype::F32, vec![3, 4], &bytes)
                .unwrap();
        let data = safetensors::serialize([("embeddings", view)], &None).unwrap();
        std::fs::write(dir.join("model.safetensors"), data).unwrap();

        ResolvedModel {
            manifest: ModelManifest {
                id: "test-static".into(),
                backend: EmbeddingBackendKind::Static,
                artifact: "model.safetensors".into(),
                tokenizer: "tokenizer.json".into(),
                dims: 4,
                matryoshka: vec![],
                pooling: Pooling::Mean,
                buckets: vec![],
                max_seq_len: 512,
                io: Default::default(),
                prefixes: Default::default(),
            },
            dir: dir.clone(),
        }
    }

    #[test]
    fn embeds_mean_of_token_rows_normalized() {
        let dir = std::env::temp_dir().join(format!("sk-static-{}", std::process::id()));
        let model = fixture(&dir);
        let e = StaticEmbedder::load(&model).unwrap();

        // "hello world" -> mean([1,0,0,0],[0,2,0,0]) = [0.5,1,0,0], normalized.
        let out = e.embed(&["Hello WORLD"], EmbedPurpose::Document).unwrap();
        let v = &out[0];
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-6);
        assert!((v[1] / v[0] - 2.0).abs() < 1e-5);

        // Identical direction for a repeated word: "hello hello" == "hello".
        let a = e.embed(&["hello hello"], EmbedPurpose::Document).unwrap();
        let b = e.embed(&["hello"], EmbedPurpose::Document).unwrap();
        let dot: f32 = a[0].iter().zip(&b[0]).map(|(x, y)| x * y).sum();
        assert!((dot - 1.0).abs() < 1e-6);

        std::fs::remove_dir_all(&dir).unwrap();
    }
}
