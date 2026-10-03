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
use sidekick_core::{EmbedLimits, EmbedPurpose, Embedder, Error, Result, Truncate};
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
            safetensors::Dtype::F32 => {
                let (values, _) = tensor.data().as_chunks::<4>();
                values.iter().map(|b| f32::from_le_bytes(*b)).collect()
            }
            safetensors::Dtype::F16 => {
                let (values, _) = tensor.data().as_chunks::<2>();
                values.iter().map(|b| f16::from_le_bytes(*b).to_f32()).collect()
            }
            other => {
                return Err(Error::Inference(format!(
                    "unsupported embeddings dtype {other:?}"
                )))
            }
        };
        let mut tokenizer = Tokenizer::from_file(model.tokenizer_path())
            .map_err(|e| Error::Tokenizer(e.to_string()))?;
        // A tokenizer.json may pad every input to a fixed length
        // (all-MiniLM ships padding to 128): the pads would be averaged in
        // as text. One input is never padded.
        tokenizer.with_padding(None);
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

    fn embed_one(&self, prefix: &str, text: &str, limits: EmbedLimits) -> Result<Vec<f32>> {
        if limits.max_tokens == Some(0) {
            return Err(Error::InvalidRequest("max_tokens must be at least 1".into()));
        }
        let max = limits.max_tokens.unwrap_or(self.max_seq_len).min(self.max_seq_len);
        // Byte caps bound tokenizer work on the text; the prefix stays whole.
        let body = match limits.truncate {
            Truncate::End => crate::byte_cap(text, max),
            Truncate::Start => crate::classify_input::byte_cap_end(text, max),
            // Beyond 16 bytes per token, the text is over-long anyway.
            Truncate::Reject if text.len() > max.saturating_mul(16) => {
                return Err(too_long(text.len(), "bytes", max))
            }
            Truncate::Reject => text,
        };
        let text = format!("{prefix}{body}");
        let encoding = self
            .tokenizer
            .encode(text.as_str(), false)
            .map_err(|e| Error::Tokenizer(e.to_string()))?;
        let all = encoding.get_ids();
        // START keeps the prompt prefix's tokens whole, then the text's end.
        let kept = encoding.get_offsets().iter().take_while(|(start, _)| *start < prefix.len()).count();
        let ids: Vec<u32> = match limits.truncate {
            _ if all.len() <= max => all.to_vec(),
            Truncate::End => all[..max].to_vec(),
            Truncate::Start if kept >= max => {
                return Err(Error::InvalidRequest(format!(
                    "max_tokens {max} leaves no room for text after the model's {kept}-token prompt prefix"
                )))
            }
            Truncate::Start => all[..kept].iter().chain(&all[all.len() - (max - kept)..]).copied().collect(),
            Truncate::Reject => return Err(too_long(all.len(), "tokens", max)),
        };
        let vocab = self.table.len() / self.dims;
        let mut out = vec![0.0f32; self.dims];
        let mut count = 0usize;
        for &id in &ids {
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
        self.embed_with(texts, purpose, EmbedLimits::default())
    }

    fn embed_with(
        &self,
        texts: &[&str],
        purpose: EmbedPurpose,
        limits: EmbedLimits,
    ) -> Result<Vec<Vec<f32>>> {
        let prefix = match purpose {
            EmbedPurpose::Query => &self.prefix_query,
            EmbedPurpose::Document => &self.prefix_document,
        };
        texts
            .iter()
            .map(|t| {
                self.embed_one(prefix, t, limits)
            })
            .collect()
    }
}

fn too_long(n: usize, unit: &str, max: usize) -> Error {
    Error::InvalidRequest(format!(
        "input of {n} {unit} exceeds the limit of {max} tokens, and truncate is NONE"
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use sidekick_core::manifest::{EmbeddingBackendKind, ModelManifest, Pooling, ResolvedModel};
    use std::path::PathBuf;

    /// Build a tiny WordLevel tokenizer + embedding table fixture on disk.
    fn fixture(dir: &PathBuf) -> ResolvedModel {
        fixture_with_padding(dir, serde_json::Value::Null)
    }

    fn fixture_with_padding(dir: &PathBuf, padding: serde_json::Value) -> ResolvedModel {
        std::fs::create_dir_all(dir).unwrap();
        let tokenizer_json = serde_json::json!({
            "version": "1.0",
            "truncation": null,
            "padding": padding,
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
                source: None,
                compute_units: None,
                ane_weight_limit: None,
                cpu_seq_limit: None,
                seq_cap: None,
                compute_units_source: Default::default(),
                placement: None,
                chunking: None,
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

    #[test]
    fn limits_keep_the_prefix_and_reject_zero_tokens() {
        let dir = std::env::temp_dir().join(format!("sk-static-limits-{}", std::process::id()));
        let mut model = fixture(&dir);
        model.manifest.prefixes.query = "world ".into();
        let e = StaticEmbedder::load(&model).unwrap();
        let start = |max| EmbedLimits { truncate: Truncate::Start, max_tokens: Some(max) };
        // START at 2 tokens: the prefix's "world", then the text's last
        // "hello"; without the prefix it would be two "hello"s.
        let v = e.embed_with(&["hello hello hello"], EmbedPurpose::Query, start(2)).unwrap();
        let inv = 1.0 / 5f32.sqrt();
        assert!((v[0][0] - inv).abs() < 1e-6 && (v[0][1] - 2.0 * inv).abs() < 1e-6, "{:?}", v[0]);
        // The prefix alone fills 1 token: no room for text.
        assert!(e.embed_with(&["hello"], EmbedPurpose::Query, start(1)).is_err());
        // max_tokens 0 is refused, not an empty (zero) vector.
        let zero = EmbedLimits { max_tokens: Some(0), ..Default::default() };
        assert!(matches!(e.embed_with(&["hello"], EmbedPurpose::Document, zero), Err(Error::InvalidRequest(_))));
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn a_tokenizer_that_pads_to_a_fixed_length_changes_nothing() {
        // As all-MiniLM-L6-v2's tokenizer.json ships it (pads to 128), here
        // with the pad id of a real row, so any pad averaged in would show.
        let dir = std::env::temp_dir().join(format!("sk-static-pad-{}", std::process::id()));
        let padding = serde_json::json!({
            "strategy": {"Fixed": 8}, "direction": "Right", "pad_to_multiple_of": null,
            "pad_id": 1, "pad_type_id": 0, "pad_token": "world"
        });
        let e = StaticEmbedder::load(&fixture_with_padding(&dir, padding)).unwrap();
        assert_eq!(e.embed(&["hello"], EmbedPurpose::Document).unwrap()[0], vec![1.0, 0.0, 0.0, 0.0]);
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
