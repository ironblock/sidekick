//! Reference files written by tools/parity_reference.py.

use serde::Deserialize;
use sha2::{Digest, Sha256};
use sidekick_core::manifest::ModelManifest;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

/// The format version this suite reads.
const FORMAT: u32 = 1;

#[derive(Debug, Deserialize)]
pub struct Reference {
    pub format: u32,
    pub corpus_sha256: String,
    pub model: RefModel,
    /// Hugging Face id and revision of the checkpoint the reference ran.
    pub source: serde_json::Value,
    /// sha256 of the model directory's tokenizer.json when generated.
    pub tokenizer_sha256: String,
    /// Per purpose: "published" (the model's own prompts) or "manifest".
    #[serde(default)]
    pub prompt_source: BTreeMap<String, String>,
    pub pooling: String,
    pub oracles: Vec<String>,
    #[serde(default)]
    pub versions: BTreeMap<String, Option<String>>,
    pub cases: Vec<Case>,
    /// `[cases][dims]` per oracle, from reference.safetensors.
    #[serde(skip)]
    pub vectors: BTreeMap<String, Vec<Vec<f32>>>,
}

#[derive(Debug, Deserialize)]
pub struct RefModel {
    pub id: String,
    pub dims: usize,
    pub buckets: Vec<usize>,
    pub max_seq_len: usize,
    pub prefixes: RefPrefixes,
}

#[derive(Debug, Deserialize)]
pub struct RefPrefixes {
    pub query: String,
    pub document: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Case {
    pub id: String,
    pub purpose: String,
    #[serde(default)]
    pub tags: Vec<String>,
    pub text: String,
    /// The token ids the reference pipeline fed the model.
    pub ids: Vec<i32>,
}

/// The committed corpus the references were generated from.
pub fn corpus_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/parity/corpus.toml")
}

pub fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

/// sha256 of the corpus without its full-line comments and blank lines,
/// so editing a comment doesn't invalidate every reference. Must match
/// `corpus_hash` in tools/parity_reference.py.
pub fn corpus_hash(text: &str) -> String {
    let lines: Vec<&str> = text
        .lines()
        .filter(|l| !l.trim().is_empty() && !l.trim_start().starts_with('#'))
        .collect();
    sha256_hex(lines.join("\n").as_bytes())
}

pub fn corpus_sha256() -> String {
    corpus_hash(&std::fs::read_to_string(corpus_path()).expect("read fixtures/parity/corpus.toml"))
}

impl Reference {
    /// The checkpoint, as `id@revision`.
    pub fn source_label(&self) -> String {
        let id = self
            .source
            .get("id")
            .and_then(|v| v.as_str())
            .unwrap_or("?");
        match self.source.get("revision").and_then(|v| v.as_str()) {
            Some(rev) => format!("{id}@{}", &rev[..rev.len().min(10)]),
            None => id.to_string(),
        }
    }

    /// Load `dir/reference.{json,safetensors}`.
    pub fn load(dir: &Path) -> Result<Self, String> {
        let json = std::fs::read_to_string(dir.join("reference.json"))
            .map_err(|e| format!("{}: {e}", dir.join("reference.json").display()))?;
        let mut r: Reference =
            serde_json::from_str(&json).map_err(|e| format!("reference.json: {e}"))?;
        if r.format != FORMAT {
            return Err(format!(
                "reference format {} (this suite reads {FORMAT})",
                r.format
            ));
        }
        let raw = std::fs::read(dir.join("reference.safetensors"))
            .map_err(|e| format!("{}: {e}", dir.join("reference.safetensors").display()))?;
        let st =
            safetensors::SafeTensors::deserialize(&raw).map_err(|e| format!("safetensors: {e}"))?;
        for oracle in &r.oracles {
            let t = st
                .tensor(oracle)
                .map_err(|e| format!("oracle {oracle}: {e}"))?;
            if t.dtype() != safetensors::Dtype::F32 || t.shape() != [r.cases.len(), r.model.dims] {
                return Err(format!(
                    "oracle {oracle}: {:?} {:?}, expected F32 [{}, {}]",
                    t.dtype(),
                    t.shape(),
                    r.cases.len(),
                    r.model.dims
                ));
            }
            let (words, _) = t.data().as_chunks::<4>();
            let flat: Vec<f32> = words.iter().map(|b| f32::from_le_bytes(*b)).collect();
            r.vectors.insert(
                oracle.clone(),
                flat.chunks(r.model.dims).map(<[f32]>::to_vec).collect(),
            );
        }
        if !r.vectors.contains_key("torch") {
            return Err("no `torch` oracle".into());
        }
        Ok(r)
    }

    /// Why this reference can't grade the model with the committed corpus,
    /// if it can't: the corpus, manifest or tokenizer changed since it was
    /// generated.
    pub fn stale(
        &self,
        manifest: &ModelManifest,
        corpus_sha256: &str,
        tokenizer_sha256: &str,
    ) -> Option<String> {
        let m = &self.model;
        let mut why = Vec::new();
        if self.corpus_sha256 != corpus_sha256 {
            why.push("fixtures/parity/corpus.toml changed".to_string());
        }
        if self.tokenizer_sha256 != tokenizer_sha256 {
            why.push("tokenizer.json changed".to_string());
        }
        if m.id != manifest.id {
            why.push(format!("model id {} != {}", m.id, manifest.id));
        }
        if m.dims != manifest.dims {
            why.push(format!("dims {} != {}", m.dims, manifest.dims));
        }
        if m.buckets != manifest.buckets || m.max_seq_len != manifest.max_seq_len {
            why.push("buckets or max_seq_len changed".into());
        }
        if m.prefixes.query != manifest.prefixes.query
            || m.prefixes.document != manifest.prefixes.document
        {
            why.push("prefixes changed".into());
        }
        (!why.is_empty()).then(|| {
            format!(
                "stale reference ({}); regenerate with tools/parity_reference.py",
                why.join("; ")
            )
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn corpus_hash_ignores_comments_and_blank_lines() {
        let a = corpus_hash("# header\n[[case]]\nid = \"x\"\n\n  # note\ntext = \"#1\"\n");
        let b = corpus_hash("[[case]]\n# changed\nid = \"x\"\ntext = \"#1\"");
        assert_eq!(a, b);
        assert_ne!(a, corpus_hash("[[case]]\nid = \"y\"\ntext = \"#1\""));
        // The value tools/parity_reference.py computes for the same input.
        assert_eq!(
            corpus_hash("a\n# c\n\nb"),
            "7e18f737311b2dc3b2f269dd78396b0351f14fb66efa879f768cb23181883c78"
        );
    }
}
