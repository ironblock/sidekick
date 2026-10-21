//! Reference files written by tools/parity_reference.py.

use serde::Deserialize;
use sha2::{Digest, Sha256};
use sidekick_core::manifest::ModelManifest;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

/// The format version this suite reads.
const FORMAT: u32 = 1;

#[derive(Debug, Clone, Deserialize)]
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
    /// The long corpus's cases past the model's max_seq_len, left out.
    #[serde(default)]
    pub skipped: Vec<String>,
    /// `[cases][dims]` per oracle, from reference.safetensors.
    #[serde(skip)]
    pub vectors: BTreeMap<String, Vec<Vec<f32>>>,
    /// The corpus it was generated from.
    #[serde(skip)]
    pub corpus: Corpus,
}

/// A parity corpus, and the reference files tools/parity_reference.py
/// writes for it (`--corpus`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum Corpus {
    /// fixtures/parity/corpus.toml: every embedder is graded on it.
    #[default]
    Main,
    /// fixtures/parity/corpus_long.toml: inputs past 512 tokens, for a
    /// model whose max_seq_len reaches its shortest case.
    Long,
}

impl Corpus {
    pub fn name(self) -> &'static str {
        match self {
            Self::Main => "main",
            Self::Long => "long",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        [Self::Main, Self::Long].into_iter().find(|c| c.name() == s)
    }

    /// The corpus's file name, in fixtures/parity.
    pub fn file(self) -> &'static str {
        match self {
            Self::Main => "corpus.toml",
            Self::Long => "corpus_long.toml",
        }
    }

    /// The stem of its reference files: `<stem>.json`, `<stem>.safetensors`.
    pub fn stem(self) -> &'static str {
        match self {
            Self::Main => "reference",
            Self::Long => "reference-long",
        }
    }

    pub fn path(self) -> PathBuf {
        crate::fixtures::dir().join("parity").join(self.file())
    }
}

/// The shortest integer `length` among a long corpus's cases: a model whose
/// max_seq_len is below it has nothing to grade there.
pub fn shortest_case(corpus_text: &str) -> Result<usize, String> {
    #[derive(Deserialize)]
    struct File {
        case: Vec<LengthCase>,
    }
    #[derive(Deserialize)]
    struct LengthCase {
        length: toml::Value,
    }
    let file: File = toml::from_str(corpus_text).map_err(|e| e.to_string())?;
    file.case
        .iter()
        .filter_map(|c| c.length.as_integer())
        .filter_map(|n| usize::try_from(n).ok())
        .min()
        .ok_or_else(|| "no case with an integer length".into())
}

/// Whether the long corpus applies to a model of this max_seq_len. A
/// fixtures directory without it (one copied from before it existed)
/// grades the main corpus alone.
pub fn long_corpus_reaches(max_seq_len: usize) -> Result<bool, String> {
    let path = Corpus::Long.path();
    if !path.exists() {
        return Ok(false);
    }
    let text = std::fs::read_to_string(&path).map_err(|e| format!("{}: {e}", path.display()))?;
    Ok(max_seq_len >= shortest_case(&text).map_err(|e| format!("{}: {e}", path.display()))?)
}

#[derive(Debug, Clone, Deserialize)]
pub struct RefModel {
    pub id: String,
    pub dims: usize,
    pub buckets: Vec<usize>,
    pub max_seq_len: usize,
    pub prefixes: RefPrefixes,
}

#[derive(Debug, Clone, Deserialize)]
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

pub fn corpus_sha256() -> Result<String, String> {
    corpus_sha256_of(Corpus::Main)
}

pub fn corpus_sha256_of(corpus: Corpus) -> Result<String, String> {
    let path = corpus.path();
    std::fs::read_to_string(&path).map(|text| corpus_hash(&text)).map_err(|e| format!("{}: {e}", path.display()))
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

    /// The cases of at most `max_tokens` reference ids, with their vectors,
    /// and the ids of the cases left out: what an ane path capped at
    /// `max_tokens` (D48) is graded against, since it doesn't run the rest.
    pub fn within(&self, max_tokens: usize) -> (Reference, Vec<String>) {
        let keep: Vec<bool> = self.cases.iter().map(|c| c.ids.len() <= max_tokens).collect();
        let mut sub = self.clone();
        sub.cases = self.cases.iter().zip(&keep).filter(|(_, k)| **k).map(|(c, _)| c.clone()).collect();
        for (name, vectors) in &self.vectors {
            let kept = vectors.iter().zip(&keep).filter(|(_, k)| **k).map(|(v, _)| v.clone()).collect();
            sub.vectors.insert(name.clone(), kept);
        }
        let left_out = self.cases.iter().zip(&keep).filter(|(_, k)| !**k).map(|(c, _)| c.id.clone()).collect();
        (sub, left_out)
    }

    /// Whether any case is graded on its own (not a stress case): grading
    /// needs one.
    pub fn has_core_case(&self) -> bool {
        self.cases.iter().any(|c| !c.tags.iter().any(|t| t == crate::grade::STRESS_TAG))
    }

    /// Load `dir/reference.{json,safetensors}`.
    pub fn load(dir: &Path) -> Result<Self, String> {
        Self::load_corpus(dir, Corpus::Main)
    }

    /// Load the reference for `corpus`: `dir/<its stem>.{json,safetensors}`.
    pub fn load_corpus(dir: &Path, corpus: Corpus) -> Result<Self, String> {
        let stem = corpus.stem();
        let json_path = dir.join(format!("{stem}.json"));
        let json = std::fs::read_to_string(&json_path).map_err(|e| format!("{}: {e}", json_path.display()))?;
        let mut r: Reference = serde_json::from_str(&json).map_err(|e| format!("{stem}.json: {e}"))?;
        r.corpus = corpus;
        if r.format != FORMAT {
            return Err(format!(
                "reference format {} (this suite reads {FORMAT})",
                r.format
            ));
        }
        let st_path = dir.join(format!("{stem}.safetensors"));
        let raw = std::fs::read(&st_path).map_err(|e| format!("{}: {e}", st_path.display()))?;
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
            why.push(format!("fixtures/parity/{} changed", self.corpus.file()));
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
        // An onnx build of the model has no buckets (D40). It is graded on
        // its Core ML build's texts, but its own max_seq_len, which may be
        // longer (Qwen3-Embedding-0.6B: 4,096 against 512), must match.
        let onnx = manifest.backend == sidekick_core::EmbeddingBackendKind::Onnx;
        if (!onnx && m.buckets != manifest.buckets) || m.max_seq_len != manifest.max_seq_len {
            why.push("buckets or max_seq_len changed".into());
        }
        if m.prefixes.query != manifest.prefixes.query
            || m.prefixes.document != manifest.prefixes.document
        {
            why.push("prefixes changed".into());
        }
        let how = match self.corpus {
            Corpus::Main => "tools/parity_reference.py".to_string(),
            c => format!("tools/parity_reference.py --corpus {}", c.name()),
        };
        (!why.is_empty()).then(|| format!("stale reference ({}); regenerate with {how}", why.join("; ")))
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

    #[test]
    fn each_corpus_has_its_own_files() {
        assert_eq!((Corpus::Main.file(), Corpus::Main.stem()), ("corpus.toml", "reference"));
        assert_eq!((Corpus::Long.file(), Corpus::Long.stem()), ("corpus_long.toml", "reference-long"));
        assert_eq!(Corpus::parse("long"), Some(Corpus::Long));
        assert_eq!(Corpus::parse("main"), Some(Corpus::Main));
        assert_eq!(Corpus::parse("short"), None);
    }

    #[test]
    fn the_long_corpus_applies_from_its_shortest_integer_case() {
        let text = "[[case]]\nid = \"a\"\nlength = 1500\n[[case]]\nid = \"b\"\nlength = { beyond_max = 188 }\n\
                    [[case]]\nid = \"c\"\nlength = 600\n[filler]\ntext = \"w\"\n";
        assert_eq!(shortest_case(text), Ok(600));
        assert!(shortest_case("[[case]]\nid = \"b\"\nlength = { beyond_max = 1 }\n").is_err());
        // The committed corpus: past every 512-token model, reached by a
        // 2,048- or 4,096-token one.
        let committed = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/parity/corpus_long.toml");
        let shortest = shortest_case(&std::fs::read_to_string(committed).unwrap()).unwrap();
        assert!(shortest > 512 && shortest <= 2048, "{shortest}");
    }
}
