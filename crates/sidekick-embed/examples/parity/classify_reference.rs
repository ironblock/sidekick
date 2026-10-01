//! Classifier references (fixtures/classify/reference.schema.json), written
//! by the classifier reference generator: per-case ids, markers, qtype,
//! labels and gold labels in reference.json, fp32 logits in
//! reference.safetensors.

use crate::reference::{corpus_hash, sha256_hex};
use serde::Deserialize;
use sidekick_core::manifest::{ClassifierManifest, ClassifyFormat};
use sidekick_core::{ClassifyParams, ClassifyTask, QuestionType};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

const FORMAT: u32 = 1;

#[derive(Debug, Deserialize)]
pub struct ClassifyReference {
    pub format: u32,
    pub corpus_sha256: String,
    pub tokenizer_sha256: String,
    pub model: RefModel,
    pub source: RefSource,
    pub oracles: Vec<String>,
    #[serde(default)]
    pub versions: BTreeMap<String, Option<String>>,
    pub cases: Vec<ClassifyCase>,
    /// Per oracle, each case's logits (its first `k` values).
    #[serde(skip)]
    pub logits: BTreeMap<String, Vec<Vec<f32>>>,
}

#[derive(Debug, Deserialize)]
pub struct RefModel {
    pub id: String,
    pub task: ClassifyTask,
    #[serde(default)]
    pub format: Option<String>,
    pub buckets: Vec<usize>,
    pub max_seq_len: usize,
    pub max_labels: usize,
    #[serde(default)]
    pub labels: Vec<String>,
}

#[derive(Debug, Deserialize)]
pub struct RefSource {
    pub repo: String,
    pub revision: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ClassifyCase {
    pub id: String,
    #[serde(default)]
    pub tags: Vec<String>,
    /// The text; for a reranker, the document.
    pub input: String,
    /// Rerank: the query `input` is paired with.
    #[serde(default)]
    pub query: Option<String>,
    /// Rerank: the (query, documents) group the case belongs to; rank flips
    /// are counted within a group.
    #[serde(default)]
    pub group: Option<String>,
    #[serde(default)]
    pub candidate_labels: Vec<String>,
    #[serde(default)]
    pub question_type: Option<QuestionType>,
    #[serde(default)]
    pub instructions: Option<String>,
    pub ids: Vec<i32>,
    /// Rerank: segment ids, when the model takes them.
    #[serde(default)]
    pub type_ids: Vec<i32>,
    #[serde(default)]
    pub markers: Vec<i32>,
    #[serde(default)]
    pub qtype: Option<i32>,
    /// Label count for this case.
    pub k: usize,
    /// The request asked for multi-label decoding (gliner2's `multi_label`):
    /// the case is graded with independent sigmoids, not the model's
    /// activation.
    #[serde(default)]
    pub multi_label: Option<bool>,
    /// Gold labels, when the corpus has them (reported, never graded).
    #[serde(default)]
    pub gold: Option<Vec<String>>,
}

/// Cases the reference truncated to the model's maximum length, as a
/// client would ask with `truncate_prompt_tokens`.
pub const TRUNCATED_TAG: &str = "truncated";

impl ClassifyCase {
    /// The rerank request this case stands for: the pair, truncated like
    /// `params` when tagged `truncated`.
    pub fn pair_params(&self, max_seq_len: usize) -> sidekick_core::PairParams {
        sidekick_core::PairParams {
            truncate_prompt_tokens: self.params(max_seq_len).truncate_prompt_tokens,
            ..Default::default()
        }
    }

    /// The request this case stands for. A case tagged `truncated` sends
    /// `truncate_prompt_tokens: max_seq_len` (HF truncation: special tokens
    /// kept, `max_seq_len` in total), since without it an over-length input
    /// is a 400.
    pub fn params(&self, max_seq_len: usize) -> ClassifyParams {
        ClassifyParams {
            truncate_prompt_tokens: self
                .tags
                .iter()
                .any(|t| t == TRUNCATED_TAG)
                .then_some(max_seq_len),
            candidate_labels: self.candidate_labels.clone(),
            question_type: self.question_type,
            instructions: self.instructions.clone(),
            ..Default::default()
        }
    }

    /// The case's labels in logit order: its candidate labels, or the
    /// model's fixed labels.
    pub fn labels<'a>(&'a self, fixed: &'a [String]) -> &'a [String] {
        if self.candidate_labels.is_empty() {
            fixed
        } else {
            &self.candidate_labels
        }
    }
}

/// The corpus a classifier's reference was generated from: the embedding
/// parity corpus for fixed-label models, `fixtures/classify/<id>.corpus.toml`
/// for zero-shot ones.
pub fn corpus_path(manifest: &ClassifierManifest) -> PathBuf {
    let fixtures = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures");
    match manifest.task {
        ClassifyTask::TextClassification => fixtures.join("parity/corpus.toml"),
        ClassifyTask::ZeroShotClassification => {
            fixtures.join(format!("classify/{}.corpus.toml", manifest.id))
        }
        ClassifyTask::TextRanking => fixtures.join("rerank/corpus.toml"),
    }
}

pub fn corpus_sha256(manifest: &ClassifierManifest) -> Result<String, String> {
    let path = corpus_path(manifest);
    std::fs::read_to_string(&path)
        .map(|text| corpus_hash(&text))
        .map_err(|e| format!("{}: {e}", path.display()))
}

impl ClassifyReference {
    pub fn source_label(&self) -> String {
        match &self.source.revision {
            Some(rev) => format!("{}@{}", self.source.repo, &rev[..rev.len().min(10)]),
            None => self.source.repo.clone(),
        }
    }

    /// Load `dir/reference.{json,safetensors}`.
    pub fn load(dir: &Path) -> Result<Self, String> {
        let json = std::fs::read_to_string(dir.join("reference.json"))
            .map_err(|e| format!("{}: {e}", dir.join("reference.json").display()))?;
        let raw = std::fs::read(dir.join("reference.safetensors"))
            .map_err(|e| format!("{}: {e}", dir.join("reference.safetensors").display()))?;
        Self::parse(&json, &raw)
    }

    pub fn parse(json: &str, safetensors_bytes: &[u8]) -> Result<Self, String> {
        let mut r: Self = serde_json::from_str(json).map_err(|e| format!("reference.json: {e}"))?;
        if r.format != FORMAT {
            return Err(format!("reference format {} (this suite reads {FORMAT})", r.format));
        }
        let kmax = r.model.max_labels;
        for case in &r.cases {
            if case.k == 0 || case.k > kmax {
                return Err(format!("case {}: k = {} outside 1..={kmax}", case.id, case.k));
            }
        }
        let st = safetensors::SafeTensors::deserialize(safetensors_bytes)
            .map_err(|e| format!("safetensors: {e}"))?;
        for oracle in &r.oracles {
            let t = st.tensor(oracle).map_err(|e| format!("oracle {oracle}: {e}"))?;
            if t.dtype() != safetensors::Dtype::F32 || t.shape() != [r.cases.len(), kmax] {
                return Err(format!(
                    "oracle {oracle}: {:?} {:?}, expected F32 [{}, {kmax}]",
                    t.dtype(),
                    t.shape(),
                    r.cases.len()
                ));
            }
            let (words, _) = t.data().as_chunks::<4>();
            let flat: Vec<f32> = words.iter().map(|b| f32::from_le_bytes(*b)).collect();
            let rows = flat
                .chunks(kmax)
                .zip(&r.cases)
                .map(|(row, case)| row[..case.k].to_vec())
                .collect();
            r.logits.insert(oracle.clone(), rows);
        }
        if !r.logits.contains_key("torch") {
            return Err("no `torch` oracle".into());
        }
        Ok(r)
    }

    /// Why this reference can't grade the model, if it can't: the corpus,
    /// manifest or tokenizer changed since it was generated.
    pub fn stale(
        &self,
        manifest: &ClassifierManifest,
        corpus_sha256: &str,
        tokenizer_sha256: &str,
    ) -> Option<String> {
        let m = &self.model;
        let mut why = Vec::new();
        if self.corpus_sha256 != corpus_sha256 {
            why.push("the corpus changed".to_string());
        }
        if self.tokenizer_sha256 != tokenizer_sha256 {
            why.push("tokenizer.json changed".to_string());
        }
        if m.id != manifest.id {
            why.push(format!("model id {} != {}", m.id, manifest.id));
        }
        if m.task != manifest.task {
            why.push("task changed".into());
        }
        let format = manifest.classify.format.map(|f| match f {
            ClassifyFormat::Laya => "laya".to_string(),
            ClassifyFormat::Gliner2 => "gliner2".to_string(),
        });
        if m.format != format {
            why.push("format changed".into());
        }
        if m.buckets != manifest.buckets || m.max_seq_len != manifest.max_seq_len {
            why.push("buckets or max_seq_len changed".into());
        }
        if m.max_labels != manifest.max_labels() {
            why.push(format!("max_labels {} != {}", m.max_labels, manifest.max_labels()));
        }
        if m.labels != manifest.classify.labels {
            why.push("labels changed".into());
        }
        (!why.is_empty()).then(|| {
            format!("stale reference ({}); regenerate it with the classifier reference generator", why.join("; "))
        })
    }
}

/// sha256 of a tokenizer file, as the reference records it.
pub fn tokenizer_sha(path: &Path) -> String {
    std::fs::read(path).map(|b| sha256_hex(&b)).unwrap_or_default()
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// A two-case zero-shot reference (k = 2 and 3 of max 4) with NaN
    /// padding, as the generator writes it.
    pub fn sample() -> (String, Vec<u8>) {
        let json = serde_json::json!({
            "format": 1,
            "corpus_sha256": "c",
            "tokenizer_sha256": "t",
            "model": {"id": "z", "task": "zero-shot-classification", "format": "laya",
                      "buckets": [128], "max_seq_len": 128, "max_labels": 4},
            "source": {"repo": "org/z", "revision": "0123456789abcdef"},
            "oracles": ["torch"],
            "cases": [
                {"id": "a", "input": "x", "candidate_labels": ["p", "q"], "question_type": "noul",
                 "ids": [1, 2], "markers": [1], "qtype": 2, "k": 2, "gold": ["q"]},
                {"id": "b", "input": "y", "candidate_labels": ["p", "q", "r"], "question_type": "choice",
                 "ids": [1, 3], "markers": [1], "qtype": 0, "k": 3, "gold": null},
            ],
        });
        let nan = f32::NAN;
        let data: Vec<u8> = [1.0, 2.0, nan, nan, 0.5, 0.0, -1.0, nan]
            .iter()
            .flat_map(|f: &f32| f.to_le_bytes())
            .collect();
        let view = safetensors::tensor::TensorView::new(safetensors::Dtype::F32, vec![2, 4], &data).unwrap();
        let st = safetensors::serialize([("torch", view)], &None).unwrap();
        (json.to_string(), st)
    }

    #[test]
    fn reads_nan_padded_logits_per_case() {
        let (json, st) = sample();
        let r = ClassifyReference::parse(&json, &st).unwrap();
        assert_eq!(r.logits["torch"], vec![vec![1.0, 2.0], vec![0.5, 0.0, -1.0]]);
        assert_eq!(r.cases[0].question_type, Some(QuestionType::Noul));
        assert_eq!(r.cases[0].params(128).candidate_labels, vec!["p", "q"]);
        assert_eq!(r.cases[0].params(128).truncate_prompt_tokens, None);
        let mut truncated = r.cases[0].clone();
        truncated.tags.push(TRUNCATED_TAG.into());
        assert_eq!(truncated.params(128).truncate_prompt_tokens, Some(128));
        assert_eq!(r.source_label(), "org/z@0123456789");
    }

    #[test]
    fn rejects_a_wrong_shape_or_a_bad_k() {
        let (json, _) = sample();
        let data = vec![0u8; 4 * 2 * 3];
        let view = safetensors::tensor::TensorView::new(safetensors::Dtype::F32, vec![2, 3], &data).unwrap();
        let st = safetensors::serialize([("torch", view)], &None).unwrap();
        assert!(ClassifyReference::parse(&json, &st).unwrap_err().contains("expected F32 [2, 4]"));
        let (_, st) = sample();
        let bad = json.replace("\"k\":3", "\"k\":5");
        assert!(ClassifyReference::parse(&bad, &st).unwrap_err().contains("k = 5"));
    }
}
