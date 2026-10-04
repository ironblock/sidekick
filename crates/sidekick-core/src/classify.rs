//! Classification: the backend-neutral interface behind `POST /v1/classify`
//! and the rerank routes (`/v1/rerank`, `/rerank`, `/v2/rerank`).
//!
//! The HTTP contract is vLLM's `/classify` and SGLang's `/v1/classify`
//! exactly (docs/design/classify.md, D28). Two tasks share it:
//! - **text-classification**: a fixed label set from the model's manifest;
//! - **zero-shot-classification**: labels arrive with each request
//!   (`candidate_labels`, the name Hugging Face's zero-shot task uses).
//!
//! A classifier turns one input into one [`Prepared`] (token ids plus, for
//! zero-shot formats, the positions its labels occupy) and runs it to raw
//! logits. Activation, label ordering and the response shape are the
//! server's job, so every backend returns the same thing: logits in label
//! order.

use crate::Result;
use serde::{Deserialize, Serialize};

/// What a classifier model does, from its manifest.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ClassifyTask {
    /// Fixed labels, declared in the manifest.
    TextClassification,
    /// Labels supplied per request as `candidate_labels`.
    ZeroShotClassification,
    /// A cross-encoder reranker: one relevance score for a (query,
    /// document) pair (docs/design/rerank.md).
    TextRanking,
}

impl ClassifyTask {
    /// The task's name: Hugging Face's pipeline name, as manifests and
    /// /v1/models spell it.
    pub fn name(self) -> &'static str {
        match self {
            ClassifyTask::TextClassification => "text-classification",
            ClassifyTask::ZeroShotClassification => "zero-shot-classification",
            ClassifyTask::TextRanking => "text-ranking",
        }
    }
}

/// How raw outputs become `probs`, following transformers'
/// text-classification pipeline: regression → none; multi-label or a single
/// output → sigmoid; otherwise softmax.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProblemType {
    SingleLabel,
    MultiLabel,
    Regression,
}

/// laya's three question types (its `QTYPES`: choice 0, score 1, noul 2).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QuestionType {
    Choice,
    Score,
    Noul,
}

impl QuestionType {
    /// The integer laya's graph takes as `qtype`.
    pub fn index(self) -> i32 {
        match self {
            QuestionType::Choice => 0,
            QuestionType::Score => 1,
            QuestionType::Noul => 2,
        }
    }
}

/// Which end of an over-length input is kept (vLLM's `truncation_side`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TruncationSide {
    /// Keep the first tokens (vLLM's `right`).
    #[default]
    Right,
    /// Keep the last tokens.
    Left,
}

/// Everything a request may say about how to classify one input, already
/// validated against the model by the server (unsupported fields are a 400
/// before a classifier sees them). Standard fields first, then sidekick's
/// extensions.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ClassifyParams {
    /// vLLM `truncate_prompt_tokens`: truncate to this many tokens. `None`:
    /// an over-length input is a 400 (vLLM's behavior) unless the model's
    /// format truncates by design (laya truncates the state; see its format).
    pub truncate_prompt_tokens: Option<usize>,
    /// vLLM `truncation_side`.
    pub truncation_side: TruncationSide,
    /// Zero-shot only: the labels, in the order `probs` reports them.
    pub candidate_labels: Vec<String>,
    /// laya format only.
    pub question_type: Option<QuestionType>,
    /// laya format only; the manifest's per-type default when absent.
    pub instructions: Option<String>,
}

/// How a (query, document) pair is truncated (rerank; see
/// docs/design/rerank.md).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct PairParams {
    /// Truncate the pair to this many tokens, special tokens included.
    /// `None`: a pair longer than the model is a 400 (vLLM's behavior).
    pub truncate_prompt_tokens: Option<usize>,
    pub truncation_side: TruncationSide,
    /// Cut the query, or each document, to this many tokens before pairing
    /// (vLLM's `max_tokens_per_query` / `max_tokens_per_doc`).
    pub max_tokens_per_query: Option<usize>,
    pub max_tokens_per_doc: Option<usize>,
    /// Truncate only the document, keeping the query whole (Cohere's
    /// contract on `/v2/rerank`). Otherwise tokenizers' `longest_first`,
    /// which is what vLLM's tokenizer call does.
    pub keep_query: bool,
}

/// One input, tokenized and laid out for the model's graph.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Prepared {
    /// Real token ids (specials included), before padding.
    pub ids: Vec<i32>,
    /// Segment ids parallel to `ids` (0 for the query, 1 for the document)
    /// for models whose graph takes `token_type_ids`; empty otherwise.
    pub type_ids: Vec<i32>,
    /// Zero-shot formats: the position of each candidate label's marker
    /// token, in label order. Empty for text-classification.
    pub markers: Vec<i32>,
    /// laya format: the question type index. `None` otherwise.
    pub qtype: Option<i32>,
    /// fev format: the decide token's position. `None` otherwise.
    pub decide_pos: Option<i32>,
    /// agentjev format: each token's segment, parallel to `ids` (0 for the
    /// shared prefix, `c` for candidate `c`, 1-based). Empty otherwise.
    pub seg: Vec<i32>,
    /// agentjev format: each token's position, parallel to `ids`; every
    /// candidate's positions continue from the prefix's end. Empty otherwise.
    pub position_ids: Vec<i32>,
    /// The smallest bucket that fits `ids`.
    pub bucket: usize,
}

/// Where a model came from, for provenance (`sidekick-model: id@revision`).
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct Source {
    pub repo: String,
    #[serde(default)]
    pub revision: Option<String>,
}

/// A classifier. Synchronous like [`crate::Embedder`]; async callers wrap in
/// `spawn_blocking`.
pub trait Classifier: Send + Sync {
    /// Stable identifier, the request's `model`.
    fn id(&self) -> &str;
    fn task(&self) -> ClassifyTask;
    fn problem_type(&self) -> ProblemType;
    /// Fixed labels in output order (text-classification); empty for
    /// zero-shot models.
    fn labels(&self) -> &[String];
    /// Most labels one request may carry (zero-shot); `labels().len()` for
    /// fixed-label models.
    fn max_labels(&self) -> usize;
    /// Most inputs one request may carry.
    fn max_batch(&self) -> usize;
    /// Opt-in calibration temperature for a request with `k` labels
    /// (`calibration: "model"`); `None` if the model declares none for it.
    fn calibration(&self, params: &ClassifyParams, k: usize) -> Option<f32>;
    fn source(&self) -> Option<&Source>;

    /// Tokenize and lay out one input. Errors are the client's
    /// (`Error::InvalidRequest`: over-length without truncation, labels that
    /// collide after the format's shrinking, …) or the tokenizer's.
    fn prepare(&self, input: &str, params: &ClassifyParams) -> Result<Prepared>;

    /// Tokenize and lay out one (query, document) pair: `text-ranking`
    /// models only. Errors as for [`prepare`](Self::prepare).
    fn prepare_pair(&self, query: &str, document: &str, params: &PairParams) -> Result<Prepared> {
        let _ = (query, document, params);
        Err(crate::Error::InvalidRequest(format!("model `{}` doesn't score pairs", self.id())))
    }

    /// Raw logits for `prepared`, one per label in label order (length
    /// `labels().len()` or `candidate_labels.len()`; 1 for a reranker).
    fn run(&self, prepared: &Prepared) -> Result<Vec<f32>>;

    /// Load the programs for `buckets` unless they're resident, without
    /// running anything. Returns the buckets this call waited on a load
    /// for, its own or a concurrent caller's; empty when all were resident.
    /// A bucket's first load compiles it, which can take minutes, so the
    /// daemon loads a request's buckets before its deadline starts.
    /// Backends without buckets have nothing to load.
    fn load_buckets(&self, buckets: &[usize]) -> Result<Vec<usize>> {
        let _ = buckets;
        Ok(Vec::new())
    }

    /// `prepare` + `run`.
    fn classify(&self, input: &str, params: &ClassifyParams) -> Result<Vec<f32>> {
        let p = self.prepare(input, params)?;
        self.run(&p)
    }
}

/// transformers' activation for a problem type and label count.
pub fn activate(problem: ProblemType, logits: &[f32], temperature: Option<f32>) -> Vec<f32> {
    let t = temperature.unwrap_or(1.0);
    match problem {
        ProblemType::Regression => logits.to_vec(),
        ProblemType::MultiLabel => logits.iter().map(|&x| 1.0 / (1.0 + (-x / t).exp())).collect(),
        ProblemType::SingleLabel if logits.len() == 1 => {
            logits.iter().map(|&x| 1.0 / (1.0 + (-x / t).exp())).collect()
        }
        ProblemType::SingleLabel => {
            let m = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let e: Vec<f32> = logits.iter().map(|&x| ((x - m) / t).exp()).collect();
            let sum: f32 = e.iter().sum();
            e.iter().map(|v| v / sum).collect()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn softmax_matches_transformers_and_temperature_is_opt_in() {
        let p = activate(ProblemType::SingleLabel, &[1.0, 2.0, 3.0], None);
        assert!((p.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        assert!((p[2] - 0.66524).abs() < 1e-4);
        let sharp = activate(ProblemType::SingleLabel, &[1.0, 2.0, 3.0], Some(0.5));
        assert!(sharp[2] > p[2]);
    }

    #[test]
    fn single_output_and_multi_label_use_sigmoid_regression_none() {
        assert!((activate(ProblemType::SingleLabel, &[0.0], None)[0] - 0.5).abs() < 1e-6);
        assert!((activate(ProblemType::MultiLabel, &[0.0, 0.0], None)[1] - 0.5).abs() < 1e-6);
        assert_eq!(activate(ProblemType::Regression, &[3.5], None), vec![3.5]);
    }

    #[test]
    fn question_types_match_laya() {
        assert_eq!(
            [QuestionType::Choice, QuestionType::Score, QuestionType::Noul].map(QuestionType::index),
            [0, 1, 2]
        );
        let q: QuestionType = serde_json::from_str("\"noul\"").unwrap();
        assert_eq!(q, QuestionType::Noul);
        for task in [ClassifyTask::TextClassification, ClassifyTask::ZeroShotClassification, ClassifyTask::TextRanking] {
            let json = serde_json::to_string(&task).unwrap();
            assert_eq!(json, format!("\"{}\"", task.name()));
            assert_eq!(serde_json::from_str::<ClassifyTask>(&json).unwrap(), task);
        }
    }
}
