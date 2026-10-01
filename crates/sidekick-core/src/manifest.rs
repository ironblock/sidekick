//! On-disk model registry.
//!
//! A models directory contains one subdirectory per model. An embedding
//! model has a `manifest.toml`, a classifier a `classifier.toml` (its own
//! filename, so builds that predate classifiers never parse one); each
//! describes the artifact, its tokenizer, and how to run it:
//!
//! ```text
//! ~/Library/Application Support/sidekick/models/
//!   embeddinggemma-300m/
//!     manifest.toml
//!     model.mlpackage/            # or model.mlmodelc, model.safetensors
//!     tokenizer.json
//!   laya-en/
//!     classifier.toml
//!     model_{128,256,512}.mlmodelc/
//!     tokenizer.json
//! ```
//!
//! A manifest that doesn't parse or validate is skipped with a warning
//! (listed by [`ModelRegistry::skipped`]); the rest of the directory still
//! loads.

use crate::classify::{ClassifyTask, ProblemType, QuestionType, Source};
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
    /// Where the weights came from (`sidekick-model: <id>@<revision>`).
    #[serde(default)]
    pub source: Option<Source>,
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

/// `classifier.toml`: a text-classification or zero-shot classifier
/// (docs/design/classify.md). Classifiers are Core ML models, one static
/// artifact per sequence-length bucket as for embedders (D15).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassifierManifest {
    /// Stable id; the request's `model`.
    pub id: String,
    pub task: ClassifyTask,
    /// Where the weights came from (`sidekick-model: <id>@<revision>`).
    #[serde(default)]
    pub source: Option<Source>,
    /// Artifact path relative to the manifest's directory, with an optional
    /// `{seq}` placeholder resolved per bucket.
    pub artifact: String,
    /// tokenizer.json path relative to the manifest's directory.
    pub tokenizer: String,
    /// Sequence-length buckets, smallest first; the largest is `max_seq_len`.
    pub buckets: Vec<usize>,
    pub max_seq_len: usize,
    /// Most inputs one request may carry.
    #[serde(default = "default_max_batch")]
    pub max_batch: usize,
    #[serde(default = "default_problem_type")]
    pub problem_type: ProblemType,
    pub classify: ClassifySection,
}

fn default_max_batch() -> usize {
    32
}

fn default_problem_type() -> ProblemType {
    ProblemType::SingleLabel
}

/// A zero-shot input format: how labels and text become one sequence.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ClassifyFormat {
    /// laya's decision format: a `[MASK]` marker before each option.
    Laya,
    /// GLiNER2's schema format: an `[L]` marker before each label, scored
    /// per token (docs/design/classify.md).
    Gliner2,
}

/// `[classify]` of a `classifier.toml`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ClassifySection {
    /// Zero-shot only: the input format.
    #[serde(default)]
    pub format: Option<ClassifyFormat>,
    /// Zero-shot only: most labels one request may carry. Must equal the
    /// artifact's `marker_pos` width (checked at load).
    #[serde(default)]
    pub max_labels: Option<usize>,
    /// Text-classification only: the labels in output order (id2label).
    #[serde(default)]
    pub labels: Vec<String>,
    #[serde(default)]
    pub laya: Option<LayaSection>,
    #[serde(default)]
    pub gliner2: Option<Gliner2Section>,
    /// Opt-in temperatures, keyed `"<question_type>:<k bucket>"`
    /// ([`calibration_key`]).
    #[serde(default)]
    pub calibration: BTreeMap<String, f32>,
    /// Core ML feature names. Which ones are required depends on the format.
    #[serde(default)]
    pub io: ClassifierIo,
}

/// `[classify.laya]`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayaSection {
    /// Token budget for the question and its options (laya's `head_max_len`).
    pub head_max_len: usize,
    /// Instructions used when a request sends none, per question type.
    /// Required with laya's rendering; without them (Julia-1 has none), a
    /// request must send `instructions`.
    #[serde(default)]
    pub default_instructions: Option<DefaultInstructions>,
    /// How a request's labels become option texts.
    #[serde(default)]
    pub option_rendering: OptionRendering,
}

/// `[classify.gliner2]`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Gliner2Section {
    /// The task prompt when a request sends no `instructions`.
    pub default_instructions: String,
}

impl LayaSection {
    /// The instructions a question gets when the request sends none.
    pub fn default_instructions(&self, question_type: QuestionType) -> Option<&str> {
        self.default_instructions.as_ref().map(|d| d.get(question_type))
    }
}

/// How a laya-format model renders a request's labels as option texts. The
/// sequence around them is the same.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OptionRendering {
    /// laya's `render_options`: `"key: description"` as given, `"level i:
    /// …"` for scores, `"false: …"`/`"true: …"` for noul.
    #[default]
    Laya,
    /// Julia-1's typed API: the description alone (or the key), score items
    /// as given, and `"false"`/`"true"` or the two descriptions for noul.
    /// At most 20 options.
    Julia,
}

/// Julia-1's most options per question.
pub const JULIA_MAX_LABELS: usize = 20;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DefaultInstructions {
    pub choice: String,
    pub score: String,
    pub noul: String,
}

impl DefaultInstructions {
    pub fn get(&self, question_type: QuestionType) -> &str {
        match question_type {
            QuestionType::Choice => &self.choice,
            QuestionType::Score => &self.score,
            QuestionType::Noul => &self.noul,
        }
    }
}

/// `[classify.io]`: Core ML feature names.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ClassifierIo {
    #[serde(default)]
    pub input_ids: Option<String>,
    #[serde(default)]
    pub attention_mask: Option<String>,
    /// laya: `[1, max_labels]` int32 marker positions, −1 in unused slots.
    #[serde(default)]
    pub marker_pos: Option<String>,
    /// laya: `[1]` int32 question type.
    #[serde(default)]
    pub qtype: Option<String>,
    /// `[1, S]` int32 segment ids, for models that take them (BERT pairs).
    #[serde(default)]
    pub token_type_ids: Option<String>,
    #[serde(default)]
    pub output: Option<String>,
}

/// laya's `temp_bucket`: the calibration key for a question type and label
/// count. A 2-option noul and a 20-option choice need different scaling.
pub fn calibration_key(question_type: QuestionType, k: usize) -> String {
    let size = match k {
        0..=2 => "2",
        3..=5 => "3-5",
        6..=10 => "6-10",
        _ => "11+",
    };
    let t = match question_type {
        QuestionType::Choice => "choice",
        QuestionType::Score => "score",
        QuestionType::Noul => "noul",
    };
    format!("{t}:{size}")
}

impl ClassifierManifest {
    /// Most labels one request may carry: the fixed label count for
    /// text-classification, `[classify] max_labels` for zero-shot.
    pub fn max_labels(&self) -> usize {
        match self.task {
            ClassifyTask::TextClassification | ClassifyTask::TextRanking => {
                self.classify.labels.len()
            }
            ClassifyTask::ZeroShotClassification => self.classify.max_labels.unwrap_or(0),
        }
    }

    /// The request's extension fields this model accepts
    /// (docs/design/classify.md), for `/v1/models`.
    pub fn extension_fields(&self) -> Vec<&'static str> {
        let mut fields = Vec::new();
        if self.task == ClassifyTask::TextRanking {
            // Jina's and Cohere v1's; vLLM doesn't define it.
            fields.push("return_documents");
        }
        if self.task == ClassifyTask::ZeroShotClassification {
            fields.push("candidate_labels");
        }
        if !self.classify.calibration.is_empty() {
            fields.push("calibration");
        }
        match self.classify.format {
            Some(ClassifyFormat::Laya) => fields.extend(["question_type", "instructions"]),
            Some(ClassifyFormat::Gliner2) => fields.extend(["instructions", "multi_label"]),
            None => {}
        }
        fields
    }

    /// The extension fields every request must send: a subset of
    /// [`extension_fields`](Self::extension_fields), listed so a client
    /// learns them before a 400 does.
    pub fn required_fields(&self) -> Vec<&'static str> {
        let mut fields = Vec::new();
        if self.task == ClassifyTask::ZeroShotClassification {
            fields.push("candidate_labels");
        }
        if let Some(laya) = self.classify.laya.as_ref().filter(|_| self.classify.format == Some(ClassifyFormat::Laya)) {
            fields.push("question_type");
            if laya.default_instructions.is_none() {
                fields.push("instructions");
            }
        }
        fields
    }

    /// The opt-in temperature for a question type and label count.
    pub fn temperature(&self, question_type: Option<QuestionType>, k: usize) -> Option<f32> {
        let key = calibration_key(question_type?, k);
        self.classify.calibration.get(&key).copied()
    }
}

/// A manifest resolved against its directory.
#[derive(Debug, Clone)]
pub struct ResolvedModel {
    pub manifest: ModelManifest,
    pub dir: PathBuf,
}

/// `dir` joined with `artifact`, its `{seq}` placeholder (if any) resolved
/// to `bucket`.
fn bucket_path(dir: &Path, artifact: &str, bucket: usize) -> PathBuf {
    dir.join(artifact.replace("{seq}", &bucket.to_string()))
}

impl ResolvedModel {
    pub fn artifact_path(&self) -> PathBuf {
        self.dir.join(&self.manifest.artifact)
    }
    /// Artifact path for one sequence-length bucket: resolves a `{seq}`
    /// placeholder if present, otherwise the shared artifact path.
    pub fn artifact_path_for_bucket(&self, bucket: usize) -> PathBuf {
        bucket_path(&self.dir, &self.manifest.artifact, bucket)
    }
    pub fn tokenizer_path(&self) -> PathBuf {
        self.dir.join(&self.manifest.tokenizer)
    }
}

/// A classifier manifest resolved against its directory.
#[derive(Debug, Clone)]
pub struct ResolvedClassifier {
    pub manifest: ClassifierManifest,
    pub dir: PathBuf,
}

impl ResolvedClassifier {
    /// Artifact path for one sequence-length bucket.
    pub fn artifact_path_for_bucket(&self, bucket: usize) -> PathBuf {
        bucket_path(&self.dir, &self.manifest.artifact, bucket)
    }
    pub fn tokenizer_path(&self) -> PathBuf {
        self.dir.join(&self.manifest.tokenizer)
    }
}

/// A manifest the registry skipped, and why.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct SkippedManifest {
    /// Relative to the models directory (`<model dir>/<manifest file>`), so
    /// reporting it never discloses where the models directory is.
    pub path: PathBuf,
    /// Any other manifest it names is relative to the models directory too.
    pub reason: String,
}

/// Embedding manifest filename.
pub const EMBEDDER_MANIFEST: &str = "manifest.toml";
/// Classifier manifest filename.
pub const CLASSIFIER_MANIFEST: &str = "classifier.toml";

/// Scans a models directory for `*/manifest.toml` (embedders) and
/// `*/classifier.toml` (classifiers).
///
/// [`get`](Self::get), [`ids`](Self::ids) and [`iter`](Self::iter) see
/// embedders only, which is what the C ABI lists; classifiers have their
/// own accessors.
#[derive(Debug, Default)]
pub struct ModelRegistry {
    models: BTreeMap<String, ResolvedModel>,
    classifiers: BTreeMap<String, ResolvedClassifier>,
    skipped: Vec<SkippedManifest>,
    /// The scanned directory.
    root: PathBuf,
}

impl ModelRegistry {
    /// Scan `models_dir`. Only an unreadable directory is an error: a bad
    /// manifest is skipped with a warning, so one broken model can't take
    /// down every other. A classifier whose id an embedder already uses is
    /// skipped too; the embedder keeps working.
    pub fn scan(models_dir: &Path) -> Result<Self> {
        let mut reg = Self { root: models_dir.to_path_buf(), ..Self::default() };
        if !models_dir.exists() {
            return Ok(reg);
        }
        let mut dirs: Vec<PathBuf> = std::fs::read_dir(models_dir)?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.is_dir())
            .collect();
        dirs.sort();

        // Embedders first, so they win an id collision whatever the order.
        for dir in &dirs {
            let path = dir.join(EMBEDDER_MANIFEST);
            if !path.is_file() {
                continue;
            }
            let loaded = read_manifest::<ModelManifest>(&path)
                .and_then(|m| validate_embedder(&m).map(|()| m));
            match loaded {
                Ok(manifest) => match reg.owner(&manifest.id) {
                    Some(other) => reg.skip(&path, format!("duplicate model id `{}` (also {})", manifest.id, reg.relative(&other).display())),
                    None => {
                        reg.models.insert(manifest.id.clone(), ResolvedModel { manifest, dir: dir.clone() });
                    }
                },
                Err(reason) => reg.skip(&path, reason),
            }
        }
        for dir in &dirs {
            let path = dir.join(CLASSIFIER_MANIFEST);
            if !path.is_file() {
                continue;
            }
            let loaded = read_manifest::<ClassifierManifest>(&path)
                .and_then(|m| validate_classifier(&m).map(|()| m));
            match loaded {
                Ok(manifest) => match reg.owner(&manifest.id) {
                    Some(other) => reg.skip(&path, format!("duplicate model id `{}` (also {})", manifest.id, reg.relative(&other).display())),
                    None => {
                        reg.classifiers.insert(manifest.id.clone(), ResolvedClassifier { manifest, dir: dir.clone() });
                    }
                },
                Err(reason) => reg.skip(&path, reason),
            }
        }
        Ok(reg)
    }

    /// The manifest path already registered under `id`.
    fn owner(&self, id: &str) -> Option<PathBuf> {
        self.models
            .get(id)
            .map(|m| m.dir.join(EMBEDDER_MANIFEST))
            .or_else(|| self.classifiers.get(id).map(|c| c.dir.join(CLASSIFIER_MANIFEST)))
    }

    /// `path` relative to the models directory.
    fn relative(&self, path: &Path) -> PathBuf {
        path.strip_prefix(&self.root).unwrap_or(path).to_path_buf()
    }

    fn skip(&mut self, path: &Path, reason: String) {
        tracing::warn!(manifest = %path.display(), "skipping model: {reason}");
        let path = self.relative(path);
        self.skipped.push(SkippedManifest { path, reason });
    }

    /// An embedding model.
    pub fn get(&self, id: &str) -> Result<&ResolvedModel> {
        self.models
            .get(id)
            .ok_or_else(|| Error::ModelNotFound(id.to_string()))
    }

    /// Embedding model ids.
    pub fn ids(&self) -> impl Iterator<Item = &str> {
        self.models.keys().map(|s| s.as_str())
    }

    /// Embedding models.
    pub fn iter(&self) -> impl Iterator<Item = &ResolvedModel> {
        self.models.values()
    }

    /// No embedding models (classifiers aside).
    pub fn is_empty(&self) -> bool {
        self.models.is_empty()
    }

    /// A classifier.
    pub fn classifier(&self, id: &str) -> Result<&ResolvedClassifier> {
        self.classifiers
            .get(id)
            .ok_or_else(|| Error::ModelNotFound(id.to_string()))
    }

    pub fn classifier_ids(&self) -> impl Iterator<Item = &str> {
        self.classifiers.keys().map(|s| s.as_str())
    }

    pub fn classifiers(&self) -> impl Iterator<Item = &ResolvedClassifier> {
        self.classifiers.values()
    }

    /// Manifests skipped by the scan.
    pub fn skipped(&self) -> &[SkippedManifest] {
        &self.skipped
    }
}

fn read_manifest<T: serde::de::DeserializeOwned>(path: &Path) -> std::result::Result<T, String> {
    let raw = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
    toml::from_str(&raw).map_err(|e| e.to_string())
}

/// Buckets strictly increasing, the largest equal to `max_seq_len`.
fn validate_buckets(buckets: &[usize], max_seq_len: usize) -> std::result::Result<(), String> {
    if buckets.is_empty() {
        return Err("coreml backend requires sequence-length `buckets`".into());
    }
    if buckets[0] == 0 || buckets.windows(2).any(|w| w[0] >= w[1]) {
        return Err("buckets must be positive and strictly increasing".into());
    }
    if *buckets.last().unwrap() != max_seq_len {
        return Err("largest bucket must equal max_seq_len".into());
    }
    Ok(())
}

fn validate_embedder(m: &ModelManifest) -> std::result::Result<(), String> {
    if m.id.is_empty() {
        return Err("empty model id".into());
    }
    if m.dims == 0 {
        return Err("dims must be > 0".into());
    }
    if let Some(&first) = m.matryoshka.first() {
        if first != m.dims {
            return Err(format!("matryoshka must start at native dims {} (got {first})", m.dims));
        }
        if m.matryoshka.windows(2).any(|w| w[0] <= w[1]) {
            return Err("matryoshka dims must be strictly decreasing".into());
        }
    }
    if m.backend == EmbeddingBackendKind::Coreml {
        validate_buckets(&m.buckets, m.max_seq_len)?;
    }
    if m.backend != EmbeddingBackendKind::Coreml && m.artifact.contains("{seq}") {
        return Err("`{seq}` artifact placeholder is only valid for the coreml backend".into());
    }
    Ok(())
}

/// Validation per docs/design/classify.md. The artifact-dependent check
/// (`max_labels` against `marker_pos`'s width) happens at load.
fn validate_classifier(m: &ClassifierManifest) -> std::result::Result<(), String> {
    if m.id.is_empty() {
        return Err("empty model id".into());
    }
    validate_buckets(&m.buckets, m.max_seq_len)?;
    if m.max_batch == 0 {
        return Err("max_batch must be > 0".into());
    }
    let c = &m.classify;
    let io = &c.io;
    let require = |name: &str, v: &Option<String>| match v {
        Some(s) if !s.is_empty() => Ok(()),
        _ => Err(format!("[classify.io] needs `{name}` for this model's format")),
    };
    require("input_ids", &io.input_ids)?;
    require("attention_mask", &io.attention_mask)?;
    require("output", &io.output)?;
    match m.task {
        ClassifyTask::TextClassification => {
            if c.format.is_some() {
                return Err("`format` is for zero-shot models; a text-classification model's labels are fixed".into());
            }
            if c.labels.is_empty() {
                return Err("text-classification needs `[classify] labels` (output order)".into());
            }
            if let Some(dup) = first_duplicate(&c.labels) {
                return Err(format!("duplicate label `{dup}`"));
            }
            if matches!(c.max_labels, Some(n) if n != c.labels.len()) {
                return Err("`max_labels` must equal the number of `labels`".into());
            }
            if c.laya.is_some() || c.gliner2.is_some() {
                return Err("`[classify.laya]` and `[classify.gliner2]` are for zero-shot formats".into());
            }
            if io.marker_pos.is_some() || io.qtype.is_some() {
                return Err("text-classification's [classify.io] has input_ids, attention_mask and output only".into());
            }
            // Calibration keys are per question type, which fixed-label
            // models don't have.
            if !c.calibration.is_empty() {
                return Err("`[classify.calibration]` is for zero-shot formats".into());
            }
        }
        ClassifyTask::TextRanking => {
            if c.format.is_some() || c.laya.is_some() || c.gliner2.is_some() {
                return Err("`format`, `[classify.laya]` and `[classify.gliner2]` are for zero-shot models".into());
            }
            if c.labels.len() != 1 {
                return Err("a text-ranking model has one output: `labels` names it (e.g. [\"score\"])".into());
            }
            if matches!(c.max_labels, Some(n) if n != 1) {
                return Err("a text-ranking model's `max_labels` is 1".into());
            }
            if io.marker_pos.is_some() || io.qtype.is_some() {
                return Err("text-ranking's [classify.io] has input_ids, attention_mask, token_type_ids and output only".into());
            }
            if !c.calibration.is_empty() {
                return Err("`[classify.calibration]` is for zero-shot formats".into());
            }
        }
        ClassifyTask::ZeroShotClassification => {
            if !c.labels.is_empty() {
                return Err("`labels` is for text-classification; zero-shot labels come with each request".into());
            }
            let Some(format) = c.format else {
                return Err("zero-shot models need `[classify] format`".into());
            };
            if m.problem_type == ProblemType::Regression {
                return Err("zero-shot models can't be `regression`".into());
            }
            match c.max_labels {
                Some(n) if n >= 2 => {}
                _ => return Err("zero-shot models need `max_labels` ≥ 2".into()),
            }
            match format {
                ClassifyFormat::Laya => {
                    let Some(laya) = &c.laya else {
                        return Err("the laya format needs `[classify.laya]`".into());
                    };
                    if laya.head_max_len == 0 || laya.head_max_len >= m.max_seq_len {
                        return Err("`head_max_len` must be in 1..max_seq_len".into());
                    }
                    if laya.option_rendering == OptionRendering::Laya && laya.default_instructions.is_none() {
                        return Err("laya's option rendering needs `default_instructions`".into());
                    }
                    if laya.option_rendering == OptionRendering::Julia && m.max_labels() > JULIA_MAX_LABELS {
                        return Err(format!(
                            "Julia-1 takes at most {JULIA_MAX_LABELS} options: `max_labels` must be at most that"
                        ));
                    }
                    require("marker_pos", &io.marker_pos)?;
                    require("qtype", &io.qtype)?;
                    if io.token_type_ids.is_some() {
                        return Err("the laya format takes no `token_type_ids`".into());
                    }
                    if c.gliner2.is_some() {
                        return Err("`[classify.gliner2]` is for the gliner2 format".into());
                    }
                }
                ClassifyFormat::Gliner2 => {
                    match &c.gliner2 {
                        Some(g) if !g.default_instructions.trim().is_empty() => {}
                        Some(_) => return Err("`[classify.gliner2] default_instructions` must not be empty".into()),
                        None => return Err("the gliner2 format needs `[classify.gliner2]`".into()),
                    }
                    if c.laya.is_some() {
                        return Err("`[classify.laya]` is for the laya format".into());
                    }
                    if io.marker_pos.is_some() || io.qtype.is_some() || io.token_type_ids.is_some() {
                        return Err("the gliner2 format's [classify.io] has input_ids, attention_mask and output only".into());
                    }
                    // A request makes itself multi-label (`multi_label`);
                    // the manifest's problem type is the default.
                    if m.problem_type != ProblemType::SingleLabel {
                        return Err("the gliner2 format's problem_type is `single_label`; requests opt into \
                                    multi-label with `multi_label`".into());
                    }
                    // Calibration keys are per question type, which the
                    // gliner2 format doesn't have.
                    if !c.calibration.is_empty() {
                        return Err("`[classify.calibration]` isn't supported by the gliner2 format".into());
                    }
                }
            }
            for (key, &t) in &c.calibration {
                if !valid_calibration_key(key) {
                    return Err(format!(
                        "calibration key `{key}` isn't `<choice|score|noul>:<2|3-5|6-10|11+>`"
                    ));
                }
                if !(t.is_finite() && t > 0.0) {
                    return Err(format!("calibration `{key}` must be a positive temperature"));
                }
            }
        }
    }
    Ok(())
}

fn valid_calibration_key(key: &str) -> bool {
    matches!(key.split_once(':'), Some(("choice" | "score" | "noul", "2" | "3-5" | "6-10" | "11+")))
}

fn first_duplicate(labels: &[String]) -> Option<&str> {
    let mut seen = std::collections::HashSet::new();
    labels.iter().find(|l| !seen.insert(l.as_str())).map(|s| s.as_str())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_manifest(dir: &Path, name: &str, body: &str) {
        write_file(dir, name, EMBEDDER_MANIFEST, body);
    }

    fn write_classifier(dir: &Path, name: &str, body: &str) {
        write_file(dir, name, CLASSIFIER_MANIFEST, body);
    }

    fn write_file(dir: &Path, name: &str, file: &str, body: &str) {
        let d = dir.join(name);
        std::fs::create_dir_all(&d).unwrap();
        std::fs::write(d.join(file), body).unwrap();
    }

    fn tmp_dir(tag: &str) -> PathBuf {
        let tmp = std::env::temp_dir().join(format!("sk-registry-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        tmp
    }

    const LAYA: &str = r#"
id = "laya-en"
task = "zero-shot-classification"
source = { repo = "convaiinnovations/laya", revision = "55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851" }
artifact = "model_{seq}.mlmodelc"
tokenizer = "tokenizer.json"
buckets = [128, 256, 512]
max_seq_len = 512
max_batch = 32
problem_type = "single_label"

[classify]
format = "laya"
max_labels = 32
labels = []

[classify.laya]
head_max_len = 192
default_instructions = { choice = "Which option fits the text best?", score = "Which level fits the text best?", noul = "Does the statement hold for the text?" }

[classify.calibration]
"choice:2" = 1.906
"choice:3-5" = 1.760
"choice:6-10" = 1.000
"score:3-5" = 1.251
"noul:2" = 1.983

[classify.io]
input_ids = "input_ids"
attention_mask = "attention_mask"
marker_pos = "marker_pos"
qtype = "qtype"
output = "logits"
"#;

    const SENTIMENT: &str = r#"
id = "sentiment"
task = "text-classification"
source = { repo = "nlptown/bert-base-multilingual-uncased-sentiment" }
artifact = "model_{seq}.mlmodelc"
tokenizer = "tokenizer.json"
buckets = [128, 512]
max_seq_len = 512

[classify]
labels = ["1 star", "2 stars", "3 stars", "4 stars", "5 stars"]

[classify.io]
input_ids = "input_ids"
attention_mask = "attention_mask"
output = "logits"
"#;

    #[test]
    fn scans_classifiers_beside_embedders() {
        let tmp = tmp_dir("cls");
        write_classifier(&tmp, "laya", LAYA);
        write_classifier(&tmp, "sentiment", SENTIMENT);
        write_manifest(
            &tmp,
            "bge",
            r#"
id = "bge"
backend = "coreml"
artifact = "model_{seq}.mlmodelc"
tokenizer = "tokenizer.json"
dims = 384
pooling = "none"
buckets = [128]
max_seq_len = 128
source = { repo = "BAAI/bge-small-en-v1.5", revision = "abc" }
"#,
        );
        let reg = ModelRegistry::scan(&tmp).unwrap();
        assert!(reg.skipped().is_empty(), "{:?}", reg.skipped());
        // Embedder accessors see embedders only (the C ABI lists these).
        assert_eq!(reg.ids().collect::<Vec<_>>(), vec!["bge"]);
        assert_eq!(reg.get("bge").unwrap().manifest.source.as_ref().unwrap().revision.as_deref(), Some("abc"));
        assert!(reg.get("laya-en").is_err());
        assert_eq!(reg.classifier_ids().collect::<Vec<_>>(), vec!["laya-en", "sentiment"]);

        let laya = &reg.classifier("laya-en").unwrap().manifest;
        assert_eq!(laya.task, ClassifyTask::ZeroShotClassification);
        assert_eq!(laya.classify.format, Some(ClassifyFormat::Laya));
        assert_eq!(laya.max_labels(), 32);
        assert_eq!(laya.temperature(Some(QuestionType::Choice), 4), Some(1.760));
        assert_eq!(laya.temperature(Some(QuestionType::Choice), 12), None);
        assert_eq!(laya.temperature(None, 4), None);
        assert_eq!(
            laya.extension_fields(),
            vec!["candidate_labels", "calibration", "question_type", "instructions"]
        );
        assert_eq!(
            laya.classify.laya.as_ref().unwrap().default_instructions(QuestionType::Noul),
            Some("Does the statement hold for the text?")
        );
        assert_eq!(laya.classify.laya.as_ref().unwrap().option_rendering, OptionRendering::Laya, "the default");

        let s = &reg.classifier("sentiment").unwrap().manifest;
        assert_eq!(s.max_labels(), 5);
        assert_eq!(s.max_batch, 32, "default");
        assert_eq!(s.problem_type, ProblemType::SingleLabel);
        assert!(s.extension_fields().is_empty());
        assert!(reg.classifier("sentiment").unwrap().artifact_path_for_bucket(128).ends_with("sentiment/model_128.mlmodelc"));
        std::fs::remove_dir_all(&tmp).unwrap();
    }

    const RERANKER: &str = r#"
id = "ms-marco"
task = "text-ranking"
artifact = "model_{seq}.mlmodelc"
tokenizer = "tokenizer.json"
buckets = [128, 512]
max_seq_len = 512
max_batch = 128
problem_type = "regression"

[classify]
labels = ["score"]

[classify.io]
input_ids = "input_ids"
attention_mask = "attention_mask"
token_type_ids = "token_type_ids"
output = "logits"
"#;

    #[test]
    fn rerankers_are_one_output_classifiers() {
        let tmp = tmp_dir("rank");
        write_classifier(&tmp, "rank", RERANKER);
        let reg = ModelRegistry::scan(&tmp).unwrap();
        assert!(reg.skipped().is_empty(), "{:?}", reg.skipped());
        let m = &reg.classifier("ms-marco").unwrap().manifest;
        assert_eq!(m.task, ClassifyTask::TextRanking);
        assert_eq!(m.max_labels(), 1);
        assert_eq!(m.problem_type, ProblemType::Regression);
        assert_eq!(m.classify.io.token_type_ids.as_deref(), Some("token_type_ids"));
        assert_eq!(m.extension_fields(), vec!["return_documents"]);
        std::fs::remove_dir_all(&tmp).unwrap();

        for (name, body, want) in [
            ("two-labels", RERANKER.replace("[\"score\"]", "[\"a\", \"b\"]"), "one output"),
            ("format", RERANKER.replace("[classify]\n", "[classify]\nformat = \"laya\"\n"), "zero-shot"),
            ("marker", RERANKER.replace("output = ", "marker_pos = \"m\"\noutput = "), "token_type_ids and output only"),
        ] {
            let tmp = tmp_dir(&format!("rank-{name}"));
            write_classifier(&tmp, name, &body);
            let reg = ModelRegistry::scan(&tmp).unwrap();
            assert!(reg.skipped()[0].reason.contains(want), "{name}: {}", reg.skipped()[0].reason);
            std::fs::remove_dir_all(&tmp).unwrap();
        }
    }

    const GLINER2: &str = r#"
id = "gliner2.5-decide"
task = "zero-shot-classification"
source = { repo = "fastino/GLiNER2.5-Decide", revision = "5a7adf72a23b4d311abae6ce050d7f0012bb3416" }
artifact = "model_{seq}.mlmodelc"
tokenizer = "tokenizer.json"
buckets = [128, 256, 512]
max_seq_len = 512
max_batch = 32
problem_type = "single_label"

[classify]
format = "gliner2"
max_labels = 32

[classify.gliner2]
default_instructions = "label"

[classify.io]
input_ids = "input_ids"
attention_mask = "attention_mask"
output = "logits"
"#;

    #[test]
    fn gliner2_classifiers_and_what_they_refuse() {
        let tmp = tmp_dir("gliner2");
        write_classifier(&tmp, "g", GLINER2);
        let reg = ModelRegistry::scan(&tmp).unwrap();
        assert!(reg.skipped().is_empty(), "{:?}", reg.skipped());
        let m = &reg.classifier("gliner2.5-decide").unwrap().manifest;
        assert_eq!(m.classify.format, Some(ClassifyFormat::Gliner2));
        assert_eq!(m.max_labels(), 32);
        assert_eq!(m.classify.gliner2.as_ref().unwrap().default_instructions, "label");
        assert_eq!(m.extension_fields(), vec!["candidate_labels", "instructions", "multi_label"]);
        std::fs::remove_dir_all(&tmp).unwrap();

        for (name, body, want) in [
            ("no-section", GLINER2.replace("[classify.gliner2]\ndefault_instructions = \"label\"\n", ""), "needs `[classify.gliner2]`"),
            ("empty-default", GLINER2.replace("= \"label\"", "= \" \""), "must not be empty"),
            ("marker-io", GLINER2.replace("output = ", "marker_pos = \"m\"\noutput = "), "output only"),
            ("multi-label", GLINER2.replace("problem_type = \"single_label\"", "problem_type = \"multi_label\""), "`multi_label`"),
            ("calibration", format!("{GLINER2}\n[classify.calibration]\n\"choice:2\" = 1.0\n"), "isn't supported by the gliner2"),
            ("laya-section", format!("{GLINER2}\n[classify.laya]\nhead_max_len = 8\ndefault_instructions = {{ choice = \"a\", score = \"b\", noul = \"c\" }}\n"), "for the laya format"),
            ("on-fixed", format!("{SENTIMENT}\n[classify.gliner2]\ndefault_instructions = \"x\"\n"), "zero-shot formats"),
        ] {
            let tmp = tmp_dir(&format!("gliner2-{name}"));
            write_classifier(&tmp, name, &body);
            let reg = ModelRegistry::scan(&tmp).unwrap();
            assert_eq!(reg.skipped().len(), 1, "{name}");
            assert!(reg.skipped()[0].reason.contains(want), "{name}: {}", reg.skipped()[0].reason);
            std::fs::remove_dir_all(&tmp).unwrap();
        }
    }

    #[test]
    fn calibration_keys_follow_laya_temp_bucket() {
        let cases = [(1, "choice:2"), (2, "choice:2"), (3, "choice:3-5"), (5, "choice:3-5"), (6, "choice:6-10"), (10, "choice:6-10"), (11, "choice:11+")];
        for (k, key) in cases {
            assert_eq!(calibration_key(QuestionType::Choice, k), key);
        }
        assert_eq!(calibration_key(QuestionType::Noul, 2), "noul:2");
        assert_eq!(calibration_key(QuestionType::Score, 5), "score:3-5");
    }

    #[test]
    fn bad_classifiers_are_skipped_with_a_reason_and_the_rest_load() {
        let cases: &[(&str, String, &str)] = &[
            ("labels-on-zero-shot", LAYA.replace("labels = []", "labels = [\"a\"]"), "`labels` is for text-classification"),
            ("format-on-fixed", SENTIMENT.replace("[classify]\n", "[classify]\nformat = \"laya\"\n"), "`format` is for zero-shot"),
            ("no-marker-io", LAYA.replace("marker_pos = \"marker_pos\"\n", ""), "`marker_pos`"),
            ("no-output-io", SENTIMENT.replace("output = \"logits\"\n", ""), "`output`"),
            ("laya-io-on-fixed", SENTIMENT.replace("output = ", "qtype = \"qtype\"\noutput = "), "output only"),
            ("no-laya-section", LAYA.replace("[classify.laya]", "[classify.other]"), "[classify.laya]"),
            ("one-label", LAYA.replace("max_labels = 32", "max_labels = 1"), "max_labels"),
            ("bad-key", LAYA.replace("\"choice:2\"", "\"choice:12\""), "calibration key"),
            ("bad-temp", LAYA.replace("1.906", "-1.0"), "positive temperature"),
            ("fixed-calibration", format!("{SENTIMENT}\n[classify.calibration]\n\"choice:2\" = 1.0\n"), "zero-shot formats"),
            ("dup-label", SENTIMENT.replace("\"2 stars\"", "\"1 star\""), "duplicate label"),
            ("zero-batch", SENTIMENT.replace("max_seq_len = 512", "max_seq_len = 512\nmax_batch = 0"), "max_batch"),
            ("bad-buckets", SENTIMENT.replace("[128, 512]", "[512, 128]"), "increasing"),
            ("julia-too-many", LAYA.replace("head_max_len = 192", "head_max_len = 192\noption_rendering = \"julia\""), "at most 20 options"),
            ("no-defaults", LAYA.replace("\ndefault_instructions", "\n# default_instructions"), "default_instructions"),
            ("bad-rendering", LAYA.replace("head_max_len = 192", "head_max_len = 192\noption_rendering = \"jules\""), "option_rendering"),
            ("not-toml", "id = ".to_string(), ""),
        ];
        for (name, body, want) in cases {
            let tmp = tmp_dir(&format!("bad-{name}"));
            write_classifier(&tmp, name, body);
            write_classifier(&tmp, "good", SENTIMENT);
            let reg = ModelRegistry::scan(&tmp).unwrap();
            assert_eq!(reg.classifier_ids().collect::<Vec<_>>(), vec!["sentiment"], "{name}");
            assert_eq!(reg.skipped().len(), 1, "{name}");
            assert!(reg.skipped()[0].reason.contains(want), "{name}: {}", reg.skipped()[0].reason);
            std::fs::remove_dir_all(&tmp).unwrap();
        }
    }

    #[test]
    fn julia_rendering_takes_up_to_20_options_and_no_default_instructions() {
        let tmp = tmp_dir("julia");
        let julia = LAYA
            .replace("id = \"laya-en\"", "id = \"julia\"")
            .replace("max_labels = 32", "max_labels = 20")
            .replace("\ndefault_instructions", "\noption_rendering = \"julia\"\n# default_instructions");
        write_classifier(&tmp, "julia", &julia);
        let reg = ModelRegistry::scan(&tmp).unwrap();
        assert!(reg.skipped().is_empty(), "{:?}", reg.skipped().first().map(|s| &s.reason));
        let laya = reg.classifier("julia").unwrap().manifest.classify.laya.clone().unwrap();
        assert_eq!(laya.option_rendering, OptionRendering::Julia);
        assert_eq!(laya.default_instructions(QuestionType::Choice), None);
        std::fs::remove_dir_all(&tmp).unwrap();
    }

    #[test]
    fn an_embedder_wins_an_id_collision_with_a_classifier() {
        let tmp = tmp_dir("dup");
        // "a-..." sorts first, so a naive single pass would register the
        // classifier and drop the embedder.
        write_classifier(&tmp, "a-classifier", &SENTIMENT.replace("id = \"sentiment\"", "id = \"shared\""));
        write_manifest(
            &tmp,
            "b-embedder",
            r#"
id = "shared"
backend = "static"
artifact = "model.safetensors"
tokenizer = "tokenizer.json"
dims = 4
max_seq_len = 512
"#,
        );
        let reg = ModelRegistry::scan(&tmp).unwrap();
        assert_eq!(reg.ids().collect::<Vec<_>>(), vec!["shared"]);
        assert_eq!(reg.classifier_ids().count(), 0);
        let reason = &reg.skipped()[0].reason;
        assert!(reason.contains("duplicate model id `shared`") && reason.contains("b-embedder"), "{reason}");
        assert_eq!(reg.skipped()[0].path, Path::new("a-classifier/classifier.toml"));
        assert!(!reason.contains(tmp.to_str().unwrap()), "no absolute paths: {reason}");
        std::fs::remove_dir_all(&tmp).unwrap();
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
    fn skips_bad_matryoshka() {
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
        let reg = ModelRegistry::scan(&tmp).unwrap();
        assert!(reg.is_empty());
        assert_eq!(reg.skipped().len(), 1);
        assert!(reg.skipped()[0].reason.contains("matryoshka"), "{:?}", reg.skipped());
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
    fn skips_seq_placeholder_on_static_backend() {
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
        let reg = ModelRegistry::scan(&tmp).unwrap();
        assert!(reg.is_empty());
        assert!(reg.skipped()[0].reason.contains("{seq}"));
        std::fs::remove_dir_all(&tmp).unwrap();
    }

    #[test]
    fn missing_dir_is_empty_registry() {
        let reg = ModelRegistry::scan(Path::new("/nonexistent/sidekick-models")).unwrap();
        assert!(reg.is_empty());
    }
}
