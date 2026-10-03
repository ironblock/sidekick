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

/// The Core ML compute units a model is loaded with: a manifest's
/// `compute_units`, `cpu_and_ne` unless it says otherwise. The ANE
/// preference is sidekick's default because it keeps background work off
/// the GPU (D14); a model the ANE runs badly can ask for another
/// (docs/design/classify.md, "Compute units").
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub enum ComputeUnits {
    /// `.all`: Core ML picks among CPU, GPU and ANE.
    #[serde(rename = "all")]
    All,
    /// `.cpuAndNeuralEngine`.
    #[default]
    #[serde(rename = "cpu_and_ne")]
    CpuAndNeuralEngine,
    /// `.cpuAndGPU`.
    #[serde(rename = "cpu_and_gpu")]
    CpuAndGpu,
    /// `.cpuOnly`.
    #[serde(rename = "cpu_only")]
    CpuOnly,
}

impl ComputeUnits {
    /// The manifest value, also what `sidekick-compute-units` and
    /// `/v1/models` report.
    pub fn name(self) -> &'static str {
        match self {
            Self::All => "all",
            Self::CpuAndNeuralEngine => "cpu_and_ne",
            Self::CpuAndGpu => "cpu_and_gpu",
            Self::CpuOnly => "cpu_only",
        }
    }
}

/// Core ML runs an ML program on the ANE only while its weights stay under
/// about 1 GiB; past that it runs the whole program elsewhere, with no error
/// and nothing in the log. Measured on an M1 Max under macOS 27.0: a program
/// with 0.964 GiB of weights ran on the ANE, one with 1.022 GiB didn't, and
/// coremltools documents a 1 GB Neural Engine limit. The converters gate on
/// the same value (tools/sidekick_convert/plan.py).
pub const MAX_ANE_PROGRAM_WEIGHT_BYTES: u64 = 1 << 30;

/// A manifest's `ane_weight_limit`: whether the registry enforces
/// [`MAX_ANE_PROGRAM_WEIGHT_BYTES`] on a model served on the ANE.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AneWeightLimit {
    /// Skip a model whose compiled weights exceed the cap (the default).
    #[default]
    Enforce,
    /// Load it anyway, for experimentation: Core ML will run it off the ANE.
    Ignore,
}

/// Core ML's fp16 CPU matmul sums in an order that depends on the
/// contraction length past 1,024 keys: accurate within its own rounding,
/// but a model's results then differ slightly between buckets (by up to
/// 0.021 in probability between Lumma-fev's 1,024 and 2,048 buckets, 0.018
/// for agent-jev). No conversion restores invariance (D33). A model served
/// with `cpu_only` therefore runs with its sequence length capped here.
pub const MAX_CPU_INVARIANT_SEQ: usize = 1024;

/// A manifest's `cpu_seq_limit`: whether the registry caps a `cpu_only`
/// model's sequence length at [`MAX_CPU_INVARIANT_SEQ`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CpuSeqLimit {
    /// Cap it (the default).
    #[default]
    Enforce,
    /// Serve every bucket, accepting small bucket-dependent differences
    /// past 1,024 tokens.
    Ignore,
}

/// A sequence-length cap the registry applied to a model.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct SeqCap {
    /// The effective `max_seq_len`: the largest bucket kept.
    pub limit: usize,
    /// The manifest's own `max_seq_len`.
    pub manifest_max_seq_len: usize,
    pub reason: &'static str,
    /// fev's state limit under the cap, when the model is fev and the cap
    /// lowered it; the manifest's own is `[classify.fev] state_max_len`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub state_max_len: Option<usize>,
}

/// `[placement]`: where Core ML placed each bucket's operations when the
/// converter read its compute plan, and on what machine. The daemon reports
/// it per bucket without reading a plan itself; see docs/design/classify.md,
/// "Where Core ML places operations".
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RecordedPlacement {
    /// The compute units the plan was read for. A plan for other units
    /// than the model is served with doesn't describe it, and isn't
    /// reported.
    pub compute_units: ComputeUnits,
    /// The chip it was read on (`sysctl machdep.cpu.brand_string`).
    pub chip: String,
    /// The macOS build it was read on (`sw_vers -buildVersion`).
    pub macos_build: String,
    /// The day it was read (`2026-10-01`).
    #[serde(default)]
    pub date: Option<String>,
    /// By bucket length (TOML table keys are strings).
    pub buckets: BTreeMap<String, RecordedPlan>,
}

/// One bucket's recorded plan: operation counts by device.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RecordedPlan {
    pub ane: usize,
    pub gpu: usize,
    pub cpu: usize,
    /// Operations with no device (constants and other bookkeeping).
    pub unassigned: usize,
    /// Every operation: the sum of the four counts.
    pub total: usize,
    /// Operator names off the ANE, with counts.
    #[serde(default)]
    pub off_ane_ops: BTreeMap<String, usize>,
    /// A chunked bucket's chunks (D37), in order, each with its own counts;
    /// the bucket's counts are their sums. Empty for one program.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub chunks: Vec<RecordedPlan>,
}

impl RecordedPlan {
    fn counts_add_up(&self) -> bool {
        self.ane + self.gpu + self.cpu + self.unassigned == self.total
    }
}

impl RecordedPlacement {
    /// The recorded plan for `bucket`.
    pub fn bucket(&self, bucket: usize) -> Option<&RecordedPlan> {
        self.buckets.get(&bucket.to_string())
    }
}

/// Why a recorded placement can't be used: it must describe the model's own
/// buckets, with consistent counts. Checked against the manifest's buckets
/// before any cap drops some. A bad record is dropped with a warning rather
/// than skipping the model: it only describes the model.
fn placement_problem(p: &RecordedPlacement, buckets: &[usize], chunks: usize) -> Option<String> {
    for (key, plan) in &p.buckets {
        let listed = key.parse::<usize>().is_ok_and(|b| buckets.contains(&b));
        if !listed {
            return Some(format!("`[placement.buckets]` has `{key}`, which isn't one of the buckets {buckets:?}"));
        }
        if !plan.counts_add_up() || !plan.chunks.iter().all(RecordedPlan::counts_add_up) {
            return Some(format!("`[placement.buckets.{key}]` counts don't add up to its `total`"));
        }
        if !plan.chunks.is_empty() && plan.chunks.len() != chunks {
            return Some(format!(
                "`[placement.buckets.{key}]` lists {} chunks, and the model has {chunks}",
                plan.chunks.len()
            ));
        }
    }
    None
}

/// `[chunking]`: a bucket compiled as an ordered chain of programs, each
/// under the ANE's per-program weight limit (D37). The artifact's
/// `{chunk}` placeholder numbers them from 0.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Chunking {
    /// Programs per bucket.
    pub chunks: usize,
    /// The fp16 weight budget the converter split under (`--chunks auto`),
    /// for reports; the limit itself is [`MAX_ANE_PROGRAM_WEIGHT_BYTES`].
    #[serde(default)]
    pub weight_budget_bytes: Option<u64>,
}

/// The chunk inputs and outputs that carry the residual stream from one
/// chunk to the next (D37).
pub const CHUNK_HIDDEN_IN: &str = "hidden_in";
pub const CHUNK_HIDDEN_OUT: &str = "hidden_out";

/// The artifact files of one bucket, in chain order: one per chunk when
/// `artifact` has a `{chunk}` placeholder, else the one program.
pub fn artifact_files(artifact: &str, bucket: usize, chunks: usize) -> Vec<String> {
    let name = artifact.replace("{seq}", &bucket.to_string());
    if !name.contains("{chunk}") {
        return vec![name];
    }
    (0..chunks).map(|i| name.replace("{chunk}", &i.to_string())).collect()
}

/// `{chunk}` and `[chunking]` come together (D37).
fn validate_chunking(artifact: &str, chunking: Option<&Chunking>) -> std::result::Result<(), String> {
    match (artifact.contains("{chunk}"), chunking) {
        (true, None) => Err("`artifact` has a `{chunk}` placeholder, and there's no `[chunking]` table".into()),
        (false, Some(_)) => Err("`[chunking]` needs a `{chunk}` placeholder in `artifact`".into()),
        (_, Some(c)) if c.chunks == 0 => Err("`[chunking] chunks` must be at least 1".into()),
        _ => Ok(()),
    }
}

/// Drop `placement` when it can't be used, with a warning naming `path`.
fn check_placement(path: &Path, placement: &mut Option<RecordedPlacement>, problem: Option<String>) {
    if let Some(problem) = problem {
        tracing::warn!(manifest = %path.display(), "ignoring the recorded compute plan: {problem}");
        *placement = None;
    }
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
    /// Core ML compute units (coreml backend only; default `cpu_and_ne`).
    #[serde(default)]
    pub compute_units: Option<ComputeUnits>,
    /// Whether the ANE weight cap is enforced (coreml backend only).
    #[serde(default)]
    pub ane_weight_limit: Option<AneWeightLimit>,
    /// Whether a `cpu_only` model's length is capped (coreml backend only).
    #[serde(default)]
    pub cpu_seq_limit: Option<CpuSeqLimit>,
    /// The cap the registry applied, if any (never read from the file).
    #[serde(skip)]
    pub seq_cap: Option<SeqCap>,
    /// Who chose `compute_units`: the manifest, or the operator's daemon
    /// config (D38). Never read from the file.
    #[serde(skip)]
    pub compute_units_source: ComputeUnitsSource,
    /// The compute plan the converter read (coreml backend only).
    #[serde(default)]
    pub placement: Option<RecordedPlacement>,
    /// Each bucket as a chain of programs (coreml backend only; D37).
    #[serde(default)]
    pub chunking: Option<Chunking>,
}

impl ModelManifest {
    /// Programs per bucket: 1 unless the model is chunked (D37).
    pub fn chunks(&self) -> usize {
        self.chunking.map_or(1, |c| c.chunks)
    }

    /// What the model runs on, as `sidekick-compute-units` reports it:
    /// `cpu` for a static model, else its Core ML compute units.
    pub fn compute_units_name(&self) -> &'static str {
        match self.backend {
            EmbeddingBackendKind::Static => "cpu",
            EmbeddingBackendKind::Coreml => self.compute_units.unwrap_or_default().name(),
        }
    }
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
    /// Core ML compute units (default `cpu_and_ne`).
    #[serde(default)]
    pub compute_units: ComputeUnits,
    /// Whether the ANE weight cap is enforced (default `enforce`).
    #[serde(default)]
    pub ane_weight_limit: AneWeightLimit,
    /// Whether a `cpu_only` model's length is capped (default `enforce`).
    #[serde(default)]
    pub cpu_seq_limit: CpuSeqLimit,
    /// The cap the registry applied, if any (never read from the file).
    #[serde(skip)]
    pub seq_cap: Option<SeqCap>,
    /// Who chose `compute_units`: the manifest, or the operator's daemon
    /// config (D38). Never read from the file.
    #[serde(skip)]
    pub compute_units_source: ComputeUnitsSource,
    /// The compute plan the converter read.
    #[serde(default)]
    pub placement: Option<RecordedPlacement>,
    /// Each bucket as a chain of programs (D37).
    #[serde(default)]
    pub chunking: Option<Chunking>,
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
    /// Lumma-fev's decision format: a causal row of the state, one question
    /// and its options, scored at each option's end and a final decide
    /// token (docs/design/classify.md).
    Fev,
    /// AgentJev's decision format: the state and question as a shared
    /// prefix, then each candidate as its own branch of a tree, scored by a
    /// head over the candidate set (docs/design/classify.md).
    Agentjev,
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
    #[serde(default)]
    pub fev: Option<FevSection>,
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

/// `[classify.fev]`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FevSection {
    /// The row's delimiter tokens, by role; each is one added token of the
    /// tokenizer, resolved at load.
    pub delimiters: FevDelimiters,
    /// The state's tokens at most, its delimiter included (the checkpoint's
    /// own limit); a longer state is truncated, keeping its start.
    pub state_max_len: usize,
}

/// fev's delimiters: `<state> text <question> instructions (<option> text
/// <option_end>)… <decide>`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FevDelimiters {
    pub state: String,
    pub question: String,
    pub option: String,
    pub option_end: String,
    pub decide: String,
}

impl FevDelimiters {
    /// (role, token) in row order.
    pub fn by_role(&self) -> [(&'static str, &str); 5] {
        [
            ("state", &self.state),
            ("question", &self.question),
            ("option", &self.option),
            ("option_end", &self.option_end),
            ("decide", &self.decide),
        ]
    }
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
    /// fev: `[1]` int32 position of the decide token.
    #[serde(default)]
    pub decide_pos: Option<String>,
    /// agentjev: `[1, S]` int32 tree segments (0 prefix, `c` candidate `c`,
    /// −1 pads), from which the graph builds its attention mask.
    #[serde(default)]
    pub seg: Option<String>,
    /// agentjev: `[1, S]` int32 positions; each candidate's restart at the
    /// prefix's end.
    #[serde(default)]
    pub position_ids: Option<String>,
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
    /// Programs per bucket: 1 unless the model is chunked (D37).
    pub fn chunks(&self) -> usize {
        self.chunking.map_or(1, |c| c.chunks)
    }

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
            Some(ClassifyFormat::Fev) => fields.extend(["question_type", "instructions"]),
            Some(ClassifyFormat::Agentjev) => fields.extend(["question_type", "instructions"]),
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
        // fev renders noul labels differently, so it needs the type; its
        // instructions may be empty, as fev's own API sends them.
        if self.classify.format == Some(ClassifyFormat::Fev) {
            fields.push("question_type");
        }
        // AgentJev's API requires a nonempty question and has no default.
        if self.classify.format == Some(ClassifyFormat::Agentjev) {
            fields.extend(["question_type", "instructions"]);
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
    /// Every program of one bucket, in chain order (D37).
    pub fn artifact_paths_for_bucket(&self, bucket: usize) -> Vec<PathBuf> {
        artifact_files(&self.manifest.artifact, bucket, self.manifest.chunks()).iter().map(|f| self.dir.join(f)).collect()
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
    /// Every program of one bucket, in chain order (D37).
    pub fn artifact_paths_for_bucket(&self, bucket: usize) -> Vec<PathBuf> {
        artifact_files(&self.manifest.artifact, bucket, self.manifest.chunks()).iter().map(|f| self.dir.join(f)).collect()
    }
    pub fn tokenizer_path(&self) -> PathBuf {
        self.dir.join(&self.manifest.tokenizer)
    }
}

/// Who chose a model's compute units (D38).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum ComputeUnitsSource {
    /// Its manifest's `compute_units`, or the default.
    #[default]
    Manifest,
    /// The daemon's config (`[models."<id>"] compute_units`).
    Operator,
}

impl ComputeUnitsSource {
    pub fn name(self) -> &'static str {
        match self {
            Self::Manifest => "manifest",
            Self::Operator => "operator",
        }
    }
}

/// How [`ModelRegistry::scan_with`] scans.
#[derive(Debug, Clone, Default)]
pub struct ScanOptions {
    /// Compute units the operator chose for some Core ML models, by model
    /// id (D38). They replace the manifest's before the weight limit (D32)
    /// and the CPU cap (D33) judge the model.
    pub compute_units: BTreeMap<String, ComputeUnits>,
    /// Load models served on the ANE even past
    /// [`MAX_ANE_PROGRAM_WEIGHT_BYTES`] (`sidekickd
    /// --ignore-ane-weight-cap`, or the parity suite, which measures every
    /// compute path itself).
    pub ignore_ane_weight_cap: bool,
    /// Serve `cpu_only` models past [`MAX_CPU_INVARIANT_SEQ`] (`sidekickd
    /// --ignore-cpu-seq-cap`, or the parity suite, which grades every
    /// bucket and reports the CPU's past-1,024 variation itself).
    pub ignore_cpu_seq_cap: bool,
}

/// The reason a capped model's listing gives.
const CPU_SEQ_CAP_REASON: &str =
    "served cpu_only: Core ML's CPU sums in a length-dependent order past 1,024 tokens, so buckets past it \
     would give slightly different results (D33)";

/// Cap a `cpu_only` model's buckets at [`MAX_CPU_INVARIANT_SEQ`]: keep the
/// buckets up to it, and make the largest kept bucket the effective
/// `max_seq_len`. `None` when no cap applies; an error when no bucket is
/// short enough.
fn cpu_seq_cap(
    buckets: &mut Vec<usize>,
    max_seq_len: &mut usize,
    units: ComputeUnits,
    limit: CpuSeqLimit,
    options: &ScanOptions,
) -> std::result::Result<Option<SeqCap>, String> {
    if units != ComputeUnits::CpuOnly
        || limit == CpuSeqLimit::Ignore
        || options.ignore_cpu_seq_cap
        || *max_seq_len <= MAX_CPU_INVARIANT_SEQ
    {
        return Ok(None);
    }
    buckets.retain(|&b| b <= MAX_CPU_INVARIANT_SEQ);
    let Some(&largest) = buckets.last() else {
        return Err(format!(
            "every bucket is longer than {MAX_CPU_INVARIANT_SEQ} tokens, past which Core ML's CPU gives \
             bucket-dependent results (D33), and the model is served `cpu_only`. Serve it on the GPU or \
             the ANE, convert a bucket of at most {MAX_CPU_INVARIANT_SEQ} tokens, or set \
             `cpu_seq_limit = \"ignore\"` (or start sidekickd with --ignore-cpu-seq-cap)"
        ));
    };
    let cap = SeqCap { limit: largest, manifest_max_seq_len: *max_seq_len, reason: CPU_SEQ_CAP_REASON, state_max_len: None };
    *max_seq_len = largest;
    Ok(Some(cap))
}

/// The question and options' share of a fev window, which fev's own
/// `limits()` keeps free of the state: the state gets `window - 640`.
const FEV_QUESTION_RESERVE: usize = 640;

/// fev's state limit for a window, by fev's own formula, applied when the
/// CPU cap shrinks the window (D33). It is sidekick's choice, not the
/// checkpoint's: the model is never served a shorter window by its own
/// code. 0 when the window leaves no room, which then fails validation.
fn fev_state_max_len(window: usize) -> usize {
    window.saturating_sub(FEV_QUESTION_RESERVE)
}

/// The bytes of a compiled Core ML artifact's weights: every file under its
/// `weights/` directory (an `.mlmodelc`), or under
/// `Data/com.apple.CoreML/weights/` (an `.mlpackage`), as the converters
/// measure them. Missing directories count as 0. File lengths are read from
/// metadata, so the check reads no weights.
pub fn artifact_weight_bytes(artifact: &Path) -> u64 {
    fn walk(dir: &Path) -> u64 {
        let Ok(entries) = std::fs::read_dir(dir) else { return 0 };
        entries
            .filter_map(|e| e.ok())
            .map(|e| {
                let path = e.path();
                match std::fs::metadata(&path) {
                    Ok(m) if m.is_dir() => walk(&path),
                    Ok(m) => m.len(),
                    Err(_) => 0,
                }
            })
            .sum()
    }
    walk(&artifact.join("weights")) + walk(&artifact.join("Data/com.apple.CoreML/weights"))
}

/// Skip a model served on the ANE (`cpu_and_ne` or `all`) when a bucket's
/// compiled weights exceed [`MAX_ANE_PROGRAM_WEIGHT_BYTES`]: Core ML would
/// run it off the ANE with no error, so the model would look ANE-served and
/// not be. Each `{seq}` bucket, and each `{chunk}` of a chained one (D37),
/// is its own program and is checked on its own.
fn check_ane_weights(
    dir: &Path,
    artifact: &str,
    buckets: &[usize],
    chunks: usize,
    units: ComputeUnits,
    source: ComputeUnitsSource,
    limit: AneWeightLimit,
    options: &ScanOptions,
) -> std::result::Result<(), String> {
    let on_ane = matches!(units, ComputeUnits::CpuAndNeuralEngine | ComputeUnits::All);
    if !on_ane || limit == AneWeightLimit::Ignore || options.ignore_ane_weight_cap {
        return Ok(());
    }
    match over_ane_weight_limit(dir, artifact, buckets, chunks) {
        None => Ok(()),
        Some(reason) if source == ComputeUnitsSource::Operator => Err(format!(
            "{reason}; served with `{}`, which the daemon config's `[models.\"<id>\"]` chose (D38), Core ML \
             would run it off the ANE without an error. Remove that override, convert it in chunks under the \
             limit (`--chunks auto`, D37) or as a quantized variant, or start sidekickd with \
             --ignore-ane-weight-cap to load it anyway",
            units.name()
        )),
        Some(reason) => Err(format!(
            "{reason}; served with `{}`, Core ML would run it off the ANE without an error. Convert it in \
             chunks under the limit (`--chunks auto`, D37), serve it on the GPU (`compute_units = \
             \"cpu_and_gpu\"`), convert a quantized variant (`--int8-embedding`), or set \
             `ane_weight_limit = \"ignore\"` (or start sidekickd with --ignore-ane-weight-cap) to load it anyway",
            units.name()
        )),
    }
}

/// The first bucket whose compiled weights exceed
/// [`MAX_ANE_PROGRAM_WEIGHT_BYTES`], as a sentence: "`model_512.mlmodelc` has
/// 1.115 GiB of weights, past Core ML's 1 GiB limit for running a program on
/// the ANE (MAX_ANE_PROGRAM_WEIGHT_BYTES)". Each `{seq}` bucket, and each
/// `{chunk}` of a chained one, is its own program and is checked on its own.
fn over_ane_weight_limit(dir: &Path, artifact: &str, buckets: &[usize], chunks: usize) -> Option<String> {
    let mut names: Vec<String> = buckets.iter().flat_map(|&b| artifact_files(artifact, b, chunks)).collect();
    names.dedup();
    names.into_iter().find_map(|name| {
        let bytes = artifact_weight_bytes(&dir.join(&name));
        (bytes > MAX_ANE_PROGRAM_WEIGHT_BYTES).then(|| {
            format!(
                "`{name}` has {:.3} GiB of weights, past Core ML's 1 GiB limit for running a program on the ANE \
                 (MAX_ANE_PROGRAM_WEIGHT_BYTES)",
                bytes as f64 / (1u64 << 30) as f64
            )
        })
    })
}

/// Why the runtime would refuse to serve this model on the ANE (D32), whatever
/// compute units its manifest names: the reason a `cpu_and_ne` manifest would
/// be skipped for, without the fixes the registry's message lists. `None`
/// when every bucket fits, or when the manifest opts out with
/// `ane_weight_limit = "ignore"`. The parity suite reports, rather than
/// grades, the ANE path of a model the runtime would refuse there.
pub fn ane_weight_refusal(
    dir: &Path,
    artifact: &str,
    buckets: &[usize],
    chunks: usize,
    limit: AneWeightLimit,
) -> Option<String> {
    if limit == AneWeightLimit::Ignore {
        return None;
    }
    over_ane_weight_limit(dir, artifact, buckets, chunks)
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
    /// Operator compute-unit overrides that named no Core ML model (D38).
    unmatched_overrides: Vec<String>,
    /// The scanned directory.
    root: PathBuf,
}

impl ModelRegistry {
    /// Scan `models_dir`. Only an unreadable directory is an error: a bad
    /// manifest is skipped with a warning, so one broken model can't take
    /// down every other. A classifier whose id an embedder already uses is
    /// skipped too; the embedder keeps working. So is a model served on the
    /// ANE whose compiled weights exceed [`MAX_ANE_PROGRAM_WEIGHT_BYTES`].
    pub fn scan(models_dir: &Path) -> Result<Self> {
        Self::scan_with(models_dir, &ScanOptions::default())
    }

    /// [`scan`](Self::scan) with options.
    pub fn scan_with(models_dir: &Path, options: &ScanOptions) -> Result<Self> {
        let mut reg = Self { root: models_dir.to_path_buf(), ..Self::default() };
        if !models_dir.exists() {
            return Ok(reg);
        }
        let mut dirs: Vec<PathBuf> = std::fs::read_dir(models_dir)?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.is_dir())
            .collect();
        dirs.sort();
        // Ids whose operator override was applied (D38).
        let mut applied: Vec<String> = Vec::new();

        // Embedders first, so they win an id collision whatever the order.
        for dir in &dirs {
            let path = dir.join(EMBEDDER_MANIFEST);
            if !path.is_file() {
                continue;
            }
            let loaded = read_manifest::<ModelManifest>(&path)
                .and_then(|m| validate_embedder(&m).map(|()| m))
                .map(|mut m| {
                    if let (Some(&units), EmbeddingBackendKind::Coreml) = (options.compute_units.get(&m.id), m.backend) {
                        m.compute_units = Some(units);
                        m.compute_units_source = ComputeUnitsSource::Operator;
                        applied.push(m.id.clone());
                    }
                    m
                })
                .map(|mut m| {
                    let problem = match (&m.placement, m.backend) {
                        (None, _) => None,
                        (Some(_), EmbeddingBackendKind::Static) => Some("a static model has no compute plan".into()),
                        (Some(p), EmbeddingBackendKind::Coreml) => placement_problem(p, &m.buckets, m.chunks()),
                    };
                    check_placement(&path, &mut m.placement, problem);
                    m
                })
                .and_then(|m| {
                    if m.backend != EmbeddingBackendKind::Coreml {
                        return Ok(m);
                    }
                    let units = m.compute_units.unwrap_or_default();
                    let limit = m.ane_weight_limit.unwrap_or_default();
                    check_ane_weights(dir, &m.artifact, &m.buckets, m.chunks(), units, m.compute_units_source, limit, options)
                        .map(|()| m)
                })
                .and_then(|mut m| {
                    if m.backend != EmbeddingBackendKind::Coreml {
                        return Ok(m);
                    }
                    let units = m.compute_units.unwrap_or_default();
                    let limit = m.cpu_seq_limit.unwrap_or_default();
                    m.seq_cap = cpu_seq_cap(&mut m.buckets, &mut m.max_seq_len, units, limit, options)?;
                    if m.seq_cap.is_some() {
                        validate_embedder(&m).map_err(|e| format!("with its length capped for the CPU (D33): {e}"))?;
                    }
                    Ok(m)
                });
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
                .and_then(|m| validate_classifier(&m).map(|()| m))
                .map(|mut m| {
                    if let Some(&units) = options.compute_units.get(&m.id) {
                        m.compute_units = units;
                        m.compute_units_source = ComputeUnitsSource::Operator;
                        applied.push(m.id.clone());
                    }
                    m
                })
                .map(|mut m| {
                    let problem = m.placement.as_ref().and_then(|p| placement_problem(p, &m.buckets, m.chunks()));
                    check_placement(&path, &mut m.placement, problem);
                    m
                })
                .and_then(|m| {
                    check_ane_weights(dir, &m.artifact, &m.buckets, m.chunks(), m.compute_units, m.compute_units_source, m.ane_weight_limit, options)
                        .map(|()| m)
                })
                .and_then(|mut m| {
                    m.seq_cap = cpu_seq_cap(&mut m.buckets, &mut m.max_seq_len, m.compute_units, m.cpu_seq_limit, options)?;
                    if let (Some(cap), Some(fev)) = (&mut m.seq_cap, &mut m.classify.fev) {
                        let capped = fev_state_max_len(m.max_seq_len);
                        if capped < fev.state_max_len {
                            fev.state_max_len = capped;
                            cap.state_max_len = Some(capped);
                        }
                    }
                    if m.seq_cap.is_some() {
                        validate_classifier(&m).map_err(|e| format!("with its length capped for the CPU (D33): {e}"))?;
                    }
                    Ok(m)
                });
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
        reg.unmatched_overrides = options.compute_units.keys().filter(|id| !applied.contains(id)).cloned().collect();
        for id in &reg.unmatched_overrides {
            tracing::warn!("the config sets compute units for `{id}`, and no Core ML model has that id");
        }
        Ok(reg)
    }

    /// Operator compute-unit overrides (D38) that named no Core ML model in
    /// the models directory.
    pub fn unmatched_overrides(&self) -> &[String] {
        &self.unmatched_overrides
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
    if m.backend != EmbeddingBackendKind::Coreml && m.compute_units.is_some() {
        return Err("`compute_units` is only valid for the coreml backend".into());
    }
    if m.backend != EmbeddingBackendKind::Coreml && m.ane_weight_limit.is_some() {
        return Err("`ane_weight_limit` is only valid for the coreml backend".into());
    }
    if m.backend != EmbeddingBackendKind::Coreml && m.cpu_seq_limit.is_some() {
        return Err("`cpu_seq_limit` is only valid for the coreml backend".into());
    }
    if m.backend != EmbeddingBackendKind::Coreml && (m.chunking.is_some() || m.artifact.contains("{chunk}")) {
        return Err("`[chunking]` is only valid for the coreml backend".into());
    }
    validate_chunking(&m.artifact, m.chunking.as_ref())
}

/// Validation per docs/design/classify.md. The artifact-dependent check
/// (`max_labels` against `marker_pos`'s width) happens at load.
fn validate_classifier(m: &ClassifierManifest) -> std::result::Result<(), String> {
    if m.id.is_empty() {
        return Err("empty model id".into());
    }
    validate_buckets(&m.buckets, m.max_seq_len)?;
    validate_chunking(&m.artifact, m.chunking.as_ref())?;
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
            if c.laya.is_some() || c.gliner2.is_some() || c.fev.is_some() {
                return Err("`[classify.laya]`, `[classify.gliner2]` and `[classify.fev]` are for zero-shot formats".into());
            }
            if io.marker_pos.is_some() || io.qtype.is_some() || io.decide_pos.is_some() || io.seg.is_some()
                || io.position_ids.is_some()
            {
                return Err("text-classification's [classify.io] has input_ids, attention_mask and output only".into());
            }
            // Calibration keys are per question type, which fixed-label
            // models don't have.
            if !c.calibration.is_empty() {
                return Err("`[classify.calibration]` is for zero-shot formats".into());
            }
        }
        ClassifyTask::TextRanking => {
            if c.format.is_some() || c.laya.is_some() || c.gliner2.is_some() || c.fev.is_some() {
                return Err("`format` and the format sections are for zero-shot models".into());
            }
            if c.labels.len() != 1 {
                return Err("a text-ranking model has one output: `labels` names it (e.g. [\"score\"])".into());
            }
            if matches!(c.max_labels, Some(n) if n != 1) {
                return Err("a text-ranking model's `max_labels` is 1".into());
            }
            if io.marker_pos.is_some() || io.qtype.is_some() || io.decide_pos.is_some() || io.seg.is_some()
                || io.position_ids.is_some()
            {
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
            // Each format's section belongs to that format alone.
            for (section, present, owner) in [
                ("laya", c.laya.is_some(), ClassifyFormat::Laya),
                ("gliner2", c.gliner2.is_some(), ClassifyFormat::Gliner2),
                ("fev", c.fev.is_some(), ClassifyFormat::Fev),
            ] {
                if present && format != owner {
                    return Err(format!("`[classify.{section}]` is for the {section} format"));
                }
            }
            if format != ClassifyFormat::Fev && io.decide_pos.is_some() {
                return Err("`decide_pos` is for the fev format".into());
            }
            if format != ClassifyFormat::Agentjev && (io.seg.is_some() || io.position_ids.is_some()) {
                return Err("`seg` and `position_ids` are for the agentjev format".into());
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
                }
                ClassifyFormat::Gliner2 => {
                    match &c.gliner2 {
                        Some(g) if !g.default_instructions.trim().is_empty() => {}
                        Some(_) => return Err("`[classify.gliner2] default_instructions` must not be empty".into()),
                        None => return Err("the gliner2 format needs `[classify.gliner2]`".into()),
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
                ClassifyFormat::Fev => {
                    let Some(fev) = &c.fev else {
                        return Err("the fev format needs `[classify.fev]`".into());
                    };
                    if fev.state_max_len == 0 || fev.state_max_len >= m.max_seq_len {
                        return Err("`[classify.fev] state_max_len` must be in 1..max_seq_len".into());
                    }
                    let names: Vec<&str> = fev.delimiters.by_role().iter().map(|(_, t)| *t).collect();
                    if names.iter().any(|t| t.is_empty()) {
                        return Err("`[classify.fev] delimiters` must name a token for every role".into());
                    }
                    if first_duplicate(&names.iter().map(|t| t.to_string()).collect::<Vec<_>>()).is_some() {
                        return Err("`[classify.fev] delimiters` must be five different tokens".into());
                    }
                    require("marker_pos", &io.marker_pos)?;
                    require("decide_pos", &io.decide_pos)?;
                    if io.qtype.is_some() || io.token_type_ids.is_some() {
                        return Err("the fev format takes no `qtype` or `token_type_ids`".into());
                    }
                    if m.problem_type != ProblemType::SingleLabel {
                        return Err("the fev format's problem_type is `single_label`".into());
                    }
                    // The checkpoint's probabilities are a plain softmax; it
                    // has no calibration of its own.
                    if !c.calibration.is_empty() {
                        return Err("`[classify.calibration]` isn't supported by the fev format".into());
                    }
                }
                ClassifyFormat::Agentjev => {
                    require("marker_pos", &io.marker_pos)?;
                    require("seg", &io.seg)?;
                    require("position_ids", &io.position_ids)?;
                    if io.qtype.is_some() || io.token_type_ids.is_some() {
                        return Err("the agentjev format takes no `qtype` or `token_type_ids`".into());
                    }
                    if m.problem_type != ProblemType::SingleLabel {
                        return Err("the agentjev format's problem_type is `single_label`".into());
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

    const FEV: &str = r#"
id = "lumma-fev"
task = "zero-shot-classification"
artifact = "model_{seq}.mlmodelc"
tokenizer = "tokenizer.json"
buckets = [128, 2048]
max_seq_len = 2048
problem_type = "single_label"

[classify]
format = "fev"
max_labels = 32

[classify.fev]
state_max_len = 1408
delimiters = { state = "<|r0|>", question = "<|r1|>", option = "<|r2|>", option_end = "<|r3|>", decide = "<|r4|>" }

[classify.io]
input_ids = "input_ids"
attention_mask = "attention_mask"
marker_pos = "marker_pos"
decide_pos = "decide_pos"
output = "logits"
"#;

    #[test]
    fn fev_classifiers_and_what_they_refuse() {
        let tmp = tmp_dir("fev");
        write_classifier(&tmp, "f", FEV);
        let reg = ModelRegistry::scan(&tmp).unwrap();
        assert!(reg.skipped().is_empty(), "{:?}", reg.skipped().first().map(|s| &s.reason));
        let m = &reg.classifier("lumma-fev").unwrap().manifest;
        let fev = m.classify.fev.as_ref().unwrap();
        assert_eq!((fev.state_max_len, fev.delimiters.decide.as_str()), (1408, "<|r4|>"));
        assert_eq!(m.extension_fields(), vec!["candidate_labels", "question_type", "instructions"]);
        assert_eq!(m.required_fields(), vec!["candidate_labels", "question_type"]);
        std::fs::remove_dir_all(&tmp).unwrap();

        for (name, body, want) in [
            ("no-section", FEV.replace("[classify.fev]", "[classify.other]"), "needs `[classify.fev]`"),
            ("state-too-long", FEV.replace("state_max_len = 1408", "state_max_len = 2048"), "state_max_len"),
            ("state-zero", FEV.replace("state_max_len = 1408", "state_max_len = 0"), "state_max_len"),
            ("same-delimiter", FEV.replace("decide = \"<|r4|>\"", "decide = \"<|r3|>\""), "five different tokens"),
            ("empty-delimiter", FEV.replace("decide = \"<|r4|>\"", "decide = \"\""), "every role"),
            ("missing-role", FEV.replace(", decide = \"<|r4|>\"", ""), "decide"),
            ("no-decide-io", FEV.replace("decide_pos = \"decide_pos\"\n", ""), "`decide_pos`"),
            ("no-marker-io", FEV.replace("marker_pos = \"marker_pos\"\n", ""), "`marker_pos`"),
            ("qtype-io", FEV.replace("output = ", "qtype = \"qtype\"\noutput = "), "no `qtype`"),
            ("multi-label", FEV.replace("problem_type = \"single_label\"", "problem_type = \"multi_label\""), "single_label"),
            ("calibration", format!("{FEV}\n[classify.calibration]\n\"choice:2\" = 1.0\n"), "isn't supported by the fev"),
            ("laya-section", format!("{FEV}\n[classify.laya]\nhead_max_len = 8\n"), "for the laya format"),
            ("fev-on-laya", format!("{LAYA}\n[classify.fev]\nstate_max_len = 8\ndelimiters = {{ state = \"a\", question = \"b\", option = \"c\", option_end = \"d\", decide = \"e\" }}\n"), "for the fev format"),
            ("decide-on-laya", LAYA.replace("qtype = \"qtype\"", "qtype = \"qtype\"\ndecide_pos = \"d\""), "`decide_pos` is for the fev format"),
            ("fev-on-fixed", format!("{SENTIMENT}\n[classify.fev]\nstate_max_len = 8\ndelimiters = {{ state = \"a\", question = \"b\", option = \"c\", option_end = \"d\", decide = \"e\" }}\n"), "zero-shot formats"),
        ] {
            let tmp = tmp_dir(&format!("fev-{name}"));
            write_classifier(&tmp, name, &body);
            let reg = ModelRegistry::scan(&tmp).unwrap();
            assert_eq!(reg.skipped().len(), 1, "{name}");
            assert!(reg.skipped()[0].reason.contains(want), "{name}: {}", reg.skipped()[0].reason);
            std::fs::remove_dir_all(&tmp).unwrap();
        }
    }

    const AGENTJEV: &str = r#"
id = "agent-jev"
task = "zero-shot-classification"
artifact = "model_{seq}.mlmodelc"
tokenizer = "tokenizer.json"
buckets = [512, 2048]
max_seq_len = 2048
problem_type = "single_label"
compute_units = "cpu_and_gpu"

[classify]
format = "agentjev"
max_labels = 32

[classify.calibration]
"noul:2" = 1.0718
"choice:2" = 1.0353
"choice:3-5" = 1.0353
"choice:6-10" = 1.0353
"choice:11+" = 1.0353
"score:2" = 1.0718
"score:3-5" = 1.0718
"score:6-10" = 1.0718

[classify.io]
input_ids = "input_ids"
attention_mask = "attention_mask"
seg = "seg"
position_ids = "position_ids"
marker_pos = "cand_end"
output = "logits"
"#;

    #[test]
    fn agentjev_classifiers_and_what_they_refuse() {
        let tmp = tmp_dir("agentjev");
        write_classifier(&tmp, "j", AGENTJEV);
        let reg = ModelRegistry::scan(&tmp).unwrap();
        assert!(reg.skipped().is_empty(), "{:?}", reg.skipped().first().map(|s| &s.reason));
        let m = &reg.classifier("agent-jev").unwrap().manifest;
        assert_eq!(m.classify.format, Some(ClassifyFormat::Agentjev));
        assert_eq!((m.classify.io.seg.as_deref(), m.classify.io.position_ids.as_deref()), (Some("seg"), Some("position_ids")));
        assert_eq!(m.extension_fields(), vec!["candidate_labels", "calibration", "question_type", "instructions"]);
        assert_eq!(m.required_fields(), vec!["candidate_labels", "question_type", "instructions"]);
        assert_eq!(m.temperature(Some(QuestionType::Noul), 2), Some(1.0718));
        assert_eq!(m.temperature(Some(QuestionType::Choice), 32), Some(1.0353));
        assert_eq!(m.temperature(Some(QuestionType::Score), 10), Some(1.0718));
        std::fs::remove_dir_all(&tmp).unwrap();

        for (name, body, want) in [
            ("no-seg-io", AGENTJEV.replace("seg = \"seg\"\n", ""), "`seg`"),
            ("no-position-io", AGENTJEV.replace("position_ids = \"position_ids\"\n", ""), "`position_ids`"),
            ("no-marker-io", AGENTJEV.replace("marker_pos = \"cand_end\"\n", ""), "`marker_pos`"),
            ("qtype-io", AGENTJEV.replace("output = ", "qtype = \"qtype\"\noutput = "), "no `qtype`"),
            ("multi-label", AGENTJEV.replace("problem_type = \"single_label\"", "problem_type = \"multi_label\""), "single_label"),
            ("laya-section", format!("{AGENTJEV}\n[classify.laya]\nhead_max_len = 8\n"), "for the laya format"),
            ("seg-on-laya", LAYA.replace("qtype = \"qtype\"", "qtype = \"qtype\"\nseg = \"seg\""), "for the agentjev format"),
            ("seg-on-fixed", SENTIMENT.replace("output = ", "position_ids = \"p\"\noutput = "), "input_ids, attention_mask and output only"),
        ] {
            let tmp = tmp_dir(&format!("agentjev-{name}"));
            write_classifier(&tmp, name, &body);
            let reg = ModelRegistry::scan(&tmp).unwrap();
            assert_eq!(reg.skipped().len(), 1, "{name}");
            assert!(reg.skipped()[0].reason.contains(want), "{name}: {}", reg.skipped()[0].reason);
            std::fs::remove_dir_all(&tmp).unwrap();
        }
    }

    #[test]
    fn a_cpu_capped_fev_window_keeps_fevs_question_reserve() {
        let cpu = |body: &str| body.replace("\n[classify]", "compute_units = \"cpu_only\"\n\n[classify]");
        let scan = |name: &str, body: &str| {
            let tmp = tmp_dir(&format!("fev-cap-{name}"));
            write_classifier(&tmp, name, body);
            let reg = ModelRegistry::scan(&tmp).unwrap();
            let out = (reg.classifier("lumma-fev").ok().map(|c| c.manifest.clone()), reg.skipped().first().map(|s| s.reason.clone()));
            std::fs::remove_dir_all(&tmp).unwrap();
            out
        };
        // Capped at 1,024, the state gets fev's own window - 640: 384.
        let three = FEV.replace("buckets = [128, 2048]", "buckets = [128, 1024, 2048]");
        let m = scan("capped", &cpu(&three)).0.unwrap();
        let cap = m.seq_cap.as_ref().unwrap();
        assert_eq!((m.max_seq_len, m.classify.fev.as_ref().unwrap().state_max_len), (1024, 384));
        assert_eq!(cap.state_max_len, Some(384));
        assert_eq!(serde_json::to_value(cap).unwrap()["state_max_len"], 384);
        // A state limit already under it is kept, and not reported.
        let m = scan("short", &cpu(&three.replace("state_max_len = 1408", "state_max_len = 300"))).0.unwrap();
        assert_eq!((m.classify.fev.as_ref().unwrap().state_max_len, m.seq_cap.as_ref().unwrap().state_max_len), (300, None));
        assert!(serde_json::to_value(m.seq_cap.as_ref().unwrap()).unwrap().get("state_max_len").is_none());
        // Not capped: the manifest's own limit.
        let m = scan("gpu", &three).0.unwrap();
        assert_eq!((m.max_seq_len, m.classify.fev.as_ref().unwrap().state_max_len), (2048, 1408));
        // A capped window with no room for a state is skipped.
        let (m, why) = scan("tiny", &cpu(FEV));
        assert!(m.is_none());
        assert!(why.as_deref().unwrap().contains("with its length capped for the CPU (D33)"), "{why:?}");
    }

    /// A classifier manifest's `[placement]`, parsed without checks.
    fn reg_placement(body: &str) -> RecordedPlacement {
        toml::from_str::<ClassifierManifest>(body).unwrap().placement.unwrap()
    }

    #[test]
    fn a_recorded_placement_is_read_and_checked() {
        let placement = "\n[placement]\ncompute_units = \"cpu_and_ne\"\nchip = \"Apple M1 Max\"\nmacos_build = \"25A354\"\n\
                         date = \"2026-10-01\"\n\
                         [placement.buckets.128]\nane = 294\ngpu = 0\ncpu = 10\nunassigned = 410\ntotal = 714\n\
                         off_ane_ops = { gather = 2, cast = 8 }\n";
        let tmp = tmp_dir("placement");
        write_classifier(&tmp, "s", &format!("{SENTIMENT}{placement}"));
        let reg = ModelRegistry::scan(&tmp).unwrap();
        assert!(reg.skipped().is_empty(), "{:?}", reg.skipped().first().map(|s| &s.reason));
        let p = reg.classifier("sentiment").unwrap().manifest.placement.clone().unwrap();
        assert_eq!((p.compute_units, p.chip.as_str(), p.macos_build.as_str()), (ComputeUnits::CpuAndNeuralEngine, "Apple M1 Max", "25A354"));
        assert_eq!(p.date.as_deref(), Some("2026-10-01"));
        let b = p.bucket(128).unwrap();
        assert_eq!((b.ane, b.cpu, b.total, b.off_ane_ops["gather"]), (294, 10, 714, 2));
        assert!(p.bucket(512).is_none(), "a bucket may go unrecorded");
        std::fs::remove_dir_all(&tmp).unwrap();

        // A record that doesn't fit the model is dropped; the model still loads.
        let sentiment = |p: &RecordedPlacement| placement_problem(p, &[128, 512], 1);
        let mut p = reg_placement(&format!("{SENTIMENT}{}", placement.replace("buckets.128", "buckets.256")));
        assert!(sentiment(&p).unwrap().contains("`256`, which isn't one of the buckets"));
        p = reg_placement(&format!("{SENTIMENT}{}", placement.replace("total = 714", "total = 700")));
        assert!(sentiment(&p).unwrap().contains("don't add up"));
        let tmp = tmp_dir("placement-dropped");
        write_classifier(&tmp, "s", &format!("{SENTIMENT}{}", placement.replace("total = 714", "total = 700")));
        let reg = ModelRegistry::scan(&tmp).unwrap();
        assert!(reg.skipped().is_empty());
        assert!(reg.classifier("sentiment").unwrap().manifest.placement.is_none());
        std::fs::remove_dir_all(&tmp).unwrap();

        // Embedders: coreml only; a cap that drops a recorded bucket is fine.
        let tmp = tmp_dir("placement-embedders");
        let embedder = |id: &str, backend: &str, extra: &str| {
            format!("id = \"{id}\"\nbackend = \"{backend}\"\nartifact = \"m_{{seq}}\"\ntokenizer = \"t\"\ndims = 4\n\
                     buckets = [128, 2048]\nmax_seq_len = 2048\n{extra}{}", placement.replace("buckets.128", "buckets.2048"))
        };
        write_manifest(&tmp, "e", &embedder("e", "coreml", "compute_units = \"cpu_only\"\n"));
        let static_model = "id = \"s\"\nbackend = \"static\"\nartifact = \"m\"\ntokenizer = \"t\"\ndims = 4\nmax_seq_len = 8\n";
        write_manifest(&tmp, "s", &format!("{static_model}{placement}"));
        let reg = ModelRegistry::scan(&tmp).unwrap();
        let e = &reg.get("e").unwrap().manifest;
        assert_eq!(e.buckets, vec![128], "capped for the CPU");
        assert!(e.placement.as_ref().unwrap().bucket(2048).is_some());
        assert!(reg.skipped().is_empty(), "{:?}", reg.skipped().first().map(|s| &s.reason));
        assert!(reg.get("s").unwrap().manifest.placement.is_none(), "a static model's is dropped");
        std::fs::remove_dir_all(&tmp).unwrap();
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
    fn compute_units_default_to_the_ane_and_unknown_values_are_skipped() {
        const COREML: &str = "id = \"e\"\nbackend = \"coreml\"\nartifact = \"m\"\ntokenizer = \"t\"\ndims = 4\nbuckets = [8]\nmax_seq_len = 8\n";
        const STATIC: &str = "id = \"s\"\nbackend = \"static\"\nartifact = \"m\"\ntokenizer = \"t\"\ndims = 4\nmax_seq_len = 8\n";
        let tmp = tmp_dir("units");
        write_classifier(&tmp, "default", SENTIMENT);
        let gpu = SENTIMENT.replace("id = \"sentiment\"", "id = \"gpu\"").replace("\n[classify]", "compute_units = \"cpu_and_gpu\"\n\n[classify]");
        write_classifier(&tmp, "gpu", &gpu);
        write_manifest(&tmp, "e-default", COREML);
        write_manifest(&tmp, "e-cpu", &format!("{}compute_units = \"cpu_only\"\n", COREML.replace("\"e\"", "\"e-cpu\"")));
        write_manifest(&tmp, "s", STATIC);
        let reg = ModelRegistry::scan(&tmp).unwrap();
        assert!(reg.skipped().is_empty(), "{:?}", reg.skipped().first().map(|s| &s.reason));
        assert_eq!(reg.classifier("sentiment").unwrap().manifest.compute_units, ComputeUnits::CpuAndNeuralEngine);
        assert_eq!(reg.classifier("gpu").unwrap().manifest.compute_units, ComputeUnits::CpuAndGpu);
        let units = |id| reg.get(id).unwrap().manifest.compute_units_name();
        assert_eq!((units("e"), units("e-cpu"), units("s")), ("cpu_and_ne", "cpu_only", "cpu"));
        std::fs::remove_dir_all(&tmp).unwrap();

        // An unknown value, or units on a static model, skips that manifest
        // with the reason; the rest load.
        let tmp = tmp_dir("units-bad");
        write_classifier(&tmp, "good", SENTIMENT);
        write_classifier(&tmp, "typo", &SENTIMENT.replace("id = \"sentiment\"", "id = \"typo\"").replace("\n[classify]", "compute_units = \"gpu\"\n\n[classify]"));
        write_manifest(&tmp, "s", &format!("{STATIC}compute_units = \"cpu_only\"\n"));
        let reg = ModelRegistry::scan(&tmp).unwrap();
        assert_eq!(reg.classifier_ids().collect::<Vec<_>>(), vec!["sentiment"]);
        let reasons: Vec<&str> = reg.skipped().iter().map(|s| s.reason.as_str()).collect();
        assert_eq!(reasons.len(), 2, "{reasons:?}");
        assert!(reasons.iter().any(|r| r.contains("compute_units") && r.contains("gpu")), "{reasons:?}");
        assert!(reasons.iter().any(|r| r.contains("only valid for the coreml backend")), "{reasons:?}");
        std::fs::remove_dir_all(&tmp).unwrap();
    }

    /// A compiled artifact whose weights total `bytes`, as sparse files
    /// (their length counts; no disk is used).
    fn weights(model_dir: &Path, artifact: &str, sub: &str, bytes: u64) {
        let dir = model_dir.join(artifact).join(sub);
        std::fs::create_dir_all(&dir).unwrap();
        let half = bytes / 2;
        for (name, len) in [("weight.bin", half), ("nested/more.bin", bytes - half)] {
            let path = dir.join(name);
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::File::create(&path).unwrap().set_len(len).unwrap();
        }
    }

    #[test]
    fn models_served_on_the_ane_past_its_weight_cap_are_skipped() {
        const CAP: u64 = MAX_ANE_PROGRAM_WEIGHT_BYTES;
        // SENTIMENT's buckets are 128 and 512, artifact model_{seq}.mlmodelc.
        let cases: &[(&str, String, u64, Option<&str>)] = &[
            // Every bucket is its own program: one over the cap skips the model.
            ("over", SENTIMENT.to_string(), CAP + 1, Some("`model_512.mlmodelc` has 1.000 GiB of weights")),
            ("at-cap", SENTIMENT.to_string(), CAP, None),
            ("all-units", SENTIMENT.replace("\n[classify]", "compute_units = \"all\"\n\n[classify]"), CAP + 1, Some("past Core ML's 1 GiB limit")),
            ("on-gpu", SENTIMENT.replace("\n[classify]", "compute_units = \"cpu_and_gpu\"\n\n[classify]"), CAP + 1, None),
            ("on-cpu", SENTIMENT.replace("\n[classify]", "compute_units = \"cpu_only\"\n\n[classify]"), CAP + 1, None),
            ("ignored", SENTIMENT.replace("\n[classify]", "ane_weight_limit = \"ignore\"\n\n[classify]"), CAP + 1, None),
        ];
        for (name, body, bytes, want) in cases {
            let tmp = tmp_dir(&format!("ane-cap-{name}"));
            write_classifier(&tmp, "s", body);
            weights(&tmp.join("s"), "model_128.mlmodelc", "weights", 1 << 20);
            weights(&tmp.join("s"), "model_512.mlmodelc", "weights", *bytes);
            let reg = ModelRegistry::scan(&tmp).unwrap();
            match want {
                Some(want) => {
                    assert!(reg.classifier("sentiment").is_err(), "{name}");
                    let reason = &reg.skipped()[0].reason;
                    assert!(reason.contains(want), "{name}: {reason}");
                    assert!(reason.contains("cpu_and_gpu") && reason.contains("ane_weight_limit"), "{name}: {reason}");
                    // The daemon's --ignore-ane-weight-cap loads it.
                    let opts = ScanOptions { ignore_ane_weight_cap: true, ..Default::default() };
                    assert!(ModelRegistry::scan_with(&tmp, &opts).unwrap().classifier("sentiment").is_ok(), "{name}");
                }
                None => assert!(reg.skipped().is_empty(), "{name}: {:?}", reg.skipped().first().map(|s| &s.reason)),
            }
            std::fs::remove_dir_all(&tmp).unwrap();
        }
    }

    #[test]
    fn ane_weight_refusal_answers_for_the_ane_whatever_the_units() {
        const CAP: u64 = MAX_ANE_PROGRAM_WEIGHT_BYTES;
        let tmp = tmp_dir("ane-refusal");
        let dir = tmp.join("s");
        weights(&dir, "model_128.mlmodelc", "weights", 1 << 20);
        weights(&dir, "model_512.mlmodelc", "weights", CAP + 1);
        let buckets = [128, 512];
        // Over the cap: refused, though the call names no compute units (a
        // GPU-served manifest gets the same answer).
        let why = ane_weight_refusal(&dir, "model_{seq}.mlmodelc", &buckets, 1, AneWeightLimit::Enforce).unwrap();
        // The reason alone: the registry's fixes aren't part of it.
        assert_eq!(
            why,
            "`model_512.mlmodelc` has 1.000 GiB of weights, past Core ML's 1 GiB limit for running a program \
             on the ANE (MAX_ANE_PROGRAM_WEIGHT_BYTES)"
        );
        // The opt-out is honored, and a model that fits isn't refused.
        assert_eq!(ane_weight_refusal(&dir, "model_{seq}.mlmodelc", &buckets, 1, AneWeightLimit::Ignore), None);
        assert_eq!(ane_weight_refusal(&dir, "model_{seq}.mlmodelc", &[128], 1, AneWeightLimit::Enforce), None);
        std::fs::remove_dir_all(&tmp).unwrap();
    }

    /// SENTIMENT in two chunks per bucket (D37).
    fn chunked_sentiment() -> String {
        SENTIMENT.replace("model_{seq}.mlmodelc", "model_{seq}.{chunk}.mlmodelc")
            + "\n[chunking]\nchunks = 2\nweight_budget_bytes = 966367641\n"
    }

    #[test]
    fn a_chunked_bucket_is_checked_against_the_weight_cap_chunk_by_chunk() {
        const CAP: u64 = MAX_ANE_PROGRAM_WEIGHT_BYTES;
        assert_eq!(artifact_files("model_{seq}.{chunk}.mlmodelc", 512, 2), ["model_512.0.mlmodelc", "model_512.1.mlmodelc"]);
        assert_eq!(artifact_files("model_{seq}.mlmodelc", 512, 1), ["model_512.mlmodelc"]);
        // Each chunk under the cap, the bucket over it in all: served.
        let tmp = tmp_dir("chunked-fits");
        write_classifier(&tmp, "s", &chunked_sentiment());
        for b in [128, 512] {
            for c in 0..2 {
                weights(&tmp.join("s"), &format!("model_{b}.{c}.mlmodelc"), "weights", CAP / 2 + (1 << 20));
            }
        }
        let reg = ModelRegistry::scan(&tmp).unwrap();
        assert!(reg.skipped().is_empty(), "{:?}", reg.skipped().first().map(|s| &s.reason));
        let m = &reg.classifier("sentiment").unwrap().manifest;
        assert_eq!(m.chunks(), 2);
        assert_eq!(m.chunking.unwrap().weight_budget_bytes, Some(966367641));
        assert_eq!(
            reg.classifier("sentiment").unwrap().artifact_paths_for_bucket(128),
            [tmp.join("s/model_128.0.mlmodelc"), tmp.join("s/model_128.1.mlmodelc")]
        );
        // One chunk over it: skipped, naming that chunk, with chunking first among the fixes.
        weights(&tmp.join("s"), "model_512.1.mlmodelc", "weights", CAP + 1);
        let reg = ModelRegistry::scan(&tmp).unwrap();
        let reason = &reg.skipped()[0].reason;
        assert!(reason.starts_with("`model_512.1.mlmodelc` has 1.000 GiB of weights"), "{reason}");
        assert!(reason.contains("Convert it in chunks under the limit (`--chunks auto`, D37)"), "{reason}");
        assert_eq!(
            ane_weight_refusal(&tmp.join("s"), &m.artifact, &m.buckets, 2, AneWeightLimit::Enforce).as_deref(),
            Some(
                "`model_512.1.mlmodelc` has 1.000 GiB of weights, past Core ML's 1 GiB limit for running a program \
                 on the ANE (MAX_ANE_PROGRAM_WEIGHT_BYTES)"
            )
        );
        std::fs::remove_dir_all(&tmp).unwrap();
    }

    #[test]
    fn chunking_and_its_placeholder_come_together() {
        let cases = [
            ("no-table", SENTIMENT.replace("model_{seq}.mlmodelc", "model_{seq}.{chunk}.mlmodelc"), "no `[chunking]` table"),
            ("no-placeholder", SENTIMENT.to_string() + "\n[chunking]\nchunks = 2\n", "needs a `{chunk}` placeholder"),
            ("zero", chunked_sentiment().replace("chunks = 2", "chunks = 0"), "must be at least 1"),
        ];
        for (name, body, want) in cases {
            let tmp = tmp_dir(&format!("chunking-{name}"));
            write_classifier(&tmp, "s", &body);
            let reg = ModelRegistry::scan(&tmp).unwrap();
            let reason = &reg.skipped().first().unwrap_or_else(|| panic!("{name}: not skipped")).reason;
            assert!(reason.contains(want), "{name}: {reason}");
            std::fs::remove_dir_all(&tmp).unwrap();
        }
        // A static embedder has no chunks.
        let tmp = tmp_dir("chunking-static");
        let dir = tmp.join("e");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join(EMBEDDER_MANIFEST),
            "id = \"e\"\nbackend = \"static\"\nartifact = \"m.{chunk}.bin\"\ntokenizer = \"t\"\ndims = 4\n\
             max_seq_len = 8\n\n[chunking]\nchunks = 2\n",
        )
        .unwrap();
        let reg = ModelRegistry::scan(&tmp).unwrap();
        assert!(reg.skipped()[0].reason.contains("only valid for the coreml backend"), "{:?}", reg.skipped());
        std::fs::remove_dir_all(&tmp).unwrap();
    }

    #[test]
    fn a_recorded_placement_lists_each_chunk() {
        let plan = |ane, chunks: Vec<RecordedPlan>| RecordedPlan {
            ane,
            gpu: 0,
            cpu: 1,
            unassigned: 2,
            total: ane + 3,
            off_ane_ops: BTreeMap::new(),
            chunks,
        };
        let placement = |chunks| RecordedPlacement {
            compute_units: ComputeUnits::CpuAndNeuralEngine,
            chip: "c".into(),
            macos_build: "b".into(),
            date: None,
            buckets: [("128".to_string(), plan(10, chunks))].into(),
        };
        // Chunks that add up, and as many as the model has: kept.
        assert_eq!(placement_problem(&placement(vec![plan(4, vec![]), plan(6, vec![])]), &[128], 2), None);
        // A chunk whose counts don't add up, or the wrong number of chunks: dropped.
        let mut bad = plan(4, vec![]);
        bad.total += 1;
        assert!(placement_problem(&placement(vec![bad, plan(6, vec![])]), &[128], 2).is_some());
        let why = placement_problem(&placement(vec![plan(4, vec![]), plan(6, vec![])]), &[128], 3).unwrap();
        assert!(why.contains("lists 2 chunks, and the model has 3"), "{why}");
        // Parsed from the converter's TOML.
        let toml = "[placement]\ncompute_units = \"cpu_and_ne\"\nchip = \"c\"\nmacos_build = \"b\"\n\n\
                    [placement.buckets.128]\nane = 10\ngpu = 0\ncpu = 1\nunassigned = 2\ntotal = 13\noff_ane_ops = {}\n\n\
                    [[placement.buckets.128.chunks]]\nane = 4\ngpu = 0\ncpu = 1\nunassigned = 2\ntotal = 7\n\
                    off_ane_ops = { cast = 1 }\n\n\
                    [[placement.buckets.128.chunks]]\nane = 6\ngpu = 0\ncpu = 0\nunassigned = 0\ntotal = 6\n";
        #[derive(Deserialize)]
        struct Doc {
            placement: RecordedPlacement,
        }
        let doc: Doc = toml::from_str(toml).unwrap();
        let b = doc.placement.bucket(128).unwrap();
        assert_eq!(b.chunks.iter().map(|c| c.ane).collect::<Vec<_>>(), [4, 6]);
        assert_eq!(b.chunks[0].off_ane_ops["cast"], 1);
    }

    #[test]
    fn the_operators_compute_units_replace_the_manifests_and_meet_the_same_limits() {
        const CAP: u64 = MAX_ANE_PROGRAM_WEIGHT_BYTES;
        let tmp = tmp_dir("operator-units");
        // Over the ANE's weight limit, and served on the GPU by its manifest.
        let gpu = SENTIMENT.replace("\n[classify]", "compute_units = \"cpu_and_gpu\"\n\n[classify]");
        write_classifier(&tmp, "big", &gpu.replace("\"sentiment\"", "\"big\""));
        weights(&tmp.join("big"), "model_128.mlmodelc", "weights", 1 << 20);
        weights(&tmp.join("big"), "model_512.mlmodelc", "weights", CAP + 1);
        // Within it, on the ANE by default.
        write_classifier(&tmp, "small", &SENTIMENT.replace("\"sentiment\"", "\"small\""));
        // Long, for the CPU cap.
        let long = SENTIMENT.replace("[128, 512]", "[128, 1024, 2048]").replace("max_seq_len = 512", "max_seq_len = 2048");
        write_classifier(&tmp, "long", &long.replace("\"sentiment\"", "\"long\""));
        let scan = |overrides: &[(&str, ComputeUnits)]| {
            let options = ScanOptions {
                compute_units: overrides.iter().map(|(id, u)| (id.to_string(), *u)).collect(),
                ..Default::default()
            };
            ModelRegistry::scan_with(&tmp, &options).unwrap()
        };
        // No overrides: the manifests decide.
        let reg = scan(&[]);
        assert!(reg.skipped().is_empty() && reg.unmatched_overrides().is_empty());
        assert_eq!(reg.classifier("big").unwrap().manifest.compute_units_source, ComputeUnitsSource::Manifest);
        // Moved to the GPU, and to the CPU: applied and reported as the operator's.
        let reg = scan(&[("small", ComputeUnits::CpuAndGpu), ("long", ComputeUnits::CpuOnly)]);
        let small = &reg.classifier("small").unwrap().manifest;
        assert_eq!((small.compute_units, small.compute_units_source), (ComputeUnits::CpuAndGpu, ComputeUnitsSource::Operator));
        // The CPU cap (D33) applies to the operator's choice.
        let long = &reg.classifier("long").unwrap().manifest;
        assert_eq!(long.seq_cap.as_ref().map(|c| c.limit), Some(1024));
        // Moved onto the ANE past its weight limit: refused as the manifest's own choice would be (D32).
        let reg = scan(&[("big", ComputeUnits::CpuAndNeuralEngine), ("ghost", ComputeUnits::CpuOnly)]);
        assert!(reg.classifier("big").is_err());
        let why = &reg.skipped()[0].reason;
        assert!(why.contains("past Core ML's 1 GiB limit") && why.contains("Remove that override"), "{why}");
        // An override naming no model is reported, not an error; one naming a skipped model is not unmatched.
        assert_eq!(reg.unmatched_overrides(), ["ghost"]);
        std::fs::remove_dir_all(&tmp).unwrap();
    }

    #[test]
    fn the_ane_weight_cap_covers_embedders_and_mlpackages() {
        const EMBEDDER: &str = "id = \"big\"\nbackend = \"coreml\"\nartifact = \"model.mlpackage\"\ntokenizer = \"t\"\ndims = 4\nbuckets = [8]\nmax_seq_len = 8\n";
        let tmp = tmp_dir("ane-cap-embedder");
        write_manifest(&tmp, "big", EMBEDDER);
        // An .mlpackage keeps its weights under Data/com.apple.CoreML/weights.
        weights(&tmp.join("big"), "model.mlpackage", "Data/com.apple.CoreML/weights", MAX_ANE_PROGRAM_WEIGHT_BYTES + 4096);
        assert_eq!(artifact_weight_bytes(&tmp.join("big/model.mlpackage")), MAX_ANE_PROGRAM_WEIGHT_BYTES + 4096);
        let reg = ModelRegistry::scan(&tmp).unwrap();
        assert!(reg.get("big").is_err());
        assert!(reg.skipped()[0].reason.contains("`model.mlpackage`"), "{}", reg.skipped()[0].reason);
        std::fs::remove_dir_all(&tmp).unwrap();
        // The key is for the coreml backend only.
        let tmp = tmp_dir("ane-cap-static");
        write_manifest(&tmp, "s", "id = \"s\"\nbackend = \"static\"\nartifact = \"m\"\ntokenizer = \"t\"\ndims = 4\nmax_seq_len = 8\nane_weight_limit = \"ignore\"\n");
        let reg = ModelRegistry::scan(&tmp).unwrap();
        assert!(reg.skipped()[0].reason.contains("only valid for the coreml backend"));
        std::fs::remove_dir_all(&tmp).unwrap();
    }

    #[test]
    fn cpu_only_models_are_capped_at_1024_tokens() {
        let long = SENTIMENT.replace("buckets = [128, 512]", "buckets = [128, 1024, 2048]").replace("max_seq_len = 512", "max_seq_len = 2048");
        let with = |top: &str| long.replace("\n[classify]", &format!("{top}\n\n[classify]"));
        let scan = |body: &str, opts: &ScanOptions| {
            let tmp = tmp_dir("cpu-cap");
            write_classifier(&tmp, "s", body);
            let reg = ModelRegistry::scan_with(&tmp, opts).unwrap();
            let out = (
                reg.classifier("sentiment").ok().map(|c| c.manifest.clone()),
                reg.skipped().first().map(|s| s.reason.clone()),
            );
            std::fs::remove_dir_all(&tmp).unwrap();
            out
        };
        let default = ScanOptions::default();

        // cpu_only past 1,024: the buckets above it are dropped and the
        // largest kept one becomes the effective maximum.
        let (m, _) = scan(&with("compute_units = \"cpu_only\""), &default);
        let m = m.unwrap();
        assert_eq!((m.buckets.clone(), m.max_seq_len), (vec![128, 1024], 1024));
        let cap = m.seq_cap.unwrap();
        assert_eq!((cap.limit, cap.manifest_max_seq_len), (1024, 2048));
        assert!(cap.reason.contains("D33"));

        // Not capped: on the ANE, on the GPU, `all` (it's for measuring),
        // the manifest's opt-out, and the daemon's.
        for (top, opts) in [
            ("", &default),
            ("compute_units = \"cpu_and_gpu\"", &default),
            ("compute_units = \"all\"", &default),
            ("compute_units = \"cpu_only\"\ncpu_seq_limit = \"ignore\"", &default),
            ("compute_units = \"cpu_only\"", &ScanOptions { ignore_cpu_seq_cap: true, ..Default::default() }),
        ] {
            let (m, why) = scan(&with(top), opts);
            let m = m.unwrap_or_else(|| panic!("{top}: skipped: {why:?}"));
            assert_eq!((m.max_seq_len, m.seq_cap), (2048, None), "{top}");
        }

        // No bucket short enough: skipped, naming the fixes.
        let only_long = long.replace("buckets = [128, 1024, 2048]", "buckets = [2048]");
        let (m, why) = scan(&only_long.replace("\n[classify]", "compute_units = \"cpu_only\"\n\n[classify]"), &default);
        assert!(m.is_none());
        let why = why.unwrap();
        assert!(why.contains("every bucket is longer than 1024") && why.contains("cpu_seq_limit"), "{why}");

        // A capped manifest must still validate: laya's head budget has to
        // fit the capped length.
        let laya = LAYA
            .replace("buckets = [128, 256, 512]", "buckets = [128, 512, 2048]")
            .replace("max_seq_len = 512", "max_seq_len = 2048")
            .replace("head_max_len = 192", "head_max_len = 1500")
            .replace("\n[classify]", "compute_units = \"cpu_only\"\n\n[classify]");
        let tmp = tmp_dir("cpu-cap-laya");
        write_classifier(&tmp, "l", &laya);
        let reg = ModelRegistry::scan(&tmp).unwrap();
        assert!(reg.skipped()[0].reason.contains("with its length capped for the CPU (D33)"), "{}", reg.skipped()[0].reason);
        std::fs::remove_dir_all(&tmp).unwrap();

        // Embedders too; the key is for the coreml backend only.
        let tmp = tmp_dir("cpu-cap-embedder");
        write_manifest(&tmp, "e", "id = \"e\"\nbackend = \"coreml\"\nartifact = \"m_{seq}\"\ntokenizer = \"t\"\ndims = 4\nbuckets = [512, 2048]\nmax_seq_len = 2048\ncompute_units = \"cpu_only\"\n");
        write_manifest(&tmp, "s", "id = \"s\"\nbackend = \"static\"\nartifact = \"m\"\ntokenizer = \"t\"\ndims = 4\nmax_seq_len = 8\ncpu_seq_limit = \"ignore\"\n");
        let reg = ModelRegistry::scan(&tmp).unwrap();
        let e = &reg.get("e").unwrap().manifest;
        assert_eq!((e.buckets.clone(), e.max_seq_len, e.seq_cap.as_ref().map(|c| c.limit)), (vec![512], 512, Some(512)));
        assert!(reg.skipped()[0].reason.contains("only valid for the coreml backend"));
        std::fs::remove_dir_all(&tmp).unwrap();
    }

    #[test]
    fn missing_dir_is_empty_registry() {
        let reg = ModelRegistry::scan(Path::new("/nonexistent/sidekick-models")).unwrap();
        assert!(reg.is_empty());
    }
}
