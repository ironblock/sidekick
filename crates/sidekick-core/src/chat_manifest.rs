//! `chat.toml`: an open causal language model for the chat route (D53).
//!
//! A model directory with a `chat.toml` holds a decoder converted to the
//! `kv-state-1` program contract (docs/design/open-llm.md, "The program
//! contract" and "The manifest"): N chunk programs over contiguous runs of
//! layers, each one static function taking T tokens per call and owning
//! its layers' KV cache as Core ML state, plus the host's embedding and
//! RoPE tables.
//!
//! This module reads the manifest's top level and its `[decoder]` table,
//! and checks what can be checked without Core ML: the table's own
//! arithmetic, the compute units, the ANE length cap (D48), the program
//! files and the per-program ANE weight limit (D32). Whether the programs
//! declare the inputs, outputs and state shapes the table describes is
//! checked when they load (`sidekick_coreml::KvDecoder`).
//!
//! The `[chat]`, `[sampling]` and `[serving]` tables are the loop's
//! (`sidekick-llm`): the chat template and stop tokens, the checkpoint's
//! sampling defaults and the model's request timeout. They are parsed here
//! so that the registry can skip a model whose tables are wrong, with the
//! reason, before anything loads ([`ChatManifest::check_serving`]); the
//! decoder alone doesn't need them. `[build]` and `[placement]` are the
//! converter's provenance and aren't read.

use crate::classify::Source;
use crate::manifest::{
    ane_seq_cap, artifact_files, artifact_weight_bytes, over_ane_weight_limit,
    symlinked_program_file, ComputeUnits, ScanOptions,
};
use serde::Deserialize;
use std::path::Path;

/// The manifest's filename.
pub const CHAT_MANIFEST: &str = "chat.toml";

/// `task` for an open causal LM.
pub const TEXT_GENERATION_TASK: &str = "text-generation";

/// The program contract this daemon runs.
pub const KV_STATE_INTERFACE: &str = "kv-state-1";

/// The additive mask's value for a key a query must not see: finite, so a
/// fully masked row can't produce NaN (D25).
pub const KV_MASK_HIDDEN: f32 = -30_000.0;

/// The names the `kv-state-1` contract gives its programs' inputs, outputs
/// and states.
pub mod kv_names {
    /// `[1, D, 1, T]` fp16: the residual stream, channels first.
    pub const HIDDEN_IN: &str = "hidden_in";
    /// `[1]` int32: the cache position of the call's first token.
    pub const POS: &str = "pos";
    /// `[1, 1, T, C]` fp16: the additive causal mask over the whole cache.
    pub const MASK: &str = "mask";
    /// `[T, head_dim]` fp16: RoPE rows for the call's positions.
    pub const COS: &str = "cos";
    /// `[T, head_dim]` fp16.
    pub const SIN: &str = "sin";
    /// `[T]` fp16, last chunk only: one-hot, which position's logits to return.
    pub const SEL: &str = "sel";
    /// `[1, D, 1, T]` fp16, every chunk but the last.
    pub const HIDDEN_OUT: &str = "hidden_out";
    /// `[1, V]` fp16, last chunk only.
    pub const LOGITS: &str = "logits";
    /// State `[layers in chunk, kv_heads, C, head_dim]` fp16.
    pub const K_CACHE: &str = "k_cache";
    /// State, the same shape as [`K_CACHE`].
    pub const V_CACHE: &str = "v_cache";
}

/// A `chat.toml`.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct ChatManifest {
    pub id: String,
    /// Where the checkpoint came from, for provenance
    /// (`sidekick-model: id@revision`).
    #[serde(default)]
    pub source: Option<Source>,
    /// Always [`TEXT_GENERATION_TASK`].
    pub task: String,
    /// The program contract; this daemon runs [`KV_STATE_INTERFACE`] and
    /// skips any other.
    pub interface: String,
    /// `cpu_and_ne`, the only value `kv-state-1` accepts; absent means it.
    #[serde(default)]
    pub compute_units: Option<ComputeUnits>,
    /// The tokenizer file, relative to the model directory.
    pub tokenizer: String,
    /// The model's ANE length cap (D48), when lower than the default; the
    /// cache capacity must fit under it.
    #[serde(default)]
    pub ane_max_seq_len: Option<usize>,
    /// The macOS the build was graded on (`"macOS 27.0"`), as
    /// tools/grade_causal_lm.py records it; `/v1/models` lists it.
    #[serde(default)]
    pub graded_os: Option<String>,
    pub decoder: DecoderSection,
    /// `[chat]`: the template and the stop tokens. Required to serve the
    /// model ([`check_serving`](Self::check_serving)); the decoder alone
    /// runs without it.
    #[serde(default)]
    pub chat: Option<ChatSection>,
    /// `[sampling]`: the checkpoint's defaults, for a request that omits
    /// them.
    #[serde(default)]
    pub sampling: SamplingSection,
    /// `[serving]`.
    #[serde(default)]
    pub serving: ServingSection,
}

/// `[chat]`: how a conversation becomes a prompt, and which tokens end a
/// reply.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ChatSection {
    /// The checkpoint's Jinja chat template, relative to the model
    /// directory.
    pub template: String,
    /// Extra variables the template is rendered with (MiniCPM5's
    /// `enable_thinking = false`).
    #[serde(default)]
    pub template_kwargs: toml::Table,
    /// `bos_token` and `eos_token` as the template sees them: the
    /// checkpoint's `tokenizer_config.json` values.
    #[serde(default)]
    pub bos_token: Option<String>,
    #[serde(default)]
    pub eos_token: Option<String>,
    /// Tokens that end a reply (`generation_config.json`'s
    /// `eos_token_id`): `finish_reason: "stop"`.
    pub stop_token_ids: Vec<u32>,
}

/// `[sampling]`: the checkpoint's `generation_config.json` values, used
/// when a request omits them. Absent, a model samples the whole
/// distribution at temperature 1, as transformers does. A key this daemon
/// can't honor (`top_k`, a penalty) is refused, so that a model never
/// samples differently from its checkpoint without saying so.
#[derive(Debug, Clone, Copy, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SamplingSection {
    #[serde(default = "one")]
    pub temperature: f32,
    #[serde(default = "one")]
    pub top_p: f32,
}

fn one() -> f32 {
    1.0
}

impl Default for SamplingSection {
    fn default() -> Self {
        Self { temperature: 1.0, top_p: 1.0 }
    }
}

/// `[serving]`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ServingSection {
    /// The model's request timeout, over the daemon's global one; a
    /// `[models."<id>"] request_timeout_secs` overrides it.
    #[serde(default)]
    pub request_timeout_secs: Option<u64>,
}

/// `[decoder]`: the chunk programs and the shapes they share.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DecoderSection {
    /// The programs, with a `{chunk}` placeholder numbering them from 0
    /// when there is more than one.
    pub artifact: String,
    pub chunks: usize,
    /// T: tokens per call, for the prompt and for generation alike.
    pub tokens_per_call: usize,
    /// C: the KV cache's capacity, the attention width of every call.
    pub cache_capacity: usize,
    /// C − T + 1: the usable context, prompt plus completion.
    pub context_length: usize,
    /// D: the residual stream's width.
    pub hidden_size: usize,
    pub head_dim: usize,
    pub kv_heads: usize,
    /// Layers per chunk: each chunk's state's first dimension.
    pub layers: Vec<usize>,
    /// V: the logits row's width.
    pub vocab_size: usize,
    pub host: DecoderHost,
}

/// `[decoder.host]`: what the host computes with, shipped as files.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DecoderHost {
    /// One tensor, `[vocab, hidden]` fp16, in safetensors.
    pub embedding: String,
    /// `cos` and `sin`, each `[C, head_dim]` fp16, in safetensors.
    pub rope: String,
}

impl ChatManifest {
    /// Parse and [`validate`](Self::validate) a `chat.toml`.
    pub fn read(path: &Path) -> Result<Self, String> {
        let raw = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
        let manifest: Self = toml::from_str(&raw).map_err(|e| e.to_string())?;
        manifest.validate()?;
        Ok(manifest)
    }

    /// The manifest on its own: the task, the interface, the compute units
    /// and the `[decoder]` table's arithmetic.
    pub fn validate(&self) -> Result<(), String> {
        if self.id.is_empty() {
            return Err("`id` is empty".into());
        }
        if self.task != TEXT_GENERATION_TASK {
            return Err(format!(
                "`task` is \"{}\"; a chat.toml serves \"{TEXT_GENERATION_TASK}\"",
                self.task
            ));
        }
        if self.interface != KV_STATE_INTERFACE {
            return Err(format!(
                "`interface` is \"{}\", which this sidekickd doesn't run (it runs \"{KV_STATE_INTERFACE}\")",
                self.interface
            ));
        }
        if let Some(units) = self.compute_units {
            compute_units_problem(units)
                .map_or(Ok(()), |reason| Err(format!("`compute_units`: {reason}")))?;
        }
        if let Some(0) = self.ane_max_seq_len {
            return Err("`ane_max_seq_len` must be at least 1".into());
        }
        if self.tokenizer.is_empty() {
            return Err("`tokenizer` is empty".into());
        }
        self.decoder.validate()?;
        if let Some(chat) = &self.chat {
            self.validate_chat(chat)?;
        }
        let s = &self.sampling;
        if !(s.temperature.is_finite() && s.temperature >= 0.0) {
            return Err(format!("`[sampling] temperature` {} must be 0 or more", s.temperature));
        }
        if !(s.top_p > 0.0 && s.top_p <= 1.0) {
            return Err(format!("`[sampling] top_p` {} must be above 0 and at most 1", s.top_p));
        }
        if self.serving.request_timeout_secs == Some(0) {
            return Err("`[serving] request_timeout_secs` must be at least 1".into());
        }
        Ok(())
    }

    fn validate_chat(&self, chat: &ChatSection) -> Result<(), String> {
        if chat.template.is_empty() {
            return Err("`[chat] template` is empty".into());
        }
        if chat.stop_token_ids.is_empty() {
            return Err("`[chat] stop_token_ids` is empty: nothing would end a reply but its length".into());
        }
        let vocab = self.decoder.vocab_size;
        if let Some(bad) = chat.stop_token_ids.iter().find(|&&t| t as usize >= vocab) {
            return Err(format!("`[chat] stop_token_ids` has {bad}, past the vocabulary ({vocab})"));
        }
        Ok(())
    }

    /// What serving the model needs beyond the decoder: the `[chat]` table
    /// and the tokenizer and template files in `dir`. Expects a
    /// [`validate`](Self::validate)d manifest.
    pub fn check_serving(&self, dir: &Path) -> Result<&ChatSection, String> {
        let chat = self.chat.as_ref().ok_or_else(|| {
            "`[chat]` is missing: serving the model needs its chat template and stop tokens".to_string()
        })?;
        for name in [&self.tokenizer, &chat.template] {
            if !dir.join(name).is_file() {
                return Err(format!("`{name}` is missing"));
            }
        }
        Ok(chat)
    }

    /// Whether this Mac and this daemon's options can serve the model from
    /// `dir`, its model directory: the reason it would be skipped, if any.
    /// Checks the operator's compute units (D38), the Neural Engine, the
    /// cache capacity against the ANE length cap (D48), the program and
    /// host files, symlinked program files (D51) and each program's weights
    /// against the ANE limit (D32). Expects a [`validate`](Self::validate)d
    /// manifest.
    pub fn check(&self, dir: &Path, options: &ScanOptions) -> Result<(), String> {
        if let Some(&units) = options.compute_units.get(&self.id) {
            if let Some(reason) = compute_units_problem(units) {
                return Err(format!(
                    "the daemon config's `[models.\"{}\"] compute_units` chose `{}` (D38): {reason}",
                    self.id,
                    units.name()
                ));
            }
        }
        if options.no_neural_engine {
            return Err(format!(
                "this Mac has no Neural Engine, and a `{KV_STATE_INTERFACE}` decoder runs only there (`cpu_and_ne`; \
                 on the CPU its state crashes)"
            ));
        }
        let d = &self.decoder;
        let cap = ane_seq_cap(self.ane_max_seq_len);
        if d.cache_capacity > cap && !options.ignore_ane_seq_cap {
            return Err(format!(
                "`[decoder] cache_capacity` {} is past the model's ANE length cap of {cap} tokens (D48)",
                d.cache_capacity
            ));
        }
        for name in d
            .chunk_files()
            .iter()
            .chain([&d.host.embedding, &d.host.rope])
        {
            if !dir.join(name).exists() {
                return Err(format!("`{name}` is missing"));
            }
        }
        if let Some(link) = symlinked_program_file(dir, &d.artifact, &[0], d.chunks) {
            return Err(format!(
                "`{link}` is a symbolic link, and Core ML runs a program with a symlinked file off the ANE without \
                 an error (D51); install the files as hard links or copies"
            ));
        }
        if !options.ignore_ane_weight_cap {
            if let Some(reason) = over_ane_weight_limit(dir, &d.artifact, &[0], d.chunks) {
                return Err(format!(
                    "{reason}; Core ML would run it off the ANE without an error, and a `{KV_STATE_INTERFACE}` \
                     decoder runs only there. Convert it in more chunks (D37)"
                ));
            }
        }
        Ok(())
    }
}

/// Why `kv-state-1` can't run on `units`, or `None` for `cpu_and_ne`.
fn compute_units_problem(units: ComputeUnits) -> Option<String> {
    match units {
        ComputeUnits::CpuAndNeuralEngine => None,
        ComputeUnits::CpuOnly => Some(format!(
            "a `{KV_STATE_INTERFACE}` decoder runs only with `cpu_and_ne`: programs with Core ML state crash on \
             the CPU"
        )),
        ComputeUnits::CpuAndGpu | ComputeUnits::All => Some(format!(
            "a `{KV_STATE_INTERFACE}` decoder runs only with `cpu_and_ne`: the GPU with Core ML state is ungraded"
        )),
    }
}

impl DecoderSection {
    /// The chunk programs' names, relative to the model directory, in
    /// chain order.
    pub fn chunk_files(&self) -> Vec<String> {
        artifact_files(&self.artifact, 0, self.chunks)
    }

    /// The usable context the shapes imply: C − T + 1. A call writes T
    /// cache positions, so its first token sits no later than C − T.
    pub fn usable_context(&self) -> usize {
        (self.cache_capacity + 1).saturating_sub(self.tokens_per_call)
    }

    /// The programs' weights on disk, which is what they hold on the ANE
    /// (the embedding table is on the host): the registry's estimate of
    /// the model's ANE share before a load.
    pub fn program_weight_bytes(&self, dir: &Path) -> u64 {
        self.chunk_files()
            .iter()
            .map(|name| artifact_weight_bytes(&dir.join(name)))
            .sum()
    }

    fn validate(&self) -> Result<(), String> {
        let fail = |message: String| Err(format!("`[decoder]`: {message}"));
        if self.artifact.is_empty() {
            return fail("`artifact` is empty".into());
        }
        if self.artifact.contains("{seq}") {
            return fail(
                "`artifact` has a `{seq}` placeholder; a decoder has one static shape, no buckets"
                    .into(),
            );
        }
        if self.chunks == 0 {
            return fail("`chunks` must be at least 1".into());
        }
        if self.chunks > 1 && !self.artifact.contains("{chunk}") {
            return fail(format!(
                "`chunks` is {}, and `artifact` has no `{{chunk}}` placeholder",
                self.chunks
            ));
        }
        if self.layers.len() != self.chunks {
            return fail(format!(
                "`layers` names {} chunks, and `chunks` is {}",
                self.layers.len(),
                self.chunks
            ));
        }
        if self.layers.contains(&0) {
            return fail("every chunk in `layers` must have at least one layer".into());
        }
        for (name, value) in [
            ("tokens_per_call", self.tokens_per_call),
            ("hidden_size", self.hidden_size),
            ("kv_heads", self.kv_heads),
            ("vocab_size", self.vocab_size),
        ] {
            if value == 0 {
                return fail(format!("`{name}` must be at least 1"));
            }
        }
        if self.head_dim == 0 || !self.head_dim.is_multiple_of(2) {
            return fail(format!(
                "`head_dim` {} must be even and positive (RoPE rotates halves)",
                self.head_dim
            ));
        }
        if self.cache_capacity < self.tokens_per_call {
            return fail(format!(
                "`cache_capacity` {} is smaller than `tokens_per_call` {}",
                self.cache_capacity, self.tokens_per_call
            ));
        }
        if self.context_length != self.usable_context() {
            return fail(format!(
                "`context_length` is {}, and cache_capacity − tokens_per_call + 1 is {}",
                self.context_length,
                self.usable_context()
            ));
        }
        if self.host.embedding.is_empty() || self.host.rope.is_empty() {
            return fail("`[decoder.host]` names an empty file".into());
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MANIFEST: &str = r#"
id = "minicpm5-2b"
task = "text-generation"
source = { repo = "openbmb/MiniCPM5-2B", revision = "0123abc" }
interface = "kv-state-1"
compute_units = "cpu_and_ne"
tokenizer = "tokenizer.json"

[decoder]
artifact = "decoder.{chunk}.mlmodelc"
chunks = 4
tokens_per_call = 16
cache_capacity = 2048
context_length = 2033
hidden_size = 2048
head_dim = 128
kv_heads = 2
layers = [11, 11, 10, 10]
vocab_size = 130560

[decoder.host]
embedding = "embedding.safetensors"
rope = "rope.safetensors"

[chat]
template = "chat_template.jinja"
stop_token_ids = [1, 130073]

[sampling]
temperature = 1.0
top_p = 0.95

[serving]
request_timeout_secs = 120

[build]
weights = "palette8"
"#;

    fn parse(text: &str) -> Result<ChatManifest, String> {
        let m: ChatManifest = toml::from_str(text).map_err(|e| e.to_string())?;
        m.validate()?;
        Ok(m)
    }

    fn edit(from: &str, to: &str) -> Result<ChatManifest, String> {
        assert!(MANIFEST.contains(from), "{from}");
        parse(&MANIFEST.replacen(from, to, 1))
    }

    #[test]
    fn the_design_example_parses_and_validates() {
        let m = parse(MANIFEST).unwrap();
        assert_eq!(m.decoder.chunk_files()[3], "decoder.3.mlmodelc");
        assert_eq!(m.decoder.usable_context(), 2033);
        assert_eq!(m.compute_units, Some(ComputeUnits::CpuAndNeuralEngine));
        let chat = m.chat.as_ref().unwrap();
        assert_eq!((chat.template.as_str(), chat.stop_token_ids.as_slice()), ("chat_template.jinja", &[1, 130073][..]));
        assert_eq!((m.sampling.temperature, m.sampling.top_p), (1.0, 0.95));
        assert_eq!(m.serving.request_timeout_secs, Some(120));
        assert_eq!(m.source.as_ref().and_then(|s| s.revision.as_deref()), Some("0123abc"));
        // The converter's provenance tables aren't read.
        assert!(edit("[build]", "[build]\nanything = 1").is_ok());
    }

    #[test]
    fn the_serving_tables_are_checked() {
        // A key the loop doesn't know is refused, not ignored.
        let err = edit("[chat]", "[chat]\nnot_read_here = 1").unwrap_err();
        assert!(err.contains("not_read_here"), "{err}");
        let err = edit("top_p = 0.95", "top_p = 0.95\ntop_k = 20").unwrap_err();
        assert!(err.contains("top_k"), "{err}");
        assert!(edit("top_p = 0.95", "top_p = 0").unwrap_err().contains("top_p"));
        assert!(edit("temperature = 1.0", "temperature = -1.0").unwrap_err().contains("temperature"));
        assert!(edit("request_timeout_secs = 120", "request_timeout_secs = 0").is_err());
        assert!(edit("stop_token_ids = [1, 130073]", "stop_token_ids = []").unwrap_err().contains("stop_token_ids"));
        assert!(edit("stop_token_ids = [1, 130073]", "stop_token_ids = [130560]").unwrap_err().contains("vocabulary"));
        // Absent tables: the decoder still runs; sampling defaults to the
        // whole distribution at temperature 1.
        let bare = MANIFEST.split("[chat]").next().unwrap().to_string();
        let m = parse(&bare).unwrap();
        assert!(m.chat.is_none());
        assert_eq!((m.sampling.temperature, m.sampling.top_p), (1.0, 1.0));
        assert_eq!(m.serving.request_timeout_secs, None);
        let dir = model_dir("serving", 10);
        let err = m.check_serving(&dir).unwrap_err();
        assert!(err.contains("[chat]"), "{err}");
        let m = parse(MANIFEST).unwrap();
        assert!(m.check_serving(&dir).unwrap_err().contains("tokenizer.json"));
        std::fs::write(dir.join("tokenizer.json"), "{}").unwrap();
        assert!(m.check_serving(&dir).unwrap_err().contains("chat_template.jinja"));
        std::fs::write(dir.join("chat_template.jinja"), "").unwrap();
        assert_eq!(m.check_serving(&dir).unwrap().stop_token_ids, [1, 130073]);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn the_decoder_table_refuses_unknown_keys() {
        let err = edit("vocab_size = 130560", "vocab_size = 130560\nvocab = 1").unwrap_err();
        assert!(err.contains("vocab"), "{err}");
        assert!(edit("rope = \"rope.safetensors\"", "rope = \"r\"\ntheta = 1").is_err());
    }

    #[test]
    fn another_task_or_interface_is_refused() {
        let err = edit("task = \"text-generation\"", "task = \"classification\"").unwrap_err();
        assert!(err.contains("text-generation"), "{err}");
        let err = edit("interface = \"kv-state-1\"", "interface = \"kv-state-2\"").unwrap_err();
        assert!(
            err.contains("kv-state-2") && err.contains("kv-state-1"),
            "{err}"
        );
    }

    #[test]
    fn only_cpu_and_ne_is_accepted() {
        assert!(edit("compute_units = \"cpu_and_ne\"\n", "").is_ok());
        let err = edit(
            "compute_units = \"cpu_and_ne\"",
            "compute_units = \"cpu_only\"",
        )
        .unwrap_err();
        assert!(err.contains("crash"), "{err}");
        for units in ["cpu_and_gpu", "all"] {
            let err = edit(
                "compute_units = \"cpu_and_ne\"",
                &format!("compute_units = \"{units}\""),
            )
            .unwrap_err();
            assert!(err.contains("ungraded"), "{err}");
        }
    }

    #[test]
    fn the_shapes_must_add_up() {
        let err = edit("context_length = 2033", "context_length = 2048").unwrap_err();
        assert!(err.contains("2033"), "{err}");
        let err = edit("layers = [11, 11, 10, 10]", "layers = [21, 21]").unwrap_err();
        assert!(err.contains("2 chunks"), "{err}");
        assert!(edit("layers = [11, 11, 10, 10]", "layers = [11, 11, 20, 0]").is_err());
        let err = edit("head_dim = 128", "head_dim = 127").unwrap_err();
        assert!(err.contains("even"), "{err}");
        assert!(edit(
            "artifact = \"decoder.{chunk}.mlmodelc\"",
            "artifact = \"decoder.mlmodelc\""
        )
        .is_err());
        assert!(edit(
            "artifact = \"decoder.{chunk}.mlmodelc\"",
            "artifact = \"d_{seq}.{chunk}.mlmodelc\""
        )
        .is_err());
        let err = edit(
            "tokens_per_call = 16\ncache_capacity = 2048",
            "tokens_per_call = 16\ncache_capacity = 8",
        )
        .unwrap_err();
        assert!(err.contains("smaller"), "{err}");
    }

    #[test]
    fn one_chunk_needs_no_placeholder() {
        let one = MANIFEST
            .replace("decoder.{chunk}.mlmodelc", "decoder.mlmodelc")
            .replace("chunks = 4", "chunks = 1")
            .replace("layers = [11, 11, 10, 10]", "layers = [42]");
        assert_eq!(
            parse(&one).unwrap().decoder.chunk_files(),
            ["decoder.mlmodelc"]
        );
    }

    /// A model directory with the example's files, each program holding
    /// `weights` bytes.
    fn model_dir(tag: &str, weights: u64) -> std::path::PathBuf {
        let dir =
            std::env::temp_dir().join(format!("sk-chat-manifest-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        for i in 0..4 {
            let w = dir.join(format!("decoder.{i}.mlmodelc/weights"));
            std::fs::create_dir_all(&w).unwrap();
            std::fs::File::create(w.join("weight.bin"))
                .unwrap()
                .set_len(weights)
                .unwrap();
        }
        for f in ["embedding.safetensors", "rope.safetensors"] {
            std::fs::write(dir.join(f), b"").unwrap();
        }
        dir
    }

    #[test]
    fn check_passes_a_complete_directory() {
        let m = parse(MANIFEST).unwrap();
        let dir = model_dir("ok", 1000);
        assert_eq!(m.check(&dir, &ScanOptions::default()), Ok(()));
        assert_eq!(m.decoder.program_weight_bytes(&dir), 4000);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn check_skips_a_mac_without_a_neural_engine() {
        let m = parse(MANIFEST).unwrap();
        let dir = model_dir("no-ane", 1000);
        let options = ScanOptions {
            no_neural_engine: true,
            ..Default::default()
        };
        let err = m.check(&dir, &options).unwrap_err();
        assert!(err.contains("no Neural Engine"), "{err}");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn check_refuses_an_operator_override_off_the_ane() {
        let m = parse(MANIFEST).unwrap();
        let dir = model_dir("override", 1000);
        let mut options = ScanOptions::default();
        options
            .compute_units
            .insert("minicpm5-2b".into(), ComputeUnits::CpuAndGpu);
        let err = m.check(&dir, &options).unwrap_err();
        assert!(
            err.contains("[models.\"minicpm5-2b\"]") && err.contains("cpu_and_gpu"),
            "{err}"
        );
        options
            .compute_units
            .insert("minicpm5-2b".into(), ComputeUnits::CpuAndNeuralEngine);
        assert_eq!(m.check(&dir, &options), Ok(()));
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn check_applies_the_ane_length_cap() {
        let m = edit(
            "cache_capacity = 2048\ncontext_length = 2033",
            "cache_capacity = 4096\ncontext_length = 4081",
        )
        .unwrap();
        let dir = model_dir("cap", 1000);
        let err = m.check(&dir, &ScanOptions::default()).unwrap_err();
        assert!(err.contains("2048"), "{err}");
        let options = ScanOptions {
            ignore_ane_seq_cap: true,
            ..Default::default()
        };
        assert_eq!(m.check(&dir, &options), Ok(()));
        let raised = edit(
            "tokenizer = \"tokenizer.json\"",
            "tokenizer = \"tokenizer.json\"\nane_max_seq_len = 4096",
        )
        .unwrap();
        assert_eq!(raised.check(&dir, &ScanOptions::default()), Ok(()));
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn check_applies_the_ane_weight_limit_per_chunk() {
        let m = parse(MANIFEST).unwrap();
        let dir = model_dir("weights", 1000);
        std::fs::File::create(dir.join("decoder.2.mlmodelc/weights/weight.bin"))
            .unwrap()
            .set_len(crate::MAX_ANE_PROGRAM_WEIGHT_BYTES + 1)
            .unwrap();
        let err = m.check(&dir, &ScanOptions::default()).unwrap_err();
        assert!(
            err.contains("decoder.2.mlmodelc") && err.contains("1 GiB"),
            "{err}"
        );
        let options = ScanOptions {
            ignore_ane_weight_cap: true,
            ..Default::default()
        };
        assert_eq!(m.check(&dir, &options), Ok(()));
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn check_reports_missing_files_and_symlinks() {
        let m = parse(MANIFEST).unwrap();
        let dir = model_dir("files", 1000);
        std::fs::remove_file(dir.join("rope.safetensors")).unwrap();
        let err = m.check(&dir, &ScanOptions::default()).unwrap_err();
        assert!(err.contains("rope.safetensors"), "{err}");
        std::fs::write(dir.join("rope.safetensors"), b"").unwrap();
        let weight = dir.join("decoder.1.mlmodelc/weights/weight.bin");
        std::fs::rename(&weight, dir.join("elsewhere.bin")).unwrap();
        std::os::unix::fs::symlink(dir.join("elsewhere.bin"), &weight).unwrap();
        let err = m.check(&dir, &ScanOptions::default()).unwrap_err();
        assert!(
            err.contains("decoder.1.mlmodelc/weights/weight.bin") && err.contains("D51"),
            "{err}"
        );
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn the_registry_scans_chat_models() {
        use crate::ModelRegistry;
        let root = std::env::temp_dir().join(format!("sk-chat-registry-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let install = |name: &str, manifest: &str| {
            let dir = model_dir(&format!("reg-{name}"), 10);
            std::fs::write(dir.join("tokenizer.json"), "{}").unwrap();
            std::fs::write(dir.join("chat_template.jinja"), "").unwrap();
            std::fs::write(dir.join(CHAT_MANIFEST), manifest).unwrap();
            std::fs::create_dir_all(&root).unwrap();
            std::fs::rename(&dir, root.join(name)).unwrap();
        };
        install("good", MANIFEST);
        install("bare", &MANIFEST.split("[chat]").next().unwrap().replace("minicpm5-2b", "bare"));
        install("afm", &MANIFEST.replace("\"minicpm5-2b\"", "\"apple-fm\""));
        install("twin", MANIFEST);
        let options = ScanOptions {
            compute_units: [("minicpm5-2b".to_string(), ComputeUnits::CpuAndNeuralEngine)].into(),
            ..Default::default()
        };
        let reg = ModelRegistry::scan_with(&root, &options).unwrap();
        assert_eq!(reg.chat_ids().collect::<Vec<_>>(), ["minicpm5-2b"]);
        assert_eq!(reg.chat("minicpm5-2b").unwrap().dir, root.join("good"));
        assert!(reg.unmatched_overrides().is_empty(), "the override named the chat model");
        let reasons: Vec<String> = reg.skipped().iter().map(|s| format!("{}: {}", s.path.display(), s.reason)).collect();
        assert_eq!(reasons.len(), 3, "{reasons:?}");
        assert!(reasons.iter().any(|r| r.starts_with("afm/chat.toml") && r.contains("Foundation Models")), "{reasons:?}");
        assert!(reasons.iter().any(|r| r.starts_with("bare/chat.toml") && r.contains("[chat]")), "{reasons:?}");
        assert!(reasons.iter().any(|r| r.starts_with("twin/chat.toml") && r.contains("duplicate")), "{reasons:?}");
        let mut reg = reg;
        reg.skip_chats("no Core ML in this build");
        assert_eq!(reg.chat_ids().count(), 0);
        assert!(reg.skipped().iter().any(|s| s.reason == "no Core ML in this build"));
        std::fs::remove_dir_all(&root).unwrap();
    }
}
