//! A stateful decoder chain: an open causal LM's `kv-state-1` chunk
//! programs on the Neural Engine, each owning its layers' KV cache as Core
//! ML state (D53; docs/design/open-llm.md, "The program contract" and "The
//! runtime").
//!
//! [`KvDecoder`] runs one T-token call at a time through the chain and
//! returns one logits row. It does the host's side of the contract too: the
//! embedding lookup from the mapped table, the RoPE rows, the causal mask
//! and the one-hot `sel`. Everything above a call (the prompt's layout
//! across requests, sampling, stop sequences, the request queue) is the
//! loop's.

#![allow(unsafe_code)]
// `dataPointer` is deprecated in favor of the block-based accessors; the
// inputs are this decoder's own arrays, written on its own thread between
// calls, and outputs are read before the call's autorelease pool drains.
#![allow(deprecated)]

use crate::decoder_health::{DecoderHealth, DecoderMonitor, Watchdog, DEFAULT_CALL_TIMEOUT};
use crate::kv_contract::{
    check_program, DecoderShape, Dtype, ProgramDecl, ProgramNames, TensorDecl,
};
use crate::model::{open, read_input_shapes};
use crate::{shape_verdict, ShapeVerdict};
use half::f16;
use objc2::rc::{autoreleasepool, Retained};
use objc2::runtime::{AnyObject, ProtocolObject};
use objc2::{available, AnyThread, Message};
use objc2_core_ml::{
    MLDictionaryFeatureProvider, MLFeatureDescription, MLFeatureProvider, MLModel, MLMultiArray,
    MLMultiArrayDataType, MLPredictionOptions, MLState,
};
use objc2_foundation::{NSDictionary, NSNumber, NSString};
use sidekick_core::chat_manifest::{kv_names, ChatManifest, KV_MASK_HIDDEN, KV_STATE_INTERFACE};
use sidekick_core::{ComputeUnits, Error, Result, ScanOptions};
use std::mem::ManuallyDrop;
use std::path::Path;
use std::time::{Duration, Instant};

/// In the error of a load refused because its programs weren't confirmed
/// on the Neural Engine ([`AneCheck`]); see [`is_placement_refusal`].
pub const PLACEMENT_REFUSAL: &str = "its programs aren't confirmed on the Neural Engine";

/// Whether `e` refused a decoder because its programs weren't confirmed on
/// the Neural Engine: a verdict on the build, on this Mac, that a strict
/// load in a process with nothing else on the ANE makes conclusively.
pub fn is_placement_refusal(e: &Error) -> bool {
    e.to_string().contains(PLACEMENT_REFUSAL)
}

/// The token whose embedding pads a call's unused slots. Their keys and
/// values land at cache positions no real query attends to, and the next
/// call's real tokens overwrite them ("One call, T tokens").
pub const PAD_TOKEN: u32 = 0;

/// How [`KvDecoder::load_with`] loads and runs a decoder.
#[derive(Debug, Clone)]
pub struct DecoderOptions {
    /// One prediction's limit before the decoder is marked failed.
    pub call_timeout: Duration,
    /// Have Core ML write each chunk's output into arrays allocated once at
    /// load (`MLPredictionOptions.outputBackings`): the residual stream
    /// passes from chunk to chunk in one buffer, and no call allocates an
    /// output or a feature provider. Off, each call's outputs are Core
    /// ML's, released with the call's autorelease pool.
    pub output_backings: bool,
    /// The residual stream's input name: the contract's, unless a build
    /// predates it.
    pub names: ProgramNames,
    /// The registry's options for [`ChatManifest::check`], which the load
    /// applies again: the ANE weight limit (D32), the length cap (D48) and
    /// the Neural Engine itself, which is detected whatever
    /// `no_neural_engine` says.
    pub scan: ScanOptions,
    /// How the load confirms that Core ML put the programs on the ANE.
    pub ane_check: AneCheck,
}

/// How [`KvDecoder::load_with`] confirms that Core ML placed the decoder on
/// the Neural Engine. Core ML runs a program it judges too small for the
/// ANE, or one whose ANE compile failed, on the CPU without an error, and a
/// `kv-state-1` decoder's state crashes there. Nothing reports where a
/// loaded program runs short of a compute plan, which costs another full
/// compile and another copy of the weights, so the load reads the process's
/// ANE footprint (`ri_neural_footprint`) before it and after its first call.
///
/// The footprint is the whole process's. A load can make the system evict
/// another of the process's idle ANE programs, so the total can grow by
/// less than the decoder holds, or not at all, while the decoder is on the
/// ANE. Only in a process that held nothing else on the ANE is the reading
/// conclusive either way: load the decoder there first to confirm it (as
/// D52's warm-up child does), and with [`AneCheck::Lenient`] where other
/// models may already be loaded.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AneCheck {
    /// Refuse the decoder unless the footprint grew by at least half the
    /// programs' weights on disk (the 2B build's ANE share was 1.01 times
    /// its weights; a palettized test program's about 1.4 times). In a
    /// process holding other ANE programs this can refuse a decoder that is
    /// on the ANE, if its load evicted theirs.
    #[default]
    Strict,
    /// Refuse only when the process holds no ANE program at all after the
    /// call, which is conclusive; when the footprint grew by less than
    /// [`Strict`](Self::Strict) asks, load anyway and log a warning.
    Lenient,
}

impl Default for DecoderOptions {
    fn default() -> Self {
        Self {
            call_timeout: DEFAULT_CALL_TIMEOUT,
            output_backings: true,
            names: ProgramNames::default(),
            scan: ScanOptions::default(),
            ane_check: AneCheck::default(),
        }
    }
}

/// One chunk program, its KV state and its prediction options.
struct Chunk {
    model: Retained<MLModel>,
    state: Retained<MLState>,
    options: Retained<MLPredictionOptions>,
    /// Built once when every input the chunk takes is an array this decoder
    /// owns: always for the first chunk, and for the rest when the previous
    /// chunk's output is a backing.
    provider: Option<Retained<MLDictionaryFeatureProvider>>,
}

/// The inputs every call rewrites in place, allocated once.
struct Inputs {
    hidden: Retained<MLMultiArray>,
    pos: Retained<MLMultiArray>,
    mask: Retained<MLMultiArray>,
    cos: Retained<MLMultiArray>,
    sin: Retained<MLMultiArray>,
    sel: Retained<MLMultiArray>,
}

/// A loaded `kv-state-1` decoder.
///
/// One thread drives it, one call at a time (`&mut self`), as the design's
/// decoder thread does; other threads see its health through
/// [`monitor`](Self::monitor). The KV state carries over from call to call
/// and from request to request: a request writes its prompt from position
/// 0, and the mask hides every key past a query's own position, so an
/// earlier request's keys never reach it. The state is replaced after a
/// failed call or a non-finite logit, before the error returns.
pub struct KvDecoder {
    id: String,
    shape: DecoderShape,
    hidden_in: String,
    chunks: Vec<Chunk>,
    /// Dropped by hand, in an autorelease pool (`Drop`).
    inputs: ManuallyDrop<Inputs>,
    /// Each chunk's output backing, when they're used.
    backings: Vec<Retained<MLMultiArray>>,
    embedding: Embedding,
    /// `[C, head_dim]` each, fp16 bits.
    cos: Vec<u16>,
    sin: Vec<u16>,
    /// The last call's logits row.
    logits: Vec<f32>,
    /// What the load added to the process's ANE footprint, when measured.
    ane_bytes: Option<u64>,
    monitor: DecoderMonitor,
    watchdog: Watchdog,
}

// SAFETY: every Objective-C object here is owned by the decoder and used
// only through `&mut self`, so from one thread at a time; the monitor and
// the watchdog's shared state are `Send + Sync`. Each call runs in its own
// autorelease pool, so nothing the decoder makes is left in a thread's pool
// when it moves.
unsafe impl Send for KvDecoder {}

impl KvDecoder {
    /// Load the decoder in `dir` with the default options.
    pub fn load(dir: &Path, manifest: &ChatManifest) -> Result<Self> {
        Self::load_with(dir, manifest, DecoderOptions::default())
    }

    /// Load every chunk program with `cpu_and_ne`, check each one's declared
    /// inputs, outputs and state shapes against the manifest's `[decoder]`
    /// table, give each a fresh KV state, map the embedding table and read
    /// the RoPE tables. Refuses the model, with the reason, before macOS 15
    /// (Core ML state), on a Mac without a Neural Engine, and wherever
    /// [`ChatManifest::check`] would skip it.
    pub fn load_with(dir: &Path, manifest: &ChatManifest, options: DecoderOptions) -> Result<Self> {
        // Core ML autoreleases objects while it loads and makes states, the
        // models among them; drained here, dropping the decoder releases its
        // programs (and their share of the ANE) rather than at thread exit.
        autoreleasepool(|_| Self::load_in_pool(dir, manifest, options))
    }

    fn load_in_pool(dir: &Path, manifest: &ChatManifest, options: DecoderOptions) -> Result<Self> {
        let id = &manifest.id;
        let refuse =
            |reason: String| Error::Inference(format!("refusing decoder `{id}`: {reason}"));
        if !available!(macos = 15.0) {
            return Err(refuse("Core ML state needs macOS 15 or later".into()));
        }
        manifest.validate().map_err(refuse)?;
        let mut scan = options.scan.clone();
        scan.no_neural_engine |= !crate::has_neural_engine();
        manifest.check(dir, &scan).map_err(refuse)?;
        let d = &manifest.decoder;
        let shape = DecoderShape::from_section(d);

        let mut chunks = Vec::with_capacity(d.chunks);
        let neural_before = settled_neural_footprint();
        for (i, file) in d.chunk_files().into_iter().enumerate() {
            let model = open(&dir.join(&file), ComputeUnits::CpuAndNeuralEngine)?;
            if let ShapeVerdict::Warn(reason) | ShapeVerdict::Refuse(reason) =
                shape_verdict(&read_input_shapes(&model), true)
            {
                return Err(refuse(format!("`{file}`: {reason}")));
            }
            check_program(&shape, i, &options.names, &declared(&model))
                .map_err(|e| refuse(format!("`{file}`: {e}")))?;
            let state = unsafe { model.newState() };
            let options = unsafe { MLPredictionOptions::new() };
            chunks.push(Chunk {
                model,
                state,
                options,
                provider: None,
            });
        }

        let embedding = Embedding::open(&dir.join(&d.host.embedding), d.vocab_size, d.hidden_size)?;
        let (cos, sin) = read_rope(&dir.join(&d.host.rope), d.cache_capacity, d.head_dim)?;
        let (t, c, hd) = (d.tokens_per_call, d.cache_capacity, d.head_dim);
        let inputs = Inputs {
            hidden: array(&[1, d.hidden_size, 1, t], MLMultiArrayDataType::Float16)?,
            pos: array(&[1], MLMultiArrayDataType::Int32)?,
            mask: array(&[1, 1, t, c], MLMultiArrayDataType::Float16)?,
            cos: array(&[t, hd], MLMultiArrayDataType::Float16)?,
            sin: array(&[t, hd], MLMultiArrayDataType::Float16)?,
            sel: array(&[t], MLMultiArrayDataType::Float16)?,
        };

        let mut backings = Vec::new();
        if options.output_backings {
            for (i, chunk) in chunks.iter().enumerate() {
                let (name, dims) = match i + 1 == d.chunks {
                    true => (kv_names::LOGITS, vec![1, d.vocab_size]),
                    false => (kv_names::HIDDEN_OUT, vec![1, d.hidden_size, 1, t]),
                };
                let backing = array(&dims, MLMultiArrayDataType::Float16)?;
                let dict = NSDictionary::from_retained_objects(
                    &[&*NSString::from_str(name)],
                    &[object(&backing)],
                );
                unsafe { chunk.options.setOutputBackings(&dict) };
                backings.push(backing);
            }
        }

        let monitor = DecoderMonitor::new();
        let watchdog = Watchdog::start(options.call_timeout, monitor.clone())
            .map_err(|e| refuse(format!("can't start its watchdog thread: {e}")))?;
        let mut decoder = Self {
            id: id.clone(),
            shape,
            hidden_in: options.names.hidden_in,
            chunks,
            inputs: ManuallyDrop::new(inputs),
            backings,
            embedding,
            cos,
            sin,
            logits: vec![0.0; d.vocab_size],
            ane_bytes: None,
            monitor,
            watchdog,
        };
        for i in 0..decoder.chunks.len() {
            let hidden = match i {
                0 => Some(decoder.inputs.hidden.clone()),
                _ => decoder.backings.get(i - 1).cloned(),
            };
            if let Some(hidden) = hidden {
                let provider = decoder.provider(i, &hidden)?;
                decoder.chunks[i].provider = Some(provider);
            }
        }
        // One call, as a request makes it, so every program is resident on
        // the ANE before the load returns, then the placement check
        // ([`AneCheck`]). If Core ML put the decoder on the CPU, this one
        // call runs there: the test build that Core ML places on the CPU
        // has never crashed in one call, nor in thousands.
        decoder
            .step(&[PAD_TOKEN], 0)
            .map_err(|e| refuse(format!("the first call failed: {e}")))?;
        let expected = d.program_weight_bytes(dir);
        match placement(neural_before, expected) {
            Placement::Confirmed(bytes) => decoder.ane_bytes = Some(bytes),
            Placement::Unreported => {
                tracing::warn!(model = %id, "this kernel doesn't report the Neural Engine footprint, so the \
                    decoder's placement on the ANE can't be confirmed");
            }
            Placement::Off => {
                return Err(refuse(format!(
                    "{PLACEMENT_REFUSAL}: Core ML didn't place them there (this process holds no ANE program after \
                     a call), and a `{KV_STATE_INTERFACE}` decoder runs only there"
                )))
            }
            Placement::Unconfirmed { grew } => {
                let reason = format!(
                    "the process's Neural Engine footprint grew by {grew} bytes across the load, under half the \
                     programs' {expected} bytes of weights: Core ML may have placed a chunk off the ANE, where a \
                     `{KV_STATE_INTERFACE}` decoder's state crashes, or the load evicted another of the process's \
                     ANE programs"
                );
                match options.ane_check {
                    AneCheck::Strict => {
                        return Err(refuse(format!(
                            "{PLACEMENT_REFUSAL}: {reason}. Load it where nothing else is on the ANE to confirm it \
                             (on the Neural Engine or not), then with `AneCheck::Lenient`"
                        )))
                    }
                    AneCheck::Lenient => tracing::warn!(model = %id, "{reason}; loading it anyway"),
                }
            }
        }
        Ok(decoder)
    }

    /// The model's id, from its manifest.
    pub fn id(&self) -> &str {
        &self.id
    }

    /// The shapes the programs share: T, C, the usable context, D, V.
    pub fn shape(&self) -> &DecoderShape {
        &self.shape
    }

    /// How much loading the decoder and making its first call added to this
    /// process's Neural Engine footprint: its share of the ANE's budget
    /// (#187 measured 2,185 MiB for MiniCPM5-2B's build). `None` when the
    /// load couldn't confirm it ([`AneCheck`]): the kernel doesn't report
    /// the footprint, or it grew by less than half the programs' weights.
    pub fn ane_bytes(&self) -> Option<u64> {
        self.ane_bytes
    }

    /// Whether each call writes into output backings: as asked at load,
    /// until Core ML once returns a chunk's output somewhere else.
    pub fn output_backings(&self) -> bool {
        !self.backings.is_empty()
    }

    /// The decoder's shared health record, for other threads.
    pub fn monitor(&self) -> DecoderMonitor {
        self.monitor.clone()
    }

    /// A snapshot of [`monitor`](Self::monitor).
    pub fn health(&self) -> DecoderHealth {
        self.monitor.snapshot()
    }

    /// One call: `tokens` (1 to T of them) at cache positions `pos`,
    /// `pos + 1`, …, the rest of the call padded with [`PAD_TOKEN`].
    /// Returns the logits of the last token, valid until the next call.
    /// Generation is one token per call; the loop may fill the padding
    /// with tokens it already knows (a grammar's forced tokens).
    pub fn step(&mut self, tokens: &[u32], pos: usize) -> Result<&[f32]> {
        self.step_select(tokens, pos, tokens.len().saturating_sub(1))
    }

    /// [`step`](Self::step), returning the logits of `tokens[sel]`.
    pub fn step_select(&mut self, tokens: &[u32], pos: usize, sel: usize) -> Result<&[f32]> {
        let (t, d, v) = (
            self.shape.tokens_per_call,
            self.shape.hidden_size,
            self.shape.vocab_size,
        );
        if tokens.is_empty() || tokens.len() > t {
            return Err(Error::InvalidRequest(format!(
                "a call takes 1 to {t} tokens, not {}",
                tokens.len()
            )));
        }
        if sel >= tokens.len() {
            return Err(Error::InvalidRequest(format!(
                "`sel` {sel} is past the call's {} tokens",
                tokens.len()
            )));
        }
        if let Some(&bad) = tokens.iter().find(|&&tok| tok as usize >= v) {
            return Err(Error::InvalidRequest(format!(
                "token id {bad} is past the vocabulary ({v})"
            )));
        }
        self.check_position(pos, tokens.len())?;
        let hidden = self.inputs.hidden.clone();
        // SAFETY: an fp16 [1, D, 1, T] array this decoder allocated, contiguous.
        let dst = unsafe { slice_mut::<u16>(&hidden, d * t) };
        for s in 0..t {
            let row = self
                .embedding
                .row(tokens.get(s).copied().unwrap_or(PAD_TOKEN) as usize);
            for (j, pair) in row.as_chunks::<2>().0.iter().enumerate() {
                dst[j * t + s] = u16::from_le_bytes(*pair);
            }
        }
        self.call(pos, sel)
    }

    /// The prompt: `tokens` in T-token calls at positions `start`,
    /// `start + T`, …, the last call padded. Returns the last token's
    /// logits; the earlier calls' rows are computed and discarded.
    pub fn prefill(&mut self, tokens: &[u32], start: usize) -> Result<&[f32]> {
        if tokens.is_empty() {
            return Err(Error::InvalidRequest("an empty prompt".into()));
        }
        self.check_position(start, tokens.len())?;
        let t = self.shape.tokens_per_call;
        let mut done = 0;
        while done + t < tokens.len() {
            self.step(&tokens[done..done + t], start + done)?;
            done += t;
        }
        self.step(&tokens[done..], start + done)
    }

    /// One call from the residual stream directly: `hidden` is the call's T
    /// rows of D fp16 values (as bits), token-major, as an embedding table
    /// holds them. For graders that bring their own inputs.
    pub fn step_hidden(&mut self, hidden: &[u16], pos: usize, sel: usize) -> Result<&[f32]> {
        let (t, d) = (self.shape.tokens_per_call, self.shape.hidden_size);
        if hidden.len() != t * d {
            return Err(Error::InvalidRequest(format!(
                "{} values for a [{t}, {d}] call",
                hidden.len()
            )));
        }
        if sel >= t {
            return Err(Error::InvalidRequest(format!(
                "`sel` {sel} is past the call's {t} tokens"
            )));
        }
        self.check_position(pos, 1)?;
        let array = self.inputs.hidden.clone();
        // SAFETY: as in `step_select`.
        let dst = unsafe { slice_mut::<u16>(&array, d * t) };
        for s in 0..t {
            for j in 0..d {
                dst[j * t + s] = hidden[s * d + j];
            }
        }
        self.call(pos, sel)
    }

    /// Replace every chunk's KV state with a fresh one (`newState`). A call
    /// that fails, or returns a non-finite logit, does this itself.
    pub fn reset_state(&mut self) {
        autoreleasepool(|_| {
            for chunk in &mut self.chunks {
                chunk.state = unsafe { chunk.model.newState() };
            }
        });
        self.monitor.record_state_reset();
    }

    /// A call writes T cache positions from `pos`, so `pos` can be at most
    /// C − T; `n` tokens from `pos` must fit the usable context.
    fn check_position(&self, pos: usize, n: usize) -> Result<()> {
        let s = &self.shape;
        if pos + s.tokens_per_call > s.cache_capacity || pos + n > s.context_length {
            return Err(Error::ContextOverflow {
                limit: s.context_length,
                actual: Some(pos + n),
            });
        }
        Ok(())
    }

    /// Run the chain on the hidden rows already written, at `pos`, and
    /// read the logits of position `sel`.
    fn call(&mut self, pos: usize, sel: usize) -> Result<&[f32]> {
        if let Some(reason) = self.monitor.failed() {
            return Err(self.failed(&reason));
        }
        self.write_position_inputs(pos, sel);
        let started = Instant::now();
        // Core ML autoreleases each prediction's outputs and its internal
        // objects. A Rust thread has no pool that drains between calls, so
        // without this one they pile up for the thread's whole life (#173's
        // loop reached a 19.7 GB footprint).
        let result = autoreleasepool(|_| self.run_chain(pos));
        let elapsed = started.elapsed();
        if let Some(reason) = self.monitor.failed() {
            return Err(self.failed(&reason));
        }
        if let Err(e) = result {
            self.reset_state();
            return Err(e);
        }
        self.monitor.record_call(elapsed);
        if let Some(i) = self.logits.iter().position(|x| !x.is_finite()) {
            self.reset_state();
            return Err(Error::Inference(format!(
                "decoder `{}`: a non-finite logit ({} at token id {i}) at position {pos}; its KV state was replaced",
                self.id, self.logits[i]
            )));
        }
        Ok(&self.logits)
    }

    fn failed(&self, reason: &str) -> Error {
        Error::Inference(format!("decoder `{}` has failed: {reason}", self.id))
    }

    fn write_position_inputs(&mut self, pos: usize, sel: usize) {
        let s = &self.shape;
        let (t, c, hd) = (s.tokens_per_call, s.cache_capacity, s.head_dim);
        let hidden_bits = f16::from_f32(KV_MASK_HIDDEN).to_bits();
        // SAFETY: each array was allocated by this decoder with this shape
        // and type, contiguous; no prediction runs while they're written.
        unsafe {
            slice_mut::<i32>(&self.inputs.pos, 1)[0] = pos as i32;
            let mask = slice_mut::<u16>(&self.inputs.mask, t * c);
            for (q, row) in mask.chunks_exact_mut(c).enumerate() {
                let visible = (pos + q + 1).min(c);
                row[..visible].fill(0);
                row[visible..].fill(hidden_bits);
            }
            let rows = pos * hd..(pos + t) * hd;
            slice_mut::<u16>(&self.inputs.cos, t * hd).copy_from_slice(&self.cos[rows.clone()]);
            slice_mut::<u16>(&self.inputs.sin, t * hd).copy_from_slice(&self.sin[rows]);
            let one_hot = slice_mut::<u16>(&self.inputs.sel, t);
            one_hot.fill(0);
            one_hot[sel] = f16::ONE.to_bits();
        }
    }

    fn run_chain(&mut self, pos: usize) -> Result<()> {
        let n = self.chunks.len();
        let mut carried: Option<Retained<MLMultiArray>> = None;
        for i in 0..n {
            let built;
            let provider = match (&self.chunks[i].provider, carried.take()) {
                (Some(p), _) => p,
                (None, Some(hidden)) => {
                    built = self.provider(i, &hidden)?;
                    &built
                }
                (None, None) => unreachable!("only a chunk after the first lacks a built provider"),
            };
            let chunk = &self.chunks[i];
            self.watchdog.arm(format!(
                "chunk {i} of decoder `{}` at position {pos}",
                self.id
            ));
            let out = unsafe {
                chunk.model.predictionFromFeatures_usingState_options_error(
                    ProtocolObject::from_ref(&**provider),
                    &chunk.state,
                    &chunk.options,
                )
            };
            self.watchdog.disarm();
            // A call that returns after the watchdog gave up on it: the
            // decoder has failed, and the rest of the chain doesn't run.
            if let Some(reason) = self.monitor.failed() {
                return Err(self.failed(&reason));
            }
            let out = out.map_err(|e| {
                Error::Inference(format!(
                    "decoder `{}`, chunk {i} at position {pos}: {}",
                    self.id,
                    e.localizedDescription()
                ))
            })?;
            if i + 1 == n {
                let logits = output(&out, kv_names::LOGITS)?;
                read_f16_row(&logits, &mut self.logits)?;
                continue;
            }
            let hidden = output(&out, kv_names::HIDDEN_OUT)?;
            match self.backings.get(i) {
                Some(backing) if same_array(&hidden, backing) => {}
                // Core ML "may not use the specified backing object"
                // (MLPredictionOptions.h). The next chunk's built provider
                // would then read a stale buffer, and the logits would be
                // wrong but finite, so carry what it returned instead.
                Some(_) => {
                    tracing::warn!(
                        model = %self.id,
                        "Core ML didn't write chunk {i}'s `{}` into its output backing; output backings are off \
                         for this decoder",
                        kv_names::HIDDEN_OUT
                    );
                    self.disable_backings();
                    carried = Some(hidden);
                }
                None => carried = Some(hidden),
            }
        }
        Ok(())
    }

    /// Stop using output backings: Core ML's own outputs from now on, and
    /// providers built per call for every chunk after the first.
    fn disable_backings(&mut self) {
        let none = NSDictionary::<NSString, AnyObject>::new();
        for (i, chunk) in self.chunks.iter_mut().enumerate() {
            unsafe { chunk.options.setOutputBackings(&none) };
            if i > 0 {
                chunk.provider = None;
            }
        }
        self.backings.clear();
    }

    /// Chunk `i`'s inputs, with `hidden` as its residual stream.
    fn provider(
        &self,
        i: usize,
        hidden: &MLMultiArray,
    ) -> Result<Retained<MLDictionaryFeatureProvider>> {
        let x = &self.inputs;
        let mut features: Vec<(&str, &MLMultiArray)> = vec![
            (&self.hidden_in, hidden),
            (kv_names::POS, &x.pos),
            (kv_names::MASK, &x.mask),
            (kv_names::COS, &x.cos),
            (kv_names::SIN, &x.sin),
        ];
        if i + 1 == self.chunks.len() {
            features.push((kv_names::SEL, &x.sel));
        }
        let keys: Vec<Retained<NSString>> = features
            .iter()
            .map(|(k, _)| NSString::from_str(k))
            .collect();
        let key_refs: Vec<&NSString> = keys.iter().map(|k| &**k).collect();
        let values: Vec<Retained<AnyObject>> = features
            .iter()
            .map(|(_, a)| {
                let value = unsafe { objc2_core_ml::MLFeatureValue::featureValueWithMultiArray(a) };
                Retained::into_super(Retained::into_super(value))
            })
            .collect();
        let dict = NSDictionary::from_retained_objects(&key_refs, &values);
        unsafe {
            MLDictionaryFeatureProvider::initWithDictionary_error(
                MLDictionaryFeatureProvider::alloc(),
                &dict,
            )
        }
        .map_err(|e| Error::Inference(format!("decoder `{}`: feature provider: {e}", self.id)))
    }
}

impl Drop for KvDecoder {
    fn drop(&mut self) {
        // Releasing a model or a state can autorelease Core ML's objects;
        // drained here rather than at the thread's exit, the programs leave
        // the ANE when the decoder goes.
        autoreleasepool(|_| {
            self.chunks.clear();
            self.backings.clear();
            // SAFETY: dropped once, here, and never used again.
            unsafe { ManuallyDrop::drop(&mut self.inputs) };
        });
    }
}

/// Whether Core ML returned `backing` itself, or another array over its
/// buffer, as a chunk's output.
fn same_array(returned: &MLMultiArray, backing: &MLMultiArray) -> bool {
    std::ptr::eq(returned, backing)
        || unsafe { returned.dataPointer().as_ptr() == backing.dataPointer().as_ptr() }
}

/// Where the load's footprint reading puts the decoder ([`AneCheck`]).
#[derive(Debug, PartialEq, Eq)]
enum Placement {
    /// The footprint grew by at least half the programs' weights: by this much.
    Confirmed(u64),
    /// The process holds no ANE program after the call.
    Off,
    /// It grew by less (a shrink reads as 0): the decoder may be off the
    /// ANE, or the load evicted another of the process's programs.
    Unconfirmed { grew: u64 },
    /// The kernel doesn't report the footprint.
    Unreported,
}

/// Read the process's ANE footprint against `before` and the programs'
/// weights. The kernel's accounting can trail a load, so a short reading is
/// retried for up to a second before it counts.
fn placement(before: Option<u64>, expected: u64) -> Placement {
    let Some(before) = before else {
        return Placement::Unreported;
    };
    let mut verdict = Placement::Unreported;
    for attempt in 0..10 {
        if attempt > 0 {
            std::thread::sleep(Duration::from_millis(100));
        }
        let Some(after) = neural_footprint() else {
            return Placement::Unreported;
        };
        verdict = judge_placement(before, after, expected);
        if matches!(verdict, Placement::Confirmed(_)) {
            break;
        }
    }
    verdict
}

fn judge_placement(before: u64, after: u64, expected: u64) -> Placement {
    let grew = after.saturating_sub(before);
    if after == 0 {
        Placement::Off
    } else if grew > 0 && grew >= expected / 2 {
        Placement::Confirmed(grew)
    } else {
        Placement::Unconfirmed { grew }
    }
}

/// This process's Neural Engine footprint, where the kernel reports it.
fn neural_footprint() -> Option<u64> {
    crate::process_memory().and_then(|m| m.neural_footprint)
}

/// The footprint once it holds still: a model dropped just before (a
/// decoder reloaded, say) leaves the ANE some tens of milliseconds later,
/// and a reading taken before that would count it against this load.
/// Waits only while the footprint is non-zero and changing, for at most a
/// second.
fn settled_neural_footprint() -> Option<u64> {
    let mut last = neural_footprint()?;
    for _ in 0..20 {
        if last == 0 {
            break;
        }
        std::thread::sleep(Duration::from_millis(50));
        let now = neural_footprint()?;
        if now == last {
            break;
        }
        last = now;
    }
    Some(last)
}

/// A new contiguous multi-array.
fn array(shape: &[usize], dtype: MLMultiArrayDataType) -> Result<Retained<MLMultiArray>> {
    let dims: Vec<Retained<NSNumber>> = shape.iter().map(|&d| NSNumber::new_usize(d)).collect();
    let dims = objc2_foundation::NSArray::from_retained_slice(&dims);
    let array =
        unsafe { MLMultiArray::initWithShape_dataType_error(MLMultiArray::alloc(), &dims, dtype) }
            .map_err(|e| Error::Inference(format!("MLMultiArray {shape:?}: {e}")))?;
    // Zeroed, so nothing uninitialized ever reaches a program.
    let bytes = shape.iter().product::<usize>()
        * if dtype == MLMultiArrayDataType::Float16 {
            2
        } else {
            4
        };
    unsafe { std::ptr::write_bytes(array.dataPointer().as_ptr().cast::<u8>(), 0, bytes) };
    Ok(array)
}

/// An array as a dictionary value.
fn object(array: &MLMultiArray) -> Retained<AnyObject> {
    Retained::into_super(Retained::into_super(array.retain()))
}

/// # Safety
/// `array` holds at least `len` contiguous elements of `T`'s type, and
/// nothing else reads or writes it while the slice lives.
/// The array is an Objective-C object Core ML reads by reference, so a
/// shared reference to it is all Rust has; the decoder's `&mut self` is
/// what keeps the writes exclusive.
#[allow(clippy::mut_from_ref)]
unsafe fn slice_mut<T>(array: &MLMultiArray, len: usize) -> &mut [T] {
    std::slice::from_raw_parts_mut(array.dataPointer().as_ptr().cast::<T>(), len)
}

/// The named output of a prediction, as a multi-array.
fn output(
    result: &ProtocolObject<dyn MLFeatureProvider>,
    name: &str,
) -> Result<Retained<MLMultiArray>> {
    unsafe { result.featureValueForName(&NSString::from_str(name)) }
        .and_then(|v| unsafe { v.multiArrayValue() })
        .ok_or_else(|| {
            Error::Inference(format!("the prediction has no multi-array output `{name}`"))
        })
}

/// A `[1, V]` fp16 row into `dst`, through the array's strides.
fn read_f16_row(array: &MLMultiArray, dst: &mut [f32]) -> Result<()> {
    if unsafe { array.dataType() } != MLMultiArrayDataType::Float16 {
        return Err(Error::Inference("the logits aren't fp16".into()));
    }
    let shape: Vec<usize> = unsafe { array.shape() }
        .iter()
        .map(|n| n.as_usize())
        .collect();
    let strides: Vec<usize> = unsafe { array.strides() }
        .iter()
        .map(|n| n.as_usize())
        .collect();
    if shape.iter().product::<usize>() != dst.len() {
        return Err(Error::Inference(format!(
            "logits {shape:?}, expected {} values",
            dst.len()
        )));
    }
    let stride = *strides.last().unwrap_or(&1);
    // SAFETY: the array holds its shape's elements at its strides, alive
    // while `array` is.
    let src = unsafe {
        std::slice::from_raw_parts(
            array.dataPointer().as_ptr().cast::<u16>(),
            (dst.len() - 1) * stride + 1,
        )
    };
    if stride == 1 {
        for (o, &h) in dst.iter_mut().zip(src) {
            *o = f16::from_bits(h).to_f32();
        }
    } else {
        for (i, o) in dst.iter_mut().enumerate() {
            *o = f16::from_bits(src[i * stride]).to_f32();
        }
    }
    Ok(())
}

/// What a loaded program declares: its multi-array inputs and outputs and
/// its states, with shapes and element types.
fn declared(model: &MLModel) -> ProgramDecl {
    let description = unsafe { model.modelDescription() };
    let dtype = |t: MLMultiArrayDataType| match t {
        t if t == MLMultiArrayDataType::Float16 => Dtype::F16,
        t if t == MLMultiArrayDataType::Float32 => Dtype::F32,
        t if t == MLMultiArrayDataType::Double => Dtype::F64,
        t if t == MLMultiArrayDataType::Int32 => Dtype::I32,
        _ => Dtype::Other,
    };
    let shape = |n: Retained<objc2_foundation::NSArray<NSNumber>>| {
        n.iter().map(|x| x.as_usize()).collect::<Vec<_>>()
    };
    let tensors = |features: &NSDictionary<NSString, MLFeatureDescription>, state: bool| {
        let (names, descs) = features.to_vecs();
        names
            .iter()
            .zip(&descs)
            .map(|(name, f)| {
                let decl = if state {
                    unsafe { f.stateConstraint() }.map(|c| TensorDecl {
                        shape: shape(unsafe { c.bufferShape() }),
                        dtype: dtype(unsafe { c.dataType() }),
                    })
                } else {
                    unsafe { f.multiArrayConstraint() }.map(|c| TensorDecl {
                        shape: shape(unsafe { c.shape() }),
                        dtype: dtype(unsafe { c.dataType() }),
                    })
                };
                (
                    name.to_string(),
                    decl.unwrap_or(TensorDecl {
                        shape: vec![],
                        dtype: Dtype::Other,
                    }),
                )
            })
            .collect()
    };
    ProgramDecl {
        inputs: tensors(&*unsafe { description.inputDescriptionsByName() }, false),
        outputs: tensors(&*unsafe { description.outputDescriptionsByName() }, false),
        states: tensors(&*unsafe { description.stateDescriptionsByName() }, true),
    }
}

/// A file mapped read-only.
struct Mapping {
    ptr: *mut std::ffi::c_void,
    len: usize,
}

impl Mapping {
    fn open(path: &Path) -> std::io::Result<Self> {
        use std::os::fd::AsRawFd;
        let file = std::fs::File::open(path)?;
        let len = file.metadata()?.len() as usize;
        if len == 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "empty file",
            ));
        }
        // SAFETY: a read-only private mapping of a whole open file; the
        // mapping outlives the descriptor, as mmap allows.
        let ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                len,
                libc::PROT_READ,
                libc::MAP_PRIVATE,
                file.as_raw_fd(),
                0,
            )
        };
        if ptr == libc::MAP_FAILED {
            return Err(std::io::Error::last_os_error());
        }
        Ok(Self { ptr, len })
    }

    fn bytes(&self) -> &[u8] {
        // SAFETY: the mapping is `len` readable bytes until it's dropped.
        unsafe { std::slice::from_raw_parts(self.ptr.cast::<u8>(), self.len) }
    }
}

impl Drop for Mapping {
    fn drop(&mut self) {
        // SAFETY: unmapping exactly what `open` mapped, once.
        unsafe { libc::munmap(self.ptr, self.len) };
    }
}

/// The host's token-embedding table, `[vocab, hidden]` fp16, mapped: a
/// lookup touches one row's pages, so the table's resident share stays
/// small ("Host assets").
struct Embedding {
    map: Mapping,
    /// Where the table's data starts in the file.
    offset: usize,
    row_bytes: usize,
}

impl Embedding {
    fn open(path: &Path, vocab: usize, hidden: usize) -> Result<Self> {
        let fail = |m: String| Error::Inference(format!("embedding table {}: {m}", path.display()));
        let map = Mapping::open(path).map_err(|e| fail(e.to_string()))?;
        let tensors =
            safetensors::SafeTensors::deserialize(map.bytes()).map_err(|e| fail(e.to_string()))?;
        let names = tensors.names();
        if names.len() != 1 {
            return Err(fail(format!("{} tensors; the table is one", names.len())));
        }
        let view = tensors.tensor(names[0]).map_err(|e| fail(e.to_string()))?;
        if view.dtype() != safetensors::Dtype::F16 || view.shape() != [vocab, hidden] {
            return Err(fail(format!(
                "{:?} {:?}, and the manifest makes it [{vocab}, {hidden}] F16",
                view.shape(),
                view.dtype()
            )));
        }
        let offset = view.data().as_ptr() as usize - map.bytes().as_ptr() as usize;
        Ok(Self {
            map,
            offset,
            row_bytes: hidden * 2,
        })
    }

    /// Token `id`'s row, as little-endian fp16 bytes.
    fn row(&self, id: usize) -> &[u8] {
        let start = self.offset + id * self.row_bytes;
        &self.map.bytes()[start..start + self.row_bytes]
    }
}

/// The RoPE tables, `cos` and `sin`, each `[C, head_dim]` fp16, as bits.
fn read_rope(path: &Path, capacity: usize, head_dim: usize) -> Result<(Vec<u16>, Vec<u16>)> {
    let fail = |m: String| Error::Inference(format!("RoPE tables {}: {m}", path.display()));
    let raw = std::fs::read(path).map_err(|e| fail(e.to_string()))?;
    let tensors = safetensors::SafeTensors::deserialize(&raw).map_err(|e| fail(e.to_string()))?;
    let table = |name: &str| {
        let view = tensors
            .tensor(name)
            .map_err(|e| fail(format!("`{name}`: {e}")))?;
        if view.dtype() != safetensors::Dtype::F16 || view.shape() != [capacity, head_dim] {
            return Err(fail(format!(
                "`{name}` is {:?} {:?}, and the manifest makes it [{capacity}, {head_dim}] F16",
                view.shape(),
                view.dtype()
            )));
        }
        Ok(view
            .data()
            .as_chunks::<2>()
            .0
            .iter()
            .map(|b| u16::from_le_bytes(*b))
            .collect())
    };
    Ok((table("cos")?, table("sin")?))
}

#[cfg(test)]
mod tests {
    use super::{judge_placement, Placement};

    #[test]
    fn placement_needs_half_the_weights() {
        let mib = 1 << 20;
        assert_eq!(
            judge_placement(0, 2185 * mib, 2173 * mib),
            Placement::Confirmed(2185 * mib)
        );
        assert_eq!(
            judge_placement(500 * mib, 1600 * mib, 2173 * mib),
            Placement::Confirmed(1100 * mib)
        );
        assert_eq!(judge_placement(0, 0, 2173 * mib), Placement::Off);
        // Another model's programs keep the total up; the decoder added nothing.
        assert_eq!(
            judge_placement(300 * mib, 300 * mib, 2173 * mib),
            Placement::Unconfirmed { grew: 0 }
        );
        assert_eq!(
            judge_placement(300 * mib, 200 * mib, 2173 * mib),
            Placement::Unconfirmed { grew: 0 }
        );
        assert_eq!(
            judge_placement(300 * mib, 900 * mib, 2173 * mib),
            Placement::Unconfirmed { grew: 600 * mib }
        );
        // Programs with next to no weights still have to show up.
        assert_eq!(
            judge_placement(10, 10, 0),
            Placement::Unconfirmed { grew: 0 }
        );
        assert_eq!(judge_placement(10, 20, 0), Placement::Confirmed(10));
    }
}
