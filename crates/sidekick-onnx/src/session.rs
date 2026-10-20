//! One graph on ONNX Runtime's CPU: inputs fed in the integer type the
//! graph declares, and its output returned as it comes.

use crate::runtime::load_runtime;
use ort::ep;
use ort::session::builder::GraphOptimizationLevel;
use ort::session::{RunOptions, Session};
use ort::value::{DynValue, Tensor, TensorElementType};
use sidekick_core::manifest::CoremlIoNames;
use sidekick_core::{Error, Result};
use std::path::Path;
use std::sync::{Mutex, PoisonError};

/// One `[io]` input: its name and the integer type the graph declares.
struct Input {
    name: String,
    int64: bool,
}

/// How a session holds the memory a run allocates for its activations.
///
/// ONNX Runtime's CPU arena keeps every block a run grew it by, so one long
/// input leaves the process at its peak for as long as the model is loaded:
/// Qwen3-Embedding-0.6B stayed 2.4 GB above its loaded footprint after one
/// 4,096-token input, through every input after it (D40). Its
/// `memory.enable_memory_arena_shrinkage` run option frees the blocks a run
/// no longer uses at the end of that run, to the system allocator, which on
/// macOS returns the pages to the system lazily. Shrinking only after long
/// runs leaves short ones exactly as they were.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MemoryOptions {
    /// The CPU arena (ONNX Runtime's default). Without it every allocation
    /// goes to the system allocator.
    pub cpu_arena: bool,
    /// Shrink the arena after a run of at least this many tokens (rows ×
    /// length); `Some(0)` after every run, `None` never.
    pub shrink_from_tokens: Option<usize>,
}

impl MemoryOptions {
    /// Whether a run of `tokens` (rows × length) shrinks the arena after.
    pub fn shrinks_after(self, tokens: usize) -> bool {
        self.cpu_arena && self.shrink_from_tokens.is_some_and(|from| tokens >= from)
    }
}

impl Default for MemoryOptions {
    /// The arena, shrunk after a run of [`SHRINK_FROM_TOKENS`] or more.
    fn default() -> Self {
        Self { cpu_arena: true, shrink_from_tokens: Some(SHRINK_FROM_TOKENS) }
    }
}

/// The run size (rows × tokens) from which a run's arena growth is given
/// back: twice the 512 tokens most embedders' limits stop at, where what a
/// run allocates grows with the square of its length. Measured on
/// Qwen3-Embedding-0.6B, shrinking after every run costs short inputs
/// nothing measurable either, but this keeps their path unchanged.
pub const SHRINK_FROM_TOKENS: usize = 1024;

/// A graph loaded into an ONNX Runtime session.
pub struct OnnxModel {
    /// The model id, for errors.
    id: String,
    /// `run` needs the session exclusively: concurrent requests to one
    /// model take turns, each using all its intra-op threads.
    session: Mutex<Session>,
    input_ids: Input,
    attention_mask: Option<Input>,
    token_type_ids: Option<Input>,
    position_ids: Option<Input>,
    output: String,
    memory: MemoryOptions,
    /// The run options of a run that shrinks the arena.
    shrink: RunOptions,
}

impl OnnxModel {
    /// Load the graph at `path`, named `id`, with the manifest's `[io]`
    /// names and `intra_op_threads` inference threads. Loads the runtime
    /// first if this process hasn't (the daemon did at startup; tools do it
    /// here).
    pub fn load(path: &Path, id: &str, io: &CoremlIoNames, intra_op_threads: usize) -> Result<Self> {
        Self::load_with(path, id, io, intra_op_threads, MemoryOptions::default())
    }

    /// [`load`](Self::load) with other memory options (measurements).
    #[doc(hidden)]
    pub fn load_with(
        path: &Path,
        id: &str,
        io: &CoremlIoNames,
        intra_op_threads: usize,
        memory: MemoryOptions,
    ) -> Result<Self> {
        load_runtime(None).map_err(Error::Inference)?;
        let file = path.file_name().map_or_else(|| path.display().to_string(), |f| f.to_string_lossy().into_owned());
        let fail = |what: &str, e: ort::Error| Error::Inference(format!("model `{id}`: {what}: {e}"));
        // Session creation is where ONNX Runtime first picks its kernels,
        // and this thread may not be the one that loaded the runtime.
        crate::avx512::enable_on_this_thread();
        let mut builder = Session::builder()
            .map_err(|e| fail("session", e))?
            .with_intra_threads(intra_op_threads)
            .map_err(|e| fail("intra-op threads", e.into()))?
            .with_inter_threads(1)
            .map_err(|e| fail("inter-op threads", e.into()))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| fail("optimization level", e.into()))?;
        if !memory.cpu_arena {
            builder = builder
                .with_execution_providers([ep::CPU::default().with_arena_allocator(false).build()])
                .map_err(|e| fail("CPU arena", e.into()))?;
        }
        let session = builder.commit_from_file(path).map_err(|e| fail(&format!("can't load {file}"), e))?;
        let mut shrink = RunOptions::new().map_err(|e| fail("run options", e))?;
        // "cpu:0": the CPU device's arena, the only one a CPU session has.
        shrink
            .set("memory.enable_memory_arena_shrinkage", "cpu:0")
            .map_err(|e| fail("arena shrinkage", e))?;
        let declared = |name: &str| -> Result<Input> {
            let outlet = session
                .inputs()
                .iter()
                .find(|i| i.name() == name)
                .ok_or_else(|| Error::Inference(format!("model `{id}`: the graph has no input `{name}`")))?;
            let int64 = match outlet.dtype().tensor_type() {
                Some(TensorElementType::Int64) => true,
                Some(TensorElementType::Int32) => false,
                other => {
                    return Err(Error::Inference(format!(
                        "model `{id}`: input `{name}` must be int32 or int64, not {other:?}"
                    )))
                }
            };
            Ok(Input { name: name.to_string(), int64 })
        };
        Ok(Self {
            id: id.to_string(),
            input_ids: declared(&io.input_ids)?,
            attention_mask: io.attention_mask.as_deref().map(declared).transpose()?,
            token_type_ids: io.token_type_ids.as_deref().map(declared).transpose()?,
            position_ids: io.position_ids.as_deref().map(declared).transpose()?,
            output: io.output.clone(),
            session: Mutex::new(session),
            memory,
            shrink,
        })
    }

    /// Run `rows` inputs of `len` tokens each: `ids` and `mask` hold
    /// `rows × len` values, row-major. Token types are fed zeros and
    /// positions 0..len, where the graph takes them. Returns the output's
    /// shape and its values.
    pub fn run(&self, ids: &[i64], mask: &[i64], rows: usize, len: usize) -> Result<(Vec<i64>, Vec<f32>)> {
        let fail = |e: ort::Error| Error::Inference(format!("model `{}`: {e}", self.id));
        if ids.len() != rows * len || mask.len() != rows * len {
            return Err(Error::Inference(format!("model `{}`: {rows} rows of {len} tokens need {} ids", self.id, rows * len)));
        }
        let positions: Vec<i64> = (0..rows).flat_map(|_| 0..len as i64).collect();
        let shape = [rows, len];
        let tensor = |input: &Input, data: &[i64]| -> std::result::Result<DynValue, ort::Error> {
            Ok(if input.int64 {
                Tensor::from_array((shape, data.to_vec()))?.into_dyn()
            } else {
                Tensor::from_array((shape, data.iter().map(|&v| v as i32).collect::<Vec<_>>()))?.into_dyn()
            })
        };
        let mut inputs: Vec<(String, DynValue)> =
            vec![(self.input_ids.name.clone(), tensor(&self.input_ids, ids).map_err(fail)?)];
        if let Some(i) = &self.attention_mask {
            inputs.push((i.name.clone(), tensor(i, mask).map_err(fail)?));
        }
        if let Some(i) = &self.token_type_ids {
            inputs.push((i.name.clone(), tensor(i, &vec![0; rows * len]).map_err(fail)?));
        }
        if let Some(i) = &self.position_ids {
            inputs.push((i.name.clone(), tensor(i, &positions).map_err(fail)?));
        }
        let mut session = self.session.lock().unwrap_or_else(PoisonError::into_inner);
        let outputs = if self.memory.shrinks_after(rows * len) {
            session.run_with_options(inputs, &self.shrink)
        } else {
            session.run(inputs)
        }
        .map_err(fail)?;
        // Checked at scan, but the file may have changed since: an error
        // naming what the graph does return, not a panic.
        let output = outputs.get(self.output.as_str()).ok_or_else(|| {
            let names: Vec<&str> = outputs.keys().collect();
            Error::Inference(format!("model `{}`: the graph returned no `{}` (its outputs: {names:?})", self.id, self.output))
        })?;
        let (out_shape, data) = output.try_extract_tensor::<f32>().map_err(fail)?;
        Ok((out_shape.iter().copied().collect(), data.to_vec()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_arena_shrinks_after_long_runs_only() {
        let default = MemoryOptions::default();
        assert!(!default.shrinks_after(512), "a short input's path is unchanged");
        assert!(!default.shrinks_after(SHRINK_FROM_TOKENS - 1));
        assert!(default.shrinks_after(SHRINK_FROM_TOKENS));
        assert!(default.shrinks_after(4096));
        let never = MemoryOptions { shrink_from_tokens: None, ..default };
        assert!(!never.shrinks_after(4096));
        // Without an arena there's nothing to shrink.
        let no_arena = MemoryOptions { cpu_arena: false, ..default };
        assert!(!no_arena.shrinks_after(4096));
        assert!(MemoryOptions { shrink_from_tokens: Some(0), ..default }.shrinks_after(1));
    }
}
