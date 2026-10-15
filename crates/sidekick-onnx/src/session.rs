//! One graph on ONNX Runtime's CPU: inputs fed in the integer type the
//! graph declares, and its output returned as it comes.

use crate::runtime::load_runtime;
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
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
}

impl OnnxModel {
    /// Load the graph at `path`, named `id`, with the manifest's `[io]`
    /// names and `intra_op_threads` inference threads. Loads the runtime
    /// first if this process hasn't (the daemon did at startup; tools do it
    /// here).
    pub fn load(path: &Path, id: &str, io: &CoremlIoNames, intra_op_threads: usize) -> Result<Self> {
        load_runtime(None).map_err(Error::Inference)?;
        let file = path.file_name().map_or_else(|| path.display().to_string(), |f| f.to_string_lossy().into_owned());
        let fail = |what: &str, e: ort::Error| Error::Inference(format!("model `{id}`: {what}: {e}"));
        // Session creation is where ONNX Runtime first picks its kernels,
        // and this thread may not be the one that loaded the runtime.
        crate::avx512::enable_on_this_thread();
        let session = Session::builder()
            .map_err(|e| fail("session", e))?
            .with_intra_threads(intra_op_threads)
            .map_err(|e| fail("intra-op threads", e.into()))?
            .with_inter_threads(1)
            .map_err(|e| fail("inter-op threads", e.into()))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| fail("optimization level", e.into()))?
            .commit_from_file(path)
            .map_err(|e| fail(&format!("can't load {file}"), e))?;
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
        let outputs = session.run(inputs).map_err(fail)?;
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
