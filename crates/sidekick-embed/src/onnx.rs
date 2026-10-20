//! The ONNX Runtime CPU backend for embeddings (D40).
//!
//! An `onnx` model is one graph with dynamic shapes, run by ONNX Runtime
//! on the CPU. Each input runs alone, at its own length, so an embedding
//! never depends on what else is in the request. The tokenizing
//! (`embed_input`) and the pooling and normalization (`pooling`) are the
//! Core ML embedder's: the two backends differ only in the forward pass,
//! which `sidekick-onnx` runs.
//!
//! ONNX Runtime isn't linked at build time. `libonnxruntime` loads the
//! first time the process calls [`load_runtime`]: the daemon does that at
//! startup when its models directory has an `onnx` model, and never
//! otherwise. It comes from the configured path, else `ORT_DYLIB_PATH`,
//! else beside the executable. A build without the `onnx` feature has no
//! runtime, and the daemon skips every `onnx` manifest with the reason.

pub use sidekick_onnx::{
    default_intra_op_threads, default_threads, detected_cores, load_runtime, runtime_state, DetectedCores,
    RuntimeInfo, RuntimeSource, FALLBACK_INTRA_OP_THREADS, MAX_DEFAULT_INTRA_OP_THREADS, PINNED_VERSION,
};

/// Whether this build has the ONNX Runtime backend (the `onnx` feature).
pub const SUPPORTED: bool = cfg!(feature = "onnx");

#[cfg(feature = "onnx")]
pub use backend::OnnxEmbedder;
#[cfg(feature = "onnx")]
pub use sidekick_onnx::{MemoryOptions, SHRINK_FROM_TOKENS};

#[cfg(feature = "onnx")]
mod backend {
    use crate::embed_input::{each_input, EmbedInput};
    use crate::pooling::{degenerate, normalize_in_place, pool};
    use sidekick_core::manifest::ResolvedModel;
    use sidekick_core::{EmbedLimits, EmbedPurpose, Embedder, Error, Pooling, Result};
    use sidekick_onnx::{MemoryOptions, OnnxModel};

    /// An embedding model on ONNX Runtime's CPU.
    pub struct OnnxEmbedder {
        id: String,
        dims: usize,
        matryoshka: Vec<usize>,
        pooling: Pooling,
        max_seq_len: usize,
        input: std::sync::Arc<EmbedInput>,
        model: OnnxModel,
        output: String,
        intra_op_threads: usize,
    }

    impl OnnxEmbedder {
        /// Load with `intra_op_threads` inference threads.
        pub fn load(model: &ResolvedModel, intra_op_threads: usize) -> Result<Self> {
            Self::load_with(model, intra_op_threads, MemoryOptions::default())
        }

        /// [`load`](Self::load) with other memory options (measurements).
        #[doc(hidden)]
        pub fn load_with(model: &ResolvedModel, intra_op_threads: usize, memory: MemoryOptions) -> Result<Self> {
            let m = &model.manifest;
            Ok(Self {
                id: m.id.clone(),
                dims: m.dims,
                matryoshka: m.matryoshka.clone(),
                pooling: m.pooling,
                max_seq_len: m.max_seq_len,
                input: EmbedInput::shared(model)?,
                model: OnnxModel::load_with(&model.dir.join(&m.artifact), &m.id, &m.io, intra_op_threads, memory)?,
                output: m.io.output.clone(),
                intra_op_threads,
            })
        }

        /// The inference threads this model's session uses.
        pub fn intra_op_threads(&self) -> usize {
            self.intra_op_threads
        }

        /// Prefix, tokenize and truncate one input, as the daemon does.
        /// Public for the parity suite.
        #[doc(hidden)]
        pub fn prepare(&self, text: &str, purpose: EmbedPurpose, limits: EmbedLimits) -> Result<Vec<i32>> {
            self.input.ids(text, purpose, limits, self.max_seq_len)
        }

        /// Run `ids` (one input, no padding), pool, and unit-normalize.
        #[doc(hidden)]
        pub fn run(&self, ids: &[i32]) -> Result<Vec<f32>> {
            self.run_batch(&[ids], ids.len()).map(|mut v| v.remove(0))
        }

        /// Run inputs padded to `len` with id 0 as one batch: the length
        /// and batch-invariance measurements (D40) compare this with `run`.
        #[doc(hidden)]
        pub fn run_batch(&self, batch: &[&[i32]], len: usize) -> Result<Vec<Vec<f32>>> {
            let n = batch.len();
            let mut ids = Vec::with_capacity(n * len);
            let mut mask = Vec::with_capacity(n * len);
            for row in batch {
                if row.len() > len || row.is_empty() {
                    return Err(Error::Inference(format!("{} tokens can't run at length {len}", row.len())));
                }
                ids.extend(row.iter().map(|&t| i64::from(t)));
                ids.resize(ids.len() + len - row.len(), 0);
                mask.extend(std::iter::repeat_n(1i64, row.len()));
                mask.resize(mask.len() + len - row.len(), 0);
            }
            let (out_shape, data) = self.model.run(&ids, &mask, n, len)?;
            let want: Vec<i64> = match self.pooling {
                Pooling::None => vec![n as i64, self.dims as i64],
                _ => vec![n as i64, len as i64, self.dims as i64],
            };
            if out_shape != want {
                return Err(Error::Inference(format!(
                    "model `{}`: output `{}` has shape {out_shape:?}, expected {want:?}",
                    self.id, self.output
                )));
            }
            let per_row = data.len() / n;
            let mask_u32: Vec<u32> = mask.iter().map(|&m| m as u32).collect();
            Ok((0..n)
                .map(|r| {
                    let mut v = pool(&data[r * per_row..(r + 1) * per_row], self.dims, &mask_u32[r * len..(r + 1) * len], self.pooling);
                    normalize_in_place(&mut v);
                    v
                })
                .collect())
        }
    }

    impl Embedder for OnnxEmbedder {
        fn id(&self) -> &str {
            &self.id
        }

        fn dims(&self) -> usize {
            self.dims
        }

        fn matryoshka_dims(&self) -> &[usize] {
            &self.matryoshka
        }

        fn embed(&self, texts: &[&str], purpose: EmbedPurpose) -> Result<Vec<Vec<f32>>> {
            self.embed_with(texts, purpose, EmbedLimits::default())
        }

        fn embed_with(&self, texts: &[&str], purpose: EmbedPurpose, limits: EmbedLimits) -> Result<Vec<Vec<f32>>> {
            // Prepare every text first, so a rejected one costs no inference.
            self.input.check_limits(purpose, limits, self.max_seq_len)?;
            let prepared = each_input(texts, |t| self.prepare(t, purpose, limits))?;
            prepared
                .iter()
                .map(|ids| {
                    let v = self.run(ids)?;
                    match degenerate(&v) {
                        Some(what) => Err(Error::Inference(format!(
                            "model `{}`: its output has {what}, which isn't an embedding",
                            self.id
                        ))),
                        None => Ok(v),
                    }
                })
                .collect()
        }
    }
}
