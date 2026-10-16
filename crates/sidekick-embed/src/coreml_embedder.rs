//! Core ML encoder embedder (ANE-targeted).
//!
//! Pipeline: tokenize (HF `tokenizers`) → pad to the smallest sequence-length
//! bucket that fits → int32 `input_ids`/`attention_mask` prediction →
//! pool per manifest → unit-normalize.
//!
//! Each bucket maps to its own static-shape artifact when the manifest uses
//! a `{seq}` placeholder; bucket models load lazily on first use and stay
//! resident. (A single enumerated-shapes artifact is also supported, but
//! hardware verification showed E5RT rejects flexible shapes at ANE plan
//! time and silently runs the whole encoder on CPU — prefer per-bucket
//! static artifacts.)

use crate::bucket_models::BucketModels;
use crate::chain_check::{check_chain, ChainInputs};
use crate::embed_input::EmbedInput;
use crate::pooling::{normalize_in_place, pool};
use sidekick_core::manifest::ResolvedModel;
use sidekick_core::{EmbedLimits, EmbedPurpose, Embedder, Error, Pooling, Result};
use sidekick_coreml::{ComputeUnits, Int32Input, ShapeVerdict};

pub struct CoremlEmbedder {
    id: String,
    dims: usize,
    matryoshka: Vec<usize>,
    /// Lazily loaded per-bucket models.
    models: BucketModels,
    /// Tokenizing, prefixes and truncation, shared with the ONNX backend.
    input: EmbedInput,
    buckets: Vec<usize>,
    pooling: Pooling,
    input_ids_name: String,
    attention_mask_name: Option<String>,
    output_name: String,
    /// Each bucket is a chain of programs (D37).
    chunked: bool,
    /// The artifact has a `{seq}` placeholder: each bucket's inputs are [1, bucket].
    per_bucket: bool,
}

/// The token ids sidekick feeds the model for one input, and the bucket they
/// go in. Exposed so tests can check tokenization against a reference
/// pipeline and run the same ids through a larger bucket.
#[doc(hidden)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Prepared {
    /// Real token ids (prefix applied, specials included, truncated to the
    /// largest bucket), before padding.
    pub ids: Vec<i32>,
    /// The smallest bucket that fits `ids`.
    pub bucket: usize,
}

impl CoremlEmbedder {
    /// Load with the manifest's compute units (`cpu_and_ne` by default), as
    /// the daemon does.
    pub fn load(model: &ResolvedModel) -> Result<Self> {
        Self::load_with(model, model.manifest.compute_units.unwrap_or_default())
    }

    /// Load with an explicit compute-unit preference, overriding the
    /// manifest's, so tests and the parity suite can compare paths.
    pub fn load_with(model: &ResolvedModel, units: ComputeUnits) -> Result<Self> {
        let m = &model.manifest;
        let embedder = Self {
            id: m.id.clone(),
            dims: m.dims,
            matryoshka: m.matryoshka.clone(),
            models: BucketModels::new(&model.dir, &m.artifact, m.chunks(), units),
            input: EmbedInput::load(model)?,
            buckets: m.buckets.clone(),
            pooling: m.pooling,
            input_ids_name: m.io.input_ids.clone(),
            attention_mask_name: m.io.attention_mask.clone(),
            output_name: m.io.output.clone(),
            chunked: m.chunks() > 1 || m.chunking.is_some(),
            per_bucket: m.artifact.contains("{seq}"),
        };
        // Check every bucket's artifact now, from its description (a CPU-only
        // open that never predicts, then released), so a broken one fails the
        // load (the pool's load-error surface), not the first request long
        // enough to need it. No bucket stays resident until a request needs it.
        std::thread::scope(|scope| {
            let checks: Vec<_> = embedder
                .buckets
                .iter()
                .map(|&b| {
                    let embedder = &embedder;
                    scope.spawn(move || embedder.check_bucket(b))
                })
                .collect();
            checks.into_iter().try_for_each(|c| {
                c.join().unwrap_or_else(|_| Err(Error::Inference("bucket check panicked".into())))
            })
        })?;
        Ok(embedder)
    }

    /// One bucket's artifact: readable, past D27's shape guard, with the
    /// manifest's input and output names. Errors name the artifact relative
    /// to the model directory.
    fn check_bucket(&self, bucket: usize) -> Result<()> {
        if self.chunked {
            return self.check_chain(bucket);
        }
        let name = self.models.artifact_name(bucket);
        let fail = |message: String| Err(Error::Inference(format!("model `{}`, {name}: {message}", self.id)));
        let iface = match sidekick_coreml::interface(&self.models.path(bucket)) {
            Ok(iface) => iface,
            Err(e) => return fail(format!("can't read the artifact: {e}")),
        };
        if let ShapeVerdict::Refuse(reason) = sidekick_coreml::load_verdict(&iface.constraints) {
            return fail(reason);
        }
        let names = std::iter::once(&self.input_ids_name).chain(self.attention_mask_name.as_ref());
        for input in names {
            if !iface.inputs.contains_key(input) {
                return fail(format!("no int32 multi-array input `{input}`"));
            }
        }
        // an artifact may not declare its outputs' shapes; check the name when it does
        if !iface.outputs.is_empty() && !iface.outputs.contains_key(&self.output_name) {
            return fail(format!("no multi-array output `{}`", self.output_name));
        }
        Ok(())
    }

    /// A chunked bucket's programs (D37): the chain takes `input_ids` and
    /// the mask at [1, bucket] and nothing else the manifest doesn't name,
    /// links `hidden_out` to `hidden_in`, and ends in the manifest's output.
    fn check_chain(&self, bucket: usize) -> Result<()> {
        let seq = self.per_bucket.then(|| vec![1, bucket]);
        let inputs = std::iter::once(&self.input_ids_name)
            .chain(self.attention_mask_name.as_ref())
            .map(|n| (n, seq.clone()))
            .collect();
        let want = ChainInputs { id: &self.id, bucket, inputs, input_ids: &self.input_ids_name, table: "[io]" };
        check_chain(&want, &self.models.artifact_names(bucket), &self.models.paths(bucket), |iface, fail| {
            // as for one program: check the name when the artifact declares its outputs
            if !iface.outputs.is_empty() && !iface.outputs.contains_key(&self.output_name) {
                return fail(format!("no multi-array output `{}`", self.output_name));
            }
            Ok(())
        })
    }

    /// Sequence-length buckets, smallest first.
    pub fn buckets(&self) -> &[usize] {
        &self.buckets
    }

    /// The compute units every bucket loads with: the ones sidekick
    /// configures Core ML with. Reading them loads nothing.
    pub fn compute_units(&self) -> Result<ComputeUnits> {
        Ok(self.models.units())
    }

    /// The compute units Core ML holds in a loaded bucket's configuration,
    /// loading the smallest if none is: for tests that Core ML honors
    /// [`compute_units`](Self::compute_units).
    #[doc(hidden)]
    pub fn loaded_compute_units(&self) -> Result<ComputeUnits> {
        let bucket = *self.buckets.first().expect("validated non-empty");
        self.models.get(bucket)?.compute_units().ok_or_else(|| {
            Error::Inference("Core ML reports compute units sidekick doesn't set".into())
        })
    }

    /// The buckets loaded and resident, smallest first.
    #[doc(hidden)]
    pub fn resident_buckets(&self) -> Vec<usize> {
        self.models.resident(&self.buckets)
    }

    /// Apply the purpose's prefix, tokenize, truncate, and pick a bucket.
    /// Public for the parity suite (examples/parity), which checks the ids
    /// against a reference pipeline.
    #[doc(hidden)]
    pub fn prepare(&self, text: &str, purpose: EmbedPurpose) -> Result<Prepared> {
        self.prepare_with(text, purpose, EmbedLimits::default())
    }

    /// `prepare` with per-request limits: at most `max_tokens` (and the
    /// largest bucket), keeping the start (`END`, the final token
    /// preserved as `prepare` does), the end (`START`: the first token
    /// preserved, since a tokenizer's leading special token is what CLS
    /// pooling reads), or a 400 (`NONE`).
    pub fn prepare_with(&self, text: &str, purpose: EmbedPurpose, limits: EmbedLimits) -> Result<Prepared> {
        let largest = *self.buckets.last().expect("validated non-empty");
        let ids = self.input.ids(text, purpose, limits, largest)?;
        let bucket = *self
            .buckets
            .iter()
            .find(|&&b| b >= ids.len())
            .unwrap_or(&largest);
        Ok(Prepared { ids, bucket })
    }

    /// Pad `ids` to `bucket` with id 0, predict, pool, and unit-normalize.
    /// `bucket` must be one of the model's buckets and hold all of `ids`; the
    /// daemon always passes the one `prepare` picked. Public for the parity
    /// suite, which also runs ids through larger buckets.
    #[doc(hidden)]
    pub fn run(&self, ids: &[i32], bucket: usize) -> Result<Vec<f32>> {
        self.run_padded(ids, bucket, &[])
    }

    /// `run` with the given ids in the pad positions (id 0 past their end).
    /// The attention mask hides pads, so the result must not change: the
    /// parity suite's check for a dropped mask.
    #[doc(hidden)]
    pub fn run_padded(&self, ids: &[i32], bucket: usize, pad_ids: &[i32]) -> Result<Vec<f32>> {
        if !self.buckets.contains(&bucket) || ids.len() > bucket {
            return Err(Error::Inference(format!(
                "{} tokens can't run in bucket {bucket} (buckets {:?})",
                ids.len(),
                self.buckets
            )));
        }
        let used = ids.len();
        let mut input_ids = ids.to_vec();
        input_ids.extend(pad_ids.iter().take(bucket - used));
        input_ids.resize(bucket, 0);
        let mut mask = vec![1i32; used];
        mask.resize(bucket, 0);

        let mut inputs = vec![Int32Input {
            name: &self.input_ids_name,
            shape: vec![1, bucket],
            data: input_ids,
        }];
        if let Some(mask_name) = &self.attention_mask_name {
            inputs.push(Int32Input {
                name: mask_name,
                shape: vec![1, bucket],
                data: mask.clone(),
            });
        }

        let model = self.models.get(bucket)?;
        let out = model.predict_int32(&inputs, &self.output_name)?;
        let n: usize = out.shape.iter().product();

        let mut vector = match self.pooling {
            Pooling::None => {
                if n != self.dims {
                    return Err(Error::Inference(format!(
                        "pooled output shape {:?} != dims {}",
                        out.shape, self.dims
                    )));
                }
                out.data
            }
            pooling => {
                if n != bucket * self.dims {
                    return Err(Error::Inference(format!(
                        "hidden-state output shape {:?} != [1, {bucket}, {}]",
                        out.shape, self.dims
                    )));
                }
                let mask_u32: Vec<u32> = mask.iter().map(|&m| m as u32).collect();
                pool(&out.data, self.dims, &mask_u32, pooling)
            }
        };
        normalize_in_place(&mut vector);
        Ok(vector)
    }
}

impl Embedder for CoremlEmbedder {
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

    fn embed_with(
        &self,
        texts: &[&str],
        purpose: EmbedPurpose,
        limits: EmbedLimits,
    ) -> Result<Vec<Vec<f32>>> {
        Ok(self.embed_bucketed(texts, purpose, limits)?.0)
    }

    fn embed_bucketed(
        &self,
        texts: &[&str],
        purpose: EmbedPurpose,
        limits: EmbedLimits,
    ) -> Result<(Vec<Vec<f32>>, Option<Vec<usize>>)> {
        self.embed_staged(texts, purpose, limits, &mut |_| {})
    }

    fn embed_staged(
        &self,
        texts: &[&str],
        purpose: EmbedPurpose,
        limits: EmbedLimits,
        loaded: &mut dyn FnMut(&[usize]),
    ) -> Result<(Vec<Vec<f32>>, Option<Vec<usize>>)> {
        // Prepare every text first, so a rejected one costs no prediction.
        let prepared = texts
            .iter()
            .map(|t| self.prepare_with(t, purpose, limits))
            .collect::<Result<Vec<_>>>()?;
        let buckets: Vec<usize> = prepared.iter().map(|p| p.bucket).collect();
        loaded(&load_buckets(&self.models, &buckets)?);
        let vectors = prepared.iter().map(|p| self.run(&p.ids, p.bucket)).collect::<Result<_>>()?;
        Ok((vectors, Some(buckets)))
    }
}

/// Each of `buckets` loaded, smallest first: the ones that waited on a load.
pub(crate) fn load_buckets(models: &BucketModels, buckets: &[usize]) -> Result<Vec<usize>> {
    let mut unique = buckets.to_vec();
    unique.sort_unstable();
    unique.dedup();
    let mut waited = Vec::new();
    for bucket in unique {
        if models.ensure(bucket)? {
            waited.push(bucket);
        }
    }
    Ok(waited)
}
