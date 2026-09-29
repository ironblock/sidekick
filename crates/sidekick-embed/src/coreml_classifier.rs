//! Core ML classifier (ANE-targeted): text-classification and the laya
//! zero-shot format (docs/design/classify.md).
//!
//! Pipeline: [`InputBuilder`] (tokenize, lay out, bucket) → pad to the
//! bucket → int32 prediction → the logits in label order. Activation is the
//! server's job ([`sidekick_core::activate`]).
//!
//! Graph interface, every input int32:
//! - text-classification: `input_ids [1,S]`, `attention_mask [1,S]` →
//!   `logits [1,N]`;
//! - laya: those plus `marker_pos [1,KMAX]` (−1 in unused slots) and
//!   `qtype [1]` → `logits [1,KMAX]`, unused slots at −1e4.

use crate::bucket_models::BucketModels;
use crate::classify_input::InputBuilder;
use sidekick_core::manifest::{ClassifierManifest, ClassifyFormat, ResolvedClassifier};
use sidekick_core::{
    Classifier, ClassifyParams, ClassifyTask, Error, Prepared, ProblemType, Result, Source,
};
use sidekick_coreml::{ComputeUnits, Int32Input};

pub struct CoremlClassifier {
    manifest: ClassifierManifest,
    inputs: InputBuilder,
    models: BucketModels,
    input_ids: String,
    attention_mask: String,
    marker_pos: Option<String>,
    qtype: Option<String>,
    output: String,
}

fn io_name(name: &Option<String>, what: &str) -> Result<String> {
    name.clone()
        .ok_or_else(|| Error::Inference(format!("classifier manifest has no `{what}` feature")))
}

impl CoremlClassifier {
    /// Load for the ANE (`.cpuAndNeuralEngine`), as the daemon does.
    pub fn load(model: &ResolvedClassifier) -> Result<Self> {
        Self::load_with(model, ComputeUnits::CpuAndNeuralEngine)
    }

    /// Load with an explicit compute-unit preference, for tests that
    /// compare paths.
    pub fn load_with(model: &ResolvedClassifier, units: ComputeUnits) -> Result<Self> {
        let m = &model.manifest;
        let io = &m.classify.io;
        let laya = m.classify.format == Some(ClassifyFormat::Laya);
        let marker_pos = if laya { Some(io_name(&io.marker_pos, "marker_pos")?) } else { None };
        let output = io_name(&io.output, "output")?;

        // Checked on every bucket as it loads: each is its own artifact.
        let max_labels = m.max_labels();
        let check_marker = marker_pos.clone();
        let check_output = output.clone();
        let fixed_labels = m.task == ClassifyTask::TextClassification;
        let models = BucketModels::new(&model.dir, &m.artifact, units).with_check(Box::new(
            move |model, path| {
                if let Some(name) = &check_marker {
                    let width = model.input_shape(name).and_then(|s| s.last().copied());
                    if width != Some(max_labels) {
                        return Err(Error::Inference(format!(
                            "{}: `max_labels` is {max_labels} but input `{name}` is {}",
                            path.display(),
                            match model.input_shape(name) {
                                Some(shape) => format!("{shape:?} wide"),
                                None => "missing".into(),
                            }
                        )));
                    }
                }
                if fixed_labels {
                    if let Some(shape) = model.output_shape(&check_output) {
                        if shape.last() != Some(&max_labels) {
                            return Err(Error::Inference(format!(
                                "{}: output `{check_output}` has shape {shape:?} for {max_labels} labels",
                                path.display()
                            )));
                        }
                    }
                }
                Ok(())
            },
        ));

        let classifier = Self {
            manifest: m.clone(),
            inputs: InputBuilder::load(model)?,
            models,
            input_ids: io_name(&io.input_ids, "input_ids")?,
            attention_mask: io_name(&io.attention_mask, "attention_mask")?,
            marker_pos,
            qtype: if laya { Some(io_name(&io.qtype, "qtype")?) } else { None },
            output,
        };
        // Load the smallest bucket eagerly so a broken artifact fails at
        // load time, not on the first request.
        classifier.models.get(*m.buckets.first().expect("validated non-empty"))?;
        Ok(classifier)
    }

    /// Sequence-length buckets, smallest first.
    pub fn buckets(&self) -> &[usize] {
        &self.manifest.buckets
    }

    /// The input builder, for tests and the parity suite.
    #[doc(hidden)]
    pub fn inputs(&self) -> &InputBuilder {
        &self.inputs
    }

    /// Run `prepared` in `bucket` with the given ids in the pad positions
    /// (id 0 past their end). The attention mask hides pads, so neither the
    /// bucket nor the pad ids may change the result: the parity suite's
    /// bucket and pad invariance checks.
    #[doc(hidden)]
    pub fn run_in(&self, prepared: &Prepared, bucket: usize, pad_ids: &[i32]) -> Result<Vec<f32>> {
        let ids = &prepared.ids;
        if !self.manifest.buckets.contains(&bucket) || ids.len() > bucket {
            return Err(Error::Inference(format!(
                "{} tokens can't run in bucket {bucket} (buckets {:?})",
                ids.len(),
                self.manifest.buckets
            )));
        }
        let used = ids.len();
        let mut input_ids = ids.clone();
        input_ids.extend(pad_ids.iter().take(bucket - used));
        input_ids.resize(bucket, 0);
        let mut mask = vec![1i32; used];
        mask.resize(bucket, 0);
        let mut inputs = vec![
            Int32Input { name: &self.input_ids, shape: vec![1, bucket], data: input_ids },
            Int32Input { name: &self.attention_mask, shape: vec![1, bucket], data: mask },
        ];

        let k = match (&self.marker_pos, &self.qtype) {
            (Some(marker_name), Some(qtype_name)) => {
                let kmax = self.manifest.max_labels();
                let k = prepared.markers.len();
                if k == 0 || k > kmax {
                    return Err(Error::Inference(format!("{k} markers for a model of {kmax} labels")));
                }
                let qtype = prepared
                    .qtype
                    .ok_or_else(|| Error::Inference("laya input without a qtype".into()))?;
                let mut markers = prepared.markers.clone();
                markers.resize(kmax, -1);
                inputs.push(Int32Input { name: marker_name, shape: vec![1, kmax], data: markers });
                inputs.push(Int32Input { name: qtype_name, shape: vec![1], data: vec![qtype] });
                k
            }
            _ => self.manifest.classify.labels.len(),
        };

        let model = self.models.get(bucket)?;
        let out = model.predict_int32(&inputs, &self.output)?;
        let width = out.shape.last().copied().unwrap_or(0);
        let expected = if self.marker_pos.is_some() { self.manifest.max_labels() } else { k };
        if out.data.len() != expected || width != expected {
            return Err(Error::Inference(format!(
                "output `{}` has shape {:?}, expected [1, {expected}]",
                self.output, out.shape
            )));
        }
        let mut logits = out.data;
        logits.truncate(k);
        Ok(logits)
    }
}

impl Classifier for CoremlClassifier {
    fn id(&self) -> &str {
        &self.manifest.id
    }

    fn task(&self) -> ClassifyTask {
        self.manifest.task
    }

    fn problem_type(&self) -> ProblemType {
        self.manifest.problem_type
    }

    fn labels(&self) -> &[String] {
        &self.manifest.classify.labels
    }

    fn max_labels(&self) -> usize {
        self.manifest.max_labels()
    }

    fn max_batch(&self) -> usize {
        self.manifest.max_batch
    }

    fn calibration(&self, params: &ClassifyParams, k: usize) -> Option<f32> {
        self.manifest.temperature(params.question_type, k)
    }

    fn source(&self) -> Option<&Source> {
        self.manifest.source.as_ref()
    }

    fn prepare(&self, input: &str, params: &ClassifyParams) -> Result<Prepared> {
        self.inputs.prepare(input, params)
    }

    fn run(&self, prepared: &Prepared) -> Result<Vec<f32>> {
        self.run_in(prepared, prepared.bucket, &[])
    }
}
