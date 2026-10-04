//! Core ML classifier (ANE-targeted): text-classification, rerankers, and the
//! laya and gliner2 zero-shot formats (docs/design/classify.md).
//!
//! Pipeline: [`InputBuilder`] (tokenize, lay out, bucket) → pad to the
//! bucket → int32 prediction → the logits in label order. Activation is the
//! server's job ([`sidekick_core::activate`]).
//!
//! Graph interface, every input int32:
//! - text-classification: `input_ids [1,S]`, `attention_mask [1,S]` →
//!   `logits [1,N]`;
//! - laya: those plus `marker_pos [1,KMAX]` (−1 in unused slots) and
//!   `qtype [1]` → `logits [1,KMAX]`, unused slots at −1e4;
//! - gliner2: `input_ids [1,S]`, `attention_mask [1,S]` → `logits [1,S]`,
//!   one per token, read at the `[L]` positions the input builder placed.

use crate::bucket_models::BucketModels;
use crate::classify_input::InputBuilder;
use sidekick_core::manifest::{ClassifierManifest, ClassifyFormat, ResolvedClassifier};
use sidekick_core::{
    Classifier, ClassifyParams, ClassifyTask, Error, PairParams, Prepared, ProblemType, Result,
    Source,
};
use sidekick_coreml::{ComputeUnits, Int32Input, ShapeVerdict};

pub struct CoremlClassifier {
    manifest: ClassifierManifest,
    inputs: InputBuilder,
    models: BucketModels,
    io: Io,
}

/// Core ML feature names, resolved from `[classify.io]` for the format.
struct Io {
    input_ids: String,
    attention_mask: String,
    token_type_ids: Option<String>,
    marker_pos: Option<String>,
    qtype: Option<String>,
    /// fev: the decide token's position, `[1]`.
    decide_pos: Option<String>,
    /// agentjev: the tree's segments and positions, `[1, S]` each.
    seg: Option<String>,
    position_ids: Option<String>,
    output: String,
    /// gliner2: the output is one logit per token, read at the markers.
    per_token: bool,
}

/// Check one bucket's artifact against the manifest: `input_ids` and
/// `attention_mask` are `[1, bucket]` (for per-bucket `{seq}` artifacts;
/// a shared artifact must only have them), and so are agentjev's `seg` and
/// `position_ids`; laya's, fev's and agentjev's `marker_pos` is
/// `[1, max_labels]`, laya's `qtype` and fev's `decide_pos` are `[1]`; the
/// output has one slot per label (`max_labels` for those formats), or exactly
/// `[1, bucket]` for gliner2's per-token
/// logits; and the inputs pass the flexible-shape guard (D27). Errors name
/// the artifact relative to the model directory.
fn check_interface(
    m: &ClassifierManifest,
    io: &Io,
    bucket: usize,
    name: &str,
    path: &std::path::Path,
) -> Result<()> {
    let fail = |message: String| Err(Error::Inference(format!("model `{}`, {name}: {message}", m.id)));
    let iface = match sidekick_coreml::interface(path) {
        Ok(iface) => iface,
        Err(e) => return fail(format!("can't read the artifact: {e}")),
    };
    if let ShapeVerdict::Refuse(reason) = sidekick_coreml::load_verdict(&iface.constraints) {
        return fail(reason);
    }
    let expect = expected_inputs(m, io, bucket);
    for (input, shape) in &expect {
        match (iface.inputs.get(*input), shape) {
            (None, _) => return fail(format!("no int32 multi-array input `{input}`")),
            (Some(got), Some(shape)) if got != shape => {
                return fail(format!("input `{input}` is {got:?}, expected {shape:?}"))
            }
            _ => {}
        }
    }
    // Every multi-array input the artifact declares must be one the
    // manifest names: an unnamed one would never be fed, and every
    // prediction would fail. This runs at classifier load only (embedders
    // don't read their artifacts' interfaces), and it refuses an unnamed
    // input even if the model marks it optional.
    let named: Vec<&String> = expect.iter().map(|(n, _)| *n).collect();
    if let Some(extra) = iface.inputs.keys().find(|k| !named.contains(k)) {
        return fail(format!(
            "takes input `{extra}`, which the manifest's [classify.io] doesn't name"
        ));
    }
    check_output(m, io, bucket, &iface, fail)
}

/// The inputs a bucket's artifact must declare, with their shapes where
/// they are fixed.
fn expected_inputs<'a>(m: &ClassifierManifest, io: &'a Io, bucket: usize) -> Vec<(&'a String, Option<Vec<usize>>)> {
    let per_bucket = m.artifact.contains("{seq}");
    let seq = per_bucket.then(|| vec![1, bucket]);
    let mut expect = vec![(&io.input_ids, seq.clone()), (&io.attention_mask, seq.clone())];
    for per_token in [&io.token_type_ids, &io.seg, &io.position_ids].into_iter().flatten() {
        expect.push((per_token, seq.clone()));
    }
    if let Some(marker) = &io.marker_pos {
        expect.push((marker, Some(vec![1, m.max_labels()])));
    }
    for one in [&io.qtype, &io.decide_pos].into_iter().flatten() {
        expect.push((one, Some(vec![1])));
    }
    expect
}

/// The output the artifact (a chain's last chunk) must produce.
fn check_output(
    m: &ClassifierManifest,
    io: &Io,
    bucket: usize,
    iface: &sidekick_coreml::ModelInterface,
    fail: impl Fn(String) -> Result<()>,
) -> Result<()> {
    let labels = m.max_labels();
    match iface.outputs.get(&io.output) {
        None => fail(format!("no multi-array output `{}`", io.output)),
        Some(shape) if io.per_token => match shape.as_slice() {
            [1, n] if *n == bucket => Ok(()),
            _ => fail(format!(
                "output `{}` is {shape:?}, expected [1, {bucket}] (one logit per token)",
                io.output
            )),
        },
        Some(shape) if !shape.is_empty() && shape.last() != Some(&labels) => fail(format!(
            "output `{}` is {shape:?}, expected {labels} slots (one per label)",
            io.output
        )),
        Some(_) => Ok(()),
    }
}

/// Check a chunked bucket's programs (D37), named relative to the model
/// directory: together they take every input the manifest names (each at
/// its shape), the first takes `input_ids` and no `hidden_in`, every later
/// one takes `hidden_in`, which the one before outputs as `hidden_out` at
/// the same [1, bucket, H] shape, no chunk takes an input the manifest
/// doesn't name, and the last produces the manifest's output.
fn check_chunks(m: &ClassifierManifest, io: &Io, bucket: usize, names: &[String], paths: &[std::path::PathBuf]) -> Result<()> {
    use sidekick_coreml::{HIDDEN_IN, HIDDEN_OUT};
    let fail_in = |name: &str, message: String| Err(Error::Inference(format!("model `{}`, {name}: {message}", m.id)));
    let mut ifaces = Vec::with_capacity(paths.len());
    for (name, path) in names.iter().zip(paths) {
        let iface = match sidekick_coreml::interface(path) {
            Ok(iface) => iface,
            Err(e) => return fail_in(name, format!("can't read the artifact: {e}")),
        };
        if let ShapeVerdict::Refuse(reason) = sidekick_coreml::load_verdict(&iface.constraints) {
            return fail_in(name, reason);
        }
        ifaces.push(iface);
    }
    let expect = expected_inputs(m, io, bucket);
    let named: Vec<&String> = expect.iter().map(|(n, _)| *n).collect();
    for (i, (name, iface)) in names.iter().zip(&ifaces).enumerate() {
        if let Some(extra) = iface.inputs.keys().find(|k| *k != HIDDEN_IN && !named.contains(k)) {
            return fail_in(name, format!("takes input `{extra}`, which the manifest's [classify.io] doesn't name"));
        }
        for (input, shape) in &expect {
            match (iface.inputs.get(*input), shape) {
                (Some(got), Some(shape)) if got != shape => {
                    return fail_in(name, format!("input `{input}` is {got:?}, expected {shape:?}"))
                }
                _ => {}
            }
        }
        let takes_hidden = iface.inputs.get(HIDDEN_IN);
        if i == 0 {
            if takes_hidden.is_some() {
                return fail_in(name, format!("the first chunk takes `{HIDDEN_IN}`; nothing comes before it"));
            }
            if !iface.inputs.contains_key(&io.input_ids) {
                return fail_in(name, format!("the first chunk doesn't take `{}`", io.input_ids));
            }
        } else {
            let produced = ifaces[i - 1].outputs.get(HIDDEN_OUT);
            match (produced, takes_hidden) {
                (_, None) => return fail_in(name, format!("a chunk after the first must take `{HIDDEN_IN}`")),
                (None, _) => return fail_in(&names[i - 1], format!("a chunk before the last must output `{HIDDEN_OUT}`")),
                (Some(out), Some(inp)) if out != inp || out.len() != 3 || out[..2] != [1, bucket] => {
                    return fail_in(name, format!(
                        "takes `{HIDDEN_IN}` {inp:?}, and the chunk before outputs `{HIDDEN_OUT}` {out:?}; both must \
                         be the same [1, {bucket}, hidden]"
                    ))
                }
                _ => {}
            }
        }
    }
    for (input, _) in &expect {
        if !ifaces.iter().any(|f| f.inputs.contains_key(*input)) {
            return fail_in(&names[0], format!("no chunk takes the int32 multi-array input `{input}`"));
        }
    }
    let last = names.last().expect("a chain has a program");
    check_output(m, io, bucket, ifaces.last().expect("a chain has a program"), |message| fail_in(last, message))
}

fn io_name(name: &Option<String>, what: &str) -> Result<String> {
    name.clone()
        .ok_or_else(|| Error::Inference(format!("classifier manifest has no `{what}` feature")))
}

impl CoremlClassifier {
    /// Load with the manifest's compute units (`cpu_and_ne` by default), as
    /// the daemon does.
    pub fn load(model: &ResolvedClassifier) -> Result<Self> {
        Self::load_with(model, model.manifest.compute_units)
    }

    /// Load with an explicit compute-unit preference, overriding the
    /// manifest's, so tests and the parity suite can compare paths.
    pub fn load_with(model: &ResolvedClassifier, units: ComputeUnits) -> Result<Self> {
        let m = &model.manifest;
        let io = &m.classify.io;
        let laya = m.classify.format == Some(ClassifyFormat::Laya);
        let fev = m.classify.format == Some(ClassifyFormat::Fev);
        let agentjev = m.classify.format == Some(ClassifyFormat::Agentjev);
        let marker_pos = if laya || fev || agentjev { Some(io_name(&io.marker_pos, "marker_pos")?) } else { None };
        let output = io_name(&io.output, "output")?;

        let models = BucketModels::new(&model.dir, &m.artifact, m.chunks(), units);
        let io = Io {
            input_ids: io_name(&io.input_ids, "input_ids")?,
            attention_mask: io_name(&io.attention_mask, "attention_mask")?,
            token_type_ids: io.token_type_ids.clone(),
            marker_pos,
            qtype: if laya { Some(io_name(&io.qtype, "qtype")?) } else { None },
            decide_pos: if fev { Some(io_name(&io.decide_pos, "decide_pos")?) } else { None },
            seg: if agentjev { Some(io_name(&io.seg, "seg")?) } else { None },
            position_ids: if agentjev { Some(io_name(&io.position_ids, "position_ids")?) } else { None },
            output,
            per_token: m.classify.format == Some(ClassifyFormat::Gliner2),
        };
        // Every bucket is its own artifact: check each one's interface now,
        // from its description (a CPU-only load that never predicts), so a
        // bad bucket fails the load instead of only the requests long
        // enough to reach it. A cold read of a large artifact takes about 2 s
        // (measured on 600 MB buckets), so the buckets are read in parallel.
        std::thread::scope(|scope| {
            let checks: Vec<_> = m
                .buckets
                .iter()
                .map(|&bucket| {
                    let (io, models) = (&io, &models);
                    scope.spawn(move || {
                        if m.chunks() > 1 || m.chunking.is_some() {
                            check_chunks(m, io, bucket, &models.artifact_names(bucket), &models.paths(bucket))
                        } else {
                            check_interface(m, io, bucket, &models.artifact_name(bucket), &models.path(bucket))
                        }
                    })
                })
                .collect();
            checks.into_iter().try_for_each(|c| {
                c.join().unwrap_or_else(|_| Err(Error::Inference("bucket check panicked".into())))
            })
        })?;

        // The checks above opened every program, from its description, on the
        // CPU, and released it: a broken artifact has already failed the load.
        // No bucket stays resident until a request needs it, so a model
        // costs the memory of the buckets it serves, not also its smallest.
        Ok(Self { manifest: m.clone(), inputs: InputBuilder::load(model)?, models, io })
    }

    /// Sequence-length buckets, smallest first.
    pub fn buckets(&self) -> &[usize] {
        &self.manifest.buckets
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
        let bucket = *self.manifest.buckets.first().expect("validated non-empty");
        self.models.get(bucket)?.compute_units().ok_or_else(|| {
            Error::Inference("Core ML reports compute units sidekick doesn't set".into())
        })
    }

    /// The buckets loaded and resident, smallest first.
    #[doc(hidden)]
    pub fn resident_buckets(&self) -> Vec<usize> {
        self.models.resident(&self.manifest.buckets)
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
            Int32Input { name: &self.io.input_ids, shape: vec![1, bucket], data: input_ids },
            Int32Input { name: &self.io.attention_mask, shape: vec![1, bucket], data: mask },
        ];
        if let Some(name) = &self.io.token_type_ids {
            // Segment ids for the real tokens (none given: all segment 0),
            // then 0 in the pads, which the mask hides.
            let mut types = prepared.type_ids.clone();
            types.resize(used, 0);
            types.resize(bucket, 0);
            inputs.push(Int32Input { name, shape: vec![1, bucket], data: types });
        }
        if let (Some(seg_name), Some(pos_name)) = (&self.io.seg, &self.io.position_ids) {
            // agentjev's tree: pads get attention_mask 0, segment -1 and
            // position 0. The graph's mask requires attention_mask 1 and a
            // segment >= 0 on every key, so no real token attends a pad and
            // random pad ids change nothing; the -1 keeps each pad's own row
            // from attending the prefix.
            if prepared.seg.len() != used || prepared.position_ids.len() != used {
                return Err(Error::Inference(format!(
                    "{used} tokens but {} segments and {} positions",
                    prepared.seg.len(),
                    prepared.position_ids.len()
                )));
            }
            let mut seg = prepared.seg.clone();
            seg.resize(bucket, -1);
            let mut positions = prepared.position_ids.clone();
            positions.resize(bucket, 0);
            inputs.push(Int32Input { name: seg_name, shape: vec![1, bucket], data: seg });
            inputs.push(Int32Input { name: pos_name, shape: vec![1, bucket], data: positions });
            // The head reads hidden states at the candidates' ends: each
            // must be a real token's, or a pad's state would be read.
            if prepared.markers.iter().any(|&m| m < 0 || m as usize >= used) {
                return Err(Error::Inference(format!(
                    "candidate ends {:?} aren't all among the {used} real tokens",
                    prepared.markers
                )));
            }
        }

        let k = match &self.io.marker_pos {
            Some(marker_name) => {
                let kmax = self.manifest.max_labels();
                let k = prepared.markers.len();
                if k == 0 || k > kmax {
                    return Err(Error::Inference(format!("{k} markers for a model of {kmax} labels")));
                }
                let mut markers = prepared.markers.clone();
                markers.resize(kmax, -1);
                inputs.push(Int32Input { name: marker_name, shape: vec![1, kmax], data: markers });
                if let Some(qtype_name) = &self.io.qtype {
                    let qtype = prepared
                        .qtype
                        .ok_or_else(|| Error::Inference("laya input without a qtype".into()))?;
                    inputs.push(Int32Input { name: qtype_name, shape: vec![1], data: vec![qtype] });
                }
                if let Some(decide_name) = &self.io.decide_pos {
                    // fev reads hidden states at these positions: each must
                    // be a real token's, or a pad's state would be read.
                    let decide = prepared
                        .decide_pos
                        .ok_or_else(|| Error::Inference("fev input without a decide position".into()))?;
                    let real = |p: i32| p >= 0 && (p as usize) < used;
                    if !real(decide) || !prepared.markers.iter().all(|&m| real(m)) {
                        return Err(Error::Inference(format!(
                            "positions {:?} and {decide} aren't all among the {used} real tokens",
                            prepared.markers
                        )));
                    }
                    inputs.push(Int32Input { name: decide_name, shape: vec![1], data: vec![decide] });
                }
                k
            }
            None => self.manifest.classify.labels.len(),
        };

        if self.io.per_token {
            // Every marker must be a real token's position: a pad's logit
            // would be read silently otherwise.
            if prepared.markers.is_empty() || prepared.markers.iter().any(|&m| m < 0 || m as usize >= used) {
                return Err(Error::Inference(format!(
                    "markers {:?} aren't all positions of the {used} real tokens",
                    prepared.markers
                )));
            }
            let model = self.models.get(bucket)?;
            let out = model.predict_int32(&inputs, &self.io.output)?;
            if out.data.len() != bucket || out.shape.last().copied() != Some(bucket) {
                return Err(Error::Inference(format!(
                    "output `{}` has shape {:?}, expected [1, {bucket}]",
                    self.io.output, out.shape
                )));
            }
            return Ok(prepared.markers.iter().map(|&m| out.data[m as usize]).collect());
        }

        let model = self.models.get(bucket)?;
        let out = model.predict_int32(&inputs, &self.io.output)?;
        let width = out.shape.last().copied().unwrap_or(0);
        let expected = if self.io.marker_pos.is_some() { self.manifest.max_labels() } else { k };
        if out.data.len() != expected || width != expected {
            return Err(Error::Inference(format!(
                "output `{}` has shape {:?}, expected [1, {expected}]",
                self.io.output, out.shape
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

    fn prepare_pair(&self, query: &str, document: &str, params: &PairParams) -> Result<Prepared> {
        self.inputs.prepare_pair(query, document, params)
    }

    fn load_buckets(&self, buckets: &[usize]) -> Result<Vec<usize>> {
        crate::coreml_embedder::load_buckets(&self.models, buckets)
    }

    fn run(&self, prepared: &Prepared) -> Result<Vec<f32>> {
        self.run_in(prepared, prepared.bucket, &[])
    }
}
