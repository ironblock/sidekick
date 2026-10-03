#![allow(unsafe_code)]
// `dataPointer` is deprecated in favor of the block-based accessors, but the
// block variants need block2 and buy us nothing for a same-thread copy of a
// freshly created / just-returned array.
#![allow(deprecated)]

use crate::{
    shape_verdict, ComputeUnits, InputShape, Int32Input, ModelInterface, ShapeConstraint,
    ShapeVerdict,
};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2::{available, AnyThread};
use objc2_core_ml::{
    MLDictionaryFeatureProvider, MLFeatureProvider, MLFeatureValue, MLModel,
    MLModelConfiguration, MLMultiArray, MLMultiArrayDataType, MLMultiArrayShapeConstraintType,
};
use objc2_foundation::{NSDictionary, NSNumber, NSString, NSURL};
use sidekick_core::{Error, Result};
use std::path::Path;

pub(crate) fn to_ml(units: ComputeUnits) -> objc2_core_ml::MLComputeUnits {
    use objc2_core_ml::MLComputeUnits;
    match units {
        ComputeUnits::All => MLComputeUnits::All,
        ComputeUnits::CpuAndNeuralEngine => MLComputeUnits::CPUAndNeuralEngine,
        ComputeUnits::CpuAndGpu => MLComputeUnits::CPUAndGPU,
        ComputeUnits::CpuOnly => MLComputeUnits::CPUOnly,
    }
}

/// The inverse of [`to_ml`]; `None` for a value this crate never sets.
fn from_ml(units: objc2_core_ml::MLComputeUnits) -> Option<ComputeUnits> {
    [ComputeUnits::All, ComputeUnits::CpuAndNeuralEngine, ComputeUnits::CpuAndGpu, ComputeUnits::CpuOnly]
        .into_iter()
        .find(|&u| to_ml(u) == units)
}

/// A float32 output tensor read back from a prediction.
#[derive(Debug, Clone)]
pub struct OutputTensor {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

/// A loaded Core ML model.
///
/// `MLModel` prediction is thread-safe per Apple's docs; we still funnel
/// sidekick predictions through one blocking task at a time at the server
/// layer, since the ANE serializes requests anyway.
pub struct CoremlModel {
    model: Retained<MLModel>,
    /// The model's input names, from its description.
    inputs: Vec<String>,
}

// SAFETY: MLModel is documented thread-safe for predictions, and we do not
// expose any mutable configuration after load.
unsafe impl Send for CoremlModel {}
unsafe impl Sync for CoremlModel {}

impl CoremlModel {
    /// Load a **compiled** model (`.mlmodelc` directory). If handed a
    /// `.mlpackage`/`.mlmodel`, compiles it first via `MLModel::compileModelAtURL`
    /// (synchronous variant) — callers should cache the compiled artifact by
    /// shipping `.mlmodelc` in the model directory to avoid recompiles.
    ///
    /// Refuses, on macOS 27 and later, a model with an input that accepts
    /// several enumerated shapes: predicting with one there can abort the
    /// process with an Objective-C exception Rust can't catch (D27). Other
    /// flexible-shape models load with a warning; they run on the CPU.
    pub fn load(path: &Path, units: ComputeUnits) -> Result<Self> {
        let model = open(path, units)?;
        match load_verdict(&read_input_shapes(&model)) {
            ShapeVerdict::Static => {}
            ShapeVerdict::Warn(reason) => {
                tracing::warn!(model = %path.display(), "flexible-shape Core ML model: {reason}");
            }
            ShapeVerdict::Refuse(reason) => {
                return Err(Error::Inference(format!(
                    "refusing Core ML model {}: {reason}",
                    path.display()
                )));
            }
        }
        let (names, _) = unsafe { model.modelDescription().inputDescriptionsByName() }.to_vecs();
        let inputs = names.iter().map(|n| n.to_string()).collect();
        Ok(Self { model, inputs })
    }

    /// The model's input names.
    pub fn input_names(&self) -> &[String] {
        &self.inputs
    }

    /// The compute units Core ML holds in this model's configuration: what
    /// it was loaded with, read back from Core ML rather than remembered.
    pub fn compute_units(&self) -> Option<ComputeUnits> {
        from_ml(unsafe { self.model.configuration().computeUnits() })
    }

    /// Run a prediction with named int32 inputs, returning the named float
    /// output. Fails if the output is missing or not a multiarray.
    pub fn predict_int32(&self, inputs: &[Int32Input<'_>], output: &str) -> Result<OutputTensor> {
        let features =
            inputs.iter().map(|i| Ok((NSString::from_str(i.name), int32_value(i)?))).collect::<Result<Vec<_>>>()?;
        read_output(&*self.run(features)?, output)
    }

    /// One prediction from named feature values.
    fn run(&self, features: Vec<(Retained<NSString>, Retained<MLFeatureValue>)>) -> Result<Prediction> {
        let (keys, values): (Vec<_>, Vec<_>) = features.into_iter().unzip();
        let key_refs: Vec<&NSString> = keys.iter().map(|k| &**k).collect();
        let value_objs: Vec<Retained<objc2::runtime::AnyObject>> = values
            .into_iter()
            .map(|v| Retained::into_super(Retained::into_super(v)))
            .collect();
        let dict: Retained<NSDictionary<NSString, objc2::runtime::AnyObject>> =
            NSDictionary::from_retained_objects(&key_refs, &value_objs);

        let provider = unsafe {
            MLDictionaryFeatureProvider::initWithDictionary_error(
                MLDictionaryFeatureProvider::alloc(),
                dict.as_ref(),
            )
        }
        .map_err(|e| Error::Inference(format!("feature provider: {e}")))?;

        let provider = ProtocolObject::from_retained::<MLDictionaryFeatureProvider>(provider);
        unsafe { self.model.predictionFromFeatures_error(&provider) }
            .map_err(|e| Error::Inference(format!("prediction: {e}")))
    }
}

type Prediction = Retained<ProtocolObject<dyn MLFeatureProvider>>;

/// An int32 multi-array feature value holding `input`'s data.
fn int32_value(input: &Int32Input<'_>) -> Result<Retained<MLFeatureValue>> {
    let expected: usize = input.shape.iter().product();
    if expected != input.data.len() {
        return Err(Error::Inference(format!(
            "input `{}`: shape {:?} does not match data length {}",
            input.name,
            input.shape,
            input.data.len()
        )));
    }
    let shape: Vec<Retained<NSNumber>> = input
        .shape
        .iter()
        .map(|&d| NSNumber::new_usize(d))
        .collect();
    let shape = objc2_foundation::NSArray::from_retained_slice(&shape);
    let array = unsafe {
        MLMultiArray::initWithShape_dataType_error(
            MLMultiArray::alloc(),
            &shape,
            MLMultiArrayDataType::Int32,
        )
    }
    .map_err(|e| Error::Inference(format!("MLMultiArray alloc: {e}")))?;

    // SAFETY: the array was just created with Int32 dtype and
    // `expected` elements; dataPointer is valid for its lifetime and
    // no other reference exists yet.
    unsafe {
        let ptr = array.dataPointer().as_ptr() as *mut i32;
        std::ptr::copy_nonoverlapping(input.data.as_ptr(), ptr, expected);
    }
    Ok(unsafe { MLFeatureValue::featureValueWithMultiArray(&array) })
}

/// The named feature value of a prediction.
fn feature(result: &ProtocolObject<dyn MLFeatureProvider>, name: &str) -> Result<Retained<MLFeatureValue>> {
    unsafe { result.featureValueForName(&NSString::from_str(name)) }
        .ok_or_else(|| Error::Inference(format!("missing output feature `{name}`")))
}

/// The address of a feature value's multi-array data, to show a handoff
/// passes the same buffer.
fn data_address(value: &MLFeatureValue) -> Option<usize> {
    let array = unsafe { value.multiArrayValue() }?;
    Some(unsafe { array.dataPointer() }.as_ptr() as usize)
}

/// A prediction's named output, read back as f32.
fn read_output(result: &ProtocolObject<dyn MLFeatureProvider>, output: &str) -> Result<OutputTensor> {
    let value = feature(result, output)?;
    let array = unsafe { value.multiArrayValue() }
        .ok_or_else(|| Error::Inference(format!("output `{output}` is not a multiarray")))?;

    let shape: Vec<usize> = unsafe { array.shape() }
        .iter()
        .map(|n| n.as_usize())
        .collect();
    let count: usize = shape.iter().product();
    let dtype = unsafe { array.dataType() };

    // SAFETY: pointer valid for the array's lifetime; we bounds-read
    // exactly `count` elements of the reported dtype.
    let data: Vec<f32> = unsafe {
        let ptr = array.dataPointer().as_ptr();
        match dtype {
            MLMultiArrayDataType::Float32 => {
                std::slice::from_raw_parts(ptr as *const f32, count).to_vec()
            }
            MLMultiArrayDataType::Float16 => {
                let halves = std::slice::from_raw_parts(ptr as *const u16, count);
                halves
                    .iter()
                    .map(|&h| half_to_f32(h))
                    .collect()
            }
            MLMultiArrayDataType::Double => {
                let doubles = std::slice::from_raw_parts(ptr as *const f64, count);
                doubles.iter().map(|&d| d as f32).collect()
            }
            other => {
                return Err(Error::Inference(format!(
                    "output `{output}`: unsupported dtype {other:?}"
                )))
            }
        }
    };

    Ok(OutputTensor { shape, data })
}

/// The input and output that carry the residual stream between chunks (D37).
pub const HIDDEN_IN: &str = sidekick_core::manifest::CHUNK_HIDDEN_IN;
pub const HIDDEN_OUT: &str = sidekick_core::manifest::CHUNK_HIDDEN_OUT;

/// A bucket's programs, run in order (D37): one program, or a chain of
/// chunks. Each chunk is given the int32 inputs it declares and, after the
/// first, the previous chunk's `hidden_out` as its `hidden_in`: the output
/// feature value itself, so the handoff copies nothing.
pub struct CoremlChain {
    chunks: Vec<CoremlModel>,
}

impl CoremlChain {
    /// Load every program, in chain order, with the same compute units.
    pub fn load(paths: &[std::path::PathBuf], units: ComputeUnits) -> Result<Self> {
        if paths.is_empty() {
            return Err(Error::Inference("a chain needs at least one program".into()));
        }
        let chunks = paths.iter().map(|p| CoremlModel::load(p, units)).collect::<Result<Vec<_>>>()?;
        Ok(Self { chunks })
    }

    /// How many programs answer each prediction.
    pub fn len(&self) -> usize {
        self.chunks.len()
    }

    /// Always false: a chain has at least one program.
    pub fn is_empty(&self) -> bool {
        self.chunks.is_empty()
    }

    /// The compute units every program was loaded with, read back from
    /// Core ML; `None` when they don't agree or aren't known.
    pub fn compute_units(&self) -> Option<ComputeUnits> {
        let first = self.chunks[0].compute_units()?;
        self.chunks.iter().all(|c| c.compute_units() == Some(first)).then_some(first)
    }

    /// [`CoremlModel::predict_int32`] through the chain. A single program
    /// is given every input, as before chunking.
    pub fn predict_int32(&self, inputs: &[Int32Input<'_>], output: &str) -> Result<OutputTensor> {
        self.predict(inputs, output, None)
    }

    /// [`predict_int32`](Self::predict_int32), also returning, per
    /// boundary, the address of the `hidden_out` data a chunk produced and
    /// of the `hidden_in` data the next chunk was given: equal when the
    /// handoff copies nothing.
    #[doc(hidden)]
    pub fn predict_int32_tracing_handoffs(
        &self,
        inputs: &[Int32Input<'_>],
        output: &str,
    ) -> Result<(OutputTensor, Vec<(usize, usize)>)> {
        let mut trace = Vec::new();
        let out = self.predict(inputs, output, Some(&mut trace))?;
        Ok((out, trace))
    }

    fn predict(
        &self,
        inputs: &[Int32Input<'_>],
        output: &str,
        mut trace: Option<&mut Vec<(usize, usize)>>,
    ) -> Result<OutputTensor> {
        if self.chunks.len() == 1 {
            return self.chunks[0].predict_int32(inputs, output);
        }
        // The previous chunk's `hidden_out`, and where its data was when produced.
        let mut carried: Option<(Retained<MLFeatureValue>, Option<usize>)> = None;
        for (i, chunk) in self.chunks.iter().enumerate() {
            let mut features = Vec::with_capacity(chunk.inputs.len());
            for input in inputs.iter().filter(|i| chunk.inputs.iter().any(|n| n == i.name)) {
                features.push((NSString::from_str(input.name), int32_value(input)?));
            }
            if let Some((hidden, produced)) = carried.take() {
                if let Some(trace) = trace.as_deref_mut() {
                    trace.push((produced.unwrap_or(0), data_address(&hidden).unwrap_or(1)));
                }
                features.push((NSString::from_str(HIDDEN_IN), hidden));
            }
            if features.len() != chunk.inputs.len() {
                return Err(Error::Inference(format!(
                    "chunk {i} of the chain takes {:?}, and the request supplies {} of them",
                    chunk.inputs,
                    features.len()
                )));
            }
            let result = chunk.run(features)?;
            if i + 1 == self.chunks.len() {
                return read_output(&result, output);
            }
            let hidden = feature(&result, HIDDEN_OUT)?;
            let produced = data_address(&hidden);
            carried = Some((hidden, produced));
        }
        unreachable!("the last chunk returns")
    }
}

/// The shape constraints of a model's multi-array inputs, sorted by name.
/// Loads the model for the CPU only, which is cheap and never predicts, so
/// it is safe on artifacts that [`CoremlModel::load`] refuses. Judge the
/// result with [`crate::shape_verdict`].
pub fn input_shapes(path: &Path) -> Result<Vec<InputShape>> {
    let model = open(path, ComputeUnits::CpuOnly)?;
    Ok(read_input_shapes(&model))
}

/// What [`CoremlModel::load`] does with a model whose inputs have these
/// shape constraints, on the running OS: [`shape_verdict`] with the macOS 27
/// rule applied when it's running (D27).
pub fn load_verdict(inputs: &[InputShape]) -> ShapeVerdict {
    shape_verdict(inputs, available!(macos = 27.0))
}

/// A model's declared interface: input shape constraints, and the declared
/// shape of every multi-array input and output. Like [`input_shapes`], it
/// loads the model for the CPU only and never predicts, so it's cheap next
/// to an ANE load and safe on artifacts [`CoremlModel::load`] refuses.
pub fn interface(path: &Path) -> Result<ModelInterface> {
    let model = open(path, ComputeUnits::CpuOnly)?;
    let description = unsafe { model.modelDescription() };
    let shapes = |features: &NSDictionary<NSString, objc2_core_ml::MLFeatureDescription>| {
        let (names, _) = features.to_vecs();
        names
            .iter()
            .filter_map(|n| Some((n.to_string(), declared_shape(features, &n.to_string())?)))
            .collect()
    };
    Ok(ModelInterface {
        constraints: read_input_shapes(&model),
        inputs: shapes(&*unsafe { description.inputDescriptionsByName() }),
        outputs: shapes(&*unsafe { description.outputDescriptionsByName() }),
    })
}

/// Compile if needed, then load with the given compute units.
fn open(path: &Path, units: ComputeUnits) -> Result<Retained<MLModel>> {
    let is_compiled = path
        .extension()
        .map(|e| e == "mlmodelc")
        .unwrap_or(false);

    let url_for = |p: &Path| -> Retained<NSURL> {
        let s = NSString::from_str(&p.to_string_lossy());
        NSURL::fileURLWithPath(&s)
    };

    let compiled_url = if is_compiled {
        url_for(path)
    } else {
        // The synchronous compiler is deprecated in favor of the
        // completion-handler variant, but it's exactly right for a
        // blocking loader and avoids a block2 dependency. Ship
        // precompiled .mlmodelc in the model dir to skip this entirely
        // (`xcrun coremlcompiler compile model.mlpackage .`).
        let src = url_for(path);
        #[allow(deprecated)]
        unsafe { MLModel::compileModelAtURL_error(&src) }.map_err(|e| {
            Error::Inference(format!("Core ML compile failed for {}: {e}", path.display()))
        })?
    };

    let config = unsafe { MLModelConfiguration::new() };
    unsafe { config.setComputeUnits(to_ml(units)) };

    unsafe { MLModel::modelWithContentsOfURL_configuration_error(&compiled_url, &config) }.map_err(
        |e| Error::Inference(format!("Core ML load failed for {}: {e}", path.display())),
    )
}

/// Read each multi-array input's shape constraint from the model
/// description. Per-bucket artifacts built by coremltools report
/// `.enumerated` with exactly one shape, so the number of shapes is what
/// tells a flexible model apart, not the constraint type.
fn read_input_shapes(model: &MLModel) -> Vec<InputShape> {
    let description = unsafe { model.modelDescription() };
    let (names, features) = unsafe { description.inputDescriptionsByName() }.to_vecs();
    let mut inputs: Vec<InputShape> = names
        .iter()
        .zip(&features)
        .filter_map(|(name, feature)| {
            let shape = unsafe { feature.multiArrayConstraint()?.shapeConstraint() };
            let constraint = match unsafe { shape.r#type() } {
                MLMultiArrayShapeConstraintType::Enumerated => ShapeConstraint::Enumerated(
                    unsafe { shape.enumeratedShapes() }
                        .iter()
                        .map(|s| s.iter().map(|n| n.as_usize()).collect())
                        .collect(),
                ),
                MLMultiArrayShapeConstraintType::Range => ShapeConstraint::Range(
                    unsafe { shape.sizeRangeForDimension() }
                        .iter()
                        .map(|v| match v.get_range() {
                            // `length` sizes starting at `location`; a fixed
                            // dimension of 128 reads (128, 1).
                            Some(r) => r.location..=r.location.saturating_add(r.length.max(1) - 1),
                            // Never observed; count it as flexible.
                            None => 0..=usize::MAX,
                        })
                        .collect(),
                ),
                _ => ShapeConstraint::Unspecified,
            };
            Some(InputShape { name: name.to_string(), constraint })
        })
        .collect();
    inputs.sort_by(|a, b| a.name.cmp(&b.name));
    inputs
}

fn declared_shape(
    features: &NSDictionary<NSString, objc2_core_ml::MLFeatureDescription>,
    name: &str,
) -> Option<Vec<usize>> {
    let feature = features.objectForKey(&NSString::from_str(name))?;
    let constraint = unsafe { feature.multiArrayConstraint()? };
    Some(unsafe { constraint.shape() }.iter().map(|n| n.as_usize()).collect())
}

/// Minimal f16 -> f32 (avoids pulling `half` into this crate). Verified
/// bit-exact against the IEEE-754 definition for all 65536 inputs (see the
/// exhaustive test below); the subnormal branch is correct — the normalized
/// leading bit becomes the *implicit* bit of the f32, encoded via the
/// exponent, which is why `f & 0x3ff` masks it off.
fn half_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) & 1) as u32;
    let exp = ((h >> 10) & 0x1f) as u32;
    let frac = (h & 0x3ff) as u32;
    let bits = match (exp, frac) {
        (0, 0) => sign << 31,
        (0, f) => {
            // subnormal: renormalize
            let mut e = 127 - 15 + 1;
            let mut f = f;
            while f & 0x400 == 0 {
                f <<= 1;
                e -= 1;
            }
            (sign << 31) | ((e as u32) << 23) | ((f & 0x3ff) << 13)
        }
        (0x1f, 0) => (sign << 31) | 0x7f80_0000,
        (0x1f, f) => (sign << 31) | 0x7f80_0000 | (f << 13),
        (e, f) => (sign << 31) | ((e + 127 - 15) << 23) | (f << 13),
    };
    f32::from_bits(bits)
}

#[cfg(test)]
mod tests {
    use super::half_to_f32;

    /// Independent reference straight from the IEEE-754 binary16 definition,
    /// via f64 arithmetic (exact: every binary16 value fits in f64).
    fn reference(h: u16) -> f32 {
        let sign = if h >> 15 & 1 == 1 { -1.0f64 } else { 1.0 };
        let exp = (h >> 10 & 0x1f) as i32;
        let frac = (h & 0x3ff) as f64;
        (match exp {
            0 => sign * (frac / 1024.0) * (2.0f64).powi(-14),
            0x1f if frac == 0.0 => sign * f64::INFINITY,
            0x1f => f64::NAN,
            e => sign * (1.0 + frac / 1024.0) * (2.0f64).powi(e - 15),
        }) as f32
    }

    #[test]
    fn half_to_f32_is_bit_exact_for_all_inputs() {
        for h in 0..=u16::MAX {
            let got = half_to_f32(h);
            let want = reference(h);
            if want.is_nan() {
                assert!(got.is_nan(), "h={h:#06x}: expected NaN, got {got}");
            } else {
                assert_eq!(
                    got.to_bits(),
                    want.to_bits(),
                    "h={h:#06x}: got {got} ({:#010x}), want {want} ({:#010x})",
                    got.to_bits(),
                    want.to_bits()
                );
            }
        }
    }
}
