#![allow(unsafe_code)]
// `dataPointer` is deprecated in favor of the block-based accessors, but the
// block variants need block2 and buy us nothing for a same-thread copy of a
// freshly created / just-returned array.
#![allow(deprecated)]

use crate::{ComputeUnits, Int32Input};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2::AnyThread;
use objc2_core_ml::{
    MLDictionaryFeatureProvider, MLFeatureProvider, MLFeatureValue, MLModel,
    MLModelConfiguration, MLMultiArray, MLMultiArrayDataType,
};
use objc2_foundation::{NSDictionary, NSNumber, NSString, NSURL};
use sidekick_core::{Error, Result};
use std::path::Path;

impl ComputeUnits {
    fn to_ml(self) -> objc2_core_ml::MLComputeUnits {
        use objc2_core_ml::MLComputeUnits;
        match self {
            ComputeUnits::All => MLComputeUnits::All,
            ComputeUnits::CpuAndNeuralEngine => MLComputeUnits::CPUAndNeuralEngine,
            ComputeUnits::CpuAndGpu => MLComputeUnits::CPUAndGPU,
            ComputeUnits::CpuOnly => MLComputeUnits::CPUOnly,
        }
    }
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
    pub fn load(path: &Path, units: ComputeUnits) -> Result<Self> {
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
        unsafe { config.setComputeUnits(units.to_ml()) };

        let model = unsafe {
            MLModel::modelWithContentsOfURL_configuration_error(&compiled_url, &config)
        }
        .map_err(|e| {
            Error::Inference(format!("Core ML load failed for {}: {e}", path.display()))
        })?;

        Ok(Self { model })
    }

    /// Run a prediction with named int32 inputs, returning the named float
    /// output. Fails if the output is missing or not a multiarray.
    pub fn predict_int32(&self, inputs: &[Int32Input<'_>], output: &str) -> Result<OutputTensor> {
        let mut keys: Vec<Retained<NSString>> = Vec::with_capacity(inputs.len());
        let mut values: Vec<Retained<MLFeatureValue>> = Vec::with_capacity(inputs.len());

        for input in inputs {
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

            keys.push(NSString::from_str(input.name));
            values.push(unsafe { MLFeatureValue::featureValueWithMultiArray(&array) });
        }

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
        let result = unsafe { self.model.predictionFromFeatures_error(&provider) }
            .map_err(|e| Error::Inference(format!("prediction: {e}")))?;

        let name = NSString::from_str(output);
        let value = unsafe { result.featureValueForName(&name) }
            .ok_or_else(|| Error::Inference(format!("missing output feature `{output}`")))?;
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
