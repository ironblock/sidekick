//! Minimal safe wrapper over Core ML.
//!
//! Scope is deliberately tiny: load a compiled model with a chosen compute
//! unit preference, run batch-1 predictions with int32/float inputs, read
//! float outputs. Everything sidekick needs for encoder models on the ANE,
//! and nothing else.
//!
//! On non-macOS targets this crate compiles to an empty stub so that the
//! workspace builds and tests everywhere; `sidekick-embed` gates its Core ML
//! backend on `target_os = "macos"` accordingly.

#[cfg(target_os = "macos")]
mod model;
#[cfg(target_os = "macos")]
mod plan;
#[cfg(target_os = "macos")]
pub use model::{input_shapes, interface, load_verdict, CoremlModel, OutputTensor};
#[cfg(target_os = "macos")]
pub use plan::compute_plan;

use std::collections::BTreeMap;
use std::ops::RangeInclusive;

/// Compute-unit preference, as a manifest's `compute_units` names it.
/// `CpuAndNeuralEngine` is sidekick's default: it keeps background work off
/// the GPU, which is the point of the project.
pub use sidekick_core::ComputeUnits;

/// A named int32 tensor input (shape is row-major, batch dim included).
#[derive(Debug)]
pub struct Int32Input<'a> {
    pub name: &'a str,
    pub shape: Vec<usize>,
    pub data: Vec<i32>,
}

/// The shapes a Core ML multi-array input accepts, as the compiled model
/// describes them (`MLMultiArrayShapeConstraint`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ShapeConstraint {
    /// No flexibility declared: the input takes its default shape only.
    Unspecified,
    /// A list of accepted shapes. Static-shape artifacts built by
    /// coremltools report exactly one; `ct.EnumeratedShapes` reports several.
    Enumerated(Vec<Vec<usize>>),
    /// An inclusive size range per dimension (`ct.RangeDim`). Fixed
    /// dimensions are single-size ranges.
    Range(Vec<RangeInclusive<usize>>),
}

impl ShapeConstraint {
    /// Whether the input accepts more than one shape.
    pub fn is_flexible(&self) -> bool {
        match self {
            ShapeConstraint::Unspecified => false,
            ShapeConstraint::Enumerated(shapes) => shapes.len() > 1,
            ShapeConstraint::Range(dims) => dims.iter().any(|d| d.start() != d.end()),
        }
    }

    /// Several enumerated shapes: the layout that can abort the process at
    /// prediction on macOS 27.
    fn is_multi_shape(&self) -> bool {
        matches!(self, ShapeConstraint::Enumerated(shapes) if shapes.len() > 1)
    }
}

/// A multi-array input of a Core ML model and the shapes it accepts.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InputShape {
    pub name: String,
    pub constraint: ShapeConstraint,
}

/// A model's interface as its description declares it, read without
/// predicting ([`interface`]).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ModelInterface {
    /// Each multi-array input's shape constraint, sorted by name (what
    /// [`shape_verdict`] judges).
    pub constraints: Vec<InputShape>,
    /// Each multi-array input's declared (default) shape, by name.
    pub inputs: BTreeMap<String, Vec<usize>>,
    /// Each multi-array output's declared shape, by name; empty when the
    /// model doesn't declare it.
    pub outputs: BTreeMap<String, Vec<usize>>,
}

/// What loading does with a model, judged from its inputs' shape
/// constraints.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ShapeVerdict {
    /// Every input takes exactly one shape: the supported layout, one
    /// static-shape artifact per sequence-length bucket (D15).
    Static,
    /// Flexible inputs that still run, off the ANE and slowly: load the
    /// model and log the reason.
    Warn(String),
    /// Flexible inputs that can abort the process at prediction: refuse
    /// the model with this reason.
    Refuse(String),
}

/// Where a converted model should come from instead of a flexible one.
const STATIC_RECIPE: &str = "convert one static-shape .mlmodelc per sequence-length bucket \
     instead (tools/convert_bge_small.py is the reference recipe; see docs/MODELS.md)";

/// Judge a model's inputs. `multi_shape_aborts` says whether predicting
/// with an input that has several enumerated shapes can abort the process
/// with an uncatchable Objective-C exception, which it can on macOS 27
/// (measured under `.cpuOnly`; D27). Such models are refused there,
/// whatever the compute units. Every other flexible layout only warns: it
/// runs, on the CPU instead of the ANE.
pub fn shape_verdict(inputs: &[InputShape], multi_shape_aborts: bool) -> ShapeVerdict {
    let multi: Vec<&InputShape> = inputs.iter().filter(|i| i.constraint.is_multi_shape()).collect();
    if multi_shape_aborts && !multi.is_empty() {
        return ShapeVerdict::Refuse(format!(
            "{} several enumerated shapes ({}). On macOS 27 and later, predicting with such a \
             model can abort the whole process; {STATIC_RECIPE}",
            describe(&multi, "accepts", "accept"),
            shapes(&multi[0].constraint),
        ));
    }
    let flexible: Vec<&InputShape> = inputs.iter().filter(|i| i.constraint.is_flexible()).collect();
    if flexible.is_empty() {
        return ShapeVerdict::Static;
    }
    let mut reason = format!(
        "{} flexible shapes ({}), which run on the CPU instead of the ANE; {STATIC_RECIPE}",
        describe(&flexible, "takes", "take"),
        shapes(&flexible[0].constraint),
    );
    if !multi.is_empty() {
        reason.push_str(". macOS 27 and later refuse models with several enumerated shapes");
    }
    ShapeVerdict::Warn(reason)
}

/// "input `a` accepts" / "inputs `a`, `b` accept".
fn describe(inputs: &[&InputShape], singular: &str, plural: &str) -> String {
    let names: Vec<String> = inputs.iter().map(|i| format!("`{}`", i.name)).collect();
    match names.len() {
        1 => format!("input {} {singular}", names[0]),
        _ => format!("inputs {} {plural}", names.join(", ")),
    }
}

fn shapes(constraint: &ShapeConstraint) -> String {
    match constraint {
        ShapeConstraint::Unspecified => "default only".into(),
        ShapeConstraint::Enumerated(shapes) => {
            shapes.iter().map(|s| format!("{s:?}")).collect::<Vec<_>>().join(", ")
        }
        ShapeConstraint::Range(dims) => format!("{dims:?}"),
    }
}

/// How Core ML's compute plan assigns a model's operations to devices, for
/// the ML program's `main` function (nested blocks included). This is what
/// the compiler *intends* for a given compute-unit preference: it is
/// load-independent and never runs the model, but it can't see failures
/// that only happen at run time (e.g. a transient ANE compile failure).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PlanSummary {
    pub ane: usize,
    pub cpu: usize,
    pub gpu: usize,
    /// Operations with no device usage (constants and other bookkeeping).
    pub unassigned: usize,
    /// Operator names (opset prefix removed) of operations not on the ANE,
    /// with counts.
    pub off_ane_ops: BTreeMap<String, usize>,
}

/// Minimum share of device-assigned operations on the ANE for a model to
/// count as ANE-eligible. The validated encoders measure 93–99.6% on macOS
/// 27 (the rest is mask and cast plumbing); a flexible-shape artifact that
/// runs entirely on the CPU measures 0%.
pub const MIN_ANE_SHARE: f64 = 0.80;

/// Operators that carry an encoder's compute. Any of them off the ANE means
/// the expensive part runs elsewhere, whatever the overall share says
/// (operation counts aren't cost-weighted: Core ML reports no costs).
pub const HEAVY_OPS: &[&str] = &[
    "matmul",
    "linear",
    "conv",
    "conv_transpose",
    "einsum",
    "scaled_dot_product_attention",
];

impl PlanSummary {
    /// Operations assigned to some device.
    pub fn assigned(&self) -> usize {
        self.ane + self.cpu + self.gpu
    }

    /// Share of assigned operations on the ANE (0 when none are assigned).
    pub fn ane_share(&self) -> f64 {
        match self.assigned() {
            0 => 0.0,
            n => self.ane as f64 / n as f64,
        }
    }

    /// Whether the plan puts the model on the ANE; `Err` explains why not.
    pub fn verdict(&self) -> Result<(), String> {
        if self.assigned() == 0 {
            return Err("no operations are assigned to any compute device (a known-good \
                        artifact reads this way when Core ML's bundle cache entry for its \
                        path is broken; re-read a copy of it at another path)"
                .into());
        }
        let heavy: Vec<&str> = self
            .off_ane_ops
            .keys()
            .map(String::as_str)
            .filter(|op| HEAVY_OPS.contains(op))
            .collect();
        if !heavy.is_empty() {
            return Err(format!("compute-heavy operations are off the ANE: {}", heavy.join(", ")));
        }
        if self.ane_share() < MIN_ANE_SHARE {
            return Err(format!(
                "only {:.1}% of operations are on the ANE (need {:.0}%)",
                self.ane_share() * 100.0,
                MIN_ANE_SHARE * 100.0
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plan(ane: usize, cpu: usize, off: &[(&str, usize)]) -> PlanSummary {
        PlanSummary {
            ane,
            cpu,
            gpu: 0,
            unassigned: 0,
            off_ane_ops: off.iter().map(|(k, v)| (k.to_string(), *v)).collect(),
        }
    }

    #[test]
    fn healthy_encoder_passes() {
        // bge-small on macOS 27: mask/cast plumbing on the CPU.
        let p = plan(229, 16, &[("cast", 4), ("add", 3), ("select", 2), ("layer_norm", 1)]);
        assert!(p.verdict().is_ok(), "{:?}", p.verdict());
        assert!((p.ane_share() - 0.9347).abs() < 1e-3);
    }

    #[test]
    fn cpu_fallback_fails() {
        // The flexible-shape bge artifact: nothing on the ANE.
        let p = plan(0, 362, &[("linear", 72)]);
        assert!(p.verdict().is_err());
    }

    #[test]
    fn heavy_op_off_the_ane_fails_even_with_a_high_share() {
        let p = plan(95, 5, &[("matmul", 1), ("cast", 4)]);
        assert!(p.ane_share() >= MIN_ANE_SHARE);
        assert!(p.verdict().unwrap_err().contains("matmul"));
    }

    #[test]
    fn low_share_fails() {
        assert!(plan(70, 30, &[("cast", 30)]).verdict().is_err());
    }

    fn input(name: &str, constraint: ShapeConstraint) -> InputShape {
        InputShape { name: name.into(), constraint }
    }

    fn enumerated(seqs: &[usize]) -> ShapeConstraint {
        ShapeConstraint::Enumerated(seqs.iter().map(|&s| vec![1, s]).collect())
    }

    /// What the per-bucket artifacts report on macOS 27: `.enumerated`
    /// with exactly one shape per input.
    fn static_bucket(seq: usize) -> Vec<InputShape> {
        vec![input("attention_mask", enumerated(&[seq])), input("input_ids", enumerated(&[seq]))]
    }

    #[test]
    fn single_shape_buckets_are_static_on_every_os() {
        for aborts in [false, true] {
            assert_eq!(shape_verdict(&static_bucket(128), aborts), ShapeVerdict::Static);
            let fixed = [input("x", ShapeConstraint::Unspecified)];
            assert_eq!(shape_verdict(&fixed, aborts), ShapeVerdict::Static);
            assert_eq!(shape_verdict(&[], aborts), ShapeVerdict::Static);
        }
    }

    #[test]
    fn several_enumerated_shapes_are_refused_where_they_abort() {
        // The `convert_bge_small.py --enumerated-shapes` negative control.
        let inputs = [
            input("attention_mask", enumerated(&[128, 256, 512])),
            input("input_ids", enumerated(&[128, 256, 512])),
        ];
        let ShapeVerdict::Refuse(reason) = shape_verdict(&inputs, true) else {
            panic!("expected a refusal");
        };
        assert!(reason.contains("inputs `attention_mask`, `input_ids` accept"), "{reason}");
        assert!(reason.contains("[1, 128], [1, 256], [1, 512]"), "{reason}");
        assert!(reason.contains("macOS 27"), "{reason}");
        assert!(reason.contains("static-shape"), "{reason}");
    }

    #[test]
    fn several_enumerated_shapes_only_warn_before_macos_27() {
        let inputs = [input("input_ids", enumerated(&[128, 256]))];
        let ShapeVerdict::Warn(reason) = shape_verdict(&inputs, false) else {
            panic!("expected a warning");
        };
        assert!(reason.contains("input `input_ids` takes"), "{reason}");
        assert!(reason.contains("macOS 27 and later refuse"), "{reason}");
    }

    #[test]
    fn only_the_multi_shape_input_is_named() {
        let inputs = [input("attention_mask", enumerated(&[128])), input("input_ids", enumerated(&[128, 256]))];
        let ShapeVerdict::Refuse(reason) = shape_verdict(&inputs, true) else {
            panic!("expected a refusal");
        };
        assert!(reason.contains("input `input_ids` accepts"), "{reason}");
        assert!(!reason.contains("attention_mask"), "{reason}");
    }

    #[test]
    fn ranges_warn_and_are_never_refused() {
        // `ct.RangeDim(1, 512)`: runs under every compute unit on macOS 27.
        let range = || ShapeConstraint::Range(vec![1..=1, 1..=512]);
        let inputs = [input("attention_mask", range()), input("input_ids", range())];
        for aborts in [false, true] {
            let ShapeVerdict::Warn(reason) = shape_verdict(&inputs, aborts) else {
                panic!("expected a warning");
            };
            assert!(reason.contains("[1..=1, 1..=512]"), "{reason}");
            assert!(!reason.contains("refuse"), "{reason}");
        }
    }

    #[test]
    fn single_size_ranges_are_static() {
        let inputs = [input("input_ids", ShapeConstraint::Range(vec![1..=1, 128..=128]))];
        assert!(!inputs[0].constraint.is_flexible());
        assert_eq!(shape_verdict(&inputs, true), ShapeVerdict::Static);
    }

    #[test]
    fn nothing_assigned_fails() {
        let p = PlanSummary { unassigned: 10, ..Default::default() };
        assert_eq!(p.ane_share(), 0.0);
        assert!(p.verdict().is_err());
    }
}
