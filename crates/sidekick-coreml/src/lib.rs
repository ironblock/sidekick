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
pub use model::{CoremlModel, OutputTensor};
#[cfg(target_os = "macos")]
pub use plan::compute_plan;

use std::collections::BTreeMap;

/// Compute-unit preference. `CpuAndNeuralEngine` is sidekick's default: it
/// keeps background work off the GPU entirely, which is the point of the
/// project. Use `All` only when measuring.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ComputeUnits {
    All,
    #[default]
    CpuAndNeuralEngine,
    CpuAndGpu,
    CpuOnly,
}

/// A named int32 tensor input (shape is row-major, batch dim included).
#[derive(Debug)]
pub struct Int32Input<'a> {
    pub name: &'a str,
    pub shape: Vec<usize>,
    pub data: Vec<i32>,
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

    #[test]
    fn nothing_assigned_fails() {
        let p = PlanSummary { unassigned: 10, ..Default::default() };
        assert_eq!(p.ane_share(), 0.0);
        assert!(p.verdict().is_err());
    }
}
