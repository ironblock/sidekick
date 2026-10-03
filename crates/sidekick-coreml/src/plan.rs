//! Core ML compute plans: which device each operation of a compiled ML
//! program is assigned to, without running the model.

#![allow(unsafe_code)]

use crate::{ComputeUnits, PlanSummary};
use block2::RcBlock;
use objc2::rc::Retained;
use objc2::runtime::NSObjectProtocol;
use objc2::{available, ClassType};
use objc2_core_ml::{
    MLComputePlan, MLGPUComputeDevice, MLModelConfiguration, MLModelStructureProgramBlock,
    MLNeuralEngineComputeDevice,
};
use objc2_foundation::{NSError, NSString, NSURL};
use sidekick_core::{Error, Result};
use std::path::Path;
use std::sync::mpsc;
use std::time::Duration;

/// Loading a plan compiles the model for the requested devices; measured up
/// to ~18 s for a 300M-parameter encoder on a cold cache.
const PLAN_TIMEOUT: Duration = Duration::from_secs(300);

/// Summarize the compute plan of a compiled model (`.mlmodelc`) for the
/// given compute units. Requires macOS 14.4. Only ML programs have
/// per-operation plans; neural-network and pipeline models are an error.
///
/// A plan can come back empty (every operation unassigned) or fail with
/// "internal failure" while the artifact is fine: Core ML caches compiled
/// bundles per executable, keyed by artifact path, and a broken entry fails
/// every read of that path. A copy at another path reads normally. An empty
/// summary fails [`PlanSummary::verdict`] without the model being
/// ineligible, so re-read from a copy before concluding anything; the
/// parity suite does.
pub fn compute_plan(path: &Path, units: ComputeUnits) -> Result<PlanSummary> {
    if !available!(macos = 14.4) {
        return Err(Error::Inference("Core ML compute plans need macOS 14.4 or later".into()));
    }
    // Core ML aborts the process, with an uncaught C++ exception, when a
    // compiled model it's asked to plan isn't there: a model uninstalled or
    // reinstalled after its load. Checking first narrows that window to the
    // read itself.
    let compiled = path.extension().is_some_and(|e| e == "mlmodelc");
    if !path.exists() || (compiled && !path.join("coremldata.bin").is_file()) {
        return Err(Error::Inference(format!("{} is gone; its compute plan can't be read", path.display())));
    }
    let url = NSURL::fileURLWithPath(&NSString::from_str(&path.to_string_lossy()));
    let config = unsafe { MLModelConfiguration::new() };
    unsafe { config.setComputeUnits(crate::model::to_ml(units)) };

    // The plan is only valid inside the handler, and MLComputePlan isn't
    // Send, so the summary is built there and only plain data crosses back.
    // A panic must not unwind into Core ML, so it's caught and reported.
    let (tx, rx) = mpsc::channel::<Result<PlanSummary>>();
    let handler = RcBlock::new(move |plan: *mut MLComputePlan, error: *mut NSError| {
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            // SAFETY: Core ML passes either a valid plan or a valid error,
            // alive for the duration of this call.
            match (unsafe { plan.as_ref() }, unsafe { error.as_ref() }) {
                (Some(plan), _) => summarize(plan),
                (None, Some(error)) => Err(Error::Inference(format!(
                    "Core ML compute plan failed: {}",
                    error.localizedDescription()
                ))),
                (None, None) => Err(Error::Inference("Core ML returned no compute plan".into())),
            }
        }))
        .unwrap_or_else(|_| Err(Error::Inference("panic while reading the compute plan".into())));
        let _ = tx.send(result);
    });
    unsafe {
        MLComputePlan::loadContentsOfURL_configuration_completionHandler(&url, &config, &handler)
    };
    // The handler runs on a Core ML queue, never this thread, so waiting
    // here can't deadlock; the timeout bounds a plan that never arrives.
    rx.recv_timeout(PLAN_TIMEOUT).map_err(|_| {
        Error::Inference(format!("no compute plan for {} within {PLAN_TIMEOUT:?}", path.display()))
    })?
}

fn summarize(plan: &MLComputePlan) -> Result<PlanSummary> {
    let structure = unsafe { plan.modelStructure() };
    let program = unsafe { structure.program() }.ok_or_else(|| {
        Error::Inference(
            "not an ML program (neural network or pipeline): no per-operation compute plan".into(),
        )
    })?;
    let functions = unsafe { program.functions() };
    let main = functions.objectForKey(&NSString::from_str("main")).ok_or_else(|| {
        let names: Vec<String> = functions.allKeys().iter().map(|k| k.to_string()).collect();
        Error::Inference(format!("ML program has no `main` function (functions: {names:?})"))
    })?;
    let mut summary = PlanSummary::default();
    walk(plan, &unsafe { main.block() }, &mut summary);
    Ok(summary)
}

fn walk(plan: &MLComputePlan, block: &Retained<MLModelStructureProgramBlock>, summary: &mut PlanSummary) {
    for op in unsafe { block.operations() }.iter() {
        match unsafe { plan.computeDeviceUsageForMLProgramOperation(&op) } {
            None => summary.unassigned += 1,
            Some(usage) => {
                let device = unsafe { usage.preferredComputeDevice() };
                if device.isKindOfClass(MLNeuralEngineComputeDevice::class()) {
                    summary.ane += 1;
                } else {
                    if device.isKindOfClass(MLGPUComputeDevice::class()) {
                        summary.gpu += 1;
                    } else {
                        summary.cpu += 1;
                    }
                    let name = unsafe { op.operatorName() }.to_string();
                    // "ios18.matmul" -> "matmul"
                    let name = name.rsplit('.').next().unwrap_or(&name).to_string();
                    *summary.off_ane_ops.entry(name).or_default() += 1;
                }
            }
        }
        for nested in unsafe { op.blocks() }.iter() {
            walk(plan, &nested, summary);
        }
    }
}
