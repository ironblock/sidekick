//! ANE check: the design doc calls silent CPU fallback "the failure mode of
//! this entire design", so measure it instead of assuming. Two signals:
//!
//! 1. **Eligibility (compute plan)** — the verdict. Core ML's compute plan
//!    says which device each operation is assigned to under
//!    `.cpuAndNeuralEngine`, without running the model, so it's unaffected
//!    by machine load. The model passes when compute-heavy operations
//!    (matmul, linear, conv, attention) are all on the ANE and at least
//!    80% of assigned operations are. Failing exits non-zero *before* any
//!    prediction: an artifact the ANE rejects can abort the process at
//!    predict time (flexible-shape models do on macOS 27).
//! 2. **Latency ratio** — runtime evidence. Median `.cpuOnly` over
//!    `.cpuAndNeuralEngine` latency. The plan is what the compiler intends
//!    and can't see failures that only happen at run time (a transient ANE
//!    compile failure); a ratio near 1.0 would. The ratio is only a
//!    warning: it swings with GPU and memory load, and a faster CPU path
//!    shrinks it on a model that is fully on the ANE (bge-small's 512 bucket
//!    measures ~1.25x on macOS 27 with the same plan as its 3x buckets).
//!
//! ```sh
//! cargo run -p sidekick-coreml --example ane_check -- \
//!     "$HOME/Library/Application Support/sidekick/models/bge-small-en-v1.5/model_128.mlmodelc" 128 \
//!     input_ids attention_mask embedding
//! ```
//!
//! Args after the path: seq_len (default 256), input_ids name,
//! attention_mask name, output name (defaults match the bge manifest).

#[cfg(target_os = "macos")]
fn main() {
    use sidekick_coreml::{compute_plan, ComputeUnits, CoremlModel, Int32Input};
    use std::time::Instant;

    let mut args = std::env::args().skip(1);
    let path = std::path::PathBuf::from(
        args.next().expect("usage: ane_check <model.mlmodelc> [seq_len] [ids] [mask] [output]"),
    );
    let seq_len: usize = args.next().map(|s| s.parse().expect("seq_len")).unwrap_or(256);
    let ids_name = args.next().unwrap_or_else(|| "input_ids".into());
    let mask_name = args.next().unwrap_or_else(|| "attention_mask".into());
    let output_name = args.next().unwrap_or_else(|| "last_hidden_state".into());

    println!("model: {} (seq_len {seq_len})", path.display());

    // 1. Eligibility from the compute plan.
    let plan = match compute_plan(&path, ComputeUnits::CpuAndNeuralEngine) {
        Ok(plan) => plan,
        Err(e) => {
            eprintln!("FAIL: could not read the compute plan: {e}");
            std::process::exit(1);
        }
    };
    println!(
        "compute plan: ANE {} / CPU {} / GPU {} of {} assigned ops ({:.1}% on ANE), {} unassigned",
        plan.ane,
        plan.cpu,
        plan.gpu,
        plan.assigned(),
        plan.ane_share() * 100.0,
        plan.unassigned
    );
    if !plan.off_ane_ops.is_empty() {
        let ops: Vec<String> = plan.off_ane_ops.iter().map(|(op, n)| format!("{op}x{n}")).collect();
        println!("off-ANE ops: {}", ops.join(", "));
    }
    if let Err(reason) = plan.verdict() {
        eprintln!("FAIL: not ANE-eligible: {reason}");
        std::process::exit(1);
    }
    println!("ANE eligibility (compute plan): OK");

    // 2. Latency ratio as runtime evidence.
    // Deterministic pseudo-token ids: content doesn't matter for latency,
    // but keep them in a small-vocab-safe range and identical across runs.
    let ids: Vec<i32> = (0..seq_len).map(|i| 1000 + (i as i32 * 7) % 20000).collect();
    let mask = vec![1i32; seq_len];
    let measure = |units: ComputeUnits| -> f64 {
        let model = CoremlModel::load(&path, units).expect("model load");
        let inputs = [
            Int32Input { name: &ids_name, shape: vec![1, seq_len], data: ids.clone() },
            Int32Input { name: &mask_name, shape: vec![1, seq_len], data: mask.clone() },
        ];
        // Warmup: first predictions include plan compilation / ANE program load.
        for _ in 0..3 {
            model.predict_int32(&inputs, &output_name).expect("warmup predict");
        }
        let mut samples: Vec<f64> = (0..20)
            .map(|_| {
                let t = Instant::now();
                model.predict_int32(&inputs, &output_name).expect("predict");
                t.elapsed().as_secs_f64() * 1000.0
            })
            .collect();
        samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
        samples[samples.len() / 2]
    };
    let cpu_ms = measure(ComputeUnits::CpuOnly);
    let ane_ms = measure(ComputeUnits::CpuAndNeuralEngine);
    let ratio = cpu_ms / ane_ms;
    println!("cpuOnly median:             {cpu_ms:8.2} ms");
    println!("cpuAndNeuralEngine median:  {ane_ms:8.2} ms");
    println!("speedup ratio:              {ratio:8.2}x");
    if ratio < 1.1 {
        eprintln!(
            "WARN: the ANE path is no faster than the CPU although the plan targets the ANE; \
             check for runtime ANE compile failures (stderr) and re-measure on a quiet machine"
        );
    }
}

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("ane_check only runs on macOS");
    std::process::exit(1);
}
