//! ANE residency check: the design doc calls silent CPU fallback "the failure
//! mode of this entire design", so measure it instead of assuming. Loads the
//! same model twice — `.cpuOnly` and `.cpuAndNeuralEngine` — and compares
//! median prediction latency. A healthy ANE-resident encoder should be
//! severalfold faster; a ratio near 1.0 means the ANE plan fell back to CPU.
//!
//! ```sh
//! cargo run -p sidekick-coreml --example ane_check -- \
//!     "$HOME/Library/Application Support/sidekick/models/bge-small-en-v1.5/model.mlmodelc"
//! ```
//!
//! Optional args after the path: seq_len (default 256), input_ids name,
//! attention_mask name, output name (defaults match the bge manifest).

#[cfg(target_os = "macos")]
fn main() {
    use sidekick_coreml::{ComputeUnits, CoremlModel, Int32Input};
    use std::time::Instant;

    let mut args = std::env::args().skip(1);
    let path = std::path::PathBuf::from(
        args.next().expect("usage: ane_check <model.mlmodelc> [seq_len] [ids] [mask] [output]"),
    );
    let seq_len: usize = args.next().map(|s| s.parse().expect("seq_len")).unwrap_or(256);
    let ids_name = args.next().unwrap_or_else(|| "input_ids".into());
    let mask_name = args.next().unwrap_or_else(|| "attention_mask".into());
    let output_name = args.next().unwrap_or_else(|| "last_hidden_state".into());

    // Deterministic pseudo-token ids: content doesn't matter for latency,
    // but keep them in a small-vocab-safe range and identical across runs.
    let ids: Vec<i32> = (0..seq_len).map(|i| 1000 + (i as i32 * 7) % 20000).collect();
    let mask = vec![1i32; seq_len];

    let measure = |units: ComputeUnits| -> (f64, Vec<usize>) {
        let model = CoremlModel::load(&path, units).expect("model load");
        let inputs = [
            Int32Input { name: &ids_name, shape: vec![1, seq_len], data: ids.clone() },
            Int32Input { name: &mask_name, shape: vec![1, seq_len], data: mask.clone() },
        ];
        // Warmup: first predictions include plan compilation / ANE program load.
        let mut shape = Vec::new();
        for _ in 0..3 {
            shape = model.predict_int32(&inputs, &output_name).expect("warmup predict").shape;
        }
        let mut samples: Vec<f64> = (0..20)
            .map(|_| {
                let t = Instant::now();
                model.predict_int32(&inputs, &output_name).expect("predict");
                t.elapsed().as_secs_f64() * 1000.0
            })
            .collect();
        samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
        (samples[samples.len() / 2], shape)
    };

    let (cpu_ms, shape) = measure(ComputeUnits::CpuOnly);
    let (ane_ms, _) = measure(ComputeUnits::CpuAndNeuralEngine);
    let ratio = cpu_ms / ane_ms;

    println!("model: {} (output shape {:?}, seq_len {seq_len})", path.display(), shape);
    println!("cpuOnly median:             {cpu_ms:8.2} ms");
    println!("cpuAndNeuralEngine median:  {ane_ms:8.2} ms");
    println!("speedup ratio:              {ratio:8.2}x");
    if ratio < 1.5 {
        eprintln!("WARN: ratio < 1.5x — the model is likely NOT resident on the ANE");
        std::process::exit(2);
    }
    println!("ANE residency: OK (>= 1.5x over CPU)");
}

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("ane_check only runs on macOS");
    std::process::exit(1);
}
