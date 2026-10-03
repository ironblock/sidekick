//! Latency of agentjev-format classifiers through sidekick's own Core ML
//! path, for comparing a chunked model (D37) with the same model unchunked,
//! and for a fixed workload while measuring power.
//!
//! Each request is a synthetic tree of exactly `--tokens` tokens: a shared
//! prefix and two two-token candidates. Per model and length it prints the
//! first-load time, then the median and 90th-percentile latency of `--reps`
//! predictions after a warm-up. With `--seconds`, it instead runs
//! predictions at the first length for that long, back to back or, with
//! `--rate`, at that many per second (each starting on schedule, or as soon
//! as the one before ends when it runs late), and prints the count and
//! their latency: the companion load for tools/companion_bench.py.
//!
//!     cargo run --release -p sidekick-embed --features coreml --example chain_timing -- \
//!         --models-dir <dir> --model agent-jev [--model ...] --units cpu_and_gpu \
//!         [--tokens 256,1024,2048] [--reps 20] [--seconds 60 [--rate 2]]
//!
//! Each `--model` is a classifier directory name under `--models-dir`; give
//! two directories holding the same model chunked and unchunked to measure
//! the boundary. `--units` overrides the manifests' compute units.

#[cfg(not(all(target_os = "macos", feature = "coreml")))]
fn main() {
    eprintln!("chain_timing needs macOS and --features coreml");
    std::process::exit(2);
}

#[cfg(all(target_os = "macos", feature = "coreml"))]
fn main() {
    use sidekick_core::manifest::{ModelRegistry, ScanOptions};
    use sidekick_core::{ComputeUnits, Prepared};
    use sidekick_embed::CoremlClassifier;
    use std::time::Instant;

    let mut args = std::env::args().skip(1);
    let (mut dir, mut models, mut units, mut tokens, mut reps, mut seconds) =
        (None, Vec::new(), None, vec![256, 1024, 2048], 20usize, None);
    let mut rate: Option<f64> = None;
    while let Some(a) = args.next() {
        let mut value = || args.next().unwrap_or_else(|| panic!("{a} needs a value"));
        match a.as_str() {
            "--models-dir" => dir = Some(std::path::PathBuf::from(value())),
            "--model" => models.push(value()),
            "--units" => {
                units = Some(match value().as_str() {
                    "cpu_and_gpu" => ComputeUnits::CpuAndGpu,
                    "cpu_and_ne" => ComputeUnits::CpuAndNeuralEngine,
                    "cpu_only" => ComputeUnits::CpuOnly,
                    "all" => ComputeUnits::All,
                    other => panic!("unknown compute units {other}"),
                })
            }
            "--tokens" => tokens = value().split(',').map(|t| t.parse().expect("token count")).collect(),
            "--reps" => reps = value().parse().expect("reps"),
            "--seconds" => seconds = Some(value().parse::<f64>().expect("seconds")),
            "--rate" => rate = Some(value().parse::<f64>().expect("requests per second")),
            other => panic!("unknown argument {other}"),
        }
    }
    let dir = dir.expect("--models-dir");
    // Like the parity suite: measure whatever the manifest says, past the caps.
    // Any other option keeps its default.
    #[allow(clippy::needless_update)]
    let options = ScanOptions { ignore_ane_weight_cap: true, ignore_cpu_seq_cap: true, ..Default::default() };
    let registry = ModelRegistry::scan_with(&dir, &options).expect("scan");
    for s in registry.skipped() {
        eprintln!("skipped {}: {}", s.path.display(), s.reason);
    }

    for id in &models {
        let model = registry.classifier(id).unwrap_or_else(|e| panic!("{id}: {e}"));
        let units = units.unwrap_or(model.manifest.compute_units);
        let t = Instant::now();
        let clf = CoremlClassifier::load_with(model, units).expect("load");
        println!("{id}: {} program(s) per bucket, {units:?}, loaded in {:.1}s", model.manifest.chunks(), t.elapsed().as_secs_f64());
        for &n in &tokens {
            let bucket = *model.manifest.buckets.iter().find(|&&b| b >= n).expect("a bucket fits");
            let p = n - 4;
            let ids: Vec<i32> = (0..n as i32).map(|i| 1000 + (i * 37) % 5000).collect();
            let mut seg = vec![0; p];
            seg.extend([1, 1, 2, 2]);
            let mut position_ids: Vec<i32> = (0..p as i32).collect();
            position_ids.extend([p as i32, p as i32 + 1, p as i32, p as i32 + 1]);
            let prepared = Prepared {
                ids,
                type_ids: vec![],
                markers: vec![p as i32 + 1, p as i32 + 3],
                qtype: None,
                decide_pos: None,
                seg,
                position_ids,
                bucket,
            };
            let t = Instant::now();
            clf.run_in(&prepared, bucket, &[]).expect("predict");
            let first = t.elapsed().as_secs_f64();
            if let Some(secs) = seconds {
                let start = Instant::now();
                let mut ms = Vec::new();
                while start.elapsed().as_secs_f64() < secs {
                    if let Some(rate) = rate {
                        let due = start + std::time::Duration::from_secs_f64(ms.len() as f64 / rate);
                        if let Some(wait) = due.checked_duration_since(Instant::now()) {
                            std::thread::sleep(wait);
                        }
                    }
                    let t = Instant::now();
                    clf.run_in(&prepared, bucket, &[]).expect("predict");
                    ms.push(t.elapsed().as_secs_f64() * 1e3);
                }
                let wall = start.elapsed().as_secs_f64();
                ms.sort_by(f64::total_cmp);
                let at = |q: f64| ms[((ms.len() as f64 * q) as usize).min(ms.len() - 1)];
                println!(
                    "{id} {n} tokens: {} predictions in {wall:.1}s ({:.2}/s); latency p50 {:.1} ms, p90 {:.1} ms, p99 {:.1} ms",
                    ms.len(),
                    ms.len() as f64 / wall,
                    at(0.5),
                    at(0.9),
                    at(0.99)
                );
                break;
            }
            for _ in 0..2 {
                clf.run_in(&prepared, bucket, &[]).expect("predict");
            }
            let mut ms: Vec<f64> = (0..reps)
                .map(|_| {
                    let t = Instant::now();
                    clf.run_in(&prepared, bucket, &[]).expect("predict");
                    t.elapsed().as_secs_f64() * 1e3
                })
                .collect();
            ms.sort_by(f64::total_cmp);
            println!(
                "{id} {n} tokens (bucket {bucket}): first {:.1}s, median {:.1} ms, p90 {:.1} ms",
                first,
                ms[ms.len() / 2],
                ms[(ms.len() * 9 / 10).min(ms.len() - 1)]
            );
        }
    }
}
