//! Latency and memory of a Core ML embedder's long buckets through
//! sidekick's own path (`CoremlEmbedder`, the daemon's), one input at a time.
//!
//! One process measures one model on one compute-unit setting, in the
//! buckets given (each bucket loads on first use and stays resident, as in
//! the daemon). Per bucket it records the time of the first prediction,
//! which includes Core ML's load (and, with an empty Core ML cache, its
//! compile for the device), then the median (of an even count, the mean of
//! the middle two) and 90th-percentile latency of `--reps` predictions after
//! two warm-ups, at each of `--lengths` that fits the bucket, and the
//! process's footprint (macOS `phys_footprint`, and its lifetime peak) and
//! resident size before loading and after each bucket. Inputs are real
//! token ids: the longest case of a parity reference (`reference-long.json`),
//! cut to each length keeping its final token, which last-token pooling
//! reads. A failed prediction is counted, with its error, rather than
//! ending the run. One JSON object goes to stdout.
//!
//! Core ML keeps its compiled bundles in a cache under the user's home
//! (`~/Library/Caches/<executable>`). To time a cold compile, run with
//! `CFFIXED_USER_HOME` set to an empty directory; run again with the same
//! directory to time a cached load; delete it afterwards. Measured on
//! Qwen3-Embedding-0.6B, the cache held 3–4 times each bucket's weights.
//!
//! ```sh
//! cargo run --release -p sidekick-embed --features coreml --example embed_timing -- \
//!     --models-dir <dir> --model qwen3-embedding-0.6b --units cpu_and_gpu \
//!     --reference <refs>/qwen3-embedding-0.6b/reference-long.json \
//!     [--buckets 1024,2048,4096] [--lengths 1024,2048,4096] [--reps 10]
//! ```

#[cfg(not(all(target_os = "macos", feature = "coreml")))]
fn main() {
    eprintln!("embed_timing needs macOS and --features coreml");
    std::process::exit(2);
}

#[cfg(all(target_os = "macos", feature = "coreml"))]
fn main() {
    use sidekick_core::manifest::{ModelRegistry, ScanOptions};
    use sidekick_core::ComputeUnits;
    use sidekick_embed::CoremlEmbedder;
    use std::time::Instant;

    /// (footprint, peak footprint, resident size) of this process, MiB. The
    /// footprint leaves out clean file-backed pages, such as weights Core ML
    /// maps from the artifact; the resident size counts them.
    fn footprint() -> (f64, f64, f64) {
        // rusage_info_v4 (<sys/resource.h>): the fields up to the two read here.
        #[repr(C)]
        struct RusageInfoV4 {
            uuid: [u8; 16],
            fields: [u64; 40],
        }
        extern "C" {
            fn proc_pid_rusage(pid: i32, flavor: i32, buffer: *mut RusageInfoV4) -> i32;
        }
        // ri_resident_size is the 7th u64 after the uuid, ri_phys_footprint
        // the 8th, ri_lifetime_max_phys_footprint the 29th.
        let mut info = RusageInfoV4 { uuid: [0; 16], fields: [0; 40] };
        let rc = unsafe { proc_pid_rusage(std::process::id() as i32, 4, &mut info) };
        assert_eq!(rc, 0, "proc_pid_rusage failed");
        let mib = |b: u64| b as f64 / (1 << 20) as f64;
        (mib(info.fields[7]), mib(info.fields[28]), mib(info.fields[6]))
    }

    fn cut(ids: &[i32], n: usize) -> Vec<i32> {
        if ids.len() <= n {
            return ids.to_vec();
        }
        let mut v = ids[..n - 1].to_vec();
        v.push(*ids.last().unwrap());
        v
    }

    /// A comma-separated list of positive numbers, or exit with a usage error.
    fn list(flag: &str, s: &str) -> Vec<usize> {
        let parsed: Option<Vec<usize>> =
            s.split(',').map(|n| n.trim().parse().ok().filter(|&n: &usize| n > 0)).collect();
        parsed.unwrap_or_else(|| {
            eprintln!("{flag} takes a comma-separated list of positive numbers, not {s:?}");
            std::process::exit(2);
        })
    }

    let mut args = std::env::args().skip(1);
    let (mut dir, mut id, mut units, mut reference) = (None, None, None, None);
    let (mut buckets, mut lengths, mut reps) = (None, vec![1024, 2048, 4096], 10usize);
    while let Some(a) = args.next() {
        let mut value = || args.next().unwrap_or_else(|| panic!("{a} needs a value"));
        match a.as_str() {
            "--models-dir" => dir = Some(std::path::PathBuf::from(value())),
            "--model" => id = Some(value()),
            "--units" => {
                units = Some(match value().as_str() {
                    "cpu_and_gpu" => ComputeUnits::CpuAndGpu,
                    "cpu_and_ne" => ComputeUnits::CpuAndNeuralEngine,
                    "cpu_only" => ComputeUnits::CpuOnly,
                    "all" => ComputeUnits::All,
                    other => panic!("unknown compute units {other}"),
                })
            }
            "--reference" => reference = Some(value()),
            "--buckets" => buckets = Some(list("--buckets", &value())),
            "--lengths" => lengths = list("--lengths", &value()),
            "--reps" => reps = value().parse().expect("--reps"),
            other => panic!("unknown argument {other}"),
        }
    }
    let (dir, id, reference) = (
        dir.expect("--models-dir"),
        id.expect("--model"),
        reference.expect("--reference"),
    );
    let reference: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&reference).expect("reference")).expect("reference json");
    let longest: Vec<i32> = reference["cases"]
        .as_array()
        .expect("cases")
        .iter()
        .map(|c| c["ids"].as_array().unwrap().iter().map(|x| x.as_i64().unwrap() as i32).collect::<Vec<i32>>())
        .max_by_key(Vec::len)
        .expect("a case");

    let before = footprint();
    // Like the parity suite: measure whatever the manifest says, past the caps.
    #[allow(clippy::needless_update)]
    let options =
        ScanOptions { ignore_ane_weight_cap: true, ignore_cpu_seq_cap: true, ignore_ane_seq_cap: true, ..Default::default() };
    let registry = ModelRegistry::scan_with(&dir, &options).expect("scan");
    let model = registry.get(&id).unwrap_or_else(|_| panic!("{id} not in {}: {:?}", dir.display(), registry.skipped()));
    let units = units.unwrap_or(model.manifest.compute_units.unwrap_or_default());
    let t = Instant::now();
    let emb = CoremlEmbedder::load_with(model, units).expect("load");
    let check_s = t.elapsed().as_secs_f64();
    let buckets = buckets.unwrap_or_else(|| emb.buckets().to_vec());

    let mut out = Vec::new();
    for &bucket in &buckets {
        let fits: Vec<usize> = lengths.iter().copied().filter(|&n| n <= bucket && n <= longest.len()).collect();
        let Some(&first_len) = fits.last() else { continue };
        // A failed prediction is recorded, not fatal: how often a bucket fails
        // is itself a result.
        let t = Instant::now();
        let first = emb.run(&cut(&longest, first_len), bucket);
        let first_s = t.elapsed().as_secs_f64();
        let first_error = first.err().map(|e| e.to_string());
        if let Some(e) = &first_error {
            eprintln!("{id} {units:?} bucket {bucket}: first prediction failed after {first_s:.1} s: {e}");
        }
        let loaded = footprint();
        let mut per_length = Vec::new();
        for &n in &fits {
            let ids = cut(&longest, n);
            let (mut ms, mut failed, mut error) = (Vec::new(), 0usize, None);
            for i in 0..reps + 2 {
                let t = Instant::now();
                match emb.run(&ids, bucket) {
                    // the first two are warm-ups
                    Ok(_) if i >= 2 => ms.push(t.elapsed().as_secs_f64() * 1e3),
                    Ok(_) => {}
                    Err(e) => {
                        failed += 1;
                        error.get_or_insert_with(|| e.to_string());
                    }
                }
            }
            ms.sort_by(f64::total_cmp);
            let median = match ms.len() {
                0 => None,
                n if n % 2 == 1 => Some(ms[n / 2]),
                n => Some((ms[n / 2 - 1] + ms[n / 2]) / 2.0),
            };
            let p90 = ms.get((ms.len() * 9 / 10).min(ms.len().saturating_sub(1))).copied();
            eprintln!(
                "{id} {units:?} bucket {bucket}, {n} tokens: median {median:.1?} ms, p90 {p90:.1?} ms, \
                 {failed} of {} predictions failed",
                reps + 2
            );
            per_length.push(serde_json::json!({
                "tokens": ids.len(), "median_ms": median, "p90_ms": p90, "timed": ms.len(),
                "failed": failed, "attempts": reps + 2, "error": error,
            }));
        }
        let after = footprint();
        eprintln!(
            "{id} {units:?} bucket {bucket}: first prediction {first_s:.1} s, footprint {:.0} MiB (peak {:.0}), \
             resident {:.0} MiB",
            after.0, after.1, after.2
        );
        out.push(serde_json::json!({
            "bucket": bucket, "first_s": first_s, "first_error": first_error, "footprint_after_first_mib": loaded.0,
            "footprint_mib": after.0, "peak_footprint_mib": after.1, "resident_mib": after.2,
            "latency": per_length,
        }));
    }
    let load_avg = std::process::Command::new("sysctl")
        .args(["-n", "vm.loadavg"])
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_default();
    println!(
        "{}",
        serde_json::json!({
            "model": id, "units": format!("{units:?}"), "reps": reps, "check_s": check_s,
            "footprint_before_mib": before.0, "resident_before_mib": before.2, "buckets": out, "load_avg_after": load_avg,
        })
    );
}
