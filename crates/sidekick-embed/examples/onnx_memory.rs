//! What an onnx model's memory does around a long input, on sidekick's own
//! path (`OnnxEmbedder`, the daemon's), and what each way of holding the
//! CPU arena costs short inputs (D40).
//!
//! One process per run, one memory mode per process:
//! - `arena`: ONNX Runtime's default, the arena never shrunk;
//! - `shrink-long`: sidekick's default, the arena shrunk after a run of at
//!   least `SHRINK_FROM_TOKENS` tokens;
//! - `shrink-always`: shrunk after every run;
//! - `no-arena`: no CPU arena at all.
//!
//! It records the process's footprint (macOS `phys_footprint` and its
//! lifetime peak; Linux RSS and its peak) before loading, after loading,
//! after a short input, after one at `--mid` tokens, after one at
//! `--long` tokens, after another short input, and after 5 and 30 seconds
//! idle; then the median latency of `--runs` warm passes at each of
//! `--lengths`, and the footprint after those. Inputs are real token
//! ids: the longest case of a parity reference (`reference-long.json` for
//! lengths past 512), cut to each length keeping its final token, as
//! tools/bench_onnx.py cuts them. One JSON object goes to stdout.
//!
//! ```sh
//! ORT_DYLIB_PATH=.../libonnxruntime.dylib cargo run --release -p sidekick-embed \
//!     --features onnx --example onnx_memory -- <models dir> <model id> <reference.json> \
//!     <mode> [--threads 8] [--mid 512] [--long 4096] \
//!     [--lengths 16,256,512] [--runs 20]
//! ```

use sidekick_core::manifest::ModelRegistry;
use sidekick_embed::onnx::{MemoryOptions, OnnxEmbedder, SHRINK_FROM_TOKENS};
use std::time::Instant;

/// (footprint, peak footprint) of this process, bytes.
#[cfg(target_os = "macos")]
fn footprint() -> (u64, u64) {
    // rusage_info_v4 (<sys/resource.h>): the fields up to the two read here.
    #[repr(C)]
    struct RusageInfoV4 {
        uuid: [u8; 16],
        fields: [u64; 40],
    }
    extern "C" {
        fn proc_pid_rusage(pid: i32, flavor: i32, buffer: *mut RusageInfoV4) -> i32;
    }
    // ri_phys_footprint is the 8th u64 after the uuid,
    // ri_lifetime_max_phys_footprint the 29th.
    let mut info = RusageInfoV4 { uuid: [0; 16], fields: [0; 40] };
    let rc = unsafe { proc_pid_rusage(std::process::id() as i32, 4, &mut info) };
    assert_eq!(rc, 0, "proc_pid_rusage failed");
    (info.fields[7], info.fields[28])
}

/// (RSS, peak RSS) of this process, bytes.
#[cfg(not(target_os = "macos"))]
fn footprint() -> (u64, u64) {
    let status = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
    let kb = |key: &str| {
        status
            .lines()
            .find_map(|l| l.strip_prefix(key))
            .and_then(|v| v.split_whitespace().next()?.parse::<u64>().ok())
            .unwrap_or(0)
            * 1024
    };
    (kb("VmRSS:"), kb("VmHWM:"))
}

fn mb(bytes: u64) -> f64 {
    bytes as f64 / (1 << 20) as f64
}

fn cut(ids: &[i32], n: usize) -> Vec<i32> {
    if ids.len() <= n {
        return ids.to_vec();
    }
    let mut v = ids[..n - 1].to_vec();
    v.push(*ids.last().unwrap());
    v
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let usage = "usage: onnx_memory <models dir> <model id> <reference.json> \
                 <mode> [--threads N] [--mid N] [--long N] \
                 [--lengths a,b,c] [--runs N]";
    let [dir, id, reference, mode, rest @ ..] = &args[..] else {
        eprintln!("{usage}");
        std::process::exit(2);
    };
    let (mut threads, mut mid, mut long, mut lengths, mut runs) = (8usize, 512usize, 4096usize, vec![16, 256, 512], 20);
    let mut it = rest.iter();
    while let Some(flag) = it.next() {
        let value = it.next().unwrap_or_else(|| panic!("{flag} needs a value\n{usage}"));
        match flag.as_str() {
            "--threads" => threads = value.parse().expect("--threads"),
            "--mid" => mid = value.parse().expect("--mid"),
            "--long" => long = value.parse().expect("--long"),
            "--lengths" => lengths = value.split(',').map(|n| n.parse().expect("--lengths")).collect(),
            "--runs" => runs = value.parse().expect("--runs"),
            _ => panic!("unknown {flag}\n{usage}"),
        }
    }
    let memory = match mode.as_str() {
        "arena" => MemoryOptions { cpu_arena: true, shrink_from_tokens: None },
        "shrink-long" => MemoryOptions::default(),
        "shrink-always" => MemoryOptions { cpu_arena: true, shrink_from_tokens: Some(0) },
        "no-arena" => MemoryOptions { cpu_arena: false, shrink_from_tokens: None },
        _ => panic!("unknown mode {mode}\n{usage}"),
    };
    let reference: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(reference).expect("reference")).expect("reference json");
    let longest: Vec<i32> = reference["cases"]
        .as_array()
        .expect("cases")
        .iter()
        .map(|c| c["ids"].as_array().unwrap().iter().map(|x| x.as_i64().unwrap() as i32).collect::<Vec<i32>>())
        .max_by_key(Vec::len)
        .expect("a case");
    assert!(longest.len() >= long, "the reference's longest case has {} tokens, under --long {long}", longest.len());

    let runtime = sidekick_embed::onnx::load_runtime(None).expect("runtime").version;
    let mut points = serde_json::Map::new();
    let mut mark = |name: &str| {
        let (now, peak) = footprint();
        points.insert(name.into(), serde_json::json!({"mb": mb(now), "peak_mb": mb(peak)}));
    };
    mark("before_load");
    let registry = ModelRegistry::scan(std::path::Path::new(dir)).expect("scan");
    let model = registry.get(id).unwrap_or_else(|_| panic!("{id} not in {dir}: {:?}", registry.skipped()));
    let emb = OnnxEmbedder::load_with(model, threads, memory).expect("load");
    mark("loaded");
    let short = cut(&longest, 16);
    emb.run(&short).unwrap();
    mark("after_short");
    emb.run(&cut(&longest, mid)).unwrap();
    mark("after_mid");
    let t = Instant::now();
    emb.run(&cut(&longest, long)).unwrap();
    let long_ms = t.elapsed().as_secs_f64() * 1e3;
    mark("after_long");
    emb.run(&short).unwrap();
    mark("after_short_again");
    // Idle, as a daemon is between requests: the system allocator may hand
    // freed pages back lazily.
    for (name, secs) in [("idle_5s", 5), ("idle_30s", 25)] {
        std::thread::sleep(std::time::Duration::from_secs(secs));
        mark(name);
    }

    let mut latency = serde_json::Map::new();
    for &n in &lengths {
        let ids = cut(&longest, n);
        for _ in 0..3 {
            emb.run(&ids).unwrap();
        }
        let mut ms: Vec<f64> = (0..runs)
            .map(|_| {
                let t = Instant::now();
                emb.run(&ids).unwrap();
                t.elapsed().as_secs_f64() * 1e3
            })
            .collect();
        ms.sort_by(f64::total_cmp);
        latency.insert(ids.len().to_string(), serde_json::json!(ms[ms.len() / 2]));
    }
    mark("after_latency");
    let load_avg = std::process::Command::new("sysctl")
        .args(["-n", "vm.loadavg"])
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_default();
    println!(
        "{}",
        serde_json::json!({
            "model": id, "mode": mode, "shrink_from_tokens": SHRINK_FROM_TOKENS, "threads": threads,
            "onnx_runtime": runtime, "mid": mid, "long": long, "long_ms": long_ms, "runs": runs,
            "footprint": points, "median_ms": latency, "load_avg_after": load_avg,
        })
    );
}
