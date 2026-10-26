//! Drive a `kv-state-1` decoder (a `chat.toml` model directory) through a
//! job of measurements, for grading a build and checking the runtime:
//!
//! ```sh
//! cargo run --release -p sidekick-coreml --example kv_decoder -- \
//!     <model dir> <job.json> <out prefix> [--no-backings] [--hidden-input NAME]
//! ```
//!
//! The job is `{"tasks": [...]}`; each task's result goes into
//! `<out prefix>.json`:
//! - `{"type": "tbench", "positions": [128, 1024], "reps": 20}`: one call's
//!   time with all T tokens real, per position (median, p90);
//! - `{"type": "greedy", "name": …, "ids": [...], "n": 96}`: the prompt in
//!   T-token calls, then `n` greedy tokens, one per call;
//! - `{"type": "trace", "name": …, "ids": [...]}`: teacher forcing, one token
//!   per call from position 0; every call's logits row goes to
//!   `<out prefix>_<name>.f32` (fp32, row-major);
//! - `{"type": "gen", "name": …, "ids": [...], "n": 128}`: a served-style
//!   generation, timed (time to first token, decode tokens per second);
//! - `{"type": "fresh", "reps": 8}`: what a fresh KV state costs: the time
//!   to replace it, and a T-token call on it against one on a reused state;
//! - `{"type": "soak", "calls": 10000, "every": 500}`: one-token calls at
//!   cycling positions, sampling the process's physical and Neural Engine
//!   footprints every `every` calls, with the decoder's stall count.
//!
//! `--no-backings` runs without output backings; `--hidden-input` names the
//! residual stream's input for a build that predates the contract's
//! `hidden_in` (#173's spike builds call it `hidden`). `--plan` first prints
//! each chunk's compute plan under `cpu_and_ne`; reading a plan compiles the
//! program again and keeps another copy of its weights in Core ML's cache.
//!
//! An ANE compile of a large build adds about its weights again to a system
//! cache that can't be deleted, and the cache key includes this binary's
//! code-signing identifier (docs/design/coreml-cache.md). Sign the example
//! with a fixed identifier after each build so a rebuild reuses the compile:
//! `codesign -s - -f -i <identifier> target/release/examples/kv_decoder`.

#[cfg(target_os = "macos")]
fn main() {
    use serde_json::{json, Value};
    use sidekick_core::chat_manifest::{ChatManifest, CHAT_MANIFEST};
    use sidekick_coreml::{process_memory, DecoderOptions, KvDecoder, ProgramNames};
    use std::io::Write;
    use std::path::PathBuf;
    use std::time::Instant;

    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let mut options = DecoderOptions::default();
    if let Some(i) = args.iter().position(|a| a == "--no-backings") {
        args.remove(i);
        options.output_backings = false;
    }
    let plan = args
        .iter()
        .position(|a| a == "--plan")
        .map(|i| args.remove(i))
        .is_some();
    if let Some(i) = args.iter().position(|a| a == "--hidden-input") {
        args.remove(i);
        options.names = ProgramNames {
            hidden_in: args.remove(i),
        };
    }
    let [dir, job, out] = <[String; 3]>::try_from(args)
        .expect("usage: kv_decoder <model dir> <job.json> <out prefix> [--no-backings] [--hidden-input NAME]");
    let dir = PathBuf::from(dir);
    let job: Value =
        serde_json::from_str(&std::fs::read_to_string(&job).expect("job")).expect("job json");

    let ms = |t: Instant| t.elapsed().as_secs_f64() * 1e3;
    let median = |x: &mut Vec<f64>| {
        x.sort_by(|a, b| a.total_cmp(b));
        x[x.len() / 2]
    };
    let pct = |x: &[f64], p: f64| x[((x.len() as f64 * p) as usize).min(x.len() - 1)];
    let argmax = |x: &[f32]| {
        x.iter()
            .enumerate()
            .fold(0, |b, (i, v)| if *v > x[b] { i } else { b }) as u32
    };
    let ids = |t: &Value| -> Vec<u32> {
        t["ids"]
            .as_array()
            .expect("ids")
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect()
    };

    let manifest = ChatManifest::read(&dir.join(CHAT_MANIFEST)).expect("chat.toml");
    if plan {
        for file in manifest.decoder.chunk_files() {
            let summary = sidekick_coreml::compute_plan(
                &dir.join(&file),
                sidekick_coreml::ComputeUnits::CpuAndNeuralEngine,
            );
            println!("{file}: {summary:?}");
        }
    }
    let before = process_memory();
    let t = Instant::now();
    let mut decoder = KvDecoder::load_with(&dir, &manifest, options).expect("load");
    let load_ms = ms(t);
    println!(
        "loaded {} in {load_ms:.0} ms (output backings: {})",
        decoder.id(),
        decoder.output_backings()
    );
    let shape = decoder.shape().clone();
    let t_tokens = shape.tokens_per_call;

    println!(
        "ANE share measured at load: {:?} bytes",
        decoder.ane_bytes()
    );
    let t = Instant::now();
    decoder.step(&[1], 0).expect("first call");
    let first_call_ms = ms(t);
    let loaded = process_memory();
    println!("first call {first_call_ms:.0} ms; memory {loaded:?}");

    let mut results = vec![];
    for task in job["tasks"].as_array().expect("tasks") {
        let kind = task["type"].as_str().expect("type");
        let name = task["name"].as_str().unwrap_or(kind).to_string();
        let started = Instant::now();
        let mut r = json!({"type": kind, "name": name});
        match kind {
            "tbench" => {
                let reps = task["reps"].as_u64().unwrap_or(20) as usize;
                let tokens: Vec<u32> = (0..t_tokens as u32).map(|i| 100 + (i * 37) % 900).collect();
                let mut per = serde_json::Map::new();
                for p in task["positions"].as_array().expect("positions") {
                    let p = p.as_u64().unwrap() as usize;
                    if p + t_tokens > shape.context_length {
                        continue;
                    }
                    decoder.step(&tokens, p).expect("call");
                    let mut times: Vec<f64> = (0..reps)
                        .map(|_| {
                            let t = Instant::now();
                            decoder.step(&tokens, p).expect("call");
                            ms(t)
                        })
                        .collect();
                    let m = median(&mut times);
                    per.insert(
                        p.to_string(),
                        json!({"ms": m, "p90_ms": pct(&times, 0.9),
                        "prompt_tok_s": t_tokens as f64 / m * 1e3, "decode_tok_s": 1e3 / m}),
                    );
                }
                r["tbench"] = per.into();
            }
            "greedy" => {
                let prompt = ids(task);
                let n = task["n"].as_u64().unwrap() as usize;
                let mut tokens = vec![argmax(decoder.prefill(&prompt, 0).expect("prefill"))];
                let mut pos = prompt.len();
                while tokens.len() < n {
                    tokens.push(argmax(
                        decoder.step(&[*tokens.last().unwrap()], pos).expect("step"),
                    ));
                    pos += 1;
                }
                r["tokens"] = json!(tokens);
            }
            "trace" => {
                let tokens = ids(task);
                let path = format!("{out}_{name}.f32");
                let mut file =
                    std::io::BufWriter::new(std::fs::File::create(&path).expect("trace file"));
                for (pos, &tok) in tokens.iter().enumerate() {
                    let row = decoder.step(&[tok], pos).expect("step");
                    for v in row {
                        file.write_all(&v.to_le_bytes()).unwrap();
                    }
                }
                r["rows"] = json!(tokens.len());
                r["vocab"] = json!(shape.vocab_size);
            }
            "gen" => {
                let prompt = ids(task);
                let n = task["n"].as_u64().unwrap() as usize;
                let t1 = Instant::now();
                let mut tok = argmax(decoder.prefill(&prompt, 0).expect("prefill"));
                let ttft = ms(t1);
                let t2 = Instant::now();
                for pos in prompt.len()..prompt.len() + n {
                    tok = argmax(decoder.step(&[tok], pos).expect("step"));
                }
                r["prompt_tokens"] = json!(prompt.len());
                r["ttft_ms"] = json!(ttft);
                r["decode_tok_s"] = json!(n as f64 / t2.elapsed().as_secs_f64());
            }
            "fresh" => {
                let reps = task["reps"].as_u64().unwrap_or(8) as usize;
                let tokens: Vec<u32> = (0..t_tokens as u32).map(|i| 100 + (i * 37) % 900).collect();
                let (mut reset, mut fresh, mut reused) = (vec![], vec![], vec![]);
                for _ in 0..reps {
                    let t = Instant::now();
                    decoder.reset_state();
                    reset.push(ms(t));
                    let t = Instant::now();
                    decoder.step(&tokens, 0).expect("call");
                    fresh.push(ms(t));
                    let t = Instant::now();
                    decoder.step(&tokens, 0).expect("call");
                    reused.push(ms(t));
                }
                r["reset_ms"] = json!(median(&mut reset));
                r["fresh_call_ms"] = json!(median(&mut fresh));
                r["reused_call_ms"] = json!(median(&mut reused));
            }
            "soak" => {
                let calls = task["calls"].as_u64().unwrap_or(10_000) as usize;
                let every = task["every"].as_u64().unwrap_or(500) as usize;
                let positions = shape.cache_capacity - t_tokens + 1;
                let mut samples = vec![];
                let mut times = Vec::with_capacity(calls);
                for i in 0..calls {
                    if i % every == 0 {
                        let m = process_memory().unwrap();
                        samples.push(json!({"call": i, "phys_footprint": m.phys_footprint,
                            "neural_footprint": m.neural_footprint}));
                    }
                    let t = Instant::now();
                    decoder
                        .step(&[(100 + i % 900) as u32], i % positions)
                        .expect("step");
                    times.push(ms(t));
                }
                let m = process_memory().unwrap();
                samples.push(json!({"call": calls, "phys_footprint": m.phys_footprint,
                    "neural_footprint": m.neural_footprint}));
                let mut sorted = times.clone();
                let med = median(&mut sorted);
                r["samples"] = json!(samples);
                r["median_ms"] = json!(med);
                r["p99_ms"] = json!(pct(&sorted, 0.99));
                r["max_ms"] = json!(sorted.last());
            }
            other => panic!("unknown task {other}"),
        }
        r["wall_s"] = json!(started.elapsed().as_secs_f64());
        println!(
            "task {name} done in {:.1} s",
            started.elapsed().as_secs_f64()
        );
        results.push(r);
    }
    let health = decoder.health();
    let summary = json!({
        "load_ms": load_ms,
        "first_call_ms": first_call_ms,
        "output_backings": decoder.output_backings(),
        "ane_bytes": decoder.ane_bytes(),
        "memory_before_load": before.map(|m| json!({"phys": m.phys_footprint, "neural": m.neural_footprint})),
        "memory_loaded": loaded.map(|m| json!({"phys": m.phys_footprint, "neural": m.neural_footprint})),
        "health": {"calls": health.calls, "stalls": health.stalls, "warm_median_ms": health.warm_median_ms,
                   "state_resets": health.state_resets, "failed": health.failed},
        "tasks": results,
    });
    std::fs::write(
        format!("{out}.json"),
        serde_json::to_string_pretty(&summary).unwrap(),
    )
    .expect("write results");
    println!(
        "done: {} calls, {} stalls, warm median {:?} ms",
        health.calls, health.stalls, health.warm_median_ms
    );
}

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("kv_decoder: Core ML is macOS only");
}
