//! Which invariances the onnx backend's CPU path has (D40): whether ONNX
//! Runtime's CPU kernels give an input the same vector at any thread
//! count, at any padded length, in any batch, and on a re-run.
//!
//! It runs a parity reference's materialized cases (the token ids the
//! suite feeds) through an `onnx` model:
//! - each input alone, unpadded, on one thread: the baseline;
//! - the same again in the same session (repeatability);
//! - alone on 2, 4 and 8 threads;
//! - padded to longer lengths, alone;
//! - in batches of 2 and 8 (with the next cases), padded to the batch's
//!   longest.
//!
//! For each it reports how many vectors are bit-identical to the
//! baseline, the largest |Δ| of any component, and the lowest cosine.
//!
//! ```sh
//! ORT_DYLIB_PATH=.../libonnxruntime.dylib cargo run --release -p sidekick-embed \
//!     --features onnx --example onnx_invariance -- <models dir> <model id> <reference.json>
//! ```

use sidekick_core::manifest::ModelRegistry;
use sidekick_core::{EmbedLimits, EmbedPurpose};
use sidekick_embed::onnx::OnnxEmbedder;

struct Tally {
    n: usize,
    identical: usize,
    max_abs: f32,
    min_cos: f64,
}

impl Tally {
    fn new() -> Self {
        Self { n: 0, identical: 0, max_abs: 0.0, min_cos: 1.0 }
    }

    fn add(&mut self, a: &[f32], b: &[f32]) {
        self.n += 1;
        if a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits()) {
            self.identical += 1;
        }
        let d = a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max);
        self.max_abs = self.max_abs.max(d);
        let dot: f64 = a.iter().zip(b).map(|(x, y)| f64::from(*x) * f64::from(*y)).sum();
        let norm = |v: &[f32]| v.iter().map(|x| f64::from(*x).powi(2)).sum::<f64>().sqrt();
        self.min_cos = self.min_cos.min(dot / (norm(a) * norm(b)));
    }

    fn print(&self, what: &str) {
        println!(
            "  {what:<34} {:>4}/{:<4} bit-identical   max |Δ| {:.3e}   min cosine {:.9}",
            self.identical, self.n, self.max_abs, self.min_cos
        );
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [dir, id, reference] = &args[..] else {
        eprintln!("usage: onnx_invariance <models dir> <model id> <reference.json>");
        std::process::exit(2);
    };
    let registry = ModelRegistry::scan(std::path::Path::new(dir)).expect("scan");
    let model = registry.get(id).unwrap_or_else(|_| panic!("{id} not in {dir}: {:?}", registry.skipped()));
    let reference: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(reference).expect("reference")).unwrap();
    let max_len = model.manifest.max_seq_len;
    let one = OnnxEmbedder::load(model, 1).expect("load");
    // Another model's reference: its texts, through this model's tokenizer.
    let own = reference["model"]["id"].as_str() == Some(id.as_str());
    let cases: Vec<Vec<i32>> = reference["cases"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|c| {
            if own {
                return Some(c["ids"].as_array().unwrap().iter().map(|x| x.as_i64().unwrap() as i32).collect());
            }
            let purpose = if c["purpose"] == "query" { EmbedPurpose::Query } else { EmbedPurpose::Document };
            let text = c["text"].as_str().unwrap_or_default();
            (!text.is_empty()).then(|| one.prepare(text, purpose, EmbedLimits::default()).unwrap())
        })
        .collect();
    if !own {
        println!("  (cases are another model's reference texts, tokenized with this model's tokenizer)");
    }
    println!(
        "{id}: {} cases, {}..{} tokens; ONNX Runtime {}",
        cases.len(),
        cases.iter().map(Vec::len).min().unwrap(),
        cases.iter().map(Vec::len).max().unwrap(),
        sidekick_embed::onnx::runtime().expect("runtime").version
    );

    let started = std::time::Instant::now();
    let base: Vec<Vec<f32>> = cases.iter().map(|ids| one.run(ids).unwrap()).collect();
    println!("  baseline: each input alone on 1 thread, {:.1?}", started.elapsed());

    let mut t = Tally::new();
    for (ids, b) in cases.iter().zip(&base) {
        t.add(b, &one.run(ids).unwrap());
    }
    t.print("re-run, same session");

    for threads in [2, 4, 8] {
        let m = OnnxEmbedder::load(model, threads).expect("load");
        let mut t = Tally::new();
        for (ids, b) in cases.iter().zip(&base) {
            t.add(b, &m.run(ids).unwrap());
        }
        t.print(&format!("alone, {threads} threads"));
    }

    for (name, pad) in [("+1", 1usize), ("+7", 7), ("+64", 64)] {
        let mut t = Tally::new();
        for (ids, b) in cases.iter().zip(&base) {
            let len = ids.len() + pad;
            if len <= max_len {
                t.add(b, &one.run_batch(&[ids], len).unwrap().remove(0));
            }
        }
        t.print(&format!("alone, padded {name} tokens"));
    }
    let mut t = Tally::new();
    for (ids, b) in cases.iter().zip(&base) {
        if ids.len() < max_len {
            t.add(b, &one.run_batch(&[ids], max_len).unwrap().remove(0));
        }
    }
    t.print(&format!("alone, padded to {max_len}"));

    for size in [2usize, 8] {
        let mut t = Tally::new();
        for chunk in (0..cases.len()).collect::<Vec<_>>().chunks(size) {
            let rows: Vec<&[i32]> = chunk.iter().map(|&i| cases[i].as_slice()).collect();
            let len = rows.iter().map(|r| r.len()).max().unwrap();
            for (&i, v) in chunk.iter().zip(one.run_batch(&rows, len).unwrap()) {
                t.add(&base[i], &v);
            }
        }
        t.print(&format!("in batches of {size}, padded"));
    }
}
