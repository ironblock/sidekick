//! Measure guided decoding's host work on a model's real tokenizer, without
//! its decoder: each item's known answer is walked through its schema's
//! grammar as a model would write it, one sampled token per step, timing
//! every mask, and counting the tokens the grammar forces and the decoder
//! calls the reply would take.
//!
//! ```sh
//! cargo run --release -p sidekick-llm --example guided_masks -- \
//!     <model dir> fixtures/tasks/chat/schemas.json fixtures/tasks/chat/extraction.jsonl
//! ```
//!
//! The model directory holds `chat.toml` (its stop tokens, vocabulary and
//! T) and the tokenizer it names. Each item of the JSONL file names a
//! schema of the schemas file (`schema`) and holds its answer (`gold`),
//! which is written as a guided reply is: on one line, with `, ` and `: `
//! between values and its keys in schema order. At each
//! step the walk takes the longest token the mask allows that matches the
//! answer's next bytes. The answer's tokens are an upper bound on a real
//! reply's: a model may write longer tokens than the walk finds.

use serde_json::Value;
use sidekick_core::chat_manifest::ChatManifest;
use sidekick_core::JsonSchema;
use sidekick_llm::guided::Guidance;
use std::time::{Duration, Instant};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [dir, schemas, items] = args.as_slice() else {
        eprintln!("usage: guided_masks <model dir> <schemas.json> <items.jsonl>");
        std::process::exit(2);
    };
    let dir = std::path::Path::new(dir);
    let manifest = ChatManifest::read(&dir.join("chat.toml")).expect("chat.toml");
    let chat = manifest.chat.as_ref().expect("[chat]");
    let tokenizer_path = dir.join(&manifest.tokenizer);
    let prompter = sidekick_llm::Prompter::open(dir, &manifest.tokenizer, chat).expect("tokenizer and template");
    let token_bytes = toktrie_hf_tokenizers::ByteTokenizer::from_file(&tokenizer_path).expect("bytes").token_bytes();
    let per_call = manifest.decoder.tokens_per_call;
    let guidance = Guidance::new(std::sync::Arc::new(prompter), &chat.stop_token_ids, manifest.decoder.vocab_size);

    let schemas: Value = serde_json::from_str(&std::fs::read_to_string(schemas).unwrap()).unwrap();
    let started = Instant::now();
    let warm = JsonSchema::parse(&serde_json::json!({"type": "integer"})).unwrap();
    guidance.guide(&warm).expect("the trie builds");
    println!("token trie and factory: {:.0} ms", started.elapsed().as_secs_f64() * 1e3);

    let mut masks: Vec<Duration> = Vec::new();
    let mut forced_total = Duration::ZERO;
    let (mut tokens, mut forced, mut calls, mut replies, mut failed) = (0usize, 0usize, 0usize, 0usize, 0usize);
    let mut compiles: Vec<(String, Duration, Duration)> = Vec::new();
    for line in std::fs::read_to_string(items).unwrap().lines().filter(|l| !l.trim().is_empty()) {
        let item: Value = serde_json::from_str(line).unwrap();
        let name = item["schema"].as_str().unwrap();
        let schema = JsonSchema::parse(&schemas[name]).unwrap();
        let t0 = Instant::now();
        let mut g = guidance.guide(&schema).unwrap();
        let took = t0.elapsed();
        if g.stats().compiled {
            let t1 = Instant::now();
            guidance.guide(&schema).unwrap();
            compiles.push((name.to_string(), took, t1.elapsed()));
        }
        let ordered: Value = serde_json::from_str(&schema.conform(&item["gold"].to_string())).unwrap();
        let answer = sidekick_llm::guided::spaced_json(&ordered);
        let text = answer.as_bytes();
        let mut at = 0;
        // The reply's opening is forced before the prompt's calls.
        let t = Instant::now();
        let opening = g.forced(usize::MAX).unwrap();
        forced_total += t.elapsed();
        let (mut n_tokens, mut n_forced, mut n_calls) = (opening.len(), opening.len(), 0);
        at += opening.iter().map(|&t| token_bytes[t as usize].len()).sum::<usize>();
        let mut ok = true;
        while at < text.len() {
            let t = Instant::now();
            let mask = g.mask().unwrap();
            masks.push(t.elapsed());
            let mut best: Option<(usize, u32)> = None;
            mask.iter_set_entries(|i| {
                let b = &token_bytes[i];
                if !b.is_empty() && text[at..].starts_with(b) && best.is_none_or(|(len, _)| b.len() > len) {
                    best = Some((b.len(), i as u32));
                }
            });
            let Some((len, token)) = best else {
                eprintln!("{}: no allowed token matches {:?}", item["id"], String::from_utf8_lossy(&text[at..]));
                ok = false;
                break;
            };
            g.consume(token).unwrap();
            at += len;
            n_tokens += 1;
            let t = Instant::now();
            let f = g.forced(usize::MAX).unwrap();
            forced_total += t.elapsed();
            let f_len: usize = f.iter().map(|&t| token_bytes[t as usize].len()).sum();
            assert!(text[at..].starts_with(&f.iter().flat_map(|&t| token_bytes[t as usize].clone()).collect::<Vec<_>>()));
            at += f_len;
            n_tokens += f.len();
            n_forced += f.len();
            // The sampled token and its forced tokens share calls of T.
            if !g.done() {
                n_calls += (1 + f.len()).div_ceil(per_call);
            }
        }
        if !ok || !g.done() {
            failed += 1;
            continue;
        }
        replies += 1;
        tokens += n_tokens;
        forced += n_forced;
        calls += n_calls;
    }

    masks.sort();
    let pct = |p: f64| masks[((masks.len() - 1) as f64 * p).round() as usize].as_secs_f64() * 1e6;
    let mean = masks.iter().sum::<Duration>().as_secs_f64() * 1e6 / masks.len() as f64;
    for (name, first, cached) in &compiles {
        println!("schema {name}: compiled in {:.1} ms; cached {:.3} ms", first.as_secs_f64() * 1e3, cached.as_secs_f64() * 1e3);
    }
    println!("replies walked: {replies} ({failed} couldn't be)");
    println!(
        "masks: {} — mean {mean:.0} µs, p50 {:.0}, p90 {:.0}, p99 {:.0}, max {:.0} µs",
        masks.len(),
        pct(0.5),
        pct(0.9),
        pct(0.99),
        pct(1.0)
    );
    println!("forced-token computation: {:.0} µs per step", forced_total.as_secs_f64() * 1e6 / (masks.len() + replies) as f64);
    println!(
        "tokens: {tokens}, forced {forced} ({:.0}%); decoder calls after the prompt: {calls} ({:.2} tokens a call)",
        100.0 * forced as f64 / tokens as f64,
        tokens as f64 / calls.max(1) as f64
    );

    // What computing the mask beside a call, on a helper thread, would add
    // to each step: a scoped thread's spawn
    // and join around no work.
    let reps = 2000;
    let t = Instant::now();
    for _ in 0..reps {
        std::thread::scope(|s| {
            s.spawn(|| std::hint::black_box(1));
        });
    }
    println!("scoped thread spawn + join: {:.0} µs", t.elapsed().as_secs_f64() * 1e6 / reps as f64);
}
