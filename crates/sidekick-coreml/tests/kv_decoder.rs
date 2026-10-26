//! The stateful decoder chain with real Core ML, on the Neural Engine: the
//! tiny `kv-state-1` decoder in tests/fixtures/tiny-kv-decoder (built by
//! tools/make_decoder_test_model.py), two chunk programs with their KV
//! caches in Core ML state. Through the chain's own load and calls, it
//! matches the fp32 torch model call for call, reuses its state across
//! requests, refuses a manifest its programs don't match and a program
//! Core ML puts off the Neural Engine, and keeps a flat memory footprint
//! over thousands of calls.
//!
//! macOS on Apple silicon only: `kv-state-1` runs with `cpu_and_ne` alone
//! (its state crashes on the CPU), so a Mac whose Core ML lists no Neural
//! Engine, a CI virtual machine among them, skips these tests.
#![cfg(target_os = "macos")]

use serde::Deserialize;
use sidekick_core::chat_manifest::ChatManifest;
use sidekick_core::Error;
use sidekick_coreml::{process_memory, DecoderOptions, KvDecoder};
use std::path::{Path, PathBuf};
use std::sync::{Mutex, MutexGuard};

/// One test at a time: they share the ANE, and the memory test reads the
/// whole process's footprint.
fn serial() -> MutexGuard<'static, ()> {
    static LOCK: Mutex<()> = Mutex::new(());
    LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

fn fixture() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny-kv-decoder")
}

fn ane() -> bool {
    let ane = sidekick_coreml::has_neural_engine();
    if !ane {
        eprintln!("skipping: Core ML lists no Neural Engine here, and a kv-state-1 decoder runs only there");
    }
    ane
}

fn manifest() -> ChatManifest {
    ChatManifest::read(&fixture().join("chat.toml")).unwrap()
}

fn load(output_backings: bool) -> KvDecoder {
    let options = DecoderOptions {
        output_backings,
        ..Default::default()
    };
    KvDecoder::load_with(&fixture(), &manifest(), options).unwrap()
}

#[derive(Deserialize)]
struct Expected {
    prompt: Vec<u32>,
    greedy: Vec<u32>,
    calls: Vec<Call>,
}

#[derive(Deserialize)]
struct Call {
    tokens: Vec<u32>,
    pos: usize,
    sel: usize,
}

/// The reference calls, each with the fp32 torch model's logits.
fn expected() -> (Expected, Vec<Vec<f32>>) {
    let e: Expected =
        serde_json::from_str(&std::fs::read_to_string(fixture().join("expected.json")).unwrap())
            .unwrap();
    let raw = std::fs::read(fixture().join("expected.safetensors")).unwrap();
    let st = safetensors::SafeTensors::deserialize(&raw).unwrap();
    let view = st.tensor("logits").unwrap();
    let v = view.shape()[1];
    let values: Vec<f32> = view
        .data()
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect();
    let rows = values.chunks_exact(v).map(<[f32]>::to_vec).collect();
    (e, rows)
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let dot: f64 = a.iter().zip(b).map(|(x, y)| *x as f64 * *y as f64).sum();
    let na: f64 = a.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
    dot / (na * nb)
}

fn argmax(x: &[f32]) -> usize {
    x.iter()
        .enumerate()
        .fold(0, |best, (i, v)| if *v > x[best] { i } else { best })
}

/// Every reference call through `decoder`, returning its logits rows.
fn run_calls(decoder: &mut KvDecoder, calls: &[Call]) -> Vec<Vec<f32>> {
    calls
        .iter()
        .map(|c| {
            decoder
                .step_select(&c.tokens, c.pos, c.sel)
                .unwrap()
                .to_vec()
        })
        .collect()
}

#[test]
fn the_chain_matches_the_torch_model_call_for_call() {
    let _serial = serial();
    if !ane() {
        return;
    }
    let (e, want) = expected();
    for backings in [true, false] {
        let mut decoder = load(backings);
        assert_eq!(decoder.output_backings(), backings);
        let got = run_calls(&mut decoder, &e.calls);
        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
            let cos = cosine(g, w);
            assert!(cos > 0.999, "call {i} (backings {backings}): cosine {cos}");
            assert_eq!(argmax(g), argmax(w), "call {i} (backings {backings})");
        }
        // The load makes one call of its own.
        let h = decoder.health();
        assert_eq!(
            (h.calls, h.state_resets, h.failed),
            (1 + e.calls.len() as u64, 0, None)
        );
    }
}

#[test]
fn output_backings_change_nothing() {
    let _serial = serial();
    if !ane() {
        return;
    }
    let (e, _) = expected();
    let with = run_calls(&mut load(true), &e.calls);
    let without = run_calls(&mut load(false), &e.calls);
    assert_eq!(
        with, without,
        "outputs differ with and without output backings"
    );
}

#[test]
fn the_prompt_and_greedy_decoding_reproduce_the_reference() {
    let _serial = serial();
    if !ane() {
        return;
    }
    let (e, _) = expected();
    let mut decoder = load(true);
    let mut next = argmax(decoder.prefill(&e.prompt, 0).unwrap()) as u32;
    let mut tokens = vec![];
    for pos in e.prompt.len().. {
        tokens.push(next);
        if tokens.len() == e.greedy.len() {
            break;
        }
        next = argmax(decoder.step(&[next], pos).unwrap()) as u32;
    }
    assert_eq!(tokens, e.greedy);
}

#[test]
fn one_state_serves_request_after_request() {
    let _serial = serial();
    if !ane() {
        return;
    }
    let (e, _) = expected();
    let mut decoder = load(true);
    let first = run_calls(&mut decoder, &e.calls);
    // Another request, with another prompt, fills the cache with its keys.
    decoder
        .prefill(
            &[
                900, 901, 902, 903, 904, 905, 906, 907, 908, 909, 910, 911, 912,
            ],
            0,
        )
        .unwrap();
    for pos in 13..20 {
        decoder.step(&[5], pos).unwrap();
    }
    // The first request again, on the same state: the mask hides every key
    // past a query's own position, so nothing of the other request reaches it.
    assert_eq!(run_calls(&mut decoder, &e.calls), first);
    decoder.reset_state();
    assert_eq!(run_calls(&mut decoder, &e.calls), first);
    assert_eq!(decoder.health().state_resets, 1);
}

#[test]
fn calls_past_the_contract_are_refused_without_running() {
    let _serial = serial();
    if !ane() {
        return;
    }
    let mut decoder = load(true);
    let s = decoder.shape().clone();
    assert_eq!(
        (s.tokens_per_call, s.cache_capacity, s.context_length),
        (64, 256, 193)
    );
    assert!(matches!(
        decoder.step(&[], 0),
        Err(Error::InvalidRequest(_))
    ));
    assert!(matches!(
        decoder.step(&[1; 65], 0),
        Err(Error::InvalidRequest(_))
    ));
    assert!(matches!(
        decoder.step(&[1024], 0),
        Err(Error::InvalidRequest(_))
    ));
    assert!(matches!(
        decoder.step_select(&[1, 2], 0, 2),
        Err(Error::InvalidRequest(_))
    ));
    // The last position a call can start at is C − T = 192.
    assert!(decoder.step(&[1], 192).is_ok());
    let err = decoder.step(&[1], 193).unwrap_err();
    assert!(
        matches!(
            err,
            Error::ContextOverflow {
                limit: 193,
                actual: Some(194)
            }
        ),
        "{err:?}"
    );
    let err = decoder.prefill(&[1; 194], 0).unwrap_err();
    assert!(
        matches!(
            err,
            Error::ContextOverflow {
                limit: 193,
                actual: Some(194)
            }
        ),
        "{err:?}"
    );
    // 193 tokens in four calls, the last one token at position 192.
    assert!(decoder.prefill(&[1; 193], 0).is_ok());
    // The load's own call, the one at 192 and the prompt's four ran.
    let h = decoder.health();
    assert_eq!((h.calls, h.state_resets), (1 + 1 + 4, 0));
}

/// The fixture's files in a directory of their own, with `edit` applied to
/// its manifest. The programs are linked, not copied (a linked program
/// directory stays on the ANE, D51).
fn edited(tag: &str, edit: impl Fn(String) -> String) -> (PathBuf, ChatManifest) {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!("tiny-kv-decoder-{tag}"));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    for name in [
        "decoder.0.mlmodelc",
        "decoder.1.mlmodelc",
        "embedding.safetensors",
        "rope.safetensors",
    ] {
        std::os::unix::fs::symlink(fixture().join(name), dir.join(name)).unwrap();
    }
    let text = edit(std::fs::read_to_string(fixture().join("chat.toml")).unwrap());
    std::fs::write(dir.join("chat.toml"), &text).unwrap();
    let manifest = ChatManifest::read(&dir.join("chat.toml")).unwrap();
    (dir, manifest)
}

fn refusal(tag: &str, edit: impl Fn(String) -> String) -> String {
    let (dir, manifest) = edited(tag, edit);
    match KvDecoder::load(&dir, &manifest) {
        Ok(_) => panic!("{tag}: loaded"),
        Err(e) => e.to_string(),
    }
}

#[test]
fn programs_that_dont_match_the_manifest_are_refused() {
    let _serial = serial();
    if !ane() {
        return;
    }
    let err = refusal("layers", |t| {
        t.replace("layers = [2, 2]", "layers = [1, 3]")
    });
    assert!(
        err.contains("decoder.0.mlmodelc")
            && err.contains("state `k_cache` is [2, 2, 256, 64] fp16"),
        "{err}"
    );
    let err = refusal("capacity", |t| {
        t.replace("cache_capacity = 256", "cache_capacity = 512")
            .replace("context_length = 193", "context_length = 449")
    });
    assert!(
        err.contains("input `mask` is [1, 1, 64, 256] fp16") && err.contains("[1, 1, 64, 512]"),
        "{err}"
    );
    let err = refusal("vocab", |t| {
        t.replace("vocab_size = 1024", "vocab_size = 1000")
    });
    assert!(
        err.contains("decoder.1.mlmodelc") && err.contains("output `logits`"),
        "{err}"
    );
    let err = refusal("kv-heads", |t| t.replace("kv_heads = 2", "kv_heads = 4"));
    assert!(err.contains("k_cache"), "{err}");
    // The manifest as shipped loads from the same layout.
    let (dir, manifest) = edited("as-shipped", |t| t);
    KvDecoder::load(&dir, &manifest).unwrap();
}

#[test]
fn host_tables_that_dont_match_the_manifest_are_refused() {
    let _serial = serial();
    if !ane() {
        return;
    }
    let (dir, manifest) = edited("rope", |t| t);
    std::fs::remove_file(dir.join("rope.safetensors")).unwrap();
    std::os::unix::fs::symlink(
        fixture().join("embedding.safetensors"),
        dir.join("rope.safetensors"),
    )
    .unwrap();
    let err = KvDecoder::load(&dir, &manifest)
        .err()
        .expect("refused")
        .to_string();
    assert!(err.contains("RoPE") && err.contains("`cos`"), "{err}");
    let (dir, manifest) = edited("embedding", |t| t);
    std::fs::remove_file(dir.join("embedding.safetensors")).unwrap();
    std::os::unix::fs::symlink(
        fixture().join("rope.safetensors"),
        dir.join("embedding.safetensors"),
    )
    .unwrap();
    let err = KvDecoder::load(&dir, &manifest)
        .err()
        .expect("refused")
        .to_string();
    assert!(
        err.contains("embedding table") && err.contains("2 tensors"),
        "{err}"
    );
}

/// The same contract at width 64, which Core ML judges too small for the
/// Neural Engine and runs on the CPU, where a decoder's state crashes: the
/// load refuses it rather than predict.
#[test]
fn a_decoder_core_ml_places_off_the_neural_engine_is_refused() {
    let _serial = serial();
    if !ane() {
        return;
    }
    // The fixture's own programs are on the ANE, and the load measures them;
    // dropped, they leave the process's ANE footprint (the system unloads
    // them within a few tens of milliseconds).
    let decoder = load(true);
    assert!(
        decoder.ane_bytes().is_some_and(|n| n > 0),
        "{:?}",
        decoder.ane_bytes()
    );
    drop(decoder);
    let released = std::time::Instant::now() + std::time::Duration::from_secs(5);
    while process_memory().unwrap().neural_footprint != Some(0)
        && std::time::Instant::now() < released
    {
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    assert_eq!(process_memory().unwrap().neural_footprint, Some(0));
    let dir = fixture().with_file_name("tiny-kv-decoder-cpu");
    let manifest = ChatManifest::read(&dir.join("chat.toml")).unwrap();
    let err = KvDecoder::load(&dir, &manifest)
        .err()
        .expect("refused")
        .to_string();
    assert!(
        err.contains("tiny-kv-decoder-cpu") && err.contains("Neural Engine"),
        "{err}"
    );
}

/// #173's loop without an autorelease pool grew without bound. Each call
/// here runs in its own pool, with inputs allocated once: after a warm-up,
/// thousands more calls leave the physical footprint where it was.
#[test]
fn memory_stays_flat_over_thousands_of_calls() {
    let _serial = serial();
    if !ane() {
        return;
    }
    for backings in [true, false] {
        let mut decoder = load(backings);
        let mut run = |n: usize| {
            for i in 0..n {
                let pos = i % 20;
                decoder.step(&[(i % 1000) as u32], pos).unwrap();
            }
        };
        run(500);
        let before = process_memory().unwrap().phys_footprint;
        run(5000);
        let after = process_memory().unwrap().phys_footprint;
        let grown = after as i64 - before as i64;
        // Measured with the pool left out: 440 MB over these 5,000 calls
        // without output backings, 1.8 MB with them. With it: under 0.1 MB.
        assert!(
            grown < 1 << 20,
            "backings {backings}: the footprint grew by {grown} bytes over 5,000 calls"
        );
        eprintln!("backings {backings}: footprint {before} -> {after} bytes over 5,000 calls");
    }
}
