//! The classifier input builder against each token-id fixture
//! (`fixtures/classify/<model id>.tokens.json`, schema `tokens.schema.json`).
//! A fixture holds the ids a model's own Python input builder produced (laya's
//! `build_sequence` for the laya format); the Rust port must reproduce them
//! exactly.
//!
//! A fixture needs its model's `classifier.toml` and tokenizer, which aren't
//! committed: they're looked up in `$SIDEKICK_MODELS_DIR` (default: the
//! daemon's models directory). A fixture whose model isn't installed is
//! skipped with a note, or fails when `SIDEKICK_REQUIRE_CLASSIFY_FIXTURES`
//! is set. `SIDEKICK_CLASSIFY_FIXTURES` points at another fixture
//! directory.

use serde::Deserialize;
use sha2::{Digest, Sha256};
use sidekick_core::{ClassifyParams, ModelRegistry, QuestionType};
use sidekick_embed::InputBuilder;
use std::path::{Path, PathBuf};

#[derive(Deserialize)]
struct Fixture {
    format: u32,
    model: String,
    tokenizer_sha256: String,
    max_len: usize,
    cases: Vec<Case>,
}

#[derive(Deserialize)]
struct Case {
    id: String,
    input: String,
    #[serde(default)]
    candidate_labels: Vec<String>,
    #[serde(default)]
    question_type: Option<QuestionType>,
    #[serde(default)]
    instructions: Option<String>,
    ids: Vec<i32>,
    #[serde(default)]
    markers: Vec<i32>,
    #[serde(default)]
    qtype: Option<i32>,
    /// fev: the decide token's position.
    #[serde(default)]
    decide: Option<i32>,
    /// agentjev: each token's tree segment and position.
    #[serde(default)]
    seg: Vec<i32>,
    #[serde(default)]
    position_ids: Vec<i32>,
}

fn fixtures_dir() -> PathBuf {
    std::env::var_os("SIDEKICK_CLASSIFY_FIXTURES").map(PathBuf::from).unwrap_or_else(|| {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/classify")
    })
}

fn models_dir() -> Option<PathBuf> {
    std::env::var_os("SIDEKICK_MODELS_DIR").map(PathBuf::from).or_else(|| {
        let home = std::env::var_os("HOME")?;
        Some(PathBuf::from(home).join("Library/Application Support/sidekick/models"))
    })
}

fn skip(message: String) {
    if std::env::var_os("SIDEKICK_REQUIRE_CLASSIFY_FIXTURES").is_some() {
        panic!("{message}");
    }
    eprintln!("skipped: {message}");
}

#[test]
fn input_builder_reproduces_every_token_fixture() {
    let mut fixtures: Vec<PathBuf> = std::fs::read_dir(fixtures_dir())
        .map(|d| d.filter_map(|e| e.ok().map(|e| e.path())).collect())
        .unwrap_or_default();
    fixtures.retain(|p| p.to_string_lossy().ends_with(".tokens.json"));
    fixtures.sort();
    if fixtures.is_empty() {
        return skip(format!("no *.tokens.json in {}", fixtures_dir().display()));
    }
    let registry = models_dir()
        .map(|d| ModelRegistry::scan(&d).expect("models dir readable"))
        .unwrap_or_default();

    for path in fixtures {
        let fixture: Fixture =
            serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        assert_eq!(fixture.format, 1, "{}", path.display());
        let Ok(model) = registry.classifier(&fixture.model) else {
            skip(format!("model `{}` isn't installed; {} not checked", fixture.model, path.display()));
            continue;
        };
        let tokenizer = std::fs::read(model.tokenizer_path()).unwrap();
        assert_eq!(
            hex(&Sha256::digest(&tokenizer)),
            fixture.tokenizer_sha256,
            "{}: the installed tokenizer isn't the one the fixture was made with",
            fixture.model
        );
        assert_eq!(fixture.max_len, model.manifest.max_seq_len, "{}", fixture.model);
        let builder = InputBuilder::load(model).unwrap();

        // The special tokens resolved from the tokenizer must be the ones
        // the fixture's sequences use, so a misread template fails here.
        if let Some(sp) = builder.laya_specials() {
            for case in &fixture.cases {
                assert_eq!(case.ids.first(), Some(&(sp.cls as i32)), "{}: CLS", case.id);
                assert_eq!(case.ids.last(), Some(&(sp.sep as i32)), "{}: SEP", case.id);
                for &m in &case.markers {
                    assert_eq!(case.ids[m as usize], sp.mask as i32, "{}: MASK", case.id);
                }
            }
        }

        for case in &fixture.cases {
            let params = ClassifyParams {
                candidate_labels: case.candidate_labels.clone(),
                question_type: case.question_type,
                instructions: case.instructions.clone(),
                ..Default::default()
            };
            let prepared = builder
                .prepare(&case.input, &params)
                .unwrap_or_else(|e| panic!("{}/{}: {e}", fixture.model, case.id));
            assert_eq!(prepared.ids, case.ids, "{}/{}: ids", fixture.model, case.id);
            assert_eq!(prepared.markers, case.markers, "{}/{}: markers", fixture.model, case.id);
            assert_eq!(prepared.qtype, case.qtype, "{}/{}: qtype", fixture.model, case.id);
            assert_eq!(prepared.decide_pos, case.decide, "{}/{}: decide", fixture.model, case.id);
            assert_eq!(prepared.seg, case.seg, "{}/{}: seg", fixture.model, case.id);
            assert_eq!(prepared.position_ids, case.position_ids, "{}/{}: position_ids", fixture.model, case.id);
            assert!(builder.buckets().contains(&prepared.bucket));
        }
        eprintln!("{}: {} cases reproduced", fixture.model, fixture.cases.len());
    }
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}
