//! The chat-template renderer against transformers (docs/design/open-llm.md,
//! "Tokenizer and chat template"): each fixture in `fixtures/chat-templates/`
//! (written by `tools/make_chat_template_fixture.py`) holds a checkpoint's
//! template and what `apply_chat_template` rendered and tokenized for a set
//! of conversations. The rendered text must match exactly, always; the token
//! ids must match whenever the checkpoint's tokenizer is in the Hugging Face
//! cache (`HF_HUB_CACHE`, else `$HF_HOME/hub`, else
//! `~/.cache/huggingface/hub`) at the fixture's revision. Without it the ids
//! are skipped with a note, or fail when `SIDEKICK_REQUIRE_CHAT_TOKENIZERS`
//! is set.

use serde::Deserialize;
use sha2::{Digest, Sha256};
use sidekick_core::chat_manifest::ChatSection;
use sidekick_core::{ChatMessage, Role};
use sidekick_llm::prompt::Prompter;
use std::path::{Path, PathBuf};
use tokenizers::Tokenizer;

#[derive(Deserialize)]
struct Fixture {
    format: u32,
    model: String,
    source: Source,
    tokenizer_sha256: String,
    template: String,
    bos_token: Option<String>,
    eos_token: Option<String>,
    template_kwargs: serde_json::Value,
    cases: Vec<Case>,
}

#[derive(Deserialize)]
struct Source {
    repo: String,
    revision: String,
}

#[derive(Deserialize)]
struct Case {
    id: String,
    messages: Vec<Message>,
    text: String,
    ids: Vec<u32>,
}

#[derive(Deserialize, serde::Serialize, Clone)]
struct Message {
    role: String,
    content: String,
}

fn fixtures() -> Vec<PathBuf> {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/chat-templates");
    let mut files: Vec<PathBuf> = std::fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("{}: {e}", dir.display()))
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|e| e == "json"))
        .collect();
    files.sort();
    assert!(files.len() >= 2, "the committed fixtures: {files:?}");
    files
}

fn hub_cache() -> Option<PathBuf> {
    if let Some(dir) = std::env::var_os("HF_HUB_CACHE") {
        return Some(PathBuf::from(dir));
    }
    if let Some(home) = std::env::var_os("HF_HOME") {
        return Some(PathBuf::from(home).join("hub"));
    }
    Some(PathBuf::from(std::env::var_os("HOME")?).join(".cache/huggingface/hub"))
}

/// The checkpoint's tokenizer at the fixture's revision, if it is cached
/// and is the file the fixture was made with.
fn cached_tokenizer(f: &Fixture) -> Result<Tokenizer, String> {
    let repo = f.source.repo.replace('/', "--");
    let path = hub_cache()
        .ok_or("no Hugging Face cache")?
        .join(format!("models--{repo}/snapshots/{}/tokenizer.json", f.source.revision));
    let bytes = std::fs::read(&path).map_err(|e| format!("{}: {e}", path.display()))?;
    let sha: String = Sha256::digest(&bytes).iter().map(|b| format!("{b:02x}")).collect();
    if sha != f.tokenizer_sha256 {
        return Err(format!("{} has sha256 {sha}, not the fixture's", path.display()));
    }
    Tokenizer::from_bytes(bytes).map_err(|e| e.to_string())
}

/// A tokenizer with no special tokens, for rendering alone.
fn no_tokenizer() -> Tokenizer {
    let json = serde_json::json!({
        "version": "1.0", "truncation": null, "padding": null, "added_tokens": [], "normalizer": null,
        "pre_tokenizer": null, "post_processor": null, "decoder": null,
        "model": {"type": "WordLevel", "vocab": {"[UNK]": 0}, "unk_token": "[UNK]"}
    });
    Tokenizer::from_bytes(serde_json::to_vec(&json).unwrap()).unwrap()
}

fn section(f: &Fixture) -> ChatSection {
    ChatSection {
        template: "chat_template.jinja".into(),
        template_kwargs: serde_json::from_value(f.template_kwargs.clone()).expect("kwargs as a TOML table"),
        bos_token: f.bos_token.clone(),
        eos_token: f.eos_token.clone(),
        stop_token_ids: vec![0],
    }
}

fn role(r: &str) -> Role {
    match r {
        "system" => Role::System,
        "user" => Role::User,
        "assistant" => Role::Assistant,
        other => panic!("role {other}"),
    }
}

#[test]
fn the_renderer_reproduces_transformers() {
    let mut skipped = Vec::new();
    for path in fixtures() {
        let f: Fixture = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        assert_eq!(f.format, 1, "{}", path.display());
        let name = format!("{} ({})", f.model, path.file_name().unwrap().to_string_lossy());

        // Rendering alone: the template's text against transformers'.
        let renderer = Prompter::new(no_tokenizer(), &f.template, &section(&f)).unwrap();
        for case in &f.cases {
            let messages = minijinja::Value::from(minijinja::value::Serde(&case.messages));
            let text = renderer.render_values(messages).unwrap_or_else(|e| panic!("{name} {}: {e}", case.id));
            assert_eq!(text, case.text, "{name}: case `{}` renders differently", case.id);
        }

        // With the checkpoint's tokenizer: the ids, and the whole path a
        // request takes (neutralizing leaves this plain text alone).
        let tokenizer = match cached_tokenizer(&f) {
            Ok(t) => t,
            Err(why) => {
                skipped.push(format!("{name}: {why}"));
                continue;
            }
        };
        let prompter = Prompter::new(tokenizer, &f.template, &section(&f)).unwrap();
        for case in &f.cases {
            assert_eq!(prompter.encode_text(&case.text).unwrap(), case.ids, "{name}: case `{}` tokenizes differently", case.id);
            let messages: Vec<ChatMessage> =
                case.messages.iter().map(|m| ChatMessage::new(role(&m.role), m.content.clone())).collect();
            assert_eq!(prompter.encode(&messages).unwrap(), case.ids, "{name}: case `{}` as a request", case.id);
        }
        eprintln!("{name}: {} cases, text and ids", f.cases.len());
    }
    if !skipped.is_empty() {
        let note = format!("token ids not checked: {}", skipped.join("; "));
        if std::env::var_os("SIDEKICK_REQUIRE_CHAT_TOKENIZERS").is_some() {
            panic!("{note}");
        }
        eprintln!("skipped: {note}");
    }
}
