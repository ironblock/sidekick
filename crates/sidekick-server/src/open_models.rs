//! The open chat models in the models directory (`chat.toml`, D53), served
//! by `sidekick-llm` over the Core ML decoder.
//!
//! A decoder loads as an encoder's bucket does (D52): through
//! `sidekick_embed::warm`'s loads, so the warm-up ledger records it and a
//! request that needs it while the warm-up process compiles it waits on the
//! same lock file. Its chunk programs are its one "bucket", labelled with
//! its cache capacity. The warm-up process loads it with the strict ANE
//! check, which is conclusive in a process holding nothing else on the
//! Neural Engine; the daemon, which may hold embedders there, with the
//! lenient one.

use crate::chat_models::{self, ChatModel};
use crate::config::{Config, RequestTimeouts};
use sidekick_core::chat_manifest::{ChatManifest, CHAT_MANIFEST};
use sidekick_core::{ComputeUnits, Error, ModelRegistry, ResolvedChat, Result, ScanOptions};
use sidekick_embed::warm::{Key, Label, Warmed};
use sidekick_llm::{ChatLane, DecoderLoader, LaneOptions, OpenModel, OpenModelSpec, Prompter};
use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

/// Why a build skips open chat models, when it does.
pub const UNSUPPORTED: &str =
    "this build has no Core ML, and an open chat model runs on the Neural Engine (D53): build sidekickd with the \
     `coreml` feature on macOS";

/// A decoder's programs, as the warm-up ledger keys them.
pub fn key(dir: &Path, manifest: &ChatManifest) -> Key {
    let paths = manifest.decoder.chunk_files().iter().map(|f| dir.join(f)).collect();
    Key { paths, units: ComputeUnits::CpuAndNeuralEngine }
}

/// The "bucket" a decoder is listed under: its cache capacity.
pub fn bucket(manifest: &ChatManifest) -> usize {
    manifest.decoder.cache_capacity
}

/// Each open model's chat template and tokenizer, read now: a model whose
/// tokenizer doesn't load, or whose template doesn't compile, is skipped
/// with the reason (`/health`'s `skipped_models`), as a bad manifest is,
/// rather than failing the daemon.
pub fn prompters(registry: &mut ModelRegistry) -> BTreeMap<String, Arc<Prompter>> {
    let mut prompters = BTreeMap::new();
    let mut skip = Vec::new();
    for chat in registry.chats() {
        let m = &chat.manifest;
        let section = m.chat.as_ref().expect("the registry checked [chat]");
        match Prompter::open(&chat.dir, &m.tokenizer, section) {
            Ok(p) => {
                prompters.insert(m.id.clone(), Arc::new(p));
            }
            Err(e) => skip.push((m.id.clone(), format!("its tokenizer or chat template can't be used: {e}"))),
        }
    }
    for (id, reason) in skip {
        registry.skip_chat(&id, &reason);
    }
    prompters
}

/// The ANE chat lane and the registry's open models, each sharing it with
/// `apple-fm`. `None` when there are no open models. `prompters` holds each
/// model's template and tokenizer ([`prompters`]).
pub fn build(
    config: &Config,
    registry: &ModelRegistry,
    mut prompters: BTreeMap<String, Arc<Prompter>>,
    timeouts: &RequestTimeouts,
    scan: &ScanOptions,
) -> anyhow::Result<Option<(Arc<ChatLane>, Vec<ChatModel>)>> {
    if registry.chat_ids().next().is_none() {
        return Ok(None);
    }
    let footprint: sidekick_llm::lane::FootprintReader = Arc::new(footprint);
    let lane = ChatLane::new(LaneOptions {
        queue_depth: config.queue_depth(registry.chat_ids()),
        idle_ttl: config.model_idle_ttl(),
        ane_budget_mib: config.ane_budget_mib,
        footprint,
    });
    let mut models = Vec::new();
    for chat in registry.chats() {
        let m = &chat.manifest;
        let section = m.chat.as_ref().expect("the registry checked [chat]");
        let prompter = prompters
            .remove(&m.id)
            .ok_or_else(|| anyhow::anyhow!("open chat model `{}` has no prompter", m.id))?;
        let spec = OpenModelSpec {
            id: m.id.clone(),
            context_length: m.decoder.context_length,
            prompter,
            stop_token_ids: section.stop_token_ids.clone(),
            vocab_size: m.decoder.vocab_size,
            sampling: m.sampling,
            estimated_ane_bytes: m.decoder.program_weight_bytes(&chat.dir),
            queue_depth: config.models.get(&m.id).and_then(|c| c.queue_depth).unwrap_or(sidekick_llm::DEFAULT_QUEUE_DEPTH),
        };
        let timeout = timeouts.resolve(&m.id, m.serving.request_timeout_secs.map(Duration::from_secs));
        let model = OpenModel::new(spec, lane.clone(), loader(chat, scan.clone()), timeout.0, config.load_timeout());
        let revision = m.source.as_ref().and_then(|s| s.revision.as_deref());
        models.push(chat_models::open_model(Arc::new(model), revision, timeout));
    }
    Ok(Some((lane, models)))
}

/// The process's Neural Engine footprint.
fn footprint() -> Option<u64> {
    #[cfg(all(feature = "coreml", target_os = "macos"))]
    {
        sidekick_llm::coreml::ane_footprint()
    }
    #[cfg(not(all(feature = "coreml", target_os = "macos")))]
    {
        None
    }
}

/// Loads `chat`'s decoder in the daemon: through the warm-up ledger and
/// lock files. A build the warm-up process's strict ANE check refused is
/// refused here too, with its reason. The daemon loads with the lenient
/// check, which can't tell an evicted program from one off the ANE, unless
/// it holds nothing on the Neural Engine, when the strict check is
/// conclusive and its refusal is recorded in turn.
fn loader(chat: &ResolvedChat, scan: ScanOptions) -> DecoderLoader {
    let (dir, manifest) = (chat.dir.clone(), chat.manifest.clone());
    let key = key(&dir, &manifest);
    Arc::new(move || {
        let label = Label { model: &manifest.id, bucket: bucket(&manifest) };
        if let Some(reason) = sidekick_embed::warm::refusal(&key) {
            return Err(Error::Inference(format!(
                "refusing decoder `{}`: a strict check of this build on this Mac refused it: {reason}",
                manifest.id
            )));
        }
        let strict = footprint() == Some(0);
        let loaded = sidekick_embed::warm::load_with(&key, label, || load(&dir, &manifest, &scan, strict));
        if let Err(e) = &loaded {
            if strict && is_placement_refusal(e) {
                sidekick_embed::warm::refuse(&key, label, &e.to_string());
            }
        }
        loaded
    })
}

/// Whether `e` refused a decoder whose programs weren't confirmed on the
/// Neural Engine.
fn is_placement_refusal(e: &Error) -> bool {
    #[cfg(all(feature = "coreml", target_os = "macos"))]
    {
        sidekick_llm::coreml::is_placement_refusal(e)
    }
    #[cfg(not(all(feature = "coreml", target_os = "macos")))]
    {
        let _ = e;
        false
    }
}

/// Load a decoder: `strict` for the ANE check that only a process with
/// nothing else on the Neural Engine can make.
#[cfg(all(feature = "coreml", target_os = "macos"))]
fn load(dir: &Path, manifest: &ChatManifest, scan: &ScanOptions, strict: bool) -> Result<Box<dyn sidekick_llm::Decoder>> {
    use sidekick_llm::coreml::AneCheck;
    let check = if strict { AneCheck::Strict } else { AneCheck::Lenient };
    let decoder = sidekick_llm::coreml::load(dir, manifest, scan.clone(), check)?;
    Ok(Box::new(decoder))
}

#[cfg(not(all(feature = "coreml", target_os = "macos")))]
fn load(_: &Path, _: &ChatManifest, _: &ScanOptions, _: bool) -> Result<Box<dyn sidekick_llm::Decoder>> {
    Err(Error::Unavailable(sidekick_core::UnavailableReason::NotSupportedInBuild))
}

/// Warm the decoder in `dir` (the warm-up process, `sidekickd warm`): load
/// it with the strict ANE check, then release it. The daemon has applied
/// its caps already, so they aren't applied again. A strict refusal is
/// recorded in the ledger, which binds the daemon.
pub fn warm(dir: &Path, key: &Key, label: Label<'_>) -> Result<Warmed> {
    let manifest = ChatManifest::read(&dir.join(CHAT_MANIFEST))
        .map_err(|e| Error::InvalidManifest { path: dir.join(CHAT_MANIFEST).display().to_string(), message: e })?;
    let scan = ScanOptions { ignore_ane_weight_cap: true, ignore_ane_seq_cap: true, ..Default::default() };
    let warmed = sidekick_embed::warm::warm_with(key, label, || load(dir, &manifest, &scan, true));
    if let Err(e) = &warmed {
        if is_placement_refusal(e) {
            sidekick_embed::warm::refuse(key, label, &e.to_string());
        }
    }
    warmed
}

#[cfg(test)]
mod tests {
    use super::*;

    const MANIFEST: &str = r#"
id = "ID"
task = "text-generation"
interface = "kv-state-1"
tokenizer = "tokenizer.json"

[decoder]
artifact = "decoder.mlmodelc"
chunks = 1
tokens_per_call = 4
cache_capacity = 64
context_length = 61
hidden_size = 8
head_dim = 4
kv_heads = 1
layers = [1]
vocab_size = 16

[decoder.host]
embedding = "embedding.safetensors"
rope = "rope.safetensors"

[chat]
template = "chat_template.jinja"
stop_token_ids = [1]
"#;

    const TOKENIZER: &str = r#"{"version": "1.0", "truncation": null, "padding": null, "added_tokens": [],
        "normalizer": null, "pre_tokenizer": {"type": "Whitespace"}, "post_processor": null, "decoder": null,
        "model": {"type": "WordLevel", "vocab": {"[UNK]": 0, "</s>": 1}, "unk_token": "[UNK]"}}"#;

    fn install(root: &Path, id: &str, tokenizer: &str, template: &str) {
        let dir = root.join(id);
        std::fs::create_dir_all(dir.join("decoder.mlmodelc/weights")).unwrap();
        std::fs::write(dir.join("decoder.mlmodelc/weights/weight.bin"), b"w").unwrap();
        for (file, body) in [
            ("chat.toml", MANIFEST.replace("ID", id)),
            ("tokenizer.json", tokenizer.to_string()),
            ("chat_template.jinja", template.to_string()),
            ("embedding.safetensors", String::new()),
            ("rope.safetensors", String::new()),
        ] {
            std::fs::write(dir.join(file), body).unwrap();
        }
    }

    #[test]
    fn a_model_whose_tokenizer_or_template_cant_be_used_is_skipped_not_fatal() {
        let root = std::env::temp_dir().join(format!("sk-open-models-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        install(&root, "good", TOKENIZER, "{% for m in messages %}{{ m.content }}{% endfor %}");
        install(&root, "bad-tokenizer", "{not json", "{{ messages }}");
        install(&root, "bad-template", TOKENIZER, "{% for m in messages %}");
        let mut registry = ModelRegistry::scan_with(&root, &ScanOptions::default()).unwrap();
        assert_eq!(registry.chat_ids().count(), 3);
        let prompters = prompters(&mut registry);
        assert_eq!(prompters.keys().collect::<Vec<_>>(), ["good"]);
        assert_eq!(registry.chat_ids().collect::<Vec<_>>(), ["good"]);
        let skipped: Vec<String> = registry.skipped().iter().map(|s| format!("{}: {}", s.path.display(), s.reason)).collect();
        assert!(skipped.iter().any(|s| s.starts_with("bad-tokenizer/chat.toml") && s.contains("tokenizer")), "{skipped:?}");
        assert!(skipped.iter().any(|s| s.starts_with("bad-template/chat.toml") && s.contains("template")), "{skipped:?}");
        // The rest serves.
        let built = build(&Config::default(), &registry, prompters, &RequestTimeouts::default(), &ScanOptions::default())
            .unwrap()
            .expect("an open model");
        assert_eq!(built.1.iter().map(|m| m.id().to_string()).collect::<Vec<_>>(), ["good"]);
        let _ = std::fs::remove_dir_all(&root);
    }
}
