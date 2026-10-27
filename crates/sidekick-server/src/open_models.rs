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

/// The ANE chat lane and the registry's open models, each sharing it with
/// `apple-fm`. `None` when there are no open models.
pub fn build(
    config: &Config,
    registry: &ModelRegistry,
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
        let prompter = Prompter::open(&chat.dir, &m.tokenizer, section)
            .map_err(|e| anyhow::anyhow!("open chat model `{}`: {e}", m.id))?;
        let spec = OpenModelSpec {
            id: m.id.clone(),
            context_length: m.decoder.context_length,
            prompter: Arc::new(prompter),
            stop_token_ids: section.stop_token_ids.clone(),
            sampling: m.sampling,
            estimated_ane_bytes: m.decoder.program_weight_bytes(&chat.dir),
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
/// lock files, with the lenient ANE check.
fn loader(chat: &ResolvedChat, scan: ScanOptions) -> DecoderLoader {
    let (dir, manifest) = (chat.dir.clone(), chat.manifest.clone());
    let key = key(&dir, &manifest);
    Arc::new(move || {
        let label = Label { model: &manifest.id, bucket: bucket(&manifest) };
        sidekick_embed::warm::load_with(&key, label, || load(&dir, &manifest, &scan, false))
    })
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
/// its caps already, so they aren't applied again.
pub fn warm(dir: &Path, key: &Key, label: Label<'_>) -> Result<Warmed> {
    let manifest = ChatManifest::read(&dir.join(CHAT_MANIFEST))
        .map_err(|e| Error::InvalidManifest { path: dir.join(CHAT_MANIFEST).display().to_string(), message: e })?;
    let scan = ScanOptions { ignore_ane_weight_cap: true, ignore_ane_seq_cap: true, ..Default::default() };
    sidekick_embed::warm::warm_with(key, label, || load(dir, &manifest, &scan, true))
}
