//! The Core ML decoder chain (`sidekick_coreml::KvDecoder`) behind
//! [`Decoder`], and the process's Neural Engine footprint.

use crate::decoder::{Decoder, DecoderProbe, DecoderSpec, DecoderStatus};
use sidekick_coreml::{DecoderMonitor, KvDecoder};
use sidekick_core::chat_manifest::ChatManifest;
use sidekick_core::{Result, ScanOptions};
use std::path::Path;
use std::sync::Arc;

pub use sidekick_coreml::{is_placement_refusal, AneCheck};

/// A loaded `kv-state-1` decoder.
pub struct CoremlDecoder(KvDecoder);

/// Load the decoder in `dir`. `check` is how the load confirms its
/// programs are on the Neural Engine: [`AneCheck::Strict`] where nothing
/// else is on it (the warm-up process, D52), [`AneCheck::Lenient`] in the
/// daemon, where other models may be.
pub fn load(dir: &Path, manifest: &ChatManifest, scan: ScanOptions, check: AneCheck) -> Result<CoremlDecoder> {
    let options = sidekick_coreml::DecoderOptions { scan, ane_check: check, ..Default::default() };
    KvDecoder::load_with(dir, manifest, options).map(CoremlDecoder)
}

/// The process's Neural Engine footprint (`ri_neural_footprint`), in bytes.
pub fn ane_footprint() -> Option<u64> {
    sidekick_coreml::process_memory()?.neural_footprint
}

impl Decoder for CoremlDecoder {
    fn spec(&self) -> DecoderSpec {
        let s = self.0.shape();
        DecoderSpec { tokens_per_call: s.tokens_per_call, context_length: s.context_length, vocab_size: s.vocab_size }
    }

    fn prefill(&mut self, tokens: &[u32]) -> Result<&[f32]> {
        self.0.prefill(tokens, 0)
    }

    fn step(&mut self, token: u32, pos: usize) -> Result<&[f32]> {
        self.0.step(&[token], pos)
    }

    fn probe(&self) -> Arc<dyn DecoderProbe> {
        Arc::new(Monitor(self.0.monitor()))
    }

    fn ane_bytes(&self) -> Option<u64> {
        self.0.ane_bytes()
    }
}

struct Monitor(DecoderMonitor);

impl DecoderProbe for Monitor {
    fn status(&self) -> DecoderStatus {
        let h = self.0.snapshot();
        DecoderStatus {
            failed: h.failed,
            calls: h.calls,
            stalls: h.stalls,
            warm_median_ms: h.warm_median_ms,
            last_stall_ms: h.last_stall_ms,
            since_last_stall: h.since_last_stall,
            state_resets: h.state_resets,
        }
    }
}
