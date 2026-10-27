//! What the loop needs of a decoder: token ids in, one logits row out
//! (docs/design/open-llm.md, "One call, T tokens"). The Core ML chain
//! (`sidekick_coreml::KvDecoder`) implements it behind the `coreml`
//! feature ([`crate::coreml`]); tests implement it with fakes.

use sidekick_core::Result;
use std::sync::Arc;
use std::time::Duration;

/// The shapes the loop plans with.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DecoderSpec {
    /// T: tokens per call.
    pub tokens_per_call: usize,
    /// The usable context: prompt plus completion, C − T + 1.
    pub context_length: usize,
    /// V: the logits row's width.
    pub vocab_size: usize,
}

/// A loaded decoder, driven by one thread at a time.
///
/// Its KV state carries over from request to request: a request writes its
/// prompt from position 0, and the mask hides every key past a query's own
/// position, so an earlier request's keys never reach it. A decoder
/// replaces its state itself after a failed call.
pub trait Decoder: Send {
    fn spec(&self) -> DecoderSpec;

    /// The prompt, from position 0: returns its last token's logits.
    fn prefill(&mut self, tokens: &[u32]) -> Result<&[f32]>;

    /// One generated token at cache position `pos`: returns its logits.
    fn step(&mut self, token: u32, pos: usize) -> Result<&[f32]>;

    /// The decoder's health, readable from any thread while it runs.
    fn probe(&self) -> Arc<dyn DecoderProbe>;

    /// What loading the decoder added to the process's Neural Engine
    /// footprint, when it was measured.
    fn ane_bytes(&self) -> Option<u64> {
        None
    }
}

/// A decoder's health, from any thread.
pub trait DecoderProbe: Send + Sync {
    fn status(&self) -> DecoderStatus;
}

/// A snapshot of a decoder's health (`sidekick_coreml::DecoderHealth`).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct DecoderStatus {
    /// Why the decoder can no longer serve: a call that hung. Its thread
    /// can't be stopped, so recovery is a restart.
    pub failed: Option<String>,
    /// T-token calls since the load.
    pub calls: u64,
    /// Calls slower than five times the warm median: other work on the
    /// Neural Engine, this daemon's or another app's.
    pub stalls: u64,
    pub warm_median_ms: Option<f64>,
    pub last_stall_ms: Option<f64>,
    pub since_last_stall: Option<Duration>,
    /// Times the KV state was replaced after a failed call.
    pub state_resets: u64,
}

/// A probe for a decoder that can't fail and keeps no counts.
#[derive(Debug, Default)]
pub struct NoProbe;

impl DecoderProbe for NoProbe {
    fn status(&self) -> DecoderStatus {
        DecoderStatus::default()
    }
}
