//! Open causal language models on sidekick's chat route (D53;
//! docs/design/open-llm.md, "The runtime" through "Sampling and the wire
//! contract").
//!
//! The decoder (`sidekick_coreml::KvDecoder`, on the Neural Engine) turns
//! token ids into one logits row per call. This crate does everything
//! around it, platform-neutrally, so it builds and tests anywhere against a
//! fake [`Decoder`]:
//! - [`prompt`]: the checkpoint's chat template (minijinja, as transformers
//!   renders it) and tokenizer, with special-token text neutralised in
//!   message content;
//! - [`sampler`]: greedy at temperature 0 (ties to the lower id), else
//!   temperature and `top_p`, with a seeded ChaCha generator for `seed`;
//! - [`generate`]: one request's loop, streaming text through the shared
//!   stop-sequence shaper;
//! - [`lane`]: the one ANE chat lane `apple-fm` and the open models share,
//!   its bounded queue, and the one resident open model;
//! - [`model`]: an open model as a `ChatBackend`, on its own decoder
//!   thread.
//!
//! The `coreml` feature adds [`coreml`]: the Core ML decoder behind
//! [`Decoder`], on macOS.

pub mod decoder;
mod engine;
pub mod generate;
pub mod lane;
pub mod model;
pub mod prompt;
pub mod sampler;

#[cfg(all(feature = "coreml", target_os = "macos"))]
pub mod coreml;

pub use decoder::{Decoder, DecoderProbe, DecoderSpec, DecoderStatus};
pub use engine::DecoderLoader;
pub use lane::{ChatLane, LaneOptions, LanedChat, DEFAULT_ANE_BUDGET_MIB, DEFAULT_QUEUE_DEPTH};
pub use model::{OpenModel, OpenModelSpec};
pub use prompt::Prompter;

/// Whether this build can run open chat models: Core ML on macOS.
pub const SUPPORTED: bool = cfg!(all(feature = "coreml", target_os = "macos"));
