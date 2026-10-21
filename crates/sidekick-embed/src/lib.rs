//! Encoder backends: embedders and classifiers.
//!
//! - [`StaticEmbedder`]: model2vec-style static token embeddings. Pure Rust,
//!   runs anywhere, microsecond lookups. The unconditional floor tier.
//! - [`CoremlEmbedder`] (feature `coreml`, macOS): a Core ML encoder
//!   (EmbeddingGemma-300m, bge-small, MiniLM, …) targeted at the ANE.
//! - [`onnx::OnnxEmbedder`] (feature `onnx`, macOS and Linux): an ONNX
//!   encoder run by ONNX Runtime on the CPU, one input at a time (D40).
//! - [`CoremlClassifier`] (feature `coreml`, macOS): a Core ML classifier,
//!   text-classification, a reranker, or zero-shot in laya's or GLiNER2's
//!   format. Its tokenization ([`InputBuilder`], [`laya`], [`gliner2`]) is
//!   platform-neutral.
//! - [`tokenize::ModelTokenizer`]: an embedder's or classifier's tokenizer
//!   without its weights, counting as its route does (the daemon's
//!   `/tokenize`, D44).

pub mod classify_input;
pub mod embed_input;
pub mod agentjev;
pub mod fev;
pub mod gliner2;
pub mod laya;
pub mod onnx;
pub mod placement;
mod pooling;
mod shared;
mod static_embedder;
pub mod tokenize;

pub use classify_input::InputBuilder;
pub use pooling::{degenerate, mean_pool, normalize_in_place, pool};
pub use static_embedder::StaticEmbedder;

#[cfg(all(feature = "coreml", target_os = "macos"))]
mod bucket_models;
#[cfg(all(feature = "coreml", target_os = "macos"))]
mod chain_check;
#[cfg(all(feature = "coreml", target_os = "macos"))]
mod coreml_classifier;
#[cfg(all(feature = "coreml", target_os = "macos"))]
mod coreml_embedder;
#[cfg(all(feature = "coreml", target_os = "macos"))]
pub use coreml_classifier::CoremlClassifier;
#[cfg(all(feature = "coreml", target_os = "macos"))]
pub use coreml_embedder::{CoremlEmbedder, Prepared};

use sidekick_core::manifest::{ResolvedClassifier, ResolvedModel};
use sidekick_core::{EmbeddingBackendKind, Result};

/// Cap input bytes before tokenization. HF `tokenizers` processes the whole
/// string before any truncation applies (its `with_truncation` runs in
/// post-processing — measured ~2.6s / ~2GB transient for a 10MB input either
/// way), so the only effective bound is up front. 16 bytes per token is far
/// above the real bytes-per-token ratio of these vocabularies, so any text
/// that survives the token-level cap is unaffected.
pub(crate) fn byte_cap(text: &str, max_tokens: usize) -> &str {
    let cap = max_tokens.saturating_mul(16);
    if text.len() <= cap {
        return text;
    }
    let mut end = cap;
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    &text[..end]
}

/// The model backends this build can serve, as manifests and `/v1/models`
/// name them: `coreml` (feature `coreml`, macOS), `onnx` (feature `onnx`)
/// and `static`, always.
pub fn compiled_backends() -> Vec<&'static str> {
    let mut backends = Vec::new();
    if cfg!(all(feature = "coreml", target_os = "macos")) {
        backends.push(EmbeddingBackendKind::Coreml.name());
    }
    if onnx::SUPPORTED {
        backends.push(EmbeddingBackendKind::Onnx.name());
    }
    backends.push(EmbeddingBackendKind::Static.name());
    backends
}

/// Load the right embedder for a registry entry, with default options.
pub fn load_embedder(model: &ResolvedModel) -> Result<Box<dyn sidekick_core::Embedder>> {
    load_embedder_with(model, &LoadOptions::default())
}

/// What the operator chose about how an embedder runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LoadOptions {
    /// Inference threads for an `onnx` model (D40).
    pub onnx_intra_op_threads: usize,
}

impl Default for LoadOptions {
    fn default() -> Self {
        Self { onnx_intra_op_threads: onnx::default_intra_op_threads() }
    }
}

/// Load the right embedder for a registry entry.
pub fn load_embedder_with(model: &ResolvedModel, options: &LoadOptions) -> Result<Box<dyn sidekick_core::Embedder>> {
    match model.manifest.backend {
        EmbeddingBackendKind::Static => Ok(Box::new(StaticEmbedder::load(model)?)),
        EmbeddingBackendKind::Coreml => {
            #[cfg(all(feature = "coreml", target_os = "macos"))]
            {
                Ok(Box::new(CoremlEmbedder::load(model)?))
            }
            #[cfg(not(all(feature = "coreml", target_os = "macos")))]
            {
                Err(sidekick_core::Error::Unavailable(
                    sidekick_core::UnavailableReason::NotSupportedInBuild,
                ))
            }
        }
        EmbeddingBackendKind::Onnx => {
            #[cfg(feature = "onnx")]
            {
                Ok(Box::new(onnx::OnnxEmbedder::load(model, options.onnx_intra_op_threads)?))
            }
            #[cfg(not(feature = "onnx"))]
            {
                let _ = options;
                Err(sidekick_core::Error::Unavailable(sidekick_core::UnavailableReason::NotSupportedInBuild))
            }
        }
    }
}

/// Whether this build can load classifiers: they're Core ML models.
pub const CLASSIFIERS_SUPPORTED: bool = cfg!(all(feature = "coreml", target_os = "macos"));

/// Load a registry classifier. Classifiers are Core ML models; without the
/// `coreml` feature (or off macOS) this is `Unavailable`.
pub fn load_classifier(model: &ResolvedClassifier) -> Result<Box<dyn sidekick_core::Classifier>> {
    #[cfg(all(feature = "coreml", target_os = "macos"))]
    {
        Ok(Box::new(CoremlClassifier::load(model)?))
    }
    #[cfg(not(all(feature = "coreml", target_os = "macos")))]
    {
        let _ = model;
        Err(sidekick_core::Error::Unavailable(
            sidekick_core::UnavailableReason::NotSupportedInBuild,
        ))
    }
}

#[cfg(test)]
mod byte_cap_tests {
    use super::byte_cap;

    #[test]
    fn caps_long_input_at_char_boundary() {
        let short = "hello";
        assert_eq!(byte_cap(short, 512), short, "short input untouched");
        // 3-byte chars: a naive slice at the cap would split one.
        let long = "日".repeat(4000);
        let capped = byte_cap(&long, 4);
        assert!(capped.len() <= 64);
        assert!(capped.chars().all(|c| c == '日'), "cut lands on a boundary");
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn the_compiled_backends_follow_the_features() {
        let backends = super::compiled_backends();
        assert_eq!(backends.contains(&"coreml"), cfg!(all(feature = "coreml", target_os = "macos")));
        assert_eq!(backends.contains(&"onnx"), cfg!(feature = "onnx"));
        assert_eq!(backends.last(), Some(&"static"), "always, and last");
    }
}
