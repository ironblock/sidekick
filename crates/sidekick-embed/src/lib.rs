//! Encoder backends: embedders and classifiers.
//!
//! - [`StaticEmbedder`]: model2vec-style static token embeddings. Pure Rust,
//!   runs anywhere, microsecond lookups. The unconditional floor tier.
//! - [`CoremlEmbedder`] (feature `coreml`, macOS): a Core ML encoder
//!   (EmbeddingGemma-300m, bge-small, MiniLM, …) targeted at the ANE.
//! - [`CoremlClassifier`] (feature `coreml`, macOS): a Core ML classifier,
//!   text-classification, a reranker, or zero-shot in laya's or GLiNER2's
//!   format. Its tokenization ([`InputBuilder`], [`laya`], [`gliner2`]) is
//!   platform-neutral.

pub mod classify_input;
pub mod gliner2;
pub mod laya;
mod pooling;
mod static_embedder;

pub use classify_input::InputBuilder;
pub use pooling::{mean_pool, normalize_in_place};
pub use static_embedder::StaticEmbedder;

#[cfg(all(feature = "coreml", target_os = "macos"))]
mod bucket_models;
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

/// Load the right embedder for a registry entry.
pub fn load_embedder(model: &ResolvedModel) -> Result<Box<dyn sidekick_core::Embedder>> {
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
