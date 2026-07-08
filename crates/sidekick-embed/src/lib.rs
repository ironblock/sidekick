//! Embedding backends.
//!
//! - [`StaticEmbedder`]: model2vec-style static token embeddings. Pure Rust,
//!   runs anywhere, microsecond lookups. The unconditional floor tier.
//! - [`CoremlEmbedder`] (feature `coreml`, macOS): a Core ML encoder
//!   (EmbeddingGemma-300m, bge-small, MiniLM, …) targeted at the ANE.

mod pooling;
mod static_embedder;

pub use pooling::{mean_pool, normalize_in_place};
pub use static_embedder::StaticEmbedder;

#[cfg(all(feature = "coreml", target_os = "macos"))]
mod coreml_embedder;
#[cfg(all(feature = "coreml", target_os = "macos"))]
pub use coreml_embedder::CoremlEmbedder;

use sidekick_core::manifest::ResolvedModel;
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
