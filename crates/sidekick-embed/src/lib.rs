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
