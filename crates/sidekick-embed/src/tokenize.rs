//! A model's tokenizer on its own, for the daemon's `/tokenize` and
//! `/detokenize`: the same code its route tokenizes with
//! ([`EmbedInput`] for an embedder, [`InputBuilder`] for a classifier),
//! loaded without the model's weights. Platform-neutral.

use crate::embed_input::EmbedInput;
use crate::InputBuilder;
use sidekick_core::manifest::{ResolvedClassifier, ResolvedModel};
use sidekick_core::{EmbedPurpose, Error, Result};

/// One text's tokens, as the model's route counts them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Tokens {
    pub ids: Vec<u32>,
    /// Each id's vocabulary entry (`Ġword`, `▁word`, `##ing`, …).
    pub strs: Vec<String>,
}

/// An embedder's or a classifier's tokenizer.
pub enum ModelTokenizer {
    Embedder {
        input: Box<EmbedInput>,
        /// The most tokens an input may have (`max_model_len`).
        limit: usize,
    },
    Classifier(Box<InputBuilder>),
}

impl ModelTokenizer {
    pub fn embedder(model: &ResolvedModel) -> Result<Self> {
        let input = Box::new(EmbedInput::load(model)?);
        // Every backend measures `NONE` against the manifest's
        // `max_seq_len` (a Core ML model's largest bucket), after any cap.
        let limit = input.limit(model.manifest.max_seq_len);
        Ok(Self::Embedder { input, limit })
    }

    pub fn classifier(model: &ResolvedClassifier) -> Result<Self> {
        Ok(Self::Classifier(Box::new(InputBuilder::load(model)?)))
    }

    /// The most tokens an input may have, special tokens (and an
    /// embedder's prompt prefix) included: the limit `/v1/embeddings`
    /// refuses an input past under `truncate: NONE`, or `/v1/classify`
    /// without `truncate_prompt_tokens`. The model's `max_seq_len`, or a
    /// tokenizer.json's own lower limit (an embedder's).
    pub fn max_model_len(&self) -> usize {
        match self {
            Self::Embedder { limit, .. } => *limit,
            Self::Classifier(builder) => builder.max_seq_len(),
        }
    }

    /// `text`'s tokens, however many: an embedder's with `purpose`'s
    /// prompt prefix, and the special tokens its route adds unless
    /// `add_special_tokens` is false. A classifier takes no purpose.
    pub fn tokenize(&self, text: &str, purpose: Option<EmbedPurpose>, add_special_tokens: bool) -> Result<Tokens> {
        let encoding = match self {
            Self::Embedder { input, .. } => input.tokenize(text, purpose.unwrap_or_default(), add_special_tokens)?,
            Self::Classifier(_) if purpose.is_some() => {
                return Err(Error::InvalidRequest(
                    "input_type selects an embedding model's prompt prefix; a classifier has none".into(),
                ))
            }
            Self::Classifier(builder) => builder.tokenize(text, add_special_tokens)?,
        };
        Ok(Tokens { ids: encoding.get_ids().to_vec(), strs: encoding.get_tokens().to_vec() })
    }

    /// Token ids back to text, special tokens kept.
    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        match self {
            Self::Embedder { input, .. } => input.decode(ids),
            Self::Classifier(builder) => builder.decode(ids),
        }
    }
}
