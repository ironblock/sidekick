//! A model's tokenizer on its own, for the daemon's `/tokenize` and
//! `/detokenize`: the same code, and the same copy, its route tokenizes
//! with ([`EmbedInput`] for an embedder, [`InputBuilder`] for a
//! classifier), without the model's weights. Platform-neutral.

use crate::embed_input::EmbedInput;
use crate::InputBuilder;
use sidekick_core::manifest::{ResolvedClassifier, ResolvedModel};
use sidekick_core::{EmbedPurpose, Error, Result};
use std::sync::Arc;

/// One text's tokens, as the model's route counts them, and whether the
/// route takes it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Measured {
    pub ids: Vec<u32>,
    /// Each id's vocabulary entry (`Ġword`, `▁word`, `##ing`, …), when
    /// asked for.
    pub strs: Option<Vec<String>>,
    /// Whether the model's route takes the text whole: `/v1/embeddings`
    /// under `truncate: NONE` (with the given `max_tokens`, as `/v2/embed`
    /// takes it), or `/v1/classify` without `truncate_prompt_tokens`.
    pub fits: bool,
}

/// What to measure a text for.
#[derive(Debug, Clone, Copy, Default)]
pub struct MeasureOptions {
    /// An embedder's prompt prefix; a classifier takes none.
    pub purpose: Option<EmbedPurpose>,
    /// An embedder's `max_tokens`, as `/v2/embed` takes it, for `fits`.
    pub max_tokens: Option<usize>,
    /// Count the special tokens the route adds (vLLM's default).
    pub add_special_tokens: bool,
    pub token_strs: bool,
}

/// An embedder's or a classifier's tokenizer.
pub enum ModelTokenizer {
    Embedder {
        input: Arc<EmbedInput>,
        /// The model's `max_seq_len` (a Core ML model's largest bucket),
        /// after any cap: what every backend measures `NONE` against.
        longest: usize,
    },
    Classifier(Arc<InputBuilder>),
}

impl ModelTokenizer {
    pub fn embedder(model: &ResolvedModel) -> Result<Self> {
        Ok(Self::Embedder { input: EmbedInput::shared(model)?, longest: model.manifest.max_seq_len })
    }

    pub fn classifier(model: &ResolvedClassifier) -> Result<Self> {
        Ok(Self::Classifier(InputBuilder::shared(model)?))
    }

    /// The most tokens an input may have, special tokens (and an
    /// embedder's prompt prefix) included: the limit `/v1/embeddings`
    /// refuses an input past under `truncate: NONE`, or `/v1/classify`
    /// without `truncate_prompt_tokens`. The model's `max_seq_len`, or a
    /// tokenizer.json's own lower limit (an embedder's).
    pub fn max_model_len(&self) -> usize {
        match self {
            Self::Embedder { input, longest } => input.limit(*longest),
            Self::Classifier(builder) => builder.max_seq_len(),
        }
    }

    /// `text`'s tokens, however many, and whether the route takes it, by
    /// the check the route runs ([`EmbedInput::check_whole`],
    /// [`InputBuilder::check_text`]). An embedder's count includes the
    /// purpose's prompt prefix.
    pub fn measure(&self, text: &str, o: MeasureOptions) -> Result<Measured> {
        // The route's verdict, and, when it takes the text, its encoding
        // (special tokens included), which is then the count.
        let verdict = match self {
            Self::Embedder { input, longest } => {
                input.check_whole(text, o.purpose.unwrap_or_default(), o.max_tokens, *longest)
            }
            Self::Classifier(_) if o.purpose.is_some() || o.max_tokens.is_some() => {
                return Err(Error::InvalidRequest(
                    "input_type and max_tokens are embedding models' options; a classifier takes neither".into(),
                ))
            }
            Self::Classifier(builder) => builder.check_text(text),
        };
        let (fits, encoding) = match verdict {
            Ok(encoding) => (true, o.add_special_tokens.then_some(encoding)),
            Err(Error::InvalidRequest(_)) => (false, None),
            Err(e) => return Err(e),
        };
        let encoding = match encoding {
            Some(encoding) => encoding,
            None => match self {
                Self::Embedder { input, .. } => input.tokenize(text, o.purpose.unwrap_or_default(), o.add_special_tokens)?,
                Self::Classifier(builder) => builder.tokenize(text, o.add_special_tokens)?,
            },
        };
        let strs = o.token_strs.then(|| encoding.get_tokens().to_vec());
        Ok(Measured { ids: encoding.get_ids().to_vec(), strs, fits })
    }

    /// Token ids back to text, special tokens kept.
    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        match self {
            Self::Embedder { input, .. } => input.decode(ids),
            Self::Classifier(builder) => builder.decode(ids),
        }
    }
}
