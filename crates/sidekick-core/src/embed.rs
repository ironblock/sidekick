use crate::Result;
use std::collections::BTreeMap;

/// What the caller will do with the vectors, which picks the prompt prefix
/// a backend puts before each text. Models like EmbeddingGemma use
/// different prompts for queries and documents (the manifest's
/// `[prefixes]`, D9), and some name prompts for other tasks (its
/// `[prompts]`, D45). Non-exhaustive: a match outside this crate needs an
/// arm for purposes added later.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum EmbedPurpose<'a> {
    /// The manifest's document prefix: the default.
    #[default]
    Document,
    /// The manifest's query prefix.
    Query,
    /// No prefix: the text exactly as sent, for a caller that brings its
    /// own prompt (the daemon's `input_type: "none"`). The tokenizer's
    /// special tokens are still added.
    Raw,
    /// One of the manifest's named prompts (`[prompts]`), by its exact
    /// name (the daemon's `prompt_name`, as in Hugging Face Text
    /// Embeddings Inference). A name the model doesn't have is an
    /// `InvalidRequest` ([`named_prompt`]).
    Named(&'a str),
}

/// The text of `model`'s prompt named `name` among its `prompts` (the
/// manifest's `[prompts]`), matched exactly, case included, as
/// sentence-transformers and Text Embeddings Inference match it. Any other
/// name is an `InvalidRequest` that lists the model's names.
pub fn named_prompt<'p>(model: &str, prompts: &'p BTreeMap<String, String>, name: &str) -> Result<&'p str> {
    if let Some(text) = prompts.get(name) {
        return Ok(text);
    }
    let message = if prompts.is_empty() {
        format!(
            "model `{model}` has no named prompts, so prompt_name `{name}` isn't one of them \
             (send input_type instead, or input_type `none` with your own prompt in the text)"
        )
    } else {
        let names: Vec<String> = prompts.keys().map(|k| format!("`{k}`")).collect();
        format!("model `{model}` has no prompt named `{name}` (its prompts: {})", names.join(", "))
    };
    Err(crate::Error::InvalidRequest(message))
}

/// Which end of an over-long input to keep (Cohere's `truncate`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Truncate {
    /// Keep the start (every embedder's behavior without limits).
    #[default]
    End,
    /// Keep the end.
    Start,
    /// Don't truncate: an over-long input is an `InvalidRequest`.
    Reject,
}

/// Per-request input limits (Cohere `/v2/embed`'s `truncate` and
/// `max_tokens`). The default is what [`Embedder::embed`] does.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct EmbedLimits {
    pub truncate: Truncate,
    /// At most this many tokens (special tokens included); `None`: the
    /// model's maximum.
    pub max_tokens: Option<usize>,
}

/// A text-embedding backend. Implementations: Core ML encoder on ANE,
/// static (model2vec-style) CPU floor tier.
///
/// Synchronous by design — Core ML predictions and static lookups are both
/// blocking; async callers wrap in `spawn_blocking`.
pub trait Embedder: Send + Sync {
    /// Stable identifier, used as the OpenAI `model` name.
    fn id(&self) -> &str;

    /// Native output dimensionality.
    fn dims(&self) -> usize;

    /// [`embed_with`](Self::embed_with), plus the sequence-length bucket
    /// each input ran in, in input order: the compiled program that
    /// answered it (`sidekick-buckets`). `None` for a backend without
    /// buckets.
    #[allow(clippy::type_complexity)]
    fn embed_bucketed(
        &self,
        texts: &[&str],
        purpose: EmbedPurpose,
        limits: EmbedLimits,
    ) -> Result<(Vec<Vec<f32>>, Option<Vec<usize>>)> {
        Ok((self.embed_with(texts, purpose, limits)?, None))
    }

    /// [`embed_bucketed`](Self::embed_bucketed), calling `loaded` once the
    /// inputs are prepared and the buckets they need are loaded, before any
    /// prediction, with the buckets it waited on a load for (empty when all
    /// were resident). A bucket's first load compiles it, which can take
    /// minutes, so the daemon starts a request's deadline there. Backends
    /// without buckets call it first.
    #[allow(clippy::type_complexity)]
    fn embed_staged(
        &self,
        texts: &[&str],
        purpose: EmbedPurpose,
        limits: EmbedLimits,
        loaded: &mut dyn FnMut(&[usize]),
    ) -> Result<(Vec<Vec<f32>>, Option<Vec<usize>>)> {
        loaded(&[]);
        self.embed_bucketed(texts, purpose, limits)
    }

    /// Dimensions this model was trained to truncate to (Matryoshka), largest
    /// first, including the native size. Empty means truncation is lossy and
    /// unsupported.
    fn matryoshka_dims(&self) -> &[usize] {
        &[]
    }

    /// Embed a batch. Returns one unit-normalized vector of `dims()` length
    /// per input, in order.
    fn embed(&self, texts: &[&str], purpose: EmbedPurpose) -> Result<Vec<Vec<f32>>>;

    /// [`embed`](Self::embed) with per-request limits. Backends that can't
    /// honor limits other than the default refuse them.
    fn embed_with(
        &self,
        texts: &[&str],
        purpose: EmbedPurpose,
        limits: EmbedLimits,
    ) -> Result<Vec<Vec<f32>>> {
        if limits == EmbedLimits::default() {
            return self.embed(texts, purpose);
        }
        Err(crate::Error::InvalidRequest(format!(
            "model `{}` supports only truncate `END` without `max_tokens`",
            self.id()
        )))
    }
}

/// Truncate a unit vector to `dims` and re-normalize (Matryoshka truncation).
/// Callers must check `matryoshka_dims()` before using this on model output.
pub fn truncate_normalized(v: &[f32], dims: usize) -> Vec<f32> {
    let mut out: Vec<f32> = v[..dims.min(v.len())].to_vec();
    let norm = out.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in &mut out {
            *x /= norm;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_named_prompt_is_matched_exactly() {
        let prompts: BTreeMap<String, String> =
            [("STS", "task: sentence similarity | query: "), ("document", "title: none | text: ")]
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .into();
        assert_eq!(named_prompt("m", &prompts, "STS").unwrap(), "task: sentence similarity | query: ");
        let message = |e| match e {
            crate::Error::InvalidRequest(m) => m,
            e => panic!("not a 400: {e:?}"),
        };
        assert_eq!(
            message(named_prompt("m", &prompts, "sts").unwrap_err()),
            "model `m` has no prompt named `sts` (its prompts: `STS`, `document`)"
        );
        let none = BTreeMap::new();
        assert!(message(named_prompt("m", &none, "STS").unwrap_err()).starts_with("model `m` has no named prompts"));
    }

    #[test]
    fn truncation_renormalizes() {
        let v = vec![3.0, 4.0, 12.0];
        let t = truncate_normalized(&v, 2);
        assert_eq!(t.len(), 2);
        let norm = t.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-6);
        // Direction preserved: 3:4 ratio.
        assert!((t[0] / t[1] - 0.75).abs() < 1e-6);
    }

    #[test]
    fn truncation_handles_zero_vector() {
        let t = truncate_normalized(&[0.0, 0.0], 2);
        assert_eq!(t, vec![0.0, 0.0]);
    }
}
