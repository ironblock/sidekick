//! One embedding input as token ids: the purpose's prompt prefix, the
//! tokenizer, and truncation (`max_tokens`, `truncate`). Shared by every
//! embedding backend, which then differ only in how they run the ids
//! (D40), and by the daemon's `/tokenize`. Platform-neutral.

use crate::shared::Shared;
use sidekick_core::manifest::ResolvedModel;
use sidekick_core::{EmbedLimits, EmbedPurpose, EmbeddingBackendKind, Error, Result, Truncate};
use std::sync::{Arc, LazyLock};
use tokenizers::{Tokenizer, TruncationDirection};

/// Every model's input in use, shared by its backend and `/tokenize`.
static INPUTS: LazyLock<Shared<EmbedInput>> = LazyLock::new(Shared::default);

/// Under truncate `NONE`, a text longer than this many bytes per token of
/// its limit is refused without being tokenized. Generous, so that text
/// with long runs of whitespace or punctuation (indented code, padded
/// tables) is still measured in tokens. `END` and `START` cap the bytes
/// they tokenize at 16 per token instead (`byte_cap`).
pub(crate) const REJECT_BYTES_PER_TOKEN: usize = 64;

/// A model's tokenizer and prompt prefixes.
pub struct EmbedInput {
    /// The model's id, for errors.
    model: String,
    pub(crate) tokenizers: Tokenizers,
    prefix_query: String,
    prefix_document: String,
    /// Whether the model's inputs carry the tokenizer's special tokens. A
    /// static model's don't: it averages the text's own token rows.
    special_tokens: bool,
    /// Whether an input with no tokens is a 400. A Core ML or ONNX model
    /// can't run an empty sequence; a static model embeds it (as zeros).
    refuse_empty: bool,
    /// Special tokens the tokenizer adds around one input (0 without them).
    added_tokens: usize,
}

impl EmbedInput {
    /// The model's input, as its backend embeds with it: the copy already
    /// in use (by its backend, or by the daemon's `/tokenize`), or a new
    /// one. Every backend and `/tokenize` get it this way, so the two count
    /// alike and hold one tokenizer between them.
    pub fn shared(model: &ResolvedModel) -> Result<Arc<Self>> {
        INPUTS.get_or_load(&model.dir, crate::shared::fingerprint(&model.manifest), || Self::load(model))
    }

    /// A copy of its own, outside the cache.
    pub fn load(model: &ResolvedModel) -> Result<Self> {
        let m = &model.manifest;
        let tokenizer = Tokenizer::from_file(model.tokenizer_path())
            .map_err(|e| Error::Tokenizer(e.to_string()))?;
        let is_static = m.backend == EmbeddingBackendKind::Static;
        let mut input = Self::with_specials(&m.id, tokenizer, &m.prefixes.query, &m.prefixes.document, !is_static)?;
        input.refuse_empty = !is_static;
        Ok(input)
    }

    #[cfg(test)]
    fn new(model: &str, tokenizer: Tokenizer, prefix_query: &str, prefix_document: &str) -> Result<Self> {
        Self::with_specials(model, tokenizer, prefix_query, prefix_document, true)
    }

    fn with_specials(
        model: &str,
        mut tokenizer: Tokenizer,
        prefix_query: &str,
        prefix_document: &str,
        special_tokens: bool,
    ) -> Result<Self> {
        // A tokenizer.json may pad every input to a fixed length
        // (all-MiniLM ships padding to 128), and every returned token is
        // marked real in the attention mask: the pads would be read as
        // text. One input is never padded; a backend pads if it must.
        tokenizer.with_padding(None);
        // Measured with padding off, or a padding tokenizer would count its
        // pads as added tokens.
        let added_tokens = tokenizer
            .encode("", special_tokens)
            .map_err(|e| Error::Tokenizer(e.to_string()))?
            .len();
        Ok(Self {
            model: model.to_string(),
            tokenizers: Tokenizers::new(tokenizer)?,
            prefix_query: prefix_query.to_string(),
            prefix_document: prefix_document.to_string(),
            special_tokens,
            refuse_empty: true,
            added_tokens,
        })
    }

    pub(crate) fn prefix(&self, purpose: EmbedPurpose) -> &str {
        match purpose {
            EmbedPurpose::Query => &self.prefix_query,
            EmbedPurpose::Document => &self.prefix_document,
        }
    }

    /// The checks of `limits` that don't depend on the input, so that a
    /// backend refuses a request for them before naming any input: a
    /// `max_tokens` of 0, or one that leaves `START` no room for text after
    /// the prompt prefix. `ids` makes them too.
    pub fn check_limits(&self, purpose: EmbedPurpose, limits: EmbedLimits, longest: usize) -> Result<()> {
        if limits.max_tokens == Some(0) {
            return Err(Error::InvalidRequest("max_tokens must be at least 1".into()));
        }
        let prefix = self.prefix(purpose);
        if limits.truncate == Truncate::Start && !prefix.is_empty() {
            let max = Limit::new(self.tokenizers.limit(longest), limits.max_tokens).tokens();
            let tokens = encode(self.tokenizers.whole(), prefix, false)?.len();
            if tokens >= max.saturating_sub(self.added_tokens) {
                return Err(no_room(max, tokens));
            }
        }
        Ok(())
    }

    /// Apply the purpose's prefix, tokenize, and truncate to at most
    /// `max_tokens` and `longest`: keeping the start (`END`, the final
    /// token preserved), the end (`START`: the first token preserved, since
    /// a tokenizer's leading special token is what CLS pooling reads), or a
    /// 400 (`NONE`). The Core ML and ONNX backends' input; the static
    /// backend truncates its own way and shares only `NONE`
    /// (`whole`).
    pub fn ids(&self, text: &str, purpose: EmbedPurpose, limits: EmbedLimits, longest: usize) -> Result<Vec<i32>> {
        if limits.max_tokens == Some(0) {
            return Err(Error::InvalidRequest("max_tokens must be at least 1".into()));
        }
        let prefix = self.prefix(purpose);
        // Byte caps bound tokenizer work on the text; the prefix stays whole.
        let ids = match limits.truncate {
            // With the tokenizer as shipped, its own truncation included:
            // models' parity references were graded with it.
            Truncate::End => {
                let max = limits.max_tokens.unwrap_or(longest).min(longest);
                let text = format!("{prefix}{}", crate::byte_cap(text, max));
                truncate_preserving_last(encode(self.tokenizers.shipped(), &text, self.special_tokens)?.get_ids(), max)
            }
            // The rest see the whole text, under the tokenizer's own limit
            // as well as the model's.
            Truncate::Start => {
                let max = Limit::new(self.tokenizers.limit(longest), limits.max_tokens).tokens();
                let text = format!("{prefix}{}", crate::classify_input::byte_cap_end(text, max));
                if prefix.is_empty() {
                    truncate_preserving_first(encode(self.tokenizers.whole(), &text, true)?.get_ids(), max)
                } else {
                    // The model's prompt prefix stays whole: the model was
                    // never trained without it.
                    keep_prefix_and_end(self.tokenizers.whole(), self.added_tokens, &text, prefix.len(), max)?
                }
            }
            Truncate::Reject => self.whole(text, purpose, limits, longest)?.iter().map(|&u| u as i32).collect(),
        };
        // A tokenizer that adds no special tokens gives nothing for an empty
        // text: a bad request, not a model failure.
        if ids.is_empty() {
            return Err(no_tokens());
        }
        Ok(ids)
    }

    /// `NONE`: the whole input's ids, or a 400 ([`check_whole`](Self::check_whole)).
    pub(crate) fn whole(&self, text: &str, purpose: EmbedPurpose, limits: EmbedLimits, longest: usize) -> Result<Vec<u32>> {
        Ok(self.check_whole(text, purpose, limits.max_tokens, longest)?.get_ids().to_vec())
    }

    /// `NONE`'s verdict on one input: its whole encoding, special tokens
    /// and prompt prefix included, or the 400 every backend refuses it
    /// with. A `max_tokens` of 0; a text of more than
    /// [`REJECT_BYTES_PER_TOKEN`] bytes per token of the limit (the lower
    /// of `longest`, the tokenizer's own limit and `max_tokens`), refused
    /// without being tokenized; more tokens than the limit; or none, for a
    /// model that can't run an empty input. The one check behind every
    /// backend's `NONE` and the daemon's `/tokenize` `fits`.
    pub fn check_whole(
        &self,
        text: &str,
        purpose: EmbedPurpose,
        max_tokens: Option<usize>,
        longest: usize,
    ) -> Result<tokenizers::Encoding> {
        if max_tokens == Some(0) {
            return Err(Error::InvalidRequest("max_tokens must be at least 1".into()));
        }
        let limit = Limit::new(self.tokenizers.limit(longest), max_tokens);
        if text.len() > limit.tokens().saturating_mul(REJECT_BYTES_PER_TOKEN) {
            return Err(too_big(&self.model, text.len(), limit));
        }
        let encoding = self.tokenize(text, purpose, true)?;
        if encoding.len() > limit.tokens() {
            return Err(too_long(&self.model, encoding.len(), limit));
        }
        if encoding.is_empty() && self.refuse_empty {
            return Err(no_tokens());
        }
        Ok(encoding)
    }

    /// The whole input as `NONE` counts it, however long: the purpose's
    /// prompt prefix, then the text, by the tokenizer without its own
    /// truncation, with the special tokens the model's inputs carry (none
    /// for a static model) unless `add_special_tokens` is false. The
    /// daemon's `/tokenize` reports this, so its count is the one
    /// `/v1/embeddings` compares with the limit. No byte bound: the caller
    /// bounds the text.
    pub fn tokenize(&self, text: &str, purpose: EmbedPurpose, add_special_tokens: bool) -> Result<tokenizers::Encoding> {
        let text = format!("{}{text}", self.prefix(purpose));
        encode(self.tokenizers.whole(), &text, add_special_tokens && self.special_tokens)
    }

    /// The most tokens `NONE` accepts for a model whose limit is `longest`:
    /// that, or the tokenizer's own limit when its tokenizer.json
    /// truncates at fewer.
    pub fn limit(&self, longest: usize) -> usize {
        self.tokenizers.limit(longest)
    }

    /// Token ids back to text, special tokens kept. An id outside the
    /// vocabulary is a 400.
    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        decode(self.tokenizers.whole(), ids)
    }
}

/// The 400 for an input with no tokens, which a Core ML or ONNX model
/// can't run.
fn no_tokens() -> Error {
    Error::InvalidRequest("the input has no tokens to embed".into())
}

/// `ids` as text by `tokenizer`, special tokens kept, or a 400 naming the
/// first id it has no token for.
pub(crate) fn decode(tokenizer: &Tokenizer, ids: &[u32]) -> Result<String> {
    if let Some(id) = ids.iter().find(|&&id| tokenizer.id_to_token(id).is_none()) {
        return Err(Error::InvalidRequest(format!(
            "token id {id} isn't in the model's vocabulary of {} tokens",
            tokenizer.get_vocab_size(true)
        )));
    }
    tokenizer.decode(ids, false).map_err(|e| Error::Tokenizer(e.to_string()))
}

/// A model's tokenizer as its tokenizer.json ships it, and, when that
/// truncates on its own (LFM2.5's does, at 512 tokens), a copy that
/// doesn't. `END` uses the tokenizer as shipped: models' parity references
/// were graded with it. `START` must see the end of the text and `NONE`
/// its whole length, so they use the copy, and apply the tokenizer's limit
/// themselves.
pub(crate) struct Tokenizers {
    shipped: Tokenizer,
    /// The copy without truncation, and the shipped tokenizer's limit.
    untruncated: Option<(Tokenizer, usize)>,
}

impl Tokenizers {
    pub(crate) fn new(shipped: Tokenizer) -> Result<Self> {
        let untruncated = match shipped.get_truncation() {
            None => None,
            Some(t) => {
                let max_length = t.max_length;
                let mut whole = shipped.clone();
                whole.with_truncation(None).map_err(|e| Error::Tokenizer(e.to_string()))?;
                Some((whole, max_length))
            }
        };
        Ok(Self { shipped, untruncated })
    }

    /// The tokenizer as shipped, for `END`.
    pub(crate) fn shipped(&self) -> &Tokenizer {
        &self.shipped
    }

    /// The tokenizer without its own truncation, for `START` and `NONE`.
    pub(crate) fn whole(&self) -> &Tokenizer {
        self.untruncated.as_ref().map_or(&self.shipped, |(t, _)| t)
    }

    /// The lower of the model's `longest` and the tokenizer's own limit.
    pub(crate) fn limit(&self, longest: usize) -> usize {
        self.untruncated.as_ref().map_or(longest, |&(_, max_length)| max_length.min(longest))
    }
}

/// What an over-length input is measured against: the model's limit, or a
/// lower `max_tokens` the request set.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Limit {
    Model(usize),
    MaxTokens(usize),
}

impl Limit {
    /// The lower of the model's `longest` and the request's `max_tokens`.
    pub(crate) fn new(longest: usize, max_tokens: Option<usize>) -> Self {
        match max_tokens {
            Some(n) if n < longest => Limit::MaxTokens(n),
            _ => Limit::Model(longest),
        }
    }

    pub(crate) fn tokens(self) -> usize {
        match self {
            Limit::Model(n) | Limit::MaxTokens(n) => n,
        }
    }

    fn describe(self, model: &str) -> String {
        match self {
            Limit::Model(n) => format!("model `{model}`'s limit of {n} tokens"),
            Limit::MaxTokens(n) => format!("max_tokens {n}"),
        }
    }
}

pub(crate) fn encode(tokenizer: &Tokenizer, text: &str, add_special_tokens: bool) -> Result<tokenizers::Encoding> {
    tokenizer.encode(text, add_special_tokens).map_err(|e| Error::Tokenizer(e.to_string()))
}

/// At most `max` tokens of `text` (whose first `prefix_bytes` are the
/// prompt prefix), special tokens included: the prefix's tokens, then
/// the end of the rest. `tokenizer` mustn't truncate on its own.
fn keep_prefix_and_end(
tokenizer: &Tokenizer,
added_tokens: usize,
text: &str,
prefix_bytes: usize,
max: usize,
) -> Result<Vec<i32>> {
    let tok_err = |e: tokenizers::Error| Error::Tokenizer(e.to_string());
    let mut encoding = tokenizer.encode(text, false).map_err(tok_err)?;
    let budget = max.saturating_sub(added_tokens);
    let prefix = encoding.get_offsets().iter().take_while(|(start, _)| *start < prefix_bytes).count();
    if prefix >= budget {
        return Err(no_room(max, prefix));
    }
    if encoding.len() > budget {
        let mut head = encoding.clone();
        head.truncate(prefix, 0, TruncationDirection::Right);
        encoding.truncate(budget - prefix, 0, TruncationDirection::Left);
        // What truncation set aside isn't needed, and merging carries it.
        head.take_overflowing();
        encoding.take_overflowing();
        head.merge_with(encoding, false);
        encoding = head;
    }
    let encoding = tokenizer.post_process(encoding, None, true).map_err(tok_err)?;
    Ok(encoding.get_ids().iter().map(|&u| u as i32).collect())
}

/// The 400 for a `START` limit of `max` tokens that the prompt prefix's
/// `prefix` tokens fill.
pub(crate) fn no_room(max: usize, prefix: usize) -> Error {
    Error::InvalidRequest(format!(
        "max_tokens {max} leaves no room for text after the model's {prefix}-token prompt prefix"
    ))
}

/// Truncate token ids to `max`, but PRESERVE THE FINAL TOKEN. Tokenizers that
/// append a terminator (EOS/SEP) put it last, and last-token pooling reads
/// exactly that token — a naive `take(max)` that drops it silently corrupts
/// any input longer than the biggest bucket (measured: 0.36 vs 1.0 cosine on
/// an over-length doc through an F2LLM last-token model). Keeping
/// `[first max-1, last]` matches HF's right-truncation and is harmless for
/// CLS/mean pooling (one dropped interior token). Only over-length inputs are
/// touched. `max` ≥ 1: a bucket size, or a `max_tokens` that `prepare_with`
/// checked, so `max - 1` never underflows.
fn truncate_preserving_last(raw: &[u32], max: usize) -> Vec<i32> {
    if raw.len() > max {
        raw[..max - 1]
            .iter()
            .chain(std::iter::once(&raw[raw.len() - 1]))
            .map(|&u| u as i32)
            .collect()
    } else {
        raw.iter().map(|&u| u as i32).collect()
    }
}

/// Truncate token ids to `max` keeping the END of the input, but PRESERVE
/// THE FIRST TOKEN (a tokenizer's leading special token). `max` ≥ 1, as for
/// `truncate_preserving_last`.
fn truncate_preserving_first(raw: &[u32], max: usize) -> Vec<i32> {
    if raw.len() > max {
        std::iter::once(&raw[0])
            .chain(&raw[raw.len() - (max - 1)..])
            .map(|&u| u as i32)
            .collect()
    } else {
        raw.iter().map(|&u| u as i32).collect()
    }
}

/// `prepare` over each text, in order, stopping at the first error. The
/// backends check the request's limits first (`check_limits`), then
/// prepare every input before they run any, so a rejected input refuses
/// the whole request at no inference cost. A 400 names the input's index
/// in the batch.
pub(crate) fn each_input<T>(texts: &[&str], mut prepare: impl FnMut(&str) -> Result<T>) -> Result<Vec<T>> {
    texts
        .iter()
        .enumerate()
        .map(|(i, text)| {
            prepare(text).map_err(|e| match e {
                Error::InvalidRequest(m) => Error::InvalidRequest(format!("input {i}: {m}")),
                e => e,
            })
        })
        .collect()
}

/// The 400 for an input of `tokens` tokens, over `limit` under truncate
/// `NONE`.
pub(crate) fn too_long(model: &str, tokens: usize, limit: Limit) -> Error {
    Error::InvalidRequest(format!(
        "{tokens} tokens, over {}, and truncate is NONE (send truncate: END or START to truncate it)",
        limit.describe(model)
    ))
}

/// The 400 for a text of `bytes` bytes, more than `NONE` tokenizes for
/// `limit` ([`REJECT_BYTES_PER_TOKEN`]): a bound on its size, not a count.
pub(crate) fn too_big(model: &str, bytes: usize, limit: Limit) -> Error {
    Error::InvalidRequest(format!(
        "{bytes} bytes, more than the {} bytes sidekick reads for {}, and truncate is NONE \
         (send truncate: END or START to truncate it)",
        limit.tokens().saturating_mul(REJECT_BYTES_PER_TOKEN),
        limit.describe(model)
    ))
}

#[cfg(test)]
mod tests {
    use super::{truncate_preserving_last, EmbedInput};
    use sidekick_core::{EmbedLimits, EmbedPurpose, Error, Truncate};
    use tokenizers::TruncationParams;

    const NONE: EmbedLimits = EmbedLimits { truncate: Truncate::Reject, max_tokens: None };
    const START: EmbedLimits = EmbedLimits { truncate: Truncate::Start, max_tokens: None };
    const END: EmbedLimits = EmbedLimits { truncate: Truncate::End, max_tokens: None };

    /// laya's test tokenizer ([CLS] text [SEP]), truncating to `at` tokens
    /// on its own when given, as LFM2.5's tokenizer.json does at 512.
    fn tokenizer(at: Option<usize>) -> tokenizers::Tokenizer {
        let mut tok = crate::laya::tests::tokenizer();
        let truncation = at.map(|max_length| TruncationParams { max_length, ..Default::default() });
        tok.with_truncation(truncation).unwrap();
        tok
    }

    fn input(at: Option<usize>) -> EmbedInput {
        EmbedInput::new("tiny", tokenizer(at), "", "").unwrap()
    }

    fn message(e: Error) -> String {
        match e {
            Error::InvalidRequest(m) => m,
            e => panic!("not a 400: {e:?}"),
        }
    }

    #[test]
    fn truncate_none_refuses_an_input_over_the_limit() {
        // "a b c d e f": 8 tokens with [CLS] and [SEP].
        let e = input(None).ids("a b c d e f", EmbedPurpose::Document, NONE, 6).unwrap_err();
        assert_eq!(
            message(e),
            "8 tokens, over model `tiny`'s limit of 6 tokens, and truncate is NONE (send truncate: END or START to truncate it)"
        );
        assert_eq!(input(None).ids("a b c d", EmbedPurpose::Document, NONE, 6).unwrap().len(), 6, "at the limit");
        // A lower max_tokens is the request's limit, and says so.
        let limits = EmbedLimits { max_tokens: Some(4), ..NONE };
        let e = input(None).ids("a b c d", EmbedPurpose::Document, limits, 6).unwrap_err();
        assert!(message(e).starts_with("6 tokens, over max_tokens 4, and truncate is NONE"));
    }

    #[test]
    fn truncate_none_refuses_an_input_the_tokenizer_would_cut() {
        // The tokenizer keeps 5 of 8 tokens, under the model's 32: END
        // runs what it kept, and NONE refuses, naming the untruncated
        // length and the tokenizer's limit.
        let tok = input(Some(5));
        let end = tok.ids("a b c d e f", EmbedPurpose::Document, END, 32).unwrap();
        assert_eq!(end.len(), 5);
        let e = tok.ids("a b c d e f", EmbedPurpose::Document, NONE, 32).unwrap_err();
        assert!(message(e).starts_with("8 tokens, over model `tiny`'s limit of 5 tokens"));
        // Within the tokenizer's limit: nothing cut, nothing refused.
        assert_eq!(tok.ids("a b c", EmbedPurpose::Document, NONE, 32).unwrap().len(), 5);
        // A tokenizer limit above the model's: the model's applies.
        let e = input(Some(64)).ids("a b c d e f", EmbedPurpose::Document, NONE, 6).unwrap_err();
        assert!(message(e).starts_with("8 tokens, over model `tiny`'s limit of 6 tokens"));
    }

    #[test]
    fn tokenize_counts_what_none_compares_with_the_limit() {
        // A tokenizer that keeps 5 tokens on its own: tokenize counts the
        // whole text, the number NONE's 400 names, and limit is the 5.
        let tok = EmbedInput::new("tiny", tokenizer(Some(5)), "", "question : ").unwrap();
        let text = "a b c d e f";
        assert_eq!(tok.tokenize(text, EmbedPurpose::Document, true).unwrap().len(), 10);
        assert_eq!(tok.tokenize(text, EmbedPurpose::Query, true).unwrap().len(), 8, "no query prefix");
        assert_eq!(tok.tokenize(text, EmbedPurpose::Document, false).unwrap().len(), 8, "no special tokens");
        assert_eq!(tok.limit(32), 5);
        let e = tok.ids(text, EmbedPurpose::Document, NONE, 32).unwrap_err();
        assert!(message(e).starts_with("10 tokens, over model `tiny`'s limit of 5 tokens"));
        // At the limit, NONE's ids are tokenize's.
        let at = tok.tokenize("a", EmbedPurpose::Document, true).unwrap();
        let ids: Vec<i32> = at.get_ids().iter().map(|&u| u as i32).collect();
        assert_eq!((ids.len(), tok.ids("a", EmbedPurpose::Document, NONE, 32).unwrap()), (5, ids));
        // NONE's verdict, the one /tokenize's `fits` reports: past the
        // byte bound (one token: a run without spaces), and no tokens.
        let e = tok.check_whole(&"x".repeat(5 * 64 + 1), EmbedPurpose::Query, None, 32).unwrap_err();
        assert!(message(e).starts_with("321 bytes, more than the 320 bytes"));
        assert_eq!(tok.check_whole("a", EmbedPurpose::Query, Some(3), 32).unwrap().len(), 3);
        assert!(tok.check_whole("a b", EmbedPurpose::Query, Some(3), 32).is_err(), "max_tokens");
        let bare = EmbedInput::with_specials("tiny", tokenizer(None), "", "", false).unwrap();
        assert_eq!(message(bare.check_whole("", EmbedPurpose::Document, None, 32).unwrap_err()), "the input has no tokens to embed");
        // A static model's inputs carry no special tokens.
        let plain = EmbedInput::with_specials("tiny", tokenizer(None), "", "", false).unwrap();
        assert_eq!(plain.tokenize(text, EmbedPurpose::Document, true).unwrap().len(), 6);
        assert_eq!(plain.decode(&[1, 2]).unwrap(), "[CLS] [SEP]");
        assert!(message(plain.decode(&[9999]).unwrap_err()).starts_with("token id 9999 isn't in the model's vocabulary"));
    }

    #[test]
    fn truncate_none_measures_in_bytes_only_far_past_the_limit() {
        // 6 tokens' limit: up to 384 bytes are tokenized, here 4 tokens.
        let padded = format!("a{}b", " ".repeat(300));
        assert_eq!(input(None).ids(&padded, EmbedPurpose::Document, NONE, 6).unwrap().len(), 4);
        let e = input(None).ids(&"a ".repeat(200), EmbedPurpose::Document, NONE, 6).unwrap_err();
        assert!(
            message(e).starts_with("400 bytes, more than the 384 bytes sidekick reads for model `tiny`'s limit of 6 tokens"),
        );
    }

    #[test]
    fn start_keeps_the_end_when_the_tokenizer_truncates() {
        let t = |w| crate::laya::tests::tokenizer().token_to_id(w).unwrap() as i32;
        // The tokenizer keeps 5 tokens on its own: START keeps the text's
        // last 3 words within that, where the tokenizer would keep its first.
        let text = "a b c d e f g h";
        let start = input(Some(5)).ids(text, EmbedPurpose::Document, START, 32).unwrap();
        assert_eq!(start, vec![1, t("f"), t("g"), t("h"), 2]);
        // The same with a prompt prefix, kept whole: the tokenizer's 6.
        let prefixed = EmbedInput::new("tiny", tokenizer(Some(6)), "", "question : ").unwrap();
        let start = prefixed.ids(text, EmbedPurpose::Document, START, 32).unwrap();
        assert_eq!(start, vec![1, t("question"), t(":"), t("g"), t("h"), 2]);
        // END is the tokenizer's own truncation, as it was.
        let end = input(Some(5)).ids(text, EmbedPurpose::Document, END, 32).unwrap();
        assert_eq!(end, vec![1, t("a"), t("b"), t("c"), 2]);
        // Without truncation in the tokenizer, START is as it was.
        let start = input(None).ids(text, EmbedPurpose::Document, START, 5).unwrap();
        assert_eq!(start, vec![1, t("f"), t("g"), t("h"), 2]);
    }

    #[test]
    fn request_wide_limits_are_checked_before_any_input() {
        let prefixed = EmbedInput::new("tiny", tokenizer(None), "", "question : ").unwrap();
        let tight = EmbedLimits { max_tokens: Some(4), ..START };
        let e = prefixed.check_limits(EmbedPurpose::Document, tight, 32).unwrap_err();
        assert_eq!(message(e), "max_tokens 4 leaves no room for text after the model's 2-token prompt prefix");
        assert!(prefixed.check_limits(EmbedPurpose::Document, START, 32).is_ok());
        assert!(prefixed.check_limits(EmbedPurpose::Query, tight, 32).is_ok(), "no query prefix");
        let zero = EmbedLimits { max_tokens: Some(0), ..END };
        assert!(prefixed.check_limits(EmbedPurpose::Query, zero, 32).is_err());
    }

    #[test]
    fn a_refused_input_is_named_by_its_index() {
        let tok = input(None);
        let e = super::each_input(&["a", "a b c d e f", "b"], |t| tok.ids(t, EmbedPurpose::Document, NONE, 6)).unwrap_err();
        assert!(message(e).starts_with("input 1: 8 tokens, over model `tiny`'s limit of 6 tokens"));
    }

    #[test]
    fn keeps_all_when_within_max() {
        assert_eq!(truncate_preserving_last(&[1, 2, 3], 5), vec![1, 2, 3]);
        assert_eq!(truncate_preserving_last(&[1, 2, 3], 3), vec![1, 2, 3]);
    }

    #[test]
    fn start_truncation_keeps_the_prompt_prefix_whole() {
        let tok = crate::laya::tests::tokenizer();
        let t = |w| tok.token_to_id(w).unwrap() as i32;
        // "question : " is the prefix (2 tokens), then 6 words of text: at 6
        // tokens, [CLS] + prefix + the text's last 2 words + [SEP].
        let text = "question : a b c d e f";
        let ids = super::keep_prefix_and_end(&tok, 2, text, "question : ".len(), 6).unwrap();
        assert_eq!(ids, vec![1, t("question"), t(":"), t("e"), t("f"), 2]);
        // Short enough: untouched.
        let ids = super::keep_prefix_and_end(&tok, 2, "question : a", "question : ".len(), 6).unwrap();
        assert_eq!(ids, vec![1, t("question"), t(":"), t("a"), 2]);
        // No room for text after the prefix.
        let e = super::keep_prefix_and_end(&tok, 2, text, "question : ".len(), 4).unwrap_err();
        assert!(e.to_string().contains("2-token prompt prefix"), "{e}");
    }

    #[test]
    fn start_truncation_keeps_the_first_token_and_the_end() {
        let raw = [99, 10, 11, 12, 13, 98];
        assert_eq!(super::truncate_preserving_first(&raw, 3), vec![99, 13, 98]);
        assert_eq!(super::truncate_preserving_first(&raw, 6), raw.map(|u| u as i32).to_vec());
    }

    #[test]
    fn preserves_final_token_when_over_max() {
        // e.g. raw ends with EOS=99; a naive take(3) would drop it. The
        // last-token pooler must still see 99.
        let raw = [10, 11, 12, 13, 99];
        assert_eq!(truncate_preserving_last(&raw, 3), vec![10, 11, 99]);
        assert_eq!(truncate_preserving_last(&raw, 4), vec![10, 11, 12, 99]);
    }
}
