//! One embedding input as token ids: the purpose's prompt prefix, the
//! tokenizer, and truncation (`max_tokens`, `truncate`). Shared by the
//! Core ML and ONNX backends, which then differ only in how they run the
//! ids (D40). Platform-neutral.

use sidekick_core::manifest::ResolvedModel;
use sidekick_core::{EmbedLimits, EmbedPurpose, Error, Result, Truncate};
use tokenizers::{Encoding, Tokenizer, TruncationDirection};

/// A model's tokenizer and prompt prefixes.
pub struct EmbedInput {
    /// The model's id, for errors.
    model: String,
    tokenizer: Tokenizer,
    prefix_query: String,
    prefix_document: String,
    /// Special tokens the tokenizer adds around one input.
    added_tokens: usize,
}

impl EmbedInput {
    pub fn load(model: &ResolvedModel) -> Result<Self> {
        let m = &model.manifest;
        let tokenizer = Tokenizer::from_file(model.tokenizer_path())
            .map_err(|e| Error::Tokenizer(e.to_string()))?;
        Self::new(&m.id, tokenizer, &m.prefixes.query, &m.prefixes.document)
    }

    fn new(model: &str, mut tokenizer: Tokenizer, prefix_query: &str, prefix_document: &str) -> Result<Self> {
        // A tokenizer.json may pad every input to a fixed length
        // (all-MiniLM ships padding to 128), and every returned token is
        // marked real in the attention mask: the pads would be read as
        // text. One input is never padded; a backend pads if it must.
        // Truncation stays as shipped: models' parity references were
        // graded with it (LFM2.5 truncates at 512 in its tokenizer.json).
        // Under truncate NONE, an input it cut is refused (`check_whole`).
        tokenizer.with_padding(None);
        // Measured with padding off, or a padding tokenizer would count its
        // pads as added tokens.
        let added_tokens = tokenizer
            .encode("", true)
            .map_err(|e| Error::Tokenizer(e.to_string()))?
            .len();
        Ok(Self {
            model: model.to_string(),
            tokenizer,
            prefix_query: prefix_query.to_string(),
            prefix_document: prefix_document.to_string(),
            added_tokens,
        })
    }

    /// Apply the purpose's prefix, tokenize, and truncate to at most
    /// `max_tokens` and `longest`: keeping the start (`END`, the final
    /// token preserved), the end (`START`: the first token preserved, since
    /// a tokenizer's leading special token is what CLS pooling reads), or a
    /// 400 (`NONE`).
    pub fn ids(&self, text: &str, purpose: EmbedPurpose, limits: EmbedLimits, longest: usize) -> Result<Vec<i32>> {
        if limits.max_tokens == Some(0) {
            return Err(Error::InvalidRequest("max_tokens must be at least 1".into()));
        }
        let prefix = match purpose {
            EmbedPurpose::Query => self.prefix_query.as_str(),
            EmbedPurpose::Document => self.prefix_document.as_str(),
        };
        let max = limits.max_tokens.unwrap_or(longest).min(longest);
        // Byte caps bound tokenizer work on the text; the prefix stays whole.
        let body = match limits.truncate {
            Truncate::End => crate::byte_cap(text, max),
            Truncate::Start => crate::classify_input::byte_cap_end(text, max),
            Truncate::Reject if text.len() > max.saturating_mul(16) => {
                return Err(too_long(&self.model, format!("{} bytes", text.len()), max))
            }
            Truncate::Reject => text,
        };
        let text = format!("{prefix}{body}");
        let ids = match limits.truncate {
            // START keeps the end of the text, and the model's prompt
            // prefix whole: the model was never trained without it.
            Truncate::Start if !prefix.is_empty() => {
                keep_prefix_and_end(&self.tokenizer, self.added_tokens, &text, prefix.len(), max)?
            }
            truncate => {
                let encoding = self
                    .tokenizer
                    .encode(text.as_str(), true)
                    .map_err(|e| Error::Tokenizer(e.to_string()))?;
                let raw = encoding.get_ids();
                match truncate {
                    Truncate::End => truncate_preserving_last(raw, max),
                    Truncate::Start => truncate_preserving_first(raw, max),
                    Truncate::Reject => {
                        check_whole(&self.tokenizer, &self.model, &text, true, &encoding, max)?;
                        raw.iter().map(|&u| u as i32).collect()
                    }
                }
            }
        };
        // A tokenizer that adds no special tokens gives nothing for an empty
        // text: a bad request, not a model failure.
        if ids.is_empty() {
            return Err(Error::InvalidRequest("the input has no tokens to embed".into()));
        }
        Ok(ids)
    }
}

/// At most `max` tokens of `text` (whose first `prefix_bytes` are the
/// prompt prefix), special tokens included: the prefix's tokens, then
/// the end of the rest.
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
        return Err(Error::InvalidRequest(format!(
            "max_tokens {max} leaves no room for text after the model's {prefix}-token prompt prefix"
        )));
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
/// backends prepare every input before they run any, so a rejected input
/// refuses the whole request at no inference cost. A 400 names the input's
/// index in the batch.
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

/// Under truncate `NONE`: an error unless `encoding` (of `text`, with or
/// without special tokens) is whole and at most `max` tokens. A
/// tokenizer.json may truncate on its own (LFM2.5's does at 512) before
/// sidekick sees the ids, so an encoding it cut is too long whatever its
/// length: the limit is then the lower of the tokenizer's and `max`, and
/// the count is the text's untruncated length.
pub(crate) fn check_whole(
    tokenizer: &Tokenizer,
    model: &str,
    text: &str,
    add_special_tokens: bool,
    encoding: &Encoding,
    max: usize,
) -> Result<()> {
    if encoding.get_overflowing().is_empty() {
        return match encoding.len() {
            n if n > max => Err(too_long(model, format!("{n} tokens"), max)),
            _ => Ok(()),
        };
    }
    let limit = tokenizer.get_truncation().map_or(max, |t| t.max_length.min(max));
    // Only an input being refused pays for the copy.
    let mut whole = tokenizer.clone();
    let count = whole
        .with_truncation(None)
        .ok()
        .and_then(|t| t.encode(text, add_special_tokens).ok())
        .map_or_else(|| format!("more than {} tokens", encoding.len()), |e| format!("{} tokens", e.len()));
    Err(too_long(model, count, limit))
}

/// The 400 for an input longer than `model`'s limit of `max` tokens under
/// truncate `NONE`. `size` is its length: `600 tokens`, or `40000 bytes`
/// for a text too long to be worth tokenizing.
pub(crate) fn too_long(model: &str, size: String, max: usize) -> Error {
    Error::InvalidRequest(format!(
        "{size}, over model `{model}`'s limit of {max} tokens, and truncate is NONE \
         (send END or START to truncate it)"
    ))
}

#[cfg(test)]
mod tests {
    use super::{truncate_preserving_last, EmbedInput};
    use sidekick_core::{EmbedLimits, EmbedPurpose, Error, Truncate};
    use tokenizers::TruncationParams;

    const NONE: EmbedLimits = EmbedLimits { truncate: Truncate::Reject, max_tokens: None };

    /// laya's test tokenizer ([CLS] text [SEP]), truncating to `at` tokens
    /// on its own when given, as LFM2.5's tokenizer.json does at 512.
    fn input(at: Option<usize>) -> EmbedInput {
        let mut tok = crate::laya::tests::tokenizer();
        let truncation = at.map(|max_length| TruncationParams { max_length, ..Default::default() });
        tok.with_truncation(truncation).unwrap();
        EmbedInput::new("tiny", tok, "", "").unwrap()
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
            "8 tokens, over model `tiny`'s limit of 6 tokens, and truncate is NONE (send END or START to truncate it)"
        );
        assert_eq!(input(None).ids("a b c d", EmbedPurpose::Document, NONE, 6).unwrap().len(), 6, "at the limit");
    }

    #[test]
    fn truncate_none_refuses_an_input_the_tokenizer_cut() {
        // The tokenizer keeps 5 of 8 tokens, under the model's 32: END
        // runs what it kept, and NONE refuses, naming the untruncated
        // length and the tokenizer's limit.
        let tok = input(Some(5));
        let end = tok.ids("a b c d e f", EmbedPurpose::Document, EmbedLimits::default(), 32).unwrap();
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
    fn a_refused_input_is_named_by_its_index() {
        let tok = input(None);
        let e = super::each_input(&["a", "a b c d e f", "b"], |t| tok.ids(t, EmbedPurpose::Document, NONE, 6)).unwrap_err();
        assert!(message(e).starts_with("input 1: 8 tokens, over model `tiny`'s limit of 6 tokens"));
        // A text too long to be worth tokenizing is measured in bytes.
        let long = "a ".repeat(100);
        let e = super::each_input(&[long.as_str()], |t| tok.ids(t, EmbedPurpose::Document, NONE, 6)).unwrap_err();
        assert!(message(e).starts_with("input 0: 200 bytes, over model `tiny`'s limit of 6 tokens"));
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
