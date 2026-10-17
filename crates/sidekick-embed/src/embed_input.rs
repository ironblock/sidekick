//! One embedding input as token ids: the purpose's prompt prefix, the
//! tokenizer, and truncation (`max_tokens`, `truncate`). Shared by the
//! Core ML and ONNX backends, which then differ only in how they run the
//! ids (D40). Platform-neutral.

use sidekick_core::manifest::ResolvedModel;
use sidekick_core::{EmbedLimits, EmbedPurpose, Error, Result, Truncate};
use tokenizers::{Tokenizer, TruncationDirection};

/// A model's tokenizer and prompt prefixes.
pub struct EmbedInput {
    tokenizer: Tokenizer,
    prefix_query: String,
    prefix_document: String,
    /// Special tokens the tokenizer adds around one input.
    added_tokens: usize,
}

impl EmbedInput {
    pub fn load(model: &ResolvedModel) -> Result<Self> {
        let m = &model.manifest;
        let mut tokenizer = Tokenizer::from_file(model.tokenizer_path())
            .map_err(|e| Error::Tokenizer(e.to_string()))?;
        // A tokenizer.json may pad every input to a fixed length
        // (all-MiniLM ships padding to 128), and every returned token is
        // marked real in the attention mask: the pads would be read as
        // text. One input is never padded; a backend pads if it must.
        // Truncation stays as shipped: models' parity references were
        // graded with it (LFM2.5 truncates at 512 in its tokenizer.json).
        tokenizer.with_padding(None);
        // Measured with padding off, or a padding tokenizer would count its
        // pads as added tokens.
        let added_tokens = tokenizer
            .encode("", true)
            .map_err(|e| Error::Tokenizer(e.to_string()))?
            .len();
        Ok(Self {
            tokenizer,
            prefix_query: m.prefixes.query.clone(),
            prefix_document: m.prefixes.document.clone(),
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
                return Err(too_long(format!("{} bytes", text.len()), max))
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
                    Truncate::Reject if raw.len() > max => {
                        return Err(too_long(format!("{} tokens", raw.len()), max))
                    }
                    Truncate::Reject => raw.iter().map(|&u| u as i32).collect(),
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

fn too_long(size: String, max: usize) -> Error {
    Error::InvalidRequest(format!(
        "input of {size} exceeds the limit of {max} tokens, and truncate is NONE"
    ))
}

#[cfg(test)]
mod tests {
    use super::truncate_preserving_last;

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
