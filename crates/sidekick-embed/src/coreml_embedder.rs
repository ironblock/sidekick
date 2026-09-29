//! Core ML encoder embedder (ANE-targeted).
//!
//! Pipeline: tokenize (HF `tokenizers`) → pad to the smallest sequence-length
//! bucket that fits → int32 `input_ids`/`attention_mask` prediction →
//! pool per manifest → unit-normalize.
//!
//! Each bucket maps to its own static-shape artifact when the manifest uses
//! a `{seq}` placeholder; bucket models load lazily on first use and stay
//! resident. (A single enumerated-shapes artifact is also supported, but
//! hardware verification showed E5RT rejects flexible shapes at ANE plan
//! time and silently runs the whole encoder on CPU — prefer per-bucket
//! static artifacts.)

use crate::bucket_models::BucketModels;
use crate::pooling::{mean_pool, normalize_in_place};
use sidekick_core::manifest::ResolvedModel;
use sidekick_core::{EmbedLimits, EmbedPurpose, Embedder, Error, Pooling, Result, Truncate};
use sidekick_coreml::{ComputeUnits, Int32Input};
use tokenizers::{Tokenizer, TruncationDirection};

pub struct CoremlEmbedder {
    id: String,
    dims: usize,
    matryoshka: Vec<usize>,
    /// Lazily loaded per-bucket models.
    models: BucketModels,
    tokenizer: Tokenizer,
    buckets: Vec<usize>,
    pooling: Pooling,
    input_ids_name: String,
    attention_mask_name: Option<String>,
    output_name: String,
    prefix_query: String,
    prefix_document: String,
    /// Special tokens the tokenizer adds around one input.
    added_tokens: usize,
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

/// The token ids sidekick feeds the model for one input, and the bucket they
/// go in. Exposed so tests can check tokenization against a reference
/// pipeline and run the same ids through a larger bucket.
#[doc(hidden)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Prepared {
    /// Real token ids (prefix applied, specials included, truncated to the
    /// largest bucket), before padding.
    pub ids: Vec<i32>,
    /// The smallest bucket that fits `ids`.
    pub bucket: usize,
}

impl CoremlEmbedder {
    /// Load for the ANE (`.cpuAndNeuralEngine`), as the daemon does.
    pub fn load(model: &ResolvedModel) -> Result<Self> {
        Self::load_with(model, ComputeUnits::CpuAndNeuralEngine)
    }

    /// Load with an explicit compute-unit preference. sidekick only serves
    /// from the ANE preference; the others exist so tests can compare paths.
    pub fn load_with(model: &ResolvedModel, units: ComputeUnits) -> Result<Self> {
        let m = &model.manifest;
        let mut tokenizer = Tokenizer::from_file(model.tokenizer_path())
            .map_err(|e| Error::Tokenizer(e.to_string()))?;
        // A tokenizer.json may pad every input to a fixed length
        // (all-MiniLM ships padding to 128), and every returned token is
        // marked real in the attention mask: the pads would be read as
        // text. One input is never padded; `run` pads to the bucket.
        // Truncation stays as shipped: models' parity references were
        // graded with it (LFM2.5 truncates at 512 in its tokenizer.json).
        tokenizer.with_padding(None);
        // Measured with padding off, or a padding tokenizer would count its
        // pads as added tokens.
        let added_tokens = tokenizer
            .encode("", true)
            .map_err(|e| Error::Tokenizer(e.to_string()))?
            .len();
        let embedder = Self {
            id: m.id.clone(),
            dims: m.dims,
            matryoshka: m.matryoshka.clone(),
            models: BucketModels::new(&model.dir, &m.artifact, units),
            tokenizer,
            buckets: m.buckets.clone(),
            pooling: m.pooling,
            input_ids_name: m.io.input_ids.clone(),
            attention_mask_name: m.io.attention_mask.clone(),
            output_name: m.io.output.clone(),
            prefix_query: m.prefixes.query.clone(),
            prefix_document: m.prefixes.document.clone(),
            added_tokens,
        };
        // Load the smallest bucket eagerly so a broken artifact fails at
        // load time (matching the pool's load-error surface), not on the
        // first request.
        let smallest = *embedder.buckets.first().expect("validated non-empty");
        embedder.models.get(smallest)?;
        Ok(embedder)
    }

    /// Sequence-length buckets, smallest first.
    pub fn buckets(&self) -> &[usize] {
        &self.buckets
    }

    /// Apply the purpose's prefix, tokenize, truncate, and pick a bucket.
    /// Public for the parity suite (examples/parity), which checks the ids
    /// against a reference pipeline.
    #[doc(hidden)]
    pub fn prepare(&self, text: &str, purpose: EmbedPurpose) -> Result<Prepared> {
        self.prepare_with(text, purpose, EmbedLimits::default())
    }

    /// `prepare` with per-request limits: at most `max_tokens` (and the
    /// largest bucket), keeping the start (`END`, the final token
    /// preserved as `prepare` does), the end (`START`: the first token
    /// preserved, since a tokenizer's leading special token is what CLS
    /// pooling reads), or a 400 (`NONE`).
    pub fn prepare_with(&self, text: &str, purpose: EmbedPurpose, limits: EmbedLimits) -> Result<Prepared> {
        if limits.max_tokens == Some(0) {
            return Err(Error::InvalidRequest("max_tokens must be at least 1".into()));
        }
        let prefix = match purpose {
            EmbedPurpose::Query => self.prefix_query.as_str(),
            EmbedPurpose::Document => self.prefix_document.as_str(),
        };
        let largest = *self.buckets.last().expect("validated non-empty");
        let max = limits.max_tokens.unwrap_or(largest).min(largest);
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
        let bucket = *self
            .buckets
            .iter()
            .find(|&&b| b >= ids.len())
            .unwrap_or(&largest);
        Ok(Prepared { ids, bucket })
    }

    /// Pad `ids` to `bucket` with id 0, predict, pool, and unit-normalize.
    /// `bucket` must be one of the model's buckets and hold all of `ids`; the
    /// daemon always passes the one `prepare` picked. Public for the parity
    /// suite, which also runs ids through larger buckets.
    #[doc(hidden)]
    pub fn run(&self, ids: &[i32], bucket: usize) -> Result<Vec<f32>> {
        self.run_padded(ids, bucket, &[])
    }

    /// `run` with the given ids in the pad positions (id 0 past their end).
    /// The attention mask hides pads, so the result must not change: the
    /// parity suite's check for a dropped mask.
    #[doc(hidden)]
    pub fn run_padded(&self, ids: &[i32], bucket: usize, pad_ids: &[i32]) -> Result<Vec<f32>> {
        if !self.buckets.contains(&bucket) || ids.len() > bucket {
            return Err(Error::Inference(format!(
                "{} tokens can't run in bucket {bucket} (buckets {:?})",
                ids.len(),
                self.buckets
            )));
        }
        let used = ids.len();
        let mut input_ids = ids.to_vec();
        input_ids.extend(pad_ids.iter().take(bucket - used));
        input_ids.resize(bucket, 0);
        let mut mask = vec![1i32; used];
        mask.resize(bucket, 0);

        let mut inputs = vec![Int32Input {
            name: &self.input_ids_name,
            shape: vec![1, bucket],
            data: input_ids,
        }];
        if let Some(mask_name) = &self.attention_mask_name {
            inputs.push(Int32Input {
                name: mask_name,
                shape: vec![1, bucket],
                data: mask.clone(),
            });
        }

        let model = self.models.get(bucket)?;
        let out = model.predict_int32(&inputs, &self.output_name)?;
        let n: usize = out.shape.iter().product();

        let mut vector = match self.pooling {
            Pooling::None => {
                if n != self.dims {
                    return Err(Error::Inference(format!(
                        "pooled output shape {:?} != dims {}",
                        out.shape, self.dims
                    )));
                }
                out.data
            }
            Pooling::Mean | Pooling::Cls => {
                if n != bucket * self.dims {
                    return Err(Error::Inference(format!(
                        "hidden-state output shape {:?} != [1, {bucket}, {}]",
                        out.shape, self.dims
                    )));
                }
                match self.pooling {
                    Pooling::Cls => out.data[..self.dims].to_vec(),
                    _ => {
                        let mask_u32: Vec<u32> = mask.iter().map(|&m| m as u32).collect();
                        mean_pool(&out.data, self.dims, &mask_u32)
                    }
                }
            }
        };
        normalize_in_place(&mut vector);
        Ok(vector)
    }
}

impl Embedder for CoremlEmbedder {
    fn id(&self) -> &str {
        &self.id
    }

    fn dims(&self) -> usize {
        self.dims
    }

    fn matryoshka_dims(&self) -> &[usize] {
        &self.matryoshka
    }

    fn embed(&self, texts: &[&str], purpose: EmbedPurpose) -> Result<Vec<Vec<f32>>> {
        self.embed_with(texts, purpose, EmbedLimits::default())
    }

    fn embed_with(
        &self,
        texts: &[&str],
        purpose: EmbedPurpose,
        limits: EmbedLimits,
    ) -> Result<Vec<Vec<f32>>> {
        // Prepare every text first, so a rejected one costs no prediction.
        let prepared = texts
            .iter()
            .map(|t| self.prepare_with(t, purpose, limits))
            .collect::<Result<Vec<_>>>()?;
        prepared.iter().map(|p| self.run(&p.ids, p.bucket)).collect()
    }
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
