//! Turning one classification input into a [`Prepared`]: tokenization,
//! truncation, the zero-shot format's layout, and bucketing.
//!
//! Platform-neutral, so the token-level contract (docs/design/classify.md)
//! is tested everywhere; [`crate::CoremlClassifier`] runs the result.

use crate::laya;
use sidekick_core::manifest::{ClassifierManifest, ClassifyFormat, LayaSection, ResolvedClassifier};
use sidekick_core::{
    ClassifyParams, ClassifyTask, Error, Prepared, Result, TruncationSide,
};
use tokenizers::{Tokenizer, TruncationDirection};

/// Everything `prepare` needs from a classifier's manifest and tokenizer.
pub struct InputBuilder {
    tokenizer: Tokenizer,
    task: ClassifyTask,
    buckets: Vec<usize>,
    max_seq_len: usize,
    max_labels: usize,
    format: Option<Format>,
    /// Tokens the post-processor adds around one sequence ([CLS] … [SEP]).
    added_tokens: usize,
}

enum Format {
    Laya { section: LayaSection, specials: laya::Specials },
}

fn invalid(message: impl Into<String>) -> Error {
    Error::InvalidRequest(message.into())
}

impl InputBuilder {
    pub fn load(model: &ResolvedClassifier) -> Result<Self> {
        let mut tokenizer = Tokenizer::from_file(model.tokenizer_path())
            .map_err(|e| Error::Tokenizer(e.to_string()))?;
        // A tokenizer.json may carry its own truncation or padding. Both
        // would hide an over-length input that should be a 400, or change
        // the ids, so `prepare` does its own.
        tokenizer
            .with_truncation(None)
            .map_err(|e| Error::Tokenizer(e.to_string()))?;
        tokenizer.with_padding(None);
        Self::new(tokenizer, &model.manifest)
    }

    pub fn new(tokenizer: Tokenizer, m: &ClassifierManifest) -> Result<Self> {
        let added_tokens = tokenizer
            .encode("", true)
            .map_err(|e| Error::Tokenizer(e.to_string()))?
            .len();
        let format = match (m.classify.format, &m.classify.laya) {
            (Some(ClassifyFormat::Laya), Some(section)) => Some(Format::Laya {
                section: section.clone(),
                specials: laya::Specials::from_tokenizer(&tokenizer)?,
            }),
            (Some(ClassifyFormat::Laya), None) => {
                return Err(Error::InvalidManifest {
                    path: m.id.clone(),
                    message: "the laya format needs `[classify.laya]`".into(),
                })
            }
            (None, _) => None,
        };
        Ok(Self {
            tokenizer,
            task: m.task,
            buckets: m.buckets.clone(),
            max_seq_len: m.max_seq_len,
            max_labels: m.max_labels(),
            format,
            added_tokens,
        })
    }

    /// The laya special tokens, when the model uses that format.
    pub fn laya_specials(&self) -> Option<&laya::Specials> {
        match &self.format {
            Some(Format::Laya { specials, .. }) => Some(specials),
            None => None,
        }
    }

    pub fn buckets(&self) -> &[usize] {
        &self.buckets
    }

    fn bucket_for(&self, len: usize) -> usize {
        *self
            .buckets
            .iter()
            .find(|&&b| b >= len)
            .unwrap_or(self.buckets.last().expect("validated non-empty"))
    }

    pub fn prepare(&self, input: &str, params: &ClassifyParams) -> Result<Prepared> {
        match &self.format {
            None => self.prepare_text(input, params),
            Some(Format::Laya { section, specials }) => {
                self.prepare_laya(input, params, section, specials)
            }
        }
    }

    /// text-classification: the whole input, wrapped in the tokenizer's
    /// special tokens. Over-length is a 400 unless `truncate_prompt_tokens`
    /// is set (vLLM's behavior); truncation keeps the special tokens.
    fn prepare_text(&self, input: &str, params: &ClassifyParams) -> Result<Prepared> {
        if self.task != ClassifyTask::TextClassification
            || !params.candidate_labels.is_empty()
            || params.question_type.is_some()
            || params.instructions.is_some()
        {
            return Err(invalid("this model takes no candidate_labels, question_type or instructions"));
        }
        let max = self.max_seq_len;
        let limit = match params.truncate_prompt_tokens {
            None => max,
            Some(n) if n > max => {
                return Err(invalid(format!(
                    "truncate_prompt_tokens {n} exceeds the model's maximum of {max} tokens"
                )))
            }
            Some(n) if n <= self.added_tokens => {
                return Err(invalid(format!(
                    "truncate_prompt_tokens {n} leaves no room for text after the model's \
                     {} special tokens",
                    self.added_tokens
                )))
            }
            Some(n) => n,
        };
        let cap = max.saturating_mul(16);
        let text = match (params.truncate_prompt_tokens, params.truncation_side) {
            // No tokenizer spends 16 bytes per token, so an input this long
            // is over-length without spending the time to tokenize it.
            (None, _) if input.len() > cap => {
                return Err(too_long(format!("{} bytes", input.len()), max))
            }
            (None, _) => input,
            (Some(_), TruncationSide::Right) => crate::byte_cap(input, max),
            (Some(_), TruncationSide::Left) => byte_cap_end(input, max),
        };
        let mut encoding = self
            .tokenizer
            .encode(text, false)
            .map_err(|e| Error::Tokenizer(e.to_string()))?;
        if params.truncate_prompt_tokens.is_some() {
            let direction = match params.truncation_side {
                TruncationSide::Right => TruncationDirection::Right,
                TruncationSide::Left => TruncationDirection::Left,
            };
            encoding.truncate(limit - self.added_tokens, 0, direction);
        }
        let encoding = self
            .tokenizer
            .post_process(encoding, None, true)
            .map_err(|e| Error::Tokenizer(e.to_string()))?;
        let ids: Vec<i32> = encoding.get_ids().iter().map(|&u| u as i32).collect();
        if ids.len() > max {
            return Err(too_long(format!("{} tokens", ids.len()), max));
        }
        Ok(Prepared { bucket: self.bucket_for(ids.len()), ids, markers: vec![], qtype: None })
    }

    /// The laya format: the state is truncated by design, keeping its
    /// start, so `truncate_prompt_tokens` and left truncation are 400s.
    fn prepare_laya(
        &self,
        input: &str,
        params: &ClassifyParams,
        section: &LayaSection,
        specials: &laya::Specials,
    ) -> Result<Prepared> {
        if params.truncate_prompt_tokens.is_some() {
            return Err(invalid(
                "truncate_prompt_tokens isn't supported by the laya format, which truncates \
                 the text itself, keeping its start",
            ));
        }
        if params.truncation_side == TruncationSide::Left {
            return Err(invalid(
                "truncation_side `left` isn't supported by the laya format, which keeps the \
                 text's start",
            ));
        }
        let Some(question_type) = params.question_type else {
            return Err(invalid("question_type is required by this model (choice, score or noul)"));
        };
        let labels = &params.candidate_labels;
        check_labels(labels, self.max_labels)?;
        let options = laya::render_options(question_type, labels)?;
        let instructions = params
            .instructions
            .as_deref()
            .unwrap_or_else(|| section.default_instructions.get(question_type));
        // Byte caps bound tokenizer work; laya cuts every piece far below
        // them (the state to the sequence, options to 48 tokens, the
        // question to the head budget).
        let options: Vec<String> =
            options.iter().map(|o| crate::byte_cap(o, 64).to_string()).collect();
        let seq = laya::build_sequence(
            &self.tokenizer,
            specials,
            crate::byte_cap(input, self.max_seq_len),
            laya::Question {
                question_type,
                instructions: crate::byte_cap(instructions, section.head_max_len),
                options: &options,
            },
            self.max_seq_len,
            section.head_max_len,
        )?;
        if seq.markers.len() != labels.len() {
            return Err(invalid(format!(
                "the {} candidate labels don't fit the model's input",
                labels.len()
            )));
        }
        for (i, a) in seq.options.iter().enumerate() {
            if let Some(j) = seq.options[..i].iter().position(|b| b == a) {
                return Err(invalid(format!(
                    "candidate labels `{}` and `{}` are identical to the model once shortened \
                     to fit; make them differ earlier",
                    labels[j], labels[i]
                )));
            }
        }
        let ids: Vec<i32> = seq.ids.iter().map(|&u| u as i32).collect();
        Ok(Prepared {
            bucket: self.bucket_for(ids.len()),
            ids,
            markers: seq.markers.iter().map(|&m| m as i32).collect(),
            qtype: Some(question_type.index()),
        })
    }
}

/// Zero-shot label checks that don't need the tokenizer.
pub fn check_labels(labels: &[String], max_labels: usize) -> Result<()> {
    if labels.len() < 2 {
        return Err(invalid("candidate_labels needs at least 2 labels"));
    }
    if labels.len() > max_labels {
        return Err(invalid(format!(
            "{} candidate_labels exceed the model's maximum of {max_labels}",
            labels.len()
        )));
    }
    for (i, l) in labels.iter().enumerate() {
        if labels[..i].contains(l) {
            return Err(invalid(format!("duplicate candidate label `{l}`")));
        }
    }
    Ok(())
}

fn too_long(size: String, max: usize) -> Error {
    invalid(format!(
        "input of {size} exceeds the model's maximum of {max} tokens; set \
         truncate_prompt_tokens to truncate it"
    ))
}

/// The last bytes of `text` that `max_tokens` could need: `byte_cap` from
/// the end, for left truncation.
fn byte_cap_end(text: &str, max_tokens: usize) -> &str {
    let cap = max_tokens.saturating_mul(16);
    if text.len() <= cap {
        return text;
    }
    let mut start = text.len() - cap;
    while !text.is_char_boundary(start) {
        start += 1;
    }
    &text[start..]
}

#[cfg(test)]
mod tests {
    use super::*;
    use sidekick_core::manifest::{ClassifierIo, ClassifySection, DefaultInstructions};
    use sidekick_core::{ProblemType, QuestionType};

    fn manifest(task: ClassifyTask) -> ClassifierManifest {
        let laya = task == ClassifyTask::ZeroShotClassification;
        ClassifierManifest {
            id: "m".into(),
            task,
            source: None,
            artifact: "model_{seq}.mlmodelc".into(),
            tokenizer: "tokenizer.json".into(),
            buckets: vec![8, 16, 32],
            max_seq_len: 32,
            max_batch: 4,
            problem_type: ProblemType::SingleLabel,
            classify: ClassifySection {
                format: laya.then_some(ClassifyFormat::Laya),
                max_labels: laya.then_some(4),
                labels: if laya { vec![] } else { vec!["neg".into(), "pos".into()] },
                laya: laya.then(|| LayaSection {
                    head_max_len: 16,
                    default_instructions: DefaultInstructions {
                        choice: "which option".into(),
                        score: "which".into(),
                        noul: "ok".into(),
                    },
                }),
                calibration: Default::default(),
                io: ClassifierIo::default(),
            },
        }
    }

    fn builder(task: ClassifyTask) -> InputBuilder {
        InputBuilder::new(crate::laya::tests::tokenizer(), &manifest(task)).unwrap()
    }

    fn words(n: usize) -> String {
        (0..n).map(|i| ["a", "b", "c"][i % 3]).collect::<Vec<_>>().join(" ")
    }

    #[test]
    fn text_classification_wraps_buckets_and_rejects_over_length() {
        let b = builder(ClassifyTask::TextClassification);
        let p = b.prepare("a b c", &ClassifyParams::default()).unwrap();
        assert_eq!(p.ids.len(), 5);
        assert_eq!((p.ids[0], p.ids[4]), (1, 2));
        assert_eq!(p.bucket, 8);
        assert!(p.markers.is_empty() && p.qtype.is_none());
        assert_eq!(b.prepare(&words(10), &ClassifyParams::default()).unwrap().bucket, 16);
        assert_eq!(b.prepare(&words(30), &ClassifyParams::default()).unwrap().bucket, 32);
        let e = b.prepare(&words(31), &ClassifyParams::default()).unwrap_err();
        assert!(e.to_string().contains("33 tokens exceeds the model's maximum of 32"), "{e}");
        // Too long to be worth tokenizing.
        let e = b.prepare(&"a ".repeat(400), &ClassifyParams::default()).unwrap_err();
        assert!(e.to_string().contains("800 bytes"), "{e}");
    }

    #[test]
    fn truncation_keeps_special_tokens_on_the_chosen_side() {
        let b = builder(ClassifyTask::TextClassification);
        let text = format!("d {} e", words(100));
        let right = ClassifyParams { truncate_prompt_tokens: Some(6), ..Default::default() };
        let p = b.prepare(&text, &right).unwrap();
        let t = |w| crate::laya::tests::tokenizer().token_to_id(w).unwrap() as i32;
        assert_eq!(p.ids, vec![1, t("d"), t("a"), t("b"), t("c"), 2]);
        let left = ClassifyParams { truncation_side: TruncationSide::Left, ..right.clone() };
        let p = b.prepare(&text, &left).unwrap();
        assert_eq!(p.ids.len(), 6);
        assert_eq!(p.ids[4], t("e"));
        assert_eq!((p.ids[0], p.ids[5]), (1, 2));
        // Over the model's maximum, or no room for text.
        for n in [33, 2, 0] {
            let params = ClassifyParams { truncate_prompt_tokens: Some(n), ..Default::default() };
            assert!(matches!(b.prepare("a", &params), Err(Error::InvalidRequest(_))), "{n}");
        }
        // A huge input with truncation is fine: byte-capped first.
        assert_eq!(b.prepare(&"a ".repeat(10_000), &ClassifyParams { truncate_prompt_tokens: Some(32), ..Default::default() }).unwrap().ids.len(), 32);
    }

    fn laya_params(qt: QuestionType, labels: &[&str]) -> ClassifyParams {
        ClassifyParams {
            question_type: Some(qt),
            candidate_labels: labels.iter().map(|s| s.to_string()).collect(),
            ..Default::default()
        }
    }

    #[test]
    fn laya_lays_out_markers_and_qtype_with_default_instructions() {
        let b = builder(ClassifyTask::ZeroShotClassification);
        let p = b.prepare("a b", &laya_params(QuestionType::Choice, &["c", "d"])).unwrap();
        let t = |w| crate::laya::tests::tokenizer().token_to_id(w).unwrap() as i32;
        // [CLS] choice question : which option [SEP] [MASK] c [MASK] d [SEP] a b [SEP]
        assert_eq!(p.ids[..7], [1, t("choice"), t("question"), t(":"), t("which"), t("option"), 2]);
        assert_eq!(p.markers, vec![7, 9]);
        assert_eq!(p.qtype, Some(0));
        assert_eq!(p.bucket, 16);
        // The state is truncated, not rejected.
        let p = b.prepare(&words(200), &laya_params(QuestionType::Noul, &["false", "true"])).unwrap();
        assert_eq!(p.ids.len(), 32);
        assert_eq!(p.qtype, Some(2));
    }

    #[test]
    fn laya_rejects_what_it_cannot_honor() {
        let b = builder(ClassifyTask::ZeroShotClassification);
        let ok = laya_params(QuestionType::Choice, &["c", "d"]);
        let cases = [
            ClassifyParams { truncate_prompt_tokens: Some(8), ..ok.clone() },
            ClassifyParams { truncation_side: TruncationSide::Left, ..ok.clone() },
            ClassifyParams { question_type: None, ..ok.clone() },
            laya_params(QuestionType::Choice, &["c"]),
            laya_params(QuestionType::Choice, &["a", "b", "c", "d", "e"]),
            laya_params(QuestionType::Choice, &["c", "c"]),
            laya_params(QuestionType::Noul, &["yes", "no"]),
        ];
        for params in cases {
            assert!(matches!(b.prepare("a", &params), Err(Error::InvalidRequest(_))), "{params:?}");
        }
    }

    #[test]
    fn laya_rejects_labels_identical_after_shrinking() {
        let b = builder(ClassifyTask::ZeroShotClassification);
        // head_max_len 16 with 3 options forces shrinking to max(4, 0) = 4
        // tokens each: [MASK] + 3 words, so these collide.
        let e = b
            .prepare("a", &laya_params(QuestionType::Choice, &["a b c d", "a b c e", "f"]))
            .unwrap_err();
        assert!(e.to_string().contains("identical to the model"), "{e}");
    }
}
