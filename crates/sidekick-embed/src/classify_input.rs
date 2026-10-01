//! Turning one classification input into a [`Prepared`]: tokenization,
//! truncation, the zero-shot format's layout, and bucketing.
//!
//! Platform-neutral, so the token-level contract (docs/design/classify.md)
//! is tested everywhere; [`crate::CoremlClassifier`] runs the result.

use crate::{gliner2, laya};
use sidekick_core::manifest::{ClassifierManifest, ClassifyFormat, Gliner2Section, LayaSection, ResolvedClassifier};
use sidekick_core::{
    ClassifyParams, ClassifyTask, Error, PairParams, Prepared, Result, TruncationSide,
};
use tokenizers::utils::truncation::{truncate_encodings, TruncationParams, TruncationStrategy};
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
    /// Tokens it adds around a pair ([CLS] q [SEP] d [SEP]: 3).
    pair_added_tokens: usize,
    /// The model's graph takes `token_type_ids`.
    segment_ids: bool,
}

enum Format {
    Laya { section: LayaSection, specials: laya::Specials },
    Gliner2 { section: Gliner2Section, specials: gliner2::Specials },
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
        let tok_err = |e: tokenizers::Error| Error::Tokenizer(e.to_string());
        let added_tokens = tokenizer.encode("", true).map_err(tok_err)?.len();
        let pair_added_tokens = tokenizer.encode(("", ""), true).map_err(tok_err)?.len();
        let segment_ids = m.classify.io.token_type_ids.is_some();
        // A tokenizer that marks the document with segment ids for a model
        // whose graph can't take them would score every pair wrong.
        if m.task == ClassifyTask::TextRanking && !segment_ids {
            let pair = tokenizer.encode(("a", "b"), true).map_err(tok_err)?;
            if pair.get_type_ids().iter().any(|&t| t != 0) {
                return Err(Error::InvalidManifest {
                    path: m.id.clone(),
                    message: "the tokenizer gives pairs segment ids, but `[classify.io]` names no \
                              `token_type_ids` input"
                        .into(),
                });
            }
        }
        let missing = |section: &str| Error::InvalidManifest {
            path: m.id.clone(),
            message: format!("the {section} format needs `[classify.{section}]`"),
        };
        let format = match m.classify.format {
            Some(ClassifyFormat::Laya) => Some(Format::Laya {
                section: m.classify.laya.clone().ok_or_else(|| missing("laya"))?,
                specials: laya::Specials::from_tokenizer(&tokenizer)?,
            }),
            Some(ClassifyFormat::Gliner2) => Some(Format::Gliner2 {
                section: m.classify.gliner2.clone().ok_or_else(|| missing("gliner2"))?,
                specials: gliner2::Specials::from_tokenizer(&tokenizer)?,
            }),
            None => None,
        };
        Ok(Self {
            tokenizer,
            task: m.task,
            buckets: m.buckets.clone(),
            max_seq_len: m.max_seq_len,
            max_labels: m.max_labels(),
            format,
            added_tokens,
            pair_added_tokens,
            segment_ids,
        })
    }

    /// The laya special tokens, when the model uses that format.
    pub fn laya_specials(&self) -> Option<&laya::Specials> {
        match &self.format {
            Some(Format::Laya { specials, .. }) => Some(specials),
            _ => None,
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
        if self.task == ClassifyTask::TextRanking {
            return Err(invalid("a text-ranking model scores (query, document) pairs; use a rerank route"));
        }
        match &self.format {
            None => self.prepare_text(input, params),
            Some(Format::Laya { section, specials }) => {
                self.prepare_laya(input, params, section, specials)
            }
            Some(Format::Gliner2 { section, specials }) => {
                self.prepare_gliner2(input, params, section, specials)
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
        Ok(Prepared { bucket: self.bucket_for(ids.len()), ids, type_ids: vec![], markers: vec![], qtype: None })
    }

    /// A text-ranking model's (query, document) pair, as its tokenizer
    /// pairs text: `[CLS] q [SEP] d [SEP]` with segment ids 0 then 1 for
    /// BERT, `<s> q </s></s> d </s>` for XLM-R. `max_tokens_per_query` and
    /// `max_tokens_per_doc` cut each text first, keeping its start. Then a
    /// pair longer than the limit is truncated `longest_first` (vLLM's
    /// default), or only in the document (`keep_query`, Cohere's contract,
    /// or whenever `max_tokens_per_doc` is set, as in vLLM). Special tokens
    /// are always kept, where vLLM's `truncation_side` slice can drop one.
    /// With no limit, an over-length pair is a 400.
    pub fn prepare_pair(&self, query: &str, document: &str, p: &PairParams) -> Result<Prepared> {
        if self.task != ClassifyTask::TextRanking {
            return Err(invalid("only text-ranking models score (query, document) pairs"));
        }
        let max = self.max_seq_len;
        let limit = match (p.truncate_prompt_tokens, p.keep_query) {
            (Some(n), _) if n > max => {
                return Err(invalid(format!(
                    "truncate_prompt_tokens {n} exceeds the model's maximum of {max} tokens"
                )))
            }
            (Some(n), _) if n <= self.pair_added_tokens => {
                return Err(invalid(format!(
                    "truncate_prompt_tokens {n} leaves no room for text after the pair's {} special tokens",
                    self.pair_added_tokens
                )))
            }
            (Some(n), _) => Some(n),
            (None, true) => Some(max),
            (None, false) => None,
        };
        // Each text is cut to its own limit first, keeping its start (vLLM's
        // truncate_text_to_tokens), then the pair is truncated. Byte caps
        // bound the tokenizer's work per text (16 bytes per token is beyond
        // any vocabulary), keeping the end that truncation keeps.
        let cap = max.saturating_mul(16);
        let bound = |text: &'_ str, cut: Option<usize>, what: &str, field: &str| -> Result<String> {
            Ok(match (cut, limit, p.truncation_side) {
                (Some(n), _, _) => crate::byte_cap(text, n),
                (None, Some(_), TruncationSide::Right) => crate::byte_cap(text, max),
                (None, Some(_), TruncationSide::Left) => byte_cap_end(text, max),
                (None, None, _) if text.len() > cap => {
                    return Err(invalid(format!(
                        "the {what} ({} bytes) is longer than the model's maximum of {max} tokens; \
                         set truncate_prompt_tokens or {field} to truncate it",
                        text.len()
                    )))
                }
                (None, None, _) => text,
            }
            .to_string())
        };
        let q = bound(query, p.max_tokens_per_query, "query", "max_tokens_per_query")?;
        let d = bound(document, p.max_tokens_per_doc, "document", "max_tokens_per_doc")?;
        let encode = |text: &str, cut: Option<usize>| -> Result<tokenizers::Encoding> {
            let mut e = self.tokenizer.encode(text, false).map_err(|e| Error::Tokenizer(e.to_string()))?;
            if let Some(n) = cut {
                e.truncate(n, 0, TruncationDirection::Right);
            }
            Ok(e)
        };
        let (mut eq, mut ed) = (encode(&q, p.max_tokens_per_query)?, encode(&d, p.max_tokens_per_doc)?);
        if let Some(n) = limit {
            let budget = n - self.pair_added_tokens;
            // Only the document is truncated under Cohere's contract, and,
            // as in vLLM, whenever max_tokens_per_doc is set; otherwise
            // tokenizers' longest_first, vLLM's default.
            let only_document = p.keep_query || p.max_tokens_per_doc.is_some();
            if only_document && eq.len() >= budget {
                return Err(invalid(format!(
                    "the query alone ({} tokens) fills the model's {n}-token input",
                    eq.len()
                )));
            }
            let params = TruncationParams {
                max_length: budget,
                strategy: if only_document { TruncationStrategy::OnlySecond } else { TruncationStrategy::LongestFirst },
                stride: 0,
                direction: match p.truncation_side {
                    TruncationSide::Right => TruncationDirection::Right,
                    TruncationSide::Left => TruncationDirection::Left,
                },
            };
            let (a, b) = truncate_encodings(eq, Some(ed), &params).map_err(|e| Error::Tokenizer(e.to_string()))?;
            eq = a;
            ed = b.expect("a pair stays a pair");
        }
        let pair = self
            .tokenizer
            .post_process(eq, Some(ed), true)
            .map_err(|e| Error::Tokenizer(e.to_string()))?;
        let ids: Vec<i32> = pair.get_ids().iter().map(|&u| u as i32).collect();
        if ids.len() > max {
            return Err(invalid(format!(
                "the (query, document) pair is {} tokens, more than the model's maximum of {max}; \
                 set truncate_prompt_tokens or max_tokens_per_doc to truncate it",
                ids.len()
            )));
        }
        let type_ids = if self.segment_ids {
            pair.get_type_ids().iter().map(|&u| u as i32).collect()
        } else {
            vec![]
        };
        Ok(Prepared { bucket: self.bucket_for(ids.len()), ids, type_ids, markers: vec![], qtype: None })
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
        let options = laya::render_options(question_type, labels, section.option_rendering)?;
        let instructions = params
            .instructions
            .as_deref()
            .or_else(|| section.default_instructions(question_type))
            .ok_or_else(|| invalid(laya::NO_INSTRUCTIONS))?;
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
            type_ids: vec![],
            markers: seq.markers.iter().map(|&m| m as i32).collect(),
            qtype: Some(question_type.index()),
        })
    }
}

impl InputBuilder {
    /// The gliner2 format (docs/design/classify.md): the schema must fit,
    /// and the text is truncated at the word level by design, keeping its
    /// start, so `truncate_prompt_tokens` and left truncation are 400s.
    fn prepare_gliner2(
        &self,
        input: &str,
        params: &ClassifyParams,
        section: &Gliner2Section,
        specials: &gliner2::Specials,
    ) -> Result<Prepared> {
        if params.truncate_prompt_tokens.is_some() {
            return Err(invalid(
                "truncate_prompt_tokens isn't supported by the gliner2 format, which truncates \
                 the text itself, keeping its start",
            ));
        }
        if params.truncation_side == TruncationSide::Left {
            return Err(invalid(
                "truncation_side `left` isn't supported by the gliner2 format, which keeps the \
                 text's start",
            ));
        }
        if params.question_type.is_some() {
            return Err(invalid("question_type is for the laya format"));
        }
        let labels = &params.candidate_labels;
        check_labels(labels, self.max_labels)?;
        let prompt = params.instructions.as_deref().unwrap_or(&section.default_instructions);
        // Byte caps bound tokenizer and splitter work: the schema has to fit
        // the model whole, and the text is truncated anyway.
        let schema_bytes = prompt.len() + labels.iter().map(String::len).sum::<usize>();
        if schema_bytes > self.max_seq_len.saturating_mul(16) {
            return Err(invalid(format!(
                "the candidate labels and instructions ({schema_bytes} bytes) can't fit the model's \
                 maximum of {} tokens",
                self.max_seq_len
            )));
        }
        let text = crate::byte_cap(input, self.max_seq_len);
        let seq = gliner2::build_sequence(
            &self.tokenizer,
            specials,
            text,
            text.len() < input.len(),
            prompt,
            labels,
            self.max_seq_len,
        )?;
        let ids: Vec<i32> = seq.ids.iter().map(|&u| u as i32).collect();
        Ok(Prepared {
            bucket: self.bucket_for(ids.len()),
            ids,
            type_ids: vec![],
            markers: seq.markers.iter().map(|&m| m as i32).collect(),
            qtype: None,
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
pub(crate) fn byte_cap_end(text: &str, max_tokens: usize) -> &str {
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
    use sidekick_core::manifest::{ClassifierIo, ClassifySection, DefaultInstructions, OptionRendering};
    use sidekick_core::{ProblemType, QuestionType};

    fn manifest(task: ClassifyTask) -> ClassifierManifest {
        let laya = task == ClassifyTask::ZeroShotClassification;
        let ranking = task == ClassifyTask::TextRanking;
        ClassifierManifest {
            compute_units: Default::default(),
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
                labels: match task {
                    ClassifyTask::ZeroShotClassification => vec![],
                    ClassifyTask::TextRanking => vec!["score".into()],
                    ClassifyTask::TextClassification => vec!["neg".into(), "pos".into()],
                },
                gliner2: None,
                laya: laya.then(|| LayaSection {
                    head_max_len: 16,
                    default_instructions: Some(DefaultInstructions {
                        choice: "which option".into(),
                        score: "which".into(),
                        noul: "ok".into(),
                    }),
                    option_rendering: OptionRendering::Laya,
                }),
                calibration: Default::default(),
                io: ClassifierIo {
                    token_type_ids: ranking.then(|| "token_type_ids".to_string()),
                    ..Default::default()
                },
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
    fn julia_renders_descriptions_and_needs_instructions_without_defaults() {
        let mut m = manifest(ClassifyTask::ZeroShotClassification);
        let section = m.classify.laya.as_mut().unwrap();
        section.option_rendering = OptionRendering::Julia;
        section.default_instructions = None;
        let b = InputBuilder::new(crate::laya::tests::tokenizer(), &m).unwrap();
        let params = laya_params(QuestionType::Choice, &["x: c", "y: d"]);
        let e = b.prepare("a b", &params).unwrap_err();
        assert!(e.to_string().contains("`instructions` is required"), "{e}");
        let params = ClassifyParams { instructions: Some("which option".into()), ..params };
        let p = b.prepare("a b", &params).unwrap();
        let t = |w| crate::laya::tests::tokenizer().token_to_id(w).unwrap() as i32;
        // The descriptions alone: [MASK] c [MASK] d, as laya's "c", "d".
        assert_eq!(p.ids[7..11], [3, t("c"), 3, t("d")]);
        assert_eq!(p.markers, vec![7, 9]);
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

    fn gliner2_builder() -> InputBuilder {
        let mut m = manifest(ClassifyTask::ZeroShotClassification);
        m.classify.format = Some(ClassifyFormat::Gliner2);
        m.classify.laya = None;
        m.classify.gliner2 = Some(Gliner2Section { default_instructions: "label".into() });
        InputBuilder::new(crate::gliner2::tests::tokenizer(), &m).unwrap()
    }

    fn gliner2_params(labels: &[&str]) -> ClassifyParams {
        ClassifyParams { candidate_labels: labels.iter().map(|s| s.to_string()).collect(), ..Default::default() }
    }

    #[test]
    fn gliner2_markers_buckets_and_default_prompt() {
        let b = gliner2_builder();
        let t = |w| crate::gliner2::tests::tokenizer().token_to_id(w).unwrap() as i32;
        let p = b.prepare("a b", &gliner2_params(&["c", "d"])).unwrap();
        // ( [P] label ( [L] c [L] d ) ) [SEP_TEXT] a b .
        assert_eq!(p.ids[..3], [t("("), t("[P]"), t("label")]);
        assert_eq!(p.markers, vec![4, 6]);
        assert_eq!((p.qtype, p.bucket, p.ids.len()), (None, 16, 14));
        // Long text is truncated to the largest bucket, not rejected.
        let long = (0..200).map(|i| ["a", "b", "c"][i % 3]).collect::<Vec<_>>().join(" ");
        let p = b.prepare(&long, &gliner2_params(&["c", "d"])).unwrap();
        assert_eq!((p.ids.len(), *p.ids.last().unwrap()), (32, t(".")));
        // A huge text is byte-capped first, and still fits.
        assert_eq!(b.prepare(&"a ".repeat(100_000), &gliner2_params(&["c", "d"])).unwrap().ids.len(), 32);
    }

    #[test]
    fn gliner2_rejects_what_it_cannot_honor() {
        let b = gliner2_builder();
        let ok = gliner2_params(&["c", "d"]);
        let cases = [
            ClassifyParams { truncate_prompt_tokens: Some(8), ..ok.clone() },
            ClassifyParams { truncation_side: TruncationSide::Left, ..ok.clone() },
            ClassifyParams { question_type: Some(QuestionType::Choice), ..ok.clone() },
            ClassifyParams { instructions: Some("x".repeat(600)), ..ok.clone() },
            gliner2_params(&["c"]),
            gliner2_params(&["a", "b", "c", "d", "e"]),
            gliner2_params(&["c: x", "c: y"]),
        ];
        for params in cases {
            assert!(matches!(b.prepare("a", &params), Err(Error::InvalidRequest(_))), "{params:?}");
        }
    }

    fn pair(truncate: Option<usize>, keep_query: bool) -> PairParams {
        PairParams { truncate_prompt_tokens: truncate, keep_query, ..Default::default() }
    }

    #[test]
    fn pairs_use_the_tokenizers_pair_template_and_segment_ids() {
        let b = builder(ClassifyTask::TextRanking);
        let t = |w| crate::laya::tests::tokenizer().token_to_id(w).unwrap() as i32;
        let p = b.prepare_pair("a b", "c d", &PairParams::default()).unwrap();
        assert_eq!(p.ids, vec![1, t("a"), t("b"), 2, t("c"), t("d"), 2]);
        assert_eq!(p.type_ids, vec![0, 0, 0, 0, 1, 1, 1]);
        assert_eq!(p.bucket, 8);
        assert!(p.markers.is_empty() && p.qtype.is_none());
        // A reranker scores pairs only, and only a reranker does.
        assert!(matches!(b.prepare("a", &ClassifyParams::default()), Err(Error::InvalidRequest(_))));
        let c = builder(ClassifyTask::TextClassification);
        assert!(matches!(c.prepare_pair("a", "b", &PairParams::default()), Err(Error::InvalidRequest(_))));
    }

    #[test]
    fn a_tokenizer_with_segment_ids_needs_the_input() {
        let mut m = manifest(ClassifyTask::TextRanking);
        m.classify.io.token_type_ids = None;
        let e = InputBuilder::new(crate::laya::tests::tokenizer(), &m).err().unwrap();
        assert!(e.to_string().contains("names no `token_type_ids`"), "{e}");
    }

    #[test]
    fn pair_truncation_follows_vllm_and_cohere() {
        let b = builder(ClassifyTask::TextRanking);
        // Over-length with no limit: a 400, as vLLM.
        let e = b.prepare_pair(&words(20), &words(20), &PairParams::default()).unwrap_err();
        assert!(e.to_string().contains("pair is 43 tokens, more than the model's maximum of 32"), "{e}");

        // longest_first to 8: 5 text tokens, split 2 + 3 (tokenizers' rule).
        let p = b.prepare_pair(&words(10), &words(10), &pair(Some(8), false)).unwrap();
        assert_eq!(p.ids.len(), 8);
        assert_eq!(p.type_ids, vec![0, 0, 0, 0, 1, 1, 1, 1]);
        // A short query leaves the rest to the document.
        let p = b.prepare_pair("a", &words(40), &pair(Some(16), false)).unwrap();
        assert_eq!((p.ids.len(), p.type_ids.iter().filter(|&&t| t == 0).count()), (16, 3));

        // Cohere: the document alone is cut to fit, the query kept whole.
        let p = b.prepare_pair(&words(3), &words(50), &pair(None, true)).unwrap();
        assert_eq!(p.ids.len(), 32);
        assert_eq!(p.type_ids.iter().filter(|&&t| t == 0).count(), 5, "[CLS] + 3 + [SEP]");
        let e = b.prepare_pair(&words(40), "a", &pair(None, true)).unwrap_err();
        assert!(e.to_string().contains("the query alone (40 tokens) fills"), "{e}");

        // Per-text cuts come first.
        let p = b
            .prepare_pair(&words(5), &words(5), &PairParams { max_tokens_per_doc: Some(2), max_tokens_per_query: Some(1), ..Default::default() })
            .unwrap();
        assert_eq!(p.ids.len(), 1 + 1 + 1 + 2 + 1);

        for n in [33, 3] {
            assert!(matches!(b.prepare_pair("a", "b", &pair(Some(n), false)), Err(Error::InvalidRequest(_))), "{n}");
        }
    }

    #[test]
    fn a_long_document_that_max_tokens_per_doc_cuts_needs_no_other_truncation() {
        let b = builder(ClassifyTask::TextRanking);
        // 2,000 words: far past the model and its byte cap, but cut to 10
        // tokens first, as vLLM cuts it, the pair fits.
        let long = words(2000);
        let per_doc = PairParams { max_tokens_per_doc: Some(10), ..Default::default() };
        let p = b.prepare_pair("a b", &long, &per_doc).unwrap();
        assert_eq!(p.ids.len(), 1 + 2 + 1 + 10 + 1);
        // Without a cut, the message names the text and the fields to set.
        let e = b.prepare_pair("a b", &long, &PairParams::default()).unwrap_err();
        assert!(e.to_string().contains("the document (") && e.to_string().contains("max_tokens_per_doc"), "{e}");
    }

    #[test]
    fn with_max_tokens_per_doc_only_the_document_is_truncated_as_in_vllm() {
        let b = builder(ClassifyTask::TextRanking);
        // longest_first would split 10 + 10 into 2 + 3 (see above); with
        // max_tokens_per_doc set, vLLM truncates only the document.
        let p = PairParams { truncate_prompt_tokens: Some(16), max_tokens_per_doc: Some(20), ..Default::default() };
        let got = b.prepare_pair(&words(10), &words(40), &p).unwrap();
        assert_eq!(got.ids.len(), 16);
        assert_eq!(got.type_ids.iter().filter(|&&t| t == 0).count(), 12, "[CLS] + 10 + [SEP]: the query whole");
        let e = b.prepare_pair(&words(14), &words(40), &p).unwrap_err();
        assert!(e.to_string().contains("the query alone (14 tokens)"), "{e}");
    }
}
