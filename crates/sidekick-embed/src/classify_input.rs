//! Turning one classification input into a [`Prepared`]: tokenization,
//! truncation, the zero-shot format's layout, and bucketing.
//!
//! Platform-neutral, so the token-level contract (docs/design/classify.md)
//! is tested everywhere; [`crate::CoremlClassifier`] runs the result.

use crate::{agentjev, fev, gliner2, laya};
use sidekick_core::manifest::{
    ClassifierManifest, ClassifyFormat, FevSection, Gliner2Section, LayaSection, NliSection, ResolvedClassifier,
};
use sidekick_core::{
    check_hypothesis_template, ClassifyParams, ClassifyTask, Error, NliPair, NliPairs, PairParams, Prepared, Result,
    TruncationSide,
};
use std::sync::{Arc, LazyLock};
use tokenizers::utils::truncation::{truncate_encodings, TruncationParams, TruncationStrategy};
use tokenizers::{Tokenizer, TruncationDirection};

/// Every classifier's input builder in use, shared by the classifier and
/// `/tokenize`.
static BUILDERS: LazyLock<crate::shared::Shared<InputBuilder>> = LazyLock::new(Default::default);

/// Under `/v1/classify` without `truncate_prompt_tokens`, a text longer
/// than this many bytes per token of the model's limit is refused without
/// being tokenized: no tokenizer spends 16 bytes per token.
const REFUSE_BYTES_PER_TOKEN: usize = 16;

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
    Fev { section: FevSection, delimiters: fev::Delimiters },
    Agentjev,
    Nli { section: NliSection },
}

fn invalid(message: impl Into<String>) -> Error {
    Error::InvalidRequest(message.into())
}

/// Whether the daemon's `/tokenize` takes a classifier's input, from its
/// manifest, so that a request is refused before the tokenizer loads: a
/// text-classification model's input is one text, counted as
/// `/v1/classify` counts it ([`InputBuilder::tokenize`]). A reranker's is
/// a pair, and a zero-shot format lays the text out with the request's
/// labels (and cuts it to fit, by design), so a count of the text alone
/// isn't one any route compares with a limit: a 400 saying so.
pub fn check_tokenizable(task: ClassifyTask, format: Option<ClassifyFormat>) -> Result<()> {
    if task == ClassifyTask::TextRanking {
        return Err(invalid(
            "a text-ranking model's input is a (query, document) pair, which /tokenize doesn't take",
        ));
    }
    if let Some(format) = format {
        let name = match format {
            ClassifyFormat::Laya => "laya",
            ClassifyFormat::Gliner2 => "gliner2",
            ClassifyFormat::Fev => "fev",
            ClassifyFormat::Agentjev => "agentjev",
            ClassifyFormat::Nli => "nli",
        };
        return Err(invalid(format!(
            "a zero-shot model in the {name} format lays its input out with the request's labels, so \
             /tokenize, which counts one text, doesn't take it"
        )));
    }
    Ok(())
}

impl InputBuilder {
    /// The classifier's builder: the copy already in use (by the
    /// classifier, or by the daemon's `/tokenize`), or a new one.
    pub fn shared(model: &ResolvedClassifier) -> Result<Arc<Self>> {
        BUILDERS.get_or_load(&model.dir, &model.manifest.id, || Self::load(model))
    }

    /// A copy of its own, outside the cache.
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
        // whose graph can't take them would score every pair wrong. Not
        // for nli: DeBERTa-v3's tokenizer gives pairs segment ids its model
        // never reads (no token-type embeddings), so there `[classify.io]`
        // decides, and the converter checks it against the checkpoint.
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
            Some(ClassifyFormat::Fev) => {
                let section = m.classify.fev.clone().ok_or_else(|| missing("fev"))?;
                let delimiters = fev::Delimiters::from_tokenizer(&tokenizer, &section.delimiters)?;
                Some(Format::Fev { section, delimiters })
            }
            Some(ClassifyFormat::Agentjev) => Some(Format::Agentjev),
            Some(ClassifyFormat::Nli) => Some(Format::Nli { section: m.classify.nli.clone().ok_or_else(|| missing("nli"))? }),
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

    /// The longest input the model takes, in tokens, special tokens
    /// included: its `max_seq_len`, after any cap (D33).
    pub fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }

    /// A text-classification model's input as `/v1/classify` counts it
    /// without `truncate_prompt_tokens`, however long: the whole text,
    /// wrapped in the tokenizer's special tokens unless
    /// `add_special_tokens` is false. The daemon's `/tokenize` reports
    /// this. Other models' inputs aren't one text on its own (a pair, or a
    /// zero-shot format's layout of text and labels), so they are a 400
    /// saying so. No byte bound: the caller bounds the text.
    pub fn tokenize(&self, text: &str, add_special_tokens: bool) -> Result<tokenizers::Encoding> {
        self.check_one_text()?;
        self.tokenizer.encode(text, add_special_tokens).map_err(|e| Error::Tokenizer(e.to_string()))
    }

    /// [`check_tokenizable`] for this model.
    fn check_one_text(&self) -> Result<()> {
        let format = self.format.as_ref().map(|f| match f {
            Format::Laya { .. } => ClassifyFormat::Laya,
            Format::Gliner2 { .. } => ClassifyFormat::Gliner2,
            Format::Fev { .. } => ClassifyFormat::Fev,
            Format::Agentjev => ClassifyFormat::Agentjev,
            Format::Nli { .. } => ClassifyFormat::Nli,
        });
        check_tokenizable(self.task, format)
    }

    /// `/v1/classify`'s verdict on one text without
    /// `truncate_prompt_tokens`: its whole encoding, special tokens
    /// included, or the 400 the route refuses it with. A text of more than
    /// 16 bytes per token of `max_seq_len` is refused without being
    /// tokenized, and one of more tokens than `max_seq_len` after. The one
    /// check behind `/v1/classify` and the daemon's `/tokenize` `fits`.
    pub fn check_text(&self, text: &str) -> Result<tokenizers::Encoding> {
        let max = self.max_seq_len;
        self.check_one_text()?;
        if text.len() > max.saturating_mul(REFUSE_BYTES_PER_TOKEN) {
            return Err(too_long(format!("{} bytes", text.len()), max));
        }
        let encoding = self.tokenize(text, true)?;
        if encoding.len() > max {
            return Err(too_long(format!("{} tokens", encoding.len()), max));
        }
        Ok(encoding)
    }

    /// Token ids back to text, special tokens kept. An id outside the
    /// vocabulary is a 400.
    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        crate::embed_input::decode(&self.tokenizer, ids)
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
            Some(Format::Fev { section, delimiters }) => self.prepare_fev(input, params, section, delimiters),
            Some(Format::Agentjev) => self.prepare_agentjev(input, params),
            Some(Format::Nli { section }) => self.prepare_nli(input, params, section),
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
        let encoding = match params.truncate_prompt_tokens {
            // The whole text, checked as /tokenize's `fits` checks it.
            None => self.check_text(input)?,
            Some(_) => {
                let text = match params.truncation_side {
                    TruncationSide::Right => crate::byte_cap(input, max),
                    TruncationSide::Left => byte_cap_end(input, max),
                };
                let tok_err = |e: tokenizers::Error| Error::Tokenizer(e.to_string());
                let mut encoding = self.tokenizer.encode(text, false).map_err(tok_err)?;
                let direction = match params.truncation_side {
                    TruncationSide::Right => TruncationDirection::Right,
                    TruncationSide::Left => TruncationDirection::Left,
                };
                encoding.truncate(limit - self.added_tokens, 0, direction);
                self.tokenizer.post_process(encoding, None, true).map_err(tok_err)?
            }
        };
        let ids: Vec<i32> = encoding.get_ids().iter().map(|&u| u as i32).collect();
        if ids.len() > max {
            return Err(too_long(format!("{} tokens", ids.len()), max));
        }
        Ok(Prepared { nli: None, bucket: self.bucket_for(ids.len()), ids, type_ids: vec![], markers: vec![], qtype: None, decide_pos: None, seg: vec![], position_ids: vec![] })
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
        Ok(Prepared { nli: None, bucket: self.bucket_for(ids.len()), ids, type_ids, markers: vec![], qtype: None, decide_pos: None, seg: vec![], position_ids: vec![] })
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
        check_labels(labels, 2, self.max_labels)?;
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
            nli: None,
            bucket: self.bucket_for(ids.len()),
            ids,
            type_ids: vec![],
            markers: seq.markers.iter().map(|&m| m as i32).collect(),
            qtype: Some(question_type.index()),
            decide_pos: None, seg: vec![], position_ids: vec![],
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
        // One label is a multi-label request's yes/no question; the server
        // requires two otherwise, before the model loads.
        check_labels(labels, 1, self.max_labels)?;
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
            nli: None,
            bucket: self.bucket_for(ids.len()),
            ids,
            type_ids: vec![],
            markers: seq.markers.iter().map(|&m| m as i32).collect(),
            qtype: None,
            decide_pos: None, seg: vec![], position_ids: vec![],
        })
    }
}

impl InputBuilder {
    /// The fev format (docs/design/classify.md): the state is truncated by
    /// design, keeping its start, so `truncate_prompt_tokens` and left
    /// truncation are 400s; the question and options are never cut.
    fn prepare_fev(
        &self,
        input: &str,
        params: &ClassifyParams,
        section: &FevSection,
        delimiters: &fev::Delimiters,
    ) -> Result<Prepared> {
        if params.truncate_prompt_tokens.is_some() {
            return Err(invalid(
                "truncate_prompt_tokens isn't supported by the fev format, which truncates the text \
                 itself, keeping its start",
            ));
        }
        if params.truncation_side == TruncationSide::Left {
            return Err(invalid(
                "truncation_side `left` isn't supported by the fev format, which keeps the text's start",
            ));
        }
        let Some(question_type) = params.question_type else {
            return Err(invalid("question_type is required by this model (choice, score or noul)"));
        };
        let labels = &params.candidate_labels;
        check_labels(labels, 2, self.max_labels)?;
        let options = fev::render_options(question_type, labels)?;
        // No instructions is an empty question, as fev's own API sends it.
        let instructions = params.instructions.as_deref().unwrap_or("");
        let seq = fev::build_sequence(
            &self.tokenizer,
            delimiters,
            input,
            instructions,
            &options,
            section.state_max_len,
            self.max_seq_len,
        )?;
        let ids: Vec<i32> = seq.ids.iter().map(|&u| u as i32).collect();
        Ok(Prepared {
            nli: None,
            bucket: self.bucket_for(ids.len()),
            ids,
            type_ids: vec![],
            markers: seq.option_ends.iter().map(|&m| m as i32).collect(),
            qtype: None,
            decide_pos: Some(seq.decide as i32), seg: vec![], position_ids: vec![],
        })
    }
}

impl InputBuilder {
    /// The agentjev format (docs/design/classify.md): the state, question
    /// and candidates as one tree. AgentJev never truncates, so neither
    /// does sidekick: a tree over the largest bucket is a 400, and so are
    /// `truncate_prompt_tokens` and left truncation.
    fn prepare_agentjev(&self, input: &str, params: &ClassifyParams) -> Result<Prepared> {
        if params.truncate_prompt_tokens.is_some() {
            return Err(invalid(
                "truncate_prompt_tokens isn't supported by the agentjev format, which never truncates; shorten the \
                 input instead",
            ));
        }
        if params.truncation_side == TruncationSide::Left {
            return Err(invalid("truncation_side `left` isn't supported by the agentjev format, which never truncates"));
        }
        let Some(question_type) = params.question_type else {
            return Err(invalid("question_type is required by this model (choice, score or noul)"));
        };
        let Some(instructions) = params.instructions.as_deref() else {
            return Err(invalid("instructions is required by this model: the question to decide"));
        };
        let labels = &params.candidate_labels;
        check_labels(labels, 2, self.max_labels)?;
        let candidates = agentjev::render_candidates(question_type, labels)?;
        let tree = agentjev::build_tree(&self.tokenizer, input, instructions, &candidates, self.max_seq_len)?;
        let ids: Vec<i32> = tree.ids.iter().map(|&u| u as i32).collect();
        Ok(Prepared {
            nli: None,
            bucket: self.bucket_for(ids.len()),
            ids,
            type_ids: vec![],
            markers: tree.cand_ends.iter().map(|&m| m as i32).collect(),
            qtype: None,
            decide_pos: None,
            seg: tree.seg,
            position_ids: tree.position_ids,
        })
    }
}

impl InputBuilder {
    /// The nli format (D41), as Hugging Face's zero-shot pipeline builds
    /// its inputs: for each label, the tokenizer's pair of the text and the
    /// hypothesis template with the label in it (`[CLS] text [SEP]
    /// hypothesis [SEP]` for BERT, `<s> text </s></s> hypothesis </s>` for
    /// RoBERTa), with its segment ids when the graph takes them. Like the
    /// pipeline (`truncation="only_first"`), only the text is truncated, to
    /// fit the model, on `truncation_side`; `truncate_prompt_tokens` may
    /// lower that limit. A hypothesis that leaves no room for the text is a
    /// 400, never cut. One label always scores against its contradiction,
    /// as the pipeline does.
    fn prepare_nli(&self, input: &str, params: &ClassifyParams, section: &NliSection) -> Result<Prepared> {
        if params.question_type.is_some() || params.instructions.is_some() {
            return Err(invalid("the nli format takes no question_type or instructions"));
        }
        let labels = &params.candidate_labels;
        check_labels(labels, 1, self.max_labels)?;
        let template = params.hypothesis_template.as_deref().unwrap_or(&section.hypothesis_template);
        check_hypothesis_template(template).map_err(invalid)?;
        let max = self.max_seq_len;
        let limit = match params.truncate_prompt_tokens {
            None => max,
            Some(n) if n > max => {
                return Err(invalid(format!("truncate_prompt_tokens {n} exceeds the model's maximum of {max} tokens")))
            }
            Some(n) if n <= self.pair_added_tokens => {
                return Err(invalid(format!(
                    "truncate_prompt_tokens {n} leaves no room for text after the pair's {} special tokens",
                    self.pair_added_tokens
                )))
            }
            Some(n) => n,
        };
        let budget = limit - self.pair_added_tokens;
        let direction = match params.truncation_side {
            TruncationSide::Right => TruncationDirection::Right,
            TruncationSide::Left => TruncationDirection::Left,
        };
        // Bound the tokenizer's work on the text (16 bytes per token is
        // beyond any vocabulary), keeping the end truncation keeps.
        let text = match params.truncation_side {
            TruncationSide::Right => crate::byte_cap(input, max),
            TruncationSide::Left => byte_cap_end(input, max),
        };
        let tok_err = |e: tokenizers::Error| Error::Tokenizer(e.to_string());
        let premise = self.tokenizer.encode(text, false).map_err(tok_err)?;
        let mut pairs = Vec::with_capacity(labels.len());
        for label in labels {
            let hypothesis = template.replacen("{}", label, 1);
            let h = self.tokenizer.encode(hypothesis.as_str(), false).map_err(tok_err)?;
            let h_len = h.len();
            let no_room = || {
                invalid(format!(
                    "the hypothesis for label `{label}` ({h_len} tokens) leaves no room for the text in the model's \
                     {limit}-token input"
                ))
            };
            if h.len() >= budget {
                return Err(no_room());
            }
            let (p, h) = if premise.len() + h.len() > budget {
                let truncation = TruncationParams {
                    max_length: budget,
                    strategy: TruncationStrategy::OnlyFirst,
                    stride: 0,
                    direction,
                };
                truncate_encodings(premise.clone(), Some(h.clone()), &truncation).map_err(|_| no_room())?
            } else {
                (premise.clone(), Some(h))
            };
            let pair = self.tokenizer.post_process(p, h, true).map_err(tok_err)?;
            let ids: Vec<i32> = pair.get_ids().iter().map(|&u| u as i32).collect();
            if ids.len() > limit {
                return Err(no_room());
            }
            let type_ids = if self.segment_ids { pair.get_type_ids().iter().map(|&u| u as i32).collect() } else { vec![] };
            pairs.push(NliPair { bucket: self.bucket_for(ids.len()), ids, type_ids });
        }
        let bucket = pairs.iter().map(|p| p.bucket).max().expect("at least one label");
        Ok(Prepared {
            nli: Some(NliPairs { pairs, per_label: params.multi_label || labels.len() == 1 }),
            bucket,
            ids: vec![],
            type_ids: vec![],
            markers: vec![],
            qtype: None,
            decide_pos: None,
            seg: vec![],
            position_ids: vec![],
        })
    }
}

/// Zero-shot label checks that don't need the tokenizer. `min_labels` is 2,
/// except for a gliner2 request with `multi_label`, where one label is a
/// yes/no question (gliner2 takes one).
pub fn check_labels(labels: &[String], min_labels: usize, max_labels: usize) -> Result<()> {
    if labels.len() < min_labels {
        return Err(invalid(format!("candidate_labels needs at least {min_labels} labels")));
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
    use sidekick_core::manifest::{ClassifierIo, ClassifySection, DefaultInstructions, FevSection, OptionRendering};
    use sidekick_core::{ProblemType, QuestionType};

    fn manifest(task: ClassifyTask) -> ClassifierManifest {
        let laya = task == ClassifyTask::ZeroShotClassification;
        let ranking = task == ClassifyTask::TextRanking;
        ClassifierManifest {
            compute_units: Default::default(),
            ane_weight_limit: Default::default(),
            cpu_seq_limit: Default::default(),
            seq_cap: None,
            compute_units_source: Default::default(),
            placement: None,
            chunking: None,
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
                nli: None,
                format: laya.then_some(ClassifyFormat::Laya),
                max_labels: laya.then_some(4),
                labels: match task {
                    ClassifyTask::ZeroShotClassification => vec![],
                    ClassifyTask::TextRanking => vec!["score".into()],
                    ClassifyTask::TextClassification => vec!["neg".into(), "pos".into()],
                },
                gliner2: None,
                fev: None,
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
    fn tokenize_counts_as_prepare_does_and_past_the_limit() {
        let b = builder(ClassifyTask::TextClassification);
        let ids = |e: tokenizers::Encoding| e.get_ids().iter().map(|&u| u as i32).collect::<Vec<_>>();
        let p = b.prepare(&words(30), &ClassifyParams::default()).unwrap();
        assert_eq!(ids(b.tokenize(&words(30), true).unwrap()), p.ids, "at the limit");
        // Past it, prepare's 400 names tokenize's count.
        assert_eq!(b.tokenize(&words(31), true).unwrap().len(), 33);
        assert_eq!(b.tokenize(&words(31), false).unwrap().len(), 31);
        assert_eq!(b.tokenize(&words(300), true).unwrap().len(), 302, "not truncated");
        // /v1/classify's verdict, the one /tokenize's `fits` reports.
        assert_eq!(ids(b.check_text(&words(30)).unwrap()), p.ids);
        assert!(b.check_text(&words(31)).unwrap_err().to_string().contains("33 tokens"));
        let run = "x".repeat(32 * 16 + 1);
        assert_eq!(b.tokenize(&run, true).unwrap().len(), 3, "one unknown token");
        assert!(b.check_text(&run).unwrap_err().to_string().contains("513 bytes"));
        // Pairs and zero-shot layouts aren't one text.
        let e = builder(ClassifyTask::TextRanking).tokenize("a", true).unwrap_err();
        assert!(e.to_string().contains("(query, document) pair"), "{e}");
        let e = builder(ClassifyTask::ZeroShotClassification).tokenize("a", true).unwrap_err();
        assert!(e.to_string().contains("in the laya format"), "{e}");
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
            gliner2_params(&[]),
            gliner2_params(&["a", "b", "c", "d", "e"]),
            gliner2_params(&["c: x", "c: y"]),
        ];
        for params in cases {
            assert!(matches!(b.prepare("a", &params), Err(Error::InvalidRequest(_))), "{params:?}");
        }
        // One label lays out (a multi-label yes/no question; the server
        // requires two for a single-label request before the model loads).
        assert_eq!(b.prepare("a", &gliner2_params(&["c"])).unwrap().markers.len(), 1);
    }

    fn fev_builder(state_max_len: usize) -> InputBuilder {
        let mut m = manifest(ClassifyTask::ZeroShotClassification);
        m.classify.format = Some(ClassifyFormat::Fev);
        m.classify.laya = None;
        m.classify.fev = Some(FevSection { delimiters: crate::fev::tests::delimiters(), state_max_len });
        InputBuilder::new(crate::fev::tests::tokenizer(), &m).unwrap()
    }

    fn fev_params(qt: QuestionType, labels: &[&str]) -> ClassifyParams {
        ClassifyParams {
            question_type: Some(qt),
            candidate_labels: labels.iter().map(|s| s.to_string()).collect(),
            ..Default::default()
        }
    }

    #[test]
    fn fev_lays_out_options_and_the_decide_position() {
        let b = fev_builder(8);
        let t = |w| crate::fev::tests::tokenizer().token_to_id(w).unwrap() as i32;
        let p = b.prepare("a", &fev_params(QuestionType::Noul, &["false", "true: dry"])).unwrap();
        // [s a] [q] [o no c] [o yes : ␣ d? …] [d]: no instructions is an
        // empty question.
        assert_eq!(p.ids[..4], [t("<|s|>"), t("a"), t("<|q|>"), t("<|o|>")]);
        assert_eq!(p.ids[4], t("no"));
        assert!(p.markers.iter().all(|&m| p.ids[m as usize] == t("<|c|>")));
        assert_eq!(p.markers.len(), 2);
        assert_eq!(p.decide_pos, Some(p.ids.len() as i32 - 1));
        assert_eq!(p.ids[p.ids.len() - 1], t("<|d|>"));
        assert_eq!(p.qtype, None);
        // A long state is cut, not refused.
        let p = b.prepare(&"a ".repeat(100), &fev_params(QuestionType::Choice, &["b", "c"])).unwrap();
        assert_eq!(p.ids[8], t("<|q|>"), "the state keeps state_max_len = 8 tokens, its delimiter included");
    }

    #[test]
    fn fev_rejects_what_it_cannot_honor() {
        let b = fev_builder(8);
        let ok = fev_params(QuestionType::Choice, &["b", "c"]);
        let cases = [
            ClassifyParams { truncate_prompt_tokens: Some(8), ..ok.clone() },
            ClassifyParams { truncation_side: TruncationSide::Left, ..ok.clone() },
            ClassifyParams { question_type: None, ..ok.clone() },
            fev_params(QuestionType::Choice, &["b"]),
            fev_params(QuestionType::Choice, &["a", "b", "c", "d", "e"]),
            fev_params(QuestionType::Choice, &["b", "b: "]),
            fev_params(QuestionType::Noul, &["yes", "no"]),
        ];
        for params in cases {
            assert!(matches!(b.prepare("a", &params), Err(Error::InvalidRequest(_))), "{params:?}");
        }
        // The window, max_seq_len 32: an option of 12 words (23 tokens with
        // their spaces) makes the row exactly 32 and fits; one word more is
        // a 400, as options are never cut.
        let option = |words: usize| "d ".repeat(words).trim().to_string();
        let p = b.prepare("a", &fev_params(QuestionType::Choice, &[&option(12), "c"])).unwrap();
        assert_eq!((p.ids.len(), p.bucket), (32, 32));
        let e = b.prepare("a", &fev_params(QuestionType::Choice, &[&option(13), "c"])).unwrap_err();
        assert!(e.to_string().contains("the model's input holds 32"), "{e}");
    }

    fn agentjev_builder(max_seq_len: usize) -> InputBuilder {
        let mut m = manifest(ClassifyTask::ZeroShotClassification);
        m.classify.format = Some(ClassifyFormat::Agentjev);
        m.classify.laya = None;
        m.max_seq_len = max_seq_len;
        m.buckets = vec![max_seq_len / 2, max_seq_len];
        InputBuilder::new(crate::agentjev::tests::tokenizer(), &m).unwrap()
    }

    fn agentjev_params(qt: QuestionType, labels: &[&str]) -> ClassifyParams {
        ClassifyParams {
            question_type: Some(qt),
            instructions: Some("done".into()),
            candidate_labels: labels.iter().map(|s| s.to_string()).collect(),
            ..Default::default()
        }
    }

    #[test]
    fn agentjev_lays_out_a_tree() {
        let b = agentjev_builder(32);
        let t = |w| crate::agentjev::tests::tokenizer().token_to_id(w).unwrap() as i32;
        let p = b.prepare("state", &agentjev_params(QuestionType::Noul, &["false", "true: yes"])).unwrap();
        assert_eq!(p.ids.len(), 15);
        assert_eq!((p.seg.len(), p.position_ids.len()), (15, 15));
        // Each marker is its candidate's last token: FALSE, then the description.
        assert_eq!(p.markers, vec![10, 14]);
        assert_eq!((p.ids[10], p.ids[14]), (t("FALSE"), t("yes")));
        assert_eq!((p.seg[7], p.seg[10], p.seg[11], p.seg[14]), (1, 1, 2, 2));
        assert_eq!((p.position_ids[7], p.position_ids[11]), (7, 7));
        assert_eq!((p.qtype, p.decide_pos, p.bucket), (None, None, 16));
        assert!(p.type_ids.is_empty());
    }

    #[test]
    fn agentjev_rejects_what_it_cannot_honor() {
        let b = agentjev_builder(32);
        let ok = agentjev_params(QuestionType::Choice, &["b", "c"]);
        let cases = [
            ClassifyParams { truncate_prompt_tokens: Some(8), ..ok.clone() },
            ClassifyParams { truncation_side: TruncationSide::Left, ..ok.clone() },
            ClassifyParams { question_type: None, ..ok.clone() },
            ClassifyParams { instructions: None, ..ok.clone() },
            ClassifyParams { instructions: Some(" ".into()), ..ok.clone() },
            agentjev_params(QuestionType::Choice, &["b"]),
            agentjev_params(QuestionType::Choice, &["a", "b", "c", "d", "e"]),
            agentjev_params(QuestionType::Choice, &["x: b", "b"]),
            agentjev_params(QuestionType::Noul, &["yes", "no"]),
        ];
        for params in cases {
            assert!(matches!(b.prepare("state", &params), Err(Error::InvalidRequest(_))), "{params:?}");
        }
        assert!(matches!(b.prepare(" ", &ok), Err(Error::InvalidRequest(_))));
        // Never truncated: the largest bucket holds 32; a longer tree is a 400.
        let e = b.prepare(&"a ".repeat(20), &ok).unwrap_err();
        assert!(e.to_string().contains("the model's input holds 32"), "{e}");
    }

    fn pair(truncate: Option<usize>, keep_query: bool) -> PairParams {
        PairParams { truncate_prompt_tokens: truncate, keep_query, ..Default::default() }
    }

    /// A RoBERTa-style tokenizer: laya's test vocabulary with `<s>` and
    /// `</s>` as its specials and RobertaProcessing's pair layout,
    /// `<s> A </s></s> B </s>`, every segment id 0.
    fn roberta_tokenizer() -> Tokenizer {
        let bert: serde_json::Value = serde_json::from_str(&crate::laya::tests::tokenizer().to_string(false).unwrap()).unwrap();
        let mut json = bert.clone();
        let mut vocab = bert["model"]["vocab"].as_object().unwrap().clone();
        vocab.remove("[CLS]");
        vocab.remove("[SEP]");
        vocab.insert("<s>".into(), 1.into());
        vocab.insert("</s>".into(), 2.into());
        json["model"]["vocab"] = vocab.into();
        let special = |id: u32, content: &str| {
            serde_json::json!({"id": id, "content": content, "single_word": false, "lstrip": false,
                               "rstrip": false, "normalized": false, "special": true})
        };
        json["added_tokens"] = serde_json::json!([special(0, "[UNK]"), special(1, "<s>"), special(2, "</s>")]);
        json["post_processor"] = serde_json::json!({
            "type": "RobertaProcessing", "sep": ["</s>", 2], "cls": ["<s>", 1],
            "trim_offsets": true, "add_prefix_space": false
        });
        Tokenizer::from_bytes(json.to_string()).unwrap()
    }

    /// An nli model over the test vocabulary: buckets 8, 16, 32.
    fn nli_builder(tokenizer: Tokenizer, segment_ids: bool) -> InputBuilder {
        let mut m = manifest(ClassifyTask::TextClassification);
        m.task = ClassifyTask::ZeroShotClassification;
        m.classify.labels = vec![];
        m.classify.format = Some(ClassifyFormat::Nli);
        m.classify.max_labels = Some(3);
        m.classify.nli = Some(NliSection {
            entailment: 2,
            contradiction: 0,
            n_classes: 3,
            hypothesis_template: "{} fits".into(),
        });
        m.classify.io.token_type_ids = segment_ids.then(|| "token_type_ids".into());
        InputBuilder::new(tokenizer, &m).unwrap()
    }

    fn nli_params(labels: &[&str]) -> ClassifyParams {
        ClassifyParams { candidate_labels: labels.iter().map(|l| l.to_string()).collect(), ..Default::default() }
    }

    fn nli(p: &Prepared) -> &NliPairs {
        p.nli.as_ref().expect("an nli input")
    }

    #[test]
    fn nli_pairs_each_label_as_the_tokenizer_pairs_text() {
        let t = |w| crate::laya::tests::tokenizer().token_to_id(w).unwrap() as i32;
        // BERT: [CLS] text [SEP] hypothesis [SEP], segments 0 then 1.
        let b = nli_builder(crate::laya::tests::tokenizer(), true);
        let p = b.prepare("c d e", &nli_params(&["a", "b"])).unwrap();
        let pairs = &nli(&p).pairs;
        assert_eq!(pairs[0].ids, vec![1, t("c"), t("d"), t("e"), 2, t("a"), t("fits"), 2]);
        assert_eq!(pairs[1].ids[5], t("b"));
        assert_eq!(pairs[0].type_ids, vec![0, 0, 0, 0, 0, 1, 1, 1]);
        assert_eq!((pairs[0].bucket, p.bucket), (8, 8));
        assert!(p.ids.is_empty() && p.markers.is_empty());
        assert_eq!(p.tokens(), 16);
        assert_eq!(p.buckets(), vec![8, 8]);
        assert!(!nli(&p).per_label, "several labels: a softmax across them");
        // Segment ids only when the graph takes them: DeBERTa-v3's tokenizer
        // gives pairs segment ids its model never reads.
        let deberta = nli_builder(crate::laya::tests::tokenizer(), false);
        assert!(nli(&deberta.prepare("c d e", &nli_params(&["a"])).unwrap()).pairs[0].type_ids.is_empty());
        // RoBERTa: <s> text </s></s> hypothesis </s>.
        let r = nli_builder(roberta_tokenizer(), false);
        let p = r.prepare("c d e", &nli_params(&["a"])).unwrap();
        assert_eq!(nli(&p).pairs[0].ids, vec![1, t("c"), t("d"), t("e"), 2, 2, t("a"), t("fits"), 2]);
        assert_eq!(p.bucket, 16);
        // One label, or multi_label: each label scored on its own.
        assert!(nli(&p).per_label);
        let multi = ClassifyParams { multi_label: true, ..nli_params(&["a", "b"]) };
        assert!(nli(&r.prepare("c", &multi).unwrap()).per_label);
    }

    #[test]
    fn nli_truncates_only_the_text_on_the_requested_side() {
        let b = nli_builder(crate::laya::tests::tokenizer(), true);
        let t = |w| crate::laya::tests::tokenizer().token_to_id(w).unwrap() as i32;
        // 40 words of text; the pair fits 32 with 3 specials and the
        // 2-token hypothesis, so 27 text tokens stay.
        let text = (0..40).map(|i| ["a", "b", "c", "d", "e"][i % 5]).collect::<Vec<_>>().join(" ");
        let right = b.prepare(&text, &nli_params(&["f"])).unwrap();
        let ids = &nli(&right).pairs[0].ids;
        assert_eq!(ids.len(), 32);
        assert_eq!(&ids[..3], &[1, t("a"), t("b")], "keeps the start");
        assert_eq!(&ids[28..], &[2, t("f"), t("fits"), 2], "the hypothesis whole");
        let left = b.prepare(&text, &ClassifyParams { truncation_side: TruncationSide::Left, ..nli_params(&["f"]) }).unwrap();
        let ids = &nli(&left).pairs[0].ids;
        assert_eq!(ids.len(), 32);
        // Word 39 is "e", and 27 words back from it, word 13, is "d".
        assert_eq!(&ids[..2], &[1, t("d")], "keeps the end");
        assert_eq!(&ids[26..], &[t("d"), t("e"), 2, t("f"), t("fits"), 2]);
        // truncate_prompt_tokens lowers the limit.
        let capped = b.prepare(&text, &ClassifyParams { truncate_prompt_tokens: Some(16), ..nli_params(&["f"]) }).unwrap();
        assert_eq!((nli(&capped).pairs[0].ids.len(), capped.bucket), (16, 16));
        // A hypothesis that leaves no room for the text is a 400, never cut.
        let long = ClassifyParams { hypothesis_template: Some(format!("{} {{}}", "a ".repeat(29))), ..nli_params(&["f"]) };
        let e = b.prepare("c", &long).unwrap_err();
        assert!(matches!(e, Error::InvalidRequest(_)) && e.to_string().contains("leaves no room"), "{e}");
    }

    #[test]
    fn nli_templates_have_one_slot() {
        let b = nli_builder(crate::laya::tests::tokenizer(), true);
        let t = |w| crate::laya::tests::tokenizer().token_to_id(w).unwrap() as i32;
        let custom = ClassifyParams { hypothesis_template: Some("which {}".into()), ..nli_params(&["a"]) };
        assert_eq!(nli(&b.prepare("c", &custom).unwrap()).pairs[0].ids, vec![1, t("c"), 2, t("which"), t("a"), 2]);
        for bad in ["which", "{} {}", "{0}", "{x} {}"] {
            let p = ClassifyParams { hypothesis_template: Some(bad.into()), ..nli_params(&["a"]) };
            assert!(b.prepare("c", &p).unwrap_err().to_string().contains("exactly one `{}`"), "{bad}");
        }
        assert!(b.prepare("c", &nli_params(&[])).is_err(), "at least one label");
        assert!(b.prepare("c", &nli_params(&["a", "b", "c", "d"])).is_err(), "at most max_labels");
        let q = ClassifyParams { question_type: Some(QuestionType::Choice), ..nli_params(&["a"]) };
        assert!(b.prepare("c", &q).is_err());
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
