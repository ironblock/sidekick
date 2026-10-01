//! AgentJev's input format: a Rust port of `encode_paths`, and of the
//! candidate rules of `prepare`, in `jev_service/contract.py`
//! (malevrigns/agent-jev at revision a965ca8, Apache-2.0), laid out as one
//! tree per question.
//!
//! AgentJev scores each candidate from the hidden state at the end of its
//! own causal path, `[STATE] state` `\n[QUESTION] question`
//! `\n[CANDIDATE] candidate`, each fragment tokenized on its own without
//! special tokens and concatenated, with no BOS or EOS. Every path of a
//! question shares the state and question, so sidekick lays them out once:
//!
//! ```text
//! [prefix: state, question][suffix 1: candidate 1]…[suffix C: candidate C]
//! ```
//!
//! Each token carries its segment (0 for the prefix, `c` for candidate `c`)
//! and its position, which continues from the prefix's end in every suffix.
//! The graph lets a token attend to the earlier tokens of the prefix and of
//! its own segment only, so each candidate sees exactly its own path, and
//! the candidate head runs on the hidden state at each suffix's last token.
//! That reproduces the per-path scoring, which was measured exact in fp32
//! (docs/design/classify.md). The token-id fixture
//! (`fixtures/classify/<id>.tokens.json`, generated with the checkpoint's own
//! `encode_paths`) checks the port.

use crate::laya::noul_description;
use sidekick_core::{Error, QuestionType, Result};
use tokenizers::Tokenizer;

/// AgentJev's default text for an undescribed boolean candidate
/// (`k.upper()` in `prepare`).
const NOUL_DEFAULT: [&str; 2] = ["FALSE", "TRUE"];

/// Python's `str.isspace()`, which AgentJev's `semantic()` uses through
/// `strip()`: Unicode White_Space plus the information separators
/// U+001C..U+001F, which Rust's `char::is_whitespace` leaves out.
fn py_space(c: char) -> bool {
    c.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&c)
}

/// AgentJev's `semantic(value, name)` for text: nonempty once Python's
/// whitespace is stripped, and kept as is.
fn semantic<'a>(text: &'a str, name: &str) -> Result<&'a str> {
    if text.chars().all(py_space) {
        return Err(Error::InvalidRequest(format!("the {name} must be nonempty text")));
    }
    Ok(text)
}

/// The request's labels as AgentJev's candidate texts, in label order. A
/// label's description is the text after its first `": "`; an empty
/// description counts as none.
/// - choice: the description, else the label (AgentJev's option values);
/// - score: the label as given (AgentJev's level descriptions);
/// - noul: labels `false` then `true` (each optionally `"false: …"`); each
///   candidate is its description, else `FALSE` or `TRUE`, as AgentJev's
///   boolean criteria default.
///
/// The candidate head is permutation-equivariant, so label order changes
/// nothing but the order of `probs`. Candidates must be nonempty and
/// distinct, as AgentJev's `prepare` requires; both are 400s.
pub fn render_candidates(question_type: QuestionType, labels: &[String]) -> Result<Vec<String>> {
    let candidates: Vec<String> = match question_type {
        QuestionType::Choice => labels
            .iter()
            .map(|l| match l.split_once(": ") {
                Some((_, desc)) if !desc.is_empty() => desc.to_string(),
                Some((key, _)) => key.to_string(),
                None => l.clone(),
            })
            .collect(),
        QuestionType::Score => labels.to_vec(),
        QuestionType::Noul => {
            let bad = || {
                Error::InvalidRequest(
                    "a noul question's candidate_labels are `false` then `true`, each \
                     optionally with a description (`\"false: …\"`, `\"true: …\"`)"
                        .into(),
                )
            };
            let [f, t] = labels else { return Err(bad()) };
            let f = noul_description(f, "false").ok_or_else(bad)?;
            let t = noul_description(t, "true").ok_or_else(bad)?;
            [f, t]
                .iter()
                .zip(NOUL_DEFAULT)
                .map(|(d, default)| (*d).filter(|d| !d.is_empty()).unwrap_or(default).to_string())
                .collect()
        }
    };
    for (i, c) in candidates.iter().enumerate() {
        semantic(c, &format!("candidate for label `{}`", labels[i]))?;
        if let Some(j) = candidates[..i].iter().position(|p| p == c) {
            return Err(Error::InvalidRequest(format!(
                "candidate labels `{}` and `{}` give the same candidate text; AgentJev's candidates must be distinct",
                labels[j], labels[i]
            )));
        }
    }
    Ok(candidates)
}

/// One question laid out as a tree.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Tree {
    pub ids: Vec<u32>,
    /// Parallel to `ids`: 0 for the prefix, `c` for candidate `c` (1-based).
    pub seg: Vec<i32>,
    /// Parallel to `ids`: 0.. over the prefix, then each candidate's
    /// positions continue from the prefix's end.
    pub position_ids: Vec<i32>,
    /// Each candidate's last token, in label order.
    pub cand_ends: Vec<usize>,
    /// The prefix's length: the state and question tokens.
    pub prefix_len: usize,
}

fn text_ids(tok: &Tokenizer, text: &str) -> Result<Vec<u32>> {
    Ok(tok.encode(text, false).map_err(|e| Error::Tokenizer(e.to_string()))?.get_ids().to_vec())
}

/// AgentJev's `encode_paths` for one question, as a tree. Nothing is
/// truncated: a tree longer than `max_len` is a 400, as AgentJev refuses a
/// path over its limit. Every fragment is capped at 16 bytes per token of
/// `max_len` before it is tokenized, which bounds the tokenizer's work; a
/// longer fragment couldn't fit.
pub fn build_tree(tok: &Tokenizer, state: &str, question: &str, candidates: &[String], max_len: usize) -> Result<Tree> {
    let cap = max_len.saturating_mul(16);
    let over = |what: &str, n: usize| {
        Error::InvalidRequest(format!(
            "the {what} takes {n} tokens; the model's input holds {max_len}. Shorten the input: AgentJev's \
             format never truncates"
        ))
    };
    if state.len() > cap {
        return Err(Error::InvalidRequest(format!(
            "the input is {} bytes, more than {cap} ({max_len} tokens × 16): shorten it; AgentJev's format never \
             truncates",
            state.len()
        )));
    }
    if question.len() > cap || candidates.iter().any(|c| c.len() > cap) {
        return Err(Error::InvalidRequest(format!(
            "the instructions and candidate labels can't fit the model's {max_len}-token input"
        )));
    }
    let state = semantic(state, "input")?;
    let question = semantic(question, "instructions")?;
    let state_text = if state.starts_with("[STATE]") { state.to_string() } else { format!("[STATE] {state}") };
    let mut ids = text_ids(tok, &state_text)?;
    ids.extend(text_ids(tok, &format!("\n[QUESTION] {question}"))?);
    let prefix_len = ids.len();
    if prefix_len > max_len {
        return Err(over("input and instructions", prefix_len));
    }
    let mut seg = vec![0; prefix_len];
    let mut position_ids: Vec<i32> = (0..prefix_len as i32).collect();
    let mut cand_ends = Vec::with_capacity(candidates.len());
    for (c, candidate) in candidates.iter().enumerate() {
        let suffix = text_ids(tok, &format!("\n[CANDIDATE] {candidate}"))?;
        if suffix.is_empty() {
            return Err(Error::Tokenizer("a candidate tokenized to nothing".into()));
        }
        position_ids.extend((prefix_len..prefix_len + suffix.len()).map(|p| p as i32));
        seg.extend(std::iter::repeat_n(c as i32 + 1, suffix.len()));
        ids.extend(suffix);
        cand_ends.push(ids.len() - 1);
        if ids.len() > max_len {
            let total = ids.len()
                + candidates[c + 1..]
                    .iter()
                    .map(|c| text_ids(tok, &format!("\n[CANDIDATE] {c}")).map(|s| s.len()))
                    .sum::<Result<usize>>()?;
            return Err(over("input, instructions and candidate labels together", total));
        }
    }
    Ok(Tree { ids, seg, position_ids, cand_ends, prefix_len })
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// A word-level tokenizer whose pre-tokenizer isolates whitespace runs,
    /// AgentJev's tags and punctuation, plus an added special token, as
    /// Qwen3's `<|endoftext|>` is.
    pub(crate) fn tokenizer() -> Tokenizer {
        let mut vocab = serde_json::Map::new();
        let words = "[UNK] <|endoftext|> [STATE] [QUESTION] [CANDIDATE] \n state done yes no a b c d e : FALSE TRUE";
        for (i, w) in words.split(' ').enumerate() {
            vocab.insert(w.to_string(), i.into());
        }
        let json = serde_json::json!({
            "version": "1.0", "truncation": null, "padding": null,
            "added_tokens": [{"id": 1, "content": "<|endoftext|>", "single_word": false, "lstrip": false,
                              "rstrip": false, "normalized": false, "special": true}],
            "normalizer": null,
            "pre_tokenizer": {"type": "Split", "pattern": {"Regex": "\\[[A-Z]+\\]|\\n| +|[:]|[^\\s:\\[]+"},
                              "behavior": "Isolated", "invert": false},
            "post_processor": null, "decoder": null,
            "model": {"type": "WordLevel", "vocab": vocab, "unk_token": "[UNK]"}
        });
        Tokenizer::from_bytes(json.to_string().as_bytes()).unwrap()
    }

    fn s(v: &[&str]) -> Vec<String> {
        v.iter().map(|x| x.to_string()).collect()
    }

    fn id(w: &str) -> u32 {
        tokenizer().token_to_id(w).unwrap()
    }

    /// The ids without the space pieces ([UNK] here), to compare layouts.
    fn words(ids: &[u32]) -> Vec<u32> {
        ids.iter().copied().filter(|&i| i != id("[UNK]")).collect()
    }

    #[test]
    fn candidates_render_as_agentjev_reads_them() {
        let r = |q, l: &[&str]| render_candidates(q, &s(l));
        // choice: the description, else the key; an empty description drops ": ".
        assert_eq!(r(QuestionType::Choice, &["a", "b: c", "d: e: f", "g: ", "h:i"]).unwrap(), s(&["a", "c", "e: f", "g", "h:i"]));
        // score: as given.
        assert_eq!(r(QuestionType::Score, &["low", "high: x"]).unwrap(), s(&["low", "high: x"]));
        // noul: each side's description, else FALSE / TRUE.
        assert_eq!(r(QuestionType::Noul, &["false", "true"]).unwrap(), s(&["FALSE", "TRUE"]));
        assert_eq!(r(QuestionType::Noul, &["false: unmet", "true"]).unwrap(), s(&["unmet", "TRUE"]));
        assert_eq!(r(QuestionType::Noul, &["false: ", "true: met"]).unwrap(), s(&["FALSE", "met"]));
        for bad in [&["true", "false"][..], &["false"], &["no", "yes"]] {
            assert!(matches!(r(QuestionType::Noul, bad), Err(Error::InvalidRequest(_))), "{bad:?}");
        }
        // Distinct and nonempty, as AgentJev's prepare requires.
        let e = r(QuestionType::Choice, &["x: same", "y: same"]).unwrap_err();
        assert!(e.to_string().contains("must be distinct"), "{e}");
        let e = r(QuestionType::Score, &["a", " \u{1c}"]).unwrap_err();
        assert!(e.to_string().contains("nonempty"), "{e}");
    }

    fn tree(state: &str, question: &str, candidates: &[&str], max_len: usize) -> Result<Tree> {
        build_tree(&tokenizer(), state, question, &s(candidates), max_len)
    }

    #[test]
    fn lays_out_one_prefix_and_a_branch_per_candidate() {
        let t = tree("state", "done", &["yes", "no"], 64).unwrap();
        let want = |w: &[&str]| w.iter().map(|w| id(w)).collect::<Vec<_>>();
        assert_eq!(words(&t.ids[..t.prefix_len]), want(&["[STATE]", "state", "\n", "[QUESTION]", "done"]));
        // Each branch: \n [CANDIDATE] candidate, ending at its candidate.
        let branch = |c: usize| {
            let start = if c == 0 { t.prefix_len } else { t.cand_ends[c - 1] + 1 };
            words(&t.ids[start..=t.cand_ends[c]])
        };
        assert_eq!(branch(0), want(&["\n", "[CANDIDATE]", "yes"]));
        assert_eq!(branch(1), want(&["\n", "[CANDIDATE]", "no"]));
        assert_eq!(t.cand_ends[1], t.ids.len() - 1);
        // Segments and positions: every branch restarts at the prefix's end.
        assert!(t.seg[..t.prefix_len].iter().all(|&s| s == 0));
        for c in 0..2 {
            let start = if c == 0 { t.prefix_len } else { t.cand_ends[c - 1] + 1 };
            assert!(t.seg[start..=t.cand_ends[c]].iter().all(|&s| s == c as i32 + 1));
            assert_eq!(t.position_ids[start], t.prefix_len as i32);
            assert_eq!(t.position_ids[t.cand_ends[c]], (t.prefix_len + t.cand_ends[c] - start) as i32);
        }
        assert_eq!(t.ids.len(), t.seg.len());
        assert_eq!(t.ids.len(), t.position_ids.len());
    }

    #[test]
    fn the_state_tag_is_added_once_and_specials_in_text_are_kept() {
        let t = tree("[STATE] state", "done", &["yes", "no"], 64).unwrap();
        assert_eq!(words(&t.ids)[..2], [id("[STATE]"), id("state")]);
        assert_eq!(t.ids.iter().filter(|&&i| i == id("[STATE]")).count(), 1);
        // AgentJev encodes text with add_special_tokens=False, which still
        // matches added tokens in the text: an <|endoftext|> in the state is
        // that token, as in AgentJev's own service.
        let t = tree("a <|endoftext|> b", "done", &["yes", "no"], 64).unwrap();
        assert!(t.ids.contains(&id("<|endoftext|>")));
    }

    #[test]
    fn nothing_is_truncated_an_over_long_tree_is_refused() {
        // [STATE] ␣ state \n [QUESTION] ␣ done = 7; each branch \n [CANDIDATE] ␣ x = 4.
        let exact = tree("state", "done", &["yes", "no"], 15).unwrap();
        assert_eq!(exact.ids.len(), 15);
        let e = tree("state", "done", &["yes", "no"], 14).unwrap_err();
        assert!(e.to_string().contains("takes 15 tokens; the model's input holds 14"), "{e}");
        let e = tree("state", "done", &["yes", "no"], 6).unwrap_err();
        assert!(e.to_string().contains("input and instructions takes 7 tokens"), "{e}");
        // Empty text, by Python's whitespace (U+001C counts).
        for (state, question) in [("", "done"), (" \u{1c}\n", "done"), ("state", "\t")] {
            let e = tree(state, question, &["yes", "no"], 64).unwrap_err();
            assert!(e.to_string().contains("must be nonempty text"), "{e}");
        }
        // The byte cap, before tokenizing.
        let e = tree(&"a".repeat(16 * 15 + 1), "done", &["yes", "no"], 15).unwrap_err();
        assert!(e.to_string().contains("bytes"), "{e}");
    }
}
