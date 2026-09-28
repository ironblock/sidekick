//! Post-processing applied to every reply, in one place so that the
//! complete and streaming paths can't drift apart:
//!
//! 1. Strip a leading `Assistant:` speaker label. Cold multi-turn replays
//!    present history as a "User:/Assistant:" transcript and the model
//!    sometimes mimics the format (seen on real hardware); a leading speaker
//!    label is never part of a wanted reply.
//! 2. Apply stop sequences (OpenAI semantics: the reply ends before the
//!    earliest occurrence of any stop sequence, which is not included).
//! 3. Never end on U+FFFD: a multi-byte character cut by truncation.
//!
//! [`shape`] applies them to a complete reply; [`StreamShaper`] applies the
//! same rules to a stream of cumulative snapshots, emitting only text that
//! no later snapshot can change.

/// Strip one leading `assistant` speaker label (ASCII case-insensitive,
/// optional spaces before the colon), at the start of the reply only.
pub fn strip_assistant_label(text: &str) -> &str {
    const LABEL: &str = "assistant";
    // `get`, not slicing, so a multi-byte char spanning the boundary can't panic.
    if let Some(prefix) = text.get(..LABEL.len()) {
        if prefix.eq_ignore_ascii_case(LABEL) {
            let rest = text[LABEL.len()..].trim_start_matches(' ');
            if let Some(after) = rest.strip_prefix(':') {
                return after.trim_start();
            }
        }
    }
    text
}

/// Byte offset where the earliest stop sequence begins, if any occurs.
pub fn find_stop(text: &str, stop: &[String]) -> Option<usize> {
    stop.iter()
        .filter(|s| !s.is_empty())
        .filter_map(|s| text.find(s.as_str()))
        .min()
}

/// A shaped reply.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Shaped {
    pub text: String,
    /// A stop sequence ended the reply early.
    pub stop_hit: bool,
}

/// Shape a complete reply.
pub fn shape(text: &str, stop: &[String]) -> Shaped {
    let text = strip_assistant_label(text);
    let (text, stop_hit) = match find_stop(text, stop) {
        Some(end) => (&text[..end], true),
        None => (text, false),
    };
    Shaped { text: text.trim_end_matches('\u{FFFD}').to_string(), stop_hit }
}

/// Where a reply's content starts, as far as the leading-label rule can
/// tell from a (possibly partial) text.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LabelDecision {
    /// Could still turn out to start with a speaker label: hold everything.
    Undecided,
    /// Content starts at this byte offset (0 when there is no label).
    ContentAt(usize),
}

fn label_decision(text: &str) -> LabelDecision {
    const LABEL: &str = "assistant";
    let Some(prefix) = text.get(..LABEL.len()) else {
        // Shorter than the label (or a multi-byte char straddles its end):
        // undecided only while it is still a prefix of the label.
        let could_be_label = text.len() < LABEL.len()
            && LABEL.as_bytes()[..text.len()].eq_ignore_ascii_case(text.as_bytes());
        return if could_be_label {
            LabelDecision::Undecided
        } else {
            LabelDecision::ContentAt(0)
        };
    };
    if !prefix.eq_ignore_ascii_case(LABEL) {
        return LabelDecision::ContentAt(0);
    }
    let rest = text[LABEL.len()..].trim_start_matches(' ');
    if rest.is_empty() {
        return LabelDecision::Undecided;
    }
    let Some(after) = rest.strip_prefix(':') else {
        return LabelDecision::ContentAt(0);
    };
    let content = after.trim_start();
    if content.is_empty() {
        // Whitespace after the colon may continue.
        return LabelDecision::Undecided;
    }
    LabelDecision::ContentAt(text.len() - content.len())
}

/// The largest char boundary of `text` that is <= `index`.
fn floor_char_boundary(text: &str, mut index: usize) -> usize {
    index = index.min(text.len());
    while !text.is_char_boundary(index) {
        index -= 1;
    }
    index
}

/// The final text of a stream disagrees with text already sent to the
/// client (a snapshot rewrote emitted text and never came back to it).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Diverged;

/// Incremental [`shape`]: feed cumulative snapshots, get deltas that are
/// safe to send. Holds back text a later snapshot could still change: the
/// whole reply while a leading label is undecided, the last
/// `longest stop − 1` bytes (a stop sequence may be arriving), and trailing
/// U+FFFD. A snapshot that doesn't extend what was already sent is skipped
/// (emission resumes if a later snapshot agrees again).
#[derive(Debug)]
pub struct StreamShaper<'a> {
    stop: &'a [String],
    holdback: usize,
    emitted: String,
    stop_hit: bool,
}

impl<'a> StreamShaper<'a> {
    pub fn new(stop: &'a [String]) -> Self {
        let longest = stop.iter().map(String::len).max().unwrap_or(0);
        Self { stop, holdback: longest.saturating_sub(1), emitted: String::new(), stop_hit: false }
    }

    /// Text sent to the client so far.
    pub fn emitted(&self) -> &str {
        &self.emitted
    }

    /// A stop sequence ended the reply; generation should stop.
    pub fn stop_hit(&self) -> bool {
        self.stop_hit
    }

    /// Feed the cumulative text of one snapshot; returns the new text to
    /// send (possibly empty).
    pub fn push(&mut self, snapshot: &str) -> String {
        if self.stop_hit {
            return String::new();
        }
        let LabelDecision::ContentAt(start) = label_decision(snapshot) else {
            return String::new();
        };
        let content = &snapshot[start..];
        if !content.starts_with(self.emitted.as_str()) {
            return String::new();
        }
        let end = match find_stop(content, self.stop) {
            Some(stop_at) => {
                self.stop_hit = true;
                stop_at
            }
            None => {
                let safe = floor_char_boundary(content, content.len().saturating_sub(self.holdback));
                content[..safe].trim_end_matches('\u{FFFD}').len()
            }
        };
        self.advance(content, end)
    }

    /// The stream completed normally with `final_text`: returns the
    /// remaining text to send, or [`Diverged`] if the final reply doesn't
    /// extend what was already sent.
    pub fn finish(&mut self, final_text: &str) -> Result<String, Diverged> {
        if self.stop_hit {
            return Ok(String::new());
        }
        let shaped = shape(final_text, self.stop);
        if !shaped.text.starts_with(self.emitted.as_str()) {
            return Err(Diverged);
        }
        self.stop_hit = shaped.stop_hit;
        Ok(self.advance(&shaped.text, shaped.text.len()))
    }

    fn advance(&mut self, content: &str, end: usize) -> String {
        if end <= self.emitted.len() {
            return String::new();
        }
        let delta = content[self.emitted.len()..end].to_string();
        self.emitted.push_str(&delta);
        delta
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stops(list: &[&str]) -> Vec<String> {
        list.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn assistant_label_variants() {
        for s in ["Assistant: x", "assistant: x", "ASSISTANT: x", "Assistant : x", "assistant  :  x"] {
            assert_eq!(strip_assistant_label(s), "x", "should strip: {s:?}");
        }
        for s in ["Assistants: x", "The Assistant: x", "Assistant x", "", "助手: x"] {
            assert_eq!(strip_assistant_label(s), s, "should not strip: {s:?}");
        }
    }

    #[test]
    fn earliest_stop_wins_and_is_excluded() {
        let s = shape("1, 2, 3, 4, 5, 6", &stops(&["5", "3"]));
        assert_eq!(s, Shaped { text: "1, 2, ".into(), stop_hit: true });
    }

    #[test]
    fn no_stop_match_keeps_text() {
        let s = shape("hello", &stops(&["zzz"]));
        assert_eq!(s, Shaped { text: "hello".into(), stop_hit: false });
        assert_eq!(shape("hello", &[]).text, "hello");
    }

    #[test]
    fn empty_stop_sequences_are_ignored() {
        assert!(!shape("hello", &stops(&[""])).stop_hit);
    }

    #[test]
    fn stop_applies_after_the_label_is_stripped() {
        // "Assistant" must not match a stop of "Assistant" once stripped...
        let s = shape("Assistant: fine. Assistant: again", &stops(&["Assistant"]));
        assert_eq!(s.text, "fine. ");
        // ...and a stop at the very start yields an empty reply.
        assert_eq!(shape("STOP now", &stops(&["STOP"])).text, "");
    }

    #[test]
    fn multibyte_text_and_stops() {
        let s = shape("東京は日本の首都です。大阪", &stops(&["。"]));
        assert_eq!(s.text, "東京は日本の首都です");
        let s = shape("ok 👍🏽 then", &stops(&["🏽"]));
        assert_eq!(s.text, "ok 👍");
    }

    #[test]
    fn trailing_replacement_chars_are_dropped() {
        assert_eq!(shape("moon\u{FFFD}", &[]).text, "moon");
        assert_eq!(shape("a\u{FFFD}b", &[]).text, "a\u{FFFD}b");
    }

    /// Feed snapshots, then finish with the last one; returns (deltas, final
    /// emitted text, stop hit).
    fn stream(snapshots: &[&str], stop: &[&str]) -> (Vec<String>, String, bool) {
        let stop = stops(stop);
        let mut shaper = StreamShaper::new(&stop);
        let mut deltas = Vec::new();
        for snap in snapshots {
            let d = shaper.push(snap);
            if !d.is_empty() {
                deltas.push(d);
            }
            if shaper.stop_hit() {
                break;
            }
        }
        let tail = shaper.finish(snapshots.last().unwrap()).unwrap();
        if !tail.is_empty() {
            deltas.push(tail);
        }
        (deltas, shaper.emitted().to_string(), shaper.stop_hit())
    }

    /// Streaming and whole-reply shaping must agree on the final text.
    fn assert_agrees(snapshots: &[&str], stop: &[&str]) {
        let (deltas, emitted, _) = stream(snapshots, stop);
        let whole = shape(snapshots.last().unwrap(), &stops(stop));
        assert_eq!(deltas.concat(), whole.text, "deltas vs shape() for {snapshots:?}");
        assert_eq!(emitted, whole.text);
    }

    #[test]
    fn plain_stream_emits_incrementally() {
        let (deltas, text, stop) = stream(&["The", "The moon", "The moon orbits."], &[]);
        assert_eq!(deltas, vec!["The", " moon", " orbits."]);
        assert_eq!((text.as_str(), stop), ("The moon orbits.", false));
    }

    #[test]
    fn label_is_held_until_decided_then_stripped() {
        let (deltas, text, _) =
            stream(&["Assi", "Assistant", "Assistant: ", "Assistant: Paris", "Assistant: Paris is big."], &[]);
        assert_eq!(text, "Paris is big.");
        assert_eq!(deltas.first().map(String::as_str), Some("Paris"), "nothing leaked before");
        // A reply that merely starts like the label is released as soon as
        // it diverges from it.
        let (deltas, _, _) = stream(&["As", "As I said", "As I said, yes."], &[]);
        assert_eq!(deltas, vec!["As I said", ", yes."]);
    }

    #[test]
    fn stop_spanning_snapshots_is_caught_and_excluded() {
        let (deltas, text, stop) = stream(&["1, 2, 3, E", "1, 2, 3, EN", "1, 2, 3, END 4"], &["END"]);
        assert_eq!(text, "1, 2, 3, ");
        assert!(stop);
        assert!(deltas.iter().all(|d| !d.contains('E')), "partial stop text never sent: {deltas:?}");
    }

    #[test]
    fn stop_hit_ends_emission() {
        let stop = stops(&["5"]);
        let mut shaper = StreamShaper::new(&stop);
        assert_eq!(shaper.push("1, 2, 3, 4, 5"), "1, 2, 3, 4, ");
        assert!(shaper.stop_hit());
        assert_eq!(shaper.push("1, 2, 3, 4, 5, 6"), "");
        assert_eq!(shaper.finish("1, 2, 3, 4, 5, 6, 7").unwrap(), "");
    }

    #[test]
    fn trailing_replacement_char_is_held_until_repaired() {
        let (deltas, text, _) = stream(&["moon\u{FFFD}", "moon’s glow"], &[]);
        assert_eq!(deltas, vec!["moon", "’s glow"]);
        assert_eq!(text, "moon’s glow");
    }

    #[test]
    fn multibyte_text_with_holdback() {
        assert_agrees(&["東京", "東京は日本", "東京は日本の首都。大阪"], &["。"]);
        assert_agrees(&["ok 👍", "ok 👍🏽 then", "ok 👍🏽 then done"], &["done"]);
        assert_agrees(&["héllo wörld", "héllo wörld, ça va"], &["ça va?"]);
    }

    #[test]
    fn a_rewrite_is_skipped_and_emission_resumes_when_it_agrees_again() {
        let mut shaper = StreamShaper::new(&[]);
        assert_eq!(shaper.push("The cat"), "The cat");
        assert_eq!(shaper.push("The dog sat"), "", "rewrote emitted text: skipped");
        assert_eq!(shaper.push("The cat sat"), " sat", "agrees again: resumes");
        assert_eq!(shaper.finish("The cat sat.").unwrap(), ".");
    }

    #[test]
    fn a_final_text_that_contradicts_emitted_text_is_divergence() {
        let mut shaper = StreamShaper::new(&[]);
        shaper.push("The cat");
        assert_eq!(shaper.finish("The dog."), Err(Diverged));
    }

    #[test]
    fn finish_flushes_holdback() {
        let stop = stops(&["STOP"]);
        let mut shaper = StreamShaper::new(&stop);
        assert_eq!(shaper.push("abcdef"), "abc", "last 3 bytes held for a possible STOP");
        assert_eq!(shaper.finish("abcdef").unwrap(), "def");
    }

    #[test]
    fn streaming_agrees_with_whole_reply_shaping() {
        assert_agrees(&["Assistant:", "Assistant: hi", "Assistant: hi there"], &[]);
        assert_agrees(&["a", "ab", "abc"], &["zz"]);
        assert_agrees(&["x"], &[]);
        assert_agrees(&["Assistant: STOP now"], &["STOP"]);
    }
}
