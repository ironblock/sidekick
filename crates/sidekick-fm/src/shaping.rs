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
}
