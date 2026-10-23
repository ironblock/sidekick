//! `--include` / `--exclude` matching, as `hf download` does it.
//!
//! huggingface_hub filters a repo's paths with Python's `fnmatchcase`: `*`
//! matches any run of characters, `/` included (so `dir/*` matches every
//! file under `dir/`, however deep), `?` one character, and `[seq]` /
//! `[!seq]` a character class; a pattern ending in `/` gets a `*` appended.
//! The fallback downloader selects files with this same matcher, so both
//! paths fetch exactly the same files for the same patterns.

/// Whether `path` is selected: it matches an include pattern (or there are
/// none), and no exclude pattern.
pub fn selected(path: &str, include: &[String], exclude: &[String]) -> bool {
    let matches = |p: &String| fnmatch(path, &with_dir_wildcard(p));
    (include.is_empty() || include.iter().any(matches)) && !exclude.iter().any(matches)
}

/// huggingface_hub's `_add_wildcard_to_directories`.
fn with_dir_wildcard(pattern: &str) -> String {
    if pattern.ends_with('/') {
        format!("{pattern}*")
    } else {
        pattern.to_string()
    }
}

/// Python's `fnmatch.fnmatchcase(name, pattern)`.
pub fn fnmatch(name: &str, pattern: &str) -> bool {
    let n: Vec<char> = name.chars().collect();
    let p: Vec<char> = pattern.chars().collect();
    let (mut ni, mut pi) = (0, 0);
    // After a `*`: where the pattern resumes, and how much of the name the
    // star has taken so far.
    let mut star: Option<(usize, usize)> = None;
    while ni < n.len() {
        let step = match p.get(pi) {
            Some('*') => {
                star = Some((pi + 1, ni));
                pi += 1;
                continue;
            }
            Some('?') => Some(pi + 1),
            Some('[') => match class(&p, pi) {
                Some((set, next)) => set.matches(n[ni]).then_some(next),
                // No closing bracket: a literal `[`.
                None => (n[ni] == '[').then_some(pi + 1),
            },
            Some(&c) => (c == n[ni]).then_some(pi + 1),
            None => None,
        };
        match (step, star) {
            (Some(next), _) => {
                ni += 1;
                pi = next;
            }
            (None, Some((resume, taken))) => {
                pi = resume;
                ni = taken + 1;
                star = Some((resume, taken + 1));
            }
            (None, None) => return false,
        }
    }
    p[pi..].iter().all(|&c| c == '*')
}

/// A `[...]` class.
struct Class {
    negated: bool,
    ranges: Vec<(char, char)>,
}

impl Class {
    fn matches(&self, c: char) -> bool {
        self.ranges.iter().any(|&(lo, hi)| lo <= c && c <= hi) != self.negated
    }
}

/// The class starting at `p[start] == '['`, and the index after its `]`;
/// `None` when it has no closing bracket. As in `fnmatch.translate`, a `]`
/// first in the class (after any `!`) is a member, not the end.
fn class(p: &[char], start: usize) -> Option<(Class, usize)> {
    let mut i = start + 1;
    let negated = p.get(i) == Some(&'!');
    if negated {
        i += 1;
    }
    let body_start = i;
    if p.get(i) == Some(&']') {
        i += 1;
    }
    while i < p.len() && p[i] != ']' {
        i += 1;
    }
    if i >= p.len() {
        return None;
    }
    let body = &p[body_start..i];
    let mut ranges = Vec::new();
    let mut k = 0;
    while k < body.len() {
        if k + 2 < body.len() && body[k + 1] == '-' {
            if body[k] <= body[k + 2] {
                ranges.push((body[k], body[k + 2]));
            }
            k += 3;
        } else {
            ranges.push((body[k], body[k]));
            k += 1;
        }
    }
    Some((Class { negated, ranges }, i + 1))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_like_fnmatchcase() {
        // Pairs checked against Python's fnmatch.fnmatchcase.
        let cases = [
            ("manifest.toml", "manifest.toml", true),
            ("manifest.toml", "Manifest.toml", false),
            ("int8/manifest.toml", "manifest.toml", false),
            ("int8/manifest.toml", "*manifest.toml", true),
            ("int8/manifest.toml", "*/manifest.toml", true),
            ("a/b/manifest.toml", "*/manifest.toml", true),
            ("model_128.mlmodelc/weights/weight.bin", "model_128.mlmodelc/*", true),
            ("model_1280.mlmodelc/model.mil", "model_128.mlmodelc/*", false),
            ("model_128.mlmodelc", "model_128.mlmodelc/*", false),
            ("model.onnx", "model.onnx?data", false),
            ("model.onnx_data", "model.onnx?data", true),
            ("x", "*", true),
            ("", "*", true),
            ("", "?", false),
            ("abc", "a*c", true),
            ("abbbc", "a*b*c", true),
            ("abd", "a*c", false),
            ("a1", "a[0-9]", true),
            ("ax", "a[0-9]", false),
            ("ax", "a[!0-9]", true),
            ("a]", "a[]]", true),
            ("a!", "a[!]", false),
            ("a[", "a[", true),
            ("a[b", "a[b", true),
            ("a-", "a[a-]", true),
        ];
        for (name, pattern, want) in cases {
            assert_eq!(fnmatch(name, pattern), want, "fnmatchcase({name:?}, {pattern:?})");
        }
    }

    #[test]
    fn a_trailing_slash_selects_a_directory() {
        let inc = vec!["model_128.mlmodelc/".to_string()];
        assert!(selected("model_128.mlmodelc/weights/weight.bin", &inc, &[]));
        assert!(!selected("model_256.mlmodelc/weights/weight.bin", &inc, &[]));
    }

    #[test]
    fn exclude_wins_and_no_include_means_everything() {
        let exc = vec!["*.md".to_string()];
        assert!(selected("model.onnx", &[], &exc));
        assert!(!selected("README.md", &[], &exc));
        assert!(!selected("README.md", &["*".to_string()], &exc));
    }
}
