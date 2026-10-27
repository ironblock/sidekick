//! What transformers adds to jinja2 for chat templates, for minijinja:
//! `raise_exception`, `strftime_now`, a `tojson` filter that writes Python's
//! `json.dumps` (no HTML escaping, `ensure_ascii` off by default), and the
//! `{% generation %}` block, which marks an assistant's text for training
//! masks and renders its body unchanged.

use minijinja::value::{Kwargs, Rest};
use minijinja::{Environment, Error, ErrorKind, Value};
use regex::Regex;
use std::borrow::Cow;
use std::sync::OnceLock;

/// Add transformers' globals and filters to `env`.
pub fn install(env: &mut Environment<'static>) {
    env.add_function("raise_exception", |message: String| -> Result<Value, Error> {
        Err(Error::new(ErrorKind::InvalidOperation, message))
    });
    env.add_function("strftime_now", |format: String| -> Result<Value, Error> {
        strftime(&format, now_local()).map(Value::from)
    });
    env.add_filter("tojson", tojson);
}

/// `{% generation %}` … `{% endgeneration %}` as a block that renders its
/// body (`{% if true %}` … `{% endif %}`), whitespace control kept.
pub fn strip_generation(template: &str) -> Cow<'_, str> {
    static OPEN: OnceLock<Regex> = OnceLock::new();
    static CLOSE: OnceLock<Regex> = OnceLock::new();
    let open = OPEN.get_or_init(|| Regex::new(r"\{%(-|\+)?\s*generation\s*(-|\+)?%\}").expect("compiles"));
    let close = CLOSE.get_or_init(|| Regex::new(r"\{%(-|\+)?\s*endgeneration\s*(-|\+)?%\}").expect("compiles"));
    match open.replace_all(template, "{%${1} if true ${2}%}") {
        Cow::Borrowed(_) => Cow::Borrowed(template),
        Cow::Owned(opened) => Cow::Owned(close.replace_all(&opened, "{%${1} endif ${2}%}").into_owned()),
    }
}

/// transformers' `tojson(x, ensure_ascii=False, indent=None,
/// separators=None, sort_keys=False)`: `json.dumps` with those arguments.
fn tojson(value: Value, args: Rest<Value>, kwargs: Kwargs) -> Result<Value, Error> {
    let positional = |i: usize| args.0.get(i).cloned();
    let pick = |i: usize, name: &str| -> Result<Option<Value>, Error> {
        match kwargs.get::<Option<Value>>(name)? {
            Some(v) => Ok(Some(v)),
            None => Ok(positional(i)),
        }
    };
    let ensure_ascii = pick(0, "ensure_ascii")?.is_some_and(|v| v.is_true());
    let indent = match pick(1, "indent")? {
        None => None,
        Some(v) if v.is_none() || v.is_undefined() => None,
        Some(v) => match v.as_str() {
            Some(s) => Some(s.to_string()),
            None => {
                let n = i64::try_from(v.clone()).map_err(|_| bad(format!("tojson: indent {v} isn't a number or string")))?;
                Some(" ".repeat(n.max(0) as usize))
            }
        },
    };
    let separators = match pick(2, "separators")? {
        None => None,
        Some(v) if v.is_none() || v.is_undefined() => None,
        Some(v) => {
            let item = v.get_item(&Value::from(0))?;
            let key = v.get_item(&Value::from(1))?;
            match (item.as_str(), key.as_str()) {
                (Some(i), Some(k)) => Some((i.to_string(), k.to_string())),
                _ => return Err(bad("tojson: separators must be two strings".into())),
            }
        }
    };
    let sort_keys = pick(3, "sort_keys")?.is_some_and(|v| v.is_true());
    if args.0.len() > 4 {
        return Err(bad("tojson takes at most four arguments".into()));
    }
    kwargs.assert_all_used()?;
    let json = serde_json::to_value(&value).map_err(|e| bad(format!("tojson: {e}")))?;
    let (item_sep, key_sep) = separators.unwrap_or_else(|| match indent {
        Some(_) => (",".into(), ": ".into()),
        None => (", ".into(), ": ".into()),
    });
    let mut out = String::new();
    let style = Style { ensure_ascii, indent, item_sep, key_sep, sort_keys };
    dump(&json, &style, 0, &mut out);
    Ok(Value::from_safe_string(out))
}

fn bad(message: String) -> Error {
    Error::new(ErrorKind::InvalidOperation, message)
}

struct Style {
    ensure_ascii: bool,
    indent: Option<String>,
    item_sep: String,
    key_sep: String,
    sort_keys: bool,
}

/// Python's `json.dumps` layout.
fn dump(v: &serde_json::Value, s: &Style, level: usize, out: &mut String) {
    use serde_json::Value as J;
    let newline = |out: &mut String, level: usize| {
        if let Some(indent) = &s.indent {
            out.push('\n');
            for _ in 0..level {
                out.push_str(indent);
            }
        }
    };
    match v {
        J::Null => out.push_str("null"),
        J::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        J::Number(n) => match n.as_f64() {
            Some(f) if n.is_f64() && f.is_finite() && f.fract() == 0.0 && f.abs() < 1e16 => out.push_str(&format!("{f:.1}")),
            _ => out.push_str(&n.to_string()),
        },
        J::String(text) => string(text, s.ensure_ascii, out),
        J::Array(items) if items.is_empty() => out.push_str("[]"),
        J::Array(items) => {
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push_str(&s.item_sep);
                }
                newline(out, level + 1);
                dump(item, s, level + 1, out);
            }
            newline(out, level);
            out.push(']');
        }
        J::Object(map) if map.is_empty() => out.push_str("{}"),
        J::Object(map) => {
            let mut entries: Vec<(&String, &J)> = map.iter().collect();
            if s.sort_keys {
                entries.sort_by(|a, b| a.0.cmp(b.0));
            }
            out.push('{');
            for (i, (k, item)) in entries.into_iter().enumerate() {
                if i > 0 {
                    out.push_str(&s.item_sep);
                }
                newline(out, level + 1);
                string(k, s.ensure_ascii, out);
                out.push_str(&s.key_sep);
                dump(item, s, level + 1, out);
            }
            newline(out, level);
            out.push('}');
        }
    }
}

/// A JSON string as Python writes it.
fn string(text: &str, ensure_ascii: bool, out: &mut String) {
    out.push('"');
    for c in text.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c if ensure_ascii && (c as u32) > 0x7e => {
                let mut units = [0u16; 2];
                for unit in c.encode_utf16(&mut units) {
                    out.push_str(&format!("\\u{unit:04x}"));
                }
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

/// A moment's calendar fields, as `struct tm` holds them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Tm {
    pub year: i64,
    /// 1–12.
    pub month: u32,
    pub day: u32,
    pub hour: u32,
    pub minute: u32,
    pub second: u32,
    /// 0 = Sunday.
    pub weekday: u32,
    /// 1–366.
    pub yearday: u32,
}

impl Tm {
    /// The UTC calendar fields of `secs` since the Unix epoch.
    pub fn utc(secs: i64) -> Self {
        let days = secs.div_euclid(86_400);
        let rem = secs.rem_euclid(86_400) as u32;
        // Howard Hinnant's civil_from_days.
        let z = days + 719_468;
        let era = z.div_euclid(146_097);
        let doe = z.rem_euclid(146_097);
        let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
        let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
        let mp = (5 * doy + 2) / 153;
        let day = (doy - (153 * mp + 2) / 5 + 1) as u32;
        let month = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
        let year = yoe + era * 400 + i64::from(month <= 2);
        let leap = (year % 4 == 0 && year % 100 != 0) || year % 400 == 0;
        const BEFORE: [u32; 12] = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334];
        let yearday = BEFORE[month as usize - 1] + day + u32::from(leap && month > 2);
        Self {
            year,
            month,
            day,
            hour: rem / 3600,
            minute: rem / 60 % 60,
            second: rem % 60,
            weekday: (days + 4).rem_euclid(7) as u32,
            yearday,
        }
    }
}

/// Now, in the local time zone where the platform says what it is (as
/// transformers' `datetime.now()`), else in UTC.
fn now_local() -> Tm {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    #[cfg(unix)]
    {
        let t: libc::time_t = secs as libc::time_t;
        let mut tm: libc::tm = unsafe { std::mem::zeroed() };
        // SAFETY: both pointers are valid for the call; localtime_r is
        // the thread-safe form.
        if !unsafe { libc::localtime_r(&t, &mut tm) }.is_null() {
            return Tm::utc(secs + tm.tm_gmtoff as i64);
        }
    }
    Tm::utc(secs)
}

/// C's `strftime` for the conversions chat templates use.
pub fn strftime(format: &str, t: Tm) -> Result<String, Error> {
    const DAYS: [&str; 7] = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"];
    const MONTHS: [&str; 12] =
        ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"];
    let mut out = String::new();
    let mut chars = format.chars();
    while let Some(c) = chars.next() {
        if c != '%' {
            out.push(c);
            continue;
        }
        let month = MONTHS[t.month as usize - 1];
        let day = DAYS[t.weekday as usize];
        match chars.next() {
            Some('Y') => out.push_str(&t.year.to_string()),
            Some('y') => out.push_str(&format!("{:02}", t.year.rem_euclid(100))),
            Some('m') => out.push_str(&format!("{:02}", t.month)),
            Some('d') => out.push_str(&format!("{:02}", t.day)),
            Some('e') => out.push_str(&format!("{:2}", t.day)),
            Some('j') => out.push_str(&format!("{:03}", t.yearday)),
            Some('H') => out.push_str(&format!("{:02}", t.hour)),
            Some('I') => out.push_str(&format!("{:02}", (t.hour + 11) % 12 + 1)),
            Some('p') => out.push_str(if t.hour < 12 { "AM" } else { "PM" }),
            Some('M') => out.push_str(&format!("{:02}", t.minute)),
            Some('S') => out.push_str(&format!("{:02}", t.second)),
            Some('b') | Some('h') => out.push_str(&month[..3]),
            Some('B') => out.push_str(month),
            Some('a') => out.push_str(&day[..3]),
            Some('A') => out.push_str(day),
            Some('F') => out.push_str(&format!("{}-{:02}-{:02}", t.year, t.month, t.day)),
            Some('%') => out.push('%'),
            other => {
                return Err(bad(format!(
                    "strftime_now: `%{}` isn't supported",
                    other.map(String::from).unwrap_or_default()
                )))
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn render(template: &str, ctx: Value) -> String {
        let mut env = Environment::new();
        install(&mut env);
        env.add_template_owned("t", strip_generation(template).into_owned()).unwrap();
        env.get_template("t").unwrap().render(ctx).unwrap()
    }

    #[test]
    fn tojson_writes_pythons_json_dumps() {
        let ctx = Value::from(minijinja::value::Serde(serde_json::json!({
            "x": {"b": [1, 2.0, "é<&>\"\n"], "a": null, "e": {}, "f": []}
        })));
        // json.dumps(x, ensure_ascii=False): ", " and ": ", no HTML escaping.
        assert_eq!(render("{{ x | tojson }}", ctx.clone()), r#"{"b": [1, 2.0, "é<&>\"\n"], "a": null, "e": {}, "f": []}"#);
        assert_eq!(
            render("{{ x | tojson(ensure_ascii=True, sort_keys=True) }}", ctx.clone()),
            r#"{"a": null, "b": [1, 2.0, "\u00e9<&>\"\n"], "e": {}, "f": []}"#
        );
        assert_eq!(
            render("{{ x | tojson(indent=2) }}", ctx.clone()),
            "{\n  \"b\": [\n    1,\n    2.0,\n    \"é<&>\\\"\\n\"\n  ],\n  \"a\": null,\n  \"e\": {},\n  \"f\": []\n}"
        );
        assert_eq!(render("{{ x.b | tojson(separators=(',', ':')) }}", ctx), r#"[1,2.0,"é<&>\"\n"]"#);
        let emoji = Value::from(minijinja::value::Serde(serde_json::json!({"s": "🚀"})));
        assert_eq!(render("{{ s | tojson(true) }}", emoji), r#""\ud83d\ude80""#);
    }

    #[test]
    fn generation_blocks_render_their_body() {
        let t = "{%- for m in messages %}<{{ m }}>{%- generation %} [{{ m }}] {% endgeneration -%}|{% endfor %}";
        let ctx = Value::from(minijinja::value::Serde(serde_json::json!({"messages": ["a", "b"]})));
        assert_eq!(render(t, ctx), "<a> [a] |<b> [b] |");
        assert!(matches!(strip_generation("{{ x }}"), Cow::Borrowed(_)));
    }

    #[test]
    fn strftime_formats_a_known_moment() {
        // 2024-02-29 13:05:09 UTC, a Thursday.
        let t = Tm::utc(1_709_211_909);
        assert_eq!(t, Tm { year: 2024, month: 2, day: 29, hour: 13, minute: 5, second: 9, weekday: 4, yearday: 60 });
        assert_eq!(strftime("%d %b %Y", t).unwrap(), "29 Feb 2024");
        assert_eq!(strftime("%A, %B %e %Y %I:%M:%S %p (%j) %F %y %%", t).unwrap(), "Thursday, February 29 2024 01:05:09 PM (060) 2024-02-29 24 %");
        assert_eq!(Tm::utc(0), Tm { year: 1970, month: 1, day: 1, hour: 0, minute: 0, second: 0, weekday: 4, yearday: 1 });
        assert!(strftime("%Q", t).is_err());
        // The global renders today's date.
        let today = render("{{ strftime_now('%Y') }}", Value::from(()));
        assert!(today.parse::<i32>().is_ok_and(|y| y >= 2024), "{today}");
    }
}
