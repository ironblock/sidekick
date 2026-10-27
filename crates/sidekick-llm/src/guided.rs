//! Constrained decoding for `response_format: json_schema` on an open model
//! (docs/design/open-llm.md, "Constrained decoding on the open path"):
//! [llguidance](https://github.com/guidance-ai/llguidance) computes, each
//! step, the tokens the schema's grammar allows, and the sampler sets every
//! other logit to −∞.
//!
//! - **One grammar layer.** The schema reaches this module as the chat
//!   route's validated representation ([`JsonSchema`]), never as the
//!   request's own JSON, so an open model accepts exactly the schemas
//!   `apple-fm` does. [`grammar_schema`] writes it back out as plain JSON
//!   Schema for llguidance: properties in the order sent, objects closed,
//!   nullable properties as an `anyOf` with `null`, descriptions dropped
//!   (the model sees only the prompt).
//! - **One line of JSON, spelled as models write it.** The grammar allows
//!   no whitespace between tokens except the separators `, ` and `: `
//!   (Python's `json.dumps` spelling), so a guided reply is one line with
//!   its keys in schema order, and identical requests at temperature 0
//!   give identical bytes. Those separators are the ones a model writes
//!   unprompted: compact `,` and `:` split its usual tokens (`": "`), and
//!   MiniCPM5-2B then wrote `{"name":", "}` for a contact it extracts
//!   correctly with spaces. A model can't spend its budget on blank lines,
//!   and more of the reply is forced than with free whitespace.
//! - **One tokenizer bridge per model** ([`Guidance`]): llguidance's token
//!   trie, built from the model's `tokenizer.json` on its first guided
//!   request, with the model's stop tokens as end-of-sequence tokens.
//!   Compiled grammars are cached by the schema written out, so a schema a
//!   client repeats compiles once.
//! - **Forced tokens.** Where the grammar allows exactly one continuation
//!   (a property name, a closing brace), [`Guide::forced`] returns it, and
//!   the loop feeds those tokens in the same call as the sampled one,
//!   filling slots the call would otherwise pad.

use llguidance::api::TopLevelGrammar;
use llguidance::toktrie::{InferenceCapabilities, SimpleVob, TokEnv};
use llguidance::{Matcher, ParserFactory};
use serde_json::{json, Map, Value};
use sidekick_core::schema::{Kind, Node, Property};
use sidekick_core::{Error, JsonSchema, Result};
use std::collections::VecDeque;
use std::sync::{Arc, Mutex, OnceLock, PoisonError};
use std::time::{Duration, Instant};
use tokenizers::Tokenizer;

/// The options every guided grammar is compiled with: no whitespace
/// between JSON tokens but the separators', which are spelled as a model
/// writes them.
fn guidance_options() -> Value {
    json!({"whitespace_flexible": false, "item_separator": ", ", "key_separator": ": "})
}

/// `schema` as the JSON Schema llguidance compiles, with the options
/// (`x-guidance`) at its root.
pub fn grammar_schema(schema: &JsonSchema) -> Value {
    let mut root = node_schema(schema.root());
    root.as_object_mut().expect("a node is an object").insert("x-guidance".into(), guidance_options());
    root
}

fn node_schema(node: &Node) -> Value {
    match &node.kind {
        Kind::Object { properties } => {
            let mut props = Map::new();
            for p in properties {
                props.insert(p.name.clone(), property_schema(p));
            }
            let required: Vec<&str> = properties.iter().filter(|p| p.required).map(|p| p.name.as_str()).collect();
            json!({"type": "object", "properties": props, "required": required, "additionalProperties": false})
        }
        Kind::Array { items, min_items, max_items } => {
            let mut m = Map::new();
            m.insert("type".into(), json!("array"));
            m.insert("items".into(), node_schema(items));
            if let Some(n) = min_items {
                m.insert("minItems".into(), json!(n));
            }
            if let Some(n) = max_items {
                m.insert("maxItems".into(), json!(n));
            }
            Value::Object(m)
        }
        Kind::String => json!({"type": "string"}),
        Kind::Enum(choices) => json!({"type": "string", "enum": choices}),
        Kind::Integer => json!({"type": "integer"}),
        Kind::Number => json!({"type": "number"}),
        Kind::Boolean => json!({"type": "boolean"}),
    }
}

fn property_schema(p: &Property) -> Value {
    let schema = node_schema(&p.schema);
    if p.nullable {
        json!({"anyOf": [schema, {"type": "null"}]})
    } else {
        schema
    }
}

/// How many compiled grammars a model keeps.
const GRAMMAR_CACHE: usize = 32;

/// A model's bridge to llguidance: its tokenizer's token trie, built on
/// first use, and its compiled grammars.
pub struct Guidance {
    tokenizer: Tokenizer,
    stop_ids: Vec<u32>,
    vocab_size: usize,
    factory: OnceLock<std::result::Result<Arc<ParserFactory>, String>>,
    /// Compiled grammars, freshly started, by the schema written out; the
    /// most recently used last.
    grammars: Mutex<VecDeque<(String, Matcher)>>,
}

impl Guidance {
    /// For a model with this tokenizer, these stop tokens and a logits row
    /// `vocab_size` wide.
    pub fn new(tokenizer: Tokenizer, stop_ids: &[u32], vocab_size: usize) -> Self {
        Self {
            tokenizer,
            stop_ids: stop_ids.to_vec(),
            vocab_size,
            factory: OnceLock::new(),
            grammars: Mutex::new(VecDeque::new()),
        }
    }

    /// The token trie and parser factory, built on first use (a few
    /// hundred milliseconds for a 130k vocabulary).
    fn factory(&self) -> Result<Arc<ParserFactory>> {
        self.factory
            .get_or_init(|| {
                let started = Instant::now();
                let factory = build_factory(self.tokenizer.clone(), &self.stop_ids, self.vocab_size);
                if factory.is_ok() {
                    tracing::info!(build_ms = started.elapsed().as_millis() as u64, "guided decoding's token trie built");
                }
                factory.map(Arc::new)
            })
            .clone()
            .map_err(|e| Error::InvalidRequest(format!("this model can't guide its output: {e}")))
    }

    /// A guide for `schema`, ready for a request's first token. Blocking:
    /// the first call builds the trie, and a schema not seen recently is
    /// compiled.
    pub fn guide(&self, schema: &JsonSchema) -> Result<Guide> {
        let factory = self.factory()?;
        let grammar = grammar_schema(schema);
        let key = grammar.to_string();
        let mut cache = self.grammars.lock().unwrap_or_else(PoisonError::into_inner);
        if let Some(i) = cache.iter().position(|(k, _)| *k == key) {
            let entry = cache.remove(i).expect("found above");
            let matcher = entry.1.clone();
            cache.push_back(entry);
            return Ok(Guide::new(matcher, false));
        }
        drop(cache);
        let parser = factory
            .create_parser(TopLevelGrammar::from_json_schema(grammar))
            .map_err(|e| Error::GuidedGeneration(format!("llguidance can't compile it: {e}")))?;
        let matcher = Matcher::new(Ok(parser));
        if let Some(e) = matcher.get_error() {
            return Err(Error::GuidedGeneration(format!("llguidance can't compile it: {e}")));
        }
        let mut cache = self.grammars.lock().unwrap_or_else(PoisonError::into_inner);
        if cache.len() == GRAMMAR_CACHE {
            cache.pop_front();
        }
        cache.push_back((key, matcher.clone()));
        Ok(Guide::new(matcher, true))
    }
}

fn build_factory(tokenizer: Tokenizer, stop_ids: &[u32], vocab_size: usize) -> std::result::Result<ParserFactory, String> {
    let mut bytes = toktrie_hf_tokenizers::ByteTokenizer::from_tokenizer(tokenizer).map_err(|e| e.to_string())?;
    let vocab = bytes.tokrx_info().vocab_size as usize;
    if vocab > vocab_size {
        return Err(format!("its tokenizer has {vocab} tokens, past its logits' {vocab_size}"));
    }
    if stop_ids.is_empty() || stop_ids.iter().any(|&t| t as usize >= vocab) {
        return Err(format!("its stop tokens {stop_ids:?} aren't all in its tokenizer's {vocab} tokens"));
    }
    bytes.set_eos_tokens(stop_ids);
    let env: TokEnv = bytes.into_tok_env(Some(vocab_size)).map_err(|e| e.to_string())?;
    let caps = InferenceCapabilities { ff_tokens: true, ..Default::default() };
    let slices = llguidance::earley::SlicedBiasComputer::json_slices();
    let mut factory = ParserFactory::new(&env, caps, &slices).map_err(|e| e.to_string())?;
    factory.quiet();
    Ok(factory)
}

/// What a guide measured over one request.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct GuideStats {
    /// Whether the request compiled its grammar (a cache miss).
    pub compiled: bool,
    /// Masks computed, and their time.
    pub masks: u32,
    pub mask_total: Duration,
    pub mask_max: Duration,
    /// Tokens the grammar forced, fed without sampling.
    pub forced: u32,
}

/// One request's grammar state.
pub struct Guide {
    /// Boxed: a parser is large, and a job carries its guide by value.
    matcher: Box<Matcher>,
    stats: GuideStats,
}

impl Guide {
    fn new(matcher: Matcher, compiled: bool) -> Self {
        Self { matcher: Box::new(matcher), stats: GuideStats { compiled, ..Default::default() } }
    }

    pub fn stats(&self) -> GuideStats {
        self.stats
    }

    /// The tokens the grammar allows next, the model's stop tokens among
    /// them once the JSON is complete.
    pub fn mask(&mut self) -> Result<SimpleVob> {
        let started = Instant::now();
        let mask = self.matcher.compute_mask().map_err(|e| failed(&e.to_string()))?;
        let took = started.elapsed();
        self.stats.masks += 1;
        self.stats.mask_total += took;
        self.stats.mask_max = self.stats.mask_max.max(took);
        if mask.is_zero() {
            return Err(failed("the grammar allows no token of this model's vocabulary"));
        }
        Ok(mask)
    }

    /// Advance past a sampled token.
    pub fn consume(&mut self, token: u32) -> Result<()> {
        self.matcher.consume_token(token).map_err(|e| failed(&e.to_string()))
    }

    /// The tokens the grammar forces next, at most `limit`, consumed.
    pub fn forced(&mut self, limit: usize) -> Result<Vec<u32>> {
        if limit == 0 || self.matcher.is_stopped() {
            return Ok(Vec::new());
        }
        let mut forced = self.matcher.compute_ff_tokens();
        if let Some(e) = self.matcher.get_error() {
            return Err(failed(&e));
        }
        forced.truncate(limit);
        if !forced.is_empty() {
            self.matcher.consume_tokens(&forced).map_err(|e| failed(&e.to_string()))?;
            self.stats.forced += forced.len() as u32;
        }
        Ok(forced)
    }

    /// Whether the JSON is complete and nothing may follow it.
    pub fn done(&self) -> bool {
        self.matcher.is_stopped()
    }
}

fn failed(reason: &str) -> Error {
    Error::Inference(format!("guided decoding failed: {reason}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prompt::tests::byte_tokenizer;

    fn schema(v: Value) -> JsonSchema {
        JsonSchema::parse(&v).unwrap()
    }

    #[test]
    fn writes_the_representation_back_out_in_order_and_closed() {
        let s = schema(json!({
            "type": "object",
            "properties": {
                "zeta": {"type": "string", "description": "dropped"},
                "alpha": {"type": ["integer", "null"]},
                "tags": {"type": "array", "items": {"enum": ["a", "b"]}, "maxItems": 3},
                "kind": {"anyOf": [{"const": "x"}, {"type": "null"}]}
            },
            "required": ["zeta", "kind"]
        }));
        let g = grammar_schema(&s);
        assert_eq!(
            g.to_string(),
            json!({
                "type": "object",
                "properties": {
                    "zeta": {"type": "string"},
                    "alpha": {"anyOf": [{"type": "integer"}, {"type": "null"}]},
                    "tags": {"type": "array", "items": {"type": "string", "enum": ["a", "b"]}, "maxItems": 3},
                    "kind": {"anyOf": [{"type": "string", "enum": ["x"]}, {"type": "null"}]}
                },
                "required": ["zeta", "kind"],
                "additionalProperties": false,
                "x-guidance": {"whitespace_flexible": false, "item_separator": ", ", "key_separator": ": "}
            })
            .to_string()
        );
    }

    fn guidance() -> Guidance {
        Guidance::new(byte_tokenizer(), &[257, 259], 300)
    }

    /// Drive a guide over `text`'s bytes, taking forced tokens where the
    /// grammar has them: returns whether every byte was allowed, and how
    /// many were forced.
    fn walk(g: &mut Guide, text: &str) -> (bool, usize) {
        let ids: Vec<u32> = text.bytes().map(byte_id).collect();
        let mut i = 0;
        let mut forced = 0;
        while i < ids.len() {
            let mask = g.mask().unwrap();
            if !mask.is_allowed(ids[i]) {
                return (false, forced);
            }
            g.consume(ids[i]).unwrap();
            i += 1;
            let f = g.forced(usize::MAX).unwrap();
            if !ids[i..].starts_with(&f) {
                return (false, forced);
            }
            i += f.len();
            forced += f.len();
        }
        (true, forced)
    }

    /// The test tokenizer's id for a byte.
    fn byte_id(b: u8) -> u32 {
        let p = crate::prompt::tests::prompter();
        p.encode_text(std::str::from_utf8(&[b]).unwrap_or("?")).unwrap()[0]
    }

    #[test]
    fn masks_force_and_finish_on_the_test_tokenizer() {
        let guidance = guidance();
        let s = schema(json!({
            "type": "object",
            "properties": {"name": {"type": "string"}, "n": {"type": ["integer", "null"]}},
            "required": ["name", "n"]
        }));
        let mut g = guidance.guide(&s).unwrap();
        assert!(g.stats().compiled);
        // `{"name": "` is forced from the start.
        assert_eq!(g.forced(usize::MAX).unwrap().len(), r#"{"name": ""#.len());
        let (ok, forced) = walk(&mut g, r#"Al", "n": null}"#);
        assert!(ok);
        // `, "n": ` after the string's closing quote, `ull}` after `n`.
        assert_eq!(forced, r#", "n": "#.len() + "ull}".len());
        assert!(g.done(), "nothing may follow the root object");

        // Whitespace isn't allowed outside the separators.
        let mut g = guidance.guide(&s).unwrap();
        assert!(!g.stats().compiled, "the second request uses the cached grammar");
        g.forced(usize::MAX).unwrap();
        assert!(!walk(&mut g, r#"Al",  "n": 1}"#).0);
        let mut g = guidance.guide(&s).unwrap();
        g.forced(usize::MAX).unwrap();
        assert!(!walk(&mut g, r#"Al","n":1}"#).0, "nor compact separators");
        let mut g = guidance.guide(&s).unwrap();
        g.forced(usize::MAX).unwrap();
        assert!(!walk(&mut g, r#"Al", "n": 1, "x": 2}"#).0, "closed object");
    }

    #[test]
    fn stop_tokens_are_allowed_once_the_json_is_complete() {
        let guidance = guidance();
        let mut g = guidance.guide(&schema(json!({"type": "integer"}))).unwrap();
        let mask = g.mask().unwrap();
        assert!(!mask.is_allowed(257) && mask.is_allowed(byte_id(b'7')));
        g.consume(byte_id(b'7')).unwrap();
        let mask = g.mask().unwrap();
        assert!(mask.is_allowed(257) && mask.is_allowed(259) && mask.is_allowed(byte_id(b'0')));
        assert!(!g.done());
    }

    #[test]
    fn a_tokenizer_llguidance_cant_read_refuses_guided_output() {
        // A word-level tokenizer has no byte-level decoder.
        let json = json!({
            "version": "1.0", "truncation": null, "padding": null, "added_tokens": [], "normalizer": null,
            "pre_tokenizer": {"type": "Whitespace"}, "post_processor": null, "decoder": null,
            "model": {"type": "WordLevel", "vocab": {"[UNK]": 0, "a": 1}, "unk_token": "[UNK]"}
        });
        let tokenizer = Tokenizer::from_bytes(serde_json::to_vec(&json).unwrap()).unwrap();
        let guidance = Guidance::new(tokenizer, &[1], 2);
        let Err(err) = guidance.guide(&schema(json!({"type": "integer"}))) else { panic!("guided") };
        assert!(matches!(&err, Error::InvalidRequest(m) if m.contains("can't guide")), "{err}");
    }
}
