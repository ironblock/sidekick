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
//!   nullable properties as an `anyOf` with `null`.
//! - **The schema in the prompt.** The grammar only forces what comes next;
//!   it can't tell the model which fields exist, which are optional, what
//!   an enum offers or what a description asks. So, as Foundation Models
//!   puts a guided request's schema in its prompt, [`with_schema_in_prompt`]
//!   adds the schema, descriptions included, to the system message, where
//!   the model's own template puts machine-readable definitions (tools).
//! - **One line of JSON, spelled as models write it.** The grammar allows
//!   no whitespace between tokens except the separators `, ` and `: `
//!   (Python's `json.dumps` spelling), so a guided reply is one line with
//!   its keys in schema order, and identical requests at temperature 0
//!   give identical bytes. Those separators are the ones a model writes
//!   unprompted: compact `,` and `:` split its usual tokens (`": "`), and
//!   MiniCPM5-2B then wrote `{"name":", "}` for a contact it extracts
//!   correctly with spaces. A model can't spend its budget on blank lines,
//!   and more of the reply is forced than with free whitespace. Numbers
//!   are as long as the model writes them: JSON Schema's `integer` and
//!   `number` have no bounds here, and JSON allows exponents past what a
//!   double holds (`1e400`).
//! - **One tokenizer bridge per model** ([`Guidance`]): llguidance's token
//!   trie, built from the model's tokenizer on its first guided request,
//!   with the model's stop tokens as end-of-sequence tokens. Compiled
//!   grammars are cached by the schema written out, so a schema a client
//!   repeats compiles once; a grammar that fails a request is evicted.
//! - **Forced tokens.** Where the grammar allows exactly one continuation
//!   (a property name, a closing brace), [`Guide::forced`] returns it, and
//!   the loop feeds those tokens in the same call as the sampled one,
//!   filling slots the call would otherwise pad.

use crate::prompt::Prompter;
use llguidance::api::{StopReason, TopLevelGrammar};
use llguidance::toktrie::{InferenceCapabilities, SimpleVob, TokEnv};
use llguidance::{Matcher, ParserFactory};
use serde_json::{json, Map, Value};
use sidekick_core::schema::{Kind, Node, Property};
use sidekick_core::{ChatMessage, Error, JsonSchema, Result, Role};
use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock, PoisonError};
use std::time::{Duration, Instant};
use tokenizers::decoders::DecoderWrapper;
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

/// `schema` as the model reads it in its prompt: JSON Schema with each
/// `description`, `required` where anything is, and a nullable property's
/// type with `"null"`, as clients usually write it.
pub fn prompt_schema(schema: &JsonSchema) -> Value {
    prompt_node(schema.root(), false)
}

fn prompt_node(node: &Node, nullable: bool) -> Value {
    let mut m = Map::new();
    let kind = |name: &str| if nullable { json!([name, "null"]) } else { json!(name) };
    match &node.kind {
        Kind::Object { properties } => {
            m.insert("type".into(), kind("object"));
            if let Some(d) = &node.description {
                m.insert("description".into(), json!(d));
            }
            let props: Map<String, Value> =
                properties.iter().map(|p| (p.name.clone(), prompt_node(&p.schema, p.nullable))).collect();
            m.insert("properties".into(), Value::Object(props));
            let required: Vec<&str> = properties.iter().filter(|p| p.required).map(|p| p.name.as_str()).collect();
            if !required.is_empty() {
                m.insert("required".into(), json!(required));
            }
            return Value::Object(m);
        }
        Kind::Array { items, min_items, max_items } => {
            m.insert("type".into(), kind("array"));
            m.insert("items".into(), prompt_node(items, false));
            if let Some(n) = min_items {
                m.insert("minItems".into(), json!(n));
            }
            if let Some(n) = max_items {
                m.insert("maxItems".into(), json!(n));
            }
        }
        Kind::Enum(choices) => {
            m.insert("type".into(), kind("string"));
            let mut values: Vec<Value> = choices.iter().map(|c| json!(c)).collect();
            if nullable {
                values.push(Value::Null);
            }
            m.insert("enum".into(), Value::Array(values));
        }
        Kind::String => {
            m.insert("type".into(), kind("string"));
        }
        Kind::Integer => {
            m.insert("type".into(), kind("integer"));
        }
        Kind::Number => {
            m.insert("type".into(), kind("number"));
        }
        Kind::Boolean => {
            m.insert("type".into(), kind("boolean"));
        }
    }
    if let Some(d) = &node.description {
        m.insert("description".into(), json!(d));
    }
    Value::Object(m)
}

/// `value` on one line with `, ` and `: ` between values and non-ASCII
/// text as it is, as Python's `json.dumps(ensure_ascii=False)` writes it,
/// and as chat templates write tool definitions.
pub fn spaced_json(value: &Value) -> String {
    match value {
        Value::Object(map) => {
            let items: Vec<String> =
                map.iter().map(|(k, v)| format!("{}: {}", Value::from(k.as_str()), spaced_json(v))).collect();
            format!("{{{}}}", items.join(", "))
        }
        Value::Array(values) => format!("[{}]", values.iter().map(spaced_json).collect::<Vec<_>>().join(", ")),
        other => other.to_string(),
    }
}

/// The instruction a guided request's prompt carries.
pub fn schema_instruction(schema: &JsonSchema) -> String {
    format!(
        "# Response format\n\nReply with only JSON that matches this JSON Schema:\n{}",
        spaced_json(&prompt_schema(schema))
    )
}

/// `messages` with the schema's instruction in the system message: after
/// the client's own system text and a blank line, or as a system message
/// of its own when there is none. That is where the model's template puts
/// tool definitions, and where Foundation Models' instructions are.
pub fn with_schema_in_prompt(messages: &[ChatMessage], schema: &JsonSchema) -> Vec<ChatMessage> {
    let instruction = schema_instruction(schema);
    let mut out = messages.to_vec();
    match out.first_mut() {
        Some(first) if first.role == Role::System => {
            first.content = if first.content.trim().is_empty() {
                instruction
            } else {
                format!("{}\n\n{instruction}", first.content.trim_end())
            };
        }
        _ => out.insert(0, ChatMessage::new(Role::System, instruction)),
    }
    out
}

/// Whether llguidance can read `tokenizer`: a byte-level or byte-fallback
/// decoder (as `toktrie_hf_tokenizers` requires), its vocabulary within the
/// logits row, and the stop tokens in it. Cheap, so the model's listing
/// can say before any request.
fn check_tokenizer(tokenizer: &Tokenizer, stop_ids: &[u32], vocab_size: usize) -> std::result::Result<(), String> {
    fn bytewise(d: &DecoderWrapper) -> bool {
        match d {
            DecoderWrapper::ByteLevel(_) | DecoderWrapper::ByteFallback(_) => true,
            DecoderWrapper::Sequence(s) => s.get_decoders().iter().any(bytewise),
            _ => false,
        }
    }
    if !tokenizer.get_decoder().is_some_and(bytewise) {
        return Err("its tokenizer isn't byte-level or byte-fallback BPE".into());
    }
    let vocab = tokenizer.get_vocab_size(true);
    if vocab > vocab_size {
        return Err(format!("its tokenizer has {vocab} tokens, past its logits' {vocab_size}"));
    }
    if stop_ids.is_empty() || stop_ids.iter().any(|&t| t as usize >= vocab) {
        return Err(format!("its stop tokens {stop_ids:?} aren't all in its tokenizer's {vocab} tokens"));
    }
    Ok(())
}

/// How many compiled grammars a model keeps.
const GRAMMAR_CACHE: usize = 32;

/// A model's compiled grammars, freshly started, by the schema written
/// out; the most recently used last. A guide clones its entry, sharing
/// llguidance's lexer state with it, and evicts it if the grammar fails.
#[derive(Default)]
struct GrammarCache {
    entries: Mutex<VecDeque<(String, Matcher)>>,
    hits: AtomicU64,
    misses: AtomicU64,
    evictions: AtomicU64,
}

impl GrammarCache {
    fn lock(&self) -> std::sync::MutexGuard<'_, VecDeque<(String, Matcher)>> {
        self.entries.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// A clone of `key`'s grammar, now the most recently used. An entry
    /// found in an error state is dropped instead.
    fn get(&self, key: &str) -> Option<Matcher> {
        let mut entries = self.lock();
        let i = entries.iter().position(|(k, _)| k == key)?;
        let entry = entries.remove(i).expect("found above");
        if entry.1.is_error() {
            self.evictions.fetch_add(1, Ordering::Relaxed);
            return None;
        }
        let matcher = entry.1.clone();
        entries.push_back(entry);
        self.hits.fetch_add(1, Ordering::Relaxed);
        Some(matcher)
    }

    /// Keep `matcher` for `key`, unless a request compiling the same
    /// schema at the same time kept its own first; returns the one kept,
    /// cloned.
    fn insert(&self, key: String, matcher: Matcher) -> Matcher {
        let mut entries = self.lock();
        if let Some((_, kept)) = entries.iter().find(|(k, _)| *k == key) {
            return kept.clone();
        }
        if entries.len() == GRAMMAR_CACHE {
            entries.pop_front();
        }
        entries.push_back((key, matcher.clone()));
        matcher
    }

    fn evict(&self, key: &str) {
        let mut entries = self.lock();
        if let Some(i) = entries.iter().position(|(k, _)| k == key) {
            entries.remove(i);
            self.evictions.fetch_add(1, Ordering::Relaxed);
        }
    }
}

/// A model's bridge to llguidance: its tokenizer's token trie, built on
/// first use, and its compiled grammars.
pub struct Guidance {
    prompter: Arc<Prompter>,
    stop_ids: Vec<u32>,
    vocab_size: usize,
    /// Why llguidance can't read the tokenizer, checked up front.
    unavailable: Option<String>,
    factory: OnceLock<std::result::Result<Arc<ParserFactory>, String>>,
    cache: Arc<GrammarCache>,
}

impl Guidance {
    /// For a model with this prompter's tokenizer, these stop tokens and a
    /// logits row `vocab_size` wide. The tokenizer is copied for llguidance
    /// only when the first guided request builds the trie.
    pub fn new(prompter: Arc<Prompter>, stop_ids: &[u32], vocab_size: usize) -> Self {
        let unavailable = check_tokenizer(prompter.tokenizer(), stop_ids, vocab_size).err();
        Self {
            prompter,
            stop_ids: stop_ids.to_vec(),
            vocab_size,
            unavailable,
            factory: OnceLock::new(),
            cache: Arc::default(),
        }
    }

    /// Whether this model can guide its output; why not when it can't.
    pub fn available(&self) -> std::result::Result<(), &str> {
        match &self.unavailable {
            None => Ok(()),
            Some(reason) => Err(reason),
        }
    }

    /// The token trie and parser factory, built on first use (about 0.4 s
    /// for a 130k vocabulary).
    fn factory(&self) -> Result<Arc<ParserFactory>> {
        if let Some(reason) = &self.unavailable {
            return Err(Error::InvalidRequest(format!("this model can't guide its output: {reason}")));
        }
        self.factory
            .get_or_init(|| {
                let started = Instant::now();
                let factory = build_factory(self.prompter.tokenizer().clone(), &self.stop_ids, self.vocab_size);
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
        if let Some(matcher) = self.cache.get(&key) {
            return Ok(Guide::new(matcher, false, self.cache.clone(), key));
        }
        self.cache.misses.fetch_add(1, Ordering::Relaxed);
        let parser = factory
            .create_parser(TopLevelGrammar::from_json_schema(grammar))
            .map_err(|e| Error::GuidedGeneration(format!("llguidance can't compile it: {e}")))?;
        let matcher = Matcher::new(Ok(parser));
        if let Some(e) = matcher.get_error() {
            return Err(Error::GuidedGeneration(format!("llguidance can't compile it: {e}")));
        }
        let matcher = self.cache.insert(key.clone(), matcher);
        Ok(Guide::new(matcher, true, self.cache.clone(), key))
    }

    /// `/health`'s facts: whether the model can guide, whether its trie is
    /// built, and its grammar cache.
    pub fn health(&self) -> Value {
        json!({
            "available": self.unavailable.is_none(),
            "reason": self.unavailable,
            "trie_built": self.factory.get().is_some_and(|f| f.is_ok()),
            "grammars": self.cache.lock().len(),
            "grammar_hits": self.cache.hits.load(Ordering::Relaxed),
            "grammar_misses": self.cache.misses.load(Ordering::Relaxed),
            "grammar_evictions": self.cache.evictions.load(Ordering::Relaxed),
        })
    }
}

fn build_factory(tokenizer: Tokenizer, stop_ids: &[u32], vocab_size: usize) -> std::result::Result<ParserFactory, String> {
    let mut bytes = toktrie_hf_tokenizers::ByteTokenizer::from_tokenizer(tokenizer).map_err(|e| e.to_string())?;
    let vocab = bytes.tokrx_info().vocab_size as usize;
    if vocab > vocab_size {
        return Err(format!("its tokenizer has {vocab} tokens, past its logits' {vocab_size}"));
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
    /// Where its grammar is cached, and under what, to evict it if it fails.
    cache: Arc<GrammarCache>,
    key: String,
}

impl Guide {
    fn new(matcher: Matcher, compiled: bool, cache: Arc<GrammarCache>, key: String) -> Self {
        Self { matcher: Box::new(matcher), stats: GuideStats { compiled, ..Default::default() }, cache, key }
    }

    pub fn stats(&self) -> GuideStats {
        self.stats
    }

    /// The tokens the grammar allows next, the model's stop tokens among
    /// them once the JSON is complete.
    pub fn mask(&mut self) -> Result<SimpleVob> {
        let started = Instant::now();
        let mask = self.matcher.compute_mask();
        let took = started.elapsed();
        self.stats.masks += 1;
        self.stats.mask_total += took;
        self.stats.mask_max = self.stats.mask_max.max(took);
        let mask = mask.map_err(|e| self.fail(&e.to_string()))?;
        if mask.is_zero() {
            return Err(self.fail("the grammar allows no token of this model's vocabulary"));
        }
        Ok(mask)
    }

    /// Advance past a sampled token.
    pub fn consume(&mut self, token: u32) -> Result<()> {
        self.matcher.consume_token(token).map_err(|e| self.fail(&e.to_string()))?;
        self.check()
    }

    /// The tokens the grammar forces next, at most `limit`, consumed.
    pub fn forced(&mut self, limit: usize) -> Result<Vec<u32>> {
        if limit == 0 || self.matcher.is_stopped() {
            return Ok(Vec::new());
        }
        let mut forced = self.matcher.compute_ff_tokens();
        if let Some(e) = self.matcher.get_error() {
            return Err(self.fail(&e));
        }
        forced.truncate(limit);
        if !forced.is_empty() {
            self.matcher.consume_tokens(&forced).map_err(|e| self.fail(&e.to_string()))?;
            self.stats.forced += forced.len() as u32;
        }
        self.check()?;
        Ok(forced)
    }

    /// Whether the JSON is complete and nothing may follow it (not a parser
    /// stopped by an error or a cancellation).
    pub fn done(&self) -> bool {
        matches!(
            self.matcher.stop_reason(),
            StopReason::NoExtension | StopReason::NoExtensionBias | StopReason::EndOfSentence
        )
    }

    /// An error if the parser stopped for any reason but a complete JSON.
    fn check(&self) -> Result<()> {
        if !self.matcher.is_stopped() || self.done() {
            return Ok(());
        }
        let reason = self.matcher.get_error().unwrap_or_else(|| format!("stopped: {}", self.matcher.stop_reason()));
        Err(self.fail(&reason))
    }

    /// The error for a failed grammar, which is evicted from the cache so
    /// that the next request with the schema compiles it afresh.
    fn fail(&self, reason: &str) -> Error {
        self.cache.evict(&self.key);
        tracing::warn!("guided decoding failed: {reason}; its grammar is evicted");
        Error::Inference(format!("guided decoding failed: {reason}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prompt::tests::{chat_section, prompter, CHATML};

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
        Guidance::new(Arc::new(prompter()), &[257, 259], 300)
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
        let mut section = chat_section();
        section.bos_token = None;
        section.eos_token = None;
        section.stop_token_ids = vec![1];
        let words = Prompter::new(tokenizer, CHATML, &section).unwrap();
        let guidance = Guidance::new(Arc::new(words), &[1], 2);
        // Known up front, for the model's listing.
        assert!(guidance.available().unwrap_err().contains("byte-level"));
        assert_eq!(guidance.health()["available"], false);
        let Err(err) = guidance.guide(&schema(json!({"type": "integer"}))) else { panic!("guided") };
        assert!(matches!(&err, Error::InvalidRequest(m) if m.contains("can't guide")), "{err}");
        // Stop tokens outside the vocabulary, or a vocabulary past the
        // logits, are refused as well.
        let bytes = Arc::new(prompter());
        assert!(Guidance::new(bytes.clone(), &[300], 300).available().is_err());
        assert!(Guidance::new(bytes.clone(), &[257], 200).available().is_err());
        assert!(Guidance::new(bytes, &[257], 300).available().is_ok());
    }

    #[test]
    fn the_prompt_carries_the_schema_with_its_descriptions() {
        let s = schema(json!({
            "type": "object",
            "description": "A meeting",
            "properties": {
                "day": {"type": ["string", "null"], "description": "Weekday"},
                "room": {"anyOf": [{"enum": ["A", "B"]}, {"type": "null"}]},
                "people": {"type": "array", "items": {"type": "string"}, "maxItems": 4},
                "note": {"type": "string"}
            },
            "required": ["day", "room"]
        }));
        assert_eq!(
            spaced_json(&prompt_schema(&s)),
            r#"{"type": "object", "description": "A meeting", "properties": {"day": {"type": ["string", "null"], "description": "Weekday"}, "room": {"type": ["string", "null"], "enum": ["A", "B", null]}, "people": {"type": "array", "items": {"type": "string"}, "maxItems": 4}, "note": {"type": "string"}}, "required": ["day", "room"]}"#
        );
        let instruction = schema_instruction(&s);
        assert!(instruction.starts_with("# Response format\n\nReply with only JSON that matches this JSON Schema:\n{"));

        // Appended to the client's system message after a blank line, or a
        // system message of its own; the conversation is otherwise as sent.
        let user = ChatMessage::new(Role::User, "Thursday, room A");
        let with = with_schema_in_prompt(&[ChatMessage::new(Role::System, "Be brief.\n"), user.clone()], &s);
        assert_eq!(with.len(), 2);
        assert_eq!(with[0].content, format!("Be brief.\n\n{instruction}"));
        assert_eq!(with[1].content, user.content);
        let with = with_schema_in_prompt(std::slice::from_ref(&user), &s);
        assert_eq!((with[0].role, with[0].content.as_str()), (Role::System, instruction.as_str()));
        assert_eq!(with[1].content, user.content);
        // Non-ASCII text is written as it is.
        let s = schema(json!({"type": "string", "description": "名前"}));
        assert_eq!(spaced_json(&prompt_schema(&s)), r#"{"type": "string", "description": "名前"}"#);
    }

    #[test]
    fn the_grammar_cache_dedupes_and_evicts_a_failed_grammar() {
        let guidance = guidance();
        let s = schema(json!({"type": "integer"}));
        let key = grammar_schema(&s).to_string();
        let g = guidance.guide(&s).unwrap();
        assert!(g.stats().compiled);
        // A second compile of the same schema (a concurrent miss) keeps one
        // entry.
        let parser = guidance.factory().unwrap().create_parser(TopLevelGrammar::from_json_schema(grammar_schema(&s))).unwrap();
        guidance.cache.insert(key.clone(), Matcher::new(Ok(parser)));
        assert_eq!(guidance.health()["grammars"], 1);
        // A guide that fails evicts its grammar; the next request compiles.
        let mut g = guidance.guide(&s).unwrap();
        assert_eq!(guidance.health()["grammar_hits"], 1);
        assert!(g.consume(byte_id(b'x')).is_err(), "not allowed by the grammar");
        assert_eq!(guidance.health()["grammars"], 0);
        assert_eq!(guidance.health()["grammar_evictions"], 1);
        assert!(guidance.guide(&s).unwrap().stats().compiled);
        assert_eq!(guidance.health()["trie_built"], true);
    }

    #[test]
    fn done_means_complete_json_not_a_stopped_parser() {
        let guidance = guidance();
        let s = schema(json!({"type": "boolean"}));
        let mut g = guidance.guide(&s).unwrap();
        assert!(!g.done());
        g.consume(byte_id(b't')).unwrap();
        assert_eq!(g.forced(usize::MAX).unwrap().len(), "rue".len());
        assert!(g.done());
        let mut g = guidance.guide(&s).unwrap();
        assert!(g.consume(byte_id(b'q')).is_err());
        assert!(!g.done(), "an errored parser is stopped, not done");
    }
}
