//! One request's generation, on the decoder's thread: the prompt, then a
//! token per call until a stop token, a stop sequence, `max_tokens`, the
//! end of a guided reply's JSON, or the client going away.
//!
//! A guided request ([`Job::guide`]) samples each token from what its
//! grammar allows. The mask is computed after each call, on this thread: it
//! takes microseconds (12 µs on average and 266 µs at most over the chat
//! benchmark's extraction answers on MiniCPM5-2B's 130k vocabulary)
//! against a call's 53 ms, less than handing it to another thread would
//! cost. When the grammar allows exactly one continuation, those tokens are
//! fed after the sampled one in the same call, in slots it would otherwise
//! pad.

use crate::decoder::Decoder;
use crate::guided::{Guide, GuideStats};
use crate::prompt::Prompter;
use crate::sampler::{Sampler, SamplingParams};
use sidekick_core::shaping::StreamShaper;
use sidekick_core::{DeltaSink, Error, FinishReason, Result};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use toktrie::SimpleVob;

/// A request, ready for the decoder.
pub struct Job {
    /// The rendered, tokenized prompt.
    pub prompt: Vec<u32>,
    /// At most this many generated tokens, the stop token included; the
    /// prompt and these fit the context.
    pub max_tokens: usize,
    pub sampling: SamplingParams,
    pub stop: Vec<String>,
    /// The grammar a guided reply follows (`response_format`
    /// `json_schema`).
    pub guide: Option<Guide>,
    /// Set when the request timed out or its client went away: generation
    /// stops before its next call.
    pub cancel: Arc<AtomicBool>,
    /// Where streamed text goes; `None` for a reply returned whole.
    pub sink: Option<DeltaSink>,
}

/// What a generation produced.
#[derive(Debug, Clone, PartialEq)]
pub struct Generated {
    pub content: String,
    pub finish: FinishReason,
    pub prompt_tokens: usize,
    /// Every generated token, the stop token and a grammar's forced tokens
    /// included.
    pub completion_tokens: usize,
    /// From the job's start to the first token.
    pub first_token: Duration,
    /// From the first token to the last.
    pub decode: Duration,
    /// Decoder calls after the prompt's.
    pub calls: usize,
    /// A guided reply's grammar work.
    pub guided: Option<GuideStats>,
}

/// The error a stopped generation returns: its request has already timed
/// out or lost its client, so nobody reads it but the logs.
pub fn cancelled() -> Error {
    Error::Other("generation stopped: the request timed out or its client went away".into())
}

/// Run `job` on `decoder`. Text reaches the sink as complete UTF-8, and
/// text that could still become a stop sequence is held back, by the same
/// shaper Foundation Models' replies go through (without its leading-label
/// rule, which is for its replayed transcripts).
pub fn generate(decoder: &mut dyn Decoder, prompter: &Prompter, stop_ids: &[u32], job: &mut Job) -> Result<Generated> {
    let started = Instant::now();
    let n = job.prompt.len();
    let max = job.max_tokens;
    if n == 0 || max == 0 {
        return Err(Error::InvalidRequest("an empty prompt or no tokens to generate".into()));
    }
    let per_call = decoder.spec().tokens_per_call.max(1);
    let mut sampler = Sampler::new(job.sampling);
    let mut out = Output {
        shaper: StreamShaper::verbatim(&job.stop),
        reply: Detokenizer::default(),
        sink: job.sink.take(),
        prompter,
        generated: 0,
    };
    let mut guide = job.guide.take();
    let finished = |guide: &Option<Guide>, out: &Output| -> Option<FinishReason> {
        if out.shaper.stop_hit() || guide.as_ref().is_some_and(Guide::done) {
            Some(FinishReason::Stop)
        } else if out.generated == max {
            Some(FinishReason::Length)
        } else {
            None
        }
    };

    // A grammar can force a reply's first tokens (`{"name":"`): they
    // follow the prompt in its calls, and are sent once those have run, so
    // a failure before the model has run keeps its own HTTP status.
    let opening = match guide.as_mut() {
        Some(g) => g.forced(max)?,
        None => Vec::new(),
    };
    let prompt: Vec<u32> = job.prompt.iter().chain(&opening).copied().collect();
    let mut first_token = None;
    let mut calls = 0;
    let finish = 'generation: {
        if opening.len() == max || guide.as_ref().is_some_and(Guide::done) {
            // The whole reply was forced: no call is needed.
            for &t in &opening {
                out.push(t)?;
            }
            break 'generation finished(&guide, &out).unwrap_or(FinishReason::Length);
        }
        let (mut logits, mut mask) = (decoder.prefill(&prompt)?, next_mask(&mut guide)?);
        for &t in &opening {
            out.push(t)?;
        }
        let mut pos = prompt.len();
        loop {
            let token = sampler.sample_masked(logits, mask.as_ref());
            first_token.get_or_insert_with(|| started.elapsed());
            if stop_ids.contains(&token) {
                out.generated += 1;
                break 'generation FinishReason::Stop;
            }
            if let Some(g) = guide.as_mut() {
                g.consume(token)?;
            }
            out.push(token)?;
            let mut pending = vec![token];
            if let Some(finish) = finished(&guide, &out) {
                break 'generation finish;
            }
            if let Some(g) = guide.as_mut() {
                for t in g.forced(max - out.generated)? {
                    out.push(t)?;
                    pending.push(t);
                }
                if let Some(finish) = finished(&guide, &out) {
                    break 'generation finish;
                }
            }
            if job.cancel.load(Ordering::Relaxed) {
                return Err(cancelled());
            }
            // More forced tokens than a call holds: the rest go in whole
            // calls first, their logits unused.
            while pending.len() > per_call {
                decoder.step(&pending[..per_call], pos)?;
                calls += 1;
                pos += per_call;
                pending.drain(..per_call);
            }
            (logits, mask) = (decoder.step(&pending, pos)?, next_mask(&mut guide)?);
            calls += 1;
            pos += pending.len();
        }
    };
    let first_token = first_token.unwrap_or_else(|| started.elapsed());
    if !out.shaper.stop_hit() {
        let tail = out.shaper.finish(&out.reply.finish(prompter)?).map_err(|_| {
            Error::Inference("the reply's text changed after part of it was sent".into())
        })?;
        if let (false, Some(sink)) = (tail.is_empty(), &mut out.sink) {
            sink(&tail);
        }
    }
    Ok(Generated {
        content: out.shaper.emitted().to_string(),
        finish,
        prompt_tokens: n,
        completion_tokens: out.generated,
        first_token,
        decode: started.elapsed() - first_token,
        calls,
        guided: guide.map(|g| g.stats()),
    })
}

/// The reply as it is generated: detokenized, shaped and streamed.
struct Output<'a> {
    shaper: StreamShaper<'a>,
    reply: Detokenizer,
    sink: Option<DeltaSink>,
    prompter: &'a Prompter,
    /// Tokens generated so far.
    generated: usize,
}

impl Output<'_> {
    /// Add a generated token (not a stop token): count it, and send the
    /// text it completes.
    fn push(&mut self, token: u32) -> Result<()> {
        self.generated += 1;
        let delta = self.shaper.push(self.reply.push(self.prompter, token)?);
        if let (false, Some(sink)) = (delta.is_empty(), &mut self.sink) {
            if !sink(&delta) {
                return Err(cancelled());
            }
        }
        Ok(())
    }
}

/// The tokens a guided reply allows next; `None` unguided.
fn next_mask(guide: &mut Option<Guide>) -> Result<Option<SimpleVob>> {
    guide.as_mut().map(Guide::mask).transpose()
}

/// Incremental detokenizing, as text-generation-inference does it: each
/// token decodes only the tokens since the last complete text, with those
/// before it as context (some decoders treat a sequence's first token
/// differently), so a reply costs time linear in its length. Text ending
/// in an incomplete UTF-8 character (U+FFFD) waits for the next token.
#[derive(Default)]
pub struct Detokenizer {
    ids: Vec<u32>,
    /// Where the context of the next decode starts.
    prefix: usize,
    /// The tokens before this are in `text`.
    read: usize,
    text: String,
}

impl Detokenizer {
    /// Add `id`; returns the reply's complete text so far.
    pub fn push(&mut self, prompter: &Prompter, id: u32) -> Result<&str> {
        self.ids.push(id);
        let before = prompter.decode(&self.ids[self.prefix..self.read])?;
        let now = prompter.decode(&self.ids[self.prefix..])?;
        if now.len() > before.len() && !now.ends_with('\u{FFFD}') {
            match now.strip_prefix(before.as_str()) {
                Some(delta) => self.text.push_str(delta),
                // A decoder whose text isn't prefix-stable here: decode the
                // whole reply.
                None => self.text = prompter.decode(&self.ids)?,
            }
            self.prefix = self.read;
            self.read = self.ids.len();
        }
        Ok(&self.text)
    }

    /// The whole reply's text, an incomplete last character included (as
    /// U+FFFD, which the shaper drops).
    pub fn finish(&self, prompter: &Prompter) -> Result<String> {
        if self.read == self.ids.len() {
            return Ok(self.text.clone());
        }
        let before = prompter.decode(&self.ids[self.prefix..self.read])?;
        let now = prompter.decode(&self.ids[self.prefix..])?;
        Ok(match now.strip_prefix(before.as_str()) {
            Some(tail) => format!("{}{tail}", self.text),
            None => prompter.decode(&self.ids)?,
        })
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::decoder::{DecoderProbe, DecoderSpec, NoProbe};
    use crate::prompt::tests::prompter;
    use std::sync::Mutex;
    use crate::guided::Guidance;
    use sidekick_core::JsonSchema;

    /// A decoder that "generates" a script: after a prompt of n tokens, the
    /// call at position n + k - 1 returns logits peaking at `script[k]`
    /// (the prefill's at `script[0]`), and at token 257 (`</s>`) once the
    /// script runs out. It checks the positions it is called at, and
    /// records every call.
    pub struct Scripted {
        pub script: Vec<u32>,
        pub context: usize,
        prompt_len: usize,
        logits: Vec<f32>,
        pub calls: Arc<Mutex<Vec<(usize, usize)>>>,
    }

    impl Scripted {
        pub fn new(script: Vec<u32>) -> Self {
            Self { script, context: 2033, prompt_len: 0, logits: vec![0.0; 300], calls: Arc::default() }
        }

        fn peak(&mut self, k: usize) -> &[f32] {
            self.logits.fill(0.0);
            let token = self.script.get(k).copied().unwrap_or(257);
            self.logits[token as usize] = 10.0;
            &self.logits
        }
    }

    impl Decoder for Scripted {
        fn spec(&self) -> DecoderSpec {
            DecoderSpec { tokens_per_call: 16, context_length: self.context, vocab_size: 300 }
        }

        fn prefill(&mut self, tokens: &[u32]) -> Result<&[f32]> {
            self.prompt_len = tokens.len();
            self.calls.lock().unwrap().push((0, tokens.len()));
            Ok(self.peak(0))
        }

        fn step(&mut self, tokens: &[u32], pos: usize) -> Result<&[f32]> {
            let k = pos + tokens.len() - self.prompt_len;
            assert_eq!(Some(tokens), self.script.get(k - tokens.len()..k), "fed back what was sampled, at its position");
            assert!(pos + tokens.len() <= self.context);
            self.calls.lock().unwrap().push((pos, tokens.len()));
            Ok(self.peak(k))
        }

        fn probe(&self) -> Arc<dyn DecoderProbe> {
            Arc::new(NoProbe)
        }
    }

    /// Byte tokens for `text` (the test tokenizer is one token a byte).
    pub fn bytes(text: &str) -> Vec<u32> {
        prompter().encode_text(text).unwrap()
    }

    fn job(max_tokens: usize, stop: &[&str], sink: Option<DeltaSink>) -> Job {
        Job {
            prompt: vec![256, 1, 2, 3],
            max_tokens,
            sampling: SamplingParams { temperature: 0.0, top_p: 1.0, seed: None },
            stop: stop.iter().map(|s| s.to_string()).collect(),
            guide: None,
            cancel: Arc::default(),
            sink,
        }
    }

    fn collect() -> (DeltaSink, Arc<Mutex<Vec<String>>>) {
        let seen: Arc<Mutex<Vec<String>>> = Arc::default();
        let into = seen.clone();
        (Box::new(move |d: &str| {
            into.lock().unwrap().push(d.to_string());
            true
        }), seen)
    }

    #[test]
    fn generates_until_a_stop_token_and_counts_it() {
        let p = prompter();
        let mut d = Scripted::new(bytes("Hi there"));
        let g = generate(&mut d, &p, &[257, 259], &mut job(100, &[], None)).unwrap();
        assert_eq!((g.content.as_str(), g.finish, g.prompt_tokens, g.completion_tokens), ("Hi there", FinishReason::Stop, 4, 9));
        // The prompt in one prefill, then one call a token from position 4;
        // the stop token is never fed back.
        let calls = d.calls.lock().unwrap().clone();
        assert_eq!(calls[0], (0, 4));
        assert_eq!(calls[1..].iter().map(|c| c.0).collect::<Vec<_>>(), (4..12).collect::<Vec<_>>());
    }

    #[test]
    fn max_tokens_ends_with_length() {
        let p = prompter();
        let mut d = Scripted::new(bytes("Hello world"));
        let g = generate(&mut d, &p, &[257], &mut job(5, &[], None)).unwrap();
        assert_eq!((g.content.as_str(), g.finish, g.completion_tokens), ("Hello", FinishReason::Length, 5));
        // The last token is never fed back: 1 prefill + 4 steps.
        assert_eq!(d.calls.lock().unwrap().len(), 5);
    }

    #[test]
    fn a_stop_sequence_ends_the_reply_before_it() {
        let p = prompter();
        let mut d = Scripted::new(bytes("one, two. three"));
        let (sink, seen) = collect();
        let g = generate(&mut d, &p, &[257], &mut job(100, &[". "], Some(sink))).unwrap();
        assert_eq!((g.content.as_str(), g.finish), ("one, two", FinishReason::Stop));
        // Generation stopped at the token that completed the stop sequence.
        assert_eq!(g.completion_tokens, "one, two. ".len());
        assert_eq!(seen.lock().unwrap().concat(), "one, two");
    }

    #[test]
    fn streamed_text_is_whole_utf8() {
        let p = prompter();
        let text = "né 東京 🚀!";
        let mut d = Scripted::new(bytes(text));
        let (sink, seen) = collect();
        let g = generate(&mut d, &p, &[257], &mut job(100, &[], Some(sink))).unwrap();
        assert_eq!(g.content, text);
        let deltas = seen.lock().unwrap().clone();
        assert_eq!(deltas.concat(), text);
        assert!(deltas.iter().all(|d| !d.contains('\u{FFFD}')), "{deltas:?}");
        // A multi-byte character cut by max_tokens is dropped, not sent half.
        let mut d = Scripted::new(bytes("a東"));
        let g = generate(&mut d, &p, &[257], &mut job(3, &[], None)).unwrap();
        assert_eq!((g.content.as_str(), g.finish), ("a", FinishReason::Length));
    }

    #[test]
    fn incremental_detokenizing_matches_decoding_the_whole_reply() {
        let p = prompter();
        for text in ["plain ascii", "né 東京 🚀!", "\u{FFFD} kept", "é", ""] {
            let ids = bytes(text);
            let mut d = Detokenizer::default();
            for (i, &id) in ids.iter().enumerate() {
                let so_far = d.push(&p, id).unwrap().to_string();
                let whole = p.decode(&ids[..=i]).unwrap();
                assert!(whole.starts_with(&so_far), "{text:?} at {i}: {so_far:?} vs {whole:?}");
                assert!(!so_far.ends_with('\u{FFFD}') || text.contains("\u{FFFD} "), "{so_far:?}");
            }
            assert_eq!(d.finish(&p).unwrap(), p.decode(&ids).unwrap(), "{text:?}");
        }
        // A cut character is in the final text as U+FFFD, for the shaper.
        let mut d = Detokenizer::default();
        let ids = bytes("a東");
        for &id in &ids[..2] {
            d.push(&p, id).unwrap();
        }
        assert_eq!(d.finish(&p).unwrap(), "a\u{FFFD}");
    }

    #[test]
    fn cancel_and_a_gone_client_stop_generation() {
        let p = prompter();
        let mut d = Scripted::new(bytes("abcdefgh"));
        let mut j = job(100, &[], Some(Box::new(|_| false)));
        assert!(generate(&mut d, &p, &[257], &mut j).is_err());
        assert_eq!(d.calls.lock().unwrap().len(), 1, "stopped after the first delta");
        let mut d = Scripted::new(bytes("abcdefgh"));
        let j = &mut job(100, &[], None);
        j.cancel.store(true, Ordering::Relaxed);
        assert!(generate(&mut d, &p, &[257], j).is_err());
        assert_eq!(d.calls.lock().unwrap().len(), 1);
    }

    /// Each call's position and tokens.
    type Calls = Arc<Mutex<Vec<(usize, Vec<u32>)>>>;

    /// A decoder whose every logits row ranks tokens in one fixed order:
    /// `order` first, then every other token, lower ids first. It records
    /// each call's position and tokens (the prompt's first, as one).
    pub struct Preferring {
        logits: Vec<f32>,
        pub per_call: usize,
        pub calls: Calls,
    }

    impl Preferring {
        pub fn new(order: &[u32]) -> Self {
            let mut logits: Vec<f32> = (0..300).map(|i| -(i as f32)).collect();
            for (rank, &t) in order.iter().enumerate() {
                logits[t as usize] = 1000.0 - rank as f32;
            }
            Self { logits, per_call: 16, calls: Arc::default() }
        }
    }

    impl Decoder for Preferring {
        fn spec(&self) -> DecoderSpec {
            DecoderSpec { tokens_per_call: self.per_call, context_length: 2033, vocab_size: 300 }
        }

        fn prefill(&mut self, tokens: &[u32]) -> Result<&[f32]> {
            self.calls.lock().unwrap().push((0, tokens.to_vec()));
            Ok(&self.logits)
        }

        fn step(&mut self, tokens: &[u32], pos: usize) -> Result<&[f32]> {
            assert!((1..=self.per_call).contains(&tokens.len()), "{} tokens in a call", tokens.len());
            self.calls.lock().unwrap().push((pos, tokens.to_vec()));
            Ok(&self.logits)
        }

        fn probe(&self) -> Arc<dyn DecoderProbe> {
            Arc::new(NoProbe)
        }
    }

    /// The test tokenizer's id for each of `text`'s bytes.
    fn ids(text: &str) -> Vec<u32> {
        text.bytes().map(|b| bytes(std::str::from_utf8(&[b]).unwrap())[0]).collect()
    }

    fn guided(schema: serde_json::Value, max_tokens: usize, sink: Option<DeltaSink>) -> (Job, JsonSchema) {
        let schema = JsonSchema::parse(&schema).unwrap();
        let guidance = Guidance::new(Arc::new(prompter()), &[257, 259], 300);
        let mut j = job(max_tokens, &[], sink);
        j.guide = Some(guidance.guide(&schema).unwrap());
        (j, schema)
    }

    fn record() -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "n": {"type": ["integer", "null"]},
                "tags": {"type": "array", "items": {"enum": ["red", "blue"]}, "maxItems": 2},
                "ok": {"type": "boolean"}
            },
            "required": ["name", "n", "tags"]
        })
    }

    #[test]
    fn guided_output_follows_the_schema_and_feeds_forced_tokens_with_the_sampled() {
        let p = prompter();
        // Close strings, objects and arrays as soon as allowed; otherwise
        // a comma, then `1`, then `b`.
        let mut d = Preferring::new(&ids("\"}],1b"));
        let (sink, seen) = collect();
        let (mut j, schema) = guided(record(), 100, Some(sink));
        let g = generate(&mut d, &p, &[257, 259], &mut j).unwrap();
        let want = r#"{"name": "", "n": 1, "tags": ["blue"]}"#;
        assert_eq!((g.content.as_str(), g.finish), (want, FinishReason::Stop));
        assert_eq!(seen.lock().unwrap().concat(), want, "streamed as generated");
        assert_eq!(schema.validate(&serde_json::from_str(&g.content).unwrap()), Ok(()));
        // Every token is counted, forced or sampled; the JSON's end needs no
        // stop token.
        assert_eq!(g.completion_tokens, want.len());
        let stats = g.guided.unwrap();
        // `{"name": "`, `, "n": ` after the closing quote, ` "tags": [`
        // after the comma, `lue"` after `b`.
        assert_eq!(stats.forced as usize, 10 + 7 + 10 + 4);
        // The forced opening follows the prompt; each sampled token then
        // shares its call with the tokens the grammar forces after it.
        let calls = d.calls.lock().unwrap().clone();
        assert_eq!(&calls[0].1[4..], ids(r#"{"name": ""#).as_slice());
        let fed: Vec<String> = calls[1..].iter().map(|(_, t)| p.decode(t).unwrap()).collect();
        assert_eq!(fed, [r#"", "n": "#, "1", r#", "tags": ["#, "\"", "blue\"", "]"]);
        assert_eq!(g.calls, 6);
        // Positions run on without a gap.
        let mut pos = calls[0].1.len();
        for (at, tokens) in &calls[1..] {
            assert_eq!(*at, pos);
            pos += tokens.len();
        }

        // More forced tokens than a call takes go in whole calls first.
        let mut d = Preferring::new(&ids("\"}],1b"));
        d.per_call = 4;
        let (mut j, _) = guided(record(), 100, None);
        let g = generate(&mut d, &p, &[257, 259], &mut j).unwrap();
        assert_eq!(g.content, want);
        let sizes: Vec<usize> = d.calls.lock().unwrap()[1..].iter().map(|(_, t)| t.len()).collect();
        assert_eq!(sizes, [4, 4, 1, 4, 4, 3, 1, 4, 1, 1]);
    }

    #[test]
    fn a_reply_the_grammar_forces_whole_needs_no_call() {
        let p = prompter();
        let mut d = Preferring::new(&[]);
        let (mut j, _) = guided(serde_json::json!({"const": "only"}), 100, None);
        let g = generate(&mut d, &p, &[257], &mut j).unwrap();
        assert_eq!((g.content.as_str(), g.finish, g.completion_tokens), ("\"only\"", FinishReason::Stop, 6));
        assert!(d.calls.lock().unwrap().is_empty());
    }

    #[test]
    fn a_stop_token_is_allowed_only_once_the_json_is_complete() {
        let p = prompter();
        // The model would stop at once, but `7` is the first thing allowed.
        let mut d = Preferring::new(&[257, ids("7")[0]]);
        let (mut j, _) = guided(serde_json::json!({"type": "integer"}), 100, None);
        let g = generate(&mut d, &p, &[257], &mut j).unwrap();
        assert_eq!((g.content.as_str(), g.finish, g.completion_tokens), ("7", FinishReason::Stop, 2));
    }

    #[test]
    fn max_tokens_cuts_guided_output_with_length() {
        let p = prompter();
        let mut d = Preferring::new(&ids("a"));
        let (mut j, schema) = guided(record(), 12, None);
        let g = generate(&mut d, &p, &[257], &mut j).unwrap();
        assert_eq!((g.content.as_str(), g.finish, g.completion_tokens), (r#"{"name": "aa"#, FinishReason::Length, 12));
        // Cut short, it isn't JSON: the client sees `length`.
        assert!(serde_json::from_str::<serde_json::Value>(&g.content).is_err());
        assert_eq!(schema.conform(&g.content), g.content);
    }

    /// A decoder with seeded random logits, structural JSON tokens and the
    /// stop token favored, so replies end within a budget.
    struct Noisy {
        rng: rand_chacha::ChaCha20Rng,
        logits: Vec<f32>,
        favored: Vec<u32>,
    }

    impl Noisy {
        fn new(seed: u64) -> Self {
            use rand_core::SeedableRng;
            let mut favored = ids("\"}],:0123456789-ntf");
            favored.push(257);
            Self { rng: rand_chacha::ChaCha20Rng::seed_from_u64(seed), logits: vec![0.0; 300], favored }
        }

        fn row(&mut self) -> &[f32] {
            use rand_core::RngCore;
            for l in self.logits.iter_mut() {
                *l = (self.rng.next_u32() % 1000) as f32 / 250.0;
            }
            for &t in &self.favored {
                self.logits[t as usize] += 3.0;
            }
            &self.logits
        }
    }

    impl Decoder for Noisy {
        fn spec(&self) -> DecoderSpec {
            DecoderSpec { tokens_per_call: 16, context_length: 2033, vocab_size: 300 }
        }
        fn prefill(&mut self, _: &[u32]) -> Result<&[f32]> {
            Ok(self.row())
        }
        fn step(&mut self, _: &[u32], _: usize) -> Result<&[f32]> {
            Ok(self.row())
        }
        fn probe(&self) -> Arc<dyn DecoderProbe> {
            Arc::new(NoProbe)
        }
    }

    #[test]
    fn sampled_guided_replies_are_valid_one_line_json_in_schema_order() {
        let p = prompter();
        let schema = serde_json::json!({
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "kind": {"enum": ["bug", "feature", "question"]},
                "score": {"type": "number"},
                "count": {"type": ["integer", "null"]},
                "done": {"type": "boolean"},
                "owner": {"type": "object", "properties": {"name": {"type": "string"}, "team": {"type": ["string", "null"]}},
                          "required": ["name"]},
                "items": {"type": "array", "minItems": 1, "maxItems": 3,
                          "items": {"type": "object", "properties": {"id": {"type": "integer"}}, "required": ["id"]}}
            },
            "required": ["title", "kind", "count", "owner", "items"]
        });
        let mut complete = 0;
        for seed in 0..40 {
            let mut d = Noisy::new(seed);
            let (mut j, parsed) = guided(schema.clone(), 400, None);
            j.sampling = SamplingParams { temperature: 1.0, top_p: 0.95, seed: Some(seed) };
            let g = generate(&mut d, &p, &[257], &mut j).unwrap();
            if g.finish == FinishReason::Length {
                continue;
            }
            complete += 1;
            let value: serde_json::Value = match serde_json::from_str(&g.content) {
                Ok(v) => v,
                // Valid JSON, but past what an f64 holds (`1e400`): random
                // logits write long exponents.
                Err(e) if e.to_string().contains("out of range") => continue,
                Err(e) => panic!("{e}: {}", g.content),
            };
            assert_eq!(parsed.validate(&value), Ok(()), "{}", g.content);
            assert!(!g.content.contains('\n') || value.to_string().contains('\n'), "one line: {}", g.content);
            // Keys in schema order: re-serializing in that order changes
            // nothing but number spelling.
            let reordered: serde_json::Value = serde_json::from_str(&parsed.conform(&g.content)).unwrap();
            let keys = |v: &serde_json::Value| v.as_object().unwrap().keys().cloned().collect::<Vec<_>>();
            assert_eq!(keys(&value), keys(&reordered), "{}", g.content);
        }
        assert!(complete >= 20, "only {complete} of 40 replies completed");
    }
}
