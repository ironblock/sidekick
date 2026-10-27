//! One request's generation, on the decoder's thread: the prompt, then a
//! token per call until a stop token, a stop sequence, `max_tokens` or the
//! client going away.

use crate::decoder::Decoder;
use crate::prompt::Prompter;
use crate::sampler::{Sampler, SamplingParams};
use sidekick_core::shaping::StreamShaper;
use sidekick_core::{DeltaSink, Error, FinishReason, Result};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// A request, ready for the decoder.
pub struct Job {
    /// The rendered, tokenized prompt.
    pub prompt: Vec<u32>,
    /// At most this many generated tokens, the stop token included; the
    /// prompt and these fit the context.
    pub max_tokens: usize,
    pub sampling: SamplingParams,
    pub stop: Vec<String>,
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
    /// Every generated token, the stop token included.
    pub completion_tokens: usize,
    /// From the job's start to the first token.
    pub first_token: Duration,
    /// From the first token to the last.
    pub decode: Duration,
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
    if n == 0 || job.max_tokens == 0 {
        return Err(Error::InvalidRequest("an empty prompt or no tokens to generate".into()));
    }
    let mut sampler = Sampler::new(job.sampling);
    let mut shaper = StreamShaper::verbatim(&job.stop);
    let mut reply = Detokenizer::default();
    let mut generated = 0;
    let mut token = sampler.sample(decoder.prefill(&job.prompt)?);
    let first_token = started.elapsed();
    let finish = loop {
        generated += 1;
        if stop_ids.contains(&token) {
            break FinishReason::Stop;
        }
        let delta = shaper.push(reply.push(prompter, token)?);
        if !delta.is_empty() {
            if let Some(sink) = &mut job.sink {
                if !sink(&delta) {
                    return Err(cancelled());
                }
            }
        }
        if shaper.stop_hit() {
            break FinishReason::Stop;
        }
        if generated == job.max_tokens {
            break FinishReason::Length;
        }
        if job.cancel.load(Ordering::Relaxed) {
            return Err(cancelled());
        }
        token = sampler.sample(decoder.step(token, n + generated - 1)?);
    };
    if !shaper.stop_hit() {
        let tail = shaper.finish(&reply.finish(prompter)?).map_err(|_| {
            Error::Inference("the reply's text changed after part of it was sent".into())
        })?;
        if let (false, Some(sink)) = (tail.is_empty(), &mut job.sink) {
            sink(&tail);
        }
    }
    Ok(Generated {
        content: shaper.emitted().to_string(),
        finish,
        prompt_tokens: n,
        completion_tokens: generated,
        first_token,
        decode: started.elapsed() - first_token,
    })
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

        fn step(&mut self, token: u32, pos: usize) -> Result<&[f32]> {
            let k = pos + 1 - self.prompt_len;
            assert_eq!(Some(&token), self.script.get(k - 1), "fed back what was sampled, at its position");
            assert!(pos < self.context);
            self.calls.lock().unwrap().push((pos, 1));
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
}
