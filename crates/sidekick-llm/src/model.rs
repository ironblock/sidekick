//! An open chat model as a [`ChatBackend`] (D53): a request is rendered and
//! tokenized, checked against the context, queued in the ANE chat lane,
//! and run on the model's decoder thread, loaded on first use.

use crate::decoder::{DecoderProbe, DecoderStatus};
use crate::engine::{DecoderLoader, Engine, Work};
use crate::generate::{Generated, Job};
use crate::guided::{self, Guidance, Guide};
use crate::lane::{ready, ChatLane, LanePermit, ReadyError};
use crate::prompt::Prompter;
use crate::sampler::SamplingParams;
use serde_json::{json, Map, Value};
use sidekick_core::chat_manifest::SamplingSection;
use sidekick_core::{
    Availability, ChatBackend, ChatRequest, ChatResponse, DeltaSink, Error, Result, UnavailableReason, Usage,
};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, PoisonError};
use std::time::Duration;
use tokio::sync::oneshot;

/// What an open model is, besides its decoder.
pub struct OpenModelSpec {
    pub id: String,
    /// Prompt plus completion: the manifest's `context_length`.
    pub context_length: usize,
    pub prompter: Arc<Prompter>,
    /// Tokens that end a reply.
    pub stop_token_ids: Vec<u32>,
    /// The decoder's logits row's width (the manifest's `vocab_size`).
    pub vocab_size: usize,
    /// Whether a guided request's schema is written into its prompt
    /// ([`guided::with_schema_in_prompt`]); on by default.
    pub schema_in_prompt: bool,
    /// The checkpoint's defaults for a request that omits them.
    pub sampling: SamplingSection,
    /// The model's programs' weights: its estimated share of the Neural
    /// Engine before it loads.
    pub estimated_ane_bytes: u64,
    /// The model's own `queue_depth`, for `/health`; the lane it shares
    /// takes the largest of the installed models'.
    pub queue_depth: usize,
}

/// An open chat model.
pub struct OpenModel {
    spec: OpenModelSpec,
    stop_ids: Arc<[u32]>,
    /// Constrained decoding for `json_schema` requests.
    guidance: Arc<Guidance>,
    lane: Arc<ChatLane>,
    loader: DecoderLoader,
    request_timeout: Duration,
    load_timeout: Duration,
    /// Why the model can't serve: its decoder hung. Recovery is a restart.
    /// Shared with requests' in-flight guards, which can find the hang
    /// after their request has timed out.
    failed: Arc<Mutex<Option<String>>>,
    /// Why the last load failed, until a load succeeds.
    load_error: Mutex<Option<String>>,
    /// The decoder's health as of its last request, for `/health` once it
    /// is unloaded.
    last_status: Mutex<Option<DecoderStatus>>,
    ane_bytes: Mutex<Option<u64>>,
    loads: AtomicU64,
    last_load_ms: AtomicU64,
    requests: AtomicU64,
}

impl OpenModel {
    pub fn new(
        spec: OpenModelSpec,
        lane: Arc<ChatLane>,
        loader: DecoderLoader,
        request_timeout: Duration,
        load_timeout: Duration,
    ) -> Self {
        Self {
            stop_ids: spec.stop_token_ids.clone().into(),
            guidance: Arc::new(Guidance::new(spec.prompter.clone(), &spec.stop_token_ids, spec.vocab_size)),
            spec,
            lane,
            loader,
            request_timeout,
            load_timeout,
            failed: Arc::default(),
            load_error: Mutex::new(None),
            last_status: Mutex::new(None),
            ane_bytes: Mutex::new(None),
            loads: AtomicU64::new(0),
            last_load_ms: AtomicU64::new(0),
            requests: AtomicU64::new(0),
        }
    }

    pub fn lane(&self) -> &Arc<ChatLane> {
        &self.lane
    }

    fn failed(&self) -> Option<String> {
        self.failed.lock().unwrap_or_else(PoisonError::into_inner).clone()
    }

    /// Whether the model has failed, counting a resident decoder whose
    /// watchdog has just reported a hang: an earlier request's lane-holding
    /// task records that only at its next check, and a request arriving
    /// before then would wait on the lane the hung call still holds.
    fn failed_now(&self) -> Option<String> {
        if let Some(reason) = self.failed() {
            return Some(reason);
        }
        let (engine, _) = self.lane.resident().filter(|(e, _)| e.model == self.spec.id)?;
        let reason = engine.facts()?.probe.status().failed?;
        record_failure(&self.failed, &self.spec.id, &reason);
        Some(reason)
    }

    fn fail(&self, reason: String) -> Error {
        record_failure(&self.failed, &self.spec.id, &reason);
        unavailable(&self.spec.id, &reason)
    }

    fn set_load_error(&self, error: Option<String>) {
        *self.load_error.lock().unwrap_or_else(PoisonError::into_inner) = error;
    }

    /// Whether this model can guide its output (`response_format`
    /// `json_schema`); why not when it can't.
    pub fn guided(&self) -> std::result::Result<(), &str> {
        self.guidance.available()
    }

    /// The prompt's token ids, how many tokens may follow them, and a
    /// guided request's grammar, made on a blocking thread: a 2 MiB body is
    /// milliseconds of work, as is compiling a schema. A guided request's
    /// schema is written into its prompt first, so it counts against the
    /// context; the grammar is compiled only once the request fits.
    async fn plan(&self, req: &ChatRequest) -> Result<(Vec<u32>, usize, Option<Guide>)> {
        if let (Some(_), Err(reason)) = (&req.schema, self.guidance.available()) {
            return Err(Error::InvalidRequest(format!(
                "`{}` can't guide its output (response_format json_schema): {reason}",
                self.spec.id
            )));
        }
        let (prompter, guidance) = (self.spec.prompter.clone(), self.guidance.clone());
        let messages = match (&req.schema, self.spec.schema_in_prompt) {
            (Some(schema), true) => guided::with_schema_in_prompt(&req.messages, schema),
            _ => req.messages.clone(),
        };
        let (schema, limit, max_tokens) = (req.schema.clone(), self.spec.context_length, req.max_tokens);
        tokio::task::spawn_blocking(move || -> Result<_> {
            let prompt = prompter.encode(&messages)?;
            let n = prompt.len();
            let max = match max_tokens {
                Some(m) => m as usize,
                None => limit.saturating_sub(n),
            };
            // The prompt and its completion fit the context, with room for
            // at least one generated token (OpenAI's and vLLM's 400).
            if n + max.max(1) > limit {
                return Err(Error::ContextOverflow { limit, actual: Some(n + max.max(1)) });
            }
            let guide = schema.map(|s| guidance.guide(&s)).transpose()?;
            Ok((prompt, max, guide))
        })
        .await
        .map_err(|e| Error::Other(format!("tokenizing task failed: {e}")))?
    }

    async fn run(&self, req: ChatRequest, sink: Option<DeltaSink>) -> Result<ChatResponse> {
        let started = tokio::time::Instant::now();
        if let Some(reason) = self.failed_now() {
            return Err(unavailable(&self.spec.id, &reason));
        }
        let (prompt, max_tokens, guide) = self.plan(&req).await?;
        let constrained = guide.is_some();
        let sampling = SamplingParams {
            temperature: req.temperature.unwrap_or(self.spec.sampling.temperature),
            top_p: req.top_p.unwrap_or(self.spec.sampling.top_p),
            seed: req.seed,
        };
        let mut deadline = started + self.request_timeout;
        let permit = self.lane.acquire(&self.spec.id, deadline, self.request_timeout).await?;
        if let Some(reason) = self.failed() {
            return Err(unavailable(&self.spec.id, &reason));
        }

        let engine = self
            .lane
            .engine(&permit, &self.spec.id, || {
                self.lane.check_budget(&self.spec.id, self.spec.estimated_ane_bytes);
                self.loads.fetch_add(1, Ordering::Relaxed);
                Engine::start(&self.spec.id, self.loader.clone(), self.spec.prompter.clone(), self.stop_ids.clone())
            })
            .await?;
        let ready = match ready(&engine, self.load_timeout).await {
            Ok(r) => r,
            Err(ReadyError::Timeout) => {
                return Err(Error::LoadTimeout { model: self.spec.id.clone(), secs: self.load_timeout.as_secs() })
            }
            Err(ReadyError::Failed(reason)) => {
                self.lane.forget(&self.spec.id);
                self.set_load_error(Some(reason.clone()));
                return Err(Error::Inference(format!("`{}` failed to load: {reason}", self.spec.id)));
            }
        };
        // A model's idle time starts when its load completes.
        self.lane.touch(&self.spec.id);
        self.set_load_error(None);
        if let Some(facts) = engine.facts() {
            *self.ane_bytes.lock().unwrap_or_else(PoisonError::into_inner) = facts.ane_bytes;
            self.last_load_ms.store(facts.load.as_millis() as u64, Ordering::Relaxed);
        }
        if ready.spec.context_length < self.spec.context_length {
            return Err(Error::Inference(format!(
                "`{}`'s decoder has a context of {} tokens, under its manifest's {}",
                self.spec.id, ready.spec.context_length, self.spec.context_length
            )));
        }
        // A load the request waited on doesn't count against its timeout,
        // as on the other routes: `load_timeout` bounds it.
        if let Some(waited) = ready.waited {
            deadline += waited;
        }
        if let Some(reason) = ready.probe.status().failed {
            return Err(self.fail(reason));
        }

        // A request whose time ran out in the queue doesn't start.
        if tokio::time::Instant::now() >= deadline {
            return Err(Error::Timeout { secs: self.request_timeout.as_secs() });
        }

        let cancel = Arc::new(AtomicBool::new(false));
        let (reply, result) = oneshot::channel();
        let job = Job { prompt, max_tokens, sampling, stop: req.stop, guide, cancel: cancel.clone(), sink };
        if !engine.submit(Work { job, reply }) {
            self.lane.forget(&self.spec.id);
            return Err(Error::Inference(format!("`{}`'s decoder thread is gone", self.spec.id)));
        }
        self.requests.fetch_add(1, Ordering::Relaxed);
        let mut flight = InFlight {
            permit: Some(permit),
            result: Some(result),
            cancel,
            probe: ready.probe.clone(),
            failed: self.failed.clone(),
            model: self.spec.id.clone(),
        };
        let outcome = tokio::time::timeout_at(deadline, flight.result.as_mut().expect("set above")).await;
        let generated = match outcome {
            Ok(Ok(result)) => {
                flight.finish();
                result
            }
            Ok(Err(_)) => Err(Error::Inference(format!("`{}`'s decoder thread stopped mid-request", self.spec.id))),
            // `flight` holds the lane until the thread stops.
            Err(_) => Err(Error::Timeout { secs: self.request_timeout.as_secs() }),
        };
        self.lane.touch(&self.spec.id);
        let status = ready.probe.status();
        *self.last_status.lock().unwrap_or_else(PoisonError::into_inner) = Some(status.clone());
        if let Some(reason) = status.failed {
            return Err(self.fail(reason));
        }
        let g = generated?;
        log(&self.spec.id, &g);
        Ok(ChatResponse {
            content: g.content,
            finish: g.finish,
            usage: Usage {
                prompt_tokens: g.prompt_tokens as u32,
                completion_tokens: g.completion_tokens as u32,
                cached_tokens: Some(0),
                reasoning_tokens: None,
            },
            constrained,
        })
    }

    /// `/health`'s facts about the model: its state in the lane, its
    /// decoder's health (live while resident, else as of its last request),
    /// and its share of the Neural Engine.
    pub fn health(&self) -> Map<String, Value> {
        let resident = self.lane.resident().filter(|(e, _)| e.model == self.spec.id);
        let live = resident.as_ref().and_then(|(e, _)| e.facts()).map(|f| f.probe.status());
        let status = live.clone().or_else(|| self.last_status.lock().unwrap_or_else(PoisonError::into_inner).clone());
        let load_error = self.load_error.lock().unwrap_or_else(PoisonError::into_inner).clone();
        let state = match (self.failed(), &resident, &load_error) {
            (Some(_), _, _) => "failed",
            (None, Some((e, _)), _) => e.state_name(),
            (None, None, Some(_)) => "load_failed",
            (None, None, None) => "unloaded",
        };
        let measured = *self.ane_bytes.lock().unwrap_or_else(PoisonError::into_inner);
        let mut m = Map::new();
        m.insert("state".into(), json!(state));
        m.insert("failed".into(), json!(self.failed()));
        m.insert("load_error".into(), json!(load_error));
        m.insert("resident".into(), json!(resident.is_some()));
        m.insert("idle_secs".into(), json!(resident.map(|(_, at)| at.elapsed().as_secs())));
        m.insert("queue_depth".into(), json!(self.spec.queue_depth));
        m.insert("guided".into(), self.guidance.health());
        m.insert("schema_in_prompt".into(), json!(self.spec.schema_in_prompt));
        m.insert("lane_queue_depth".into(), json!(self.lane.options().queue_depth));
        m.insert("requests".into(), json!(self.requests.load(Ordering::Relaxed)));
        m.insert("loads".into(), json!(self.loads.load(Ordering::Relaxed)));
        let last_load = self.last_load_ms.load(Ordering::Relaxed);
        m.insert("last_load_ms".into(), json!((last_load > 0).then_some(last_load)));
        m.insert(
            "ane".into(),
            json!({
                // What the load added to the process's footprint, when it
                // was measured; the programs' weights is the estimate
                // before a load.
                "measured_bytes": measured,
                "estimated_bytes": self.spec.estimated_ane_bytes,
            }),
        );
        m.insert(
            "decoder".into(),
            match status {
                None => Value::Null,
                Some(s) => json!({
                    "live": live.is_some(),
                    "calls": s.calls,
                    "stalls": s.stalls,
                    "warm_median_ms": s.warm_median_ms,
                    "last_stall_ms": s.last_stall_ms,
                    "since_last_stall_secs": s.since_last_stall.map(|d| d.as_secs()),
                    "state_resets": s.state_resets,
                    "failed": s.failed,
                }),
            },
        );
        m
    }
}

/// Record that `id`'s decoder has failed; the first reason sticks.
fn record_failure(failed: &Mutex<Option<String>>, id: &str, reason: &str) {
    let mut failed = failed.lock().unwrap_or_else(PoisonError::into_inner);
    if failed.is_none() {
        tracing::error!(model = %id, "open chat model failed: {reason}; restart sidekickd to recover");
        *failed = Some(reason.to_string());
    }
}

fn unavailable(id: &str, reason: &str) -> Error {
    Error::Unavailable(UnavailableReason::Other(format!(
        "`{id}` has failed and can't serve until sidekickd restarts: {reason}"
    )))
}

fn log(id: &str, g: &Generated) {
    let decoded = g.completion_tokens.saturating_sub(1);
    let rate = (decoded > 0 && !g.decode.is_zero()).then(|| decoded as f64 / g.decode.as_secs_f64());
    let guided = g.guided.as_ref();
    let mask_mean_us = guided.and_then(|s| (s.masks > 0).then(|| s.mask_total.as_micros() as u64 / u64::from(s.masks)));
    tracing::info!(
        model = %id,
        prompt_tokens = g.prompt_tokens,
        completion_tokens = g.completion_tokens,
        calls = g.calls,
        first_token_ms = g.first_token.as_millis() as u64,
        decode_ms = g.decode.as_millis() as u64,
        tokens_per_sec = rate.map(|r| (r * 10.0).round() / 10.0),
        finish = ?g.finish,
        guided = guided.is_some(),
        forced = guided.map(|s| s.forced),
        grammar_compiled = guided.map(|s| s.compiled),
        mask_mean_us,
        mask_max_us = guided.map(|s| s.mask_max.as_micros() as u64),
        "generated"
    );
}

/// A job on the decoder thread. The lane stays taken until the thread has
/// finished it: when the request stops waiting first (a timeout, or a
/// client that went away), the job is told to stop and a task keeps the
/// lane until it has, or until its decoder is found to have failed, which
/// it records on the model: the hang outlives the request that met it, and
/// an idle unload of the model.
struct InFlight {
    permit: Option<LanePermit>,
    result: Option<oneshot::Receiver<Result<Generated>>>,
    cancel: Arc<AtomicBool>,
    probe: Arc<dyn DecoderProbe>,
    failed: Arc<Mutex<Option<String>>>,
    model: String,
}

impl InFlight {
    fn finish(&mut self) {
        self.result = None;
        self.permit = None;
    }
}

impl Drop for InFlight {
    fn drop(&mut self) {
        let (Some(mut result), Some(permit)) = (self.result.take(), self.permit.take()) else { return };
        self.cancel.store(true, Ordering::Relaxed);
        let (probe, failed, model) = (self.probe.clone(), self.failed.clone(), self.model.clone());
        let Ok(runtime) = tokio::runtime::Handle::try_current() else { return };
        runtime.spawn(async move {
            let mut tick = tokio::time::interval(Duration::from_millis(250));
            loop {
                tokio::select! {
                    _ = &mut result => break,
                    _ = tick.tick() => {
                        if let Some(reason) = probe.status().failed {
                            record_failure(&failed, &model, &reason);
                            break;
                        }
                    }
                }
            }
            drop(permit);
        });
    }
}

#[async_trait::async_trait]
impl ChatBackend for OpenModel {
    fn id(&self) -> &str {
        &self.spec.id
    }

    fn context_limit(&self) -> Option<usize> {
        Some(self.spec.context_length)
    }

    async fn availability(&self) -> Availability {
        match self.failed() {
            None => Availability::Available,
            Some(reason) => Availability::unavailable(UnavailableReason::Other(reason)),
        }
    }

    async fn complete(&self, req: ChatRequest) -> Result<ChatResponse> {
        self.run(req, None).await
    }

    async fn complete_stream(&self, req: ChatRequest, sink: DeltaSink) -> Result<ChatResponse> {
        self.run(req, Some(sink)).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decoder::{Decoder, DecoderSpec};
    use crate::generate::tests::bytes;
    use crate::lane::{LaneOptions, LanedChat};
    use crate::prompt::tests::prompter;
    use sidekick_core::{ChatMessage, FinishReason, Role};
    use std::sync::atomic::AtomicUsize;

    /// A decoder that replies with `reply`'s bytes then `</s>`, whatever it
    /// is asked, waiting `delay` per call. It can be made to hang, its probe
    /// then reporting it failed, as the watchdog would.
    struct Fake {
        reply: Vec<u32>,
        delay: Duration,
        prompt_len: usize,
        logits: Vec<f32>,
        hang: Arc<AtomicBool>,
        /// How long after a hang starts the watchdog reports it.
        watchdog: Duration,
        probe: Arc<FakeProbe>,
        /// Decoders alive at once, across the models sharing it.
        alive: Arc<AtomicUsize>,
    }

    impl Drop for Fake {
        fn drop(&mut self) {
            self.alive.fetch_sub(1, Ordering::SeqCst);
        }
    }

    #[derive(Default)]
    struct FakeProbe {
        failed: Mutex<Option<String>>,
        calls: AtomicU64,
    }

    impl DecoderProbe for FakeProbe {
        fn status(&self) -> DecoderStatus {
            DecoderStatus {
                failed: self.failed.lock().unwrap().clone(),
                calls: self.calls.load(Ordering::SeqCst),
                ..Default::default()
            }
        }
    }

    impl Fake {
        fn call(&mut self, k: usize) -> &[f32] {
            self.probe.calls.fetch_add(1, Ordering::SeqCst);
            std::thread::sleep(self.delay);
            if self.hang.load(Ordering::SeqCst) {
                let (probe, after) = (self.probe.clone(), self.watchdog);
                std::thread::spawn(move || {
                    std::thread::sleep(after);
                    *probe.failed.lock().unwrap() = Some("a call hung".into());
                });
                std::thread::sleep(Duration::from_secs(3600));
            }
            self.logits.fill(0.0);
            self.logits[self.reply.get(k).copied().unwrap_or(257) as usize] = 1.0;
            &self.logits
        }
    }

    impl Decoder for Fake {
        fn spec(&self) -> DecoderSpec {
            DecoderSpec { tokens_per_call: 16, context_length: 64, vocab_size: 300 }
        }
        fn prefill(&mut self, tokens: &[u32]) -> Result<&[f32]> {
            self.prompt_len = tokens.len();
            Ok(self.call(0))
        }
        fn step(&mut self, tokens: &[u32], pos: usize) -> Result<&[f32]> {
            let k = pos + tokens.len() - self.prompt_len;
            Ok(self.call(k))
        }
        fn probe(&self) -> Arc<dyn DecoderProbe> {
            self.probe.clone()
        }
        fn ane_bytes(&self) -> Option<u64> {
            Some(1234)
        }
    }

    struct Setup {
        loads: Arc<AtomicUsize>,
        hang: Arc<AtomicBool>,
    }

    /// More of a fake model's behavior.
    #[derive(Clone, Default)]
    struct Knobs {
        load_delay: Duration,
        watchdog: Duration,
        alive: Arc<AtomicUsize>,
        most_alive: Arc<AtomicUsize>,
        /// Leave guided requests' schemas out of their prompts.
        no_schema_in_prompt: bool,
    }

    fn model(id: &str, lane: &Arc<ChatLane>, reply: &str, delay: Duration, timeout: Duration) -> (OpenModel, Setup) {
        model_with(id, lane, reply, delay, timeout, Knobs::default())
    }

    fn model_with(id: &str, lane: &Arc<ChatLane>, reply: &str, delay: Duration, timeout: Duration, knobs: Knobs) -> (OpenModel, Setup) {
        let loads = Arc::new(AtomicUsize::new(0));
        let hang = Arc::new(AtomicBool::new(false));
        let setup = Setup { loads: loads.clone(), hang: hang.clone() };
        let reply = bytes(reply);
        let loader: DecoderLoader = Arc::new(move || {
            loads.fetch_add(1, Ordering::SeqCst);
            std::thread::sleep(knobs.load_delay);
            let now = knobs.alive.fetch_add(1, Ordering::SeqCst) + 1;
            knobs.most_alive.fetch_max(now, Ordering::SeqCst);
            Ok(Box::new(Fake {
                reply: reply.clone(),
                delay,
                prompt_len: 0,
                logits: vec![0.0; 300],
                hang: hang.clone(),
                watchdog: knobs.watchdog,
                probe: Arc::default(),
                alive: knobs.alive.clone(),
            }) as Box<dyn Decoder>)
        });
        let spec = OpenModelSpec {
            id: id.into(),
            context_length: 64,
            prompter: Arc::new(prompter()),
            stop_token_ids: vec![257, 259],
            vocab_size: 300,
            schema_in_prompt: !knobs.no_schema_in_prompt,
            sampling: SamplingSection { temperature: 0.0, top_p: 1.0 },
            estimated_ane_bytes: 1000,
            queue_depth: 4,
        };
        (OpenModel::new(spec, lane.clone(), loader, timeout, Duration::from_secs(30)), setup)
    }

    fn lane(depth: usize) -> Arc<ChatLane> {
        ChatLane::new(LaneOptions { queue_depth: depth, ..Default::default() })
    }

    /// Wait (up to 10 s, for slow CI machines) until `ready` holds.
    async fn until(ready: impl Fn() -> bool) {
        let started = std::time::Instant::now();
        while !ready() {
            assert!(started.elapsed() < Duration::from_secs(10), "timed out waiting");
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    }

    fn ask(text: &str, max_tokens: Option<u32>) -> ChatRequest {
        ChatRequest { messages: vec![ChatMessage::new(Role::User, text)], max_tokens, ..Default::default() }
    }

    #[tokio::test]
    async fn answers_with_exact_usage_and_loads_once() {
        let lane = lane(4);
        let (m, setup) = model("m", &lane, "Hello!", Duration::ZERO, Duration::from_secs(10));
        let prompt = prompter().encode(&ask("Hi", None).messages).unwrap().len() as u32;
        let r = m.complete(ask("Hi", None)).await.unwrap();
        assert_eq!((r.content.as_str(), r.finish), ("Hello!", FinishReason::Stop));
        assert_eq!((r.usage.prompt_tokens, r.usage.completion_tokens, r.usage.cached_tokens), (prompt, 7, Some(0)));
        let r = m.complete(ask("Hi", Some(3))).await.unwrap();
        assert_eq!((r.content.as_str(), r.finish, r.usage.completion_tokens), ("Hel", FinishReason::Length, 3));
        assert_eq!(setup.loads.load(Ordering::SeqCst), 1, "resident after the first request");
        let h = m.health();
        assert_eq!((h["state"].as_str(), h["requests"].as_u64()), (Some("ready"), Some(2)));
        assert_eq!(h["ane"]["measured_bytes"], 1234);
        assert_eq!(h["decoder"]["calls"].as_u64(), Some(7 + 3));
    }

    #[tokio::test]
    async fn streams_and_refuses_what_it_cant_honor() {
        let lane = lane(4);
        let (m, _) = model("m", &lane, "a b", Duration::ZERO, Duration::from_secs(10));
        let seen = Arc::new(Mutex::new(String::new()));
        let into = seen.clone();
        let sink: DeltaSink = Box::new(move |d: &str| {
            into.lock().unwrap().push_str(d);
            true
        });
        let r = m.complete_stream(ask("Hi", None), sink).await.unwrap();
        assert_eq!((r.content.as_str(), seen.lock().unwrap().as_str()), ("a b", "a b"));
        // The context: prompt plus max_tokens.
        let err = m.complete(ask("Hi", Some(60))).await.unwrap_err();
        assert!(matches!(err, Error::ContextOverflow { limit: 64, actual: Some(_) }), "{err}");
        let err = m.complete(ask(&"x".repeat(80), None)).await.unwrap_err();
        assert!(matches!(err, Error::ContextOverflow { limit: 64, .. }), "{err}");
        // Guided output: the schema written into the prompt counts against
        // the context (64 tokens here, one a byte).
        let schema = sidekick_core::JsonSchema::parse(&serde_json::json!({"type": "object", "properties": {}})).unwrap();
        let guided = |text: &str| ChatRequest { schema: Some(schema.clone()), ..ask(text, None) };
        let err = m.complete(guided("Hi")).await.unwrap_err();
        let unguided = prompter().encode(&ask("Hi", None).messages).unwrap().len();
        let with_schema = prompter()
            .encode(&crate::guided::with_schema_in_prompt(&ask("Hi", None).messages, &schema))
            .unwrap()
            .len();
        assert!(matches!(err, Error::ContextOverflow { limit: 64, actual: Some(a) } if a == with_schema + 1), "{err}");
        assert!(with_schema > unguided + crate::guided::schema_instruction(&schema).len() - 8);
        // Without it, an empty object's grammar forces the whole reply.
        let knobs = Knobs { no_schema_in_prompt: true, ..Default::default() };
        let (m, _) = model_with("m2", &lane, "a b", Duration::ZERO, Duration::from_secs(10), knobs);
        let r = m.complete(guided("Hi")).await.unwrap();
        assert_eq!((r.content.as_str(), r.finish, r.usage.completion_tokens, r.constrained), ("{}", FinishReason::Stop, 2, true));
        assert_eq!(r.usage.prompt_tokens as usize, unguided);
        assert_eq!(m.health()["schema_in_prompt"], false);
        assert_eq!(m.health()["guided"]["grammars"], 1);
    }

    #[tokio::test]
    async fn a_full_queue_is_busy_and_waiting_counts_against_the_timeout() {
        let lane = lane(1);
        // A generous timeout: the second request waits out the first, and
        // macOS CI runners can take several times longer than a laptop.
        let (m, _) = model("m", &lane, "abcdef", Duration::from_millis(40), Duration::from_secs(10));
        let m = Arc::new(m);
        let first = tokio::spawn({
            let m = m.clone();
            async move { m.complete(ask("1", None)).await }
        });
        until(|| !lane.health()["running"].is_null()).await;
        let second = tokio::spawn({
            let m = m.clone();
            async move { m.complete(ask("2", None)).await }
        });
        until(|| lane.health()["waiting"] == 1).await;
        let err = m.complete(ask("3", None)).await.unwrap_err();
        assert!(matches!(err, Error::Busy { retry_after_secs: 5, .. }), "{err}");
        assert_eq!(lane.health()["waiting"], 1);
        assert_eq!(first.await.unwrap().unwrap().content, "abcdef");
        assert_eq!(second.await.unwrap().unwrap().content, "abcdef");
        // A request that outlasts its timeout is a timeout, and the lane
        // stays taken until the decoder has stopped.
        let (slow, _) = model("slow", &lane, &"z".repeat(40), Duration::from_millis(50), Duration::from_millis(300));
        let err = slow.complete(ask("4", None)).await.unwrap_err();
        assert!(matches!(err, Error::Timeout { .. }), "{err}");
        let r = m.complete(ask("5", None)).await.unwrap();
        assert_eq!(r.content, "abcdef");
    }

    /// A Foundation Models stand-in that records when it ran.
    struct Afm(Arc<Mutex<Vec<&'static str>>>);

    #[async_trait::async_trait]
    impl ChatBackend for Afm {
        fn id(&self) -> &str {
            "apple-fm"
        }
        async fn availability(&self) -> Availability {
            Availability::Available
        }
        async fn complete(&self, _: ChatRequest) -> Result<ChatResponse> {
            self.0.lock().unwrap().push("afm");
            Ok(ChatResponse { content: "fm".into(), finish: FinishReason::Stop, usage: Usage::default(), constrained: false })
        }
    }

    #[tokio::test]
    async fn one_open_model_is_resident_and_apple_fm_shares_the_lane() {
        let lane = lane(4);
        let (a, sa) = model("a", &lane, "A", Duration::ZERO, Duration::from_secs(10));
        let (b, sb) = model("b", &lane, "B", Duration::from_millis(100), Duration::from_secs(10));
        assert_eq!(a.complete(ask("x", None)).await.unwrap().content, "A");
        assert_eq!(b.complete(ask("x", None)).await.unwrap().content, "B");
        assert_eq!(a.complete(ask("x", None)).await.unwrap().content, "A");
        assert_eq!((sa.loads.load(Ordering::SeqCst), sb.loads.load(Ordering::SeqCst)), (2, 1));
        assert_eq!(b.health()["state"], "unloaded");
        assert_eq!(b.health()["decoder"]["live"], false, "as of its last request");

        // apple-fm waits while an open model generates.
        let order = Arc::new(Mutex::new(Vec::new()));
        let afm = LanedChat::new(Arc::new(Afm(order.clone())), lane.clone(), Duration::from_secs(10));
        let b = Arc::new(b);
        let generation = tokio::spawn({
            let (b, order) = (b.clone(), order.clone());
            async move {
                let r = b.complete(ask("x", None)).await;
                order.lock().unwrap().push("open");
                r
            }
        });
        until(|| lane.health()["running"] == "b").await;
        assert_eq!(afm.complete(ask("y", None)).await.unwrap().content, "fm");
        generation.await.unwrap().unwrap();
        assert_eq!(*order.lock().unwrap(), ["open", "afm"]);
    }

    #[tokio::test]
    async fn a_hung_decoder_fails_the_model_and_frees_the_lane() {
        let lane = lane(4);
        let (m, setup) = model("m", &lane, "abc", Duration::ZERO, Duration::from_millis(500));
        assert_eq!(m.complete(ask("x", None)).await.unwrap().content, "abc");
        setup.hang.store(true, Ordering::SeqCst);
        let err = m.complete(ask("x", None)).await.unwrap_err();
        assert!(matches!(err, Error::Unavailable(_) | Error::Timeout { .. }), "{err}");
        // Later requests fail at once with the reason, and /health says so.
        let err = m.complete(ask("x", None)).await.unwrap_err();
        assert!(matches!(&err, Error::Unavailable(UnavailableReason::Other(r)) if r.contains("hung")), "{err}");
        assert_eq!(m.health()["state"], "failed");
        assert!(!m.availability().await.is_available());
        // The hung call doesn't hold the lane: another model runs.
        let (other, _) = model("other", &lane, "ok", Duration::ZERO, Duration::from_secs(5));
        assert_eq!(other.complete(ask("x", None)).await.unwrap().content, "ok");
    }

    #[tokio::test]
    async fn a_model_switch_waits_for_the_old_decoder_to_go() {
        let lane = lane(4);
        // Decoders drop slowly: each call sleeps, and the old one finishes
        // its job before the switch.
        let knobs = Knobs { load_delay: Duration::from_millis(30), ..Default::default() };
        let (a, _) = model_with("a", &lane, "A", Duration::from_millis(20), Duration::from_secs(10), knobs.clone());
        let (b, _) = model_with("b", &lane, "B", Duration::from_millis(20), Duration::from_secs(10), knobs.clone());
        for m in [&a, &b, &a, &b] {
            m.complete(ask("x", None)).await.unwrap();
        }
        assert_eq!(knobs.most_alive.load(Ordering::SeqCst), 1, "never two decoders at once");
        lane.unload_idle_now();
        until(|| knobs.alive.load(Ordering::SeqCst) == 0).await;
    }

    #[tokio::test]
    async fn a_model_still_loading_isnt_idle() {
        let lane = ChatLane::new(LaneOptions { idle_ttl: Duration::from_millis(20), ..Default::default() });
        let knobs = Knobs { load_delay: Duration::from_millis(300), ..Default::default() };
        let (m, setup) = model_with("m", &lane, "ok", Duration::ZERO, Duration::from_secs(10), knobs);
        // The sweep runs every 10 ms through the load; unloading it then
        // would stop its thread before the job arrives.
        assert_eq!(m.complete(ask("x", None)).await.unwrap().content, "ok");
        assert_eq!(setup.loads.load(Ordering::SeqCst), 1);
        // Idle after its request, it goes.
        until(|| m.health()["state"] == "unloaded").await;
    }

    #[tokio::test]
    async fn a_hang_found_after_the_request_timed_out_still_fails_the_model() {
        let lane = ChatLane::new(LaneOptions { idle_ttl: Duration::from_millis(50), ..Default::default() });
        // The request times out at 200 ms; the watchdog reports at 600 ms.
        let knobs = Knobs { watchdog: Duration::from_millis(600), ..Default::default() };
        let (m, setup) = model_with("m", &lane, "abc", Duration::ZERO, Duration::from_millis(200), knobs);
        setup.hang.store(true, Ordering::SeqCst);
        let err = m.complete(ask("x", None)).await.unwrap_err();
        assert!(matches!(err, Error::Timeout { .. }), "{err}");
        assert_eq!(m.health()["state"], "ready", "not known yet");
        until(|| m.health()["state"] == "failed").await;
        // The failure outlives the model's unloading.
        lane.unload_idle_now();
        assert_eq!(m.health()["state"], "failed");
        let err = m.complete(ask("x", None)).await.unwrap_err();
        assert!(matches!(&err, Error::Unavailable(UnavailableReason::Other(r)) if r.contains("hung")), "{err}");
    }

    /// A backend whose work outlives its reply, as Foundation Models'
    /// blocking thread outlives a timeout: it keeps the request's `hold`
    /// for `linger` after answering.
    struct Lingering {
        linger: Duration,
        deadlines: Arc<Mutex<Vec<bool>>>,
    }

    #[async_trait::async_trait]
    impl ChatBackend for Lingering {
        fn id(&self) -> &str {
            "apple-fm"
        }
        async fn availability(&self) -> Availability {
            Availability::Available
        }
        async fn complete(&self, req: ChatRequest) -> Result<ChatResponse> {
            self.deadlines.lock().unwrap().push(req.deadline.is_some());
            let (hold, linger) = (req.hold, self.linger);
            std::thread::spawn(move || {
                std::thread::sleep(linger);
                drop(hold);
            });
            Err(Error::Timeout { secs: 1 })
        }
    }

    #[tokio::test]
    async fn apple_fm_holds_the_lane_until_its_work_has_stopped() {
        let lane = lane(4);
        let deadlines = Arc::new(Mutex::new(Vec::new()));
        let afm = LanedChat::new(
            Arc::new(Lingering { linger: Duration::from_millis(400), deadlines: deadlines.clone() }),
            lane.clone(),
            Duration::from_secs(10),
        );
        let (m, _) = model("m", &lane, "ok", Duration::ZERO, Duration::from_secs(10));
        let started = std::time::Instant::now();
        assert!(afm.complete(ask("y", None)).await.is_err());
        assert_eq!(*deadlines.lock().unwrap(), [true], "the backend is given the lane's deadline");
        // The open model waits for apple-fm's abandoned work to finish.
        assert_eq!(m.complete(ask("x", None)).await.unwrap().content, "ok");
        assert!(started.elapsed() >= Duration::from_millis(400), "{:?}", started.elapsed());
    }

    #[tokio::test]
    async fn a_failed_load_is_reported_and_retried() {
        let lane = lane(4);
        let tries = Arc::new(AtomicUsize::new(0));
        let counter = tries.clone();
        let loader: DecoderLoader = Arc::new(move || {
            counter.fetch_add(1, Ordering::SeqCst);
            Err(Error::Inference("refusing decoder `m`: not on the ANE".into()))
        });
        let spec = OpenModelSpec {
            id: "m".into(),
            context_length: 64,
            prompter: Arc::new(prompter()),
            stop_token_ids: vec![257],
            vocab_size: 300,
            schema_in_prompt: true,
            sampling: SamplingSection::default(),
            estimated_ane_bytes: 0,
            queue_depth: 4,
        };
        let m = OpenModel::new(spec, lane.clone(), loader, Duration::from_secs(5), Duration::from_secs(5));
        assert_eq!(m.health()["queue_depth"], 4);
        let err = m.complete(ask("x", None)).await.unwrap_err();
        assert!(matches!(&err, Error::Inference(msg) if msg.contains("not on the ANE")), "{err}");
        assert!(m.complete(ask("x", None)).await.is_err());
        assert_eq!(tries.load(Ordering::SeqCst), 2);
        let h = m.health();
        assert_eq!(h["state"], "load_failed");
        assert!(h["load_error"].as_str().unwrap().contains("not on the ANE"), "{h:?}");
    }
}
