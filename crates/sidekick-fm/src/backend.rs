use crate::cache::{conversation_key, ConversationCache};
use crate::engine::{EngineUsage, RespondOptions, SessionEngine};
use crate::shaping::{shape, Diverged, StreamShaper};
use sidekick_core::{
    Availability, ChatBackend, ChatMessage, ChatRequest, ChatResponse, DeltaSink, Error,
    FinishReason, ModelInfo, Result, Role, Usage,
};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

/// Context budget assumed until the model reports its own: macOS 26's fixed
/// on-device budget (also AFM 3 Core's on macOS 27).
const DEFAULT_CONTEXT_LIMIT: usize = 4096;

/// `ChatBackend` over any [`SessionEngine`], with prefix-keyed session reuse.
pub struct SessionChatBackend<E: SessionEngine> {
    engine: Arc<E>,
    cache: Arc<Mutex<ConversationCache<E::Session>>>,
    request_timeout: Duration,
    /// Last context size the model reported (0 = not yet known). Refreshed
    /// on every `model_info()` call, so a model asset update is picked up.
    context_size: AtomicUsize,
}

/// Usage from the engine when it reports it (macOS 27), otherwise an
/// estimate at ~4 bytes/token, rounded up so a non-empty text never counts
/// as zero tokens.
fn usage(real: Option<EngineUsage>, messages: &[ChatMessage], reply: &str) -> Usage {
    match real {
        Some(u) => Usage {
            prompt_tokens: u.input,
            completion_tokens: u.output,
            cached_tokens: Some(u.cached),
            reasoning_tokens: (u.reasoning > 0).then_some(u.reasoning),
        },
        None => {
            let estimate = |bytes: usize| bytes.div_ceil(4) as u32;
            Usage {
                prompt_tokens: estimate(messages.iter().map(|m| m.content.len()).sum()),
                completion_tokens: estimate(reply.len()),
                cached_tokens: None,
                reasoning_tokens: None,
            }
        }
    }
}

/// A runtime failure worth one retry on a fresh session: transient conditions
/// where the model runtime was busy, interrupted, or timed out internally.
/// The shim classifies these by type (`Error::Transient`); the keyword set
/// is a fallback for unclassified errors, carried over from real-hardware
/// failures observed in the predecessor project. "cancelled" is deliberately
/// absent: a cancellation is intentional and must not be retried. Context
/// overflow isn't recoverable either — a fresh session cannot make the input
/// smaller.
fn is_recoverable(err: &Error) -> bool {
    let message = match err {
        Error::Transient(_) => return true,
        Error::Inference(m) => m.to_lowercase(),
        _ => return false,
    };
    ["timeout", "timed out", "resource", "busy", "interrupted"]
        .iter()
        .any(|k| message.contains(k))
}

impl<E: SessionEngine> SessionChatBackend<E> {
    pub fn new(engine: E, session_ttl: Duration, request_timeout: Duration) -> Self {
        Self {
            engine: Arc::new(engine),
            cache: Arc::new(Mutex::new(ConversationCache::new(session_ttl, 8))),
            request_timeout,
            context_size: AtomicUsize::new(0),
        }
    }

    /// System messages become session instructions; the rest is the dialogue.
    fn split(messages: &[ChatMessage]) -> (String, Vec<ChatMessage>) {
        let instructions = messages
            .iter()
            .filter(|m| m.role == Role::System)
            .map(|m| m.content.as_str())
            .collect::<Vec<_>>()
            .join("\n\n");
        let history: Vec<ChatMessage> = messages
            .iter()
            .filter(|m| m.role != Role::System)
            .cloned()
            .collect();
        (instructions, history)
    }

    /// Cold-start prompt: replay prior turns as labeled transcript text, then
    /// the new user message. Single-turn requests pass through untouched.
    fn replay_prompt(history: &[ChatMessage]) -> String {
        if history.len() == 1 {
            return history[0].content.clone();
        }
        let mut out = String::from(
            "Continue this conversation. Reply to the last user message only, \
             without any speaker label. Prior turns:\n\n",
        );
        for m in &history[..history.len() - 1] {
            let label = match m.role {
                Role::User => "User",
                Role::Assistant => "Assistant",
                Role::System => continue,
            };
            out.push_str(label);
            out.push_str(": ");
            out.push_str(&m.content);
            out.push_str("\n\n");
        }
        out.push_str("User: ");
        out.push_str(&history[history.len() - 1].content);
        out
    }

    /// Validate a request and take the session to answer it on: the cached
    /// session for this conversation prefix (the prompt is then just the new
    /// user message), or a fresh one (the prompt replays the history).
    fn begin(
        engine: &E,
        cache: &Mutex<ConversationCache<E::Session>>,
        messages: &[ChatMessage],
    ) -> Result<Turn<E::Session>> {
        let (instructions, history) = Self::split(messages);
        let last = history.last().ok_or_else(|| {
            Error::Other("chat request must contain at least one non-system message".into())
        })?;
        if last.role != Role::User {
            return Err(Error::Other("last message must have role `user`".into()));
        }
        let prefix = conversation_key(&instructions, &history[..history.len() - 1]);
        let cached = cache.lock().unwrap().take(&prefix);
        let (session, prompt) = match cached {
            Some(session) => (session, last.content.clone()),
            None => (engine.create(&instructions)?, Self::replay_prompt(&history)),
        };
        Ok(Turn { instructions, history, session, prompt })
    }

    /// File a session under the conversation extended by `reply`, so a
    /// follow-up that sends exactly that history resumes it.
    fn file_session(
        cache: &Mutex<ConversationCache<E::Session>>,
        turn: Turn<E::Session>,
        reply: &str,
    ) {
        let mut extended = turn.history;
        extended.push(ChatMessage::new(Role::Assistant, reply));
        let key = conversation_key(&turn.instructions, &extended);
        cache.lock().unwrap().insert(key, turn.session);
    }

    fn run_sync(
        engine: &E,
        cache: &Mutex<ConversationCache<E::Session>>,
        req: ChatRequest,
    ) -> Result<ChatResponse> {
        let mut turn = Self::begin(engine, cache, &req.messages)?;
        let constrained = req.schema.is_some();
        let opts = RespondOptions {
            schema: req.schema,
            temperature: req.temperature,
            max_tokens: req.max_tokens,
        };

        // A `LanguageModelSession` that failed mid-generation may be left in
        // a bad state, so an errored session is always dropped (it was taken
        // out of the cache and is never re-filed). Transient failures get one
        // retry on a fresh session, which needs the full replay prompt since
        // it has no history.
        let response = match engine.respond(&mut turn.session, &turn.prompt, &opts) {
            Ok(response) => response,
            Err(e) if is_recoverable(&e) => {
                turn.session = engine.create(&turn.instructions)?;
                turn.prompt = Self::replay_prompt(&turn.history);
                engine.respond(&mut turn.session, &turn.prompt, &opts)?
            }
            Err(e) => return Err(e),
        };
        let shaped = shape(&response.text, &req.stop);

        // Keep the session for follow-ups — unless a stop sequence cut the
        // reply: its transcript then holds text the client never saw, so it
        // must not be resumed. (A max_tokens cut is fine: the transcript is
        // exactly the reply.)
        if !shaped.stop_hit {
            Self::file_session(cache, turn, &shaped.text);
        }
        Ok(ChatResponse {
            usage: usage(response.usage, &req.messages, &shaped.text),
            finish: finish_reason(shaped.stop_hit, response.truncated),
            content: shaped.text,
            constrained,
        })
    }

    /// Streamed plain-text turn. Deltas go to `sink` as soon as no later
    /// snapshot can change them (see [`StreamShaper`]). Generation stops
    /// early when a stop sequence appears, when `sink` reports the client
    /// gone, or when `cancel` is set (request timeout).
    ///
    /// Rules that keep this safe:
    /// - No retry once anything was sent: the client would see the reply
    ///   start over.
    /// - A session whose generation was stopped early, errored, or whose
    ///   final text disagrees with what was sent is dropped, never reused:
    ///   on macOS 27, using an interrupted session again traps the process.
    /// - Only a stream that completed on its own, with the client holding
    ///   exactly the final reply, is filed for follow-ups.
    fn run_stream_sync(
        engine: &E,
        cache: &Mutex<ConversationCache<E::Session>>,
        req: ChatRequest,
        sink: &mut (dyn FnMut(&str) -> bool + Send),
        cancel: &AtomicBool,
    ) -> Result<ChatResponse> {
        let mut turn = Self::begin(engine, cache, &req.messages)?;
        let opts = RespondOptions {
            schema: None,
            temperature: req.temperature,
            max_tokens: req.max_tokens,
        };

        let mut retried = false;
        let (streamed, shaper, halted) = loop {
            let mut shaper = StreamShaper::new(&req.stop);
            let mut sent_any = false;
            let mut halted = false;
            let result = {
                let mut on_snapshot = |snapshot: &str| -> bool {
                    if cancel.load(Ordering::Relaxed) {
                        halted = true;
                        return false;
                    }
                    let delta = shaper.push(snapshot);
                    if !delta.is_empty() {
                        sent_any = true;
                        if !sink(&delta) {
                            halted = true;
                            return false;
                        }
                    }
                    if shaper.stop_hit() {
                        halted = true;
                        return false;
                    }
                    true
                };
                engine.respond_stream(&mut turn.session, &turn.prompt, &opts, &mut on_snapshot)
            };
            match result {
                Ok(streamed) => break (streamed, shaper, halted),
                Err(e) if !retried && !sent_any && !halted && is_recoverable(&e) => {
                    retried = true;
                    turn.session = engine.create(&turn.instructions)?;
                    turn.prompt = Self::replay_prompt(&turn.history);
                }
                Err(e) => return Err(e),
            }
        };
        let mut shaper = shaper;

        if halted && !shaper.stop_hit() {
            // Client gone or timed out; the interrupted session is dropped.
            tracing::debug!(
                sent_chars = shaper.emitted().len(),
                "stopped generation early: the client went away or the request timed out"
            );
            return Err(Error::Other("generation stopped before completion".into()));
        }
        if !halted && !streamed.cancelled {
            match shaper.finish(&streamed.response.text) {
                Ok(tail) => {
                    if !tail.is_empty() {
                        sink(&tail);
                    }
                }
                Err(Diverged) => {
                    return Err(Error::Inference(
                        "the streamed reply was rewritten after part of it was sent".into(),
                    ))
                }
            }
        }
        let content = shaper.emitted().to_string();
        let stop_hit = shaper.stop_hit();
        if !halted && !streamed.cancelled && !stop_hit {
            Self::file_session(cache, turn, &content);
        }
        Ok(ChatResponse {
            usage: usage(streamed.response.usage, &req.messages, &content),
            finish: finish_reason(stop_hit, streamed.response.truncated),
            content,
            constrained: false,
        })
    }

    /// Run a streamed turn on a blocking thread, bounded by the request
    /// timeout. On timeout the generation is told to stop at its next
    /// snapshot (it can't be interrupted before the first one).
    async fn stream(&self, req: ChatRequest, sink: DeltaSink) -> Result<ChatResponse> {
        let engine = self.engine.clone();
        let cache = self.cache.clone();
        let cancel = Arc::new(AtomicBool::new(false));
        let flag = cancel.clone();
        let task = tokio::task::spawn_blocking(move || {
            let mut sink = sink;
            Self::run_stream_sync(&engine, &cache, req, &mut *sink, &flag)
        });
        match tokio::time::timeout(self.request_timeout, task).await {
            Ok(joined) => {
                joined.map_err(|e| Error::Other(format!("blocking task failed: {e}")))?
            }
            Err(_) => {
                cancel.store(true, Ordering::Relaxed);
                Err(Error::Timeout { secs: self.request_timeout.as_secs() })
            }
        }
    }
}

/// A turn in progress: the session it runs on and what to send it.
struct Turn<S> {
    instructions: String,
    history: Vec<ChatMessage>,
    session: S,
    prompt: String,
}

/// A stop hit is a normal stop even if the token limit was also reached.
fn finish_reason(stop_hit: bool, truncated: Option<bool>) -> FinishReason {
    if !stop_hit && truncated == Some(true) {
        FinishReason::Length
    } else {
        FinishReason::Stop
    }
}

#[async_trait::async_trait]
impl<E: SessionEngine> ChatBackend for SessionChatBackend<E> {
    fn id(&self) -> &str {
        "apple-fm"
    }

    fn context_limit(&self) -> Option<usize> {
        match self.context_size.load(Ordering::Relaxed) {
            0 => Some(DEFAULT_CONTEXT_LIMIT),
            known => Some(known),
        }
    }

    async fn model_info(&self) -> Option<ModelInfo> {
        let engine = self.engine.clone();
        let info = tokio::task::spawn_blocking(move || engine.model_info()).await.ok().flatten();
        if let Some(size) = info.as_ref().and_then(|i| i.context_size) {
            self.context_size.store(size, Ordering::Relaxed);
        }
        info
    }

    async fn availability(&self) -> Availability {
        let engine = self.engine.clone();
        tokio::task::spawn_blocking(move || engine.availability())
            .await
            .unwrap_or_else(|e| {
                Availability::unavailable(sidekick_core::UnavailableReason::Other(e.to_string()))
            })
    }

    async fn complete(&self, req: ChatRequest) -> Result<ChatResponse> {
        // With stop sequences, stream internally so generation ends at the
        // stop instead of running to completion and being cut afterwards.
        if !req.stop.is_empty() && req.schema.is_none() {
            return self.stream(req, Box::new(|_| true)).await;
        }
        let engine = self.engine.clone();
        let cache = self.cache.clone();
        let task = tokio::task::spawn_blocking(move || Self::run_sync(&engine, &cache, req));
        // The timeout abandons the *wait*, not the work: the blocking thread
        // (and the Swift task behind it) runs to completion and its session
        // is dropped when it finishes. That leaks a blocking-pool thread for
        // the duration of the hung call, which is bounded and preferable to
        // hanging the request forever.
        match tokio::time::timeout(self.request_timeout, task).await {
            Ok(joined) => {
                joined.map_err(|e| Error::Other(format!("blocking task failed: {e}")))?
            }
            Err(_) => Err(Error::Timeout { secs: self.request_timeout.as_secs() }),
        }
    }

    async fn complete_stream(&self, req: ChatRequest, mut sink: DeltaSink) -> Result<ChatResponse> {
        if req.schema.is_some() {
            // Constrained output is sent whole: partial JSON snapshots
            // aren't prefix-stable.
            let response = self.complete(req).await?;
            sink(&response.content);
            return Ok(response);
        }
        self.stream(req, sink).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::{EngineResponse, EngineUsage, StreamedResponse};
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// Mock engine that records prompts; sessions count their turns. Can be
    /// primed with errors to return before succeeding.
    struct MockEngine {
        creates: AtomicUsize,
        last_prompt: Mutex<String>,
        fail_with: Mutex<Vec<Error>>,
    }

    struct MockSession {
        turns: usize,
    }

    impl SessionEngine for MockEngine {
        type Session = MockSession;

        fn availability(&self) -> Availability {
            Availability::Available
        }

        fn create(&self, _instructions: &str) -> Result<MockSession> {
            self.creates.fetch_add(1, Ordering::SeqCst);
            Ok(MockSession { turns: 0 })
        }

        fn respond(
            &self,
            session: &mut MockSession,
            prompt: &str,
            opts: &RespondOptions,
        ) -> Result<EngineResponse> {
            session.turns += 1;
            *self.last_prompt.lock().unwrap() = prompt.to_string();
            if let Some(err) = self.fail_with.lock().unwrap().pop() {
                return Err(err);
            }
            if opts.schema.is_some() {
                Ok(EngineResponse::text(format!("{{\"turns\": {}}}", session.turns)))
            } else {
                Ok(EngineResponse::text(format!("reply-{}", session.turns)))
            }
        }
    }

    fn backend() -> SessionChatBackend<MockEngine> {
        SessionChatBackend::new(
            MockEngine {
                creates: AtomicUsize::new(0),
                last_prompt: Mutex::new(String::new()),
                fail_with: Mutex::new(Vec::new()),
            },
            Duration::from_secs(60),
            Duration::from_secs(60),
        )
    }

    fn req(messages: Vec<ChatMessage>) -> ChatRequest {
        ChatRequest { messages, ..Default::default() }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn single_turn_passes_prompt_through() {
        let b = backend();
        let r = b
            .complete(req(vec![
                ChatMessage::new(Role::System, "be brief"),
                ChatMessage::new(Role::User, "title this session"),
            ]))
            .await
            .unwrap();
        assert_eq!(r.content, "reply-1");
        assert_eq!(
            *b.engine.last_prompt.lock().unwrap(),
            "title this session",
            "single-turn prompt is not wrapped in replay scaffolding"
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn follow_up_reuses_cached_session() {
        let b = backend();
        let first = vec![ChatMessage::new(Role::User, "hi")];
        let r1 = b.complete(req(first.clone())).await.unwrap();

        // Extend exactly as an OpenAI client would.
        let mut second = first;
        second.push(ChatMessage::new(Role::Assistant, r1.content.clone()));
        second.push(ChatMessage::new(Role::User, "again"));
        let r2 = b.complete(req(second)).await.unwrap();

        assert_eq!(r2.content, "reply-2", "same session, turn count advanced");
        assert_eq!(b.engine.creates.load(Ordering::SeqCst), 1, "no second create");
        assert_eq!(*b.engine.last_prompt.lock().unwrap(), "again");
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn unrelated_history_gets_cold_replay() {
        let b = backend();
        b.complete(req(vec![ChatMessage::new(Role::User, "hi")]))
            .await
            .unwrap();

        let r = b
            .complete(req(vec![
                ChatMessage::new(Role::User, "one"),
                ChatMessage::new(Role::Assistant, "two"),
                ChatMessage::new(Role::User, "three"),
            ]))
            .await
            .unwrap();
        assert_eq!(r.content, "reply-1", "fresh session");
        assert_eq!(b.engine.creates.load(Ordering::SeqCst), 2);
        let prompt = b.engine.last_prompt.lock().unwrap().clone();
        assert!(prompt.contains("User: one") && prompt.contains("Assistant: two"));
        assert!(prompt.ends_with("User: three"));
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn rejects_history_not_ending_in_user() {
        let b = backend();
        let err = b
            .complete(req(vec![
                ChatMessage::new(Role::User, "hi"),
                ChatMessage::new(Role::Assistant, "hello"),
            ]))
            .await
            .unwrap_err();
        assert!(matches!(err, Error::Other(_)));
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn leading_assistant_label_is_stripped() {
        struct LabelingEngine;
        struct NoSession;
        impl SessionEngine for LabelingEngine {
            type Session = NoSession;
            fn availability(&self) -> Availability {
                Availability::Available
            }
            fn create(&self, _instructions: &str) -> Result<NoSession> {
                Ok(NoSession)
            }
            fn respond(
                &self,
                _session: &mut NoSession,
                _prompt: &str,
                _opts: &RespondOptions,
            ) -> Result<EngineResponse> {
                Ok(EngineResponse::text("Assistant: Paris has 2.1 million people."))
            }
        }

        let b = SessionChatBackend::new(
            LabelingEngine,
            Duration::from_secs(60),
            Duration::from_secs(60),
        );
        let r = b
            .complete(req(vec![ChatMessage::new(Role::User, "population?")]))
            .await
            .unwrap();
        assert_eq!(r.content, "Paris has 2.1 million people.");
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn typed_transient_error_retries_on_fresh_session() {
        let b = backend();
        b.engine
            .fail_with
            .lock()
            .unwrap()
            .push(Error::Transient("concurrentRequests".into()));
        let r = b.complete(req(vec![ChatMessage::new(Role::User, "hi")])).await.unwrap();
        assert_eq!(r.content, "reply-1");
        assert_eq!(b.engine.creates.load(Ordering::SeqCst), 2, "retried on a fresh session");
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn cancellation_is_not_retried() {
        let b = backend();
        b.engine
            .fail_with
            .lock()
            .unwrap()
            .push(Error::Inference("respond: CancellationError: cancelled".into()));
        let err = b
            .complete(req(vec![ChatMessage::new(Role::User, "hi")]))
            .await
            .unwrap_err();
        assert!(matches!(err, Error::Inference(_)));
        assert_eq!(b.engine.creates.load(Ordering::SeqCst), 1, "no retry session");
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn non_recoverable_error_propagates_without_retry() {
        let b = backend();
        b.engine
            .fail_with
            .lock()
            .unwrap()
            .push(Error::ContextOverflow { limit: 4096, actual: None });

        let err = b
            .complete(req(vec![ChatMessage::new(Role::User, "hi")]))
            .await
            .unwrap_err();
        assert!(matches!(err, Error::ContextOverflow { .. }));
        assert_eq!(b.engine.creates.load(Ordering::SeqCst), 1, "no retry session");
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn slow_generation_times_out() {
        struct SlowEngine;
        struct NoSession;
        impl SessionEngine for SlowEngine {
            type Session = NoSession;
            fn availability(&self) -> Availability {
                Availability::Available
            }
            fn create(&self, _instructions: &str) -> Result<NoSession> {
                Ok(NoSession)
            }
            fn respond(
                &self,
                _session: &mut NoSession,
                _prompt: &str,
                _opts: &RespondOptions,
            ) -> Result<EngineResponse> {
                // Short: runtime shutdown waits for started blocking tasks.
                std::thread::sleep(Duration::from_millis(400));
                Ok(EngineResponse::text("too late"))
            }
        }

        let b = SessionChatBackend::new(
            SlowEngine,
            Duration::from_secs(60),
            Duration::from_millis(50),
        );
        let err = b
            .complete(req(vec![ChatMessage::new(Role::User, "hi")]))
            .await
            .unwrap_err();
        assert!(matches!(err, Error::Timeout { .. }));
    }

    /// Engine returning a fixed response; counts sessions created and
    /// reports a fixed model info.
    struct ScriptedEngine {
        response: EngineResponse,
        creates: AtomicUsize,
        info: Option<ModelInfo>,
    }

    impl SessionEngine for ScriptedEngine {
        type Session = ();
        fn availability(&self) -> Availability {
            Availability::Available
        }
        fn model_info(&self) -> Option<ModelInfo> {
            self.info.clone()
        }
        fn create(&self, _instructions: &str) -> Result<()> {
            self.creates.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }
        fn respond(&self, _s: &mut (), _p: &str, _o: &RespondOptions) -> Result<EngineResponse> {
            Ok(self.response.clone())
        }
    }

    fn scripted(response: EngineResponse) -> SessionChatBackend<ScriptedEngine> {
        SessionChatBackend::new(
            ScriptedEngine { response, creates: AtomicUsize::new(0), info: None },
            Duration::from_secs(60),
            Duration::from_secs(60),
        )
    }

    /// Ask, then follow up exactly as a client would; returns the first
    /// response and whether the follow-up reused the session.
    async fn ask_then_follow_up(
        b: &SessionChatBackend<ScriptedEngine>,
        first: ChatRequest,
    ) -> (ChatResponse, bool) {
        let messages = first.messages.clone();
        let r1 = b.complete(first).await.unwrap();
        let mut second = messages;
        second.push(ChatMessage::new(Role::Assistant, r1.content.clone()));
        second.push(ChatMessage::new(Role::User, "and then?"));
        b.complete(req(second)).await.unwrap();
        let reused = b.engine.creates.load(Ordering::SeqCst) == 1;
        (r1, reused)
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn real_usage_passes_through() {
        let b = scripted(EngineResponse {
            text: "Madrid.".into(),
            usage: Some(EngineUsage { input: 86, cached: 78, output: 5, reasoning: 0 }),
            truncated: Some(false),
        });
        let r = b.complete(req(vec![ChatMessage::new(Role::User, "capital of Spain?")])).await.unwrap();
        assert_eq!(
            r.usage,
            Usage { prompt_tokens: 86, completion_tokens: 5, cached_tokens: Some(78), reasoning_tokens: None }
        );
        assert_eq!(r.finish, FinishReason::Stop);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn estimated_usage_rounds_up() {
        let b = scripted(EngineResponse::text("4"));
        let r = b.complete(req(vec![ChatMessage::new(Role::User, "2+2?")])).await.unwrap();
        assert_eq!(
            r.usage,
            Usage { prompt_tokens: 1, completion_tokens: 1, cached_tokens: None, reasoning_tokens: None },
            "a one-character reply is one token, not zero"
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn truncated_reply_finishes_with_length_and_stays_cacheable() {
        let b = scripted(EngineResponse {
            text: "Beneath the moon".into(),
            usage: None,
            truncated: Some(true),
        });
        let (r, reused) =
            ask_then_follow_up(&b, req(vec![ChatMessage::new(Role::User, "a poem")])).await;
        assert_eq!(r.finish, FinishReason::Length);
        assert!(reused, "a max_tokens cut leaves the transcript equal to the reply");
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn stop_truncates_and_is_not_cached() {
        let b = scripted(EngineResponse::text("1, 2, 3, 4, 5, 6"));
        let mut first = req(vec![ChatMessage::new(Role::User, "count")]);
        first.stop = vec!["4".into()];
        let (r, reused) = ask_then_follow_up(&b, first).await;
        assert_eq!(r.content, "1, 2, 3, ");
        assert_eq!(r.finish, FinishReason::Stop);
        assert!(!reused, "the transcript holds text the client never saw");
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn stop_wins_over_length() {
        let b = scripted(EngineResponse {
            text: "a b c STOP d".into(),
            usage: None,
            truncated: Some(true),
        });
        let mut r = req(vec![ChatMessage::new(Role::User, "go")]);
        r.stop = vec!["STOP".into()];
        let r = b.complete(r).await.unwrap();
        assert_eq!((r.content.as_str(), r.finish), ("a b c ", FinishReason::Stop));
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn context_limit_follows_reported_model_info() {
        let b = SessionChatBackend::new(
            ScriptedEngine {
                response: EngineResponse::text("x"),
                creates: AtomicUsize::new(0),
                info: Some(ModelInfo {
                    variant: Some("AFM 3 Core Advanced".into()),
                    variant_id: Some("core_advanced3".into()),
                    context_size: Some(8192),
                    capabilities: Some(vec!["guided_generation".into()]),
                }),
            },
            Duration::from_secs(60),
            Duration::from_secs(60),
        );
        assert_eq!(b.context_limit(), Some(4096), "default before the model reports");
        let info = b.model_info().await.unwrap();
        assert_eq!(info.variant_id.as_deref(), Some("core_advanced3"));
        assert_eq!(b.context_limit(), Some(8192));
    }

    /// Streams scripted snapshots. Panics if a session is used again after
    /// its stream was interrupted — on macOS 27 that traps the process.
    struct StreamEngine {
        snapshots: Vec<String>,
        final_text: Option<String>,
        /// Fail with this error once, after delivering this many snapshots.
        fail_after: Mutex<Option<(usize, Error)>>,
        creates: AtomicUsize,
        delivered: AtomicUsize,
        stream_calls: AtomicUsize,
    }

    #[derive(Default)]
    struct StreamSession {
        interrupted: bool,
    }

    impl StreamEngine {
        fn new(snapshots: &[&str]) -> Self {
            Self {
                snapshots: snapshots.iter().map(|s| s.to_string()).collect(),
                final_text: None,
                fail_after: Mutex::new(None),
                creates: AtomicUsize::new(0),
                delivered: AtomicUsize::new(0),
                stream_calls: AtomicUsize::new(0),
            }
        }
        fn final_text(&self) -> String {
            self.final_text.clone().unwrap_or_else(|| self.snapshots.last().unwrap().clone())
        }
    }

    impl SessionEngine for StreamEngine {
        type Session = StreamSession;
        fn availability(&self) -> Availability {
            Availability::Available
        }
        fn create(&self, _instructions: &str) -> Result<StreamSession> {
            self.creates.fetch_add(1, Ordering::SeqCst);
            Ok(StreamSession::default())
        }
        fn respond(&self, s: &mut StreamSession, _p: &str, _o: &RespondOptions) -> Result<EngineResponse> {
            assert!(!s.interrupted, "reused an interrupted session");
            Ok(EngineResponse::text(self.final_text()))
        }
        fn respond_stream(
            &self,
            s: &mut StreamSession,
            _p: &str,
            _o: &RespondOptions,
            on_snapshot: &mut (dyn FnMut(&str) -> bool + Send),
        ) -> Result<StreamedResponse> {
            assert!(!s.interrupted, "reused an interrupted session");
            self.stream_calls.fetch_add(1, Ordering::SeqCst);
            let fail = self.fail_after.lock().unwrap().take();
            for (i, snapshot) in self.snapshots.iter().enumerate() {
                if let Some((after, _)) = &fail {
                    if i == *after {
                        s.interrupted = true;
                        return Err(fail.unwrap().1);
                    }
                }
                self.delivered.fetch_add(1, Ordering::SeqCst);
                if !on_snapshot(snapshot) {
                    s.interrupted = true;
                    return Ok(StreamedResponse {
                        response: EngineResponse::text(snapshot.clone()),
                        cancelled: true,
                    });
                }
            }
            Ok(StreamedResponse { response: EngineResponse::text(self.final_text()), cancelled: false })
        }
    }

    fn streaming(engine: StreamEngine) -> SessionChatBackend<StreamEngine> {
        SessionChatBackend::new(engine, Duration::from_secs(60), Duration::from_secs(60))
    }

    /// A sink collecting deltas; `accept` = false simulates a gone client.
    fn collector(accept: bool) -> (DeltaSink, Arc<Mutex<Vec<String>>>) {
        let deltas = Arc::new(Mutex::new(Vec::new()));
        let seen = deltas.clone();
        let sink: DeltaSink = Box::new(move |d: &str| {
            seen.lock().unwrap().push(d.to_string());
            accept
        });
        (sink, deltas)
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn stream_sends_deltas_then_keeps_the_session() {
        let b = streaming(StreamEngine::new(&["Hel", "Hello wor", "Hello world."]));
        let (sink, deltas) = collector(true);
        let first = vec![ChatMessage::new(Role::User, "hi")];
        let r = b.complete_stream(req(first.clone()), sink).await.unwrap();
        assert_eq!(*deltas.lock().unwrap(), vec!["Hel", "lo wor", "ld."]);
        assert_eq!(r.content, "Hello world.");
        let mut second = first;
        second.push(ChatMessage::new(Role::Assistant, r.content));
        second.push(ChatMessage::new(Role::User, "more"));
        b.complete_stream(req(second), collector(true).0).await.unwrap();
        assert_eq!(b.engine.creates.load(Ordering::SeqCst), 1, "follow-up resumed the session");
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn stop_mid_stream_halts_generation_and_drops_the_session() {
        let b = streaming(StreamEngine::new(&["1, 2", "1, 2, 3", "1, 2, 3, 4, 5", "1, 2, 3, 4, 5, 6"]));
        let (sink, deltas) = collector(true);
        let mut first = req(vec![ChatMessage::new(Role::User, "count")]);
        first.stop = vec!["4".into()];
        let r = b.complete_stream(first, sink).await.unwrap();
        assert_eq!(r.content, "1, 2, 3, ");
        assert_eq!(deltas.lock().unwrap().concat(), "1, 2, 3, ");
        assert_eq!(r.finish, FinishReason::Stop);
        assert_eq!(b.engine.delivered.load(Ordering::SeqCst), 3, "generation stopped at the stop");
        // The follow-up must not resume the interrupted session (the mock
        // panics if it does).
        let follow_up = vec![
            ChatMessage::new(Role::User, "count"),
            ChatMessage::new(Role::Assistant, r.content),
            ChatMessage::new(Role::User, "again"),
        ];
        b.complete_stream(req(follow_up), collector(true).0).await.unwrap();
        assert_eq!(b.engine.creates.load(Ordering::SeqCst), 2);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn non_stream_request_with_stop_streams_internally() {
        let b = streaming(StreamEngine::new(&["1, 2", "1, 2, 3, 4", "1, 2, 3, 4, 5, 6"]));
        let mut r = req(vec![ChatMessage::new(Role::User, "count")]);
        r.stop = vec!["3".into()];
        let r = b.complete(r).await.unwrap();
        assert_eq!(r.content, "1, 2, ");
        assert_eq!(b.engine.stream_calls.load(Ordering::SeqCst), 1);
        assert_eq!(b.engine.delivered.load(Ordering::SeqCst), 2, "ended at the stop");
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn a_gone_client_stops_generation() {
        let b = streaming(StreamEngine::new(&["one", "one two", "one two three"]));
        let err = b
            .complete_stream(req(vec![ChatMessage::new(Role::User, "hi")]), collector(false).0)
            .await
            .unwrap_err();
        assert!(matches!(err, Error::Other(_)), "{err:?}");
        assert_eq!(b.engine.delivered.load(Ordering::SeqCst), 1);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn no_retry_once_text_was_sent() {
        let engine = StreamEngine::new(&["partial", "partial reply"]);
        *engine.fail_after.lock().unwrap() = Some((1, Error::Transient("timeout".into())));
        let b = streaming(engine);
        let (sink, deltas) = collector(true);
        let err = b
            .complete_stream(req(vec![ChatMessage::new(Role::User, "hi")]), sink)
            .await
            .unwrap_err();
        assert!(matches!(err, Error::Transient(_)));
        assert_eq!(deltas.lock().unwrap().concat(), "partial");
        assert_eq!(b.engine.creates.load(Ordering::SeqCst), 1, "no retry session");
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn transient_failure_before_any_text_retries() {
        let engine = StreamEngine::new(&["fine", "fine reply"]);
        *engine.fail_after.lock().unwrap() = Some((0, Error::Transient("concurrentRequests".into())));
        let b = streaming(engine);
        let (sink, deltas) = collector(true);
        let r = b.complete_stream(req(vec![ChatMessage::new(Role::User, "hi")]), sink).await.unwrap();
        assert_eq!(r.content, "fine reply");
        assert_eq!(deltas.lock().unwrap().concat(), "fine reply");
        assert_eq!(b.engine.creates.load(Ordering::SeqCst), 2, "retried on a fresh session");
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn streamed_label_is_held_and_stripped() {
        let b = streaming(StreamEngine::new(&["Assistant:", "Assistant: Hi", "Assistant: Hi there"]));
        let (sink, deltas) = collector(true);
        let r = b.complete_stream(req(vec![ChatMessage::new(Role::User, "hi")]), sink).await.unwrap();
        assert_eq!(r.content, "Hi there");
        assert_eq!(deltas.lock().unwrap().concat(), "Hi there");
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn a_rewritten_final_reply_is_an_error_and_not_cached() {
        let mut engine = StreamEngine::new(&["The cat", "The cat sat"]);
        engine.final_text = Some("The dog sat.".into());
        let b = streaming(engine);
        let err = b
            .complete_stream(req(vec![ChatMessage::new(Role::User, "hi")]), collector(true).0)
            .await
            .unwrap_err();
        assert!(matches!(err, Error::Inference(_)), "{err:?}");
        assert!(b.cache.lock().unwrap().is_empty());
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn constrained_requests_are_sent_whole() {
        let b = backend();
        let (sink, deltas) = collector(true);
        let mut r = req(vec![ChatMessage::new(Role::User, "extract")]);
        r.schema = Some(serde_json::json!({"type": "object"}));
        let resp = b.complete_stream(r, sink).await.unwrap();
        assert!(resp.constrained);
        assert_eq!(*deltas.lock().unwrap(), vec!["{\"turns\": 1}"]);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn schema_marks_constrained() {
        let b = backend();
        let mut r = req(vec![ChatMessage::new(Role::User, "extract")]);
        r.schema = Some(serde_json::json!({"type": "object"}));
        let resp = b.complete(r).await.unwrap();
        assert!(resp.constrained);
        assert_eq!(resp.content, "{\"turns\": 1}");
    }
}
