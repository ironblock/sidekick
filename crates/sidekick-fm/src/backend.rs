use crate::cache::{conversation_key, ConversationCache};
use crate::engine::{EngineUsage, RespondOptions, SessionEngine};
use crate::shaping::shape;
use sidekick_core::{
    Availability, ChatBackend, ChatMessage, ChatRequest, ChatResponse, Error, FinishReason,
    ModelInfo, Result, Role, Usage,
};
use std::sync::atomic::{AtomicUsize, Ordering};
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

    fn run_sync(
        engine: &E,
        cache: &Mutex<ConversationCache<E::Session>>,
        req: ChatRequest,
    ) -> Result<ChatResponse> {
        let (instructions, history) = Self::split(&req.messages);
        let last = history.last().ok_or_else(|| {
            Error::Other("chat request must contain at least one non-system message".into())
        })?;
        if last.role != Role::User {
            return Err(Error::Other("last message must have role `user`".into()));
        }

        let constrained = req.schema.is_some();
        let opts = RespondOptions {
            schema: req.schema,
            temperature: req.temperature,
            max_tokens: req.max_tokens,
        };

        let prefix = conversation_key(&instructions, &history[..history.len() - 1]);
        let (mut session, prompt) = {
            let cached = cache.lock().unwrap().take(&prefix);
            match cached {
                Some(session) => (session, last.content.clone()),
                None => (engine.create(&instructions)?, Self::replay_prompt(&history)),
            }
        };

        // A `LanguageModelSession` that failed mid-generation may be left in
        // a bad state, so an errored session is always dropped (it was taken
        // out of the cache and is never re-filed). Transient failures get one
        // retry on a fresh session, which needs the full replay prompt since
        // it has no history.
        let response = match engine.respond(&mut session, &prompt, &opts) {
            Ok(response) => response,
            Err(e) if is_recoverable(&e) => {
                drop(session);
                session = engine.create(&instructions)?;
                engine.respond(&mut session, &Self::replay_prompt(&history), &opts)?
            }
            Err(e) => return Err(e),
        };
        let shaped = shape(&response.text, &req.stop);

        // File the session under the extended conversation for follow-ups —
        // unless a stop sequence cut the reply: the session's transcript then
        // holds text the client never saw, so it must not be resumed.
        // (A max_tokens cut is fine: the transcript is exactly the reply.)
        if !shaped.stop_hit {
            let mut extended = history;
            extended.push(ChatMessage::new(Role::Assistant, shaped.text.clone()));
            let key = conversation_key(&instructions, &extended);
            cache.lock().unwrap().insert(key, session);
        }

        // A stop hit is a normal stop even if the limit was also reached.
        let finish = if !shaped.stop_hit && response.truncated == Some(true) {
            FinishReason::Length
        } else {
            FinishReason::Stop
        };
        Ok(ChatResponse {
            usage: usage(response.usage, &req.messages, &shaped.text),
            content: shaped.text,
            finish,
            constrained,
        })
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::{EngineResponse, EngineUsage};
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
