use crate::cache::{conversation_key, ConversationCache};
use crate::engine::{RespondOptions, SessionEngine};
use sidekick_core::{
    Availability, ChatBackend, ChatMessage, ChatRequest, ChatResponse, Error, FinishReason,
    Result, Role, Usage,
};
use std::sync::{Arc, Mutex};
use std::time::Duration;

/// The Foundation Models on-device combined input+output token budget.
const FM_CONTEXT_LIMIT: usize = 4096;

/// `ChatBackend` over any [`SessionEngine`], with prefix-keyed session reuse.
pub struct SessionChatBackend<E: SessionEngine> {
    engine: Arc<E>,
    cache: Arc<Mutex<ConversationCache<E::Session>>>,
    request_timeout: Duration,
}

/// A runtime failure worth one retry on a fresh session: transient conditions
/// where the model runtime was busy, interrupted, or timed out internally.
/// Keyword set carried over from real-hardware failures observed in the
/// predecessor project. Context overflow is deliberately not here — a fresh
/// session cannot make the input smaller.
fn is_recoverable(err: &Error) -> bool {
    let message = match err {
        Error::Inference(m) => m.to_lowercase(),
        _ => return false,
    };
    ["timeout", "timed out", "resource", "busy", "interrupted", "cancelled"]
        .iter()
        .any(|k| message.contains(k))
}

impl<E: SessionEngine> SessionChatBackend<E> {
    pub fn new(engine: E, session_ttl: Duration, request_timeout: Duration) -> Self {
        Self {
            engine: Arc::new(engine),
            cache: Arc::new(Mutex::new(ConversationCache::new(session_ttl, 8))),
            request_timeout,
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
        let text = match engine.respond(&mut session, &prompt, &opts) {
            Ok(text) => text,
            Err(e) if is_recoverable(&e) => {
                drop(session);
                session = engine.create(&instructions)?;
                engine.respond(&mut session, &Self::replay_prompt(&history), &opts)?
            }
            Err(e) => return Err(e),
        };
        // Cold replays present history as a "User:/Assistant:" transcript and
        // the model sometimes mimics the format (seen on real hardware);
        // a leading speaker label is never part of a wanted reply.
        let text = text.strip_prefix("Assistant:").map(str::trim_start).map(String::from).unwrap_or(text);

        // File the session under the extended conversation for follow-ups.
        let mut extended = history;
        extended.push(ChatMessage::new(Role::Assistant, text.clone()));
        let key = conversation_key(&instructions, &extended);
        cache.lock().unwrap().insert(key, session);

        // Foundation Models (macOS 26) does not report token usage; estimate
        // at ~4 chars/token so OpenAI clients see plausible numbers.
        let prompt_chars: usize = req
            .messages
            .iter()
            .map(|m| m.content.len())
            .sum::<usize>();
        Ok(ChatResponse {
            usage: Usage {
                prompt_tokens: (prompt_chars / 4) as u32,
                completion_tokens: (text.len() / 4) as u32,
            },
            content: text,
            finish: FinishReason::Stop,
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
        Some(FM_CONTEXT_LIMIT)
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
        ) -> Result<String> {
            session.turns += 1;
            *self.last_prompt.lock().unwrap() = prompt.to_string();
            if let Some(err) = self.fail_with.lock().unwrap().pop() {
                return Err(err);
            }
            if opts.schema.is_some() {
                Ok(format!("{{\"turns\": {}}}", session.turns))
            } else {
                Ok(format!("reply-{}", session.turns))
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
            ) -> Result<String> {
                Ok("Assistant: Paris has 2.1 million people.".into())
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
    async fn recoverable_error_retries_on_fresh_session() {
        let b = backend();
        b.engine
            .fail_with
            .lock()
            .unwrap()
            .push(Error::Inference("respond: model runtime busy".into()));

        let history = vec![
            ChatMessage::new(Role::User, "one"),
            ChatMessage::new(Role::Assistant, "two"),
            ChatMessage::new(Role::User, "three"),
        ];
        let r = b.complete(req(history)).await.unwrap();
        assert_eq!(r.content, "reply-1", "retry ran on a fresh session");
        assert_eq!(
            b.engine.creates.load(Ordering::SeqCst),
            2,
            "failed session dropped, fresh one created"
        );
        let prompt = b.engine.last_prompt.lock().unwrap().clone();
        assert!(
            prompt.contains("User: one") && prompt.ends_with("User: three"),
            "retry replays full history since the fresh session has none"
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn non_recoverable_error_propagates_without_retry() {
        let b = backend();
        b.engine
            .fail_with
            .lock()
            .unwrap()
            .push(Error::ContextOverflow { actual: 5000, limit: 4096 });

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
            ) -> Result<String> {
                // Short: runtime shutdown waits for started blocking tasks.
                std::thread::sleep(Duration::from_millis(400));
                Ok("too late".into())
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
