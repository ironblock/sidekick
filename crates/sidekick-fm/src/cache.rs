//! TTL'd session reuse keyed by conversation prefix.
//!
//! The OpenAI chat API is stateless: every request carries full history.
//! Foundation Models sessions are stateful and benefit from reuse (prewarmed
//! model, existing transcript). This cache bridges the two: after each
//! response we file the live session under a hash of the *entire*
//! conversation including our reply; a follow-up request whose history
//! matches that hash takes the session back and only sends the new user
//! message. Anything else falls through to a cold session with history
//! replay.

use sha2::{Digest, Sha256};
use sidekick_core::ChatMessage;
use std::collections::HashMap;
use std::time::{Duration, Instant};

pub type ConversationKey = [u8; 32];

pub fn conversation_key(instructions: &str, messages: &[ChatMessage]) -> ConversationKey {
    let mut h = Sha256::new();
    h.update((instructions.len() as u64).to_le_bytes());
    h.update(instructions.as_bytes());
    for m in messages {
        h.update([match m.role {
            sidekick_core::Role::System => 0u8,
            sidekick_core::Role::User => 1,
            sidekick_core::Role::Assistant => 2,
        }]);
        h.update((m.content.len() as u64).to_le_bytes());
        h.update(m.content.as_bytes());
    }
    h.finalize().into()
}

struct Entry<S> {
    session: S,
    last_used: Instant,
}

pub struct ConversationCache<S> {
    ttl: Duration,
    cap: usize,
    entries: HashMap<ConversationKey, Entry<S>>,
}

impl<S> ConversationCache<S> {
    pub fn new(ttl: Duration, cap: usize) -> Self {
        Self { ttl, cap: cap.max(1), entries: HashMap::new() }
    }

    /// Remove and return the session for `key` if present and fresh.
    pub fn take(&mut self, key: &ConversationKey) -> Option<S> {
        self.sweep();
        self.entries.remove(key).map(|e| e.session)
    }

    /// File a session under a new conversation key. Evicts expired entries,
    /// then the least-recently-used entry if at capacity.
    pub fn insert(&mut self, key: ConversationKey, session: S) {
        self.sweep();
        if self.entries.len() >= self.cap {
            if let Some(oldest) = self
                .entries
                .iter()
                .min_by_key(|(_, e)| e.last_used)
                .map(|(k, _)| *k)
            {
                self.entries.remove(&oldest);
            }
        }
        self.entries.insert(key, Entry { session, last_used: Instant::now() });
    }

    fn sweep(&mut self) {
        let ttl = self.ttl;
        self.entries.retain(|_, e| e.last_used.elapsed() < ttl);
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sidekick_core::{ChatMessage, Role};

    fn msgs(turns: &[(&str, Role)]) -> Vec<ChatMessage> {
        turns
            .iter()
            .map(|(c, r)| ChatMessage::new(*r, *c))
            .collect()
    }

    #[test]
    fn key_is_stable_and_prefix_sensitive() {
        let a = msgs(&[("hi", Role::User), ("hello!", Role::Assistant)]);
        let k1 = conversation_key("sys", &a);
        let k2 = conversation_key("sys", &a);
        assert_eq!(k1, k2);
        assert_ne!(k1, conversation_key("other-sys", &a));
        assert_ne!(k1, conversation_key("sys", &a[..1]));
        // Role matters, not just concatenated text.
        let swapped = msgs(&[("hi", Role::Assistant), ("hello!", Role::User)]);
        assert_ne!(k1, conversation_key("sys", &swapped));
    }

    #[test]
    fn take_removes_and_ttl_expires() {
        let mut cache: ConversationCache<u32> =
            ConversationCache::new(Duration::from_millis(30), 4);
        let k = [1u8; 32];
        cache.insert(k, 7);
        assert_eq!(cache.take(&k), Some(7));
        assert_eq!(cache.take(&k), None, "take removes");

        cache.insert(k, 8);
        std::thread::sleep(Duration::from_millis(40));
        assert_eq!(cache.take(&k), None, "expired by ttl");
    }

    #[test]
    fn capacity_evicts_oldest() {
        let mut cache: ConversationCache<u32> =
            ConversationCache::new(Duration::from_secs(60), 2);
        cache.insert([1u8; 32], 1);
        std::thread::sleep(Duration::from_millis(5));
        cache.insert([2u8; 32], 2);
        std::thread::sleep(Duration::from_millis(5));
        cache.insert([3u8; 32], 3);
        assert_eq!(cache.len(), 2);
        assert_eq!(cache.take(&[1u8; 32]), None, "oldest evicted");
        assert_eq!(cache.take(&[3u8; 32]), Some(3));
    }
}
