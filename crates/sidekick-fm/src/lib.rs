//! Apple Foundation Models chat backend (macOS 26+).
//!
//! Layering:
//! - [`engine::SessionEngine`] — the tiny surface a session provider must
//!   implement (create session, blocking respond). Platform-neutral.
//! - [`cache::ConversationCache`] — TTL'd reuse of live sessions keyed by
//!   conversation-prefix hash, so OpenAI-style stateless requests that
//!   extend a recent conversation don't replay history. Platform-neutral,
//!   tested everywhere.
//! - [`backend::SessionChatBackend`] — implements `sidekick_core::ChatBackend`
//!   over any engine.
//! - `ffi` — the real engine, calling the Swift shim (macOS, non-stub builds).
//!
//! On non-macOS targets (or when the shim can't build) `FmChatBackend` is an
//! alias for the backend over a stub engine that reports `Unavailable`.

pub mod backend;
pub mod cache;
pub mod engine;

#[cfg(all(target_os = "macos", not(fm_stub)))]
mod ffi;

pub use backend::SessionChatBackend;
pub use engine::{RespondOptions, SessionEngine};

#[cfg(all(target_os = "macos", not(fm_stub)))]
pub type FmChatBackend = SessionChatBackend<ffi::FfiEngine>;

#[cfg(all(target_os = "macos", not(fm_stub)))]
pub fn fm_backend(
    session_ttl: std::time::Duration,
    request_timeout: std::time::Duration,
) -> FmChatBackend {
    SessionChatBackend::new(ffi::FfiEngine, session_ttl, request_timeout)
}

#[cfg(not(all(target_os = "macos", not(fm_stub))))]
pub type FmChatBackend = SessionChatBackend<engine::StubEngine>;

#[cfg(not(all(target_os = "macos", not(fm_stub))))]
pub fn fm_backend(
    session_ttl: std::time::Duration,
    request_timeout: std::time::Duration,
) -> FmChatBackend {
    SessionChatBackend::new(engine::StubEngine, session_ttl, request_timeout)
}
