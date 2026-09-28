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
pub mod envelope;
pub mod shaping;

#[cfg(all(target_os = "macos", not(fm_stub)))]
mod ffi;

pub use backend::SessionChatBackend;
pub use engine::{EngineResponse, EngineUsage, RespondOptions, SessionEngine};

/// The macOS SDK version the Foundation Models shim was compiled against
/// (e.g. `"27.0"`), or `"none"` in stub builds. macOS 27 features — real
/// token usage, model variant, typed errors — are only compiled in with the
/// 27 SDK, so a binary built with an older SDK behaves like macOS 26 even on
/// a macOS 27 machine; this makes that visible.
pub const FM_SDK: &str = env!("SIDEKICK_FM_SDK");

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
