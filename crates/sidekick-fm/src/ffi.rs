//! The real engine: calls the Swift shim (swift/bridge.swift) over a C ABI.
//! Compiled only on macOS when the shim built (`not(fm_stub)`).

#![allow(unsafe_code)]

use crate::engine::{EngineResponse, RespondOptions, SessionEngine, StreamedResponse};
use crate::envelope;
use sidekick_core::{Availability, Error, ModelInfo, Result, UnavailableReason};
use std::ffi::{c_char, c_void, CStr};
use std::ptr::NonNull;

extern "C" {
    fn sk_fm_availability() -> i32;
    fn sk_fm_session_create(
        instructions: *const u8,
        instructions_len: usize,
        err: *mut *mut c_char,
    ) -> *mut c_void;
    fn sk_fm_session_free(session: *mut c_void);
    fn sk_fm_respond(
        session: *mut c_void,
        prompt: *const u8,
        prompt_len: usize,
        schema_json: *const u8,
        schema_len: usize,
        temperature: f64,
        max_tokens: i64,
        out: *mut *mut u8,
        out_len: *mut usize,
        err: *mut *mut c_char,
    ) -> i32;
    fn sk_fm_respond_stream(
        session: *mut c_void,
        prompt: *const u8,
        prompt_len: usize,
        temperature: f64,
        max_tokens: i64,
        on_snapshot: SnapshotCallback,
        ctx: *mut c_void,
        out: *mut *mut u8,
        out_len: *mut usize,
        err: *mut *mut c_char,
    ) -> i32;
    fn sk_fm_model_info(out: *mut *mut u8, out_len: *mut usize) -> i32;
    fn sk_fm_buf_free(ptr: *mut u8, len: usize);
    fn sk_fm_string_free(ptr: *mut c_char);
    #[cfg(test)]
    fn sk_fm_selftest(out: *mut *mut u8, out_len: *mut usize) -> i32;
}

type SnapshotCallback = extern "C" fn(ctx: *mut c_void, text: *const u8, len: usize) -> i32;

/// What the snapshot trampoline needs: the caller's callback, and a slot for
/// a panic caught inside it (unwinding across the FFI boundary is undefined
/// behavior, so it is caught there and re-raised after the call returns).
struct SnapshotContext<'a> {
    callback: &'a mut (dyn FnMut(&str) -> bool + Send),
    panic: Option<Box<dyn std::any::Any + Send>>,
}

/// Called by the shim with each snapshot's cumulative text. Returns nonzero
/// to stop generation.
extern "C" fn snapshot_trampoline(ctx: *mut c_void, text: *const u8, len: usize) -> i32 {
    // SAFETY: `ctx` is the `SnapshotContext` that `respond_stream` passed to
    // sk_fm_respond_stream. It outlives that call, and the shim invokes this
    // callback one call at a time, all before the call returns, while the
    // owning thread is blocked in it — so this is the only live reference.
    let context = unsafe { &mut *(ctx as *mut SnapshotContext<'_>) };
    if context.panic.is_some() {
        return 1;
    }
    let bytes = if text.is_null() || len == 0 {
        &[][..]
    } else {
        // SAFETY: the shim passes a valid buffer of `len` bytes for the
        // duration of this call.
        unsafe { std::slice::from_raw_parts(text, len) }
    };
    let text = String::from_utf8_lossy(bytes);
    let callback = &mut context.callback;
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| callback(&text))) {
        Ok(true) => 0,
        Ok(false) => 1,
        Err(payload) => {
            context.panic = Some(payload);
            1
        }
    }
}

/// Context size assumed when Foundation Models reports none (macOS 26's
/// fixed on-device budget).
const DEFAULT_CONTEXT_SIZE: usize = 4096;

/// Copy a shim-allocated UTF-8 buffer into a String and free it.
///
/// # Safety
/// `ptr` must be a buffer of `len` bytes allocated by the shim, not yet freed.
unsafe fn take_buffer(ptr: *mut u8, len: usize) -> String {
    let bytes = std::slice::from_raw_parts(ptr, len).to_vec();
    sk_fm_buf_free(ptr, len);
    String::from_utf8_lossy(&bytes).into_owned()
}

/// Take ownership of an error string from the shim. These are failures of
/// the shim itself (null arguments, encoding); model errors arrive typed in
/// the respond envelope instead.
unsafe fn take_error(err: *mut c_char, context: &str) -> Error {
    if err.is_null() {
        return Error::Inference(format!("{context}: unknown error"));
    }
    let message = CStr::from_ptr(err).to_string_lossy().into_owned();
    sk_fm_string_free(err);
    Error::Inference(format!("{context}: {message}"))
}

pub struct FfiSession(NonNull<c_void>);

// SAFETY: the session pointer is only ever used by one caller at a time
// (ConversationCache hands out exclusive ownership), and the shim's session
// box is safe to move between threads.
unsafe impl Send for FfiSession {}

impl Drop for FfiSession {
    fn drop(&mut self) {
        unsafe { sk_fm_session_free(self.0.as_ptr()) };
    }
}

pub struct FfiEngine;

impl SessionEngine for FfiEngine {
    type Session = FfiSession;

    fn availability(&self) -> Availability {
        match unsafe { sk_fm_availability() } {
            0 => Availability::Available,
            1 => Availability::unavailable(UnavailableReason::DeviceNotEligible),
            2 => Availability::unavailable(UnavailableReason::AppleIntelligenceNotEnabled),
            3 => Availability::unavailable(UnavailableReason::ModelNotReady),
            5 => Availability::unavailable(UnavailableReason::Other(
                "Foundation Models requires macOS 26 or later".into(),
            )),
            _ => Availability::unavailable(UnavailableReason::Other(
                "Foundation Models unavailable for an unknown reason".into(),
            )),
        }
    }

    fn model_info(&self) -> Option<ModelInfo> {
        let mut out: *mut u8 = std::ptr::null_mut();
        let mut out_len: usize = 0;
        if unsafe { sk_fm_model_info(&mut out, &mut out_len) } != 0 || out.is_null() {
            return None;
        }
        // SAFETY: on success the shim hands us an owned UTF-8 buffer.
        let json = unsafe { take_buffer(out, out_len) };
        match serde_json::from_str(&json) {
            Ok(info) => Some(info),
            Err(e) => {
                tracing::warn!("malformed model info from the shim: {e}");
                None
            }
        }
    }

    fn respond_stream(
        &self,
        session: &mut FfiSession,
        prompt: &str,
        opts: &RespondOptions,
        on_snapshot: &mut (dyn FnMut(&str) -> bool + Send),
    ) -> Result<StreamedResponse> {
        if opts.schema.is_some() {
            // Constrained output isn't streamed (partial JSON snapshots
            // aren't prefix-stable): deliver it whole.
            let response = self.respond(session, prompt, opts)?;
            let cancelled = !on_snapshot(&response.text);
            return Ok(StreamedResponse { response, cancelled });
        }
        let mut context = SnapshotContext { callback: on_snapshot, panic: None };
        let mut out: *mut u8 = std::ptr::null_mut();
        let mut out_len: usize = 0;
        let mut err: *mut c_char = std::ptr::null_mut();
        let code = unsafe {
            sk_fm_respond_stream(
                session.0.as_ptr(),
                prompt.as_ptr(),
                prompt.len(),
                opts.temperature.map(f64::from).unwrap_or(-1.0),
                opts.max_tokens.map(i64::from).unwrap_or(0),
                snapshot_trampoline,
                &mut context as *mut SnapshotContext<'_> as *mut c_void,
                &mut out,
                &mut out_len,
                &mut err,
            )
        };
        if let Some(payload) = context.panic.take() {
            // SAFETY: owned shim allocations; freed before unwinding.
            unsafe {
                if !out.is_null() {
                    drop(take_buffer(out, out_len));
                }
                if !err.is_null() {
                    sk_fm_string_free(err);
                }
            }
            std::panic::resume_unwind(payload);
        }
        if code != 0 {
            return Err(unsafe { take_error(err, "respond") });
        }
        if out.is_null() {
            return Err(Error::Inference("respond: shim returned null buffer".into()));
        }
        // SAFETY: shim guarantees `out` is a valid UTF-8 buffer of `out_len`
        // bytes that we own.
        let json = unsafe { take_buffer(out, out_len) };
        envelope::parse_streamed(&json, DEFAULT_CONTEXT_SIZE)
    }

    fn create(&self, instructions: &str) -> Result<FfiSession> {
        let mut err: *mut c_char = std::ptr::null_mut();
        let ptr = unsafe {
            sk_fm_session_create(instructions.as_ptr(), instructions.len(), &mut err)
        };
        match NonNull::new(ptr) {
            Some(p) => Ok(FfiSession(p)),
            None => Err(unsafe { take_error(err, "session create") }),
        }
    }

    fn respond(
        &self,
        session: &mut FfiSession,
        prompt: &str,
        opts: &RespondOptions,
    ) -> Result<EngineResponse> {
        let schema_text = opts
            .schema
            .as_ref()
            .map(|s| s.to_string())
            .unwrap_or_default();
        let mut out: *mut u8 = std::ptr::null_mut();
        let mut out_len: usize = 0;
        let mut err: *mut c_char = std::ptr::null_mut();

        let code = unsafe {
            sk_fm_respond(
                session.0.as_ptr(),
                prompt.as_ptr(),
                prompt.len(),
                schema_text.as_ptr(),
                schema_text.len(),
                opts.temperature.map(f64::from).unwrap_or(-1.0),
                opts.max_tokens.map(i64::from).unwrap_or(0),
                &mut out,
                &mut out_len,
                &mut err,
            )
        };

        if code != 0 {
            return Err(unsafe { take_error(err, "respond") });
        }
        if out.is_null() {
            return Err(Error::Inference("respond: shim returned null buffer".into()));
        }
        // SAFETY: shim guarantees `out` is a valid UTF-8 buffer of `out_len`
        // bytes that we own.
        let json = unsafe { take_buffer(out, out_len) };
        envelope::parse_response(&json, DEFAULT_CONTEXT_SIZE)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    /// The shim's typed error classification, exercised over constructed
    /// errors — no model needed, so this runs in CI on both SDK paths.
    #[test]
    fn shim_classifies_foundation_models_errors() {
        let mut out: *mut u8 = std::ptr::null_mut();
        let mut out_len: usize = 0;
        assert_eq!(unsafe { sk_fm_selftest(&mut out, &mut out_len) }, 0);
        let report: Value = serde_json::from_str(&unsafe { take_buffer(out, out_len) }).unwrap();
        let cases = report["cases"].as_array().expect("cases");
        let kind_of = |name: &str| -> &Value {
            cases
                .iter()
                .find(|c| c["case"] == name)
                .unwrap_or_else(|| panic!("self-test case {name} missing: {report}"))
        };

        for (name, kind) in [
            ("gen.exceededContextWindowSize", "context_overflow"),
            ("gen.assetsUnavailable", "model_not_ready"),
            ("gen.guardrailViolation", "content_filter"),
            ("gen.refusal", "content_filter"),
            ("gen.unsupportedGuide", "unsupported_guide"),
            ("gen.unsupportedLanguageOrLocale", "unsupported_language"),
            ("gen.decodingFailure", "other"),
            ("gen.rateLimited", "rate_limited"),
            ("gen.concurrentRequests", "transient"),
            ("message.contextOverflow", "context_overflow"),
            ("message.other", "other"),
        ] {
            assert_eq!(kind_of(name)["kind"], kind, "{name}");
        }
        // The message fallback recovers both numbers.
        assert_eq!(kind_of("message.contextOverflow")["token_count"], 4459);
        assert_eq!(kind_of("message.contextOverflow")["context_size"], 4096);

        if report["runtime27"] == true {
            for (name, kind) in [
                ("lm.contextSizeExceeded", "context_overflow"),
                ("lm.rateLimited", "rate_limited"),
                ("lm.guardrailViolation", "content_filter"),
                ("lm.refusal", "content_filter"),
                ("lm.timeout", "transient"),
                ("lm.unsupportedLanguageOrLocale", "unsupported_language"),
                ("lm.unsupportedGenerationGuide", "unsupported_guide"),
                ("session.concurrentRequests", "transient"),
                ("session.transcriptMutationWhileResponding", "transient"),
                ("system.assetsUnavailable", "model_not_ready"),
            ] {
                assert_eq!(kind_of(name)["kind"], kind, "{name}");
            }
            let overflow = kind_of("lm.contextSizeExceeded");
            assert_eq!(overflow["token_count"], 5000);
            assert_eq!(overflow["context_size"], 4096);
            let retry = kind_of("lm.rateLimited")["retry_after_secs"].as_u64().unwrap();
            assert!((1..=30).contains(&retry), "retry_after_secs {retry}");
        }
    }
}
