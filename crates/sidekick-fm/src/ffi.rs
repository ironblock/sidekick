//! The real engine: calls the Swift shim (swift/bridge.swift) over a C ABI.
//! Compiled only on macOS when the shim built (`not(fm_stub)`).

#![allow(unsafe_code)]

use crate::engine::{RespondOptions, SessionEngine};
use sidekick_core::{Availability, Error, Result, UnavailableReason};
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
    fn sk_fm_buf_free(ptr: *mut u8, len: usize);
    fn sk_fm_string_free(ptr: *mut c_char);
}

/// Take ownership of an error string from the shim.
unsafe fn take_error(err: *mut c_char, context: &str) -> Error {
    if err.is_null() {
        return Error::Inference(format!("{context}: unknown error"));
    }
    let message = CStr::from_ptr(err).to_string_lossy().into_owned();
    sk_fm_string_free(err);
    if message.contains("exceededContextWindowSize") {
        // The 4096-token combined budget was blown mid-generation.
        return Error::ContextOverflow { actual: 0, limit: 4096 };
    }
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
    ) -> Result<String> {
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
        // bytes that we own; copy then free via the shim's deallocator.
        let text = unsafe {
            let bytes = std::slice::from_raw_parts(out, out_len).to_vec();
            sk_fm_buf_free(out, out_len);
            String::from_utf8_lossy(&bytes).into_owned()
        };
        Ok(text)
    }
}
