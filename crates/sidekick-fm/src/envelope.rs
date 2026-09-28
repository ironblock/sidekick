//! The JSON envelope the Swift shim returns from `sk_fm_respond`, and its
//! mapping to engine responses and sidekick errors.
//!
//! The shim classifies Foundation Models errors by *type* into a small set
//! of kinds (see `classify` in swift/bridge.swift); this module turns kinds
//! into [`sidekick_core::Error`] variants, which the server maps to HTTP
//! statuses. It is platform-neutral so the mapping is tested everywhere; the
//! FFI itself only moves bytes.

use crate::engine::{EngineResponse, EngineUsage};
use serde::Deserialize;
use sidekick_core::{Error, Result, UnavailableReason};

#[derive(Debug, Deserialize)]
struct Envelope {
    #[serde(default)]
    text: Option<String>,
    #[serde(default)]
    usage: Option<EngineUsage>,
    #[serde(default)]
    truncated: Option<bool>,
    #[serde(default)]
    error: Option<EnvelopeError>,
}

/// A classified Foundation Models error.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct EnvelopeError {
    pub kind: String,
    pub message: String,
    #[serde(default)]
    pub token_count: Option<usize>,
    #[serde(default)]
    pub context_size: Option<usize>,
    #[serde(default)]
    pub retry_after_secs: Option<u64>,
}

/// Parse a respond envelope. `known_limit` is the context size to report
/// for an overflow whose error carries none (macOS 26).
pub fn parse_response(json: &str, known_limit: usize) -> Result<EngineResponse> {
    let envelope: Envelope = serde_json::from_str(json)
        .map_err(|e| Error::Inference(format!("respond: malformed shim envelope: {e}")))?;
    if let Some(error) = envelope.error {
        return Err(to_error(error, known_limit));
    }
    let text = envelope
        .text
        .ok_or_else(|| Error::Inference("respond: shim envelope has no text".into()))?;
    Ok(EngineResponse { text, usage: envelope.usage, truncated: envelope.truncated })
}

/// Map a classified error kind to a sidekick error.
pub fn to_error(error: EnvelopeError, known_limit: usize) -> Error {
    match error.kind.as_str() {
        "context_overflow" => Error::ContextOverflow {
            limit: error.context_size.unwrap_or(known_limit),
            actual: error.token_count,
        },
        "content_filter" => Error::ContentFiltered(error.message),
        "rate_limited" => Error::RateLimited {
            message: error.message,
            retry_after_secs: error.retry_after_secs,
        },
        "transient" => Error::Transient(error.message),
        "model_not_ready" => Error::Unavailable(UnavailableReason::ModelNotReady),
        "unsupported_guide" => Error::GuidedGeneration(error.message),
        "unsupported_language" => Error::UnsupportedLanguage(error.message),
        _ => Error::Inference(format!("respond: {}", error.message)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn err(json: &str) -> Error {
        parse_response(json, 4096).unwrap_err()
    }

    #[test]
    fn success_carries_usage_and_truncation() {
        let r = parse_response(
            r#"{"text":"Paris.","usage":{"input":70,"cached":0,"output":5,"reasoning":0},"truncated":false,"error":null}"#,
            4096,
        )
        .unwrap();
        assert_eq!(r.text, "Paris.");
        assert_eq!(r.usage, Some(EngineUsage { input: 70, cached: 0, output: 5, reasoning: 0 }));
        assert_eq!(r.truncated, Some(false));
    }

    #[test]
    fn success_without_macos27_fields() {
        let r = parse_response(r#"{"text":"hi","usage":null,"truncated":null,"error":null}"#, 4096)
            .unwrap();
        assert_eq!(r, EngineResponse::text("hi"));
    }

    #[test]
    fn overflow_with_counts_uses_them() {
        let e = err(r#"{"text":null,"error":{"kind":"context_overflow","message":"m","token_count":4459,"context_size":8192}}"#);
        assert!(matches!(e, Error::ContextOverflow { limit: 8192, actual: Some(4459) }), "{e:?}");
    }

    #[test]
    fn overflow_without_counts_falls_back_to_known_limit() {
        let e = parse_response(
            r#"{"text":null,"error":{"kind":"context_overflow","message":"m"}}"#,
            8192,
        )
        .unwrap_err();
        assert!(matches!(e, Error::ContextOverflow { limit: 8192, actual: None }), "{e:?}");
    }

    #[test]
    fn every_kind_maps() {
        let case = |kind: &str| err(&format!(r#"{{"error":{{"kind":"{kind}","message":"m"}}}}"#));
        assert!(matches!(case("content_filter"), Error::ContentFiltered(_)));
        assert!(matches!(case("transient"), Error::Transient(_)));
        assert!(matches!(
            case("model_not_ready"),
            Error::Unavailable(UnavailableReason::ModelNotReady)
        ));
        assert!(matches!(case("unsupported_guide"), Error::GuidedGeneration(_)));
        assert!(matches!(case("unsupported_language"), Error::UnsupportedLanguage(_)));
        assert!(matches!(case("other"), Error::Inference(_)));
        assert!(matches!(case("a kind from a newer shim"), Error::Inference(_)));
        let e = err(r#"{"error":{"kind":"rate_limited","message":"m","retry_after_secs":30}}"#);
        assert!(matches!(e, Error::RateLimited { retry_after_secs: Some(30), .. }), "{e:?}");
    }

    #[test]
    fn malformed_envelopes_are_inference_errors() {
        assert!(matches!(err("not json"), Error::Inference(_)));
        assert!(matches!(err(r#"{"text":null,"error":null}"#), Error::Inference(_)));
    }
}
