pub mod chat;
pub mod embeddings;
pub mod misc;
pub mod wire;

use crate::state::AppState;
use axum::extract::{Request, State};
use axum::http::StatusCode;
use axum::middleware::{self, Next};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use sidekick_core::{Error, UnavailableReason};

pub fn build_router(state: AppState) -> Router {
    let v1 = Router::new()
        .route("/models", get(misc::list_models))
        .route("/chat/completions", post(chat::chat_completions))
        .route("/embeddings", post(embeddings::embeddings))
        .layer(middleware::from_fn_with_state(state.clone(), require_auth));

    Router::new()
        .nest("/v1", v1)
        .route("/health", get(misc::health))
        // axum's default; stated explicitly because it is what bounds the
        // tokenizer cost of one embeddings request.
        .layer(axum::extract::DefaultBodyLimit::max(2 * 1024 * 1024))
        .with_state(state)
}

/// Constant-time byte comparison, so the auth check leaks length only.
fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    a.len() == b.len() && a.iter().zip(b).fold(0u8, |acc, (x, y)| acc | (x ^ y)) == 0
}

async fn require_auth(
    State(state): State<AppState>,
    req: Request,
    next: Next,
) -> Response {
    if let Some(expected) = &state.api_key {
        let ok = req
            .headers()
            .get(axum::http::header::AUTHORIZATION)
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.strip_prefix("Bearer "))
            .map(|token| constant_time_eq(token.as_bytes(), expected.as_bytes()))
            .unwrap_or(false);
        if !ok {
            return ApiError::new(
                StatusCode::UNAUTHORIZED,
                "invalid_api_key",
                "Missing or invalid Authorization header",
            )
            .into_response();
        }
    }
    next.run(req).await
}

/// OpenAI-shaped error responses: `{"error": {"message", "type", "code"}}`.
#[derive(Debug)]
pub struct ApiError {
    pub status: StatusCode,
    pub code: &'static str,
    pub message: String,
}

impl ApiError {
    pub fn new(status: StatusCode, code: &'static str, message: impl Into<String>) -> Self {
        Self { status, code, message: message.into() }
    }

    pub fn invalid(message: impl Into<String>) -> Self {
        Self::new(StatusCode::BAD_REQUEST, "invalid_request_error", message)
    }

    pub fn model_not_found(model: &str) -> Self {
        Self::new(
            StatusCode::NOT_FOUND,
            "model_not_found",
            format!("The model `{model}` does not exist or is not loaded"),
        )
    }
}

impl From<Error> for ApiError {
    fn from(e: Error) -> Self {
        match e {
            Error::Unavailable(reason) => {
                let message = match &reason {
                    UnavailableReason::DeviceNotEligible => {
                        "This device cannot run Apple Foundation Models".into()
                    }
                    UnavailableReason::AppleIntelligenceNotEnabled => {
                        "Apple Intelligence is disabled in System Settings".into()
                    }
                    UnavailableReason::ModelNotReady => {
                        "The on-device model is still downloading; try again shortly".into()
                    }
                    UnavailableReason::NotSupportedInBuild => {
                        "This backend was not compiled into this build".into()
                    }
                    UnavailableReason::Other(s) => s.clone(),
                };
                Self::new(StatusCode::SERVICE_UNAVAILABLE, "backend_unavailable", message)
            }
            Error::ModelNotFound(m) => Self::model_not_found(&m),
            Error::ContextOverflow { limit, .. } => Self::new(
                StatusCode::BAD_REQUEST,
                "context_length_exceeded",
                format!("Request exceeds the on-device context budget of {limit} tokens"),
            ),
            Error::Timeout { secs } => Self::new(
                StatusCode::GATEWAY_TIMEOUT,
                "timeout",
                format!("Generation did not complete within {secs}s"),
            ),
            other => Self::new(
                StatusCode::INTERNAL_SERVER_ERROR,
                "internal_error",
                other.to_string(),
            ),
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        // The one choke point every ApiError passes through — log here so
        // failures are visible server-side, not just in the client's body.
        if self.status.is_server_error() {
            tracing::error!(status = %self.status, code = self.code, message = %self.message, "request failed");
        } else if self.status == StatusCode::UNAUTHORIZED {
            tracing::warn!(status = %self.status, code = self.code, "request rejected");
        } else {
            tracing::debug!(status = %self.status, code = self.code, message = %self.message, "request rejected");
        }
        let body = serde_json::json!({
            "error": {
                "message": self.message,
                "type": if self.status.is_client_error() {
                    "invalid_request_error"
                } else {
                    "server_error"
                },
                "code": self.code,
            }
        });
        (self.status, Json(body)).into_response()
    }
}
