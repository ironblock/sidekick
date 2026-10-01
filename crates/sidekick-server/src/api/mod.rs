pub mod chat;
pub mod classify;
pub mod embeddings;
pub mod misc;
pub mod pooling;
pub mod rerank;
pub mod embed_v2;
pub mod wire;

use crate::state::AppState;
use axum::extract::rejection::JsonRejection;
use axum::extract::{FromRequest, Request, State};
use axum::http::{HeaderName, HeaderValue, StatusCode};
use axum::middleware::{self, Next};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use sidekick_core::{EmbeddingBackendKind, Error, UnavailableReason};

pub fn build_router(state: AppState) -> Router {
    // Every inference and listing route needs the API key when one is set.
    let api = Router::new()
        .route("/v1/models", get(misc::list_models))
        .route("/v1/chat/completions", post(chat::chat_completions))
        .route("/v1/embeddings", post(embeddings::embeddings))
        .route("/v1/classify", post(classify::classify))
        // vLLM serves its rerank shape at both paths (D29).
        .route("/v1/rerank", post(rerank::rerank_v1))
        .route("/rerank", post(rerank::rerank_v1))
        // Cohere's v2 shapes.
        .route("/v2/rerank", post(rerank::rerank_v2))
        .route("/v2/embed", post(embed_v2::embed_v2))
        .layer(middleware::from_fn_with_state(state.clone(), require_auth));

    Router::new()
        .merge(api)
        .route("/health", get(misc::health))
        // axum's default; stated explicitly because it is what bounds the
        // tokenizer cost of one embeddings request.
        .layer(axum::extract::DefaultBodyLimit::max(2 * 1024 * 1024))
        .with_state(state)
}

/// A JSON request body. Every way a body can fail to parse (bad syntax,
/// wrong types, a missing field, no JSON content type) is an [`ApiError`]
/// 400, where axum's own `Json` answers a plain-text 422 or 415.
pub struct ApiJson<T>(pub T);

impl<T, S> FromRequest<S> for ApiJson<T>
where
    Json<T>: FromRequest<S, Rejection = JsonRejection>,
    S: Send + Sync,
{
    type Rejection = ApiError;

    async fn from_request(req: Request, state: &S) -> Result<Self, Self::Rejection> {
        match Json::<T>::from_request(req, state).await {
            Ok(Json(value)) => Ok(Self(value)),
            // Body too large (413) and failures reading it keep their status.
            Err(JsonRejection::BytesRejection(e)) => {
                Err(ApiError::new(e.status(), "invalid_request_error", e.body_text()))
            }
            Err(e) => Err(ApiError::invalid(format!("Invalid JSON body: {}", e.body_text()))),
        }
    }
}

/// What a model does, in Hugging Face's pipeline vocabulary, for listings
/// and for errors that send a request to the wrong route.
pub fn model_task(state: &AppState, id: &str) -> Option<&'static str> {
    if let Ok(c) = state.registry.classifier(id) {
        return Some(c.manifest.task.name());
    }
    if state.registry.get(id).is_ok() {
        return Some("feature-extraction");
    }
    (id == state.chat.id()).then_some("text-generation")
}

/// The 400 for a model sent to a route that doesn't serve its task.
pub fn wrong_route(id: &str, task: &str, route: &str) -> ApiError {
    let use_instead = match task {
        "feature-extraction" => "/v1/embeddings",
        "text-generation" => "/v1/chat/completions",
        "text-ranking" => "/v1/rerank",
        _ => "/v1/classify",
    };
    ApiError::new(
        StatusCode::BAD_REQUEST,
        "invalid_request_error",
        format!("model `{id}` is a {task} model, which {route} doesn't serve; use {use_instead}"),
    )
}

/// Provenance headers for an inference response (docs/design/classify.md):
/// the daemon version, the model (`<id>@<revision>` when its manifest
/// names a source revision), and the compute units the serving instance
/// was loaded with. Chat has no compute units to report.
pub struct Provenance {
    pub model: String,
    pub compute_units: Option<&'static str>,
}

/// The compute units every Core ML model is loaded with (D14).
pub const CORE_ML_UNITS: &str = "cpu_and_ne";

impl Provenance {
    pub fn model_id(id: &str, source: Option<&sidekick_core::Source>) -> String {
        match source.and_then(|s| s.revision.as_deref()) {
            Some(rev) => format!("{id}@{rev}"),
            None => id.to_string(),
        }
    }

    pub fn embedder(state: &AppState, id: &str) -> Self {
        let m = state.registry.get(id).ok().map(|r| &r.manifest);
        Self {
            model: Self::model_id(id, m.and_then(|m| m.source.as_ref())),
            compute_units: Some(match m.map(|m| m.backend) {
                Some(EmbeddingBackendKind::Static) => "cpu",
                _ => CORE_ML_UNITS,
            }),
        }
    }

    pub fn apply(self, mut response: Response) -> Response {
        let headers = response.headers_mut();
        let mut set = |name: &'static str, value: &str| {
            // A model id that isn't a valid header value just goes without.
            if let Ok(v) = HeaderValue::from_str(value) {
                headers.insert(HeaderName::from_static(name), v);
            }
        };
        set("sidekick-version", env!("CARGO_PKG_VERSION"));
        set("sidekick-model", &self.model);
        if let Some(units) = self.compute_units {
            set("sidekick-compute-units", units);
        }
        response
    }
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
    /// Overrides the `type` derived from the status class.
    pub kind: Option<&'static str>,
    /// Sent as a `Retry-After` header (seconds).
    pub retry_after_secs: Option<u64>,
}

impl ApiError {
    pub fn new(status: StatusCode, code: &'static str, message: impl Into<String>) -> Self {
        Self { status, code, message: message.into(), kind: None, retry_after_secs: None }
    }

    pub fn invalid(message: impl Into<String>) -> Self {
        Self::new(StatusCode::BAD_REQUEST, "invalid_request_error", message)
    }

    /// The OpenAI error body, also sent as an SSE event when a stream fails
    /// after it started.
    pub fn body(&self) -> serde_json::Value {
        let kind = self.kind.unwrap_or(if self.status.is_client_error() {
            "invalid_request_error"
        } else {
            "server_error"
        });
        serde_json::json!({
            "error": {
                "message": self.message,
                "type": kind,
                "code": self.code,
            }
        })
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
            Error::InvalidRequest(message) => Self::invalid(message),
            Error::ContextOverflow { limit, actual } => Self::new(
                StatusCode::BAD_REQUEST,
                "context_length_exceeded",
                match actual {
                    Some(actual) => format!(
                        "Request uses {actual} tokens, exceeding the on-device context \
                         budget of {limit} tokens"
                    ),
                    None => format!("Request exceeds the on-device context budget of {limit} tokens"),
                },
            ),
            Error::ContentFiltered(message) => Self::new(
                StatusCode::BAD_REQUEST,
                "content_filter",
                format!("The on-device model's guardrails rejected this request: {message}"),
            ),
            Error::RateLimited { message, retry_after_secs } => Self {
                kind: Some("rate_limit_error"),
                retry_after_secs,
                ..Self::new(StatusCode::TOO_MANY_REQUESTS, "rate_limit_exceeded", message)
            },
            Error::Transient(message) => Self::new(
                StatusCode::SERVICE_UNAVAILABLE,
                "backend_busy",
                format!("The on-device model is busy; try again: {message}"),
            ),
            Error::UnsupportedLanguage(message) => Self::new(
                StatusCode::BAD_REQUEST,
                "unsupported_language",
                message,
            ),
            Error::GuidedGeneration(message) => Self::new(
                StatusCode::BAD_REQUEST,
                "unsupported_schema",
                format!("response_format schema can't be used for guided generation: {message}"),
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
        let body = self.body();
        let mut response = (self.status, Json(body)).into_response();
        if let Some(secs) = self.retry_after_secs {
            response
                .headers_mut()
                .insert(axum::http::header::RETRY_AFTER, secs.into());
        }
        response
    }
}
