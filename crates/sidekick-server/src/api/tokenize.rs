//! `POST /tokenize` and `POST /detokenize`: vLLM's and SGLang's shapes over
//! an embedder's or a text-classification model's tokenizer (D44).
//!
//! The count is the one the model's route compares with its limit: for an
//! embedder, the prompt prefix `input_type` selects (as on
//! /v1/embeddings: the document prefix by default), the text, and the
//! special tokens the backend adds, by the code /v1/embeddings tokenizes
//! with (`EmbedInput`). It is never truncated: a text past the limit is
//! counted whole, a tokenizer.json's own truncation included, so a client
//! can see by how much. The request body limit (2 MiB) bounds the text.
//! The tokenizer loads on its own, without the model's weights.

use super::wire::*;
use super::{embeddings, ApiError, ApiJson, Provenance};
use crate::state::AppState;
use axum::extract::State;
use axum::response::{IntoResponse, Response};
use axum::Json;
use sidekick_core::{EmbedPurpose, Error, UnavailableReason};
use sidekick_embed::tokenize::ModelTokenizer;
use std::sync::Arc;

pub async fn tokenize(
    State(state): State<AppState>,
    ApiJson(req): ApiJson<TokenizeRequest>,
) -> Result<Response, ApiError> {
    let model = served(&state, &req.model, "/tokenize")?;
    if req.messages.as_ref().is_some_and(|m| !m.is_null()) {
        return Err(ApiError::invalid(
            "`messages` isn't supported: embedding and classification models have no chat template; send `prompt`",
        ));
    }
    let Some(prompt) = req.prompt else {
        return Err(ApiError::invalid("`prompt` is required"));
    };
    let purpose: Option<EmbedPurpose> = match model.kind {
        // The same reading of `input_type`, and the same default, as
        // /v1/embeddings'.
        Kind::Embedder => Some(embeddings::purpose(req.input_type.as_deref())?),
        Kind::Classifier if req.input_type.is_some() => {
            return Err(ApiError::invalid(format!(
                "input_type selects an embedding model's prompt prefix; model `{}` is a classifier",
                req.model
            )))
        }
        Kind::Classifier => None,
    };
    let add_special_tokens = req.add_special_tokens.unwrap_or(true);
    let tokenizer = state.tokenizers.get(&req.model).await?;
    let max_model_len = tokenizer.max_model_len();
    let tokens = blocking(tokenizer, move |t| t.tokenize(&prompt, purpose, add_special_tokens)).await?;
    Ok(model.provenance.apply(
        Json(TokenizeResponse {
            count: tokens.ids.len(),
            max_model_len,
            tokens: tokens.ids,
            token_strs: req.return_token_strs.unwrap_or(false).then_some(tokens.strs),
        })
        .into_response(),
    ))
}

/// Token ids back to text, special tokens kept, for any embedder or
/// classifier: decoding doesn't depend on how a route lays its input out.
pub async fn detokenize(
    State(state): State<AppState>,
    ApiJson(req): ApiJson<DetokenizeRequest>,
) -> Result<Response, ApiError> {
    let model = served(&state, &req.model, "/detokenize")?;
    let tokenizer = state.tokenizers.get(&req.model).await?;
    let ids = req.tokens;
    let prompt = blocking(tokenizer, move |t| t.decode(&ids)).await?;
    Ok(model.provenance.apply(Json(DetokenizeResponse { prompt }).into_response()))
}

enum Kind {
    Embedder,
    Classifier,
}

struct Served {
    kind: Kind,
    provenance: Provenance,
}

/// The model, if this build serves it and `route` takes it, from its
/// manifest: nothing loads to refuse a request.
fn served(state: &AppState, id: &str, route: &str) -> Result<Served, ApiError> {
    let provenance = |source| Provenance { model: Provenance::model_id(id, source), compute_units: None, buckets: None };
    let unavailable = || ApiError::from(Error::Unavailable(UnavailableReason::NotSupportedInBuild));
    if let Ok(m) = state.registry.get(id) {
        // As /v1/embeddings: a model this build can't run is a 503.
        if !sidekick_embed::compiled_backends().contains(&m.manifest.backend.name()) {
            return Err(unavailable());
        }
        return Ok(Served { kind: Kind::Embedder, provenance: provenance(m.manifest.source.as_ref()) });
    }
    if let Ok(c) = state.registry.classifier(id) {
        // As /v1/classify.
        if !state.classifiers_supported {
            return Err(unavailable());
        }
        if route == "/tokenize" {
            sidekick_embed::classify_input::check_tokenizable(c.manifest.task, c.manifest.classify.format)?;
        }
        return Ok(Served { kind: Kind::Classifier, provenance: provenance(c.manifest.source.as_ref()) });
    }
    if id == state.chat.id() {
        return Err(ApiError::invalid(format!(
            "model `{id}` runs on Apple's Foundation Models, which report token counts but not token ids, \
             so {route} can't serve it"
        )));
    }
    Err(ApiError::model_not_found(id))
}

/// `work` on a blocking thread: tokenizing a long text takes a while.
async fn blocking<T: Send + 'static>(
    tokenizer: Arc<ModelTokenizer>,
    work: impl FnOnce(&ModelTokenizer) -> sidekick_core::Result<T> + Send + 'static,
) -> Result<T, ApiError> {
    let result = tokio::task::spawn_blocking(move || work(&tokenizer))
        .await
        .map_err(|e| ApiError::from(Error::Other(format!("tokenize task failed: {e}"))))?;
    Ok(result?)
}
