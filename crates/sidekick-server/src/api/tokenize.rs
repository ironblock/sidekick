//! `POST /tokenize` and `POST /detokenize`: vLLM's and SGLang's shapes over
//! an embedder's or a text-classification model's tokenizer (D44).
//!
//! The count is the one the model's route compares with its limit: for an
//! embedder, the prompt prefix `input_type` selects (by /v1/embeddings' or
//! /v2/embed's names, with their default, the document prefix) or the
//! named prompt `prompt_name` selects (D45), the text,
//! and the special tokens the backend adds, by the code and the tokenizer
//! the backend uses. It is never truncated: a text past the limit is
//! counted whole, a tokenizer.json's own truncation included. `fits` is the
//! route's own verdict on the text, by the check the route runs, so a
//! client needn't re-derive it from the count. The request body limit
//! (2 MiB) bounds one text, and a byte budget (`BUDGET_BYTES`) bounds the
//! text tokenized at once across requests; the wait for room in it is
//! bounded by the request timeout.

use super::deadline::{Deadline, Work};
use super::wire::*;
use super::{embed_v2, embeddings, ApiError, ApiJson, Provenance};
use crate::state::AppState;
use axum::extract::State;
use axum::response::{IntoResponse, Response};
use axum::Json;
use sidekick_core::{EmbedPurpose, Error, UnavailableReason};
use sidekick_embed::tokenize::{MeasureOptions, ModelTokenizer};
use std::sync::{Arc, LazyLock};
use tokio::sync::Semaphore;

/// Text tokenized at once, across requests. Tokenizing takes memory in
/// proportion to the text: 24 concurrent 2 MiB requests of random
/// alphanumerics (1.6 million tokens each to LFM2.5-Embedding's tokenizer)
/// peaked at 0.45 GB under this budget, the allocator's cache of freed
/// large blocks aside (D44). A request waits until its text fits; one of
/// the largest size the body limit allows always fits alone.
const BUDGET_BYTES: usize = 4 * 1024 * 1024;

/// The budget's unit: a request takes one permit per 64 KiB of text, and
/// at least one.
const BYTES_PER_PERMIT: usize = 64 * 1024;

static BUDGET: LazyLock<Arc<Semaphore>> = LazyLock::new(|| Arc::new(Semaphore::new(BUDGET_BYTES / BYTES_PER_PERMIT)));

/// The permits a request of `bytes` of text takes from the budget.
fn permits(bytes: usize) -> u32 {
    let budget = BUDGET_BYTES / BYTES_PER_PERMIT;
    bytes.div_ceil(BYTES_PER_PERMIT).clamp(1, budget) as u32
}

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
    let (chosen, max_tokens) = match model.kind {
        Kind::Embedder => (
            Some(embeddings::prompt(&state, &req.model, req.input_type.as_deref(), req.prompt_name, purpose)?),
            embed_v2::max_tokens(&state, &req.model, req.max_tokens)?,
        ),
        Kind::Classifier if req.input_type.is_some() || req.prompt_name.is_some() || req.max_tokens.is_some() => {
            return Err(ApiError::invalid(format!(
                "input_type, prompt_name and max_tokens are embedding models' options; model `{}` is a classifier",
                req.model
            )))
        }
        Kind::Classifier => (None, None),
    };
    let add_special_tokens = req.add_special_tokens.unwrap_or(true);
    let token_strs = req.return_token_strs.unwrap_or(false);
    let (max_model_len, measured) = run(&state, &req.model, prompt.len(), move |t| {
        let options = MeasureOptions {
            purpose: chosen.as_ref().map(embeddings::Prompt::purpose),
            max_tokens,
            add_special_tokens,
            token_strs,
        };
        Ok((t.max_model_len(), t.measure(&prompt, options)?))
    })
    .await?;
    Ok(model.provenance.apply(
        Json(TokenizeResponse {
            count: measured.ids.len(),
            max_model_len,
            tokens: measured.ids,
            token_strs: measured.strs,
            fits: measured.fits,
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
    let ids = req.tokens;
    // The text decoded is a few bytes per id.
    let bytes = ids.len().saturating_mul(4);
    let prompt = run(&state, &req.model, bytes, move |t| t.decode(&ids)).await?;
    Ok(model.provenance.apply(Json(DetokenizeResponse { prompt }).into_response()))
}

/// `input_type` by /v1/embeddings' names or /v2/embed's, which map alike:
/// `query`, `search_query`; `document` (the default), `passage`,
/// `search_document`; and `none`, no prompt.
fn purpose(input_type: Option<&str>) -> Result<EmbedPurpose<'static>, ApiError> {
    embeddings::purpose(input_type).or_else(|_| embed_v2::purpose(input_type)).map_err(|_| {
        ApiError::invalid(format!(
            "unsupported input_type `{}` (use `query` or `search_query`; `document`, `passage` or \
             `search_document`; or `none`)",
            input_type.unwrap_or_default()
        ))
    })
}

/// `work` on `bytes` of text with the model's tokenizer, on a blocking
/// thread once the text fits in the budget: the tokenizer's load bounded
/// as a model's is, and the wait for room and then the work by the request
/// timeout (api::deadline). A request that times out waiting never runs;
/// one that times out running gets its 504, while the work finishes and
/// holds its share of the budget until it does.
async fn run<T: Send + 'static>(
    state: &AppState,
    model: &str,
    bytes: usize,
    work: impl FnOnce(&ModelTokenizer) -> sidekick_core::Result<T> + Send + 'static,
) -> Result<T, ApiError> {
    let mut deadline = Deadline::new(Work::Tokenizing, model, state.request_timeout, state.load_timeout);
    let tokenizer = deadline.model(state.tokenizers.get_tracked(model)).await?;
    let turn = deadline
        .within(BUDGET.clone().acquire_many_owned(permits(bytes)))
        .await?
        .map_err(|e| ApiError::from(Error::Other(format!("tokenize budget: {e}"))))?;
    // Nothing loads from here: the loading phase is over.
    let (_, loaded) = tokio::sync::oneshot::channel();
    let task = tokio::task::spawn_blocking(move || {
        let _turn = turn;
        work(&tokenizer)
    });
    deadline.finish(loaded, task, "tokenize").await
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

#[cfg(test)]
mod tests {
    use super::{permits, BUDGET_BYTES, BYTES_PER_PERMIT};

    #[test]
    fn the_largest_body_fits_the_budget_alone() {
        assert_eq!(permits(0), 1);
        assert_eq!(permits(BYTES_PER_PERMIT + 1), 2);
        // The request body limit is 2 MiB, so a prompt is at most that.
        assert_eq!(permits(2 * 1024 * 1024), 32);
        assert!(permits(2 * 1024 * 1024) as usize * BYTES_PER_PERMIT <= BUDGET_BYTES);
        assert_eq!(permits(usize::MAX) as usize, BUDGET_BYTES / BYTES_PER_PERMIT);
    }
}
