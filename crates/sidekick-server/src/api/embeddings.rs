use super::wire::*;
use super::ApiError;
use crate::state::AppState;
use axum::extract::State;
use axum::Json;
use base64::Engine as _;
use sidekick_core::{truncate_normalized, EmbedPurpose};

/// Hard cap on batch size to keep one request from monopolizing the ANE.
const MAX_BATCH: usize = 256;

pub async fn embeddings(
    State(state): State<AppState>,
    Json(req): Json<EmbeddingsRequest>,
) -> Result<Json<EmbeddingsResponse>, ApiError> {
    let texts: Vec<String> = match req.input {
        EmbeddingsInput::One(s) => vec![s],
        EmbeddingsInput::Many(v) => v,
    };
    if texts.is_empty() {
        return Err(ApiError::invalid("`input` must not be empty"));
    }
    if texts.len() > MAX_BATCH {
        return Err(ApiError::invalid(format!(
            "batch of {} exceeds the maximum of {MAX_BATCH}",
            texts.len()
        )));
    }

    let purpose = match req.input_type.as_deref() {
        None | Some("document") | Some("passage") => EmbedPurpose::Document,
        Some("query") => EmbedPurpose::Query,
        Some(other) => {
            return Err(ApiError::invalid(format!(
                "unsupported input_type `{other}` (use `query` or `document`)"
            )))
        }
    };

    let as_base64 = match req.encoding_format.as_deref() {
        None | Some("float") => false,
        Some("base64") => true,
        Some(other) => {
            return Err(ApiError::invalid(format!(
                "unsupported encoding_format `{other}`"
            )))
        }
    };

    // Bound load + prediction under one request deadline. This abandons the
    // wait, not the work: an in-flight predict runs to completion on its
    // blocking thread, and a timed-out model load still finishes and becomes
    // resident (the pool loads in a detached task), so a retry benefits.
    let deadline = tokio::time::Instant::now() + state.request_timeout;
    let timeout_err =
        || sidekick_core::Error::Timeout { secs: state.request_timeout.as_secs() };

    let embedder = tokio::time::timeout_at(deadline, state.embedders.get(&req.model))
        .await
        .map_err(|_| timeout_err())??;

    // Validate requested dimensions against the model's Matryoshka set.
    let target_dims = match req.dimensions {
        None => None,
        Some(d) if d == embedder.dims() => None,
        Some(d) => {
            if embedder.matryoshka_dims().contains(&d) {
                Some(d)
            } else {
                return Err(ApiError::invalid(format!(
                    "model `{}` supports dimensions {:?}, got {d}",
                    req.model,
                    if embedder.matryoshka_dims().is_empty() {
                        vec![embedder.dims()]
                    } else {
                        embedder.matryoshka_dims().to_vec()
                    },
                )));
            }
        }
    };

    let approx_tokens: usize = texts.iter().map(|t| t.len() / 4).sum();
    let vectors = {
        let texts = texts.clone();
        let task = tokio::task::spawn_blocking(move || {
            let refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
            embedder.embed(&refs, purpose)
        });
        tokio::time::timeout_at(deadline, task)
            .await
            .map_err(|_| timeout_err())?
            .map_err(|e| ApiError::from(sidekick_core::Error::Other(format!("embed task: {e}"))))??
    };

    let data = vectors
        .into_iter()
        .enumerate()
        .map(|(index, v)| {
            let v = match target_dims {
                Some(d) => truncate_normalized(&v, d),
                None => v,
            };
            let embedding = if as_base64 {
                let bytes: Vec<u8> = v.iter().flat_map(|f| f.to_le_bytes()).collect();
                EmbeddingPayload::Base64(base64::engine::general_purpose::STANDARD.encode(bytes))
            } else {
                EmbeddingPayload::Floats(v)
            };
            EmbeddingObject { object: "embedding", index, embedding }
        })
        .collect();

    Ok(Json(EmbeddingsResponse {
        object: "list",
        data,
        model: req.model,
        usage: WireUsage {
            prompt_tokens: approx_tokens as u32,
            completion_tokens: 0,
            total_tokens: approx_tokens as u32,
        },
    }))
}
