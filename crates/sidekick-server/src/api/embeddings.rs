use super::wire::*;
use super::deadline::{signal, Deadline, Work};
use super::{embed_v2, model_task, wrong_route, ApiError, ApiJson, Provenance};
use crate::state::AppState;
use axum::extract::State;
use axum::response::{IntoResponse, Response};
use axum::Json;
use base64::Engine as _;
use serde_json::Value;
use sidekick_core::{truncate_normalized, EmbedLimits, EmbedPurpose, Truncate};

/// Hard cap on batch size to keep one request from monopolizing the ANE.
const MAX_BATCH: usize = 256;

pub async fn embeddings(
    State(state): State<AppState>,
    ApiJson(req): ApiJson<EmbeddingsRequest>,
) -> Result<Response, ApiError> {
    match model_task(&state, &req.model) {
        Some(task) if task != "feature-extraction" => {
            return Err(wrong_route(&req.model, task, "/v1/embeddings"))
        }
        _ => {}
    }
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

    let purpose = purpose(req.input_type.as_deref())?;

    let as_base64 = match req.encoding_format.as_deref() {
        None | Some("float") => false,
        Some("base64") => true,
        Some(other) => {
            return Err(ApiError::invalid(format!(
                "unsupported encoding_format `{other}`"
            )))
        }
    };

    // By default an over-length input refuses the request with a 400
    // naming the input (`NONE`), as OpenAI's embeddings route does (D43);
    // `END` and `START` truncate it.
    let limits = EmbedLimits { truncate: truncate(req.truncate.as_ref())?, max_tokens: None };

    // Loading isn't counted against the request timeout (api::deadline).
    let mut deadline = Deadline::new(Work::Embedding, &req.model, state.request_timeout, state.load_timeout);
    let embedder = deadline.model(state.embedders.get_tracked(&req.model)).await?;

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
    let (vectors, buckets) = {
        let texts = texts.clone();
        let (loaded_tx, loaded) = tokio::sync::oneshot::channel();
        let task = tokio::task::spawn_blocking(move || {
            let refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
            let mut tx = Some(loaded_tx);
            embedder.embed_staged(&refs, purpose, limits, &mut |b| signal(&mut tx, b))
        });
        deadline.finish(loaded, task, "embed").await?
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

    let provenance = Provenance::embedder(&state, &req.model).with_buckets(buckets);
    Ok(provenance.apply(
        Json(EmbeddingsResponse {
            object: "list",
            data,
            model: req.model,
            usage: WireUsage::new(approx_tokens as u32, 0),
        })
        .into_response(),
    ))
}

/// `input_type` on /v1/embeddings, and on /tokenize for an embedder: which
/// of the model's prompt prefixes applies. Absent, the document prefix.
pub(super) fn purpose(input_type: Option<&str>) -> Result<EmbedPurpose, ApiError> {
    match input_type {
        None | Some("document") | Some("passage") => Ok(EmbedPurpose::Document),
        Some("query") => Ok(EmbedPurpose::Query),
        Some(other) => Err(ApiError::invalid(format!("unsupported input_type `{other}` (use `query` or `document`)"))),
    }
}

/// `truncate` on /v1/embeddings: /v2/embed's values in any case, or Hugging
/// Face TEI's boolean, `true` truncating (`END`) and `false` refusing
/// (`NONE`), so that clients built for either work here. Absent (or
/// `null`), it is `NONE`: unlike /v2/embed, whose Cohere contract defaults
/// to `END`, this route refuses over-length input unless asked (D43).
fn truncate(value: Option<&Value>) -> Result<Truncate, ApiError> {
    let unsupported =
        |v: String| ApiError::invalid(format!("unsupported truncate {v} (use NONE, START or END, or true or false)"));
    match value {
        None | Some(Value::Bool(false)) => Ok(Truncate::Reject),
        Some(Value::Bool(true)) => Ok(Truncate::End),
        Some(Value::String(s)) => {
            embed_v2::truncate(Some(&s.to_ascii_uppercase())).map_err(|_| unsupported(format!("`{s}`")))
        }
        Some(other) => Err(unsupported(other.to_string())),
    }
}
