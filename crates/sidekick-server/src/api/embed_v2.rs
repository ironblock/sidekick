//! `POST /v2/embed`: Cohere's v2 embed shape over the registry's embedders,
//! as vLLM serves it (docs/design/rerank.md, D29).

use super::wire::*;
use super::{model_task, pooling, wrong_route, ApiError, ApiJson, Provenance};
use crate::state::AppState;
use axum::extract::State;
use axum::http::HeaderMap;
use axum::response::{IntoResponse, Response};
use axum::Json;
use base64::Engine as _;
use sidekick_core::{truncate_normalized, EmbedLimits, EmbedPurpose, Error, Truncate};

/// Texts per request, as `/v1/embeddings` caps them.
const MAX_TEXTS: usize = 256;

pub async fn embed_v2(
    State(state): State<AppState>,
    headers: HeaderMap,
    ApiJson(req): ApiJson<CohereEmbedRequest>,
) -> Result<Response, ApiError> {
    match model_task(&state, &req.model) {
        Some("feature-extraction") => {}
        Some(task) => return Err(wrong_route(&req.model, task, "/v2/embed")),
        None => return Err(ApiError::model_not_found(&req.model)),
    }
    if req.images.as_ref().is_some_and(|v| !v.is_null()) || req.inputs.as_ref().is_some_and(|v| !v.is_null()) {
        return Err(ApiError::invalid("`images` and `inputs` aren't supported: this daemon embeds text only; send `texts`"));
    }
    let texts = match &req.texts {
        Some(t) if !t.is_empty() => t.clone(),
        _ => return Err(ApiError::invalid("`texts` is required and must not be empty")),
    };
    if texts.len() > MAX_TEXTS {
        return Err(ApiError::invalid(format!("{} texts exceed the maximum of {MAX_TEXTS}", texts.len())));
    }
    // Cohere requires `input_type`; vLLM makes it optional. Absent, the
    // document prefix applies, as on /v1/embeddings (a deliberate default).
    let purpose = match req.input_type.as_deref() {
        None | Some("search_document") | Some("document") => EmbedPurpose::Document,
        Some("search_query") | Some("query") => EmbedPurpose::Query,
        Some(other) => {
            return Err(ApiError::invalid(format!(
                "unsupported input_type `{other}`: embedders here declare prompts for \
                 `search_query` (`query`) and `search_document` (`document`) only"
            )))
        }
    };
    let types: Vec<String> = req.embedding_types.clone().unwrap_or_else(|| vec!["float".into()]);
    if types.is_empty() {
        return Err(ApiError::invalid("`embedding_types` must not be empty"));
    }
    for t in &types {
        match t.as_str() {
            "float" | "base64" | "binary" | "ubinary" => {}
            "int8" | "uint8" => {
                return Err(ApiError::invalid(format!(
                    "embedding type `{t}` isn't supported: it needs calibration ranges (use float, base64, binary or ubinary)"
                )))
            }
            other => return Err(ApiError::invalid(format!("unknown embedding type `{other}`"))),
        }
    }
    let truncate = match req.truncate.as_deref() {
        None | Some("END") => Truncate::End,
        Some("START") => Truncate::Start,
        Some("NONE") => Truncate::Reject,
        Some(other) => {
            return Err(ApiError::invalid(format!("unsupported truncate `{other}` (use NONE, START or END)")))
        }
    };
    // As in vLLM: positive, and within the model's maximum.
    let model_max = state.registry.get(&req.model).map(|m| m.manifest.max_seq_len).unwrap_or(usize::MAX);
    let max_tokens = match req.max_tokens {
        None => None,
        Some(n) if n >= 1 && (n as u64) <= model_max as u64 => Some(n as usize),
        Some(n) if n >= 1 => {
            return Err(ApiError::invalid(format!(
                "max_tokens {n} exceeds model `{}`'s maximum of {model_max} tokens",
                req.model
            )))
        }
        Some(n) => return Err(ApiError::invalid(format!("max_tokens must be positive (got {n})"))),
    };
    pooling::check_priority(&req.priority)?;
    let request_id = match headers.get("x-request-id") {
        Some(h) => h
            .to_str()
            .map(str::to_string)
            .map_err(|_| ApiError::invalid("the X-Request-Id header must be visible ASCII"))?,
        None => uuid::Uuid::new_v4().simple().to_string(),
    };

    let deadline = tokio::time::Instant::now() + state.request_timeout;
    let timeout_err = || Error::Timeout { secs: state.request_timeout.as_secs() };
    let embedder = tokio::time::timeout_at(deadline, state.embedders.get(&req.model))
        .await
        .map_err(|_| timeout_err())??;

    // Matryoshka dimensions, as `dimensions` on /v1/embeddings.
    let target = match req.output_dimension {
        None => None,
        Some(d) if d == embedder.dims() => None,
        Some(d) if embedder.matryoshka_dims().contains(&d) => Some(d),
        Some(d) => {
            let supported = if embedder.matryoshka_dims().is_empty() {
                vec![embedder.dims()]
            } else {
                embedder.matryoshka_dims().to_vec()
            };
            return Err(ApiError::invalid(format!(
                "model `{}` supports output_dimension {supported:?}, got {d}",
                req.model
            )));
        }
    };
    let dims = target.unwrap_or(embedder.dims());
    if types.iter().any(|t| t == "binary" || t == "ubinary") && dims % 8 != 0 {
        return Err(ApiError::invalid(format!(
            "binary embedding types pack 8 dimensions per byte; {dims} isn't a multiple of 8"
        )));
    }

    let limits = EmbedLimits { truncate, max_tokens };
    let vectors = {
        let texts = texts.clone();
        let task = tokio::task::spawn_blocking(move || {
            let refs: Vec<&str> = texts.iter().map(String::as_str).collect();
            embedder.embed_with(&refs, purpose, limits)
        });
        tokio::time::timeout_at(deadline, task)
            .await
            .map_err(|_| timeout_err())?
            .map_err(|e| ApiError::from(Error::Other(format!("embed task: {e}"))))??
    };
    let vectors: Vec<Vec<f32>> = vectors
        .into_iter()
        .map(|v| match target {
            Some(d) => truncate_normalized(&v, d),
            None => v,
        })
        .collect();

    let mut embeddings = CohereEmbeddings::default();
    for t in &types {
        match t.as_str() {
            "float" => embeddings.float = Some(vectors.clone()),
            "base64" => {
                embeddings.base64 = Some(
                    vectors
                        .iter()
                        .map(|v| {
                            let bytes: Vec<u8> = v.iter().flat_map(|f| f.to_le_bytes()).collect();
                            base64::engine::general_purpose::STANDARD.encode(bytes)
                        })
                        .collect(),
                )
            }
            "binary" => {
                embeddings.binary = Some(vectors.iter().map(|v| pack_bits(v).map(|b| b as i32 - 128).collect()).collect())
            }
            "ubinary" => embeddings.ubinary = Some(vectors.iter().map(|v| pack_bits(v).collect()).collect()),
            _ => unreachable!("validated above"),
        }
    }

    // Estimated token usage (~4 characters per token), as /v1/embeddings
    // reports it (D7). vLLM reports real counts; embedders don't return them.
    let approx_tokens: usize = texts.iter().map(|t| t.len() / 4).sum();
    let provenance = Provenance::embedder(&state, &req.model);
    Ok(provenance.apply(
        Json(CohereEmbedResponse {
            id: format!("embd-{request_id}"),
            embeddings,
            texts,
            meta: CohereMeta::v2(approx_tokens as u32),
            response_type: "embeddings_by_type",
        })
        .into_response(),
    ))
}

/// Sign bits (x ≥ 0 → 1), packed MSB-first, eight per byte: Cohere's and
/// vLLM's `ubinary`. `binary` is the same bytes minus 128.
fn pack_bits(v: &[f32]) -> impl Iterator<Item = u8> + '_ {
    v.chunks(8).map(|chunk| {
        chunk
            .iter()
            .enumerate()
            .fold(0u8, |byte, (i, &x)| if x >= 0.0 { byte | (0x80 >> i) } else { byte })
    })
}

#[cfg(test)]
mod tests {
    use super::pack_bits;

    #[test]
    fn packs_sign_bits_msb_first() {
        // numpy.packbits([1,0,0,0,0,0,0,1, 0,1,1,1,1,1,1,1]) == [129, 127]
        let v = [0.5, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 0.0, -0.1, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        assert_eq!(pack_bits(&v).collect::<Vec<_>>(), vec![129, 127]);
        assert_eq!(pack_bits(&v).map(|b| b as i32 - 128).collect::<Vec<_>>(), vec![1, -1]);
    }
}
