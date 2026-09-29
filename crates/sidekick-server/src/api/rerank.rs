//! `POST /v1/rerank` and `/rerank` (vLLM's `RerankRequest`/`RerankResponse`,
//! the Jina shape) and `POST /v2/rerank` (Cohere's v2 shape, answered with
//! the superset both clients parse). docs/design/rerank.md, D29.
//!
//! A reranker is a `text-ranking` classifier: one relevance score per
//! (query, document) pair. As for classify, a request is checked in full
//! before any work, and every pair is prepared before any runs.

use super::wire::*;
use super::{model_task, pooling, wrong_route, ApiError, ApiJson, Provenance, CORE_ML_UNITS};
use crate::state::AppState;
use axum::extract::State;
use axum::http::HeaderMap;
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde_json::Value;
use sidekick_core::manifest::ClassifierManifest;
use sidekick_core::{activate, ClassifyTask, Error, PairParams, TruncationSide, UnavailableReason};

/// Which contract a route serves.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Shape {
    /// vLLM / Jina: `/v1/rerank`, `/rerank`.
    V1,
    /// Cohere v2: `/v2/rerank`.
    V2,
}

pub async fn rerank_v1(
    state: State<AppState>,
    headers: HeaderMap,
    req: ApiJson<RerankRequest>,
) -> Result<Response, ApiError> {
    rerank(state, headers, req, Shape::V1).await
}

pub async fn rerank_v2(
    state: State<AppState>,
    headers: HeaderMap,
    req: ApiJson<RerankRequest>,
) -> Result<Response, ApiError> {
    rerank(state, headers, req, Shape::V2).await
}

async fn rerank(
    State(state): State<AppState>,
    headers: HeaderMap,
    ApiJson(req): ApiJson<RerankRequest>,
    shape: Shape,
) -> Result<Response, ApiError> {
    let route = if shape == Shape::V1 { "/v1/rerank" } else { "/v2/rerank" };
    let manifest = match state.registry.classifier(&req.model) {
        Ok(c) if c.manifest.task == ClassifyTask::TextRanking => c.manifest.clone(),
        _ => {
            return Err(match model_task(&state, &req.model) {
                Some(task) => wrong_route(&req.model, task, route),
                None => ApiError::model_not_found(&req.model),
            })
        }
    };
    if !state.classifiers_supported {
        return Err(Error::Unavailable(UnavailableReason::NotSupportedInBuild).into());
    }
    let request_id = pooling::request_id(&headers, &req.pooling)?;
    pooling::check(&req.pooling)?;
    let parsed = parse(&req, &manifest, shape)?;

    // One deadline for load + prediction, as for classify.
    let deadline = tokio::time::Instant::now() + state.request_timeout;
    let timeout_err = || Error::Timeout { secs: state.request_timeout.as_secs() };
    let classifier = tokio::time::timeout_at(deadline, state.classifiers.get(&req.model))
        .await
        .map_err(|_| timeout_err())??;

    let task = {
        let classifier = classifier.clone();
        let (query, documents, params) = (parsed.query.clone(), parsed.documents.clone(), parsed.params.clone());
        let batch = documents.len() > 1;
        tokio::task::spawn_blocking(move || {
            // Every pair is prepared, and so validated, before any runs.
            let prepared = documents
                .iter()
                .enumerate()
                .map(|(i, doc)| {
                    classifier.prepare_pair(&query, doc, &params).map_err(|e| match e {
                        Error::InvalidRequest(m) if batch => Error::InvalidRequest(format!("document {i}: {m}")),
                        e => e,
                    })
                })
                .collect::<Result<Vec<_>, Error>>()?;
            prepared
                .iter()
                .map(|p| Ok((p.ids.len(), classifier.run(p)?)))
                .collect::<Result<Vec<_>, Error>>()
        })
    };
    let outputs = tokio::time::timeout_at(deadline, task)
        .await
        .map_err(|_| timeout_err())?
        .map_err(|e| ApiError::from(Error::Other(format!("rerank task: {e}"))))??;

    let mut prompt_tokens = 0usize;
    let mut scored = Vec::with_capacity(outputs.len());
    for (index, (tokens, logits)) in outputs.into_iter().enumerate() {
        prompt_tokens += tokens;
        let [logit] = logits[..] else {
            return Err(Error::Inference(format!(
                "model `{}` returned {} values for a pair; a reranker returns one",
                req.model,
                logits.len()
            ))
            .into());
        };
        if !logit.is_finite() {
            return Err(Error::Inference(format!("model `{}` returned a non-finite score", req.model)).into());
        }
        let score = if parsed.use_activation {
            activate(classifier.problem_type(), &[logit], None)[0]
        } else {
            logit
        };
        let document = parsed.return_documents.then(|| RerankDocument { text: parsed.documents[index].clone() });
        // `+ 0.0` turns -0.0 into 0.0, which total_cmp would order below it.
        scored.push((logit + 0.0, RerankResult { index, document, relevance_score: score }));
    }
    // Highest first, as vLLM and Cohere sort; ties keep request order. By
    // the raw logit: every activation is monotonic, and sigmoid saturates
    // to 1.0 above a logit of about 17, where it would tie strong matches.
    scored.sort_by(|a, b| b.0.total_cmp(&a.0));
    let mut results: Vec<RerankResult> = scored.into_iter().map(|(_, r)| r).collect();
    if let Some(n) = parsed.top_n {
        results.truncate(n);
    }

    let prompt_tokens = prompt_tokens as u32;
    let provenance = Provenance {
        model: Provenance::model_id(&req.model, manifest.source.as_ref()),
        compute_units: Some(CORE_ML_UNITS),
    };
    Ok(provenance.apply(
        Json(RerankResponse {
            // vLLM serves rerank from its scoring handler, prefix `score`.
            id: format!("score-{request_id}"),
            model: req.model,
            usage: RerankUsage { prompt_tokens, total_tokens: prompt_tokens },
            results,
            meta: (shape == Shape::V2).then(|| CohereMeta::v2(prompt_tokens)),
        })
        .into_response(),
    ))
}

/// A request's settings, checked against the model's manifest.
struct Parsed {
    query: String,
    documents: Vec<String>,
    /// `None`: every result.
    top_n: Option<usize>,
    use_activation: bool,
    return_documents: bool,
    params: PairParams,
}

fn parse(req: &RerankRequest, m: &ClassifierManifest, shape: Shape) -> Result<Parsed, ApiError> {
    let query = match &req.query {
        Some(Value::String(q)) => q.clone(),
        None | Some(Value::Null) => return Err(ApiError::invalid("`query` is required")),
        Some(_) => return Err(ApiError::invalid("`query` must be a string (multimodal input isn't supported)")),
    };
    let not_text = || {
        ApiError::invalid("`documents` must be a string or an array of strings (document objects and multimodal input aren't supported)")
    };
    let documents: Vec<String> = match &req.documents {
        None | Some(Value::Null) => return Err(ApiError::invalid("`documents` is required")),
        // vLLM accepts a single document as well as a list.
        Some(Value::String(d)) => vec![d.clone()],
        Some(Value::Array(items)) => items
            .iter()
            .map(|v| v.as_str().map(str::to_string).ok_or_else(not_text))
            .collect::<Result<_, _>>()?,
        Some(_) => return Err(not_text()),
    };
    if documents.is_empty() {
        return Err(ApiError::invalid("`documents` must not be empty"));
    }
    if documents.len() > m.max_batch {
        return Err(ApiError::invalid(format!(
            "{} documents exceed model `{}`'s maximum of {}",
            documents.len(),
            m.id,
            m.max_batch
        )));
    }
    // vLLM: an integer ≥ 0, where 0 means every result. Cohere: omitted.
    let top_n = match &req.top_n {
        None | Some(Value::Null) => None,
        Some(v) => match v.as_u64() {
            Some(0) => None,
            Some(n) => Some(n as usize),
            None => return Err(ApiError::invalid("`top_n` must be a non-negative integer")),
        },
    };
    if req.instruction.as_ref().is_some_and(|v| !v.is_null())
        || req.chat_template_kwargs.as_ref().is_some_and(|v| !v.is_null())
    {
        return Err(ApiError::invalid(format!(
            "`instruction` and `chat_template_kwargs` feed a chat template, and model `{}` (a cross-encoder) has none",
            m.id
        )));
    }

    let max = m.max_seq_len;
    let truncation_side = match req.truncation_side.as_deref() {
        None | Some("right") => TruncationSide::Right,
        Some("left") => TruncationSide::Left,
        Some(other) => {
            return Err(ApiError::invalid(format!(
                "unsupported truncation_side `{other}` (use `right` or `left`)"
            )))
        }
    };
    let truncate_prompt_tokens = match req.truncate_prompt_tokens {
        None => None,
        // vLLM: -1 truncates to the model's maximum.
        Some(-1) => Some(max),
        Some(n) if n >= 1 && n as u64 <= max as u64 => Some(n as usize),
        Some(n) => {
            return Err(ApiError::invalid(format!(
                "truncate_prompt_tokens must be -1 or 1..={max} (got {n})"
            )))
        }
    };
    // vLLM: 0 is off; otherwise less than the model's maximum.
    let token_limit = |name: &str, v: Option<i64>| match v {
        None | Some(0) => Ok(None),
        Some(n) if n > 0 && (n as u64) < max as u64 => Ok(Some(n as usize)),
        // Cohere's `max_tokens_per_doc` (default 4096) is a ceiling, not a
        // demand: anything the model can't hold is the model's maximum.
        Some(n) if n > 0 && shape == Shape::V2 && name == "max_tokens_per_doc" => Ok(Some(max)),
        Some(n) => Err(ApiError::invalid(format!(
            "{name} must be 0 (off) or 1..{max}, less than the model's maximum (got {n})"
        ))),
    };
    let max_tokens_per_query = token_limit("max_tokens_per_query", req.max_tokens_per_query)?;
    let max_tokens_per_doc = match (shape, req.max_tokens_per_doc) {
        // Cohere's default.
        (Shape::V2, None) => Some(4096.min(max)),
        (_, v) => token_limit("max_tokens_per_doc", v)?,
    };
    Ok(Parsed {
        query,
        documents,
        top_n,
        use_activation: req.use_activation.unwrap_or(true),
        // Cohere v2 has no `return_documents`, and vLLM's result requires
        // `document`: /v2 always returns it and ignores the field.
        return_documents: shape == Shape::V2 || req.return_documents.unwrap_or(true),
        params: PairParams {
            truncate_prompt_tokens,
            truncation_side,
            max_tokens_per_query,
            max_tokens_per_doc,
            // Cohere's route truncates documents unless the client asked
            // for vLLM's truncation explicitly.
            keep_query: shape == Shape::V2 && truncate_prompt_tokens.is_none(),
        },
    })
}
