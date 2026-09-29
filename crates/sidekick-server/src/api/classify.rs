//! `POST /v1/classify`: vLLM's `/classify` and SGLang's `/v1/classify`,
//! field for field, with sidekick's additive extensions
//! (docs/design/classify.md). Every field vLLM defines is honored or
//! rejected with a 400, never silently dropped (D22); other unknown fields
//! are ignored.

use super::wire::*;
use super::{model_task, wrong_route, ApiError, ApiJson, Provenance, CORE_ML_UNITS};
use crate::state::AppState;
use axum::extract::State;
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde_json::Value;
use sidekick_core::manifest::{ClassifierManifest, ClassifyFormat};
use sidekick_core::{activate, ClassifyParams, ClassifyTask, Error, QuestionType, TruncationSide};

pub async fn classify(
    State(state): State<AppState>,
    ApiJson(req): ApiJson<ClassifyRequest>,
) -> Result<Response, ApiError> {
    let manifest = match state.registry.classifier(&req.model) {
        Ok(c) => c.manifest.clone(),
        Err(_) => {
            return Err(match model_task(&state, &req.model) {
                Some(task) => wrong_route(&req.model, task, "/v1/classify"),
                None => ApiError::model_not_found(&req.model),
            })
        }
    };
    let inputs = parse_input(&req, &manifest)?;
    let (params, calibrate, use_activation) = parse_params(&req, &manifest)?;

    // One deadline for load + prediction, as for embeddings: it abandons
    // the wait, not the work (a timed-out load still becomes resident).
    let deadline = tokio::time::Instant::now() + state.request_timeout;
    let timeout_err = || Error::Timeout { secs: state.request_timeout.as_secs() };
    let classifier = tokio::time::timeout_at(deadline, state.classifiers.get(&req.model))
        .await
        .map_err(|_| timeout_err())??;

    let k = match manifest.task {
        ClassifyTask::TextClassification => classifier.labels().len(),
        ClassifyTask::ZeroShotClassification => params.candidate_labels.len(),
    };
    let temperature = if calibrate {
        Some(classifier.calibration(&params, k).ok_or_else(|| {
            ApiError::invalid(format!(
                "model `{}` declares no calibration temperature for {}{k} labels",
                req.model,
                params.question_type.map(|q| format!("{} questions with ", qtype_name(q))).unwrap_or_default(),
            ))
        })?)
    } else {
        None
    };
    let labels: Vec<String> = match manifest.task {
        ClassifyTask::TextClassification => classifier.labels().to_vec(),
        ClassifyTask::ZeroShotClassification => params.candidate_labels.clone(),
    };

    let task = {
        let classifier = classifier.clone();
        let params = params.clone();
        tokio::task::spawn_blocking(move || {
            inputs
                .iter()
                .map(|text| {
                    let prepared = classifier.prepare(text, &params)?;
                    let logits = classifier.run(&prepared)?;
                    Ok((prepared.ids.len(), logits))
                })
                .collect::<Result<Vec<_>, Error>>()
        })
    };
    let results = tokio::time::timeout_at(deadline, task)
        .await
        .map_err(|_| timeout_err())?
        .map_err(|e| ApiError::from(Error::Other(format!("classify task: {e}"))))??;

    let mut prompt_tokens = 0;
    let mut data = Vec::with_capacity(results.len());
    for (index, (tokens, logits)) in results.into_iter().enumerate() {
        prompt_tokens += tokens;
        if logits.len() != k || logits.iter().any(|x| !x.is_finite()) {
            return Err(Error::Inference(format!(
                "model `{}` returned {} values ({}) for {k} labels",
                req.model,
                logits.len(),
                if logits.iter().all(|x| x.is_finite()) { "all finite" } else { "non-finite" },
            ))
            .into());
        }
        let probs = if use_activation {
            activate(classifier.problem_type(), &logits, temperature)
        } else {
            logits
        };
        // The argmax of `probs` is the argmax of the logits: every
        // activation is monotonic.
        let best = probs
            .iter()
            .enumerate()
            .fold(0, |best, (i, &p)| if p > probs[best] { i } else { best });
        data.push(ClassifyData { index, label: labels[best].clone(), probs, num_classes: k });
    }

    let provenance = Provenance {
        model: Provenance::model_id(&req.model, manifest.source.as_ref()),
        compute_units: Some(CORE_ML_UNITS),
    };
    Ok(provenance.apply(
        Json(ClassifyResponse {
            id: format!("classify-{}", uuid::Uuid::new_v4().simple()),
            object: "list",
            created: now_unix(),
            model: req.model,
            data,
            usage: WireUsage::new(prompt_tokens as u32, 0),
        })
        .into_response(),
    ))
}

fn qtype_name(q: QuestionType) -> &'static str {
    match q {
        QuestionType::Choice => "choice",
        QuestionType::Score => "score",
        QuestionType::Noul => "noul",
    }
}

/// `input`: a string or an array of strings, at most the model's
/// `max_batch`. vLLM's token-id and chat-form inputs are 400s.
fn parse_input(req: &ClassifyRequest, m: &ClassifierManifest) -> Result<Vec<String>, ApiError> {
    if req.messages.as_ref().is_some_and(|v| !v.is_null()) {
        return Err(ApiError::invalid("chat-form `messages` input isn't supported; send `input`"));
    }
    let not_text = || ApiError::invalid("`input` must be a string or an array of strings (token-id input isn't supported)");
    let inputs = match &req.input {
        None | Some(Value::Null) => return Err(ApiError::invalid("`input` is required")),
        Some(Value::String(s)) => vec![s.clone()],
        Some(Value::Array(items)) => items
            .iter()
            .map(|v| v.as_str().map(str::to_string).ok_or_else(not_text))
            .collect::<Result<_, _>>()?,
        Some(_) => return Err(not_text()),
    };
    if inputs.is_empty() {
        return Err(ApiError::invalid("`input` must not be empty"));
    }
    if inputs.len() > m.max_batch {
        return Err(ApiError::invalid(format!(
            "batch of {} exceeds model `{}`'s maximum of {}",
            inputs.len(),
            m.id,
            m.max_batch
        )));
    }
    Ok(inputs)
}

/// Every other field, checked against the model's task and format. Returns
/// the classifier's params, whether to calibrate, and whether to activate.
fn parse_params(
    req: &ClassifyRequest,
    m: &ClassifierManifest,
) -> Result<(ClassifyParams, bool, bool), ApiError> {
    let laya = m.classify.format == Some(ClassifyFormat::Laya);
    let zero_shot = m.task == ClassifyTask::ZeroShotClassification;
    let unsupported = |field: &str| {
        ApiError::invalid(format!(
            "`{field}` isn't supported by model `{}` ({})",
            m.id,
            match (m.task, laya) {
                (ClassifyTask::TextClassification, _) => "a text-classification model",
                (_, true) => "a zero-shot model in the laya format",
                _ => "a zero-shot model",
            }
        ))
    };

    if req.add_special_tokens == Some(false) {
        return Err(ApiError::invalid("`add_special_tokens: false` isn't supported"));
    }
    let truncation_side = match req.truncation_side.as_deref() {
        None | Some("right") => TruncationSide::Right,
        Some("left") if laya => {
            return Err(ApiError::invalid(
                "`truncation_side: left` isn't supported by the laya format, which keeps the text's start",
            ))
        }
        Some("left") => TruncationSide::Left,
        Some(other) => {
            return Err(ApiError::invalid(format!(
                "unsupported truncation_side `{other}` (use `right` or `left`)"
            )))
        }
    };
    let truncate_prompt_tokens = match req.truncate_prompt_tokens {
        None => None,
        Some(_) if laya => {
            return Err(ApiError::invalid(
                "`truncate_prompt_tokens` isn't supported by the laya format, which truncates the text itself",
            ))
        }
        // vLLM: -1 truncates to the model's maximum.
        Some(-1) => Some(m.max_seq_len),
        Some(n) if n >= 1 && n as u64 <= m.max_seq_len as u64 => Some(n as usize),
        Some(n) => {
            return Err(ApiError::invalid(format!(
                "truncate_prompt_tokens must be -1 or 1..={} (got {n})",
                m.max_seq_len
            )))
        }
    };

    let candidate_labels = match (&req.candidate_labels, zero_shot) {
        (Some(labels), true) => {
            sidekick_embed::classify_input::check_labels(labels, m.max_labels())?;
            labels.clone()
        }
        (None, true) => return Err(ApiError::invalid(format!("model `{}` needs `candidate_labels`", m.id))),
        (Some(_), false) => return Err(unsupported("candidate_labels")),
        (None, false) => vec![],
    };
    let question_type = match (req.question_type.as_deref(), laya) {
        (Some(q), true) => Some(match q {
            "choice" => QuestionType::Choice,
            "score" => QuestionType::Score,
            "noul" => QuestionType::Noul,
            other => {
                return Err(ApiError::invalid(format!(
                    "unsupported question_type `{other}` (use `choice`, `score` or `noul`)"
                )))
            }
        }),
        (None, true) => {
            return Err(ApiError::invalid(format!(
                "model `{}` needs `question_type` (`choice`, `score` or `noul`)",
                m.id
            )))
        }
        (Some(_), false) => return Err(unsupported("question_type")),
        (None, false) => None,
    };
    if req.instructions.is_some() && !laya {
        return Err(unsupported("instructions"));
    }
    let calibrate = match req.calibration.as_deref() {
        None | Some("none") => false,
        Some("model") => true,
        Some(other) => {
            return Err(ApiError::invalid(format!(
                "unsupported calibration `{other}` (use `none` or `model`)"
            )))
        }
    };

    let params = ClassifyParams {
        truncate_prompt_tokens,
        truncation_side,
        candidate_labels,
        question_type,
        instructions: req.instructions.clone(),
    };
    Ok((params, calibrate, req.use_activation.unwrap_or(true)))
}
