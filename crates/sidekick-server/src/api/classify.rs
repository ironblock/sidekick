//! `POST /v1/classify`: vLLM's `/classify` and SGLang's `/v1/classify`,
//! field for field, with sidekick's additive extensions
//! (docs/design/classify.md). Every field vLLM's classify request defines
//! is honored, accepted in the form that changes nothing, or rejected with
//! a 400, never silently dropped (D22); other unknown fields are ignored.
//!
//! A request is checked in full before any work: every field against the
//! model's manifest before the model loads, and every input's `prepare`
//! (which needs the tokenizer, loaded with the model) before any input runs.

use super::wire::*;
use super::{model_task, wrong_route, ApiError, ApiJson, Provenance, CORE_ML_UNITS};
use crate::state::AppState;
use axum::extract::State;
use axum::http::HeaderMap;
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde_json::Value;
use sidekick_core::manifest::{ClassifierManifest, ClassifyFormat};
use sidekick_core::{
    activate, ClassifyParams, ClassifyTask, Error, QuestionType, TruncationSide, UnavailableReason,
};

pub async fn classify(
    State(state): State<AppState>,
    headers: HeaderMap,
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
    if !state.classifiers_supported {
        return Err(Error::Unavailable(UnavailableReason::NotSupportedInBuild).into());
    }
    let request_id = request_id(&headers, &req)?;
    check_pooling_fields(&req)?;
    let inputs = parse_input(&req, &manifest)?;
    let request = parse_params(&req, &manifest)?;
    let (params, temperature, use_activation) = (request.params, request.temperature, request.use_activation);
    let k = match manifest.task {
        ClassifyTask::TextClassification => manifest.classify.labels.len(),
        ClassifyTask::ZeroShotClassification => params.candidate_labels.len(),
    };

    // One deadline for load + prediction, as for embeddings: it abandons
    // the wait, not the work (a timed-out load still becomes resident).
    let deadline = tokio::time::Instant::now() + state.request_timeout;
    let timeout_err = || Error::Timeout { secs: state.request_timeout.as_secs() };
    let classifier = tokio::time::timeout_at(deadline, state.classifiers.get(&req.model))
        .await
        .map_err(|_| timeout_err())??;
    let labels: Vec<String> = match manifest.task {
        ClassifyTask::TextClassification => classifier.labels().to_vec(),
        ClassifyTask::ZeroShotClassification => params.candidate_labels.clone(),
    };

    let task = {
        let classifier = classifier.clone();
        let batch = inputs.len() > 1;
        tokio::task::spawn_blocking(move || {
            // Every input is prepared, and so validated, before any runs.
            let prepared = inputs
                .iter()
                .enumerate()
                .map(|(i, text)| {
                    classifier.prepare(text, &params).map_err(|e| match e {
                        Error::InvalidRequest(m) if batch => Error::InvalidRequest(format!("input {i}: {m}")),
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
            id: format!("classify-{request_id}"),
            object: "list",
            created: now_unix(),
            model: req.model,
            data,
            usage: WireUsage::new(prompt_tokens as u32, 0),
        })
        .into_response(),
    ))
}

/// The response id's suffix, as vLLM picks it: the `X-Request-Id` header,
/// else the body's `request_id`, else a random UUID.
fn request_id(headers: &HeaderMap, req: &ClassifyRequest) -> Result<String, ApiError> {
    if let Some(h) = headers.get("x-request-id") {
        return h
            .to_str()
            .map(str::to_string)
            .map_err(|_| ApiError::invalid("the X-Request-Id header must be visible ASCII"));
    }
    match &req.request_id {
        None => Ok(uuid::Uuid::new_v4().simple().to_string()),
        Some(Value::String(id)) => Ok(id.clone()),
        Some(_) => Err(ApiError::invalid("`request_id` must be a string")),
    }
}

/// vLLM's pooling fields that sidekick has no use for: each is accepted in
/// the form that changes nothing, and otherwise a 400.
fn check_pooling_fields(req: &ClassifyRequest) -> Result<(), ApiError> {
    // vLLM rejects these in any form, with these messages.
    if req.normalize.is_some() {
        return Err(ApiError::invalid("Parameter `normalize` was removed; use `use_activation` instead."));
    }
    match req.task.as_ref().and_then(Value::as_str) {
        Some("score") => return Err(ApiError::invalid("`score` task was removed; use `classify` instead.")),
        Some("encode") => {
            return Err(ApiError::invalid(
                "`encode` task was removed; use `token_embed` or `token_classify` instead.",
            ))
        }
        _ => {}
    }
    // Priority scheduling: vLLM errors on any priority but 0 when the model
    // isn't served with it, and sidekick never is.
    match &req.priority {
        None => {}
        Some(v) if v.as_i64() == Some(0) => {}
        Some(v) if v.is_i64() || v.is_u64() => {
            return Err(ApiError::invalid("`priority` other than 0 isn't supported: sidekick has no priority scheduling"))
        }
        Some(_) => return Err(ApiError::invalid("`priority` must be an integer")),
    }
    // Every classifier here takes an attention mask, so the prompt is never
    // padded (`do_not_pad`). `max_length` would add attended pad tokens.
    match req.padding.as_ref() {
        None => {}
        Some(Value::String(p)) if p == "do_not_pad" => {}
        Some(Value::String(p)) if p == "max_length" => {
            return Err(ApiError::invalid("`padding: max_length` isn't supported; inputs are not padded (`do_not_pad`)"))
        }
        Some(_) => return Err(ApiError::invalid("`padding` must be `max_length` or `do_not_pad`")),
    }
    // A salt keeps vLLM's prefix cache from leaking prompts between users.
    // sidekick keeps no prefix cache, so a valid salt has nothing to change;
    // it's validated as vLLM does, so a request valid here is valid there.
    match &req.cache_salt {
        None => {}
        Some(Value::String(salt))
            if !salt.is_empty()
                && salt.chars().count() <= 128
                && !salt.chars().any(|c| matches!(c, '@' | '/' | '\\' | '\0')) => {}
        Some(_) => {
            return Err(ApiError::invalid(
                "Parameter 'cache_salt' must be a non-empty string of at most 128 characters, \
                 without '@', '/', '\\' or NUL.",
            ))
        }
    }
    // Multimodal processor options: this daemon takes text only.
    match &req.mm_processor_kwargs {
        None => {}
        Some(Value::Object(o)) if o.is_empty() => {}
        Some(_) => return Err(ApiError::invalid("`mm_processor_kwargs` isn't supported: inputs are text only")),
    }
    Ok(())
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

/// A request's classification settings, checked against the model's
/// manifest.
struct Parsed {
    params: ClassifyParams,
    /// The calibration temperature, when `calibration: "model"` applies one.
    temperature: Option<f32>,
    use_activation: bool,
}

/// Every other field, checked against the model's task and format before
/// the model loads.
fn parse_params(req: &ClassifyRequest, m: &ClassifierManifest) -> Result<Parsed, ApiError> {
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
    // laya's noul labels are fixed (`false`, `true`); checked here, where
    // the manifest alone can answer, not after the model loads.
    if let Some(q) = question_type {
        sidekick_embed::laya::render_options(q, &candidate_labels)?;
    }

    // Calibration is an extension of models that declare temperatures;
    // /v1/models lists it for exactly those.
    let use_activation = req.use_activation.unwrap_or(true);
    let calibrate = match req.calibration.as_deref() {
        None => false,
        Some(_) if m.classify.calibration.is_empty() => return Err(unsupported("calibration")),
        Some("none") => false,
        Some("model") => true,
        Some(other) => {
            return Err(ApiError::invalid(format!(
                "unsupported calibration `{other}` (use `none` or `model`)"
            )))
        }
    };
    let k = match m.task {
        ClassifyTask::TextClassification => m.classify.labels.len(),
        ClassifyTask::ZeroShotClassification => candidate_labels.len(),
    };
    // Raw logits (`use_activation: false`) take no temperature.
    let temperature = match (calibrate && use_activation, m.temperature(question_type, k)) {
        (false, _) => None,
        (true, Some(t)) => Some(t),
        (true, None) => {
            return Err(ApiError::invalid(format!(
                "model `{}` declares no calibration temperature for {}{k} labels",
                m.id,
                question_type.map(|q| format!("{} questions with ", qtype_name(q))).unwrap_or_default(),
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
    Ok(Parsed { params, temperature, use_activation })
}
