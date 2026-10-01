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
use super::{model_task, pooling, wrong_route, ApiError, ApiJson, Provenance};
use crate::state::AppState;
use axum::extract::State;
use axum::http::HeaderMap;
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde_json::Value;
use sidekick_core::manifest::{ClassifierManifest, ClassifyFormat};
use sidekick_core::{
    activate, ClassifyParams, ClassifyTask, Error, ProblemType, QuestionType, TruncationSide,
    UnavailableReason,
};

pub async fn classify(
    State(state): State<AppState>,
    headers: HeaderMap,
    ApiJson(req): ApiJson<ClassifyRequest>,
) -> Result<Response, ApiError> {
    let manifest = match state.registry.classifier(&req.model) {
        Ok(c) if c.manifest.task != ClassifyTask::TextRanking => c.manifest.clone(),
        _ => {
            return Err(match model_task(&state, &req.model) {
                Some(task) => wrong_route(&req.model, task, "/v1/classify"),
                None => ApiError::model_not_found(&req.model),
            })
        }
    };
    if !state.classifiers_supported {
        return Err(Error::Unavailable(UnavailableReason::NotSupportedInBuild).into());
    }
    let request_id = pooling::request_id(&headers, &req.pooling)?;
    pooling::check(&req.pooling)?;
    let inputs = parse_input(&req, &manifest)?;
    let request = parse_params(&req, &manifest)?;
    let (params, temperature, use_activation) = (request.params, request.temperature, request.use_activation);
    let multi_label = request.multi_label;
    let k = match manifest.task {
        ClassifyTask::TextClassification | ClassifyTask::TextRanking => manifest.classify.labels.len(),
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
        ClassifyTask::TextClassification | ClassifyTask::TextRanking => classifier.labels().to_vec(),
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
        // `multi_label` (gliner2) makes this request multi-label; the
        // manifest's problem type is the default.
        let problem = if multi_label { ProblemType::MultiLabel } else { classifier.problem_type() };
        let probs = if use_activation {
            activate(problem, &logits, temperature)
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
        compute_units: Some(manifest.compute_units.name()),
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
    /// gliner2: sigmoid per label instead of the model's softmax.
    multi_label: bool,
}

/// Every other field, checked against the model's task and format before
/// the model loads.
fn parse_params(req: &ClassifyRequest, m: &ClassifierManifest) -> Result<Parsed, ApiError> {
    let laya = m.classify.format == Some(ClassifyFormat::Laya);
    let gliner2 = m.classify.format == Some(ClassifyFormat::Gliner2);
    // Formats that truncate the text themselves, keeping its start.
    let self_truncating = match m.classify.format {
        Some(ClassifyFormat::Laya) => Some("laya"),
        Some(ClassifyFormat::Gliner2) => Some("gliner2"),
        None => None,
    };
    let zero_shot = m.task == ClassifyTask::ZeroShotClassification;
    let unsupported = |field: &str| {
        ApiError::invalid(format!(
            "`{field}` isn't supported by model `{}` ({})",
            m.id,
            match (m.task, m.classify.format) {
                (ClassifyTask::TextClassification, _) => "a text-classification model",
                (_, Some(ClassifyFormat::Laya)) => "a zero-shot model in the laya format",
                (_, Some(ClassifyFormat::Gliner2)) => "a zero-shot model in the gliner2 format",
                _ => "a zero-shot model",
            }
        ))
    };

    if req.add_special_tokens == Some(false) {
        return Err(ApiError::invalid("`add_special_tokens: false` isn't supported"));
    }
    let truncation_side = match req.truncation_side.as_deref() {
        None | Some("right") => TruncationSide::Right,
        Some("left") if self_truncating.is_some() => {
            return Err(ApiError::invalid(format!(
                "`truncation_side: left` isn't supported by the {} format, which keeps the text's start",
                self_truncating.unwrap_or_default()
            )))
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
        Some(_) if self_truncating.is_some() => {
            return Err(ApiError::invalid(format!(
                "`truncate_prompt_tokens` isn't supported by the {} format, which truncates the text itself",
                self_truncating.unwrap_or_default()
            )))
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

    // `multi_label: false` is the default, so every model takes it, as
    // Hugging Face clients send it; `true` is the gliner2 format's alone.
    let multi_label = match (req.multi_label, gliner2) {
        (Some(true), false) => return Err(unsupported("multi_label")),
        (value, _) => value.unwrap_or(false),
    };
    let candidate_labels = match (&req.candidate_labels, zero_shot) {
        (Some(labels), true) => {
            // A multi-label request scores each label alone, so one label
            // is a yes/no question, as gliner2 allows; a softmax needs two.
            let min = if multi_label { 1 } else { 2 };
            sidekick_embed::classify_input::check_labels(labels, min, m.max_labels())?;
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
    if req.instructions.is_some() && !(laya || gliner2) {
        return Err(unsupported("instructions"));
    }
    // The labels must render (noul's are fixed: `false`, `true`), and a
    // model without default instructions needs them; checked here, where
    // the manifest alone can answer, not after the model loads.
    if let (Some(q), Some(section)) = (question_type, m.classify.laya.as_ref()) {
        sidekick_embed::laya::render_options(q, &candidate_labels, section.option_rendering)?;
        if req.instructions.is_none() && section.default_instructions(q).is_none() {
            return Err(ApiError::invalid(sidekick_embed::laya::NO_INSTRUCTIONS));
        }
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
        ClassifyTask::TextClassification | ClassifyTask::TextRanking => m.classify.labels.len(),
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
    Ok(Parsed { params, temperature, use_activation, multi_label })
}
