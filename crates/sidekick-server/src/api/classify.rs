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
use super::deadline::{Deadline, Work};
use super::{model_task, pooling, wrong_route, ApiError, ApiJson, Provenance};
use crate::state::AppState;
use axum::extract::State;
use axum::http::HeaderMap;
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde_json::Value;
use sidekick_core::manifest::{ClassifierManifest, ClassifyFormat, NliSection};
use sidekick_core::{
    activate, check_hypothesis_template, nli_value, ClassifyParams, ClassifyTask, Error, Prepared, ProblemType,
    QuestionType, TruncationSide, UnavailableReason,
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
    let nli_classes = request.nli_classes;
    let nli_probs = nli_classes.is_some();
    let k = match manifest.task {
        ClassifyTask::TextClassification | ClassifyTask::TextRanking => manifest.classify.labels.len(),
        ClassifyTask::ZeroShotClassification => params.candidate_labels.len(),
    };

    // Loading isn't counted against the request timeout (api::deadline).
    let mut deadline = Deadline::new(Work::Classification, &req.model, state.request_timeout, state.load_timeout);
    let classifier = deadline.model(state.classifiers.get_tracked(&req.model)).await?;
    let labels: Vec<String> = match manifest.task {
        ClassifyTask::TextClassification | ClassifyTask::TextRanking => classifier.labels().to_vec(),
        ClassifyTask::ZeroShotClassification => params.candidate_labels.clone(),
    };

    let (loaded_tx, loaded) = tokio::sync::oneshot::channel();
    let nli_section = manifest.classify.nli.clone();
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
            let mut buckets: Vec<usize> = prepared.iter().flat_map(|p| p.buckets()).collect();
            buckets.sort_unstable();
            buckets.dedup();
            let _ = loaded_tx.send(classifier.load_buckets(&buckets)?);
            prepared
                .iter()
                .map(|p| {
                    if !nli_probs {
                        return Ok((p.tokens(), p.bucket, classifier.run(p)?, None));
                    }
                    // The pairs' class logits, from which the values `run`
                    // would return are derived as it derives them: one
                    // prediction per pair serves both (D47).
                    let pairs = classifier.run_nli(p)?;
                    let values = nli_values(&pairs, p, nli_section.as_ref())?;
                    Ok((p.tokens(), p.bucket, values, Some(pairs)))
                })
                .collect::<Result<Vec<_>, Error>>()
        })
    };
    let results = deadline.finish(loaded, task, "classify").await?;

    let mut prompt_tokens = 0;
    let mut data = Vec::with_capacity(results.len());
    let mut buckets = Vec::with_capacity(results.len());
    for (index, (tokens, bucket, logits, pairs)) in results.into_iter().enumerate() {
        prompt_tokens += tokens;
        buckets.push(bucket);
        if logits.len() != k || logits.iter().any(|x| !x.is_finite()) {
            return Err(Error::Inference(format!(
                "model `{}` returned {} values ({}) for {k} labels",
                req.model,
                logits.len(),
                if logits.iter().all(|x| x.is_finite()) { "all finite" } else { "non-finite" },
            ))
            .into());
        }
        // `multi_label` (gliner2, nli, and nli's single label) makes this
        // request multi-label; the manifest's problem type is the default.
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
        let nli_probs = match (pairs, &nli_classes) {
            (Some(pairs), Some(classes)) => Some(class_probs(&req.model, &pairs, classes, use_activation)?),
            _ => None,
        };
        data.push(ClassifyData { index, label: labels[best].clone(), probs, num_classes: k, nli_probs });
    }

    let provenance = Provenance {
        model: Provenance::model_id(&req.model, manifest.source.as_ref()),
        compute_units: Some(manifest.compute_units.name()),
        buckets: Some(buckets),
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

/// The values `run` returns for an nli input, from its pairs' class logits
/// (`Classifier::run_nli`): [`nli_value`] of each, so `probs` is the same
/// with or without `nli_probs`.
fn nli_values(pairs: &[Vec<f32>], p: &Prepared, nli: Option<&NliSection>) -> Result<Vec<f32>, Error> {
    let (Some(input), Some(nli)) = (&p.nli, nli) else {
        return Err(Error::Inference("an nli input was prepared without its pairs".into()));
    };
    if pairs.len() != input.pairs.len() {
        return Err(Error::Inference(format!("{} pairs returned {} results", input.pairs.len(), pairs.len())));
    }
    pairs
        .iter()
        .map(|logits| {
            if logits.len() != nli.n_classes {
                return Err(Error::Inference(format!(
                    "a pair returned {} values for {} classes",
                    logits.len(),
                    nli.n_classes
                )));
            }
            Ok(nli_value(logits, nli.entailment, nli.contradiction, input.per_label))
        })
        .collect()
}

/// `nli_probs`: each pair's softmax over the model's classes, keyed by
/// class name in output order; the logits themselves with
/// `use_activation: false`, as `probs` then carries logits.
fn class_probs(model: &str, pairs: &[Vec<f32>], classes: &[String], use_activation: bool) -> Result<Vec<ClassProbs>, ApiError> {
    pairs
        .iter()
        .map(|logits| {
            if logits.iter().any(|x| !x.is_finite()) {
                return Err(Error::Inference(format!("model `{model}` returned non-finite class logits")).into());
            }
            let values = if use_activation { activate(ProblemType::SingleLabel, logits, None) } else { logits.clone() };
            Ok(ClassProbs(classes.iter().cloned().zip(values).collect()))
        })
        .collect()
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
    /// gliner2 and nli: sigmoid per label instead of the model's softmax.
    /// For nli, also a request with one label, as Hugging Face scores it.
    multi_label: bool,
    /// nli with `nli_probs: true`: the model's classes, which key them.
    nli_classes: Option<Vec<String>>,
}

/// Every other field, checked against the model's task and format before
/// the model loads.
fn parse_params(req: &ClassifyRequest, m: &ClassifierManifest) -> Result<Parsed, ApiError> {
    let laya = m.classify.format == Some(ClassifyFormat::Laya);
    let gliner2 = m.classify.format == Some(ClassifyFormat::Gliner2);
    let fev = m.classify.format == Some(ClassifyFormat::Fev);
    // agentjev never truncates: an over-long tree is a 400, as AgentJev
    // refuses an over-long path.
    let agentjev = m.classify.format == Some(ClassifyFormat::Agentjev);
    // nli truncates the text itself, as Hugging Face's pipeline does, but
    // on either side and to a lower limit when asked.
    let nli = m.classify.format == Some(ClassifyFormat::Nli);
    // Formats that truncate the text themselves, keeping its start.
    let self_truncating = match m.classify.format {
        Some(ClassifyFormat::Laya) => Some("laya"),
        Some(ClassifyFormat::Gliner2) => Some("gliner2"),
        Some(ClassifyFormat::Fev) => Some("fev"),
        Some(ClassifyFormat::Agentjev | ClassifyFormat::Nli) | None => None,
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
                (_, Some(ClassifyFormat::Fev)) => "a zero-shot model in the fev format",
                (_, Some(ClassifyFormat::Agentjev)) => "a zero-shot model in the agentjev format",
                (_, Some(ClassifyFormat::Nli)) => "a zero-shot model in the nli format",
                _ => "a zero-shot model",
            }
        ))
    };

    if req.add_special_tokens == Some(false) {
        return Err(ApiError::invalid("`add_special_tokens: false` isn't supported"));
    }
    if agentjev && (req.truncate_prompt_tokens.is_some() || req.truncation_side.as_deref() == Some("left")) {
        return Err(ApiError::invalid(
            "the agentjev format never truncates, so `truncate_prompt_tokens` and `truncation_side: left` \
             aren't supported; shorten the input instead",
        ));
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
    // Hugging Face clients send it; `true` is the gliner2 and nli formats'.
    let multi_label = match (req.multi_label, gliner2 || nli) {
        (Some(true), false) => return Err(unsupported("multi_label")),
        (value, _) => value.unwrap_or(false),
    };
    // `nli_probs: false`, the default, is accepted everywhere, as
    // `multi_label: false` is; `true` needs an nli model that names its
    // classes (D47).
    let nli_classes = match (req.nli_probs, nli) {
        (None | Some(false), _) => None,
        (Some(true), false) => return Err(unsupported("nli_probs")),
        (Some(true), true) => match m.classify.nli.as_ref().and_then(|n| n.classes.clone()) {
            Some(classes) => Some(classes),
            None => {
                return Err(ApiError::invalid(format!(
                    "`nli_probs` isn't supported by model `{}`: its manifest doesn't name its classes \
                     (`[classify.nli] classes`)",
                    m.id
                )))
            }
        },
    };
    let hypothesis_template = match (&req.hypothesis_template, nli) {
        (Some(t), true) => {
            check_hypothesis_template(t).map_err(ApiError::invalid)?;
            Some(t.clone())
        }
        (Some(_), false) => return Err(unsupported("hypothesis_template")),
        (None, _) => None,
    };
    let candidate_labels = match (&req.candidate_labels, zero_shot) {
        (Some(labels), true) => {
            // A multi-label request scores each label alone, so one label
            // is a yes/no question, as gliner2 allows; a softmax needs two.
            // nli takes one label either way: Hugging Face's pipeline
            // scores a lone label against its contradiction.
            let min = if multi_label || nli { 1 } else { 2 };
            sidekick_embed::classify_input::check_labels(labels, min, m.max_labels())?;
            labels.clone()
        }
        (None, true) => return Err(ApiError::invalid(format!("model `{}` needs `candidate_labels`", m.id))),
        (Some(_), false) => return Err(unsupported("candidate_labels")),
        (None, false) => vec![],
    };
    let question_type = match (req.question_type.as_deref(), laya || fev || agentjev) {
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
    if req.instructions.is_some() && !(laya || gliner2 || fev || agentjev) {
        return Err(unsupported("instructions"));
    }
    if agentjev && req.instructions.is_none() {
        return Err(ApiError::invalid(format!(
            "model `{}` needs `instructions`: the question to decide",
            m.id
        )));
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
    if let (Some(q), true) = (question_type, fev) {
        sidekick_embed::fev::render_options(q, &candidate_labels)?;
    }
    if let (Some(q), true) = (question_type, agentjev) {
        sidekick_embed::agentjev::render_candidates(q, &candidate_labels)?;
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

    // nli scores a lone label on its own, whatever `multi_label` says.
    let per_label = multi_label || (nli && candidate_labels.len() == 1);
    let params = ClassifyParams {
        truncate_prompt_tokens,
        truncation_side,
        candidate_labels,
        question_type,
        instructions: req.instructions.clone(),
        multi_label,
        hypothesis_template,
    };
    Ok(Parsed { params, temperature, use_activation, multi_label: per_label, nli_classes })
}

#[cfg(test)]
mod tests {
    use super::*;
    use sidekick_core::{NliPair, NliPairs};

    fn names(list: &[&str]) -> Vec<String> {
        list.iter().map(|s| s.to_string()).collect()
    }

    fn section(entailment: usize, contradiction: usize, classes: &[&str]) -> NliSection {
        NliSection {
            entailment,
            contradiction,
            n_classes: classes.len(),
            classes: Some(names(classes)),
            hypothesis_template: "This example is {}.".into(),
        }
    }

    fn prepared(k: usize, per_label: bool) -> Prepared {
        let pair = NliPair { ids: vec![1, 2, 3], type_ids: vec![], bucket: 16 };
        Prepared {
            ids: vec![],
            type_ids: vec![],
            markers: vec![],
            qtype: None,
            decide_pos: None,
            seg: vec![],
            position_ids: vec![],
            nli: Some(NliPairs { pairs: vec![pair; k], per_label }),
            bucket: 16,
        }
    }

    #[test]
    fn class_probs_are_each_pairs_softmax_keyed_in_output_order() {
        // nli-deberta-v3's order, and a two-way model's.
        let three = names(&["contradiction", "entailment", "neutral"]);
        let pairs = vec![vec![0.5f32, 2.0, 0.1], vec![-1.0, -2.0, 3.0]];
        let got = class_probs("m", &pairs, &three, true).unwrap();
        for (dist, logits) in got.iter().zip(&pairs) {
            let softmax = activate(ProblemType::SingleLabel, logits, None);
            let sum: f32 = dist.0.iter().map(|(_, p)| p).sum();
            assert!((sum - 1.0).abs() < 1e-6, "{dist:?}");
            for ((class, p), (want_class, want)) in dist.0.iter().zip(three.iter().zip(&softmax)) {
                assert_eq!((class, p), (want_class, want));
            }
        }
        // By hand: e^2 / (e^0.5 + e^2 + e^0.1).
        let e = |x: f32| x.exp();
        assert!((got[0].0[1].1 - e(2.0) / (e(0.5) + e(2.0) + e(0.1))).abs() < 1e-6);
        // A JSON object, its keys in the model's output order.
        let json = serde_json::to_string(&ClassProbs(vec![("entailment".into(), 0.25), ("not_entailment".into(), 0.75)]))
            .unwrap();
        assert_eq!(json, r#"{"entailment":0.25,"not_entailment":0.75}"#);
        // Raw logits with `use_activation: false`; non-finite ones are a 500.
        let raw = class_probs("m", &pairs, &three, false).unwrap();
        assert_eq!(raw[1].0.iter().map(|(_, v)| *v).collect::<Vec<_>>(), pairs[1]);
        assert!(class_probs("m", &[vec![f32::NAN, 0.0, 0.0]], &three, true).is_err());
    }

    #[test]
    fn nli_values_are_what_the_classifier_returns_from_the_same_logits() {
        let pairs = vec![vec![0.5f32, 0.1, 2.0], vec![1.5, 0.3, -0.5]];
        // roberta-large-mnli's order: entailment 2, contradiction 0.
        let mnli = section(2, 0, &["contradiction", "neutral", "entailment"]);
        assert_eq!(nli_values(&pairs, &prepared(2, false), Some(&mnli)).unwrap(), vec![2.0, -0.5]);
        assert_eq!(nli_values(&pairs, &prepared(2, true), Some(&mnli)).unwrap(), vec![1.5, -2.0]);
        // A two-way model: entailment 0 against not_entailment 1.
        let two = section(0, 1, &["entailment", "not_entailment"]);
        assert_eq!(nli_values(&[vec![1.0, 4.0]], &prepared(1, true), Some(&two)).unwrap(), vec![-3.0]);
        // The wrong width, or a pair missing, is the model's fault.
        assert!(nli_values(&[vec![1.0, 4.0]], &prepared(1, true), Some(&mnli)).is_err());
        assert!(nli_values(&pairs, &prepared(3, false), Some(&mnli)).is_err());
    }
}
