use super::wire::*;
use crate::state::AppState;
use axum::extract::State;
use axum::Json;
use serde_json::{json, Map, Value};
use sidekick_core::manifest::{artifact_weight_bytes, RecordedPlacement, ResolvedModel};
use sidekick_core::{
    Chunking, ClassifyTask, ComputeUnits, ComputeUnitsSource, EmbeddingBackendKind, MAX_ANE_PROGRAM_WEIGHT_BYTES,
};
use sidekick_embed::placement::{this_machine, Placement};
use std::path::Path;

pub async fn list_models(State(state): State<AppState>) -> Json<ModelList> {
    let created = now_unix();
    let mut data: Vec<ModelObject> = Vec::new();

    // A chat model is listed whenever it could plausibly serve requests;
    // /health carries the detailed availability story.
    let availability = futures::future::join_all(state.chat.iter().map(|m| m.backend().availability())).await;
    for (m, availability) in state.chat.iter().zip(availability) {
        let not_built = sidekick_core::Availability::Unavailable {
            reason: sidekick_core::UnavailableReason::NotSupportedInBuild,
        };
        if availability == not_built {
            continue;
        }
        let l = &m.listing;
        data.push(ModelObject {
            backend: Some(l.backend),
            context_length: m.backend().context_limit(),
            honored_fields: Some(l.honored_fields.clone()),
            guided_streaming: Some(l.guided_streaming),
            guardrails: Some(l.guardrails),
            compute_units: l.compute_units,
            graded_os: l.graded_os.clone(),
            ..ModelObject::new(m.id().to_string(), created, "text-generation")
        });
    }

    // Each Core ML bucket's state in Core ML's compile cache (D52).
    let warm_targets = crate::warm::targets(&state.registry);
    for id in state.registry.ids() {
        let r = state.registry.get(id).ok();
        let m = r.map(|r| &r.manifest);
        data.push(ModelObject {
            backend: m.map(|m| m.backend.name()),
            compute_units: m.map(|m| m.compute_units_name()),
            compute_units_source: m
                .filter(|m| m.backend == EmbeddingBackendKind::Coreml)
                .map(|m| m.compute_units_source.name()),
            seq_cap: m.and_then(|m| m.seq_cap.clone()),
            prompts: m.map(|m| Prompts { query: m.prefixes.query.clone(), document: m.prefixes.document.clone() }),
            named_prompts: m.map(|m| m.prompts.clone()),
            // Effective values: the registry has already applied any cap (D33).
            max_seq_len: m.map(|m| m.max_seq_len),
            buckets: m.filter(|m| m.backend == EmbeddingBackendKind::Coreml).map(|m| m.buckets.clone()),
            placement: r.and_then(|r| embedder_placement(&state, r, false)),
            compile_cache: state.warmup.compile_cache(&warm_targets, id),
            ..ModelObject::new(id.to_string(), created, "feature-extraction")
        });
    }

    // Classifiers, from their manifests: nothing is loaded to list them.
    // A build without Core ML can't serve them, so it doesn't list them.
    let classifiers = state.registry.classifiers().filter(|_| state.classifiers_supported);
    for c in classifiers {
        let m = &c.manifest;
        let (labels, max_labels) = match m.task {
            ClassifyTask::TextClassification => (Some(m.classify.labels.clone()), None),
            ClassifyTask::ZeroShotClassification => (None, Some(m.max_labels())),
            // One score per pair: nothing to list.
            ClassifyTask::TextRanking => (None, None),
        };
        data.push(ModelObject {
            labels,
            max_labels,
            hypothesis_template: m.classify.nli.as_ref().map(|n| n.hypothesis_template.clone()),
            nli_classes: m.classify.nli.as_ref().and_then(|n| n.classes.clone()),
            max_batch: Some(m.max_batch),
            extensions: Some(m.extension_fields()),
            required: Some(m.required_fields()),
            compute_units: Some(m.compute_units.name()),
            compute_units_source: Some(m.compute_units_source.name()),
            seq_cap: m.seq_cap.clone(),
            max_seq_len: Some(m.max_seq_len),
            placement: placement(&state, &c.dir, &m.artifact, &m.buckets, m.chunks(), m.compute_units, m.placement.as_ref(), false),
            compile_cache: state.warmup.compile_cache(&warm_targets, &m.id),
            calibration: (!m.classify.calibration.is_empty()).then(|| m.classify.calibration.clone()),
            ..ModelObject::new(m.id.clone(), created, m.task.name())
        });
    }

    Json(ModelList { object: "list", data })
}

/// A Core ML model's placement on the ANE, GPU and CPU, by bucket; `None`
/// when there is none to report. Per bucket, the best available:
/// 1. `live`: this daemon read the plan (`report_compute_plans`), after the
///    bucket's first load;
/// 2. `conversion`: the plan the converter recorded in the manifest, for the
///    compute units the model is served with, marked `stale` when it was
///    read on another chip or macOS build;
/// 3. a live read still pending, or failed.
///
/// `detail` adds the unassigned count, the operators off the ANE, where a
/// recorded plan was read, a failed live read's reason, and a chunked
/// bucket's per-chunk counts (D37), for /health; /v1/models keeps the
/// counts. A chunked bucket's counts are its chunks' sums.
#[allow(clippy::too_many_arguments)]
fn placement(
    state: &AppState,
    dir: &Path,
    artifact: &str,
    buckets: &[usize],
    chunks: usize,
    units: ComputeUnits,
    recorded: Option<&RecordedPlacement>,
    detail: bool,
) -> Option<Map<String, Value>> {
    let live =
        state.placements.as_ref().map(|p| p.for_model(dir, artifact, buckets, chunks, units)).unwrap_or_default();
    // Only a plan recorded for the units the model runs on describes it:
    // the main record, or an alternative read for those units (D38).
    let recorded = recorded.filter(|r| r.covers(units));
    let machine = this_machine();
    let reports: Map<String, Value> = buckets
        .iter()
        .filter_map(|&bucket| {
            let live = live.get(&bucket);
            let value = match (live, recorded.and_then(|r| r.bucket_for(units, bucket).map(|plan| (r, plan)))) {
                (Some(Placement::Ready(c)), _) => {
                    let mut v = json!({"source": "live", "state": "ready", "ane": c.ane, "gpu": c.gpu, "cpu": c.cpu});
                    if detail {
                        v["unassigned"] = json!(c.unassigned);
                        v["off_ane_ops"] = json!(c.off_ane_ops);
                    }
                    v
                }
                (live, Some((r, plan))) => {
                    let mut v = json!({"source": "conversion", "state": "ready", "ane": plan.ane, "gpu": plan.gpu, "cpu": plan.cpu});
                    if r.chip != machine.chip || r.macos_build != machine.macos_build {
                        v["stale"] = json!(format!("measured on {}, macOS {}", r.chip, r.macos_build));
                    }
                    if detail {
                        v["unassigned"] = json!(plan.unassigned);
                        v["off_ane_ops"] = json!(plan.off_ane_ops);
                        v["measured_on"] = json!({"chip": r.chip, "macos_build": r.macos_build});
                        if !plan.chunks.is_empty() {
                            v["chunks"] = plan
                                .chunks
                                .iter()
                                .map(|c| json!({"ane": c.ane, "gpu": c.gpu, "cpu": c.cpu, "unassigned": c.unassigned,
                                                "off_ane_ops": c.off_ane_ops}))
                                .collect();
                        }
                        if let Some(date) = &r.date {
                            v["measured_on"]["date"] = json!(date);
                        }
                        match live {
                            Some(Placement::Pending) => v["live"] = json!("pending"),
                            Some(Placement::Failed(e)) => v["live_error"] = json!(e),
                            _ => {}
                        }
                    }
                    v
                }
                (Some(Placement::Pending), None) => json!({"source": "live", "state": "pending"}),
                (Some(Placement::Failed(e)), None) if detail => json!({"source": "live", "state": "error", "error": e}),
                (Some(Placement::Failed(_)), None) => json!({"source": "live", "state": "error"}),
                (None, None) => return None,
            };
            Some((bucket.to_string(), value))
        })
        .collect();
    (!reports.is_empty()).then_some(reports)
}

/// [`placement`] for an embedder: Core ML ones only.
fn embedder_placement(state: &AppState, r: &ResolvedModel, detail: bool) -> Option<Map<String, Value>> {
    let m = &r.manifest;
    if m.backend != EmbeddingBackendKind::Coreml {
        return None;
    }
    let units = m.compute_units.unwrap_or_default();
    placement(state, &r.dir, &m.artifact, &m.buckets, m.chunks(), units, m.placement.as_ref(), detail)
}

/// `/health`'s `onnx` section. It names the runtime's source and file, not
/// its full path, as skipped manifests name theirs relative to the models
/// directory.
fn onnx_health(state: &AppState) -> Value {
    let runtime = match sidekick_embed::onnx::runtime_state() {
        // Loaded only when the models directory has an onnx model (D40).
        None => json!({"state": "not loaded", "reason": "no onnx models"}),
        Some(Ok(r)) => json!({
            "state": "loaded",
            "version": r.version,
            "source": r.source.name(),
            "file": r.library.file_name().map(|f| f.to_string_lossy().into_owned()),
            "warning": r.warning,
        }),
        Some(Err(reason)) => json!({"state": "unavailable", "reason": reason}),
    };
    let onnx: Vec<&sidekick_core::manifest::ModelManifest> = state
        .registry
        .iter()
        .filter(|r| r.manifest.backend == EmbeddingBackendKind::Onnx)
        .map(|r| &r.manifest)
        .collect();
    let onnx_ids: Vec<&str> = onnx.iter().map(|m| m.id.as_str()).collect();
    let models: Map<String, Value> = onnx
        .iter()
        .map(|m| {
            // A quantized graph's output depends on its batch-mates: it
            // always runs one input at a time (D40).
            let model_default = m.onnx.as_ref().and_then(|o| o.default_intra_op_threads);
            let (threads, source) = state.onnx_threads.resolve(&m.id, model_default);
            (
                m.id.clone(),
                json!({"intra_op_threads": threads, "source": source, "model_default": model_default, "quantized": m.onnx_quantized}),
            )
        })
        .collect();
    let unmatched: Vec<&String> =
        state.onnx_threads.by_model.keys().filter(|id| !onnx_ids.contains(&id.as_str())).collect();
    json!({
        "supported": sidekick_embed::onnx::SUPPORTED,
        "pinned_version": sidekick_embed::onnx::PINNED_VERSION,
        "runtime": runtime,
        // The machine's default for a model without its own: its fast
        // cores, capped at 8 (D40).
        "default_intra_op_threads": state.onnx_threads.machine_default(),
        // What the built-in default is based on (D40).
        "detected": {
            "source": sidekick_embed::onnx::detected_cores().source,
            "cores": sidekick_embed::onnx::detected_cores().cores,
        },
        "models": models,
        // Thread settings that named no onnx model.
        "unmatched_thread_overrides": unmatched,
    })
}

/// `/health`'s `chat` section: each chat model's backend, availability
/// and the reason when it is unavailable, context, request timeout and
/// what its backend reports about it.
async fn chat_health(state: &AppState) -> Value {
    let models = futures::future::join_all(state.chat.iter().map(|m| async move {
        let availability = m.backend().availability().await;
        // Fetched before context_limit(), which it refreshes.
        let info = m.refresh_provenance().await.unwrap_or_default();
        let (timeout, source) = m.request_timeout;
        let mut v = json!({
            "backend": m.listing.backend,
            "availability": availability,
            "context_limit": m.backend().context_limit(),
            "request_timeout_secs": timeout.as_secs(),
            // Who chose it: `config (model)` ([models."<id>"]), `manifest`
            // or `config` (request_timeout_secs).
            "request_timeout_source": source,
            "variant": info.variant,
            "variant_id": info.variant_id,
            // What the underlying model supports; this daemon serves text.
            "model_capabilities": info.capabilities,
        });
        v.as_object_mut().unwrap().extend(m.health.clone());
        (m.id().to_string(), v)
    }))
    .await;
    json!({"models": models.into_iter().collect::<Map<String, Value>>()})
}

/// `/health`'s `request_timeout_overrides`: each `[models."<id>"]
/// request_timeout_secs`, and those that name no model this daemon serves.
fn request_timeout_health(state: &AppState) -> Value {
    let by_model = &state.request_timeouts.by_model;
    let served = |id: &str| {
        state.chat.get(id).is_some() || state.registry.get(id).is_ok() || state.registry.classifier(id).is_ok()
    };
    let applied: Map<String, Value> =
        by_model.iter().filter(|(id, _)| served(id)).map(|(id, t)| (id.clone(), json!(t.as_secs()))).collect();
    let unmatched: Vec<&String> = by_model.keys().filter(|id| !served(id)).collect();
    json!({"default_secs": state.request_timeouts.global.as_secs(), "applied": applied, "unmatched": unmatched})
}

pub async fn health(State(state): State<AppState>) -> Json<serde_json::Value> {
    let embedding_models: Vec<&str> = state.registry.ids().collect();
    let classifier_models: Vec<&str> = state.registry.classifier_ids().collect();
    // Models running with a sequence-length cap (D33's on the CPU, D48's on
    // the ANE), by id.
    let seq_caps: serde_json::Map<String, serde_json::Value> = state
        .registry
        .iter()
        .filter_map(|r| r.manifest.seq_cap.as_ref().map(|c| (r.manifest.id.clone(), json!(c))))
        .chain(
            state
                .registry
                .classifiers()
                .filter_map(|r| r.manifest.seq_cap.as_ref().map(|c| (r.manifest.id.clone(), json!(c)))),
        )
        .collect();
    // Models whose compute units the operator chose (D38), by id.
    let overrides: Map<String, Value> = state
        .registry
        .iter()
        .filter(|r| r.manifest.compute_units_source == ComputeUnitsSource::Operator)
        .map(|r| (r.manifest.id.clone(), json!(r.manifest.compute_units_name())))
        .chain(
            state
                .registry
                .classifiers()
                .filter(|c| c.manifest.compute_units_source == ComputeUnitsSource::Operator)
                .map(|c| (c.manifest.id.clone(), json!(c.manifest.compute_units.name()))),
        )
        .collect();
    // Where Core ML places each bucket's operations, by model id.
    let plans: Map<String, Value> = state
        .registry
        .iter()
        .filter_map(|r| embedder_placement(&state, r, true).map(|p| (r.manifest.id.clone(), Value::Object(p))))
        .chain(state.registry.classifiers().filter_map(|c| {
            let m = &c.manifest;
            placement(&state, &c.dir, &m.artifact, &m.buckets, m.chunks(), m.compute_units, m.placement.as_ref(), true)
                .map(|p| (m.id.clone(), Value::Object(p)))
        }))
        .collect();
    // Chunked models (D37): each bucket's programs and their weights, against
    // the ANE's per-program limit and the budget the converter split under.
    let chunked: Map<String, Value> = state
        .registry
        .iter()
        .filter_map(|r| {
            let m = &r.manifest;
            chunking(&r.dir, &m.artifact, &m.buckets, m.chunking.as_ref()).map(|v| (m.id.clone(), v))
        })
        .chain(state.registry.classifiers().filter_map(|c| {
            let m = &c.manifest;
            chunking(&c.dir, &m.artifact, &m.buckets, m.chunking.as_ref()).map(|v| (m.id.clone(), v))
        }))
        .collect();
    Json(json!({
        "status": "ok",
        "uptime_secs": state.started_at.elapsed().as_secs(),
        // The model backends this build serves (`static` always).
        "backends": sidekick_embed::compiled_backends(),
        // Each chat model, by id (D53).
        "chat": chat_health(&state).await,
        // Request timeouts the operator set per model in the daemon config.
        "request_timeout_overrides": request_timeout_health(&state),
        "embeddings": {
            "models": embedding_models,
            "resident": state.embedders.resident().await,
        },
        "classifiers": {
            // False on builds without Core ML: every classify request is a 503.
            "supported": state.classifiers_supported,
            "models": classifier_models,
            "resident": state.classifiers.resident().await,
        },
        // The ONNX Runtime CPU backend (D40): what loaded, and each onnx
        // model's inference threads.
        "onnx": onnx_health(&state),
        // Compute units the operator chose in the daemon config (D38), and
        // overrides that named no Core ML model.
        "compute_unit_overrides": {
            "applied": overrides,
            "unmatched": state.registry.unmatched_overrides(),
        },
        // Manifests the registry couldn't use, and why.
        "skipped_models": state.registry.skipped(),
        // Models served with a shorter maximum than their manifest's (D33, D48).
        "seq_caps": seq_caps,
        // Where Core ML places each bucket's operations: recorded at
        // conversion, or read live after a bucket's first load.
        "compute_plans": {
            // Whether this daemon reads plans itself (`report_compute_plans`).
            "live": state.placements.is_some(),
            "models": plans,
        },
        "chunked_models": chunked,
        // Core ML's compile cache (D52): the startup warm-up's progress,
        // and each Core ML bucket's state, by model.
        "warmup": state.warmup.health(&state.registry),
    }))
}

/// A chunked model's programs (D37), for /health: per bucket, each chunk's
/// artifact and weight bytes; the ANE's per-program limit; and the budget
/// the converter split under, when it recorded one. `None` for a model
/// that isn't chunked.
fn chunking(dir: &Path, artifact: &str, buckets: &[usize], chunking: Option<&Chunking>) -> Option<Value> {
    let c = chunking?;
    let programs: Map<String, Value> = buckets
        .iter()
        .map(|&b| {
            let files = sidekick_core::artifact_files(artifact, b, c.chunks);
            let list: Vec<Value> = files
                .iter()
                .map(|f| json!({"artifact": f, "weight_bytes": artifact_weight_bytes(&dir.join(f))}))
                .collect();
            (b.to_string(), Value::Array(list))
        })
        .collect();
    Some(json!({
        "chunks": c.chunks,
        "ane_limit_bytes": MAX_ANE_PROGRAM_WEIGHT_BYTES,
        "weight_budget_bytes": c.weight_budget_bytes,
        "buckets": programs,
    }))
}
