use super::wire::*;
use crate::state::AppState;
use axum::extract::State;
use axum::Json;
use serde_json::{json, Map, Value};
use sidekick_core::manifest::{RecordedPlacement, ResolvedModel};
use sidekick_core::{ClassifyTask, ComputeUnits, EmbeddingBackendKind};
use sidekick_embed::placement::{this_machine, Placement};
use std::path::Path;

pub async fn list_models(State(state): State<AppState>) -> Json<ModelList> {
    let created = now_unix();
    let mut data: Vec<ModelObject> = Vec::new();

    // The chat model is listed whenever it could plausibly serve requests;
    // /health carries the detailed availability story.
    let availability = state.chat.availability().await;
    if !matches!(
        availability,
        sidekick_core::Availability::Unavailable {
            reason: sidekick_core::UnavailableReason::NotSupportedInBuild
        }
    ) {
        data.push(ModelObject::new(state.chat.id().to_string(), created, "text-generation"));
    }

    for id in state.registry.ids() {
        let r = state.registry.get(id).ok();
        let m = r.map(|r| &r.manifest);
        data.push(ModelObject {
            compute_units: m.map(|m| m.compute_units_name()),
            seq_cap: m.and_then(|m| m.seq_cap.clone()),
            placement: r.and_then(|r| embedder_placement(&state, r, false)),
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
            max_batch: Some(m.max_batch),
            extensions: Some(m.extension_fields()),
            required: Some(m.required_fields()),
            compute_units: Some(m.compute_units.name()),
            seq_cap: m.seq_cap.clone(),
            placement: placement(&state, &c.dir, &m.artifact, &m.buckets, m.compute_units, m.placement.as_ref(), false),
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
/// recorded plan was read, and a failed live read's reason, for /health;
/// /v1/models keeps the counts.
fn placement(
    state: &AppState,
    dir: &Path,
    artifact: &str,
    buckets: &[usize],
    units: ComputeUnits,
    recorded: Option<&RecordedPlacement>,
    detail: bool,
) -> Option<Map<String, Value>> {
    let live = state.placements.as_ref().map(|p| p.for_model(dir, artifact, buckets, units)).unwrap_or_default();
    // A plan recorded for other compute units doesn't describe this model.
    let recorded = recorded.filter(|r| r.compute_units == units);
    let machine = this_machine();
    let reports: Map<String, Value> = buckets
        .iter()
        .filter_map(|&bucket| {
            let live = live.get(&bucket);
            let value = match (live, recorded.and_then(|r| r.bucket(bucket).map(|plan| (r, plan)))) {
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
    placement(state, &r.dir, &m.artifact, &m.buckets, units, m.placement.as_ref(), detail)
}

pub async fn health(State(state): State<AppState>) -> Json<serde_json::Value> {
    let chat_availability = state.chat.availability().await;
    // Fetched before context_limit(), which it refreshes.
    let info = state.chat.model_info().await.unwrap_or_default();
    state.chat_model.set(info.variant_id.clone());
    let embedding_models: Vec<&str> = state.registry.ids().collect();
    let classifier_models: Vec<&str> = state.registry.classifier_ids().collect();
    // Models running with a sequence-length cap (D33), by id.
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
    // Where Core ML places each bucket's operations, by model id.
    let plans: Map<String, Value> = state
        .registry
        .iter()
        .filter_map(|r| embedder_placement(&state, r, true).map(|p| (r.manifest.id.clone(), Value::Object(p))))
        .chain(state.registry.classifiers().filter_map(|c| {
            let m = &c.manifest;
            placement(&state, &c.dir, &m.artifact, &m.buckets, m.compute_units, m.placement.as_ref(), true)
                .map(|p| (m.id.clone(), Value::Object(p)))
        }))
        .collect();
    Json(json!({
        "status": "ok",
        "uptime_secs": state.started_at.elapsed().as_secs(),
        "chat": {
            "model": state.chat.id(),
            "availability": chat_availability,
            "context_limit": state.chat.context_limit(),
            "variant": info.variant,
            "variant_id": info.variant_id,
            // What the underlying model supports; this daemon serves text.
            "model_capabilities": info.capabilities,
            // The SDK the Foundation Models shim was compiled against; the
            // macOS 27 features are only present when it is 27 or newer.
            "fm_sdk": sidekick_fm::FM_SDK,
        },
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
        // Manifests the registry couldn't use, and why.
        "skipped_models": state.registry.skipped(),
        // Models served with a shorter maximum than their manifest's (D33).
        "seq_caps": seq_caps,
        // Where Core ML places each bucket's operations: recorded at
        // conversion, or read live after a bucket's first load.
        "compute_plans": {
            // Whether this daemon reads plans itself (`report_compute_plans`).
            "live": state.placements.is_some(),
            "models": plans,
        },
    }))
}
