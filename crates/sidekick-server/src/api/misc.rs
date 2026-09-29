use super::wire::*;
use crate::state::AppState;
use axum::extract::State;
use axum::Json;
use serde_json::json;
use sidekick_core::ClassifyTask;

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
        data.push(ModelObject::new(id.to_string(), created, "feature-extraction"));
    }

    // Classifiers, from their manifests: nothing is loaded to list them.
    for c in state.registry.classifiers() {
        let m = &c.manifest;
        let (task, labels, max_labels) = match m.task {
            ClassifyTask::TextClassification => {
                ("text-classification", Some(m.classify.labels.clone()), None)
            }
            ClassifyTask::ZeroShotClassification => {
                ("zero-shot-classification", None, Some(m.max_labels()))
            }
        };
        data.push(ModelObject {
            labels,
            max_labels,
            max_batch: Some(m.max_batch),
            extensions: Some(m.extension_fields()),
            calibration: (!m.classify.calibration.is_empty()).then(|| m.classify.calibration.clone()),
            ..ModelObject::new(m.id.clone(), created, task)
        });
    }

    Json(ModelList { object: "list", data })
}

pub async fn health(State(state): State<AppState>) -> Json<serde_json::Value> {
    let chat_availability = state.chat.availability().await;
    // Fetched before context_limit(), which it refreshes.
    let info = state.chat.model_info().await.unwrap_or_default();
    let embedding_models: Vec<&str> = state.registry.ids().collect();
    let classifier_models: Vec<&str> = state.registry.classifier_ids().collect();
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
            "models": classifier_models,
            "resident": state.classifiers.resident().await,
        },
        // Manifests the registry couldn't use, and why.
        "skipped_models": state.registry.skipped(),
    }))
}
