use super::wire::*;
use crate::state::AppState;
use axum::extract::State;
use axum::Json;
use serde_json::json;

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
        data.push(ModelObject {
            id: state.chat.id().to_string(),
            object: "model",
            created,
            owned_by: "sidekick",
        });
    }

    for id in state.embedders.registry().ids() {
        data.push(ModelObject {
            id: id.to_string(),
            object: "model",
            created,
            owned_by: "sidekick",
        });
    }

    Json(ModelList { object: "list", data })
}

pub async fn health(State(state): State<AppState>) -> Json<serde_json::Value> {
    let chat_availability = state.chat.availability().await;
    // Fetched before context_limit(), which it refreshes.
    let info = state.chat.model_info().await.unwrap_or_default();
    let embedding_models: Vec<&str> = state.embedders.registry().ids().collect();
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
    }))
}
