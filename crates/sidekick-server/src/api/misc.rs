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
    let embedding_models: Vec<&str> = state.embedders.registry().ids().collect();
    Json(json!({
        "status": "ok",
        "uptime_secs": state.started_at.elapsed().as_secs(),
        "chat": {
            "model": state.chat.id(),
            "availability": chat_availability,
            "context_limit": state.chat.context_limit(),
        },
        "embeddings": {
            "models": embedding_models,
            "resident": state.embedders.resident().await,
        },
    }))
}
