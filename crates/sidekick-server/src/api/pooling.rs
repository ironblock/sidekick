//! vLLM's pooling fields shared by `/v1/classify` and the rerank routes
//! (vllm/entrypoints/pooling/base/protocol.py): each is honored, accepted
//! in the form that changes nothing, or a 400 (D22, D28).

use super::wire::PoolingFields;
use super::ApiError;
use axum::http::HeaderMap;
use serde_json::Value;

/// The response id's suffix, as vLLM picks it: the `X-Request-Id` header,
/// else the body's `request_id`, else a random UUID.
pub fn request_id(headers: &HeaderMap, req: &PoolingFields) -> Result<String, ApiError> {
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
pub fn check(req: &PoolingFields) -> Result<(), ApiError> {
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
    check_priority(&req.priority)?;
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

/// Priority scheduling: vLLM errors on any priority but 0 when the model
/// isn't served with it, and sidekick never is. 0 or null is accepted.
pub fn check_priority(priority: &Option<Value>) -> Result<(), ApiError> {
    match priority {
        None | Some(Value::Null) => Ok(()),
        Some(v) if v.is_i64() || v.is_u64() => match v.as_i64() {
            Some(0) => Ok(()),
            _ => Err(ApiError::invalid("`priority` other than 0 isn't supported: sidekick has no priority scheduling")),
        },
        Some(_) => Err(ApiError::invalid("`priority` must be an integer")),
    }
}
