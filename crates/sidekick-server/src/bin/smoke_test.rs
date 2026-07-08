//! Hardware smoke test for the Foundation Models backend.
//!
//! Run on a Mac with Apple Intelligence enabled:
//!
//! ```sh
//! cargo run -p sidekick-server --bin smoke-test
//! ```
//!
//! Exercises the real Swift shim end-to-end: availability probe, plain
//! completion, session-reuse follow-up, and schema-constrained generation.
//! Exits non-zero on the first failure so it can gate a release or run in
//! a self-hosted CI job. This is the check `docs/DECISIONS.md` lists under
//! "Needs hardware verification".

use sidekick_core::{Availability, ChatBackend, ChatMessage, ChatRequest, Role};
use sidekick_fm::fm_backend;
use std::time::{Duration, Instant};

fn req(messages: Vec<ChatMessage>) -> ChatRequest {
    ChatRequest { messages, ..Default::default() }
}

#[tokio::main]
async fn main() {
    let backend = fm_backend(Duration::from_secs(300), Duration::from_secs(60));

    println!("== availability ==");
    let availability = backend.availability().await;
    match &availability {
        Availability::Available => println!("Foundation Models: available"),
        Availability::Unavailable { reason } => {
            eprintln!("FAIL: Foundation Models unavailable: {reason:?}");
            std::process::exit(1);
        }
    }

    println!("\n== plain completion ==");
    let start = Instant::now();
    let first = vec![ChatMessage::new(Role::User, "What is 2+2? Answer with just the number.")];
    let r1 = match backend.complete(req(first.clone())).await {
        Ok(r) => r,
        Err(e) => {
            eprintln!("FAIL: plain completion errored: {e}");
            std::process::exit(1);
        }
    };
    println!("[{:?}] {:?} (usage: {:?})", start.elapsed(), r1.content, r1.usage);
    if r1.content.trim().is_empty() {
        eprintln!("FAIL: plain completion returned empty content");
        std::process::exit(1);
    }

    println!("\n== follow-up (session reuse) ==");
    let start = Instant::now();
    let mut second = first;
    second.push(ChatMessage::new(Role::Assistant, r1.content.clone()));
    second.push(ChatMessage::new(Role::User, "Now double it. Just the number."));
    match backend.complete(req(second)).await {
        Ok(r) => {
            println!("[{:?}] {:?}", start.elapsed(), r.content);
            if r.content.trim().is_empty() {
                eprintln!("FAIL: follow-up returned empty content");
                std::process::exit(1);
            }
        }
        Err(e) => {
            eprintln!("FAIL: follow-up errored: {e}");
            std::process::exit(1);
        }
    }

    println!("\n== constrained generation (json_schema) ==");
    let start = Instant::now();
    let mut constrained = req(vec![ChatMessage::new(
        Role::User,
        "Name one planet in our solar system and its position from the sun.",
    )]);
    constrained.schema = Some(serde_json::json!({
        "type": "object",
        "properties": {
            "planet": { "type": "string" },
            "position": { "type": "integer" }
        },
        "required": ["planet", "position"]
    }));
    match backend.complete(constrained).await {
        Ok(r) => {
            println!("[{:?}] {:?} (constrained: {})", start.elapsed(), r.content, r.constrained);
            if !r.constrained {
                eprintln!("FAIL: response not marked constrained");
                std::process::exit(1);
            }
            let parsed: serde_json::Value = match serde_json::from_str(&r.content) {
                Ok(v) => v,
                Err(e) => {
                    eprintln!("FAIL: constrained output is not valid JSON: {e}");
                    std::process::exit(1);
                }
            };
            if !(parsed.get("planet").is_some_and(|v| v.is_string())
                && parsed.get("position").is_some_and(|v| v.is_i64()))
            {
                eprintln!("FAIL: constrained output missing required fields: {parsed}");
                std::process::exit(1);
            }
        }
        Err(e) => {
            eprintln!("FAIL: constrained completion errored: {e}");
            std::process::exit(1);
        }
    }

    println!("\nSMOKE TEST PASSED");
}
