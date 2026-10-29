//! Print the instruction an open model's guided request carries in its
//! prompt ([`sidekick_llm::guided::schema_instruction`]), for each JSON
//! Schema given, so a tool can render a guided request's prompt as the
//! daemon does and count its tokens (tools/chat_bench/gate_score.py, for the
//! release gate's usage check).
//!
//! ```sh
//! cargo run -q -p sidekick-llm --example schema_instruction < schemas.json
//! ```
//!
//! Reads a JSON array of schemas on stdin and writes a JSON array of
//! instructions, in the same order, on stdout.

use sidekick_core::schema::JsonSchema;
use std::io::Read;

fn main() {
    let mut input = String::new();
    std::io::stdin().read_to_string(&mut input).expect("reading stdin");
    let schemas: Vec<serde_json::Value> = serde_json::from_str(&input).expect("a JSON array of schemas");
    let out: Vec<String> = schemas
        .iter()
        .map(|s| {
            let schema = JsonSchema::parse(s).unwrap_or_else(|e| panic!("a schema sidekick refuses: {e}"));
            sidekick_llm::guided::schema_instruction(&schema)
        })
        .collect();
    println!("{}", serde_json::to_string(&out).expect("writing JSON"));
}
