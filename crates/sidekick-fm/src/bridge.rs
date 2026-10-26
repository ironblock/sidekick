//! What a respond call hands the Swift shim besides the prompt: the
//! guided-generation schema in the shim's own form, and the sampling mode.
//! Platform-neutral, so the mapping is tested everywhere; `ffi` only moves
//! the bytes.

use serde_json::{json, Value};
use sidekick_core::schema::{JsonSchema, Kind, Node};

/// The schema as swift/bridge.swift's `dynamicSchema` reads it. Object
/// properties are a list rather than a JSON object, because a Swift
/// dictionary from `JSONSerialization` doesn't keep key order: the shim
/// builds the generation schema, and so generates the fields, in the order
/// the client sent them.
///
/// A property is optional to Foundation Models when it isn't `required`
/// (JSON Schema's meaning), and when it is nullable: Foundation Models has
/// no null value in a dynamic schema before macOS 26.4, so a nullable
/// property is one the model may leave out, and [`JsonSchema::conform`]
/// writes `null` for it when it is required.
pub fn schema_json(schema: &JsonSchema) -> Value {
    node_json(schema.root(), "Response")
}

/// `name` names object and enum schemas, which Foundation Models requires:
/// the root is "Response", a property's schema takes the property's name,
/// and an array's items take the array's name plus "Item".
fn node_json(node: &Node, name: &str) -> Value {
    let mut out = match &node.kind {
        Kind::Object { properties } => json!({
            "kind": "object",
            "name": name,
            "properties": properties
                .iter()
                .map(|p| {
                    let mut property = json!({
                        "name": p.name,
                        "optional": !p.required || p.nullable,
                        "schema": node_json(&p.schema, &p.name),
                    });
                    if let Some(d) = &p.schema.description {
                        property["description"] = json!(d);
                    }
                    property
                })
                .collect::<Vec<_>>(),
        }),
        Kind::Array {
            items,
            min_items,
            max_items,
        } => json!({
            "kind": "array",
            "items": node_json(items, &format!("{name}Item")),
            "min_items": min_items,
            "max_items": max_items,
        }),
        Kind::Enum(values) => json!({"kind": "enum", "name": name, "values": values}),
        Kind::String => json!({"kind": "string"}),
        Kind::Integer => json!({"kind": "integer"}),
        Kind::Number => json!({"kind": "number"}),
        Kind::Boolean => json!({"kind": "boolean"}),
    };
    if let Some(d) = &node.description {
        out["description"] = json!(d);
    }
    out
}

/// Foundation Models' sampling mode (`GenerationOptions.SamplingMode`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Sampling {
    /// Leave it to Foundation Models (its default: random sampling).
    Default,
    /// Always the most likely token.
    Greedy,
    /// `.random(probabilityThreshold:seed:)`: nucleus sampling, repeatable
    /// when seeded.
    Random {
        probability_threshold: f64,
        seed: Option<u64>,
    },
}

impl Sampling {
    /// The mode for a request's `temperature`, `top_p` and `seed`:
    /// - temperature 0, or `top_p` 0, is greedy (the most likely token is
    ///   the whole nucleus);
    /// - otherwise a `top_p` or a `seed` selects random sampling with that
    ///   threshold (1, the whole distribution, when only `seed` is set);
    /// - otherwise Foundation Models' default.
    pub fn for_request(temperature: Option<f32>, top_p: Option<f32>, seed: Option<u64>) -> Self {
        if temperature == Some(0.0) || top_p == Some(0.0) {
            return Sampling::Greedy;
        }
        if top_p.is_some() || seed.is_some() {
            return Sampling::Random {
                probability_threshold: top_p.map(f64::from).unwrap_or(1.0),
                seed,
            };
        }
        Sampling::Default
    }

    /// The shim's encoding: (kind, probability threshold, seed, has seed),
    /// kind 0 = default, 1 = greedy, 2 = random.
    pub fn ffi(self) -> (i32, f64, u64, i32) {
        match self {
            Sampling::Default => (0, 0.0, 0, 0),
            Sampling::Greedy => (1, 0.0, 0, 0),
            Sampling::Random {
                probability_threshold,
                seed,
            } => (
                2,
                probability_threshold,
                seed.unwrap_or(0),
                seed.is_some() as i32,
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schema_json_keeps_order_and_maps_optionality() {
        let schema = JsonSchema::parse(&json!({
            "type": "object",
            "description": "a reply",
            "properties": {
                "zeta": {"type": "string", "description": "last letter"},
                "alpha": {"type": ["integer", "null"]},
                "tags": {"type": "array", "items": {"type": "string", "enum": ["a", "b"]}, "maxItems": 3},
                "flag": {"type": "boolean"}
            },
            "required": ["zeta", "alpha", "tags"]
        }))
        .unwrap();
        // Properties are a list, so their order is part of the value; the
        // shim reads every other object as a dictionary.
        assert_eq!(
            schema_json(&schema),
            json!({
                "kind": "object", "name": "Response", "description": "a reply",
                "properties": [
                    {"name": "zeta", "optional": false, "description": "last letter",
                     "schema": {"kind": "string", "description": "last letter"}},
                    {"name": "alpha", "optional": true, "schema": {"kind": "integer"}},
                    {"name": "tags", "optional": false, "schema": {
                        "kind": "array", "min_items": null, "max_items": 3,
                        "items": {"kind": "enum", "name": "tagsItem", "values": ["a", "b"]}}},
                    {"name": "flag", "optional": true, "schema": {"kind": "boolean"}}
                ]
            })
        );
    }

    #[test]
    fn sampling_follows_temperature_top_p_and_seed() {
        use Sampling::*;
        assert_eq!(Sampling::for_request(None, None, None), Default);
        assert_eq!(Sampling::for_request(Some(0.7), None, None), Default);
        assert_eq!(Sampling::for_request(Some(0.0), Some(0.9), Some(7)), Greedy);
        assert_eq!(Sampling::for_request(Some(0.8), Some(0.0), None), Greedy);
        assert_eq!(
            Sampling::for_request(Some(0.8), None, Some(7)),
            Random {
                probability_threshold: 1.0,
                seed: Some(7)
            }
        );
        assert_eq!(
            Sampling::for_request(None, Some(0.5), None),
            Random {
                probability_threshold: 0.5,
                seed: None
            }
        );
        assert_eq!(
            Random {
                probability_threshold: 0.5,
                seed: Some(9)
            }
            .ffi(),
            (2, 0.5, 9, 1)
        );
        assert_eq!(Greedy.ffi().0, 1);
        assert_eq!(Default.ffi().0, 0);
    }
}
