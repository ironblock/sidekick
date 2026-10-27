//! The chat route's JSON Schema layer: one supported subset for every chat
//! backend.
//!
//! [`JsonSchema::parse`] reads `response_format.json_schema.schema` into an
//! ordered, validated representation, or refuses it with an
//! [`Error::UnsupportedSchema`] naming the keyword and its JSON pointer.
//! Backends consume the representation; none reads the request's schema
//! itself, so a schema means the same thing whichever model answers. The
//! subset, and why each keyword is in or out, is documented in
//! docs/API.md ("Structured output").
//!
//! The representation keeps `properties` in the order the request sent
//! them (the workspace builds `serde_json` with `preserve_order`), and
//! [`JsonSchema::conform`] re-serializes a reply in that order, so guided
//! output is byte-identical across runs whatever order a backend produced
//! its keys in.

use crate::{Error, Result};
use serde_json::{Map, Value};

/// A validated schema in the supported subset.
#[derive(Debug, Clone, PartialEq)]
pub struct JsonSchema {
    root: Node,
}

/// One schema in the tree: what it generates, and its description.
#[derive(Debug, Clone, PartialEq)]
pub struct Node {
    pub kind: Kind,
    /// The schema's `description`, which a backend may show its model.
    pub description: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Kind {
    /// A closed object: only the listed properties, in this order.
    Object {
        properties: Vec<Property>,
    },
    Array {
        items: Box<Node>,
        min_items: Option<u64>,
        max_items: Option<u64>,
    },
    String,
    /// One of these strings (`enum` of strings, or a `const` string).
    Enum(Vec<String>),
    Integer,
    Number,
    Boolean,
}

/// An object property.
#[derive(Debug, Clone, PartialEq)]
pub struct Property {
    pub name: String,
    pub schema: Node,
    /// Listed in the object's `required`.
    pub required: bool,
    /// May be `null` (`type: [T, "null"]`, or `anyOf` of a schema and
    /// `{"type": "null"}`). Nullable schemas are supported only here, as an
    /// object property's value.
    pub nullable: bool,
}

/// Keywords that annotate a schema without constraining it.
const ANNOTATIONS: &[&str] = &["description", "title", "$schema", "examples", "default"];

/// Every keyword the subset accepts somewhere.
const KEYWORDS: &[&str] = &[
    "type",
    "properties",
    "required",
    "additionalProperties",
    "enum",
    "const",
    "items",
    "minItems",
    "maxItems",
    "anyOf",
];

/// Why some well-known keywords are refused, for a more useful message.
fn refusal_reason(keyword: &str) -> &'static str {
    match keyword {
        "oneOf" | "allOf" | "not" | "if" | "then" | "else" => {
            "schema composition is not supported (`anyOf` only as a nullable union)"
        }
        "$ref" | "$defs" | "definitions" | "$id" | "$anchor" | "$dynamicRef" => {
            "references are not supported; write the schema inline"
        }
        "minimum" | "maximum" | "exclusiveMinimum" | "exclusiveMaximum" | "multipleOf" => {
            "number bounds are not supported"
        }
        "pattern" | "format" | "minLength" | "maxLength" | "contentEncoding"
        | "contentMediaType" => "string constraints are not supported",
        "patternProperties"
        | "propertyNames"
        | "minProperties"
        | "maxProperties"
        | "dependentRequired"
        | "dependentSchemas"
        | "unevaluatedProperties" => "only closed objects with listed properties are supported",
        "prefixItems" | "contains" | "minContains" | "maxContains" | "uniqueItems"
        | "unevaluatedItems" | "additionalItems" => {
            "only `items`, `minItems` and `maxItems` are supported on arrays"
        }
        _ => "not in the supported subset",
    }
}

fn refuse(keyword: &str, pointer: &str, reason: impl std::fmt::Display) -> Error {
    let at = if pointer.is_empty() {
        "the schema root".to_string()
    } else {
        format!("`{pointer}`")
    };
    Error::UnsupportedSchema(format!("unsupported keyword `{keyword}` at {at}: {reason}"))
}

/// Append a JSON pointer segment (RFC 6901 escaping).
fn child(pointer: &str, segment: &str) -> String {
    format!(
        "{pointer}/{}",
        segment.replace('~', "~0").replace('/', "~1")
    )
}

/// What a parsed schema allows: a node, and whether `null` is allowed too.
struct Parsed {
    node: Node,
    nullable: bool,
    /// The keyword that made it nullable, for the refusal where it can't be.
    nullable_by: &'static str,
}

impl JsonSchema {
    /// Validate a JSON Schema against the supported subset.
    pub fn parse(schema: &Value) -> Result<Self> {
        let parsed = parse_node(schema, "")?;
        if parsed.nullable {
            return Err(refuse(parsed.nullable_by, "", NULLABLE_ONLY_AS_PROPERTY));
        }
        Ok(Self { root: parsed.node })
    }

    pub fn root(&self) -> &Node {
        &self.root
    }

    /// Re-serialize a reply that matches this schema in schema order:
    /// object keys in the order `properties` lists them (any key the schema
    /// doesn't list keeps its place after them), and `null` written for a
    /// required nullable property the reply left out. A reply that isn't
    /// JSON (a guided reply cut short by `max_tokens`) is returned as it is.
    pub fn conform(&self, reply: &str) -> String {
        match serde_json::from_str::<Value>(reply) {
            Ok(value) => conform_value(&self.root, value).to_string(),
            Err(_) => reply.to_string(),
        }
    }

    /// Check that `value` matches this schema, as a guided reply must: an
    /// error names the first mismatch and its JSON pointer.
    pub fn validate(&self, value: &Value) -> std::result::Result<(), String> {
        validate_node(&self.root, value, "")
    }
}

fn validate_node(node: &Node, value: &Value, at: &str) -> std::result::Result<(), String> {
    let fail = |what: String| Err(format!("{}: {what}", if at.is_empty() { "the root" } else { at }));
    match (&node.kind, value) {
        (Kind::Object { properties }, Value::Object(map)) => {
            if let Some(key) = map.keys().find(|k| !properties.iter().any(|p| &p.name == *k)) {
                return fail(format!("`{key}` isn't one of the object's properties"));
            }
            for p in properties {
                match map.get(&p.name) {
                    None if p.required => return fail(format!("the required property `{}` is missing", p.name)),
                    None => {}
                    Some(Value::Null) if p.nullable => {}
                    Some(v) => validate_node(&p.schema, v, &child(at, &p.name))?,
                }
            }
            Ok(())
        }
        (Kind::Array { items, min_items, max_items }, Value::Array(values)) => {
            let n = values.len() as u64;
            if min_items.is_some_and(|m| n < m) || max_items.is_some_and(|m| n > m) {
                return fail(format!("{n} items, outside minItems {min_items:?} and maxItems {max_items:?}"));
            }
            for (i, v) in values.iter().enumerate() {
                validate_node(items, v, &child(at, &i.to_string()))?;
            }
            Ok(())
        }
        (Kind::String, Value::String(_)) | (Kind::Number, Value::Number(_)) | (Kind::Boolean, Value::Bool(_)) => Ok(()),
        (Kind::Enum(choices), Value::String(s)) if choices.contains(s) => Ok(()),
        (Kind::Integer, Value::Number(n)) if n.is_i64() || n.is_u64() || n.as_f64().is_some_and(|f| f.fract() == 0.0) => {
            Ok(())
        }
        (kind, value) => fail(format!("{value} doesn't match {}", kind_name(kind))),
    }
}

fn kind_name(kind: &Kind) -> String {
    match kind {
        Kind::Object { .. } => "an object".into(),
        Kind::Array { .. } => "an array".into(),
        Kind::String => "a string".into(),
        Kind::Enum(choices) => format!("one of {choices:?}"),
        Kind::Integer => "an integer".into(),
        Kind::Number => "a number".into(),
        Kind::Boolean => "a boolean".into(),
    }
}

const NULLABLE_ONLY_AS_PROPERTY: &str =
    "a nullable schema is supported only as an object property's value";

fn conform_value(node: &Node, value: Value) -> Value {
    match (&node.kind, value) {
        (Kind::Object { properties }, Value::Object(mut map)) => {
            let mut out = Map::with_capacity(map.len());
            for p in properties {
                match map.remove(&p.name) {
                    Some(Value::Null) => {
                        out.insert(p.name.clone(), Value::Null);
                    }
                    Some(v) => {
                        out.insert(p.name.clone(), conform_value(&p.schema, v));
                    }
                    None if p.required && p.nullable => {
                        out.insert(p.name.clone(), Value::Null);
                    }
                    None => {}
                }
            }
            out.extend(map);
            Value::Object(out)
        }
        (Kind::Array { items, .. }, Value::Array(values)) => Value::Array(
            values
                .into_iter()
                .map(|v| conform_value(items, v))
                .collect(),
        ),
        (_, value) => value,
    }
}

fn parse_node(schema: &Value, pointer: &str) -> Result<Parsed> {
    let obj = match schema {
        Value::Object(obj) => obj,
        Value::Bool(_) => {
            return Err(Error::UnsupportedSchema(format!(
                "a boolean schema at {} is not supported; write an object schema",
                if pointer.is_empty() {
                    "the schema root".into()
                } else {
                    format!("`{pointer}`")
                }
            )))
        }
        _ => {
            return Err(Error::UnsupportedSchema(format!(
                "the schema at {} must be a JSON object",
                if pointer.is_empty() {
                    "the schema root".into()
                } else {
                    format!("`{pointer}`")
                }
            )))
        }
    };
    for key in obj.keys() {
        if !KEYWORDS.contains(&key.as_str()) && !ANNOTATIONS.contains(&key.as_str()) {
            return Err(refuse(key, pointer, refusal_reason(key)));
        }
    }
    let description = match obj.get("description") {
        None => None,
        Some(Value::String(s)) => Some(s.clone()),
        Some(_) => return Err(refuse("description", pointer, "must be a string")),
    };

    if let Some(any_of) = obj.get("anyOf") {
        return parse_nullable_any_of(obj, any_of, pointer, description);
    }

    // `type`: one name, or two with "null".
    let mut nullable = false;
    let mut nullable_by = "type";
    // `nullable_by` names the keyword a refusal points at.
    let type_name: Option<&str> = match obj.get("type") {
        None => None,
        Some(Value::String(t)) => Some(t.as_str()),
        Some(Value::Array(names)) => {
            let mut rest = Vec::new();
            for name in names {
                match name.as_str() {
                    Some("null") => nullable = true,
                    Some(t) if !rest.contains(&t) => rest.push(t),
                    Some(_) => {}
                    None => return Err(refuse("type", pointer, "type names must be strings")),
                }
            }
            match rest.as_slice() {
                [one] => Some(*one),
                [] => None,
                _ => return Err(refuse(
                    "type",
                    pointer,
                    "a union of types is not supported (only one type, optionally with \"null\")",
                )),
            }
        }
        Some(_) => {
            return Err(refuse(
                "type",
                pointer,
                "must be a string or an array of strings",
            ))
        }
    };
    if type_name == Some("null") || (type_name.is_none() && nullable) {
        return Err(refuse(
            "type",
            pointer,
            "a schema that allows only null is not supported",
        ));
    }

    // `enum` / `const`: strings, and null in an enum. A value must satisfy
    // both `type` and `enum`, so the schema is nullable when every one of
    // them that is present allows null.
    let mut enum_has_null = false;
    let choices = match (obj.get("enum"), obj.get("const")) {
        (Some(_), Some(_)) => {
            return Err(refuse("const", pointer, "use `enum` or `const`, not both"))
        }
        (Some(Value::Array(values)), None) => {
            let mut choices = Vec::new();
            for v in values {
                match v {
                    Value::String(s) => {
                        if !choices.contains(s) {
                            choices.push(s.clone())
                        }
                    }
                    Value::Null => enum_has_null = true,
                    _ => return Err(refuse("enum", pointer, "only string values are supported")),
                }
            }
            if choices.is_empty() {
                return Err(refuse("enum", pointer, "needs at least one string value"));
            }
            Some(("enum", choices))
        }
        (Some(_), None) => return Err(refuse("enum", pointer, "must be an array")),
        (None, Some(Value::String(s))) => Some(("const", vec![s.clone()])),
        (None, Some(_)) => {
            return Err(refuse("const", pointer, "only a string value is supported"))
        }
        (None, None) => None,
    };
    match (&choices, obj.contains_key("type")) {
        (Some(("enum", _)), true) => nullable &= enum_has_null,
        (Some(("enum", _)), false) => {
            nullable = enum_has_null;
            nullable_by = "enum";
        }
        (Some(_), _) => nullable = false,
        (None, _) => {}
    }

    let kind_name = match (type_name, &choices) {
        (Some(t), _) => t,
        (None, Some(_)) => "string",
        (None, None) if obj.contains_key("properties") => "object",
        (None, None) if obj.contains_key("items") => "array",
        (None, None) => {
            return Err(refuse(
                "type",
                pointer,
                "missing; every schema needs a `type`",
            ))
        }
    };
    if let Some((keyword, _)) = &choices {
        if kind_name != "string" {
            return Err(refuse(keyword, pointer, "is supported only for strings"));
        }
    }

    // Keywords that belong to another type are refused, not ignored.
    let allowed: &[&str] = match kind_name {
        "object" => &["properties", "required", "additionalProperties"],
        "array" => &["items", "minItems", "maxItems"],
        _ => &[],
    };
    for key in [
        "properties",
        "required",
        "additionalProperties",
        "items",
        "minItems",
        "maxItems",
    ] {
        if obj.contains_key(key) && !allowed.contains(&key) {
            return Err(refuse(
                key,
                pointer,
                format!("does not apply to type `{kind_name}`"),
            ));
        }
    }

    let kind = match kind_name {
        "object" => parse_object(obj, pointer)?,
        "array" => parse_array(obj, pointer)?,
        "string" => match choices {
            Some((_, values)) => Kind::Enum(values),
            None => Kind::String,
        },
        "integer" => Kind::Integer,
        "number" => Kind::Number,
        "boolean" => Kind::Boolean,
        other => return Err(refuse("type", pointer, format!("unknown type `{other}`"))),
    };
    Ok(Parsed {
        node: Node { kind, description },
        nullable,
        nullable_by,
    })
}

/// `anyOf` is accepted only as a nullable union: one schema and
/// `{"type": "null"}`, in either order, with only annotations beside it.
fn parse_nullable_any_of(
    obj: &Map<String, Value>,
    any_of: &Value,
    pointer: &str,
    description: Option<String>,
) -> Result<Parsed> {
    const ONLY_NULLABLE: &str =
        "supported only as a nullable union: one schema and {\"type\": \"null\"}";
    if let Some(key) = obj
        .keys()
        .find(|k| *k != "anyOf" && !ANNOTATIONS.contains(&k.as_str()))
    {
        return Err(refuse(
            key,
            pointer,
            "is not supported beside `anyOf`; put it in the non-null branch",
        ));
    }
    let Value::Array(branches) = any_of else {
        return Err(refuse("anyOf", pointer, "must be an array"));
    };
    let is_null = |v: &Value| {
        v.as_object().is_some_and(|o| {
            o.get("type") == Some(&Value::String("null".into()))
                && o.keys()
                    .all(|k| k == "type" || ANNOTATIONS.contains(&k.as_str()))
        })
    };
    let other_at = match branches.as_slice() {
        [a, b] if is_null(b) && !is_null(a) => 0,
        [a, b] if is_null(a) && !is_null(b) => 1,
        _ => return Err(refuse("anyOf", pointer, ONLY_NULLABLE)),
    };
    let any_of_pointer = child(pointer, "anyOf");
    let inner = parse_node(
        &branches[other_at],
        &child(&any_of_pointer, &other_at.to_string()),
    )?;
    let mut node = inner.node;
    if description.is_some() {
        node.description = description;
    }
    Ok(Parsed {
        node,
        nullable: true,
        nullable_by: "anyOf",
    })
}

fn parse_object(obj: &Map<String, Value>, pointer: &str) -> Result<Kind> {
    match obj.get("additionalProperties") {
        None | Some(Value::Bool(false)) => {}
        Some(_) => {
            return Err(refuse(
                "additionalProperties",
                pointer,
                "only `false` (or leaving it out) is supported: objects are closed",
            ))
        }
    }
    let props = match obj.get("properties") {
        None => Map::new(),
        Some(Value::Object(props)) => props.clone(),
        Some(_) => return Err(refuse("properties", pointer, "must be an object")),
    };
    let required: Vec<&str> = match obj.get("required") {
        None => Vec::new(),
        Some(Value::Array(names)) => {
            let mut out = Vec::new();
            for name in names {
                let Some(name) = name.as_str() else {
                    return Err(refuse(
                        "required",
                        pointer,
                        "must list property names as strings",
                    ));
                };
                if !props.contains_key(name) {
                    return Err(refuse(
                        "required",
                        pointer,
                        format!("names `{name}`, which `properties` doesn't define"),
                    ));
                }
                out.push(name);
            }
            out
        }
        Some(_) => return Err(refuse("required", pointer, "must be an array")),
    };
    let properties_pointer = child(pointer, "properties");
    let mut properties = Vec::with_capacity(props.len());
    for (name, value) in &props {
        let parsed = parse_node(value, &child(&properties_pointer, name))?;
        properties.push(Property {
            name: name.clone(),
            schema: parsed.node,
            required: required.contains(&name.as_str()),
            nullable: parsed.nullable,
        });
    }
    Ok(Kind::Object { properties })
}

fn parse_array(obj: &Map<String, Value>, pointer: &str) -> Result<Kind> {
    let bound = |key: &str| -> Result<Option<u64>> {
        match obj.get(key) {
            None => Ok(None),
            Some(v) => v
                .as_u64()
                .map(Some)
                .ok_or_else(|| refuse(key, pointer, "must be a non-negative integer")),
        }
    };
    let (min_items, max_items) = (bound("minItems")?, bound("maxItems")?);
    if let (Some(min), Some(max)) = (min_items, max_items) {
        if min > max {
            return Err(refuse(
                "minItems",
                pointer,
                format!("{min} is greater than maxItems {max}"),
            ));
        }
    }
    let items = match obj.get("items") {
        None => {
            return Err(refuse(
                "items",
                pointer,
                "missing; an array needs an `items` schema",
            ))
        }
        Some(Value::Object(_)) => {
            let items_pointer = child(pointer, "items");
            let parsed = parse_node(&obj["items"], &items_pointer)?;
            if parsed.nullable {
                return Err(refuse(
                    parsed.nullable_by,
                    &items_pointer,
                    NULLABLE_ONLY_AS_PROPERTY,
                ));
            }
            parsed.node
        }
        Some(_) => return Err(refuse("items", pointer, "must be a single schema object")),
    };
    Ok(Kind::Array {
        items: Box::new(items),
        min_items,
        max_items,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn refused(schema: Value) -> String {
        match JsonSchema::parse(&schema) {
            Err(Error::UnsupportedSchema(message)) => message,
            other => panic!("expected a refusal for {schema}, got {other:?}"),
        }
    }

    fn object(property: Value) -> Value {
        json!({"type": "object", "properties": {"x": property}, "required": ["x"]})
    }

    #[test]
    fn every_refused_keyword_is_named_with_its_pointer() {
        for keyword in [
            ("minimum", json!(0)),
            ("maximum", json!(10)),
            ("exclusiveMinimum", json!(0)),
            ("multipleOf", json!(2)),
            ("pattern", json!("^a")),
            ("format", json!("date")),
            ("minLength", json!(1)),
            ("maxLength", json!(9)),
            ("patternProperties", json!({})),
            ("oneOf", json!([{"type": "string"}])),
            ("allOf", json!([{"type": "string"}])),
            ("not", json!({})),
            ("$ref", json!("#/$defs/a")),
            ("$defs", json!({})),
            ("uniqueItems", json!(true)),
            ("prefixItems", json!([])),
            ("x-custom", json!(1)),
        ] {
            let (name, value) = keyword;
            let mut schema = json!({"type": "integer"});
            schema[name] = value;
            let message = refused(object(schema));
            assert!(
                message.contains(&format!("keyword `{name}` at `/properties/x`")),
                "{name}: {message}"
            );
        }
    }

    #[test]
    fn unions_and_other_any_of_forms_are_refused() {
        let message = refused(object(json!({"type": ["string", "integer"]})));
        assert!(message.contains("keyword `type`"), "{message}");
        let message = refused(object(
            json!({"anyOf": [{"type": "string"}, {"type": "integer"}]}),
        ));
        assert!(
            message.contains("keyword `anyOf` at `/properties/x`"),
            "{message}"
        );
        let message = refused(object(
            json!({"anyOf": [{"type": "string"}, {"type": "null"}, {"type": "integer"}]}),
        ));
        assert!(message.contains("keyword `anyOf`"), "{message}");
        let message = refused(json!({"anyOf": [{"type": "object"}, {"type": "object"}]}));
        assert!(
            message.contains("keyword `anyOf` at the schema root"),
            "{message}"
        );
        // A keyword beside anyOf belongs in its branch.
        let message = refused(object(
            json!({"type": "string", "anyOf": [{"type": "string"}, {"type": "null"}]}),
        ));
        assert!(message.contains("keyword `type`"), "{message}");
    }

    #[test]
    fn enums_of_non_strings_and_open_objects_are_refused() {
        assert!(refused(object(json!({"enum": [1, 2]}))).contains("keyword `enum`"));
        assert!(refused(object(json!({"enum": []}))).contains("keyword `enum`"));
        assert!(
            refused(object(json!({"type": "integer", "enum": ["a"]}))).contains("keyword `enum`")
        );
        assert!(refused(object(json!({"const": 3}))).contains("keyword `const`"));
        assert!(
            refused(json!({"type": "object", "additionalProperties": true}))
                .contains("keyword `additionalProperties` at the schema root")
        );
        assert!(
            refused(json!({"type": "object", "additionalProperties": {"type": "string"}}))
                .contains("keyword `additionalProperties`")
        );
    }

    #[test]
    fn structural_mistakes_are_refused() {
        assert!(refused(object(json!({"type": "array"}))).contains("keyword `items`"));
        assert!(refused(object(
            json!({"type": "array", "items": [{"type": "string"}]})
        ))
        .contains("keyword `items`"));
        assert!(refused(object(
            json!({"type": "array", "items": {"type": "string"}, "minItems": 3, "maxItems": 1})
        ))
        .contains("keyword `minItems`"));
        assert!(refused(object(
            json!({"type": "array", "items": {"type": "string"}, "minItems": -1})
        ))
        .contains("keyword `minItems`"));
        assert!(refused(object(
            json!({"type": "string", "items": {"type": "string"}})
        ))
        .contains("keyword `items` at `/properties/x`: does not apply to type `string`"));
        assert!(refused(object(json!({"type": "date"}))).contains("unknown type `date`"));
        assert!(refused(object(json!({"description": "no type"}))).contains("keyword `type`"));
        assert!(refused(object(json!({"type": "null"}))).contains("keyword `type`"));
        assert!(
            refused(json!({"type": "object", "properties": {}, "required": ["ghost"]}))
                .contains("keyword `required`")
        );
        assert!(refused(object(json!(true))).contains("boolean schema"));
        // Pointer segments are escaped (RFC 6901).
        let message = refused(
            json!({"type": "object", "properties": {"a/b~c": {"type": "string", "format": "uri"}}}),
        );
        assert!(message.contains("`/properties/a~1b~0c`"), "{message}");
    }

    #[test]
    fn the_subset_parses() {
        let schema = JsonSchema::parse(&json!({
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "title": "Extraction",
            "type": "object",
            "description": "A person",
            "properties": {
                "name": {"type": "string", "description": "Full name", "examples": ["Ada"]},
                "age": {"type": "integer", "default": 0},
                "score": {"type": "number"},
                "active": {"type": "boolean"},
                "tier": {"type": "string", "enum": ["free", "pro"]},
                "kind": {"const": "person"},
                "tags": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 5},
                "address": {"type": "object", "properties": {"city": {"type": "string"}},
                            "additionalProperties": false}
            },
            "required": ["name", "tags"],
            "additionalProperties": false
        }))
        .unwrap();
        assert_eq!(schema.root().description.as_deref(), Some("A person"));
        let Kind::Object { properties } = &schema.root().kind else {
            panic!()
        };
        let names: Vec<&str> = properties.iter().map(|p| p.name.as_str()).collect();
        assert_eq!(
            names,
            ["name", "age", "score", "active", "tier", "kind", "tags", "address"]
        );
        assert!(
            properties[0].required && !properties[1].required,
            "required absent => optional"
        );
        assert_eq!(
            properties[0].schema.description.as_deref(),
            Some("Full name")
        );
        assert_eq!(
            properties[4].schema.kind,
            Kind::Enum(vec!["free".into(), "pro".into()])
        );
        assert_eq!(properties[5].schema.kind, Kind::Enum(vec!["person".into()]));
        assert!(matches!(
            &properties[6].schema.kind,
            Kind::Array { min_items: Some(1), max_items: Some(5), items } if items.kind == Kind::String
        ));
    }

    #[test]
    fn properties_keep_the_order_sent() {
        // Deliberately not alphabetical, nor reverse-alphabetical.
        let schema = JsonSchema::parse(&json!({
            "type": "object",
            "properties": {"zeta": {"type": "string"}, "alpha": {"type": "string"},
                           "mid": {"type": "string"}, "beta": {"type": "string"}}
        }))
        .unwrap();
        let Kind::Object { properties } = &schema.root().kind else {
            panic!()
        };
        let names: Vec<&str> = properties.iter().map(|p| p.name.as_str()).collect();
        assert_eq!(names, ["zeta", "alpha", "mid", "beta"]);
    }

    #[test]
    fn nullable_properties_in_both_spellings() {
        let schema = JsonSchema::parse(&json!({
            "type": "object",
            "properties": {
                "a": {"type": ["string", "null"]},
                "b": {"anyOf": [{"type": "integer"}, {"type": "null"}], "description": "outer", "default": null},
                "c": {"anyOf": [{"type": "null"}, {"type": "string", "enum": ["x", "y"]}]},
                "d": {"type": ["string", "null"], "enum": ["x", null]},
                "e": {"type": "string"},
                "f": {"enum": ["x", null]},
                "g": {"type": ["string", "null"], "enum": ["x"]},
                "h": {"type": "string", "enum": ["x", null]}
            },
            "required": ["a", "b", "c", "d", "e", "f", "g", "h"]
        }))
        .unwrap();
        let Kind::Object { properties } = &schema.root().kind else {
            panic!()
        };
        let nullable: Vec<bool> = properties.iter().map(|p| p.nullable).collect();
        // A value must satisfy both `type` and `enum`: g and h can't be null.
        assert_eq!(
            nullable,
            [true, true, true, true, false, true, false, false]
        );
        assert_eq!(properties[1].schema.kind, Kind::Integer);
        assert_eq!(properties[1].schema.description.as_deref(), Some("outer"));
        assert_eq!(
            properties[2].schema.kind,
            Kind::Enum(vec!["x".into(), "y".into()])
        );
        assert_eq!(properties[3].schema.kind, Kind::Enum(vec!["x".into()]));
    }

    #[test]
    fn nullable_outside_a_property_is_refused() {
        let message = refused(json!({"type": ["object", "null"]}));
        assert!(
            message.contains("keyword `type` at the schema root"),
            "{message}"
        );
        let message = refused(object(
            json!({"type": "array", "items": {"type": ["string", "null"]}}),
        ));
        assert!(
            message.contains("keyword `type` at `/properties/x/items`"),
            "{message}"
        );
        let message = refused(object(json!({"type": "array",
            "items": {"anyOf": [{"type": "string"}, {"type": "null"}]}})));
        assert!(
            message.contains("keyword `anyOf` at `/properties/x/items`"),
            "{message}"
        );
    }

    #[test]
    fn conform_orders_keys_and_fills_required_nulls() {
        let schema = JsonSchema::parse(&json!({
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "middle": {"type": ["string", "null"]},
                "nick": {"type": ["string", "null"]},
                "people": {"type": "array", "items": {"type": "object",
                    "properties": {"name": {"type": "string"}, "age": {"type": "integer"}}}},
                "note": {"type": "string"}
            },
            "required": ["title", "middle", "people"]
        }))
        .unwrap();
        let reply = r#"{"people":[{"age":3,"name":"a"}],"note":"n","title":"t"}"#;
        assert_eq!(
            schema.conform(reply),
            r#"{"title":"t","middle":null,"people":[{"name":"a","age":3}],"note":"n"}"#,
            "schema order; null for the required nullable `middle`; optional `nick` stays absent"
        );
        // An explicit null is kept; unknown keys keep their place after.
        assert_eq!(
            schema.conform(r#"{"extra":1,"middle":null,"title":"t","people":[]}"#),
            r#"{"title":"t","middle":null,"people":[],"extra":1}"#
        );
        // Not JSON (cut short): untouched.
        assert_eq!(schema.conform(r#"{"title":"t"#), r#"{"title":"t"#);
    }

    #[test]
    fn validate_checks_types_bounds_and_closed_objects() {
        let schema = JsonSchema::parse(&json!({
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "n": {"type": ["integer", "null"]},
                "tags": {"type": "array", "items": {"enum": ["a", "b"]}, "maxItems": 2},
                "ok": {"type": "boolean"}
            },
            "required": ["name", "n"]
        }))
        .unwrap();
        let ok = |v: Value| schema.validate(&v);
        assert_eq!(ok(json!({"name": "x", "n": null})), Ok(()));
        assert_eq!(ok(json!({"name": "x", "n": 3, "tags": ["a", "b"], "ok": true})), Ok(()));
        assert!(ok(json!({"name": "x"})).unwrap_err().contains("`n` is missing"));
        assert!(ok(json!({"name": "x", "n": 1.5})).unwrap_err().contains("/n"));
        assert!(ok(json!({"name": "x", "n": 1, "tags": ["c"]})).unwrap_err().contains("/tags/0"));
        assert!(ok(json!({"name": "x", "n": 1, "tags": ["a", "a", "a"]})).unwrap_err().contains("maxItems"));
        assert!(ok(json!({"name": "x", "n": 1, "extra": 1})).unwrap_err().contains("`extra`"));
        assert!(ok(json!({"name": null, "n": 1})).is_err(), "only nullable properties may be null");
        assert!(ok(json!([])).unwrap_err().starts_with("the root"));
    }
}
