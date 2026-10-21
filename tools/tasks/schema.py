"""Validate records against fixtures/tasks/verification/schema.json.

A small validator for the subset of JSON Schema that file uses (`type`,
`required`, `properties`, `additionalProperties`, `items`, `enum`, `const`,
`pattern`, `minItems`, `maxItems`, `minLength`, `minimum` and `$ref` into
`$defs`), so the tools need nothing outside the standard library.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

SCHEMA_PATH = Path(__file__).resolve().parents[2] / "fixtures" / "tasks" / "verification" / "schema.json"

_TYPES = {
    "object": dict, "array": list, "string": str, "boolean": bool, "null": type(None),
    "integer": int, "number": (int, float),
}


class SchemaError(ValueError):
    pass


@lru_cache(maxsize=None)
def _root(path: str = str(SCHEMA_PATH)) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _type_ok(value, t) -> bool:
    if isinstance(t, list):
        return any(_type_ok(value, x) for x in t)
    if t in ("integer", "number") and isinstance(value, bool):
        return False
    return isinstance(value, _TYPES[t])


def _check(value, sch: dict, root: dict, where: str, errors: list[str]) -> None:
    if "$ref" in sch:
        name = sch["$ref"].split("/")[-1]
        sch = {**root["$defs"][name], **{k: v for k, v in sch.items() if k != "$ref"}}
    if "type" in sch and not _type_ok(value, sch["type"]):
        errors.append(f"{where}: expected {sch['type']}, got {type(value).__name__}")
        return
    if "const" in sch and value != sch["const"]:
        errors.append(f"{where}: must be {sch['const']!r}")
    if "enum" in sch and value not in sch["enum"]:
        errors.append(f"{where}: {value!r} is not one of {sch['enum']}")
    if isinstance(value, str):
        if "pattern" in sch and not re.search(sch["pattern"], value):
            errors.append(f"{where}: {value!r} does not match {sch['pattern']!r}")
        if len(value) < sch.get("minLength", 0):
            errors.append(f"{where}: shorter than {sch['minLength']}")
    if isinstance(value, (int, float)) and not isinstance(value, bool) and "minimum" in sch \
            and value < sch["minimum"]:
        errors.append(f"{where}: below {sch['minimum']}")
    if isinstance(value, list):
        if len(value) < sch.get("minItems", 0):
            errors.append(f"{where}: fewer than {sch['minItems']} items")
        if "maxItems" in sch and len(value) > sch["maxItems"]:
            errors.append(f"{where}: more than {sch['maxItems']} items")
        if "items" in sch:
            for i, item in enumerate(value):
                _check(item, sch["items"], root, f"{where}[{i}]", errors)
    if isinstance(value, dict):
        for k in sch.get("required", []):
            if k not in value:
                errors.append(f"{where}: missing {k!r}")
        props = sch.get("properties", {})
        extra = sch.get("additionalProperties", True)
        for k, v in value.items():
            if k in props:
                _check(v, props[k], root, f"{where}.{k}", errors)
            elif extra is False:
                errors.append(f"{where}: unexpected key {k!r}")
            elif isinstance(extra, dict):
                _check(v, extra, root, f"{where}.{k}", errors)


def errors(value, name: str, schema_path: Path | None = None) -> list[str]:
    root = _root(str(schema_path or SCHEMA_PATH))
    out: list[str] = []
    _check(value, {"$ref": f"#/$defs/{name}"}, root, name, out)
    return out


def validate(value, name: str, schema_path: Path | None = None) -> None:
    errs = errors(value, name, schema_path)
    if errs:
        ident = value.get("id", "") if isinstance(value, dict) else ""
        raise SchemaError(f"{name} {ident}: " + "; ".join(errs[:8]))
