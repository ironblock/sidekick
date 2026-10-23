"""Read and validate docs/data/licences.toml, the licence audit that decides
each model's publishing tier (#137; docs/PUBLISHING.md).

Usage:
    python3 tools/licences.py            # validate, and print the tier table
    python3 tools/licences.py --check    # validate only (CI, through models_table.py)

A tier is reviewable because it follows from the recorded data: `problems()`
recomputes the lowest tier each entry's training data, disclosure and
concerns allow, and refuses an entry whose stated tier differs. Tier 3 is
the one judgement it can't recompute (a licence that forbids
redistribution or commercial use, or makes commercial use depend on the
user's circumstances), so a tier-3 entry must say why in `reason`.

Standard library only.
"""

import argparse
import re
import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
LICENCES = REPO / "docs" / "data" / "licences.toml"
MODELS = REPO / "docs" / "data" / "models.toml"

SCHEMA = 2
TIERS = ("1", "2a", "2b", "3")
TERMS = ("permissive", "share-alike", "non-commercial", "unknown")
DISCLOSURE = ("full", "partial", "undisclosed")
CONCERNS = ("non-commercial-data", "copyleft-data", "unlicensed-data", "undisclosed-data",
            "unnamed-synthetic-generator", "custom-terms", "use-policy")
TRAVELLING = ("custom-terms", "use-policy")

REQUIRED = {
    "source": str, "licence": str, "licence_sources": list, "base_model": str, "base_model_licence": str,
    "base_model_licence_sources": list, "training_disclosure": str, "training_data": list,
    "base_pretraining": str, "custom_terms": list, "concerns": list, "tier": str, "reason": str,
    "unverified": list, "read": str,
}
OPTIONAL = {"ship_files": list, "notice": str, "ship_copies": list}
DATASET = ("name", "licence", "terms", "source", "stage")

# A link to a host that versions its files must name the revision it was read at.
PINNED = [
    (re.compile(r"^https://huggingface\.co/"), re.compile(r"/(blob|tree|resolve)/[0-9a-f]{40}(/|$)")),
    (re.compile(r"^https://github\.com/"), re.compile(r"/(blob|tree)/[0-9a-f]{40}(/|$)")),
]
HF_REPO = re.compile(r"^https://huggingface\.co/(?!datasets/)([^/]+/[^/]+)/")
HF_DATASET = re.compile(r"^https://huggingface\.co/datasets/([^/]+/[^/]+)/")

# Hugging Face's licence identifiers (model-card `license:`), by our names.
HF_LICENSE = {"Apache-2.0": "apache-2.0", "MIT": "mit", "Gemma Terms of Use": "gemma"}


def load(path=LICENCES):
    return tomllib.loads(Path(path).read_text())


def load_models():
    return tomllib.loads(MODELS.read_text())["models"]


def floor_tier(entry):
    """The lowest tier the recorded data allows. Tier 3 is never implied:
    it is a judgement about the licence, which `reason` must give."""
    terms = {d.get("terms") for d in entry.get("training_data", [])}
    concerns = set(entry.get("concerns", []))
    if "non-commercial" in terms or concerns & set(TRAVELLING):
        return "2b"
    if (terms & {"share-alike", "unknown"} or entry.get("training_disclosure") != "full"
            or "unnamed-synthetic-generator" in concerns or not entry.get("training_data")):
        return "2a"
    return "1"


def pinned(url):
    for host, rev in PINNED:
        if host.match(url):
            return bool(rev.search(url))
    return True


def hf_datasets(entry):
    """Hugging Face dataset ids named by the entry's training-data sources."""
    out = []
    for d in entry.get("training_data", []):
        m = HF_DATASET.match(d.get("source", ""))
        if m and m.group(1) not in out:
            out.append(m.group(1))
    return out


def hf_license(entry):
    return HF_LICENSE.get(entry["licence"], "other")


def problems(data=None, models=None):
    data = load() if data is None else data
    models = load_models() if models is None else models
    out = []
    if data.get("schema") != SCHEMA:
        out.append(f"licences.toml: schema {data.get('schema')!r}, expected {SCHEMA}")
    entries = data.get("models", {})
    for mid in sorted(set(models) - set(entries)):
        out.append(f"licences.toml: no verdict for {mid} (models.toml)")
    for mid in sorted(set(entries) - set(models)):
        out.append(f"licences.toml: {mid} isn't in models.toml")
    for mid, e in sorted(entries.items()):
        w = f"licences.toml {mid}"
        bad = False
        for k, t in REQUIRED.items():
            if not isinstance(e.get(k), t):
                out.append(f"{w}: {k} is missing or has the wrong type")
                bad = True
        for k, t in OPTIONAL.items():
            if k in e and not isinstance(e[k], t):
                out.append(f"{w}: {k} has the wrong type")
                bad = True
        if extra := set(e) - set(REQUIRED) - set(OPTIONAL):
            out.append(f"{w}: unknown fields {sorted(extra)}")
        if bad:
            continue
        if mid in models and e["source"] != models[mid]["source"]:
            out.append(f"{w}: source {e['source']} isn't models.toml's {models[mid]['source']}")
        if e["tier"] not in TIERS:
            out.append(f"{w}: tier {e['tier']!r} isn't one of {', '.join(TIERS)}")
        if e["training_disclosure"] not in DISCLOSURE:
            out.append(f"{w}: training_disclosure {e['training_disclosure']!r}")
        if unknown := set(e["concerns"]) - set(CONCERNS):
            out.append(f"{w}: unknown concerns {sorted(unknown)}")
        if not e["reason"].strip():
            out.append(f"{w}: no reason")
        if not re.match(r"^\d{4}-\d{2}-\d{2}$", e["read"]):
            out.append(f"{w}: read must be a date")
        urls = list(e["licence_sources"]) + list(e["base_model_licence_sources"])
        for i, d in enumerate(e["training_data"]):
            if not isinstance(d, dict) or any(not isinstance(d.get(k), str) or not d[k] for k in DATASET):
                out.append(f"{w}: training_data[{i}] needs {', '.join(DATASET)}")
                continue
            if d["terms"] not in TERMS:
                out.append(f"{w}: {d['name']}: terms {d['terms']!r} isn't one of {', '.join(TERMS)}")
            urls.append(d["source"])
        for t in e["custom_terms"]:
            if not isinstance(t, dict) or not all(isinstance(t.get(k), str) and t[k] for k in ("name", "url", "requires")):
                out.append(f"{w}: a custom_terms entry needs name, url and requires")
            else:
                urls.append(t["url"])
        for c in e.get("ship_copies", []):
            if not isinstance(c, dict) or not all(isinstance(c.get(k), str) and c[k] for k in ("file", "of", "url")):
                out.append(f"{w}: a ship_copies entry needs file, of and url")
            else:
                urls.append(c["url"])
        if e["tier"] == "3" and (e.get("ship_files") or e.get("ship_copies") or e.get("notice")):
            out.append(f"{w}: tier 3 isn't published, so it ships nothing")
        for u in urls:
            if not u.startswith("https://"):
                out.append(f"{w}: {u!r} isn't an https link")
            elif not pinned(u):
                out.append(f"{w}: {u} isn't pinned to a revision")
        terms = {d.get("terms") for d in e["training_data"] if isinstance(d, dict)}
        concerns = set(e["concerns"])
        if ("non-commercial" in terms) != ("non-commercial-data" in concerns):
            out.append(f"{w}: concerns and training data disagree on non-commercial data")
        if ("share-alike" in terms) != ("copyleft-data" in concerns):
            out.append(f"{w}: concerns and training data disagree on share-alike data")
        if e["custom_terms"] and not concerns & set(TRAVELLING):
            out.append(f"{w}: custom_terms but no custom-terms or use-policy concern")
        if e["tier"] in TIERS:
            floor = floor_tier(e)
            if e["tier"] == "3":
                if "custom-terms" not in concerns:
                    out.append(f"{w}: tier 3 needs the licence's terms recorded (custom-terms)")
            elif e["tier"] != floor:
                out.append(f"{w}: tier {e['tier']}, but its data puts it at {floor}")
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--check", action="store_true", help="validate only")
    args = p.parse_args()
    found = problems()
    if found:
        raise SystemExit("docs/data/licences.toml:\n  " + "\n  ".join(found))
    data = load()["models"]
    if args.check:
        print(f"licences: {len(data)} verdicts, valid")
        return
    for mid, e in data.items():
        print(f"{mid:34} {e['tier']:3} {e['licence']}")


if __name__ == "__main__":
    main()
