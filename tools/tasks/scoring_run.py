"""Runners for the verification checkers. Each writes run records that
tools/tasks/scoring_report.py scores. Maintainer-run, one heavy job at a
time; never in CI.

    # sidekick's nli checkers (A3 and A3g, or A2 and A2g), through /v1/classify
    python3 tools/tasks/scoring_run.py nli --model nli-deberta-v3-base \\
        --model-dir <installed model dir> --out <runs dir> [--base-url http://127.0.0.1:8790]

    # Foundation Models (F), through /v1/chat/completions
    python3 tools/tasks/scoring_run.py fm --out <runs dir>

    # External baselines, in their own code at their defaults (separate
    # environments: MiniCheck's package and nltk; transformers and torch)
    python3 tools/tasks/scoring_run.py minicheck --out <runs dir>
    python3 tools/tasks/scoring_run.py hhem --out <runs dir>

`--items controls` (or `--limit N`) runs a subset, as a check of the request
shapes. Every run records the corpus lock, the control file, the scoring
settings, the request shape's hash, the model's sha256 (its installed files,
or the pinned revision), the sidekick commit and version, and the macOS
build. The base URL is a command-line argument and is never recorded; only
the route is.

The nli runner tokenizes with the installed tokenizer.json (checked against
its pinned sha256), cuts windows as the ceilings do, predicts each distinct
(window, claim) once, and asserts that nothing was truncated: each request's
`usage.prompt_tokens` must equal the sum of its pairs' lengths, and every
pair must fit `max_seq_len`. Three-way models must return `nli_probs`
summing to 1 within 1e-5.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
import tomllib
import urllib.error
import urllib.request
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tasks import canon, scoring_controls, scoring_judge, scoring_rules as rules  # noqa: E402
from tasks import scoring_set, scoring_windows as sw  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
RUN_SCHEMA = "sidekick-verification-run/1"


# ---------------------------------------------------------------- provenance


def _cmd(args: list[str]) -> str | None:
    try:
        return subprocess.run(args, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def tree_sha256(d: Path) -> str:
    """sha256 over a directory's files: one `<sha256>  <relative path>` line
    per file, sorted by path."""
    lines = []
    for p in sorted(d.rglob("*")):
        if p.is_file():
            lines.append(f"{canon.sha256_file(p)}  {p.relative_to(d).as_posix()}\n")
    return canon.sha256_text("".join(lines))


def shape_hash(route: str, body: dict, extra: dict) -> str:
    """The request's shape: every field but the texts, plus the route, τ,
    the prompt sha and the output mode."""
    shape = {k: ("<text>" if k in ("input", "messages", "candidate_labels") else v) for k, v in body.items()}
    return canon.sha256_text(canon.canonical_line({"route": route, "body": shape, **extra}))


def provenance(lock: dict, cfg: dict, extra: dict) -> dict:
    head = _cmd(["git", "-C", str(REPO), "rev-parse", "HEAD"])
    dirty = _cmd(["git", "-C", str(REPO), "status", "--porcelain", "--untracked-files=no"])
    return {"set": cfg["set"], "set_sha256": lock["set_sha256"],
            "controls_sha256": canon.sha256_file(scoring_controls.CONTROLS),
            "scoring_sha256": canon.sha256_file(scoring_set.SCORING),
            "harness_commit": head, "harness_dirty": bool(dirty),
            "macos_build": _cmd(["sw_vers", "-buildVersion"]), "macos_version": _cmd(["sw_vers", "-productVersion"]),
            "machine": platform.machine(), "tau": cfg["tau"], "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            **extra}


# ---------------------------------------------------------------- HTTP


def http(base: str, route: str, body: dict | None = None, key: str | None = None, timeout: float = 600):
    """(status, headers, json) for GET (body None) or POST."""
    data = None if body is None else json.dumps(body).encode("utf-8")
    req = urllib.request.Request(base.rstrip("/") + route, data=data, method="GET" if body is None else "POST")
    req.add_header("Content-Type", "application/json")
    if key:
        req.add_header("Authorization", f"Bearer {key}")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, {k.lower(): v for k, v in r.headers.items()}, json.loads(r.read() or b"null")
    except urllib.error.HTTPError as e:
        raw = e.read()
        try:
            payload = json.loads(raw)
        except ValueError:
            payload = {"raw": raw.decode("utf-8", "replace")}
        return e.code, {k.lower(): v for k, v in e.headers.items()}, payload


# ---------------------------------------------------------------- items


def select(items: list[dict], which: str, limit: int | None) -> list[dict]:
    if which == "controls":
        items = [i for i in items if "control" in i]
    elif which == "corpus":
        items = [i for i in items if "control" not in i]
    elif which == "fallback":
        items = [i for i in items if scoring_set.in_fallback(i)]
    return items[:limit] if limit else items


def load_items(which: str, limit: int | None):
    cfg = scoring_set.config()
    triples, _claims, lock = scoring_set.load(cfg["set"])
    items = scoring_set.items(triples, scoring_controls.build())
    return cfg, lock, select(items, which, limit)


def write_run(out: Path, name: str, record: dict) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    p = out / f"{name.replace(':', '__')}.json"
    canon.write_json(p, record)
    return p


# ---------------------------------------------------------------- sidekick nli


class Interface:
    """Assertions on every request (design §2.1)."""

    def __init__(self):
        self.requests = 0
        self.pairs = 0
        self.probs_checked = 0
        self.failures: list[str] = []

    def fail(self, msg: str):
        self.failures.append(msg)
        raise AssertionError(msg)


def run_nli(a) -> int:
    cfg, lock, items = load_items(a.items, a.limit)
    model = next((m for m in cfg["nli"] if m["id"] == a.model), None)
    if model is None:
        raise SystemExit(f"{a.model} is not an nli checker in scoring.toml")
    mdir = Path(a.model_dir)
    manifest = tomllib.loads((mdir / "classifier.toml").read_text())
    tok = sw.HFTokenizer(mdir / "tokenizer.json", model["sha256_tokenizer"])
    seq = model["max_seq_len"]
    if manifest.get("max_seq_len") != seq:
        raise SystemExit(f"installed max_seq_len {manifest.get('max_seq_len')} != scoring.toml's {seq}")
    max_batch = int(manifest.get("max_batch", 8))
    key = os.environ.get("SIDEKICK_API_KEY")
    st, hdr, models = http(a.base_url, "/v1/models", key=key)
    listed = {m["id"]: m for m in (models or {}).get("data", [])}
    if a.model not in listed:
        raise SystemExit(f"{a.model} is not served by the daemon")
    classes = listed[a.model].get("nli_classes")
    three = model["ways"] == 3
    if three and not (classes and {"entailment", "neutral", "contradiction"} <= set(classes)):
        raise SystemExit(f"{a.model}: three-way scoring needs nli_classes with neutral; the daemon lists {classes}")

    # Windows, gates and the distinct predictions.
    plan, todo = {}, {}
    for it in items:
        ws = sw.windows(it["evidence"], tok, it["claim"], seq, cfg["overlap"], cfg["margin"])
        lens = sw.fits(ws, tok, it["claim"], seq)
        mask = sw.gated(ws, sw.anchors(it["claim"]))
        keys = [sw.key(w.text, it["claim"]) for w in ws]
        plan[it["item"]] = (keys, mask)
        for k, w, n in zip(keys, ws, lens):
            todo.setdefault(k, (w.text, it["claim"], n))
    Path(a.out).mkdir(parents=True, exist_ok=True)
    cache_path = Path(a.out) / f"cache.{a.model}.jsonl"
    cache = {}
    if cache_path.exists():
        for line in cache_path.read_text().splitlines():
            r = json.loads(line)
            cache[r["key"]] = r
    body_shape = None
    iface = Interface()
    by_claim: dict[str, list[str]] = {}
    for k, (_text, claim, _n) in todo.items():
        if k not in cache:
            by_claim.setdefault(claim, []).append(k)
    buckets_seen, version = set(), None
    with open(cache_path, "a", encoding="utf-8") as cf:
        for claim, ks in by_claim.items():
            for i in range(0, len(ks), max_batch):
                batch = ks[i:i + max_batch]
                body = {"model": a.model, "input": [todo[k][0] for k in batch], "candidate_labels": [claim],
                        "hypothesis_template": "{}"}
                if three:
                    body["nli_probs"] = True
                body_shape = body_shape or body
                st, hdr, resp = http(a.base_url, "/v1/classify", body, key)
                iface.requests += 1
                if st != 200:
                    iface.fail(f"/v1/classify {st}: {json.dumps(resp)[:300]}")
                want = sum(todo[k][2] for k in batch)
                got = resp["usage"]["prompt_tokens"]
                if got != want:
                    iface.fail(f"usage.prompt_tokens {got} != the harness's {want}: truncation or a tokenizer mismatch")
                buckets_seen.update(x for x in hdr.get("sidekick-buckets", "").split(",") if x)
                version = version or hdr.get("sidekick-version")
                for k, res in zip(batch, sorted(resp["data"], key=lambda d: d["index"])):
                    rec = {"key": k, "probs": res["probs"]}
                    if three:
                        p = res["nli_probs"][0]
                        if abs(sum(p.values()) - 1) > cfg["probs_tolerance"]:
                            iface.fail(f"nli_probs sum {sum(p.values())}")
                        iface.probs_checked += 1
                        rec["nli_probs"] = p
                    iface.pairs += 1
                    cache[k] = rec
                    cf.write(json.dumps(rec, sort_keys=True) + "\n")
                cf.flush()
    if version is None and todo:  # everything was cached: ask once for the daemon's version
        text, claim, _n = next(iter(todo.values()))
        _st, hdr, _ = http(a.base_url, "/v1/classify", {"model": a.model, "input": [text],
                                                         "candidate_labels": [claim], "hypothesis_template": "{}"}, key)
        version = hdr.get("sidekick-version")
    body_shape = body_shape or {"model": a.model, "input": [], "candidate_labels": [], "hypothesis_template": "{}",
                                **({"nli_probs": True} if three else {})}
    prov = provenance(lock, cfg, {
        "route": "/v1/classify", "model": a.model, "model_revision": model["revision"],
        "model_sha256": tree_sha256(mdir), "tokenizer_sha256": tok.sha256, "max_seq_len": seq,
        "sidekick_version": version, "nli_classes": classes, "buckets_seen": sorted(buckets_seen, key=int),
        "request_shape_sha256": shape_hash("/v1/classify", body_shape, {"tau": cfg["tau"]}),
        "interface": {"requests": iface.requests, "pairs": iface.pairs, "nli_probs_checked": iface.probs_checked,
                      "truncation_asserted": True, "failures": iface.failures}})
    fam = "A3" if three else "A2"
    for gated in (False, True):
        out_items = []
        for it in items:
            keys, mask = plan[it["item"]]
            use, none = rules.gate(keys, mask) if gated else (keys, False)
            if three:
                agg = rules.three_way([cache[k]["nli_probs"] for k in use], cfg["tau"])
            else:
                agg = rules.two_way([cache[k]["probs"][0] for k in use], cfg["tau"])
            out_items.append({"item": it["item"], "verdict": agg["verdict"], "E": agg["E"], "C": agg.get("C"),
                              "windows": len(keys), "scored_windows": len(use), "no_anchor": bool(gated and none)})
        name = f"{fam}{'g' if gated else ''}:{a.model}"
        p = write_run(Path(a.out), name, {"schema": RUN_SCHEMA, "checker": name, "family": fam + ("g" if gated else ""),
                                          "mode": "three-way" if three else "binary", "provenance": prov,
                                          "items": out_items})
        print(f"wrote {p}")
    return 0


# ---------------------------------------------------------------- Foundation Models


def run_fm(a) -> int:
    cfg, lock, items = load_items(a.items, a.limit)
    fm = cfg["foundation_models"]
    prompt = scoring_judge.load()
    key = os.environ.get("SIDEKICK_API_KEY")
    st, hdr, health = http(a.base_url, "/health", key=key)
    chat = (health or {}).get("chat", {})
    out_items, shape, parse_fail, version, variant = [], None, 0, hdr.get("sidekick-version"), None
    for it in items:
        body = {"model": chat.get("model") or "foundation-models",
                "messages": scoring_judge.messages(prompt, "full", it["claim"], it["evidence"]),
                "temperature": fm["temperature"], "max_tokens": fm["max_tokens"],
                "response_format": {"type": "json_schema",
                                    "json_schema": {"name": "verdict", "schema": scoring_judge.SCHEMA}}}
        shape = shape or body
        st, h, resp = http(a.base_url, "/v1/chat/completions", body, key)
        version = version or h.get("sidekick-version")
        variant = variant or h.get("sidekick-model")
        err = (resp or {}).get("error", {}) if isinstance(resp, dict) else {}
        if st != 200 and "context" in json.dumps(err):
            out_items.append({"item": it["item"], "verdict": None, "not_run": True, "reason": "context"})
            continue
        if st != 200:
            raise SystemExit(f"/v1/chat/completions {st}: {json.dumps(resp)[:300]}")
        verdict, ok = scoring_judge.parse(scoring_judge.final_content({"response": resp}))
        parse_fail += not ok
        out_items.append({"item": it["item"], "verdict": verdict, "parse_ok": ok, "not_run": False,
                          "prompt_tokens": resp.get("usage", {}).get("prompt_tokens")})
    prov = provenance(lock, cfg, {
        "route": fm["route"], "model": chat.get("model"), "availability": chat.get("availability"),
        "variant": chat.get("variant"),
        "variant_id": chat.get("variant_id"), "context_limit": chat.get("context_limit"),
        "sidekick_version": version, "sidekick_model_header": variant,
        "prompt_sha256": prompt["sha256"], "output_mode": "json_schema",
        "request_shape_sha256": shape_hash(fm["route"], shape or {}, {"prompt_sha256": prompt["sha256"],
                                                                       "output_mode": "json_schema"}),
        "interface": {"requests": len(items), "parse_failures": parse_fail}})
    p = write_run(Path(a.out), "F:foundation-models", {"schema": RUN_SCHEMA, "checker": "F:foundation-models",
                                                       "family": "F", "mode": "three-way", "provenance": prov,
                                                       "items": out_items})
    print(f"wrote {p}")
    return 0


# ---------------------------------------------------------------- external baselines


def _hf_snapshot(repo: str) -> str | None:
    """The commit the local cache's `main` points at (repo ids compare
    without regard to case, as the Hub resolves them)."""
    home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
    want = ("models--" + repo.replace("/", "--")).lower()
    for d in home.glob("models--*") if home.exists() else []:
        ref = d / "refs" / "main"
        if d.name.lower() == want and ref.exists():
            return ref.read_text().strip()
    return None


def run_minicheck(a) -> int:
    """MiniCheck's own code at its defaults. Its loader reads revision
    `main`, so the run requires the local cache's `main` to be the pinned
    revision (fetch it once at that revision, then run offline)."""
    cfg, lock, items = load_items(a.items, a.limit)
    mc = cfg["minicheck"]
    snap = _hf_snapshot(mc["model"])
    if snap != mc["revision"]:
        raise SystemExit(f"{mc['model']}: cached main is {snap}, not the pinned {mc['revision']}")
    from minicheck.minicheck import MiniCheck  # noqa: E402  (MiniCheck's package, at code_revision)

    scorer = MiniCheck(model_name="deberta-v3-large")
    out_items = []
    for it in items:
        label, prob, chunks_, per_chunk = scorer.score(docs=[it["evidence"]], claims=[it["claim"]])
        out_items.append({"item": it["item"], "verdict": "S" if label[0] == 1 else "not-S", "E": float(prob[0]),
                          "windows": len(chunks_[0]), "chunk_sha256": [canon.sha256_text(c) for c in chunks_[0]]})
    prov = provenance(lock, cfg, {"model": mc["model"], "model_revision": mc["revision"], "code": mc["code"],
                                  "code_revision": mc["code_revision"], "chunk_size": mc["chunk_size"],
                                  "defaults": True})
    p = write_run(Path(a.out), f"M:{mc['id']}", {"schema": RUN_SCHEMA, "checker": f"M:{mc['id']}", "family": "M",
                                                 "mode": "binary", "provenance": prov, "items": out_items})
    print(f"wrote {p}")
    return 0


def run_hhem(a) -> int:
    """HHEM-2.1-Open at its defaults: the whole haystack as the premise.
    `--max-premise-tokens` reports longer items as not run instead of
    attempting them (its attention is quadratic in the premise)."""
    cfg, lock, items = load_items(a.items, a.limit)
    hh = cfg["hhem"]
    from transformers import AutoModelForSequenceClassification  # noqa: E402

    m = AutoModelForSequenceClassification.from_pretrained(hh["model"], revision=hh["revision"],
                                                           trust_remote_code=True)
    out_items = []
    for it in items:
        n = len(m.tokenzier(it["evidence"])["input_ids"])
        if a.max_premise_tokens and n > a.max_premise_tokens:
            out_items.append({"item": it["item"], "verdict": None, "not_run": True, "premise_tokens": n})
            continue
        score = float(m.predict([(it["evidence"], it["claim"])])[0])
        out_items.append({"item": it["item"], "verdict": "S" if score >= hh["threshold"] else "not-S", "E": score,
                          "not_run": False, "premise_tokens": n})
    prov = provenance(lock, cfg, {"model": hh["model"], "model_revision": hh["revision"], "defaults": True,
                                  "max_premise_tokens": a.max_premise_tokens})
    p = write_run(Path(a.out), f"H:{hh['id']}", {"schema": RUN_SCHEMA, "checker": f"H:{hh['id']}", "family": "H",
                                                 "mode": "binary", "provenance": prov, "items": out_items})
    print(f"wrote {p}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("nli", "fm", "minicheck", "hhem"):
        p = sub.add_parser(name)
        p.add_argument("--out", type=Path, required=True)
        p.add_argument("--items", choices=("all", "corpus", "controls", "fallback"), default="all")
        p.add_argument("--limit", type=int)
        if name in ("nli", "fm"):
            p.add_argument("--base-url", default="http://127.0.0.1:8790")
        if name == "nli":
            p.add_argument("--model", required=True)
            p.add_argument("--model-dir", required=True, help="the installed model's directory")
        if name == "hhem":
            p.add_argument("--max-premise-tokens", type=int)
    a = ap.parse_args(argv)
    return {"nli": run_nli, "fm": run_fm, "minicheck": run_minicheck, "hhem": run_hhem}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
