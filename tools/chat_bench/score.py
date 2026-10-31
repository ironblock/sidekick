"""Score chat benchmark runs (tools/chat_bench/run.py) against the task set's
gold labels, with bootstrap 95% confidence intervals over items, and write
docs/data/chat-bench/results.json.

    python tools/chat_bench/score.py --runs <runs dir> [--embed-url http://127.0.0.1:8806] \\
        [--out docs/data/chat-bench/results.json] [--doc docs/CHAT-BENCHMARK.md]

`--embed-url` names a sidekickd serving `--embed-model` (default
bge-small-en-v1.5) for the titles' embedding similarity; without it that
metric is left out. Embeddings are cached under .dev-data/chat-bench/.
`--doc` rewrites the generated tables between the doc's
`<!-- chat-bench:begin -->` and `<!-- chat-bench:end -->` markers.

What each metric means is documented in docs/CHAT-BENCHMARK.md; how a
field value is compared is in fixtures/tasks/chat/tasks.toml.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
import sys
import unicodedata
import urllib.request
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from chat_bench import tasks  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
OUT_SCHEMA = "sidekick-chat-bench-results/1"
BOOT = 2000
SEED = 174
ROW_ORDER = ["afm", "afm-prompt", "afm-required", "minicpm5-1b", "minicpm5-1b-guided", "minicpm5-1b-ref", "minicpm5-2b",
             "minicpm5-2b-torch", "minicpm5-2b-torch-guided", "lfm3-1.7b-nothink", "lfm3-1.7b-nothink-guided",
             "lfm3-1.7b-think", "lfm3-1.7b-think-guided", "lfm3-1.7b-think-512",
             "smollm3-3b", "k2-horizon-0.9b-nothink", "k2-horizon-0.9b-low", "k2-horizon-0.9b-high", "smollm2-360m"]


# ---------------------------------------------------------------- statistics


def boot_mean(values, seed=SEED) -> dict:
    v = np.asarray([x for x in values if x is not None], dtype=float)
    if len(v) == 0:
        return {"value": None, "lo": None, "hi": None, "n": 0}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), size=(BOOT, len(v)))
    means = v[idx].mean(axis=1)
    return {"value": round(float(v.mean()), 4), "lo": round(float(np.percentile(means, 2.5)), 4),
            "hi": round(float(np.percentile(means, 97.5)), 4), "n": int(len(v))}


def boot_ratio(num, den, seed=SEED) -> dict:
    """A micro-averaged rate (sum num / sum den) with items resampled."""
    n, d = np.asarray(num, float), np.asarray(den, float)
    if d.sum() == 0:
        return {"value": None, "lo": None, "hi": None, "n": 0}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(n), size=(BOOT, len(n)))
    dd = d[idx].sum(1)
    r = n[idx].sum(1)[dd > 0] / dd[dd > 0]
    return {"value": round(float(n.sum() / d.sum()), 4), "lo": round(float(np.percentile(r, 2.5)), 4),
            "hi": round(float(np.percentile(r, 97.5)), 4), "n": int(d.sum())}


def boot_paired(a: dict, b: dict, seed=SEED) -> dict:
    """Mean of a[item] - b[item] over shared items, with its bootstrap CI."""
    keys = sorted(set(a) & set(b))
    if not keys:
        return {"value": None, "lo": None, "hi": None, "n": 0}
    return boot_mean([a[k] - b[k] for k in keys], seed)


# ---------------------------------------------------------------- text helpers

_THINK = re.compile(r"<(ifm\|)?think(_fast|_faster)?>.*?</(ifm\|)?think(_fast|_faster)?>", re.S)
_OPEN_THINK = re.compile(r"<(ifm\|)?think(_fast|_faster)?>")
_CLOSE_THINK = re.compile(r"</(?:ifm\|)?think(?:_fast|_faster)?>")


def final_text(content: str) -> str:
    """The reply without any thinking block a server left in the content.
    A closing tag with no opening one (a model that thinks after the template
    already closed its block) ends the thinking too: the answer is what
    follows the last one, as a reasoning parser would split it."""
    text = _THINK.sub("", content or "")
    if _OPEN_THINK.search(text):          # thinking never closed: no answer
        return ""
    return _CLOSE_THINK.split(text)[-1].strip()


def norm(s) -> str:
    s = unicodedata.normalize("NFKC", str(s)).casefold()
    s = re.sub(r"[\s\-_/]+", " ", s)
    s = re.sub(r"[^\w\s]", "", s)
    s = re.sub(r"^the ", "", s.strip())
    return re.sub(r"\s+", " ", s).strip()


_STEM = None


def stem_tokens(s: str) -> list[str]:
    global _STEM
    if _STEM is None:
        from nltk.stem.porter import PorterStemmer
        _STEM = PorterStemmer()
    return [_STEM.stem(t) for t in norm(s).split()]


def parse_json(content: str) -> tuple[object | None, bool]:
    """(value, strict): strict when the whole reply is the JSON value; else
    the first {...} block, from a code fence or surrounding prose."""
    text = final_text(content)
    try:
        return json.loads(text), True
    except ValueError:
        pass
    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    candidates = [m.group(1)] if m else []
    start = text.find("{")
    if start >= 0:
        depth = 0
        for i, ch in enumerate(text[start:], start):
            depth += ch == "{"
            depth -= ch == "}"
            if depth == 0:
                candidates.append(text[start:i + 1])
                break
    for c in candidates:
        try:
            return json.loads(c), False
        except ValueError:
            continue
    return None, False


# ---------------------------------------------------------------- titles

_TITLE_PREFIX = re.compile(r"^\s*(\*\*)?title\s*:", re.I)


def clean_title(text: str) -> str:
    t = final_text(text).splitlines()[0] if final_text(text) else ""
    t = _TITLE_PREFIX.sub("", t).strip().strip("*#").strip()
    t = t.strip("\"'“”‘’`").strip().rstrip(".!?:;").strip()
    return t


def title_format(text: str, max_words: int) -> dict:
    raw = final_text(text)
    words = len(raw.split())
    checks = {
        "nonempty": bool(raw),
        "single_line": "\n" not in raw,
        "max_words": 0 < words <= max_words,
        "no_quotes": not re.match(r"^[\"'“‘`]", raw) and not re.search(r"[\"'”’`]$", raw),
        "no_prefix": not _TITLE_PREFIX.match(raw),
        "no_trailing_punct": not re.search(r"[.!?:;]$", raw),
        "no_markdown": not re.search(r"[*#`]", raw),
    }
    return checks


class Embedder:
    def __init__(self, url: str | None, model: str, cache: Path):
        self.url, self.model, self.cache_path = url, model, cache
        self.cache = json.loads(cache.read_text()) if cache.exists() else {}

    def key(self, s: str) -> str:
        return hashlib.sha256(f"{self.model}\0{s}".encode()).hexdigest()

    def embed(self, texts: list[str]) -> list[np.ndarray] | None:
        if not self.url:
            return None
        todo = sorted({t for t in texts if self.key(t) not in self.cache})
        for i in range(0, len(todo), 64):
            batch = todo[i:i + 64]
            body = json.dumps({"model": self.model, "input": batch, "input_type": "none"}).encode()
            req = urllib.request.Request(self.url.rstrip("/") + "/v1/embeddings", data=body,
                                         headers={"content-type": "application/json"})
            with urllib.request.urlopen(req, timeout=120) as r:
                data = json.loads(r.read())["data"]
            for t, d in zip(batch, data):
                self.cache[self.key(t)] = d["embedding"]
        if todo:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_text(json.dumps(self.cache))
        out = []
        for t in texts:
            v = np.asarray(self.cache[self.key(t)], float)
            out.append(v / (np.linalg.norm(v) or 1.0))
        return out


def loo_best(scores: list[float]) -> float:
    """A title's score against the references, on the ceiling's footing: the
    best match among all references but one, averaged over which one is left
    out. The ceiling scores each reference against the other two, so a model
    and a reference are both compared with two references."""
    n = len(scores)
    return sum(max(scores[:j] + scores[j + 1:]) for j in range(n)) / n


def score_titles(recs: dict, items: list[dict], cfg: dict, emb: Embedder) -> tuple[dict, dict]:
    from rouge_score import rouge_scorer
    rs = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    max_words = cfg["titles"]["max_words"]
    per: dict[str, dict] = {}
    for it in items:
        r = recs.get(it["id"])
        if r is None:
            continue
        content = r["content"] if r["status"] == 200 else ""
        fmt = title_format(content, max_words)
        title = clean_title(content)
        rouge = loo_best([rs.score(ref, title)["rougeL"].fmeasure for ref in it["references"]]) \
            if title else 0.0
        per[it["id"]] = {"format_ok": float(all(fmt.values())), "len_ok": float(fmt["max_words"]),
                         "rouge_l": rouge, "title": title, **{f"fmt_{k}": float(v) for k, v in fmt.items()}}
    if emb.url and per:
        ids = list(per)
        titles = [per[i]["title"] or " " for i in ids]
        refs = {it["id"]: it["references"] for it in items}
        flat = [x for i in ids for x in refs[i]]
        tv, rv = emb.embed(titles), emb.embed(flat)
        k = 0
        for i, t in zip(ids, tv):
            n = len(refs[i])
            sims = [float(t @ rv[k + j]) for j in range(n)]
            per[i]["emb_sim"] = loo_best(sims) if per[i]["title"] else 0.0
            k += n
    vals = list(per.values())
    m = {"format_ok": boot_mean([v["format_ok"] for v in vals]),
         "len_ok": boot_mean([v["len_ok"] for v in vals]),
         "rouge_l": boot_mean([v["rouge_l"] for v in vals])}
    if vals and "emb_sim" in vals[0]:
        m["emb_sim"] = boot_mean([v["emb_sim"] for v in vals])
    for k in [k for k in (vals[0] if vals else {}) if k.startswith("fmt_")]:
        m[k] = boot_mean([v[k] for v in vals])
    return m, {i: {"rouge_l": v["rouge_l"], "emb_sim": v.get("emb_sim"), "format_ok": v["format_ok"]}
               for i, v in per.items()}


def title_ceiling(items: list[dict], emb: Embedder) -> dict:
    """Reference against the other references (leave one out): how well a
    human-quality title scores, the softest metric's ceiling."""
    from rouge_score import rouge_scorer
    rs = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rouge, sim = [], []
    for it in items:
        refs = it["references"]
        for j, ref in enumerate(refs):
            others = refs[:j] + refs[j + 1:]
            rouge.append(max(rs.score(o, ref)["rougeL"].fmeasure for o in others))
    out = {"rouge_l": boot_mean(rouge)}
    if emb.url:
        flat = [x for it in items for x in it["references"]]
        vs = emb.embed(flat)
        k = 0
        for it in items:
            n = len(it["references"])
            block = vs[k:k + n]
            for j in range(n):
                sim.append(max(float(block[j] @ block[o]) for o in range(n) if o != j))
            k += n
        out["emb_sim"] = boot_mean(sim)
    return out


# ---------------------------------------------------------------- tags


def score_tags_vocab(recs: dict, items: list[dict], cfg: dict) -> tuple[dict, dict]:
    import jsonschema
    vocab = set(cfg["tags"]["vocabulary"])
    schema = tasks.tags_schema(cfg)
    per = {}
    for it in items:
        r = recs.get(it["id"])
        if r is None:
            continue
        val, strict = parse_json(r["content"]) if r["status"] == 200 else (None, False)
        valid = val is not None and jsonschema.Draft202012Validator(schema).is_valid(val)
        tags = []
        if isinstance(val, dict) and isinstance(val.get("tags"), list):
            tags = [str(t).strip().lower() for t in val["tags"]]
        off = [t for t in tags if t not in vocab]
        pred = list(dict.fromkeys(t for t in tags if t in vocab))
        gold = set(it["vocab_gold"])
        tp = len(set(pred) & gold)
        p = tp / len(pred) if pred else 0.0
        rcl = tp / len(gold)
        f1 = 2 * p * rcl / (p + rcl) if p + rcl else 0.0
        per[it["id"]] = {"parse_ok": float(val is not None), "strict_json": float(val is not None and strict),
                         "schema_valid": float(valid), "precision": p, "recall": rcl, "f1": f1,
                         "top1": float(bool(pred) and pred[0] in gold), "off_vocab": len(off), "n_tags": len(tags),
                         "fail": float(r["status"] != 200 or val is None)}
    v = list(per.values())
    m = {k: boot_mean([x[k] for x in v]) for k in ("f1", "precision", "recall", "top1", "schema_valid",
                                                   "parse_ok", "strict_json", "fail")}
    m["off_vocab_rate"] = boot_ratio([x["off_vocab"] for x in v], [x["n_tags"] for x in v])
    return m, {i: {"f1": x["f1"], "schema_valid": x["schema_valid"]} for i, x in per.items()}


def split_free_tags(text: str) -> list[str]:
    t = final_text(text)
    t = re.sub(r"^\s*tags?\s*:", "", t, flags=re.I)
    parts = re.split(r"[,\n;]", t)
    out = []
    for p in parts:
        p = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", p).strip().strip("#").strip().strip("\"'").strip()
        if p:
            out.append(p)
    return out


def tag_matches(tag: str, aliases: list[str]) -> bool:
    ts = set(stem_tokens(tag))
    if not ts:
        return False
    for a in aliases:
        av = set(stem_tokens(a))
        if av and (av <= ts or ts <= av):
            return True
    return False


def score_tags_free(recs: dict, items: list[dict], cfg: dict) -> tuple[dict, dict]:
    fc = cfg["tags"]["free"]
    per = {}
    for it in items:
        r = recs.get(it["id"])
        if r is None:
            continue
        raw = final_text(r["content"]) if r["status"] == 200 else ""
        tags = split_free_tags(raw)
        fmt = (fc["min_tags"] <= len(tags) <= fc["max_tags"]
               and all(t == t.lower() for t in tags)
               and all(len(t.split()) <= fc["max_tag_words"] for t in tags)
               and "\n" not in raw and not re.match(r"^\s*tags?\s*:", raw, re.I)
               and len(set(norm(t) for t in tags)) == len(tags))
        concepts = it["free_gold"]
        hit = [any(tag_matches(t, c) for t in tags) for c in concepts]
        on_gold = [any(tag_matches(t, c) for c in concepts) for t in tags]
        text_stems = set(stem_tokens(it["text"]))
        grounded = [g or (bool(stem_tokens(t)) and set(stem_tokens(t)) <= text_stems)
                    for t, g in zip(tags, on_gold)]
        per[it["id"]] = {"format_ok": float(fmt), "concept_recall": sum(hit) / len(concepts),
                         "on_gold": sum(on_gold), "grounded": sum(grounded), "n": len(tags)}
    v = list(per.values())
    m = {"concept_recall": boot_mean([x["concept_recall"] for x in v]),
         "format_ok": boot_mean([x["format_ok"] for x in v]),
         "on_gold_precision": boot_ratio([x["on_gold"] for x in v], [x["n"] for x in v]),
         "grounded_rate": boot_ratio([x["grounded"] for x in v], [x["n"] for x in v])}
    return m, {i: {"concept_recall": x["concept_recall"], "format_ok": x["format_ok"]} for i, x in per.items()}


# ---------------------------------------------------------------- extraction


def _num(x):
    if isinstance(x, bool):
        return None
    if isinstance(x, (int, float)):
        return float(x)
    try:
        return float(str(x).replace(",", "").strip())
    except ValueError:
        return None


def text_match(p, g) -> bool:
    """Equal after normalising, or one is a whole-word run inside the other
    ("Shakshuka" in "Shakshuka for two", "Grand Lodge Hotel" in "the ballroom
    of the Grand Lodge Hotel")."""
    a, b = norm(p), norm(g)
    if not a or not b:
        return False
    if a == b:
        return True
    short, long_ = sorted((a, b), key=len)
    return len(short) >= 3 and f" {short} " in f" {long_} "


def token_f1(p, g) -> float:
    a, b = stem_tokens(str(p)), stem_tokens(str(g))
    if not a or not b:
        return 0.0
    common = sum(min(a.count(t), b.count(t)) for t in set(a))
    if not common:
        return 0.0
    pr, rc = common / len(a), common / len(b)
    return 2 * pr * rc / (pr + rc)


def set_f1(p, g) -> float:
    if not isinstance(p, list):
        return 0.0
    p = [str(x) for x in p if str(x).strip()]
    used, tp = set(), 0
    for gv in g:
        for j, pv in enumerate(p):
            if j not in used and (text_match(pv, gv) or token_f1(pv, gv) >= 0.5):
                used.add(j)
                tp += 1
                break
    if not p or not tp:
        return 0.0
    pr, rc = tp / len(p), tp / len(g)
    return 2 * pr * rc / (pr + rc)


def field_score(rule: str, p, g) -> float:
    if p is None:
        return 0.0
    if rule == "text":
        return float(text_match(p, g))
    if rule == "summary":
        return float(token_f1(p, g) >= 0.5)
    if rule == "code":
        return float(re.sub(r"[\s\-]", "", str(p)).casefold() == re.sub(r"[\s\-]", "", str(g)).casefold())
    if rule == "digits":
        return float(re.sub(r"\D", "", str(p)) == re.sub(r"\D", "", str(g)) != "")
    if rule == "number":
        x = _num(p)
        return float(x is not None and abs(x - float(g)) < 0.005)
    if rule == "set":
        return set_f1(p, g)
    # exact: booleans and integers compare as values; strings trimmed
    if isinstance(g, bool):
        return float(p is g)
    if isinstance(g, int):
        return float(not isinstance(p, bool) and _num(p) is not None and _num(p) == g)
    return float(str(p).strip() == str(g).strip())


def is_empty(v) -> bool:
    return v is None or v == "" or v == [] or v == {}


def score_extraction(recs: dict, items: list[dict], cfg: dict, schemas: dict) -> tuple[dict, dict]:
    import jsonschema
    rules = cfg["extraction"]["match"]
    per = {}
    for it in items:
        r = recs.get(it["id"])
        if r is None:
            continue
        schema = schemas[it["schema"]]
        val, strict = parse_json(r["content"]) if r["status"] == 200 else (None, False)
        obj = val if isinstance(val, dict) else None
        valid = obj is not None and jsonschema.Draft202012Validator(schema).is_valid(obj)
        props = schema["properties"]
        extra = [k for k in (obj or {}) if k not in props]
        # `either` fields are inferred rather than stated: leaving one out
        # is as right as giving the gold value.
        either = set(it.get("either", []))
        present = {k: 1.0 if k in either and obj is not None and is_empty(obj.get(k))
                   else field_score(rules.get(k, "exact"), (obj or {}).get(k), g)
                   for k, g in it["gold"].items()}
        absent = {k: float(is_empty(obj.get(k))) for k in props if k not in it["gold"]} if obj is not None else {}
        refusal = r["status"] == 400 or bool(re.search(r"\b(i can'?t|i cannot|i'm sorry|unable to)\b",
                                                        final_text(r["content"]), re.I) and obj is None)
        per[it["id"]] = {"parse_ok": float(obj is not None), "strict_json": float(obj is not None and strict),
                         "schema_valid": float(valid), "extra_keys": float(bool(extra)),
                         "field_hits": sum(present.values()), "fields": len(present),
                         "item_field_acc": sum(present.values()) / len(present),
                         "absent_ok": sum(absent.values()), "absent_n": len(absent),
                         "exact": float(obj is not None and all(v == 1.0 for v in present.values())
                                        and all(absent.values())),
                         "fail": float(obj is None), "refusal": float(refusal),
                         "length_cut": float(r.get("finish_reason") == "length")}
    v = list(per.values())
    m = {"field_acc": boot_ratio([x["field_hits"] for x in v], [x["fields"] for x in v]),
         "item_field_acc": boot_mean([x["item_field_acc"] for x in v]),
         "absent_ok": boot_ratio([x["absent_ok"] for x in v], [x["absent_n"] for x in v]),
         **{k: boot_mean([x[k] for x in v]) for k in ("exact", "schema_valid", "parse_ok", "strict_json",
                                                      "extra_keys", "fail", "refusal", "length_cut")}}
    return m, {i: {"item_field_acc": x["item_field_acc"], "schema_valid": x["schema_valid"]}
               for i, x in per.items()}


# ---------------------------------------------------------------- instructions

_LANGS = {"en": "ENGLISH", "fr": "FRENCH", "es": "SPANISH", "de": "GERMAN", "it": "ITALIAN",
          "pt": "PORTUGUESE", "ja": "JAPANESE", "zh": "CHINESE", "ko": "KOREAN"}
_DETECTOR = None


def detect_lang(text: str) -> str | None:
    global _DETECTOR
    from lingua import Language, LanguageDetectorBuilder
    if _DETECTOR is None:
        _DETECTOR = LanguageDetectorBuilder.from_languages(
            *[getattr(Language, v) for v in _LANGS.values()]).build()
    lang = _DETECTOR.detect_language_of(text)
    if lang is None:
        return None
    inv = {v: k for k, v in _LANGS.items()}
    return inv.get(lang.name)


def _answer_norm(s: str) -> str:
    s = s.strip().replace("−", "-")
    s = s.strip("\"'“”‘’`*").strip()
    if s.endswith("."):
        s = s[:-1]
    return s.strip().casefold()


def sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?。！？])\s+|(?<=[。！？])", text.strip())
    return [p for p in parts if p.strip()]


def run_check(c: dict, text: str) -> bool:
    t = c["type"]
    if t == "max_words":
        return 0 < len(text.split()) <= c["n"]
    if t == "max_sentences":
        return 0 < len(sentences(text)) <= c["n"]
    if t == "bullet_count":
        lines = [ln for ln in text.splitlines() if ln.strip()]
        return len(lines) == c["n"] and all(ln.lstrip().startswith("- ") for ln in lines)
    if t == "language":
        return bool(text) and detect_lang(text) == c["lang"]
    if t == "answer_only":
        return _answer_norm(text) in {o.casefold() for o in c["options"]}
    if t == "answer_only_number":
        return bool(re.fullmatch(r"[+-]?\d+(?:\.\d+)?", _answer_norm(text)))
    if t == "lowercase":
        return bool(text) and text == text.lower()
    if t == "starts_with":
        return text.startswith(c["prefix"])
    if t == "ends_with":
        return text.rstrip().endswith(c["suffix"])
    if t == "forbidden_words":
        return bool(text) and not any(re.search(rf"\b{re.escape(w)}\b", text, re.I) for w in c["words"])
    if t == "max_chars":
        return 0 < len(text) <= c["n"]
    if t == "single_line":
        return bool(text) and "\n" not in text
    raise ValueError(f"unknown check {t}")


def answer_ok(it: dict, text: str) -> bool:
    accept = it["answer"]["accept"]
    exact = any(c["type"] in ("answer_only", "answer_only_number") for c in it["checks"])
    if exact:
        a = _answer_norm(text)
        for acc in accept:
            x, y = _num(a), _num(acc)
            if a == acc.casefold() or (x is not None and y is not None and abs(x - y) < 1e-9):
                return True
        return False
    return any(acc.casefold() in text.casefold() for acc in accept)


def score_instructions(recs: dict, items: list[dict]) -> tuple[dict, dict]:
    per, by_type = {}, {}
    for it in items:
        r = recs.get(it["id"])
        if r is None:
            continue
        text = final_text(r["content"]) if r["status"] == 200 else ""
        res = [run_check(c, text) for c in it["checks"]]
        for c, ok in zip(it["checks"], res):
            by_type.setdefault(c["type"], []).append(float(ok))
        per[it["id"]] = {"all_checks": float(all(res)), "checks_ok": sum(res), "checks": len(res),
                         "answer": float(answer_ok(it, text)) if "answer" in it else None}
    v = list(per.values())
    m = {"all_checks": boot_mean([x["all_checks"] for x in v]),
         "check_rate": boot_ratio([x["checks_ok"] for x in v], [x["checks"] for x in v]),
         "answer_acc": boot_mean([x["answer"] for x in v]),
         "by_check": {k: boot_mean(vs) for k, vs in sorted(by_type.items())}}
    return m, {i: {"all_checks": x["all_checks"]} for i, x in per.items()}


# ---------------------------------------------------------------- latency


def latency(records: list[dict]) -> dict:
    ok = [r for r in records if r["status"] == 200 and r["total_s"] is not None]
    if not ok:
        return {}
    ttft = [r["ttft_s"] for r in ok if r["ttft_s"] is not None]
    # First answer token, after any thinking (what a caller waiting for the
    # reply sees); a reply with no content counts its total time.
    ttfc = [r["ttft_content_s"] if r.get("ttft_content_s") is not None else r["total_s"] for r in ok]
    total = [r["total_s"] for r in ok]
    toks = [r["usage"]["completion_tokens"] for r in ok if r.get("usage")]
    reason = [((r["usage"].get("completion_tokens_details") or {}).get("reasoning_tokens") or 0)
              for r in ok if r.get("usage")]
    rates = [(r["usage"]["completion_tokens"] - 1) / (r["total_s"] - r["ttft_s"]) for r in ok
             if r.get("usage") and r["ttft_s"] is not None and r["usage"]["completion_tokens"] > 4
             and r["total_s"] - r["ttft_s"] > 0.05]

    def q(xs, p):
        return round(float(np.percentile(xs, p)), 3) if xs else None
    e2e = [r["usage"]["completion_tokens"] / r["total_s"] for r in ok if r.get("usage") and r["total_s"] > 0]
    return {"n": len(ok), "ttft_median_s": q(ttft, 50), "ttft_p90_s": q(ttft, 90),
            "ttft_content_median_s": q(ttfc, 50), "ttft_content_p90_s": q(ttfc, 90),
            "output_tok_s_median": round(statistics.median(e2e), 1) if e2e else None,
            "total_median_s": q(total, 50), "total_p90_s": q(total, 90),
            "output_tokens_mean": round(statistics.mean(toks), 1) if toks else None,
            "reasoning_tokens_mean": round(statistics.mean(reason), 1) if reason and any(reason) else None,
            "decode_tok_s_median": round(statistics.median(rates), 1) if rates else None,
            "length_cut_rate": round(sum(r.get("finish_reason") == "length" for r in ok) / len(ok), 4)}


# ---------------------------------------------------------------- report

GUIDED_PAIRS = [("afm", "afm-prompt"), ("afm-required", "afm-prompt"), ("minicpm5-1b-guided", "minicpm5-1b-ref"),
                ("minicpm5-2b-torch-guided", "minicpm5-2b-torch"), ("lfm3-1.7b-nothink-guided", "lfm3-1.7b-nothink"),
                ("lfm3-1.7b-think-guided", "lfm3-1.7b-think")]
HEADLINE = {"titles": "rouge_l", "tags_vocab": "f1", "tags_free": "concept_recall",
            "extraction": "item_field_acc", "instructions": "all_checks"}


def score_run(run: dict, cfg: dict, items: dict, schemas: dict, emb: Embedder) -> tuple[dict, dict]:
    recs: dict[str, dict] = {}
    for r in run["records"]:
        recs.setdefault(r["family"], {})[r["item"]] = r
    fam, per = {}, {}
    if "titles" in recs:
        fam["titles"], per["titles"] = score_titles(recs["titles"], items["titles"], cfg, emb)
    if "tags_vocab" in recs:
        fam["tags_vocab"], per["tags_vocab"] = score_tags_vocab(recs["tags_vocab"], items["tags"], cfg)
    if "tags_free" in recs:
        fam["tags_free"], per["tags_free"] = score_tags_free(recs["tags_free"], items["tags"], cfg)
    if "extraction" in recs:
        fam["extraction"], per["extraction"] = score_extraction(recs["extraction"], items["extraction"], cfg,
                                                                schemas)
    if "instructions" in recs:
        fam["instructions"], per["instructions"] = score_instructions(recs["instructions"], items["instructions"])
    for f in fam:
        rs = [r for r in run["records"] if r["family"] == f]
        fam[f]["latency"] = latency(rs)
        # A refusal (the model's guardrails, an HTTP 400 content_filter) scores
        # zero on every metric above; here is how often it happened.
        fam[f]["refused"] = boot_mean([float(r["status"] == 400 and (r["error"] or {}).get("code") == "content_filter")
                                       for r in rs])
        fam[f]["errors"] = boot_mean([float(r["status"] != 200 and (r["error"] or {}).get("code") != "content_filter")
                                      for r in rs])
        # The headline metric over the requests the model answered.
        key = HEADLINE[f]
        fam[f]["headline_answered"] = boot_mean([v[key] for i, v in per[f].items()
                                                 if recs[f][i]["status"] == 200 and v.get(key) is not None])
    return fam, per


def repeatability(a: dict, b: dict) -> dict:
    """Share of requests whose reply is byte-identical between two runs of a
    row, and the share that is the same once JSON replies are parsed (the
    same values with the object's keys in another order)."""
    ra = {(r["family"], r["item"]): r["content"] for r in a["records"]}
    rb = {(r["family"], r["item"]): r["content"] for r in b["records"]}
    keys = sorted(set(ra) & set(rb))
    fams, same = {}, {}
    for k in keys:
        fams.setdefault(k[0], []).append(float(ra[k] == rb[k]))
        pa, pb = parse_json(ra[k])[0], parse_json(rb[k])[0]
        same.setdefault(k[0], []).append(float(ra[k] == rb[k] or (pa is not None and pa == pb)))

    def share(d):
        return round(sum(sum(v) for v in d.values()) / max(len(keys), 1), 4)
    return {"identical": share(fams), "same_values": share(same), "n": len(keys),
            "by_family": {f: round(sum(v) / len(v), 4) for f, v in fams.items()},
            "same_values_by_family": {f: round(sum(v) / len(v), 4) for f, v in same.items()}}


def summarise_row(name: str, run: dict) -> dict:
    row = run["row"]
    label = row["label"]
    if name != row["id"]:                          # a run from another machine (--suffix)
        label += f" ({run['machine']['chip']}, macOS {run['machine']['macos']})"
    return {"label": label, "server": row["server"], "json": row["json"],
            "model": run["model"], "stack": run["stack"], "machine": run["machine"],
            "started": run["started"], "requests": len(run["records"]),
            "chat_template_args": row.get("chat_template_args"),
            "reasoning_budget": row.get("reasoning_budget"), "prefill": row.get("prefill")}


def fmt_ci(m: dict | None, pct=True) -> str:
    if not m or m.get("value") is None:
        return "–"
    f = (lambda x: f"{100 * x:.0f}") if pct else (lambda x: f"{x:.2f}")
    return f"{f(m['value'])} ({f(m['lo'])}–{f(m['hi'])})"


def markdown(res: dict) -> str:
    rows = [r for r in ROW_ORDER if r in res["rows"]] + sorted(r for r in res["rows"] if r not in ROW_ORDER)
    lab = {r: res["rows"][r]["label"] for r in rows}
    L = []

    def table(title, fam, cols):
        rs = [r for r in rows if fam in res["rows"][r]["scores"]]
        if not rs:
            return
        L.append(f"**{title}**\n")
        L.append("| Row | " + " | ".join(c[0] for c in cols) + " |")
        L.append("|---|" + "---|" * len(cols))
        for r in rs:
            s = res["rows"][r]["scores"][fam]
            cells = []
            for _, key, pct in cols:
                v = s
                for part in key.split("."):
                    v = v.get(part) if isinstance(v, dict) else None
                cells.append(fmt_ci(v, pct))
            L.append(f"| {lab[r]} | " + " | ".join(cells) + " |")
        L.append("")

    table("Titles", "titles", [("ROUGE-L", "rouge_l", True), ("Embedding similarity", "emb_sim", False),
                               ("Format OK %", "format_ok", True), ("≤ 6 words %", "len_ok", True)])
    c = res.get("title_ceiling")
    if c:
        L.append(f"Reference against the other references: ROUGE-L {fmt_ci(c['rouge_l'])}, embedding "
                 f"similarity {fmt_ci(c.get('emb_sim'), False)}.\n")
    table("Tags, fixed vocabulary", "tags_vocab",
          [("F1", "f1", True), ("Precision", "precision", True), ("Recall", "recall", True),
           ("Top-1 %", "top1", True), ("Schema-valid %", "schema_valid", True),
           ("Off-vocabulary tags %", "off_vocab_rate", True)])
    table("Tags, free-form", "tags_free",
          [("Concept recall", "concept_recall", True), ("On-gold precision", "on_gold_precision", True),
           ("Grounded %", "grounded_rate", True), ("Format OK %", "format_ok", True)])
    table("Extraction to a JSON Schema", "extraction",
          [("Field accuracy (item mean)", "item_field_acc", True), ("Field accuracy (all fields)", "field_acc", True),
           ("Absent fields left out %", "absent_ok", True), ("Exact items %", "exact", True),
           ("Schema-valid %", "schema_valid", True), ("Bare JSON %", "strict_json", True),
           ("Failed %", "fail", True)])
    table("Short instruction following", "instructions",
          [("All checks pass %", "all_checks", True), ("Checks pass %", "check_rate", True),
           ("Answer correct %", "answer_acc", True)])
    # per-check breakdown
    types = sorted({t for r in rows for t in res["rows"][r]["scores"].get("instructions", {}).get("by_check", {})})
    if types:
        L.append("**Instruction following by check**\n")
        L.append("| Check | " + " | ".join(lab[r] for r in rows if "instructions" in res["rows"][r]["scores"]) + " |")
        L.append("|---|" + "---|" * sum("instructions" in res["rows"][r]["scores"] for r in rows))
        for t in types:
            cells = []
            for r in rows:
                s = res["rows"][r]["scores"].get("instructions")
                if s is None:
                    continue
                b = s["by_check"].get(t)
                cells.append(f"{100 * b['value']:.0f} (n={b['n']})" if b else "–")
            L.append(f"| `{t}` | " + " | ".join(cells) + " |")
        L.append("")
    # refusals
    L.append("**Refused by guardrails** (% of requests; a refusal scores zero above)\n")
    L.append("| Row | " + " | ".join(HEADLINE) + " |")
    L.append("|---|" + "---|" * len(HEADLINE))
    for r in rows:
        sc = res["rows"][r]["scores"]
        L.append(f"| {lab[r]} | " + " | ".join(fmt_ci(sc[f]["refused"]) if f in sc else "–" for f in HEADLINE) + " |")
    L.append("")
    for r in rows:
        sc = res["rows"][r]["scores"]
        if any(sc[f]["refused"]["value"] for f in sc):
            parts = [f"{f} {HEADLINE[f]} {fmt_ci(sc[f]['headline_answered'])}" for f in HEADLINE if f in sc]
            L.append(f"- {lab[r]}, over the requests it answered: " + "; ".join(parts) + ".")
    L.append("")
    # paired differences
    if res.get("paired"):
        L.append("**Difference from Foundation Models (paired over items, points)**\n")
        fams = list(HEADLINE)
        L.append("| Row | " + " | ".join(f"{f} ({HEADLINE[f]})" for f in fams) + " |")
        L.append("|---|" + "---|" * len(fams))
        for r in rows:
            p = res["paired"].get(r)
            if not p:
                continue
            cells = []
            for f in fams:
                d = p.get(f)
                cells.append("–" if not d or d["value"] is None else
                             f"{100 * d['value']:+.0f} ({100 * d['lo']:+.0f} to {100 * d['hi']:+.0f})")
            L.append(f"| {lab[r]} | " + " | ".join(cells) + " |")
        L.append("")
    # guided minus prompt-only
    if res.get("guided_vs_prompt"):
        L.append("**Guided minus prompt-only JSON** (same model and items, points)\n")
        L.append("| Rows | tags_vocab f1 | tags_vocab schema-valid | extraction item_field_acc | extraction schema-valid |")
        L.append("|---|---|---|---|---|")

        def pts(d):
            return "–" if not d or d["value"] is None else \
                f"{100 * d['value']:+.0f} ({100 * d['lo']:+.0f} to {100 * d['hi']:+.0f})"
        for pair, fams in res["guided_vs_prompt"].items():
            g, p = pair.split(" - ")
            tv, ex = fams.get("tags_vocab", {}), fams.get("extraction", {})
            L.append(f"| {lab.get(g, g)} − {lab.get(p, p)} | {pts(tv.get('f1'))} | {pts(tv.get('schema_valid'))} | "
                     f"{pts(ex.get('item_field_acc'))} | {pts(ex.get('schema_valid'))} |")
        L.append("")
    # latency
    L.append("**Latency** (median time to the first answer token / median total, seconds; mean output tokens, "
             "thinking included; decode rate is the median of (tokens − 1) / (total − first token))\n")
    L.append("| Row | " + " | ".join(HEADLINE) + " | Decode tok/s |")
    L.append("|---|" + "---|" * (len(HEADLINE) + 1))
    for r in rows:
        sc = res["rows"][r]["scores"]
        cells, rates = [], []
        for f in HEADLINE:
            lt = sc.get(f, {}).get("latency")
            if not lt:
                cells.append("–")
                continue
            extra = f", {lt['reasoning_tokens_mean']:.0f} thinking" if lt.get("reasoning_tokens_mean") else ""
            cells.append(f"{lt['ttft_content_median_s']:.2f} / {lt['total_median_s']:.2f}; "
                         f"{lt['output_tokens_mean']:.0f} tok{extra}")
            if lt.get("decode_tok_s_median"):
                rates.append(lt["decode_tok_s_median"])
        L.append(f"| {lab[r]} | " + " | ".join(cells) + f" | {statistics.median(rates):.0f} |" if rates else
                 f"| {lab[r]} | " + " | ".join(cells) + " | – |")
    L.append("")
    if res.get("repeatability"):
        L.append("**Repeatability** (share of replies byte-identical between two runs of the row)\n")
        for r, rep in res["repeatability"].items():
            fams = ", ".join(f"{f} {100 * v:.0f}%" for f, v in rep["by_family"].items())
            L.append(f"- {lab.get(r, r)}: {100 * rep['identical']:.0f}% of {rep['n']} ({fams}); "
                     f"{100 * rep['same_values']:.0f}% with the same values once JSON is parsed")
        L.append("")
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--embed-url")
    ap.add_argument("--embed-model", default="bge-small-en-v1.5")
    ap.add_argument("--out", default=str(REPO / "docs" / "data" / "chat-bench" / "results.json"))
    ap.add_argument("--doc")
    a = ap.parse_args()
    cfg = tasks.load_config()
    items = tasks.load_items()
    schemas = tasks.load_schemas()
    emb = Embedder(a.embed_url, a.embed_model, REPO / ".dev-data" / "chat-bench" / "embed-cache.json")
    runs = {}
    for p in sorted(Path(a.runs).glob("*.json")):
        run = json.loads(p.read_text())
        if run.get("schema") == "sidekick-chat-bench-run/1" and not run.get("limit"):
            runs[p.stem] = run
    lock = tasks.task_set_lock()
    res = {"schema": OUT_SCHEMA, "task_set": lock, "bootstrap": {"resamples": BOOT, "seed": SEED, "ci": 0.95},
           "embedding_model": a.embed_model if a.embed_url else None, "rows": {}, "paired": {},
           "repeatability": {}}
    per_items = {}
    for name, run in runs.items():
        if name.endswith(".repeat"):
            continue                               # a repeat run (<row>.repeat.json)
        if run["task_set"]["requests"] != lock["requests"]:
            print(f"warning: {name} ran a different task set ({run['task_set']['version']})", file=sys.stderr)
        scores, per = score_run(run, cfg, items, schemas, emb)
        res["rows"][name] = {**summarise_row(name, run), "scores": scores}
        per_items[name] = per
    for name, run in runs.items():
        if name.endswith(".repeat"):
            base = name.removesuffix(".repeat")
            if base in runs:
                res["repeatability"][base] = repeatability(runs[base], run)
    res["title_ceiling"] = title_ceiling(items["titles"], emb)
    if "afm" in per_items:
        for name, per in per_items.items():
            if name == "afm":
                continue
            d = {}
            for fam, key in HEADLINE.items():
                ref = per_items["afm"].get(fam)
                if fam in per and ref:
                    d[fam] = boot_paired({i: v[key] for i, v in per[fam].items() if v.get(key) is not None},
                                         {i: v[key] for i, v in ref.items() if v.get(key) is not None})
            # prompt-only JSON rows compare with each other too
            res["paired"][name] = d
    # What the constraint changes, same model and items: guided minus prompt-only.
    res["guided_vs_prompt"] = {}
    for g, p in GUIDED_PAIRS:
        if g in per_items and p in per_items:
            res["guided_vs_prompt"][f"{g} - {p}"] = {
                fam: {k: boot_paired({i: v[k] for i, v in per_items[g][fam].items()},
                                     {i: v[k] for i, v in per_items[p][fam].items()})
                      for k in (HEADLINE[fam], "schema_valid")}
                for fam in ("tags_vocab", "extraction") if fam in per_items[g] and fam in per_items[p]}
    Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n")
    md = markdown(res)
    print(md)
    if a.doc:
        doc = Path(a.doc)
        text = doc.read_text()
        b, e = "<!-- chat-bench:begin -->", "<!-- chat-bench:end -->"
        if b in text and e in text:
            head, rest = text.split(b, 1)
            _, tail = rest.split(e, 1)
            doc.write_text(f"{head}{b}\n<!-- Generated by tools/chat_bench/score.py from docs/data/chat-bench/results.json. -->\n\n"
                           f"{md}\n{e}{tail}")
        else:
            print(f"warning: {doc} has no chat-bench markers", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
