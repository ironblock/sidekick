"""Export the LLM judge (J) and second-reader requests as JSONL batches, and
read their responses back (design §2.4, §6.1, §6.5).

The batch runs outside this repository, one request at a time, on a model
the maintainer chooses, with reasoning on. Nothing here names an endpoint,
a key or a model: the model's public id is read from the response file.

Reader forms, each given exactly what the checker it shadows sees:
- `window-512`: the windows of the 512-token nli models, cut with
  nli-deberta-v3-base's tokenizer (exact for it; the other 512-token models'
  tokenizers cut a little differently), one window per request, with the
  per-window question. Aggregated by the same max rule; the gated variants
  use the subset of windows marked `anchored`.
- `window-2048`: the same for the 2,048-token model (bge-m3-zeroshot-v2.0-c).
- `chunk-400`: MiniCheck-DeBERTa-v3-L's 400-token chunks, approximated at
  line granularity (tools/tasks/scoring_windows.py `chunks`).
- `full`: the whole haystack. These requests are J itself, and the reader
  for full-context judging (Foundation Models, HHEM).

Every form includes the 18 controls, short and long. Identical requests are
sent once: a request's `serves` lists every (triple, variant, window) or
(control, window) it answers, and `shadows` the checkers it stands beside.

    python3 tools/tasks/scoring_export.py --out <dir> \\
        --tokenizer nli-deberta-v3-base=<path/to/tokenizer.json> \\
        --tokenizer bge-m3-zeroshot-v2.0-c=<path> --tokenizer minicheck=<path>

writes <dir>/full/ (every window) and <dir>/fallback/ (the 1k and 4k items,
the obstacle set and the controls in the window and chunk forms; the full
form unchanged), each with requests.jsonl, manifest.json and README.md, and
<dir>/SHA256SUMS.

Response file (one JSON object per line): `id`; `model` (the model's public
registry id); and either `content` (the final answer's text) or `response`
(an OpenAI chat.completion object, of which only
`choices[0].message.content` is read); optionally `sha256`, of `content`.
The verdict is parsed by scoring_judge's lenient (scored) and strict rules,
and both parse rates are reported per form.

    python3 tools/tasks/scoring_export.py --out <dir> --reestimate --measured-on <date> \\
        --seconds window-512=<s> --seconds window-2048=<s> --seconds chunk-400=<s> --seconds full=<s>

replaces the estimates in an exported batch with measured seconds per
request, rewriting only manifest.json, README.md and SHA256SUMS.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tasks import canon, scoring_controls, scoring_judge, scoring_set, scoring_windows as sw  # noqa: E402

OUTPUT_MODE = "final-json"
FORMS = ("window-512", "window-2048", "chunk-400", "full")
SHADOWS = {
    "window-512": ["A3:nli-deberta-v3-base", "A3g:nli-deberta-v3-base"],
    "window-2048": ["A2:bge-m3-zeroshot-v2.0-c", "A2g:bge-m3-zeroshot-v2.0-c"],
    "chunk-400": ["M:minicheck-deberta-v3-large"],
    "full": ["J", "F:foundation-models", "H:HHEM-2.1-Open"],
}


def request_id(form: str, body: dict) -> str:
    return f"{form}-{canon.sha256_text(canon.canonical_line(body))[:20]}"


def build(items: list[dict], cuts: dict[str, dict[str, list[sw.Window]]], prompt: dict, judge_cfg: dict,
          forms=FORMS, keep=lambda item, form: True) -> list[dict]:
    """Requests for every item in every form. `cuts[form][item]` are the
    item's windows (not needed for `full`)."""
    reqs: dict[str, dict] = {}
    for form in forms:
        for it in items:
            if not keep(it, form):
                continue
            tag = ({"triple": it["triple"], "variant": it["variant"]} if "triple" in it
                   else {"control": it["control"]})
            pieces = ([(None, it["evidence"])] if form == "full"
                      else [(i, w.text) for i, w in enumerate(cuts[form][it["item"]])])
            anchors = sw.anchors(it["claim"])
            for idx, text in pieces:
                msgs = scoring_judge.messages(prompt, "full" if form == "full" else "window", it["claim"], text)
                body = {"messages": msgs, "output_mode": OUTPUT_MODE, "max_tokens": judge_cfg["max_tokens"],
                        "reasoning_effort": judge_cfg["reasoning_effort"]}
                rid = request_id(form, body)
                if rid not in reqs:
                    reqs[rid] = {"id": rid, "form": form, "prompt_sha256": prompt["sha256"], **body,
                                 "shadows": SHADOWS[form], "serves": []}
                    if form != "full":
                        reqs[rid]["anchored"] = any(a in text for a in anchors)
                serve = dict(tag)
                if idx is not None:
                    serve["window" if form.startswith("window") else "chunk"] = idx
                reqs[rid]["serves"].append(serve)
    return sorted(reqs.values(), key=lambda r: (FORMS.index(r["form"]), r["id"]))


def estimate(reqs: list[dict], judge_cfg: dict, prefill_tps: float, decode_tps: float,
             answer_tokens: int = 16) -> dict:
    """Request count, prompt tokens (approx-v1) and wall-clock time at one
    request at a time, by form."""
    out, tot = {}, {"requests": 0, "prompt_tokens": 0, "output_tokens": 0, "hours": 0.0}
    for form in FORMS:
        rs = [r for r in reqs if r["form"] == form]
        if not rs:
            continue
        pt = sum(canon.approx_tokens("".join(m["content"] for m in r["messages"])) for r in rs)
        ot = len(rs) * (judge_cfg["assumed_reasoning_tokens"] + answer_tokens)
        hours = (pt / prefill_tps + ot / decode_tps) / 3600
        largest = max(canon.approx_tokens("".join(m["content"] for m in r["messages"])) for r in rs)
        out[form] = {"requests": len(rs), "prompt_tokens": pt, "output_tokens": ot, "hours": round(hours, 2),
                     "largest_prompt_tokens": largest}
        for k in ("requests", "prompt_tokens", "output_tokens"):
            tot[k] += out[form][k]
        tot["hours"] += hours
    tot["hours"] = round(tot["hours"], 2)
    return {"by_form": out, "total": tot,
            "assumptions": {"token_unit": canon.LENGTH_UNIT, "prefill_tokens_per_s": prefill_tps,
                            "decode_tokens_per_s": decode_tps,
                            "reasoning_tokens_per_request": judge_cfg["assumed_reasoning_tokens"],
                            "answer_tokens_per_request": answer_tokens, "concurrency": 1}}


def measured_estimate(reqs: list[dict], seconds: dict[str, float], measured_on: str) -> dict:
    """Wall-clock time at one request at a time from measured seconds per
    request, by form (replacing the token-rate assumptions)."""
    out, tot = {}, {"requests": 0, "prompt_tokens": 0, "hours": 0.0}
    for form in FORMS:
        rs = [r for r in reqs if r["form"] == form]
        if not rs:
            continue
        pt = sum(canon.approx_tokens("".join(m["content"] for m in r["messages"])) for r in rs)
        largest = max(canon.approx_tokens("".join(m["content"] for m in r["messages"])) for r in rs)
        hours = len(rs) * seconds[form] / 3600
        out[form] = {"requests": len(rs), "prompt_tokens": pt, "seconds_per_request": seconds[form],
                     "hours": round(hours, 2), "largest_prompt_tokens": largest}
        tot["requests"] += len(rs)
        tot["prompt_tokens"] += pt
        tot["hours"] += hours
    tot["hours"] = round(tot["hours"], 2)
    return {"by_form": out, "total": tot,
            "measured": {"on": measured_on, "seconds_per_request": dict(seconds), "concurrency": 1,
                         "token_unit": canon.LENGTH_UNIT}}


README = """# Verification scoring: judge and second-reader batch

`requests.jsonl` holds one request per line. Run them one at a time, with
reasoning on at effort `reasoning_effort`, and an output budget of
`max_tokens` (reasoning included). Send `messages` as given.

- `id`: return it unchanged.
- `form`: `full` (a whole haystack: the LLM judge J), `window-512`,
  `window-2048` (one window of a per-window nli checker) or `chunk-400`
  (one MiniCheck chunk).
- `prompt_sha256`: the judge prompt file's sha256.
- `output_mode`: `final-json`: the final answer (`content`) must end with
  one JSON object, `{{"verdict": "supported" | "contradicted" | "not_mentioned"}}`.
  Only `content` is read, never the reasoning (`reasoning_content`). The
  scored (lenient) rule allows prose before the object; prose only, two
  objects, or anything after the object is a failure, and counts as wrong.
  The strict rule (the content is exactly the object) is reported beside it.
- `serves`: the items this request answers: a triple id and its variant
  (S, C or NM), or a control id, with the window or chunk index.
- `shadows`: the checkers this request stands beside.
- `anchored` (window and chunk forms): whether the window holds one of the
  claim's anchors (the gated checkers use only those).

Return `responses.jsonl`, one line per request:
`{{"id": ..., "model": "<public registry id>", "content": "<final answer>", "sha256": "<sha256 of content>"}}`
(or `"response": <the chat.completion object>` in place of `content`), and
its sha256.

{summary}
"""


def write_batch(d: Path, reqs: list[dict], manifest: dict) -> None:
    d.mkdir(parents=True, exist_ok=True)
    body = "".join(canon.canonical_line(r) for r in reqs)
    (d / "requests.jsonl").write_text(body, encoding="utf-8")
    manifest = {**manifest, "requests_sha256": canon.sha256_text(body)}
    canon.write_json(d / "manifest.json", manifest)
    write_readme(d, manifest)


def write_readme(d: Path, manifest: dict) -> None:
    est = manifest["estimate"]
    measured = est.get("measured")
    head = "| form | requests | prompt tokens (approx-v1) |" + (" s/request |" if measured else "") + " est. hours |"
    lines = [head, "|---|---|---|" + ("---|" if measured else "") + "---|"]
    for form, e in est["by_form"].items():
        rate = f" {e['seconds_per_request']} |" if measured else ""
        lines.append(f"| {form} | {e['requests']} | {e['prompt_tokens']:,} |{rate} {e['hours']} |")
    t = est["total"]
    lines.append(f"| total | {t['requests']} | {t['prompt_tokens']:,} |" + (" |" if measured else "")
                 + f" {t['hours']} |")
    lines.append("")
    largest = max(e["largest_prompt_tokens"] for e in est["by_form"].values())
    if measured:
        lines.append(f"Hours use the seconds per request measured on {measured['on']}, one request at a time. "
                     f"The largest prompt is {largest:,} tokens (approx-v1), so the model needs that much context "
                     "plus the output budget.")
    else:
        a = est["assumptions"]
        lines.append(f"Hours assume one request at a time, {a['prefill_tokens_per_s']} prompt tokens/s, "
                     f"{a['decode_tokens_per_s']} output tokens/s and {a['reasoning_tokens_per_request']} reasoning "
                     f"tokens per request. The largest prompt is {largest:,} tokens (approx-v1), so the model needs "
                     "that much context plus the output budget.")
    (d / "README.md").write_text(README.format(summary="\n".join(lines)), encoding="utf-8")


def reestimate(root: Path, seconds: dict[str, float], measured_on: str) -> dict[str, dict]:
    """Replace each batch's estimate with measured rates. Only manifest.json
    and README.md change: requests.jsonl is read, checked against the
    manifest's sha256, and never written."""
    out = {}
    for d in sorted(p for p in root.iterdir() if (p / "requests.jsonl").exists()):
        manifest = canon.read_json(d / "manifest.json")
        if canon.sha256_file(d / "requests.jsonl") != manifest["requests_sha256"]:
            raise SystemExit(f"{d.name}/requests.jsonl does not match its manifest")
        reqs = canon.read_jsonl(d / "requests.jsonl")
        manifest["estimate"] = measured_estimate(reqs, seconds, measured_on)
        canon.write_json(d / "manifest.json", manifest)
        write_readme(d, manifest)
        out[d.name] = manifest["estimate"]
    sha256sums(root)
    return out


def sha256sums(root: Path) -> None:
    lines = []
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.name != "SHA256SUMS":
            lines.append(f"{canon.sha256_file(p)}  {p.relative_to(root).as_posix()}")
    (root / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------- responses


def read_responses(path: Path, requests: list[dict]) -> dict:
    """{"model", "by_id": {id: {"verdict", "parsed", "strict", "fenced"}},
    "missing", "bad_sha", "errors", "bad_prompt"}. `parsed` is the lenient
    (scored) rule, `strict` the strict one. Only `content` is read; any
    reasoning field, `usage` and timing fields are ignored. A line with a
    non-null `error`, or whose `prompt_sha256` differs from its request's,
    gets no verdict, as does a missing line.
    A request with no response, or one whose `sha256` doesn't match its
    content, gets no verdict (wrong, and counted)."""
    want = {r["id"] for r in requests}
    by_id, models, bad = {}, set(), []
    prompt_of = {r["id"]: r.get("prompt_sha256") for r in requests}
    errors, bad_prompt = [], []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        content = scoring_judge.final_content(r)
        if "sha256" in r and content is not None and hashlib.sha256(content.encode("utf-8")).hexdigest() != r["sha256"]:
            bad.append(r["id"])
            content = None
        if r.get("error"):  # a transport or server error: no answer
            errors.append(r["id"])
            content = None
        if r.get("prompt_sha256") and prompt_of.get(r["id"]) and r["prompt_sha256"] != prompt_of[r["id"]]:
            bad_prompt.append(r["id"])  # answered a different prompt than the request names
            content = None
        verdict, ok = scoring_judge.parse(content)
        by_id[r["id"]] = {"verdict": verdict, "parsed": ok, "strict": scoring_judge.parse_strict(content)[1],
                          "fenced": scoring_judge.fenced(content),
                          "literal": scoring_judge.parse(content, quoted_json_is_prose=False)[1],
                          "truncated": r.get("finish_reason") == "length"}
        if r.get("model"):
            models.add(r["model"])
    if len(models) > 1:
        raise SystemExit(f"{path.name}: responses name {len(models)} models: {sorted(models)}")
    missing = sorted(want - set(by_id))
    for rid in missing:
        by_id[rid] = {"verdict": None, "parsed": False, "strict": False, "fenced": False, "literal": False,
                      "truncated": False}
    return {"model": next(iter(models), None), "by_id": by_id, "missing": missing, "bad_sha": bad,
            "errors": errors, "bad_prompt": bad_prompt}


def parse_rates(requests: list[dict], responses: dict) -> dict[str, dict]:
    """Per reader form: the share of requests whose final answer parses by
    the lenient (scored) rule and by the strict one, and the share that
    wrap a valid verdict object in a code fence (scored as failures); the
    counts, with the lenient rule's literal reading beside; and, by id, the
    answers cut off by the output budget (finish_reason "length")."""
    out = {}
    for form in FORMS:
        ids = [r["id"] for r in requests if r["form"] == form]
        if not ids:
            continue
        b = responses["by_id"]
        lenient = sum(1 for i in ids if b[i]["parsed"])
        strict = sum(1 for i in ids if b[i]["strict"])
        fence = sum(1 for i in ids if b[i].get("fenced"))
        literal = sum(1 for i in ids if b[i].get("literal"))
        truncated = sorted(i for i in ids if b[i].get("truncated"))
        out[form] = {"requests": len(ids), "lenient": round(lenient / len(ids), 4),
                     "strict": round(strict / len(ids), 4), "fenced": round(fence / len(ids), 4),
                     "counts": {"strict": strict, "lenient_only": lenient - strict, "failed": len(ids) - lenient,
                                "fenced": fence, "lenient_literal": literal},
                     # finish_reason "length": the output budget ran out (scored wrong, listed by id)
                     "budget_truncations": truncated}
    return out


def reader_windows(requests: list[dict], responses: dict, form: str) -> dict[str, list[dict]]:
    """Per served item, its windows' verdicts in window order:
    {item: [{"index", "verdict", "parsed", "anchored"}]}."""
    out: dict[str, list[dict]] = {}
    for r in requests:
        if r["form"] != form:
            continue
        ans = responses["by_id"][r["id"]]
        for s in r["serves"]:
            item = (scoring_set.item_id(s["triple"], {"S": "s", "C": "c", "NM": "nm"}[s["variant"]])
                    if "triple" in s else s["control"])
            out.setdefault(item, []).append({"index": s.get("window", s.get("chunk", 0)), "verdict": ans["verdict"],
                                             "parsed": ans["parsed"], "anchored": r.get("anchored", True)})
    for v in out.values():
        v.sort(key=lambda w: w["index"])
    return out


# ---------------------------------------------------------------- CLI


def cut_all(items: list[dict], form: str, tok: sw.Tokenizer, cfg: dict) -> dict[str, list[sw.Window]]:
    if form == "chunk-400":
        size = cfg["minicheck"]["chunk_size"]
        return {it["item"]: sw.chunks(it["evidence"], tok, size) for it in items}
    seq = 512 if form == "window-512" else 2048
    out = {}
    for it in items:
        ws = sw.windows(it["evidence"], tok, it["claim"], seq, cfg["overlap"], cfg["margin"])
        sw.fits(ws, tok, it["claim"], seq)
        out[it["item"]] = ws
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tokenizer", action="append", default=[], metavar="ID=PATH",
                    help="nli-deberta-v3-base, bge-m3-zeroshot-v2.0-c and minicheck tokenizer.json files")
    ap.add_argument("--prefill-tps", type=float, default=800.0, help="assumed prompt tokens per second")
    ap.add_argument("--decode-tps", type=float, default=40.0, help="assumed output tokens per second")
    ap.add_argument("--fallback-hours", type=float, default=3.0)
    ap.add_argument("--reestimate", action="store_true",
                    help="replace the estimates in an exported --out with measured --seconds; requests.jsonl is "
                         "never rewritten")
    ap.add_argument("--seconds", action="append", default=[], metavar="FORM=SECONDS",
                    help="measured seconds per request, per form (with --reestimate)")
    ap.add_argument("--measured-on", help="the date the --seconds were measured (with --reestimate)")
    a = ap.parse_args(argv)
    if a.reestimate:
        seconds = {f: float(s) for f, s in (x.split("=", 1) for x in a.seconds)}
        missing = set(FORMS) - set(seconds)
        if missing or not a.measured_on:
            raise SystemExit(f"--reestimate needs --seconds for {sorted(missing)} and --measured-on")
        for name, est in reestimate(a.out, seconds, a.measured_on).items():
            print(f"{name}: {est['total']['requests']} requests, ~{est['total']['hours']} h (measured {a.measured_on})")
            for form, e in est["by_form"].items():
                print(f"  {form}: {e['requests']} × {e['seconds_per_request']} s = {e['hours']} h")
        return 0
    cfg = scoring_set.config()
    toks = dict(t.split("=", 1) for t in a.tokenizer)
    pins = {m["id"]: m["sha256_tokenizer"] for m in cfg["nli"]}
    pins["minicheck"] = cfg["minicheck"]["sha256_tokenizer"]
    form_tok = {"window-512": "nli-deberta-v3-base", "window-2048": "bge-m3-zeroshot-v2.0-c", "chunk-400": "minicheck"}
    triples, _claims, lock = scoring_set.load(cfg["set"])
    controls = scoring_controls.build()
    items = scoring_set.items(triples, controls)
    prompt = scoring_judge.load()
    cuts, tok_shas = {}, {}
    for form, tid in form_tok.items():
        tok = sw.HFTokenizer(Path(toks[tid]), pins[tid])
        tok_shas[form] = tok.sha256
        cuts[form] = cut_all(items, form, tok, cfg)
    base = {"schema": "sidekick-verification-judge-batch/1", "set": cfg["set"], "set_sha256": lock["set_sha256"],
            "controls_sha256": canon.sha256_file(scoring_controls.CONTROLS),
            "prompt_sha256": prompt["sha256"], "tokenizer_sha256": tok_shas, "output_mode": OUTPUT_MODE,
            "max_tokens": cfg["judge"]["max_tokens"], "reasoning_effort": cfg["judge"]["reasoning_effort"],
            "chunk_note": "chunk-400 approximates MiniCheck's sentence chunks at line granularity"}
    full = build(items, cuts, prompt, cfg["judge"])
    fb = build(items, cuts, prompt, cfg["judge"],
               keep=lambda it, form: form == "full" or scoring_set.in_fallback(it))
    for name, reqs, scope in (("full", full, "every window, every item"),
                              ("fallback", fb, "window and chunk forms: the 1k and 4k items, the obstacle set "
                                               "and the controls; the full form: every item")):
        est = estimate(reqs, cfg["judge"], a.prefill_tps, a.decode_tps)
        write_batch(a.out / name, reqs, {**base, "batch": name, "scope": scope, "estimate": est})
        print(f"{name}: {est['total']['requests']} requests, {est['total']['prompt_tokens']:,} prompt tokens, "
              f"~{est['total']['hours']} h")
        for form, e in est["by_form"].items():
            print(f"  {form}: {e['requests']} requests, {e['prompt_tokens']:,} tokens, ~{e['hours']} h")
    sha256sums(a.out)
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
