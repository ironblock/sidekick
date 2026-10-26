"""chat_slice.py job <bench_dir> <hf_src> <out_job.json> [--per 12]
   chat_slice.py score <bench_dir> <hf_src> <job.json> <skrun_out.json> <gpu_run.json> <out.json>

Task quality of an ANE build on a slice of the chat benchmark (#174's task
set, fixtures/tasks/chat/ and tools/chat_bench/ on the chat-benchmark
branch, extracted to <bench_dir>) against that benchmark's GPU bf16 run of
the same model (docs/data/chat-bench/runs/<row>.json).

job: the first --per requests of each family, chat-templated as the GPU row
was (enable_thinking false), as skrun greedy tasks of max_tokens tokens.
score: cut each generation at the first EOS, then score the ANE replies and
the GPU replies to the same requests with the benchmark's own scorers
(titles, which need an embedding server, get exact match only), and report
how often the two are identical."""
import argparse
import json
import sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("mode", choices=["job", "score"])
ap.add_argument("bench")
ap.add_argument("src")
ap.add_argument("rest", nargs="*")
ap.add_argument("--per", type=int, default=12)
a = ap.parse_args()
sys.path.insert(0, str(Path(a.bench) / "tools"))
from chat_bench import tasks  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

tok = AutoTokenizer.from_pretrained(a.src)
reqs = tasks.requests(Path(a.bench) / "fixtures" / "tasks" / "chat")
seen, sel = {}, []
for r in reqs:
    if seen.get(r["family"], 0) < a.per:
        seen[r["family"]] = seen.get(r["family"], 0) + 1
        sel.append(r)

if a.mode == "job":
    out, = a.rest
    jt = []
    for r in sel:
        ids = tok.apply_chat_template(r["messages"], add_generation_prompt=True, tokenize=True, enable_thinking=False)
        jt.append({"type": "greedy", "name": f"{r['family']}/{r['item']}", "ids": list(ids), "n": r["max_tokens"]})
    json.dump({"tasks": jt}, open(out, "w"))
    print(len(jt), "requests;", sum(t["n"] for t in jt), "tokens at most")
    sys.exit(0)

job_p, ane_p, gpu_p, out = a.rest
from chat_bench import score as S  # noqa: E402

eos = {1, 130073}
ane = {}
for t in json.load(open(ane_p))["tasks"]:
    if t["type"] != "greedy":
        continue
    toks = t["tokens"]
    cut = next((i for i, x in enumerate(toks) if x in eos), None)
    fam, item = t["name"].split("/", 1)
    ane[(fam, item)] = {"family": fam, "item": item, "status": 200, "reasoning": "",
                        "content": tok.decode(toks[:cut] if cut is not None else toks, skip_special_tokens=True),
                        "finish_reason": "stop" if cut is not None else "length"}
gpu_run = json.load(open(gpu_p))
gpu = {(r["family"], r["item"]): r for r in gpu_run["records"]}
keys = [k for k in ane if k in gpu]
cfg = tasks.load_config(Path(a.bench) / "fixtures" / "tasks" / "chat")
items = tasks.load_items(Path(a.bench) / "fixtures" / "tasks" / "chat")
schemas = tasks.load_schemas(Path(a.bench) / "fixtures" / "tasks" / "chat")
res = {}
for fam in sorted({k[0] for k in keys}):
    fk = [k for k in keys if k[0] == fam]
    same = sum(S.final_text(ane[k]["content"]).strip() == S.final_text(gpu[k]["content"]).strip() for k in fk)
    rec = {"n": len(fk), "identical": same}
    if fam != "titles":
        ids = {k[1] for k in fk}
        its = {f: [it for it in v if it["id"] in ids] for f, v in items.items()}
        for name, src in (("ane", ane), ("gpu", gpu)):
            recs = {k[1]: src[k] for k in fk}
            if fam == "tags_vocab":
                fm, _ = S.score_tags_vocab(recs, its["tags"], cfg)
            elif fam == "tags_free":
                fm, _ = S.score_tags_free(recs, its["tags"], cfg)
            elif fam == "extraction":
                fm, _ = S.score_extraction(recs, its["extraction"], cfg, schemas)
            else:
                fm, _ = S.score_instructions(recs, its["instructions"])
            head = S.HEADLINE[fam]
            rec[name] = {head: fm[head]["value"] if isinstance(fm[head], dict) else fm[head]}
    res[fam] = rec
    print("RESULT", fam, json.dumps(rec))
json.dump(res, open(out, "w"), indent=1)
