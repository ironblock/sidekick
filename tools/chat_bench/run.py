"""Run one row of the chat benchmark: send every request of the task set
(fixtures/tasks/chat/) to one model through POST /v1/chat/completions and
record each reply with its timing. Maintainer-run, one model at a time; never
in CI. Run it under tools/chat_bench/watchdog.py.

    # Foundation Models, through a sidekickd you started (any free port)
    python tools/chat_bench/run.py afm --base-url http://127.0.0.1:8805 --out <runs dir>
    python tools/chat_bench/run.py afm-prompt --base-url http://127.0.0.1:8805 --out <runs dir>

    # An open model: run.py starts its server on --port, runs, and stops it
    python tools/chat_bench/run.py minicpm5-1b --port 8811 --out <runs dir>

Rows are defined in tools/chat_bench/rows.toml. `--limit N` runs the first N
requests of each family, as a check of the request shapes.

Every request streams (`stream: true`, with usage), at temperature 0, with
a per-request timeout; a timeout or an HTTP error other than a model's own
refusal ends the run rather than retrying. Each reply records the time to
the first streamed token (any text, thinking included) and to the first
content token, the total time, the output tokens the server reports, the
finish reason, and the raw content (and reasoning, for a thinking model).
Foundation Models streams schema-constrained replies whole, so its guided
JSON rows' first-token time equals their total time.

A run file records the task set's lock, the row, the request shape, the
model's revision and the serving stack, the sidekick commit and version, the
Foundation Models variant /health reports, and the macOS build and chip. The
base URL is never recorded.
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
from chat_bench import tasks  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
RUN_SCHEMA = "sidekick-chat-bench-run/1"


def load_rows() -> dict[str, dict]:
    with open(HERE / "rows.toml", "rb") as f:
        return {r["id"]: r for r in tomllib.load(f)["row"]}


def _cmd(args: list[str]) -> str | None:
    try:
        return subprocess.run(args, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def machine() -> dict:
    return {"macos": _cmd(["sw_vers", "-productVersion"]), "macos_build": _cmd(["sw_vers", "-buildVersion"]),
            "chip": _cmd(["sysctl", "-n", "machdep.cpu.brand_string"]),
            "memory_gb": round(int(_cmd(["sysctl", "-n", "hw.memsize"]) or 0) / 2**30),
            "arch": platform.machine()}


def get_json(base: str, path: str, timeout: float = 10) -> tuple[int, dict, dict | None]:
    try:
        with urllib.request.urlopen(base + path, timeout=timeout) as r:
            return r.status, dict(r.headers), json.loads(r.read() or b"null")
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers), None
    except (urllib.error.URLError, OSError):
        return 0, {}, None


def stream_chat(base: str, body: dict, timeout: float) -> dict:
    """POST a streaming chat request; return the reply and its timing."""
    data = json.dumps(body).encode()
    req = urllib.request.Request(base + "/v1/chat/completions", data=data,
                                 headers={"content-type": "application/json"})
    t0 = time.perf_counter()
    rec = {"status": None, "content": "", "reasoning": "", "finish_reason": None, "usage": None,
           "ttft_s": None, "ttft_content_s": None, "total_s": None, "error": None, "constrained": None,
           "headers": {}}
    try:
        resp = urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.HTTPError as e:
        rec["status"] = e.code
        rec["total_s"] = time.perf_counter() - t0
        try:
            rec["error"] = json.loads(e.read()).get("error")
        except (ValueError, AttributeError):
            rec["error"] = {"message": f"HTTP {e.code}"}
        return rec
    rec["status"] = resp.status
    rec["headers"] = {k.lower(): v for k, v in resp.headers.items() if k.lower().startswith("sidekick-")}
    with resp:
        for raw in resp:
            line = raw.decode("utf-8").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            ev = json.loads(payload)
            if "error" in ev and "choices" not in ev:
                rec["error"] = ev["error"]
                continue
            if ev.get("usage"):
                rec["usage"] = ev["usage"]
            if ev.get("constrained") is not None:
                rec["constrained"] = ev["constrained"]
            for ch in ev.get("choices") or []:
                d = ch.get("delta") or {}
                now = time.perf_counter() - t0
                text = d.get("content") or ""
                think = d.get("reasoning") or d.get("reasoning_content") or ""
                if (text or think) and rec["ttft_s"] is None:
                    rec["ttft_s"] = now
                if text and rec["ttft_content_s"] is None:
                    rec["ttft_content_s"] = now
                rec["content"] += text
                rec["reasoning"] += think
                if ch.get("finish_reason"):
                    rec["finish_reason"] = ch["finish_reason"]
    rec["total_s"] = time.perf_counter() - t0
    return rec


def wait_ready(base: str, proc: subprocess.Popen, timeout: float) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            raise SystemExit(f"model server exited with {proc.returncode} before it was ready")
        st, _, _ = get_json(base, "/health", timeout=2)
        if st == 200:
            return
        # mlx_lm.server has no /health on every version; /v1/models answers too
        st, _, _ = get_json(base, "/v1/models", timeout=2)
        if st == 200:
            return
        time.sleep(1)
    raise SystemExit(f"model server not ready after {timeout:.0f} s")


def hf_dir(repo: str, revision: str) -> Path:
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(repo, revision=revision))


def start_server(row: dict, port: int, log: Path) -> tuple[subprocess.Popen, dict]:
    """Start the row's model server in this process group (so the watchdog's
    kill reaches it)."""
    py = sys.executable
    model_dir = hf_dir(row["repo"], row["revision"])
    ct = json.dumps(row.get("chat_template_args", {}))
    if row["server"] == "mlx_lm":
        import mlx.core as mx
        import mlx_lm
        cmd = [py, "-m", "mlx_lm.server", "--model", str(model_dir), "--host", "127.0.0.1",
               "--port", str(port), "--temp", "0", "--chat-template-args", ct]
        stack = {"server": "mlx_lm.server", "mlx_lm": mlx_lm.__version__, "mlx": mx.__version__,
                 "device": "gpu", "dtype": "bf16 (repository weights, unquantised)"}
    else:
        cmd = [py, str(HERE / "serve_ref.py"), "--backend", row["backend"], "--model", str(model_dir),
               "--port", str(port), "--chat-template-args", ct]
        if row.get("think_tags"):
            cmd += ["--think-tags", *row["think_tags"]]
        if row.get("reasoning_budget") is not None:
            cmd += ["--reasoning-budget", str(row["reasoning_budget"])]
        if row.get("prefill"):
            cmd += ["--prefill", row["prefill"]]
        if row.get("constrained"):
            cmd += ["--constrained"]
        stack = {"server": "tools/chat_bench/serve_ref.py", "backend": row["backend"], "greedy": "argmax"}
        if row["backend"] == "torch":
            import torch
            import transformers
            stack |= {"torch": torch.__version__, "transformers": transformers.__version__,
                      "device": "mps", "dtype": "bf16"}
        else:
            import mlx_lm
            stack |= {"mlx_lm": mlx_lm.__version__, "mlx": __import__("mlx.core", fromlist=["x"]).__version__,
                      "device": "gpu", "dtype": "bf16 (repository weights, unquantised)"}
        if row.get("constrained"):
            import llguidance  # noqa: F401
            stack["llguidance"] = _cmd([py, "-c", "import importlib.metadata as m; print(m.version('llguidance'))"])
    logf = open(log, "a")
    proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT,
                            env=dict(os.environ, PYTHONUNBUFFERED="1", HF_HUB_OFFLINE="1"))
    return proc, stack


def sidekick_provenance(base: str) -> dict:
    st, hdr, health = get_json(base, "/health")
    if st != 200 or not health:
        raise SystemExit(f"sidekickd /health answered {st}")
    chat = health.get("chat", {})
    return {"model": chat.get("model"), "availability": chat.get("availability"),
            "variant": chat.get("variant"), "variant_id": chat.get("variant_id"),
            "context_limit": chat.get("context_limit"),
            "sidekick_version": hdr.get("sidekick-version") or health.get("version"),
            "sidekick_commit": _cmd(["git", "-C", str(REPO), "rev-parse", "HEAD"])}


def build_body(row: dict, r: dict, model: str, temperature: float) -> dict:
    body = {"model": model, "messages": r["messages"], "temperature": temperature,
            "max_tokens": r["max_tokens"] + row.get("extra_max_tokens", 0),
            "stream": True, "stream_options": {"include_usage": True}}
    if r["schema"] is not None and row["json"] == "guided":
        schema = r["schema"]
        if row.get("schema_all_required"):
            # A diagnostic: the same schema with every property required, to
            # separate what optional properties cost from what guidance does.
            schema = {**schema, "required": list(schema["properties"])}
        body["response_format"] = {"type": "json_schema",
                                   "json_schema": {"name": r["family"], "schema": schema}}
    return body


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("row")
    ap.add_argument("--out", required=True, help="directory for the run file")
    ap.add_argument("--base-url", help="a running sidekickd (sidekick rows)")
    ap.add_argument("--port", type=int, default=8811, help="port for the row's own model server")
    ap.add_argument("--timeout", type=float, default=180, help="seconds per request")
    ap.add_argument("--ready-timeout", type=float, default=600)
    ap.add_argument("--limit", type=int, help="first N requests of each family")
    ap.add_argument("--suffix", default="", help="appended to the run file's name (a repeat run)")
    a = ap.parse_args()

    row = load_rows()[a.row]
    cfg = tasks.load_config()
    reqs = [r for r in tasks.requests() if r["family"] in row.get("families", tasks.FAMILIES)]
    if a.limit:
        reqs = [r for f in tasks.FAMILIES for r in [q for q in reqs if q["family"] == f][:a.limit]]
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    proc = None
    if row["server"] == "sidekick":
        if not a.base_url:
            raise SystemExit("sidekick rows need --base-url")
        base = a.base_url.rstrip("/")
        prov = sidekick_provenance(base)
        model = prov["model"]
        stack = {"server": "sidekickd", "route": "/v1/chat/completions"}
    else:
        base = f"http://127.0.0.1:{a.port}"
        if get_json(base, "/health", timeout=2)[0] or get_json(base, "/v1/models", timeout=2)[0]:
            raise SystemExit(f"port {a.port} is already serving; one model at a time")
        proc, stack = start_server(row, a.port, out_dir / f"{a.row}.server.log")
        t_load = time.time()
        wait_ready(base, proc, a.ready_timeout)
        prov = {"repo": row["repo"], "revision": row["revision"], "load_s": round(time.time() - t_load, 1)}
        model = "default_model" if row["server"] == "mlx_lm" else row["repo"]

    records = []
    try:
        # One warm-up request, not recorded: first-call compilation and caches.
        warm = build_body(row, reqs[0], model, cfg["temperature"])
        w = stream_chat(base, warm, a.timeout)
        if w["status"] != 200:
            raise SystemExit(f"warm-up failed: {w['status']} {w['error']}")
        t_start = time.time()
        for i, r in enumerate(reqs):
            body = build_body(row, r, model, cfg["temperature"])
            try:
                rec = stream_chat(base, body, a.timeout)
            except (TimeoutError, urllib.error.URLError):
                raise SystemExit(f"request {r['family']}/{r['item']} timed out after {a.timeout:.0f} s; stopping")
            if row["server"] == "sidekick" and not prov.get("model_header"):
                prov["model_header"] = rec["headers"].get("sidekick-model")
                prov["sidekick_version"] = prov["sidekick_version"] or rec["headers"].get("sidekick-version")
            # A model's own refusal or limit (400 content_filter, unsupported
            # language, context) is a result; anything else ends the run.
            if rec["status"] != 200 and rec["status"] != 400:
                raise SystemExit(f"{r['family']}/{r['item']}: HTTP {rec['status']} {rec['error']}")
            rec.pop("headers")
            records.append({"family": r["family"], "item": r["item"], **rec})
            if (i + 1) % 20 == 0:
                print(f"[{a.row}] {i + 1}/{len(reqs)} {time.time() - t_start:.0f} s", flush=True)
    finally:
        if proc is not None:
            proc.terminate()
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                proc.kill()

    shape = build_body(row, reqs[0], model, cfg["temperature"]) | {"messages": "<item>"}
    run = {"schema": RUN_SCHEMA, "row": row, "task_set": tasks.task_set_lock(), "model": prov,
           "stack": stack, "machine": machine(), "request_shape": shape,
           "started": time.strftime("%Y-%m-%dT%H:%M:%S"), "timeout_s": a.timeout,
           "limit": a.limit, "records": records}
    path = out_dir / f"{a.row}{a.suffix}.json"
    path.write_text(json.dumps(run, ensure_ascii=False, indent=1) + "\n")
    print(f"wrote {path} ({len(records)} requests)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
