"""What a companion model costs the model it runs beside.

sidekick can serve a small model (a classifier) next to a larger one that
the GPU serves, a local LLM say. Served on the GPU, the companion competes
with the larger model for it; served on the ANE (a chunked artifact, D37,
the operator choosing its compute units, D38), it leaves the GPU alone but
answers more slowly. This measures the trade on one machine.

The primary is any OpenAI-compatible server that streams chat completions
(oMLX, mlx-lm's server, llama.cpp's server). Each measurement streams one
completion of a fixed prompt at temperature 0 and a fixed max_tokens, and
records the time to the first token (prefill) and the decode rate after it
(tokens after the first over the time after the first), separately: they
contend for the GPU differently. Speculative decoding makes the decode rate
depend on how many drafted tokens are accepted, so the prompt never
changes, each phase repeats it (`--runs`), and any usage field the server
reports beyond the standard ones (an acceptance rate, draft counts) is
recorded as given.

The companion is served by sidekickd itself, started by this script for
each phase with a config whose `[models."<id>"]` sets its compute units
(D38), so one install serves every phase. It sends classify requests of one
length at a fixed rate (each on schedule, or as soon as the one before ends
when it runs late) and records their latency and the bucket that answered.

The companion's load runs for the whole of each phase it is in. Every
response's HTTP status is recorded (anything but 200 counts as an error),
and so are its probabilities: the request never changes, so within a phase
they should never change either, and between the GPU and ANE phases they
differ only by the two paths' arithmetic, which the report states.

Phases, in order, each after `--cooldown` seconds idle, with the thermal
state (`pmset -g therm`) recorded before it:
1. the companion alone at the first rate on `cpu_and_gpu`, then on
   `cpu_and_ne`, with the primary loaded but idle (`--companion-seconds`):
   its own latency with the GPU free;
2. the primary alone;
3. for each `--rates` value, the primary with the companion on
   `cpu_and_gpu`, then on `cpu_and_ne`: what the companion costs the
   primary, and whether a companion off the GPU stays fast while the
   primary keeps the GPU busy;
4. the primary alone again, to show drift.
Before and after each phase it records memory pressure and swap: a phase
during which swap grew is marked invalid. Every phase's start and end are
logged with their time zone, so a power trace taken across the whole run
can be split by phase (`--power`).

Usage:
    python tools/companion_bench.py --primary-model <id> --sidekickd target/release/sidekickd \\
        --models-dir <dir> --model agent-jev [--primary-url http://127.0.0.1:2345] [--max-tokens 256]
        [--runs 5] [--rates 1,2] [--tokens 1024] [--port 8791] [--out companion_report.json]

    python tools/companion_bench.py --report companion_report.json --power power.txt

Power, in a second terminal, started before the run and stopped (Ctrl-C)
after it:

    sudo powermetrics --samplers cpu_power,gpu_power,ane_power -i 1000 -o power.txt

Nothing else heavy should run meanwhile: the primary's own weights and KV
cache fill most of the machine's memory.
"""

import argparse
import atexit
import datetime
import json
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

PROMPT = ("Write a detailed, step-by-step explanation of how a hash map handles collisions with open "
          "addressing and with separate chaining, then compare their cache behavior and their worst cases. "
          "Use plain prose, no code.")
STANDARD_USAGE = {"prompt_tokens", "completion_tokens", "total_tokens"}


def now():
    return datetime.datetime.now().astimezone()


def post(url, body, timeout=1800):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"content-type": "application/json"})
    return urllib.request.urlopen(req, timeout=timeout)


def primary_run(args):
    """One streamed completion: (ttft s, decode tok/s, completion tokens, extra usage)."""
    body = {"model": args.primary_model, "messages": [{"role": "user", "content": PROMPT}],
            "temperature": 0, "max_tokens": args.max_tokens, "stream": True,
            "stream_options": {"include_usage": True}}
    t0 = time.perf_counter()
    first = last = None
    pieces, usage, extra = 0, None, {}
    with post(f"{args.primary_url}/v1/chat/completions", body) as r:
        for raw in r:
            line = raw.decode(errors="replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            for choice in chunk.get("choices") or []:
                delta = choice.get("delta") or {}
                if delta.get("content") or delta.get("reasoning_content") or delta.get("reasoning"):
                    t = time.perf_counter()
                    first = first or t
                    last = t
                    pieces += 1
            if chunk.get("usage"):
                usage = chunk["usage"]
            for k, v in chunk.items():
                if k not in ("id", "object", "created", "model", "choices", "usage", "system_fingerprint"):
                    extra[k] = v
    if first is None:
        raise SystemExit("the primary streamed no tokens")
    tokens = (usage or {}).get("completion_tokens") or pieces
    extra.update({k: v for k, v in (usage or {}).items() if k not in STANDARD_USAGE})
    decode = (tokens - 1) / (last - first) if last > first and tokens > 1 else float("nan")
    return first - t0, decode, tokens, extra


def memory():
    pressure = subprocess.run(["memory_pressure"], capture_output=True, text=True).stdout
    free = re.search(r"free percentage: (\d+)%", pressure)
    swap = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout
    used = re.search(r"used = ([0-9.]+)M", swap)
    return {"free_percent": int(free.group(1)) if free else None, "swap_used_mb": float(used.group(1)) if used else None}


class Companion:
    """sidekickd serving the companion on `units`, and a fixed-rate load."""

    def __init__(self, args, units):
        self.args, self.units = args, units
        self.home = tempfile.mkdtemp(prefix="companion-bench-")
        config = Path(self.home) / "config.toml"
        config.write_text(f'addr = "127.0.0.1:{args.port}"\nmodels_dir = "{args.models_dir}"\n'
                          # the first ANE load compiles each chunk, minutes for a large bucket
                          f"model_idle_ttl_secs = 86400\nrequest_timeout_secs = 1800\n\n"
                          f"[models.\"{args.model}\"]\ncompute_units = \"{units}\"\n")
        self.log = open(Path(self.home) / "sidekickd.log", "w")
        self.proc = subprocess.Popen([str(args.sidekickd), "--config", str(config)], stdout=self.log,
                                     stderr=subprocess.STDOUT)
        atexit.register(self.close)
        try:
            self._ready()
        except BaseException:
            self.close()
            raise

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def _ready(self):
        args, units = self.args, self.units
        self.url = f"http://127.0.0.1:{args.port}"
        for _ in range(120):
            try:
                with urllib.request.urlopen(f"{self.url}/health", timeout=2) as r:
                    health = json.load(r)
                break
            except (urllib.error.URLError, ConnectionError):
                time.sleep(0.5)
        else:
            raise SystemExit(f"sidekickd didn't come up; see {self.log.name}")
        applied = health.get("compute_unit_overrides", {}).get("applied", {})
        if applied.get(args.model) != units:
            raise SystemExit(f"sidekickd isn't serving {args.model} on {units}: {health.get('compute_unit_overrides')}"
                             f"; skipped: {health.get('skipped_models')}")
        self.body = self.calibrate()

    def request(self, body):
        """(latency s, HTTP status, bucket, compute units, probabilities)."""
        t = time.perf_counter()
        try:
            with post(f"{self.url}/v1/classify", body) as r:
                data = json.load(r)
                return (time.perf_counter() - t, r.status, r.headers.get("sidekick-buckets"),
                        r.headers.get("sidekick-compute-units"), data["data"][0]["probs"])
        except urllib.error.HTTPError as e:
            return time.perf_counter() - t, e.code, None, None, None

    def calibrate(self):
        """A request whose tree fills the bucket of `--tokens`, found by
        growing the state until sidekick-buckets names that bucket. The
        first requests also load the model (on the ANE, compiling it)."""
        words = int(self.args.tokens * 0.6)
        for _ in range(12):
            body = {"model": self.args.model, "input": "[STATE] " + " ".join(["status"] * words),
                    "candidate_labels": ["done: the task is finished", "test: run the test suite"],
                    "question_type": "choice", "instructions": "Which candidate action is useful?"}
            _, status, bucket, units, _ = self.request(body)
            if status != 200 or bucket is None:
                raise SystemExit(f"the companion answered HTTP {status}, bucket header {bucket!r}; see "
                                 f"{self.log.name}")
            if int(bucket.split(",")[0]) >= self.args.tokens:
                self.bucket, self.served = int(bucket.split(",")[0]), units
                for _ in range(3):
                    self.request(body)
                return body
            words = int(words * 1.12)
        raise SystemExit(f"couldn't fill the {self.args.tokens} bucket")

    def start(self, rate):
        self.latencies, self.errors, self.statuses, self.probs = [], 0, {}, []
        self.sent, self.stop = 0, threading.Event()

        def load():
            start = time.perf_counter()
            while not self.stop.is_set():
                due = start + self.sent / rate
                wait = due - time.perf_counter()
                if wait > 0 and self.stop.wait(wait):
                    break
                self.sent += 1
                try:
                    latency, status, _, _, probs = self.request(self.body)
                except (urllib.error.URLError, ConnectionError):
                    status, probs = "connection", None
                self.statuses[str(status)] = self.statuses.get(str(status), 0) + 1
                if status == 200:
                    self.latencies.append(latency)
                    self.probs.append(probs)
                else:
                    self.errors += 1
            self.wall = time.perf_counter() - start

        self.thread = threading.Thread(target=load, daemon=True)
        self.thread.start()

    def finish(self):
        self.stop.set()
        self.thread.join()
        ms = sorted(x * 1e3 for x in self.latencies)
        at = lambda q: ms[min(int(len(ms) * q), len(ms) - 1)] if ms else None  # noqa: E731
        spread = (max(max(abs(a - b) for a, b in zip(p, self.probs[0])) for p in self.probs)
                  if self.probs else None)
        return {"units": self.units, "served_on": self.served, "bucket": self.bucket, "requests": self.sent,
                "ok": len(ms), "rate": len(ms) / self.wall if self.wall else None, "errors": self.errors,
                "statuses": self.statuses, "p50_ms": at(0.5), "p90_ms": at(0.9), "p99_ms": at(0.99),
                "probs": self.probs[0] if self.probs else None, "probs_spread_in_phase": spread}

    def close(self):
        """Stop sidekickd and remove its config and log (once the log's last
        lines are printed if it failed). Safe to call more than once."""
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
        if not self.log.closed:
            self.log.close()
            if self.proc.returncode not in (0, -15):
                print(Path(self.log.name).read_text(errors="replace")[-2000:], file=sys.stderr)
        shutil.rmtree(self.home, ignore_errors=True)


def thermal():
    return subprocess.run(["pmset", "-g", "therm"], capture_output=True, text=True).stdout.strip()


def phase(args, name, units=None, rate=None, primary=True):
    """One phase: the primary's runs (unless `primary` is False, when the
    companion runs alone for --companion-seconds), with the companion's
    load, when there is one, from before the first run to after the last."""
    time.sleep(args.cooldown)
    therm = thermal()
    companion = Companion(args, units) if units else None
    try:
        before = memory()
        if companion:
            companion.start(rate)
        start = now()
        print(f"phase {name}: start {start.isoformat(timespec='seconds')}", flush=True)
        if primary:
            runs = [primary_run(args) for _ in range(args.runs)]
        else:
            runs = []
            time.sleep(args.companion_seconds)
        end = now()
        c = companion.finish() if companion else None
    finally:
        if companion:
            companion.close()
    after = memory()
    result = {"phase": name, "units": units, "rate": rate, "start": start.isoformat(), "end": end.isoformat(),
              "thermal_before": therm, "memory_before": before, "memory_after": after,
              "valid": None not in (before["swap_used_mb"], after["swap_used_mb"])
              and after["swap_used_mb"] - before["swap_used_mb"] < 64}
    if runs:
        result.update({"ttft_s": [r[0] for r in runs], "decode_tok_s": [r[1] for r in runs],
                       "completion_tokens": [r[2] for r in runs], "server_extra": [r[3] for r in runs],
                       "ttft_s_median": statistics.median(r[0] for r in runs),
                       "decode_tok_s_median": statistics.median(r[1] for r in runs)})
    if companion:
        result["companion"] = c
    line = []
    if runs:
        d = result["decode_tok_s"]
        line.append(f"decode {result['decode_tok_s_median']:.2f} tok/s (min {min(d):.2f}, max {max(d):.2f}), "
                    f"TTFT {result['ttft_s_median']:.2f} s")
    if companion:
        fmt = lambda v, spec: "-" if v is None else format(v, spec)  # noqa: E731
        line.append(f"companion {c['ok']}/{c['requests']} ok, {fmt(c['rate'], '.2f')}/s on {c['served_on']}, "
                    f"bucket {c['bucket']}, p50 {fmt(c['p50_ms'], '.0f')} ms, p99 {fmt(c['p99_ms'], '.0f')} ms, "
                    f"statuses {c['statuses']}, probs spread in phase {c['probs_spread_in_phase']}")
    print(f"phase {name}: " + "; ".join(line) + ("" if result["valid"] else "; INVALID: swap grew"), flush=True)
    return result


def compare_scores(phases):
    """Max |dp| of the companion's answer between each pair of phases that
    ran it: the GPU and ANE paths' arithmetic, measured under load."""
    ran = [(p["phase"], p["companion"]["probs"]) for p in phases if p.get("companion", {}).get("probs")]
    return {f"{a} vs {b}": max(abs(x - y) for x, y in zip(pa, pb))
            for i, (a, pa) in enumerate(ran) for b, pb in ran[i + 1:]}


_POWER = re.compile(r"^(CPU|GPU|ANE) Power: (\d+) mW", re.M)


def power_by_phase(path, phases):
    """Mean CPU, GPU and ANE power (mW) per phase from a powermetrics log."""
    samples = []
    for block in Path(path).read_text(errors="replace").split("*** Sampled system activity")[1:]:
        when = re.match(r" \((.+?)\)", block)
        if not when:
            continue
        stamp = datetime.datetime.strptime(when.group(1).strip(), "%a %b %d %H:%M:%S %Y %z")
        samples.append((stamp, {k: int(v) for k, v in _POWER.findall(block)}))
    for p in phases:
        start, end = datetime.datetime.fromisoformat(p["start"]), datetime.datetime.fromisoformat(p["end"])
        inside = [s for t, s in samples if start <= t <= end]
        p["power_mw"] = ({k: statistics.mean(s.get(k, 0) for s in inside) for k in ("CPU", "GPU", "ANE")}
                         if inside else None)
        p["power_samples"] = len(inside)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--primary-url", default="http://127.0.0.1:2345")
    ap.add_argument("--primary-model")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--sidekickd", type=Path)
    ap.add_argument("--models-dir", type=Path)
    ap.add_argument("--model")
    ap.add_argument("--rates", default="1,2", help="companion requests per second, comma-separated")
    ap.add_argument("--tokens", type=int, default=1024, help="the companion's request length (a bucket)")
    ap.add_argument("--port", type=int, default=8791)
    ap.add_argument("--cooldown", type=float, default=30.0, help="seconds idle before each phase")
    ap.add_argument("--companion-seconds", type=float, default=60.0,
                    help="length of the companion-alone phases")
    ap.add_argument("--out", type=Path, default=Path("companion_report.json"))
    ap.add_argument("--report", type=Path, help="add --power to an existing report instead of measuring")
    ap.add_argument("--power", type=Path)
    args = ap.parse_args()
    if args.report:
        report = json.loads(args.report.read_text())
    else:
        for need in ("primary_model", "sidekickd", "models_dir", "model"):
            if getattr(args, need) is None:
                ap.error(f"--{need.replace('_', '-')} is required")
        args.models_dir = args.models_dir.resolve()
        rates = [float(r) for r in args.rates.split(",")]
        phases = [phase(args, f"companion-gpu@{rates[0]:g}", "cpu_and_gpu", rates[0], primary=False),
                  phase(args, f"companion-ane@{rates[0]:g}", "cpu_and_ne", rates[0], primary=False)]
        print("warming the primary up", flush=True)
        primary_run(args)
        phases.append(phase(args, "alone"))
        for rate in rates:
            phases.append(phase(args, f"gpu@{rate:g}", "cpu_and_gpu", rate))
            phases.append(phase(args, f"ane@{rate:g}", "cpu_and_ne", rate))
        phases.append(phase(args, "alone-again"))
        report = {"primary": {"url": args.primary_url, "model": args.primary_model, "max_tokens": args.max_tokens,
                              "prompt": PROMPT, "runs": args.runs},
                  "companion": {"model": args.model, "tokens": args.tokens}, "phases": phases,
                  "companion_probs_max_dp": compare_scores(phases)}
        for pair, dp in report["companion_probs_max_dp"].items():
            print(f"companion probabilities, {pair}: max |dp| {dp:.2e}")
    if args.power:
        power_by_phase(args.power, report["phases"])
        for p in report["phases"]:
            w = p.get("power_mw")
            print(f"phase {p['phase']}: " + (", ".join(f"{k} {v / 1000:.2f} W" for k, v in w.items()) if w
                                             else "no power samples in this phase"))
    out = args.report or args.out
    out.write_text(json.dumps(report, indent=1) + "\n")
    print(f"report -> {out}")


if __name__ == "__main__":
    sys.exit(main())
