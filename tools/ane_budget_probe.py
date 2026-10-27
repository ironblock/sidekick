#!/usr/bin/env python3
"""ane_budget_probe.py <scenario.json> <out.json>: time a stateful ANE decoder's
steps while other ANE work is loaded and running, phase by phase.

What it measures. The decoder runs back-to-back generations through the
scenario; each phase adds or removes load: a second decoder process,
sidekickd serving ANE embedders (each sequence-length bucket is its own
program, loaded on first use, so a request of a given length adds one
bucket's weights to the ANE), or Foundation Models requests through
sidekickd's `apple-fm`. Per phase it reports the decoder's per-call latency
(and how many calls took over 2x and 5x the first phase's median), the other
work's request latency, each started process's ANE ledger and physical
footprint (proc_pid_rusage: `ri_neural_footprint`, `ri_phys_footprint`), and
the system's wired memory. A resident-weight budget shows as a knee: the
per-call time jumping by an order of magnitude when the active programs no
longer fit and reload on every call.

The decoder runner is sidekick-coreml's `kv_decoder` example, which loads a
`chat.toml` build and runs its `loop` task: back-to-back generations, every
call's time appended to a file as it happens. Build it once and sign it with
a fixed identifier:

    cargo build --release -p sidekick-coreml --example kv_decoder
    codesign -s - -f -i <identifier> target/release/examples/kv_decoder

Core ML keys a compiled ANE program by, among other things, the loading
executable's name and code-signing identifier, the home
(`CFFIXED_USER_HOME`, else `HOME`) and the program's path
(docs/design/coreml-cache.md, "The cache key"). Run the decoder under the
identifier, home and model path an earlier run compiled it with: otherwise
every program compiles again and leaves another copy of the weights in the
ANE's undeletable system cache. The linker's own identifier changes from
build to build, so re-sign after each one. The same holds for sidekickd:
give it the binary, home and model directories whose ANE programs are
already compiled, such as the release gate's (tools/release_gate.py). A
bucket no earlier run loaded may still compile; the result counts compiles.

Safety. A watchdog samples the system every second or so and kills every
process it started (by process group, never by name) when swap use grows
past `swap_max_gib` over its starting value, free memory falls under
`free_min_pct`, free disk falls under `disk_min_gib`, or the scenario passes
`max_seconds`. Every HTTP request has a timeout, and the decoder exits on
its own (status 4) at the first prediction past `predict_timeout`; either
stops the scenario (a hung ANE call is not retried), as does a decoder that
exits while it should be running. It counts the ANE compiler's finished
compiles in the unified log over the run.

Scenario (JSON):
  {
    "decoder": "<kv_decoder binary>", "decoder_model": "<chat.toml build dir>",
    "decoder_home": "<CFFIXED_USER_HOME the build was compiled under>",
    "prompt_ids": [..], "gen_tokens": 32,
    "sidekickd": "<sidekickd binary>", "sidekickd_home": "<its home>",
    "servers": {"lfm": {"models_dir": "<dir holding the model>"}},
    "phases": [
      {"name": "baseline", "seconds": 20},
      {"name": "second decoder", "start_decoder": "d2", "seconds": 20},
      {"name": "lfm 128", "stop_decoder": "d2", "start_server": "lfm",
       "touch": [["lfm", "lfm2.5-embedding-350m", 128]],
       "hammer": [["lfm", "lfm2.5-embedding-350m", [128]]], "seconds": 20},
      {"name": "lfm every 3 s", "hammer": [["lfm", "lfm2.5-embedding-350m", [128], 3.0]],
       "seconds": 20, "footprint": true},
      {"name": "afm", "afm": "lfm", "afm_warm": 1, "seconds": 30}
    ]
  }
- Decoder `d1` starts first unless "autostart" is false; others start and
  stop where a phase says ("start_decoder", "stop_decoder"). Each runs the
  same build. "decoder_home" is optional: without it a decoder inherits the
  environment's home.
- "servers" are sidekickd instances, one per entry, started and stopped by
  "start_server" / "stop_server". Serve each model from the directory its
  ANE programs were compiled at: a copy or a symlink is a new cache key.
- "touch": one embeddings request per [server, model, bucket] (a text sized
  to land in that bucket), loading that bucket's program.
- "hammer": back-to-back embeddings requests per [server, model, buckets]
  for the phase, cycling through the buckets; an optional fourth element
  is a pause between requests, in seconds.
- "afm": back-to-back `apple-fm` chat requests through that server
  ("afm_interval" pauses between them); "afm_warm": that many requests
  first, outside the timed window, each logged.
- "footprint": also save `footprint -p` for each process at the end of the
  window.
A phase's actions run before its timed window, timed as "<name> (setup)".
Optional: "swap_max_gib" (4), "free_min_pct" (15), "disk_min_gib" (25),
"max_seconds" (1800), "predict_timeout" (30), "request_timeout" (60),
"settle_seconds" (5), "load_timeout" (300).
"""
import ctypes
import json
import os
import re
import signal
import socket
import statistics
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

RESERVED_PORTS = range(8790, 8800)
SENTENCE = "The quick brown fox jumps over the lazy dog near the quiet river bank. "


def sh(*cmd, timeout=30):
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout).stdout


def swap_gib():
    m = re.search(r"used = ([\d.]+)M", sh("sysctl", "-n", "vm.swapusage"))
    return float(m.group(1)) / 1024 if m else 0.0


def free_pct():
    m = re.search(r"free percentage: (\d+)%", sh("memory_pressure", "-Q"))
    return float(m.group(1)) if m else 100.0


def wired_gib():
    out = sh("vm_stat")
    page = int(re.search(r"page size of (\d+)", out).group(1))
    return int(re.search(r"Pages wired down:\s+(\d+)", out).group(1)) * page / 2**30


def free_disk_gib(path="/"):
    st = os.statvfs(path)
    return st.f_bavail * st.f_frsize / 2**30


def rss_mib(pids):
    if not pids:
        return {}
    out = sh("ps", "-o", "pid=,rss=", "-p", ",".join(str(p) for p in pids))
    return {int(a): int(b) / 1024 for a, b in (l.split() for l in out.splitlines() if l.strip())}


class _RusageV6(ctypes.Structure):
    _fields_ = [("uuid", ctypes.c_uint8 * 16), ("f", ctypes.c_uint64 * 56)]


_libc = ctypes.CDLL("/usr/lib/libSystem.B.dylib", use_errno=True)
RI_PHYS_FOOTPRINT, RI_NEURAL_FOOTPRINT = 7, 44     # field indexes after ri_uuid (sys/resource.h)


def ledgers_mib(pid):
    """A process's physical footprint and its ANE ("neural") ledger, MiB, from
    proc_pid_rusage(RUSAGE_INFO_V6); None for a process we may not inspect."""
    r = _RusageV6()
    if _libc.proc_pid_rusage(int(pid), 6, ctypes.byref(r)) != 0:
        return None
    return {"phys": r.f[RI_PHYS_FOOTPRINT] / 2**20, "neural": r.f[RI_NEURAL_FOOTPRINT] / 2**20}


def parse_mib(pattern, text):
    """A `footprint` size ("2185 MB", "498 KB") matched by `pattern`, in MiB."""
    m = re.search(pattern, text)
    if not m:
        return None
    num, unit = m.group(1).split()
    return round(float(num) * {"B": 1 / 2**20, "KB": 1 / 1024, "MB": 1, "GB": 1024}[unit], 1)


def free_port():
    for _ in range(100):
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        if port not in RESERVED_PORTS:
            return port
    raise SystemExit("no free port")


def post(url, body, timeout):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t = time.monotonic()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = json.loads(r.read())
    return (time.monotonic() - t) * 1000, data


def text_for_bucket(bucket):
    """A text whose token count lands in `bucket` (about 0.7 of it) for an
    English subword tokenizer: about 15 tokens a sentence."""
    return SENTENCE * max(1, int(bucket * 0.7 / 15))


class Hang(Exception):
    pass


class Probe:
    def __init__(self, sc, out):
        self.sc, self.out = sc, Path(out)
        self.work = self.out.with_suffix("")
        self.work.mkdir(parents=True, exist_ok=True)
        self.procs = {}          # name -> Popen (each its own process group)
        self.ports = {}
        self.stop_flags = {}
        self.events = []
        self.samples = []
        self.requests = []       # (kind, name, t_end, ms, ok)
        self.abort = None
        self.lock = threading.Lock()
        self.t0 = time.time()
        self.swap0 = swap_gib()

    # ------------------------------------------------------------ processes
    def spawn(self, name, cmd, env, log):
        f = open(self.work / log, "w")
        p = subprocess.Popen(cmd, env=env, stdout=f, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                             start_new_session=True)
        self.procs[name] = p
        self.event(f"start {name} pid {p.pid}")
        return p

    def kill_all(self):
        for name, p in self.procs.items():
            if p.poll() is None:
                try:
                    os.killpg(p.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass

    def event(self, what):
        with self.lock:
            self.events.append((time.time(), what))
        print(f"[{time.time() - self.t0:7.1f}] {what}", flush=True)

    # ------------------------------------------------------------ decoders
    def start_decoder(self, name):
        sc = self.sc
        stop = self.work / f"{name}.stop"
        stop.unlink(missing_ok=True)
        job = self.work / f"{name}_job.json"
        job.write_text(json.dumps({"tasks": [{"type": "loop", "name": "loop", "ids": sc["prompt_ids"],
                                              "n": sc.get("gen_tokens", 32), "seconds": sc.get("max_seconds", 1800),
                                              "stop_file": str(stop)}]}))
        env = dict(os.environ)
        if "decoder_home" in sc:
            env["CFFIXED_USER_HOME"] = sc["decoder_home"]
        self.stop_flags[name] = stop
        t = time.time()
        self.spawn(name, [sc["decoder"], sc["decoder_model"], str(job), str(self.work / name),
                          "--call-timeout", str(sc.get("predict_timeout", 30))], env, f"{name}.log")
        log = self.work / f"{name}.log"
        while True:
            # The runner prints "first call" once the build is loaded and has run once.
            if "first call" in log.read_text(errors="replace"):
                break
            if self.procs[name].poll() is not None:
                raise RuntimeError(f"{name} exited while loading:\n{log.read_text(errors='replace')[-2000:]}")
            if self.abort:
                raise RuntimeError(self.abort)
            if time.time() - t > sc.get("load_timeout", 300):
                raise Hang(f"{name} didn't load within {sc.get('load_timeout', 300)} s")
            time.sleep(0.5)
        self.event(f"{name} loaded in {time.time() - t:.1f} s")

    def stop_decoder(self, name):
        self.stop_flags[name].touch()
        p = self.procs[name]
        try:
            p.wait(timeout=self.sc.get("predict_timeout", 30) + 30)
        except subprocess.TimeoutExpired:
            raise Hang(f"{name} didn't stop")
        self.event(f"stop {name} exit {p.returncode}")
        if p.returncode != 0:
            raise RuntimeError(f"{name} exited {p.returncode}: " + (self.work / f"{name}.log").read_text()[-1500:])

    # ------------------------------------------------------------ sidekickd
    def start_server(self, name):
        sc, cfg = self.sc, self.sc["servers"][name]
        port = free_port()
        run = self.work / name
        run.mkdir(exist_ok=True)
        lines = [f'addr = "127.0.0.1:{port}"', f"models_dir = {json.dumps(cfg['models_dir'])}",
                 "request_timeout_secs = 120", "load_timeout_secs = 600", "model_idle_ttl_secs = 3600",
                 "warm_on_start = false"]
        (run / "config.toml").write_text("\n".join(lines) + "\n")
        home = sc["sidekickd_home"]
        env = dict(os.environ, HOME=home, CFFIXED_USER_HOME=home, TMPDIR=str(Path(home) / "tmp"))
        env.pop("SIDEKICK_API_KEY", None)
        self.spawn(name, [sc["sidekickd"], "--config", str(run / "config.toml")], env, f"{name}.log")
        self.ports[name] = port
        t = time.time()
        while time.time() - t < 120:
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as r:
                    if r.status == 200:
                        self.event(f"{name} up on {port}")
                        return
            except OSError:
                pass
            if self.procs[name].poll() is not None:
                raise RuntimeError(f"{name} exited at startup")
            time.sleep(0.3)
        raise RuntimeError(f"{name} didn't answer /health")

    def stop_server(self, name):
        p = self.procs[name]
        p.terminate()
        try:
            p.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGKILL)
            p.wait()
        self.event(f"stop {name}")

    def embed(self, server, model, bucket):
        ms, data = post(f"http://127.0.0.1:{self.ports[server]}/v1/embeddings",
                        {"model": model, "input": text_for_bucket(bucket)}, self.sc.get("request_timeout", 60))
        return ms, data.get("usage", {}).get("prompt_tokens")

    def chat(self, server):
        ms, data = post(f"http://127.0.0.1:{self.ports[server]}/v1/chat/completions",
                        {"model": "apple-fm", "max_tokens": 48, "temperature": 0,
                         "messages": [{"role": "user", "content": "Name three uses of a paperclip, one line each."}]},
                        self.sc.get("request_timeout", 60) + 60)
        return ms, data.get("usage", {}).get("completion_tokens")

    def footprints(self, tag):
        """Each live process's footprint and its ANE ("neural") ledger, MiB."""
        out = {}
        for n, p in list(self.procs.items()):
            if p.poll() is None:
                text = sh("footprint", "-p", str(p.pid), timeout=60)
                (self.work / f"footprint_{n}_{tag}.txt").write_text(text)
                out[n] = {"footprint": parse_mib(r"Footprint:\s+([\d.]+\s+[KMG]?B)", text),
                          "neural": parse_mib(r"neural:\s+([\d.]+\s+[KMG]?B)", text) or 0.0,
                          "iosurface": parse_mib(r"([\d.]+\s+[KMG]?B)\s+\S+\s+\S+\s+\d+\s+IOSurface", text)}
        return out

    # ------------------------------------------------------------ watchdog
    def watchdog(self, stop):
        sc = self.sc
        while not stop.is_set():
            s = {"t": time.time(), "swap_gib": swap_gib(), "free_pct": free_pct(), "wired_gib": wired_gib(),
                 "disk_gib": free_disk_gib(),
                 "rss_mib": {n: rss_mib([p.pid]).get(p.pid) for n, p in list(self.procs.items()) if p.poll() is None},
                 "ledgers": {n: ledgers_mib(p.pid) for n, p in list(self.procs.items()) if p.poll() is None}}
            self.samples.append(s)
            why = None
            if s["swap_gib"] - self.swap0 > sc.get("swap_max_gib", 4.0):
                why = f"swap grew {s['swap_gib'] - self.swap0:.2f} GiB"
            elif s["free_pct"] < sc.get("free_min_pct", 15):
                why = f"free memory {s['free_pct']:.0f}%"
            elif s["disk_gib"] < sc.get("disk_min_gib", 25):
                why = f"free disk {s['disk_gib']:.1f} GiB"
            elif time.time() - self.t0 > sc.get("max_seconds", 1800):
                why = "scenario past max_seconds"
            for n, p in list(self.procs.items()):
                if n in self.stop_flags and p.poll() is not None and not self.stop_flags[n].exists():
                    why = why or (f"{n}: a prediction ran past predict_timeout (ANE hang)" if p.returncode == 4
                                  else f"{n} exited {p.returncode} while running")
            if why:
                self.abort = why
                self.event(f"WATCHDOG: {why}; killing everything")
                self.kill_all()
                return
            stop.wait(1.0)

    # ------------------------------------------------------------ load threads
    def hammer(self, stop, kind, server, model=None, buckets=None, interval=0.0):
        i = 0
        while not stop.is_set() and not self.abort:
            if i and interval:
                stop.wait(interval)
                if stop.is_set():
                    break
            try:
                if kind == "afm":
                    ms, n = self.chat(server)
                    name = "apple-fm"
                else:
                    b = buckets[i % len(buckets)]
                    ms, n = self.embed(server, model, b)
                    name = f"{model}@{b}"
                with self.lock:
                    self.requests.append((kind, name, time.time(), ms, n))
            except Exception as e:     # noqa: BLE001 -- a timeout is a hang: stop the scenario
                self.abort = f"{kind} request failed or timed out: {e!r}"
                self.event(self.abort)
                return
            i += 1

    # ------------------------------------------------------------ scenario
    def run(self):
        sc = self.sc
        logf = open(self.work / "unified.log", "w")
        lg = subprocess.Popen(["/usr/bin/log", "stream", "--style", "compact", "--predicate",
                               'process == "aned" OR (process == "kernel" AND sender CONTAINS "ANE")'],
                              stdout=logf,
                              stderr=subprocess.STDOUT, start_new_session=True)
        wstop = threading.Event()
        wd = threading.Thread(target=self.watchdog, args=(wstop,), daemon=True)
        wd.start()
        disk0 = free_disk_gib()
        phases = []
        wired0 = wired_gib()
        try:
            if sc.get("autostart", True):
                self.start_decoder("d1")
                time.sleep(sc.get("settle_seconds", 5))
            for ph in sc["phases"]:
                if self.abort:
                    break
                ts = time.time()
                for d in ph.get("stop_decoders", []) + ([ph["stop_decoder"]] if "stop_decoder" in ph else []):
                    self.stop_decoder(d)
                for s in ph.get("stop_servers", []) + ([ph["stop_server"]] if "stop_server" in ph else []):
                    self.stop_server(s)
                if "start_decoder" in ph:
                    self.start_decoder(ph["start_decoder"])
                if "start_server" in ph:
                    self.start_server(ph["start_server"])
                for server, model, bucket in ph.get("touch", []):
                    w0 = wired_gib()
                    ms, n = self.embed(server, model, bucket)
                    self.event(f"touch {model}@{bucket}: {ms:.0f} ms, {n} tokens, wired +{wired_gib() - w0:.2f} GiB")
                    self.requests.append(("touch", f"{model}@{bucket}", time.time(), ms, n))
                for _ in range(ph.get("afm_warm", 0)):
                    ms, n = self.chat(ph["afm"])
                    self.event(f"afm warm-up: {ms:.0f} ms, {n} tokens")
                if time.time() - ts > 0.5:
                    phases.append({"name": ph["name"] + " (setup)", "start": ts, "end": time.time()})
                hstop = threading.Event()
                threads = [threading.Thread(target=self.hammer, args=(hstop, "embed", *h), daemon=True)
                           for h in ph.get("hammer", [])]
                if "afm" in ph:
                    threads.append(threading.Thread(target=self.hammer,
                                                    args=(hstop, "afm", ph["afm"], None, None, ph.get("afm_interval", 0.0)),
                                                    daemon=True))
                start = time.time()
                self.event(f"phase {ph['name']}")
                for t in threads:
                    t.start()
                end_at = start + ph["seconds"]
                while time.time() < end_at and not self.abort:
                    time.sleep(0.25)
                fps = self.footprints(len(phases)) if ph.get("footprint") else None
                hstop.set()
                for t in threads:
                    t.join(timeout=sc.get("request_timeout", 60) + 70)
                    if t.is_alive():
                        self.abort = self.abort or "a request didn't return (hang)"
                rec = {"name": ph["name"], "start": start, "end": time.time()}
                if fps:
                    rec["footprint_mib"] = fps
                phases.append(rec)
            for d in list(self.stop_flags):
                if self.procs[d].poll() is None and not self.abort and not self.stop_flags[d].exists():
                    self.stop_decoder(d)
        except Exception as e:      # noqa: BLE001
            self.abort = self.abort or f"{type(e).__name__}: {e}"
            self.event(f"ERROR {self.abort}")
        finally:
            wstop.set()
            for n, p in self.procs.items():
                if n not in self.stop_flags and p.poll() is None:
                    self.stop_server(n)
            self.kill_all()
            time.sleep(2)
            os.killpg(lg.pid, signal.SIGTERM)
            lg.wait()
        fmt = "%Y-%m-%d %H:%M:%S"
        compiles = sh("/usr/bin/log", "show", "--style", "compact", "--start",
                      time.strftime(fmt, time.localtime(self.t0 - 1)), "--end",
                      time.strftime(fmt, time.localtime(time.time() + 1)), "--predicate",
                      'process == "ANECompilerService" AND eventMessage BEGINSWITH "End of compilation of network"',
                      timeout=600).count("End of compilation of network")
        result = {"abort": self.abort, "disk_used_gib": round(disk0 - free_disk_gib(), 2),
                  "wired_gib_before": round(wired0, 2),
                  "ane_compiles": compiles,
                  "phases": phases, "events": [(round(t - self.t0, 2), w) for t, w in self.events],
                  "decoders": {}, "requests": {}, "memory": {}}
        for d in self.stop_flags:
            result["decoders"][d] = self.decoder_stats(d, phases)
        result["requests"] = self.request_stats(phases)
        result["memory"] = self.memory_stats(phases)
        self.out.write_text(json.dumps(result, indent=1))
        return result

    # ------------------------------------------------------------ analysis
    def decoder_stats(self, name, phases):
        calls = self.work / f"{name}_loop_calls.tsv"
        if not calls.exists():
            return {"error": "no output (killed while loading?)"}
        times = []      # (end time, ms, is_decode)
        for line in calls.read_text().splitlines():
            fields = line.split("\t")
            if len(fields) == 3:        # a killed run can leave a partial last line
                times.append((float(fields[0]), float(fields[1]), fields[2] == "decode"))
        summary = self.work / f"{name}.json"    # written when the decoder stops cleanly
        r = json.loads(summary.read_text()) if summary.exists() else {}
        out = {"load_s": round(r["load_ms"] / 1000, 1) if "load_ms" in r else None, "calls": len(times),
               "stalls": r.get("health", {}).get("stalls"), "phases": {}}
        base = None
        for ph in phases:
            xs = [ms for t, ms, dec in times if ph["start"] <= t < ph["end"] and dec]
            if not xs:
                continue
            xs.sort()
            med = statistics.median(xs)
            base = base or med
            out["phases"][ph["name"]] = {
                "n": len(xs), "median_ms": round(med, 1), "p90_ms": round(xs[int(len(xs) * 0.9)], 1),
                "max_ms": round(xs[-1], 1), "tok_s": round(1000 / med, 1),
                "over_2x_base": sum(x > 2 * base for x in xs), "over_5x_base": sum(x > 5 * base for x in xs)}
        return out

    def request_stats(self, phases):
        out = {}
        for ph in phases:
            byname = {}
            for kind, name, t, ms, n in self.requests:
                if ph["start"] <= t <= ph["end"] + 0.01:
                    byname.setdefault(name, []).append((ms, n))
            if byname:
                out[ph["name"]] = {k: {"n": len(v), "median_ms": round(statistics.median(x for x, _ in v), 1),
                                       "max_ms": round(max(x for x, _ in v), 1), "tokens": v[0][1]}
                                   for k, v in byname.items()}
        return out

    def memory_stats(self, phases):
        out = {}
        for ph in phases:
            ss = [s for s in self.samples if ph["start"] <= s["t"] <= ph["end"]]
            if not ss:
                continue
            rss, neural, phys = {}, {}, {}
            for s in ss:
                for n, v in s["rss_mib"].items():
                    if v:
                        rss[n] = max(rss.get(n, 0), v)
                for n, v in s.get("ledgers", {}).items():
                    if v:
                        neural.setdefault(n, []).append(v["neural"])
                        phys[n] = max(phys.get(n, 0), v["phys"])
            totals = [sum(v["neural"] for v in s.get("ledgers", {}).values() if v) for s in ss]
            out[ph["name"]] = {"wired_gib": round(statistics.median(s["wired_gib"] for s in ss), 2),
                               "min_free_pct": min(s["free_pct"] for s in ss),
                               "max_swap_gib": round(max(s["swap_gib"] for s in ss), 2),
                               "peak_rss_mib": {n: round(v) for n, v in rss.items()},
                               "peak_phys_footprint_mib": {n: round(v) for n, v in phys.items()},
                               "neural_mib": {n: [round(min(v)), round(statistics.median(v)), round(max(v))]
                                              for n, v in neural.items()},
                               "neural_total_mib": [round(min(totals)), round(statistics.median(totals)),
                                                    round(max(totals))]}
            if "footprint_mib" in ph:
                out[ph["name"]]["footprint_mib"] = ph["footprint_mib"]
        return out


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    sc = json.loads(Path(sys.argv[1]).read_text())
    res = Probe(sc, sys.argv[2]).run()
    print(json.dumps({k: res[k] for k in ("abort", "disk_used_gib", "ane_compiles", "wired_gib_before")}))
    for d, s in res["decoders"].items():
        for ph, v in s.get("phases", {}).items():
            print(f"{d:3} {ph:40} {v}")
    for ph, v in res["requests"].items():
        print(f"req {ph:40} {v}")
    for ph, v in res["memory"].items():
        print(f"mem {ph:40} {v}")
    sys.exit(1 if res["abort"] else 0)
