#!/usr/bin/env python3
"""Thread scaling of sidekickd's onnx-cpu path, through the HTTP API.

For each intra-op thread count: start sidekickd with that
`onnx_intra_op_threads`, check in /health that each model runs with it,
then for each model time sequential single-input /v1/embeddings requests
at ~16 and ~256 tokens. Token counts are sidekickd's `usage` figure, an
estimate of four bytes per token, not the tokenizer's count. Results are
written after every setting. Standard library only.

  bench_threads.py --daemon PATH --models-dir DIR --model ID [--model ID ...]
                   --threads 1,2,4,8 [--reps 20] [--out results.json]
"""
import argparse, json, math, os, platform, signal, socket, statistics, subprocess, sys, tempfile, time, urllib.request

SENTENCE = ("The committee reviewed the quarterly figures, noted a modest rise in "
            "operating costs, and asked the regional teams to report on supplier delays. ")


def machine():
    info = {"platform": platform.platform(), "machine": platform.machine(), "logical_cpus": os.cpu_count()}
    if sys.platform == "darwin":
        def sysctl(k):
            try:
                return subprocess.check_output(["sysctl", "-n", k], text=True).strip()
            except Exception:
                return None
        info.update(cpu=sysctl("machdep.cpu.brand_string") or sysctl("hw.model"),
                    physical_cpus=sysctl("hw.physicalcpu"),
                    perflevel0_physical=sysctl("hw.perflevel0.physicalcpu"),
                    perflevel1_physical=sysctl("hw.perflevel1.physicalcpu"))
    else:
        try:
            out = subprocess.check_output(["lscpu"], text=True)
            for line in out.splitlines():
                k, _, v = line.partition(":")
                if k.strip() in ("Model name", "Thread(s) per core", "Core(s) per socket", "Socket(s)", "CPU(s)"):
                    info[k.strip()] = v.strip()
        except Exception:
            pass
    return info


def loadavg():
    try:
        return list(os.getloadavg())
    except OSError:
        return None


def post(port, body):
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/embeddings", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.loads(r.read())


def tokens(resp):
    u = resp.get("usage") or {}
    return u.get("prompt_tokens")


def text_for(port, model, target):
    """The shortest repetition of SENTENCE whose token count reaches target."""
    words = SENTENCE.split()
    if target <= 24:
        for n in range(1, len(words) + 1):
            t = " ".join(words[:n])
            if (tokens(post(port, {"model": model, "input": t})) or 0) >= target:
                return t
        return SENTENCE
    n = 1
    while True:
        t = SENTENCE * n
        k = tokens(post(port, {"model": model, "input": t}))
        if k is None or k >= target or n > 64:
            return t
        n += 1


def configured_threads(health, model):
    """The intra-op threads /health reports for an onnx model, or None."""
    stack = [health]
    while stack:
        node = stack.pop()
        if isinstance(node, dict):
            entry = node.get(model)
            if isinstance(entry, dict) and "intra_op_threads" in entry:
                return entry["intra_op_threads"]
            stack.extend(node.values())
        elif isinstance(node, list):
            stack.extend(node)
    return None


def p90(times):
    """Nearest-rank 90th percentile."""
    ordered = sorted(times)
    return ordered[max(math.ceil(0.9 * len(ordered)) - 1, 0)]


def wait_gpu_idle(threshold=5, quiet_secs=20):
    """Block until nvidia-smi reports GPU utilization under threshold for
    quiet_secs in a row, so a CPU-heavy setting doesn't slow GPU serving."""
    calm = 0
    while calm < quiet_secs:
        try:
            out = subprocess.check_output(["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"], text=True)
            busy = max(int(x) for x in out.split())
        except Exception:
            return
        if busy < threshold:
            calm += 2
        else:
            print(f"  GPU busy ({busy}%), waiting", flush=True)
            calm = 0
        time.sleep(2)


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--daemon", required=True)
    ap.add_argument("--models-dir", required=True)
    ap.add_argument("--model", action="append", required=True)
    ap.add_argument("--threads", required=True)
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--out", default="bench_threads.json")
    ap.add_argument("--wait-gpu-idle-from", type=int, default=0,
                    help="before each setting with at least this many threads, wait for an NVIDIA GPU to be idle, "
                         "by nvidia-smi (0: never; without nvidia-smi it doesn't wait)")
    a = ap.parse_args()
    if a.reps < 1:
        sys.exit("--reps must be at least 1")
    threads = [int(x) for x in a.threads.split(",")]
    result = {"machine": machine(), "reps": a.reps, "runs": []}
    texts = {}
    work = tempfile.mkdtemp(prefix="sk-bench-")
    for n in threads:
        port = free_port()
        cfg = os.path.join(work, f"config-{n}.toml")
        with open(cfg, "w") as f:
            # json.dumps quotes and escapes the path as a TOML basic string
            f.write(f'addr = "127.0.0.1:{port}"\nmodels_dir = {json.dumps(os.path.abspath(a.models_dir))}\n'
                    f'onnx_intra_op_threads = {n}\n')
        log = open(os.path.join(work, f"daemon-{n}.log"), "w")
        proc = subprocess.Popen([a.daemon, "--config", cfg], stdout=log, stderr=subprocess.STDOUT)
        try:
            health = None
            for _ in range(300):
                if proc.poll() is not None:
                    sys.exit(f"sidekickd exited ({proc.returncode}); see {log.name}")
                try:
                    health = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2).read())
                    break
                except Exception:
                    time.sleep(0.1)
            else:
                sys.exit(f"sidekickd didn't start; see {log.name}")
            for model in a.model:
                got = configured_threads(health, model)
                if got != n:
                    sys.exit(f"{model}: /health reports {got} intra-op threads, not {n}: is it an onnx model, "
                             f"or does a per-model setting override the config? See {log.name}")
                if a.wait_gpu_idle_from and n >= a.wait_gpu_idle_from:
                    wait_gpu_idle()
                if model not in texts:
                    texts[model] = {"short": text_for(port, model, 16), "long": text_for(port, model, 256)}
                run = {"threads": n, "model": model, "load_before": loadavg()}
                for kind in ("short", "long"):
                    body = {"model": model, "input": texts[model][kind]}
                    for _ in range(3):
                        resp = post(port, body)
                    times = []
                    for _ in range(a.reps):
                        t0 = time.perf_counter()
                        post(port, body)
                        times.append((time.perf_counter() - t0) * 1000)
                    run[kind] = {"tokens": tokens(resp), "median_ms": statistics.median(times),
                                 "min_ms": min(times), "p90_ms": p90(times)}
                run["load_after"] = loadavg()
                result["runs"].append(run)
                load = run["load_after"][0] if run["load_after"] else float("nan")
                print(f"threads {n:>2}  {model:<24} short {run['short']['tokens'] or '?':>4} tok "
                      f"{run['short']['median_ms']:8.1f} ms   long {run['long']['tokens'] or '?':>4} tok "
                      f"{run['long']['median_ms']:8.1f} ms   load {load:.1f}", flush=True)
                with open(a.out, "w") as f:
                    json.dump(result, f, indent=1)
        finally:
            proc.send_signal(signal.SIGTERM)
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                proc.kill()
            log.close()
    with open(a.out, "w") as f:
        json.dump(result, f, indent=1)
    print(f"wrote {a.out}; daemon logs in {work}")


if __name__ == "__main__":
    main()
