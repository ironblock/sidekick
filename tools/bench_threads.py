#!/usr/bin/env python3
"""Thread scaling of sidekickd's onnx-cpu path, through the HTTP API.

For each intra-op thread count: start sidekickd with that
`onnx_intra_op_threads`, then for each model time sequential single-input
/v1/embeddings requests at ~16 and ~256 tokens. Standard library only.

  bench_threads.py --daemon PATH --models-dir DIR --model ID [--model ID ...]
                   --threads 1,2,4,8 [--reps 20] [--out results.json]
"""
import argparse, json, os, platform, signal, socket, statistics, subprocess, sys, tempfile, time, urllib.request

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
                    help="before each setting with at least this many threads, wait for the GPU to be idle (0: never)")
    a = ap.parse_args()
    threads = [int(x) for x in a.threads.split(",")]
    result = {"machine": machine(), "reps": a.reps, "runs": []}
    texts = {}
    work = tempfile.mkdtemp(prefix="sk-bench-")
    for n in threads:
        port = free_port()
        cfg = os.path.join(work, f"config-{n}.toml")
        with open(cfg, "w") as f:
            f.write(f'addr = "127.0.0.1:{port}"\nmodels_dir = "{os.path.abspath(a.models_dir)}"\nonnx_intra_op_threads = {n}\n')
        log = open(os.path.join(work, f"daemon-{n}.log"), "w")
        proc = subprocess.Popen([a.daemon, "--config", cfg], stdout=log, stderr=subprocess.STDOUT)
        try:
            for _ in range(300):
                try:
                    urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2).read()
                    break
                except Exception:
                    time.sleep(0.1)
            else:
                sys.exit(f"sidekickd didn't start; see {log.name}")
            for model in a.model:
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
                                 "min_ms": min(times), "p90_ms": sorted(times)[int(0.9 * (len(times) - 1))]}
                run["load_after"] = loadavg()
                result["runs"].append(run)
                print(f"threads {n:>2}  {model:<24} short {run['short']['tokens']:>4} tok {run['short']['median_ms']:8.1f} ms"
                      f"   long {run['long']['tokens']:>4} tok {run['long']['median_ms']:8.1f} ms   load {run['load_after'][0]:.1f}",
                      flush=True)
        finally:
            proc.send_signal(signal.SIGTERM)
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                proc.kill()
            log.close()
    with open(a.out, "w") as f:
        json.dump(result, f, indent=1)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
