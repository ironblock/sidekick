"""watchdog.py <timeout_s> <log> <cmd> [args...]: run one benchmark row under a
memory watchdog.

The row (tools/chat_bench/run.py, which starts and stops its own model server
in the same process group) runs in a new process group. Every 2 s the
watchdog checks the system: swap used over SWAP_MAX_GB (`sysctl
vm.swapusage`), free memory under FREE_MIN_PCT (`memory_pressure -Q`) or the
timeout kills the whole group and records why. It waits on the child's PID,
never on a process-name match. It logs the row's output, the peak RSS of the
group, the lowest free-memory percentage and the most swap seen.
Exit 0 ok, 1 the row failed, 3 killed (reason in the log).
"""

import json
import os
import re
import signal
import subprocess
import sys
import threading
import time

SWAP_MAX_GB = float(os.environ.get("CHAT_BENCH_SWAP_MAX_GB", 4.0))
FREE_MIN_PCT = float(os.environ.get("CHAT_BENCH_FREE_MIN_PCT", 15.0))


def swap_gb() -> float:
    s = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout
    m = re.search(r"used = ([\d.]+)M", s)
    return float(m.group(1)) / 1024 if m else 0.0


def free_pct() -> float:
    s = subprocess.run(["memory_pressure", "-Q"], capture_output=True, text=True).stdout
    m = re.search(r"free percentage: (\d+)%", s)
    return float(m.group(1)) if m else 100.0


def group_rss_mb(pgid: int) -> float:
    s = subprocess.run(["ps", "-A", "-o", "pgid=,rss="], capture_output=True, text=True).stdout
    rows = (line.split() for line in s.splitlines() if line.strip())
    return sum(int(r) for g, r in rows if int(g) == pgid) / 1024


def main() -> int:
    timeout, log, *cmd = sys.argv[1:]
    timeout = float(timeout)
    if cmd[0].endswith(".py"):
        cmd = [sys.executable, *cmd]
    logf = open(log, "a", buffering=1)
    t0 = time.time()
    logf.write(f"\n[wd] start {time.strftime('%H:%M:%S')}: {' '.join(cmd)}\n")
    p = subprocess.Popen(cmd, env=dict(os.environ, PYTHONUNBUFFERED="1"), stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, start_new_session=True)
    pgid = p.pid

    def pump():
        for line in p.stdout:
            logf.write(line)

    th = threading.Thread(target=pump, daemon=True)
    th.start()
    peak_rss, min_free, max_swap, reason = 0.0, 100.0, 0.0, None
    while p.poll() is None:
        time.sleep(2)
        sw, fr = swap_gb(), free_pct()
        peak_rss = max(peak_rss, group_rss_mb(pgid))
        min_free, max_swap = min(min_free, fr), max(max_swap, sw)
        if sw > SWAP_MAX_GB:
            reason = f"swap used {sw:.2f} GB > {SWAP_MAX_GB} GB"
        elif fr < FREE_MIN_PCT:
            reason = f"free memory {fr:.0f}% < {FREE_MIN_PCT:.0f}%"
        elif time.time() - t0 > timeout:
            reason = f"timeout {timeout:.0f} s"
        if reason:
            os.killpg(pgid, signal.SIGKILL)
            break
    p.wait()
    th.join(timeout=5)
    # Kill anything the row left behind in its group (a server it failed to stop).
    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    summary = {"cmd": " ".join(os.path.basename(c) for c in cmd[:3]), "exit": p.returncode, "killed": reason,
               "wall_s": round(time.time() - t0, 1), "peak_group_rss_MB": round(peak_rss),
               "min_free_pct": min_free, "max_swap_GB": round(max_swap, 2)}
    logf.write(f"[wd] {json.dumps(summary)}\n")
    print(json.dumps(summary))
    if reason:
        print(f"KILLED: {reason}")
        return 3
    return 0 if p.returncode == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
