"""wd.py <timeout_s> <log> <cmd> [args...]: run one heavy step under a watchdog.

- The step runs in its own process group (under /usr/bin/time -l, for its peak
  memory footprint), with a private Core ML home under $SPIKE_WORK.
- Every 2 s the watchdog checks the system: swap used over SWAP_MAX_GB
  (sysctl vm.swapusage) or free memory under FREE_MIN_PCT (memory_pressure -Q)
  kills the whole group and records why. So does the timeout (an ANE call
  that hangs). It waits on the child's PID, never on a process-name match.
- Logs the step's output, the peak RSS of its group, the minimum free-memory
  percentage and maximum swap seen, and the peak footprint /usr/bin/time
  reports. Exit 0 ok, 1 step failed, 3 killed (reason in the log).
.py commands run with this interpreter; anything else is executed as given."""
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

SWAP_MAX_GB = 4.0
FREE_MIN_PCT = 15.0
WORK = Path(os.environ.get("SPIKE_WORK", ".dev-data/spike-ane-decoder"))   # scratch: Core ML home, memory log
HOME = WORK / "home"

timeout, log, *cmd = sys.argv[1:]
timeout = float(timeout)
if cmd[0].endswith(".py"):
    cmd = [sys.executable, *cmd]
logf = open(log, "a", buffering=1)


def swap_gb():
    s = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout
    m = re.search(r"used = ([\d.]+)M", s)
    return float(m.group(1)) / 1024 if m else 0.0


def free_pct():
    s = subprocess.run(["memory_pressure", "-Q"], capture_output=True, text=True).stdout
    m = re.search(r"free percentage: (\d+)%", s)
    return float(m.group(1)) if m else 100.0


def group_rss_mb(pgid):
    s = subprocess.run(["ps", "-A", "-o", "pgid=,rss="], capture_output=True, text=True).stdout
    return sum(int(r) for g, r in (l.split() for l in s.splitlines() if l.strip()) if int(g) == pgid) / 1024


def free_kib():
    return int(subprocess.run(["df", "-k", "/"], capture_output=True, text=True).stdout.splitlines()[1].split()[3])


env = dict(os.environ, CFFIXED_USER_HOME=str(HOME), PYTHONUNBUFFERED="1")
HOME.mkdir(exist_ok=True)
disk0 = free_kib()
t0 = time.time()
logf.write(f"\n[wd] start {time.strftime('%H:%M:%S')}: {' '.join(cmd)}\n")
p = subprocess.Popen(["/usr/bin/time", "-l", *cmd], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                     text=True, start_new_session=True)
pgid = p.pid
out_lines = []
import threading


def pump():
    for line in p.stdout:
        out_lines.append(line)
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
        reason = f"timeout {timeout:.0f} s (possible ANE hang)"
    if reason:
        os.killpg(pgid, signal.SIGKILL)
        break
p.wait()
th.join(timeout=5)
text = "".join(out_lines)
m = re.search(r"(\d+)\s+peak memory footprint", text)
summary = {"cmd": " ".join(os.path.basename(c) for c in cmd[:4]), "exit": p.returncode, "killed": reason,
           "wall_s": round(time.time() - t0, 1), "peak_group_rss_MB": round(peak_rss),
           "peak_footprint_MB": round(int(m.group(1)) / 2**20) if m else None,
           "min_free_pct": min_free, "max_swap_GB": round(max_swap, 2),
           "disk_free_change_MiB": round((disk0 - free_kib()) / 1024)}
logf.write(f"[wd] {json.dumps(summary)}\n")
(WORK / "results").mkdir(parents=True, exist_ok=True)
with open(WORK / "results" / "memory_log.jsonl", "a") as f:
    f.write(json.dumps({"time": time.strftime("%Y-%m-%d %H:%M:%S"), **summary}) + "\n")
print(json.dumps(summary))
for l in out_lines:
    if l.startswith("RESULT "):
        print(l.strip())
if reason:
    print(f"KILLED: {reason}")
    sys.exit(3)
sys.exit(0 if p.returncode == 0 else 1)
