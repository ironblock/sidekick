"""powerlog.py <out.jsonl> [interval_s]: whole-system power without sudo, from the
battery controller's telemetry (AppleSmartBattery PowerTelemetryData, read
with ioreg): SystemPowerIn is the power entering the system from the adapter
(mW), so on AC it is the machine's total draw, display included. One JSON line
per sample. powermetrics needs root, and IOReport's per-rail energy counters
read back frozen here (powermon.c), so this is the per-rail-free fallback.

Averages over a window come from the accumulators:
  mean mW = d(AccumulatedSystemPowerIn) / d(SystemPowerInAccumulatorCount)."""
import json
import plistlib
import subprocess
import sys
import time

out = open(sys.argv[1], "a", buffering=1)
interval = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
KEYS = ("SystemPowerIn", "AccumulatedSystemPowerIn", "SystemPowerInAccumulatorCount", "BatteryPower",
        "AccumulatedBatteryPower", "BatteryPowerAccumulatorCount")
while True:
    raw = subprocess.run(["ioreg", "-r", "-c", "AppleSmartBattery", "-a"], capture_output=True).stdout
    try:
        t = plistlib.loads(raw)[0].get("PowerTelemetryData", {})
        out.write(json.dumps({"t": round(time.time(), 3), **{k: t.get(k) for k in KEYS}}) + "\n")
    except Exception as e:  # keep sampling through a bad read
        out.write(json.dumps({"t": time.time(), "error": str(e)}) + "\n")
    time.sleep(interval)
