"""probe_load.py <mlmodelc> <units> [function]: does Core ML load this program?"""
import sys
import coremltools as ct

path, units = sys.argv[1], sys.argv[2]
kw = {"function_name": sys.argv[3]} if len(sys.argv) > 3 else {}
try:
    ct.models.CompiledMLModel(path, compute_units=getattr(ct.ComputeUnit, units), **kw)
    print("RESULT {\"loaded\": true}")
except Exception as e:
    print("load error:", e)
