"""The command line every converter shares:

    python tools/convert_<model>.py [flags] <hf-model-dir> <install-dir> [buckets...]

Flags may appear anywhere. `--time` adds latency to the gate report; leave it
off unless the machine is quiet, since latency moves with load and accuracy
doesn't. `--int8-embedding` stores the token-embedding table in int8, and
`--ignore-ane-weight-cap` converts a model served on the ANE past the Neural
Engine's per-program weight limit with a warning instead of an error
(docs/CONVERTING.md). A backbone that can be chunked converts each bucket
over the weight budget as a chain of programs split at layer boundaries
(chunking.py, D37); `--chunks auto|N|cuts` chooses the split, and
`--chunks 1` keeps one program per bucket; `--chunk-identity-all` runs the
chain's bit-identity gate on every bucket, not only the smallest. A
converter passes these on with job_options().

Core ML caches a compiled bundle for every model a process loads, keyed by
the model's path, under ~/Library/Caches/<executable>. A conversion loads
each bucket from a temporary directory and then moves it, so every entry it
leaves is orphaned: about the bucket's weights again per compute path
(~11 GB for one 0.6B model). parse() therefore runs the converter in a
child process whose Core ML cache lives in a temporary home
(CFFIXED_USER_HOME), deleted when it exits. A caller that sets
CFFIXED_USER_HOME itself keeps that cache. A
converter adds its own flags (negative controls) through `flags`.
"""

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path


def _private_coreml_cache():
    """Re-run this converter with Core ML's cache in a temporary home that
    is deleted afterwards (see the module docstring), unless the caller
    chose one. The child gets the same interpreter options. Ctrl-C reaches
    it too (one process group); the parent waits for it to finish cleaning
    up its own temporary files, then removes the home. A child killed by a
    signal exits the parent with 128 + the signal, as a shell reports it."""
    if sys.platform != "darwin" or "CFFIXED_USER_HOME" in os.environ:
        return
    with tempfile.TemporaryDirectory(prefix="sidekick-convert-cache-") as home:
        child = subprocess.Popen([sys.executable, *sys.orig_argv[1:]], env={**os.environ, "CFFIXED_USER_HOME": home})
        while True:
            try:
                code = child.wait()
                break
            except KeyboardInterrupt:
                continue
    raise SystemExit(128 - code if code < 0 else code)


def parse(description, *, flags=(), default_buckets=(128, 256, 512), argv=None):
    """flags: (name, argparse kwargs) pairs, e.g. ("--no-pad-zeroing",
    {"action": "store_true", "help": ...}). default_buckets=None leaves
    args.buckets empty when none are given, for a converter that takes them
    from the manifest. Without `argv` (a converter's command line), the
    conversion runs with a private Core ML cache."""
    p = argparse.ArgumentParser(description=description)
    p.add_argument("src", type=lambda s: Path(s).expanduser(), help="local Hugging Face snapshot")
    p.add_argument("install_dir", type=lambda s: Path(s).expanduser(),
                   help="model directory the daemon scans, named after the model id")
    p.add_argument("buckets", nargs="*", type=int,
                   help=f"default {' '.join(map(str, default_buckets))}" if default_buckets else
                   "default: the manifest's buckets")
    p.add_argument("--time", action="store_true", help="also measure latency (only on a quiet machine)")
    p.add_argument("--int8-embedding", action="store_true",
                   help="store the token-embedding table in int8 (halves it; graded like any rewrite)")
    p.add_argument("--ignore-ane-weight-cap", action="store_true",
                   help="convert past the Neural Engine's 1 GiB per-program weight limit, with a warning "
                        "recorded in the installed manifest")
    p.add_argument("--chunks", default=None,
                   help="split each bucket into programs at layer boundaries: auto (each under 0.9 GiB; the "
                        "default for a backbone that can be chunked), a count, the layers at which chunks after "
                        "the first begin (10,20), or 1 for one program per bucket; D37")
    p.add_argument("--chunk-identity-all", action="store_true",
                   help="run a chunked bucket's GPU bit-identity gate in every bucket, not only the smallest")
    p.add_argument("--format", choices=("coreml", "onnx"), default="coreml",
                   help="coreml: per-bucket .mlmodelc for the ANE (the default); onnx: one dynamic-shape "
                        "fp32 model.onnx for ONNX Runtime's CPU backend")
    for name, kwargs in flags:
        p.add_argument(name, **kwargs)
    args = p.parse_intermixed_args(argv)
    if argv is None:
        _private_coreml_cache()
    args.buckets = sorted(args.buckets) or list(default_buckets or [])
    return args


def job_options(args):
    """The shared flags that shape a Job, for recipes.* or core.Job."""
    from .chunking import parse_spec
    return {"int8_embedding": args.int8_embedding, "ignore_ane_weight_cap": args.ignore_ane_weight_cap,
            "chunks": parse_spec(args.chunks), "chunk_identity_all": args.chunk_identity_all,
            "format": args.format}
