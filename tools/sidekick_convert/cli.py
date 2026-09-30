"""The command line every converter shares:

    python tools/convert_<model>.py [flags] <hf-model-dir> <install-dir> [buckets...]

Flags may appear anywhere. `--time` adds latency to the gate report; leave it
off unless the machine is quiet, since latency moves with load and accuracy
doesn't. A converter adds its own flags (negative controls) through `flags`.
"""

import argparse
from pathlib import Path


def parse(description, *, flags=(), default_buckets=(128, 256, 512), argv=None):
    """flags: (name, argparse kwargs) pairs, e.g. ("--no-pad-zeroing",
    {"action": "store_true", "help": ...}). default_buckets=None leaves
    args.buckets empty when none are given, for a converter that takes them
    from the manifest."""
    p = argparse.ArgumentParser(description=description)
    p.add_argument("src", type=lambda s: Path(s).expanduser(), help="local Hugging Face snapshot")
    p.add_argument("install_dir", type=lambda s: Path(s).expanduser(),
                   help="model directory the daemon scans, named after the model id")
    p.add_argument("buckets", nargs="*", type=int,
                   help=f"default {' '.join(map(str, default_buckets))}" if default_buckets else
                   "default: the manifest's buckets")
    p.add_argument("--time", action="store_true", help="also measure latency (only on a quiet machine)")
    for name, kwargs in flags:
        p.add_argument(name, **kwargs)
    args = p.parse_intermixed_args(argv)
    args.buckets = sorted(args.buckets) or list(default_buckets or [])
    return args
