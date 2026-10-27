"""Convert a LlamaForCausalLM checkpoint into a `kv-state-1` decoder for
sidekick's chat route (docs/DECISIONS.md D53; docs/CONVERTING.md, "Causal
language models").

    python tools/convert_causal_lm.py <hf-model-dir> <install-dir> --work <dir> [flags]

The install directory is named after the model id, whose committed manifest
is examples/chat/<id>/chat.toml. The recipe (sidekick_convert.recipes.
causal_lm) rotates the weights (R1, R2), calibrates the residual scale and
norm pre-scales, quantises by GPTQ onto 8-bit palettes on a seeded draw of
the committed calibration fixture (fixtures/causal-lm/), screens the result
in PyTorch, with a second draw that must grade alike, and only then
converts, gates and installs the chunk programs.

`--work` keeps every intermediate (fp32 references, ranges, palettes, GPTQ
weights per draw, screens, the staged programs), so a rerun picks up where
the last stopped; give a fresh directory to reproduce a build from scratch.
Grade the installed build on the ANE with tools/grade_causal_lm.py.

Memory: each heavy step runs in its own process. The largest are the fp32
references (the checkpoint in fp32: about 10 GB for a 2B model) and the
conversion of the largest chunk; GPTQ and the screen hold one layer in fp32
at a time. ANE cache: the compute-plan gate reads each chunk's plan, which
compiles it for the ANE once and leaves about its weights in a system cache
that can't be deleted (docs/design/coreml-cache.md); `--no-plan` skips it,
for a rebuild whose programs only need comparing.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sidekick_convert import cli, recipes  # noqa: E402
from sidekick_convert.causal_lm import run  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("src", type=lambda s: Path(s).expanduser(), help="local Hugging Face snapshot")
    p.add_argument("install_dir", type=lambda s: Path(s).expanduser(),
                   help="model directory the daemon scans, named after the model id")
    p.add_argument("--work", type=lambda s: Path(s).expanduser(), required=True,
                   help="where intermediates are kept and reused (a persistent directory, not the system's "
                        "temporary one, which macOS prunes)")
    p.add_argument("--draw", type=int, default=None, help="the calibration draw's seed (default: the recipe's)")
    p.add_argument("--stability-draw", type=int, default=None,
                   help="the second draw's seed, screened in PyTorch only (default: the recipe's)")
    p.add_argument("--no-stability", action="store_true", help="skip the second draw")
    p.add_argument("--tokens", type=int, default=None,
                   help="calibration tokens per draw, a multiple of 2,048 (default: the recipe's); a recipe "
                        "variant: screen it first (--screen-only)")
    p.add_argument("--chat-sequences", type=int, default=None,
                   help="of each draw's sequences, how many are long chat requests (default: the recipe's)")
    p.add_argument("--fp16-projections", default=None,
                   help="projections to leave unquantised, as layer.proj (3.down,3.up) or a whole layer (3.*); "
                        "each adds its fp16 size over its 8-bit palette to the ANE's weights (default: the "
                        "recipe's). Find them with tools/screen_fp16_projections.py")
    p.add_argument("--screen-only", action="store_true", help="stop after the PyTorch screen: compile nothing")
    p.add_argument("--no-plan", action="store_true",
                   help="skip the compute-plan gate (no ANE compile); for a rebuild compared byte for byte")
    p.add_argument("--keep-packages", action="store_true",
                   help="keep each chunk's .mlpackage under <work>/stage/packages")
    args = p.parse_args()
    cli._private_coreml_cache()
    model_id = args.install_dir.name
    job = recipes.causal_lm(model_id=model_id, src=args.src, work=args.work, draw=args.draw,
                            stability_draw=-1 if args.no_stability else args.stability_draw, tokens=args.tokens,
                            chat_sequences=args.chat_sequences, fp16_projections=args.fp16_projections,
                            plan=not args.no_plan, screen_only=args.screen_only, keep_packages=args.keep_packages)
    run(job, args.install_dir)


if __name__ == "__main__":
    main()
