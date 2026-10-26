"""lut_extract.py <qweights_dir> <out.pt>: the palettes of a gptq.py palette run,
{"<layer>.<proj>": lut, "head": lut}, for gptq.py --lut-from (k-means depends
only on the weights, so another calibration can reuse them)."""
import sys
from pathlib import Path

import torch

d = Path(sys.argv[1])
out = {}
for f in sorted(d.glob("layer_*.pt")):
    i = int(f.stem.split("_")[1])
    for p, (q, lut) in torch.load(f).items():
        out[f"{i}.{p}"] = lut
if (d / "head.pt").exists():
    out["head"] = torch.load(d / "head.pt")["head"][1]
torch.save(out, sys.argv[2])
print(len(out), "palettes")
