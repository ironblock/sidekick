"""ONNX export of an embedder, for ONNX Runtime's CPU backend.

The same converter, backbone, head and gate texts as the Core ML artifact,
so the two compare model for model (`--format onnx` on a converter). What
differs is the graph's shape contract:

- DYNAMIC SHAPES. One model.onnx with symbolic batch and sequence axes, no
  buckets: ONNX Runtime pads a batch to its longest input.
- int64 INPUTS, input_ids and attention_mask, as ONNX exports conventionally
  take them, so a published export and sidekick's take the same feed.
- TOKEN-LEVEL OUTPUT. `last_hidden_state` [batch, seq, hidden], and the
  server pools it (mean, cls or last_token, from the manifest), as it would
  a published export's. A head that is more than pooling (EmbeddingGemma's
  dense stack) stays in the graph and outputs the pooled vector.
- fp32, with none of the ANE's precision rewrites: they are exact in fp32
  and only add operations on the CPU.

Gates, in fp32 through ONNX Runtime's Python API on the CPU:
- every gate case, unpadded, pooled as the server pools it, against the
  checkpoint's own fp32 output (the Core ML gate's references): cosine at
  least FP32_COS;
- batch invariance: all the gate cases in one right-padded batch give each
  case's unpadded result (cosine at least BATCH_COS);
- pad invariance: random pad ids instead of 0 change nothing.
"""

import json
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

from .core import GateFailure

FP32_COS = 0.99999
BATCH_COS = 0.999999
OPSET = 17
POOLED_OUTPUT = "embedding"
TOKEN_OUTPUT = "last_hidden_state"


def pooling_of(head):
    """The server-side pooling for a head, or None when the graph pools."""
    mode = getattr(head, "mode", None)
    if mode in ("cls", "mean", "last_token") and not getattr(head, "l2", False):
        return mode
    return None


class TokenLevel(torch.nn.Module):
    """The backbone's model on dynamic int64 inputs, returning its last
    hidden state. The backbone's patches (finite masks, traceable helpers)
    stay; position ids and token types come from the model itself, sized
    from the input, rather than from the Core ML wrapper's per-bucket
    buffers."""

    def __init__(self, backbone, token_type_ids=False):
        super().__init__()
        self.model = backbone.model
        self.call = getattr(backbone, "onnx_call", None)
        self.token_type_ids = token_type_ids

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        if self.call is not None:
            return self.call(self.model, input_ids, attention_mask)
        kw = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            kw["token_type_ids"] = token_type_ids
        return self.model(**kw).last_hidden_state


def pool(hidden, mask, mode):
    """The server's pooling, in float64: hidden [n, hidden] of one input,
    mask [n]."""
    real = np.flatnonzero(mask)
    if mode == "cls":
        return hidden[0]
    if mode == "mean":
        return hidden[real].mean(axis=0)
    if mode == "last_token":
        return hidden[real[-1]]
    raise ValueError(mode)


def cosine(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    den = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / den) if den > 0 and np.isfinite(den) else float("nan")


def export(module, inputs, output, path):
    """torch.onnx's TorchScript exporter, with batch and sequence dynamic.
    Weights over protobuf's 2 GB limit go to a model.onnx_data file."""
    names = list(inputs)
    dynamic = {n: {0: "batch", 1: "sequence"} for n in names}
    dynamic[output] = {0: "batch"} if output == POOLED_OUTPUT else {0: "batch", 1: "sequence"}
    with torch.no_grad():
        torch.onnx.export(module, tuple(inputs[n] for n in names), str(path), input_names=names,
                          output_names=[output], dynamic_axes=dynamic, opset_version=OPSET, dynamo=False,
                          do_constant_folding=True)
    import onnx
    # Past protobuf's 2 GB the exporter writes each weight to its own file
    # beside the model; gather them into one model.onnx_data.
    scattered = [f for f in path.parent.iterdir() if f != path]
    if scattered or path.stat().st_size > 1_900_000_000:
        model = onnx.load(str(path), load_external_data=True)
        for f in scattered:
            f.unlink()
        onnx.save_model(model, str(path), save_as_external_data=True, all_tensors_to_one_file=True,
                        location=path.name + "_data", size_threshold=1024)
    onnx.checker.check_model(str(path))


class Runner:
    """An ONNX Runtime CPU session that feeds int64 and pools like the server."""

    def __init__(self, path, pooling, threads=None, token_type_ids=None):
        import onnxruntime as ort
        opts = ort.SessionOptions()
        if threads:
            opts.intra_op_num_threads = threads
        self.session = ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])
        self.inputs = [i.name for i in self.session.get_inputs()]
        names = [o.name for o in self.session.get_outputs()]
        self.output = TOKEN_OUTPUT if TOKEN_OUTPUT in names else names[0]
        self.pooling = pooling

    def run(self, rows, pad_id=0, pad_ids=None):
        """Pooled float64 vectors for token-id rows, right-padded to the longest."""
        n = max(len(r) for r in rows)
        ids = np.full((len(rows), n), pad_id, dtype=np.int64)
        mask = np.zeros((len(rows), n), dtype=np.int64)
        for i, r in enumerate(rows):
            ids[i, :len(r)] = r
            mask[i, :len(r)] = 1
            if pad_ids is not None and len(r) < n:
                ids[i, len(r):] = pad_ids[:n - len(r)]
        feed = {"input_ids": ids, "attention_mask": mask}
        if "token_type_ids" in self.inputs:
            feed["token_type_ids"] = np.zeros_like(ids)
        if "position_ids" in self.inputs:
            feed["position_ids"] = np.broadcast_to(np.arange(n, dtype=np.int64), ids.shape).copy()
        for i in self.session.get_inputs():
            if i.name.startswith("past_key_values."):
                # a decoder exported with a KV cache (transformers.js style):
                # run it as an encoder, with an empty cache
                heads, dim = i.shape[1], i.shape[3]
                feed[i.name] = np.zeros((len(rows), heads, 0, dim), dtype=np.float32)
        out = self.session.run([self.output], {k: v for k, v in feed.items() if k in self.inputs})[0]
        out = out.astype(np.float64)
        if self.pooling is None:
            return list(out)
        return [pool(out[i], mask[i], self.pooling) for i in range(len(rows))]


def gate(runner, cases, pad_id_range, report=print):
    """The fp32, batch and pad gates; returns the numbers, raises on failure."""
    rng = np.random.default_rng(0)
    single = [runner.run([c.ids])[0] for c in cases]
    fp32 = [cosine(v, c.ref) for v, c in zip(single, cases)]
    batched = runner.run([c.ids for c in cases])
    batch = [cosine(a, b) for a, b in zip(batched, single)]
    longest = max(c.n for c in cases)
    pads = rng.integers(*pad_id_range, size=longest)
    padded = runner.run([c.ids for c in cases], pad_ids=pads)
    pad = [cosine(a, b) for a, b in zip(padded, batched)]
    t = time.perf_counter()
    for c in cases:
        runner.run([c.ids])
    ms = (time.perf_counter() - t) * 1000 / len(cases)
    result = {"n": len(cases), "fp32_min": min(fp32), "fp32_worst_case": cases[int(np.argmin(fp32))].label,
              "batch_min": min(batch), "pad_min": min(pad), "ms_per_case": ms}
    report(f"onnx fp32 gate: n={len(cases)} min cosine vs the checkpoint {result['fp32_min']:.7f} "
           f"(case {result['fp32_worst_case']}); batch {result['batch_min']:.8f}; pads {result['pad_min']:.8f}; "
           f"{ms:.1f} ms/case unbatched")
    failures = []
    if not (result["fp32_min"] >= FP32_COS):
        failures.append(f"cosine vs the checkpoint {result['fp32_min']:.7f} < {FP32_COS}")
    if not (result["batch_min"] >= BATCH_COS):
        failures.append(f"batch invariance {result['batch_min']:.8f} < {BATCH_COS}")
    if not (result["pad_min"] >= BATCH_COS):
        failures.append(f"pad invariance {result['pad_min']:.8f} < {BATCH_COS}")
    if failures:
        raise GateFailure("onnx: " + "; ".join(failures))
    return result


def manifest_text(coreml_text, pooling, output):
    """The installed manifest for the ONNX artifact, from the committed
    Core ML one: the same model (tokenizer, dims, prefixes, Matryoshka,
    max_seq_len, source), with backend, artifact, pooling and [io] set for
    ONNX and no buckets. Provisional until D40 settles the format."""
    out, section, header = [], None, True
    for line in coreml_text.splitlines():
        s = line.strip()
        if header and (s.startswith("#") or not s):
            continue                   # the Core ML artifact's description
        header = False
        if s.startswith("["):
            section = s
        key = s.split("=", 1)[0].strip() if "=" in s and not s.startswith("#") else None
        if section is None and key == "backend":
            line = 'backend = "onnx"'
        elif section is None and key == "artifact":
            line = 'artifact = "model.onnx"'
        elif section is None and key == "pooling":
            line = f'pooling = "{pooling or "none"}"'
        elif section is None and key == "buckets":
            continue
        elif section == "[io]" and key == "output":
            line = f'output = "{output}"'
        out.append(line)
    intro = ("# Exported for ONNX Runtime's CPU backend by the conversion library\n"
             "# (tools/sidekick_convert/onnx_export.py): one fp32 model.onnx with dynamic batch and\n"
             f"# sequence axes; {'the server pools its ' + output + ' (' + pooling + ')' if pooling else 'pooled in the graph'}.\n\n")
    return intro + "\n".join(out) + "\n"


def run(job, install_dir, report=print):
    """Export, gate and install one embedder as model.onnx + manifest.toml."""
    install_dir = Path(install_dir).expanduser()
    install_dir.mkdir(parents=True, exist_ok=True)
    pooling = pooling_of(job.head)
    if pooling is None and not hasattr(job.head, "onnx_module"):
        raise SystemExit(f"{job.name}: the {type(job.head).__name__} head has no ONNX form yet")
    token_types = any(p.name == "token_type_ids" for p in job.ports)
    if pooling is not None:
        module, output = TokenLevel(job.backbone, token_types).eval(), TOKEN_OUTPUT
    else:
        module, output = job.head.onnx_module(job.backbone).eval(), POOLED_OUTPUT
    example = job.evaluation.cases[0]
    ids = torch.tensor([example.ids, example.ids], dtype=torch.long)
    inputs = {"input_ids": ids, "attention_mask": torch.ones_like(ids)}
    if token_types:
        inputs["token_type_ids"] = torch.zeros_like(ids)
    with tempfile.TemporaryDirectory() as work:
        path = Path(work) / "model.onnx"
        t = time.perf_counter()
        export(module, inputs, output, path)
        report(f"{job.name}: exported in {time.perf_counter() - t:.1f}s; pooling {pooling or 'in the graph'}; "
               f"output {output}")
        result = gate(Runner(path, pooling), job.evaluation.cases, job.gates.pad_id_range, report)
        for f in Path(work).iterdir():
            dest = install_dir / f.name
            if dest.exists():
                dest.unlink()
            shutil.move(str(f), dest)
    for source, name in job.install_files:
        if name == "manifest.toml":
            (install_dir / name).write_text(manifest_text(Path(source).read_text(), pooling, output))
    (install_dir / "onnx_gates.json").write_text(json.dumps(result, indent=2) + "\n")
    report(f"installed model.onnx and manifest.toml -> {install_dir}")
    return result
