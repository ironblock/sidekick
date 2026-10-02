"""Convert AgentJev (aimeigaoshou/agent-jev) into Core ML artifacts for
sidekick's `POST /v1/classify`, in the agentjev zero-shot format
(docs/design/classify.md, "The agentjev format").

Produces one static-shape .mlmodelc per sequence-length bucket that takes
`input_ids`, `attention_mask`, `seg`, `position_ids` [1, S] and `cand_end`
[1, 32], and outputs `logits` [1, 32]: one per candidate slot, unused slots
at -1e4. sidekick lays a question out as a tree (the state and question
once, then each candidate as its own branch) and the graph scores every
candidate in one pass.

Usage:
    python tools/convert_agentjev.py <checkpoint-dir> <install-dir> [buckets...]
        [--tokenizer-sha256 HEX] [--time]

    checkpoint-dir: local snapshot of aimeigaoshou/agent-jev (config.json,
                    model.safetensors, tokenizer.json, temperatures.json)
    install-dir:    classifier directory the daemon scans, named after the
                    classifier id: the manifest is copied from
                    examples/classifiers/<name>/classifier.toml, e.g.
                    "~/Library/Application Support/sidekick/models/agent-jev"
    buckets:        default: the manifest's

Requires: torch, transformers, tokenizers, coremltools, numpy, safetensors
(arm64-native Python), plus Xcode for `xcrun coremlcompiler`. AgentJev's own
code isn't needed: the head is rebuilt from the checkpoint's tensors, and
both it and the input layout were checked against AgentJev's code (the
token-id fixture, tools/classifier_reference.py).

The recipe (tools/sidekick_convert; docs/CONVERTING.md):
- backbones.qwen3.load_tree: transformers' Qwen3Model loaded from the
  checkpoint's path_encoder.backbone.* tensors; the tree mask built
  in-graph from `seg` (techniques.masks.tree); RoPE from constant cos/sin
  tables selected by a one-hot of `position_ids`, since fp16 can't hold
  angles near 2,047 radians; explicit attention (sdpa with a scale);
- heads.agentjev: the candidates' last states selected by a one-hot of
  `cand_end`, then AgentJev's set transformer (written out) and scorer;
- a bucket-invariant softmax (qwen3.matmul_softmax, as fev's constraint
  D): every attention computes exp(w - rowmax) and one matmul against
  [V | 1]. Core ML's own softmax, after the in-graph score matmul, rounds
  differently on the CPU for different key lengths below 1,024
  (tools/repro_cpu_softmax_length.py), so the same input differed between
  the 512 and 1,024 buckets by up to |dp| 0.019; the matmul form is
  bit-identical across them, at the same accuracy against fp32;
- no precision or range rewrites: the model is GPU-served, where the
  fp16 graph is already at its ideal-fp16 ceiling, and a 2^-6 RMSNorm
  pre-scale measured worse on every path;
- tokenizer.json copied from the checkpoint (Qwen3-0.6B-Base's) without
  padding or truncation;
- the manifest checked against the checkpoint (format, io, revision,
  sequence limits, temperatures).

Gates, per bucket: the fp32 wrapper reproduces AgentJev's per-path scoring
(each candidate's path encoded on its own, its last token's state, the
head over them) within 1e-3 logits; no fused attention op; and on the path
the manifest serves (CPU_AND_GPU), finite logits, argmax agreement with
fp32 wherever its top-2 margin is >= 0.05, max |dp| <= 0.02 after the
softmax, unused slots at -1e4, and pad invariance (random pad ids). The
ANE and the CPU are reported, not gated, and so is the compute plan: the
fp16 weights (1.2 GB) exceed Core ML's ~1 GiB limit for the ANE.

Measured with a probe of this graph (M1 Max, macOS 27): the fp32 graph
matches AgentJev's per-path scoring within |dp| 5e-7; on the GPU, 76 ms at
512 tokens and 381 ms at 2,048, max |dp| 4.8e-4 against an ideal-fp16
ceiling of 7.3e-4.
"""

import json

import numpy as np
import torch

from sidekick_convert import cli, core, manifest, tokenizer
from sidekick_convert.backbones import qwen3
from sidekick_convert.gates import ClassifierGates
from sidekick_convert.heads.agentjev import AgentJevHead
from sidekick_convert.wrapper import compose

PREFIX = "path_encoder.backbone."   # the backbone's tensors in model.safetensors
KMAX = 32
PAD_LOGIT = -1e4

_DIFF = ("--- a/auth/token.py\n+++ b/auth/token.py\n@@ -41,7 +41,7 @@ def verify_token(token, now):\n"
         "-    if token.expires_at < now:\n+    if token.expires_at <= now:\n         raise ExpiredToken(token.id)\n")
_LOG = "Tests run: 14, Failures: 1 - test_expired_token (AssertionError: expected ExpiredToken)\n"
# The converter's own gate requests, never the graded corpus: (name, state,
# question, question type, labels). Lengths are reached by repeating the
# state, so every bucket has cases that land in it.
GATES = [
    ("done-short", "Fix the expired-token failure in UserAuthService.verifyToken. " + _LOG,
     "Does this implementation meet the stated requirements?", "noul",
     ["false: A requirement is still unmet.", "true: The requirement is met."]),
    ("action", "Task: repair a data parser. Two tests fail; the trace points at a boundary condition. " + _DIFF,
     "Which candidate action is useful?", "choice",
     ["read: Read source lines 1 through 40 in parser.py", "test: Run the full test suite",
      "done: Declare the task finished", "ask: Ask the user which behavior is intended"]),
    ("risk", "The patch rewrites the retry loop and removes a lock around the shared cache. " + _DIFF,
     "How risky is merging this patch?", "score",
     ["none", "low", "moderate", "high", "severe"]),
    ("k32", "File this ticket under the right queue: the printer on floor 3 jams again.",
     "Which queue?", "choice", [f"queue {i}: tickets about office subsystem {i}" for i in range(32)]),
    ("medium", ("Task: make the token check strict at the boundary. " + _DIFF + _LOG) * 6,
     "Has the task been completed (full test suite passing)?", "noul", ["false", "true"]),
    ("long", ("Task: make the token check strict at the boundary. " + _DIFF + _LOG) * 13,
     "Which candidate action is useful?", "choice",
     [f"step {i}: inspect the expiry check in module {i} and rerun its tests" for i in range(12)]),
    ("mid-choice", ("Task: make the token check strict at the boundary. " + _DIFF + _LOG) * 8,
     "Which candidate action is useful?", "choice",
     ["read: Read the expiry check", "test: Run the auth tests", "done: Declare the task finished"]),
    ("longest", ("Task: make the token check strict at the boundary. " + _DIFF + _LOG) * 21,
     "How risky is merging this patch?", "score",
     ["none", "low", "moderate", "high", "severe"]),
]


def candidates(question_type, labels):
    """docs/design/classify.md: a choice label's description, else the label;
    a score label as given; a noul question's descriptions, else FALSE/TRUE."""
    if question_type == "choice":
        return [l.split(": ", 1)[1] if ": " in l and l.split(": ", 1)[1] else l.split(": ", 1)[0] for l in labels]
    if question_type == "score":
        return list(labels)
    return [(l.split(": ", 1)[1] if ": " in l else "") or default for l, default in zip(labels, ("FALSE", "TRUE"))]


def tree(tok, state, question, cands):
    """The format's tree: (prefix, [suffix per candidate]), each fragment
    tokenized on its own without special tokens."""
    enc = lambda text: tok.encode(text, add_special_tokens=False).ids  # noqa: E731
    prefix = enc(state if state.startswith("[STATE]") else "[STATE] " + state) + enc("\n[QUESTION] " + question)
    return prefix, [enc("\n[CANDIDATE] " + c) for c in cands]


class TreeCase(core.Case):
    """A Case whose seg and position_ids are per-token: padded per bucket
    (segment -1, position 0) as the runtime pads them."""

    def feed(self, seq, ports, pad_ids=None):
        n = len(self.ids)
        plain = [p for p in ports if p.name in ("input_ids", "attention_mask")]
        out = super().feed(seq, plain, pad_ids)
        for name, pad in (("seg", -1), ("position_ids", 0)):
            x = np.full((1, seq), pad, dtype=np.int32)
            x[0, :n] = self.extra[name]
            out[name] = x
        out["cand_end"] = np.asarray(self.extra["cand_end"], dtype=np.int32).reshape(1, KMAX)
        return {p.name: out[p.name] for p in ports}


def per_path(backbone, head, prefix, suffixes):
    """AgentJev's own scoring: each candidate's path on its own, causal and
    unpadded, its last token's state, then the head over the candidates (as
    nn modules, not the traced forms)."""
    with torch.no_grad():
        states = []
        for s in suffixes:
            ids = torch.tensor([prefix + s])
            h = backbone.model(input_ids=ids, attention_mask=torch.ones_like(ids)).last_hidden_state
            states.append(h[0, -1])
        v = torch.stack(states)[None]
        z = head.proj_in(v)
        for layer in head.set_layers:
            z = layer(z)
        return head.scorer(v + head.proj_out(z))[0].numpy()


def main():
    args = cli.parse(__doc__.split("\n\n")[0], default_buckets=None, flags=[
        ("--tokenizer-sha256", {"help": "expected SHA-256 of the installed tokenizer.json"}),
    ])
    model_id = args.install_dir.name
    path = manifest.classifier_path(model_id)
    m = manifest.load(path)
    buckets = args.buckets or m["buckets"]
    tok = tokenizer.load(tokenizer.prepare(args.src, args.install_dir / "tokenizer.json", mode="clean",
                                           expected_sha256=args.tokenizer_sha256))
    backbone, rest = qwen3.load_tree(args.src, tok, PREFIX)
    head = AgentJevHead.load(rest, backbone.hidden_size, kmax=KMAX)
    temperatures = json.loads((args.src / "temperatures.json").read_text())
    manifest.check_agentjev(m, src=args.src, buckets=buckets, backbone=backbone, head=head,
                            temperatures=temperatures)

    cases = []
    for name, state, question, qtype, labels in GATES:
        prefix, suffixes = tree(tok, state, question, candidates(qtype, labels))
        ids, seg, pos, ends = list(prefix), [0] * len(prefix), list(range(len(prefix))), []
        for c, s in enumerate(suffixes, 1):
            ids += s
            seg += [c] * len(s)
            pos += range(len(prefix), len(prefix) + len(s))
            ends.append(len(ids) - 1)
        if len(ids) > max(buckets):
            core.fail(f"gate {name}: {len(ids)} tokens, over the largest bucket {max(buckets)}")
        cand_end = np.full((1, KMAX), -1, dtype=np.int32)
        cand_end[0, :len(ends)] = ends
        cases.append(TreeCase(ids=ids, ref=per_path(backbone, head, prefix, suffixes), label=name,
                              extra={"seg": seg, "position_ids": pos, "cand_end": cand_end}))
    print(f"gate set: {len(cases)} requests, {min(c.n for c in cases)}-{max(c.n for c in cases)} tokens, "
          f"buckets {[core.bucket_of(c.n, buckets) for c in cases]}", flush=True)

    # After the references: the fp32 gate proves the rewrite exact.
    qwen3.matmul_softmax(backbone)
    ports = head.ports()
    make_wrapper, example = compose(backbone, head, ports)
    gates = ClassifierGates(activation="softmax", pad_value=PAD_LOGIT, pad_id_range=(1000, 30000),
                            **ClassifierGates.paths(manifest.served_path(m)))
    job = core.Job(name=model_id, buckets=buckets, ports=ports, output=head.output, make_wrapper=make_wrapper,
                   example=example, evaluation=core.Evaluation(cases), gates=gates,
                   forbid_ops=frozenset({core.FUSED_ATTENTION}) | backbone.forbid_ops,
                   install_files=[(path, "classifier.toml")], landing_required=True, gate_cases="landing",
                   timing=args.time)
    core.run(job, args.install_dir)


if __name__ == "__main__":
    main()
