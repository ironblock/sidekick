"""A minimal OpenAI-compatible chat server over a model's reference weights,
for the rows mlx_lm.server can't serve: a model whose architecture only its
own PyTorch code implements (torch backend), and guided JSON generation on an
open model (`--constrained`, with llguidance).

    python tools/chat_bench/serve_ref.py --backend torch --model <hf repo or dir> \\
        --revision <commit> --port 8811 [--chat-template-args '{"reasoning_effort": "low"}'] \\
        [--think-tags '<ifm|think_faster>' '</ifm|think_faster>'] [--reasoning-budget 256] \\
        [--prefill '</ifm|think_faster>\\n']
    python tools/chat_bench/serve_ref.py --backend mlx --model openbmb/MiniCPM5-1B --constrained ...

Decoding is greedy (argmax) whatever `temperature` says, one request at a
time, with the KV cache of the backend's own model code. It serves
`POST /v1/chat/completions` (streaming and not, with `stream_options.
include_usage`) and `GET /health`.

- `--constrained` honours `response_format: json_schema`: every content token
  is masked to what the schema's grammar allows (llguidance), as guided
  generation does. Without it, response_format is ignored (prompt-only).
- `--think-tags OPEN CLOSE` names a reasoning model's thinking delimiters.
  Thinking goes out as `delta.reasoning`, the rest as `delta.content`.
  `--reasoning-budget N` closes the thinking after N tokens by appending
  CLOSE and a newline, then lets the model answer (budget forcing).
- `--prefill TEXT` is appended after the chat template's generation prompt:
  with `CLOSE\\n` it starts the reply with an empty thinking block.

Trusts the model repository's own code (`trust_remote_code`) on the torch
backend: run it only on a pinned revision whose code you have read.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import numpy as np


def stop_ids(tokenizer, model_dir: Path) -> set[int]:
    ids: set[int] = set()
    gen = model_dir / "generation_config.json"
    if gen.exists():
        e = json.loads(gen.read_text()).get("eos_token_id")
        ids.update(e if isinstance(e, list) else [e] if e is not None else [])
    if tokenizer.eos_token_id is not None:
        ids.add(tokenizer.eos_token_id)
    vocab = tokenizer.get_vocab()
    for t in ("<|im_end|>", "<|ifm|im_end|>", "<|endoftext|>", "<|eot_id|>"):
        if t in vocab:
            ids.add(vocab[t])
    return ids


class TorchLM:
    def __init__(self, path: Path):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            path, dtype=torch.bfloat16, trust_remote_code=True).to(self.device).eval()
        self.dtype = "bfloat16"
        self.backend = f"transformers {__import__('transformers').__version__}, torch {torch.__version__}, {self.device}"

    def start(self, ids: list[int]):
        torch = self.torch
        with torch.no_grad():
            out = self.model(input_ids=torch.tensor([ids], device=self.device), use_cache=True)
        self.cache = out.past_key_values
        return out.logits[0, -1, :].float()

    def step(self, ids: list[int]):
        torch = self.torch
        with torch.no_grad():
            out = self.model(input_ids=torch.tensor([ids], device=self.device),
                             past_key_values=self.cache, use_cache=True)
        self.cache = out.past_key_values
        return out.logits[0, -1, :].float()

    def argmax(self, logits, mask) -> int:
        if mask is not None:
            from llguidance.torch import apply_token_bitmask_inplace
            logits = logits.unsqueeze(0)
            apply_token_bitmask_inplace(logits, self.torch.from_numpy(mask).to(logits.device))
            logits = logits[0]
        return int(self.torch.argmax(logits).item())


class MlxLM:
    def __init__(self, path: Path):
        import mlx.core as mx
        import mlx_lm
        from mlx_lm import load
        from mlx_lm.models.cache import make_prompt_cache
        self.mx = mx
        self.model, wrapper = load(str(path))
        self.tokenizer = wrapper._tokenizer
        self.make_cache = make_prompt_cache
        self.dtype = "bfloat16 (as stored)"
        self.backend = f"mlx-lm {mlx_lm.__version__}, mlx {mx.__version__}, gpu"

    def start(self, ids: list[int]):
        self.cache = self.make_cache(self.model)
        return self.model(self.mx.array([ids]), cache=self.cache)[0, -1, :]

    def step(self, ids: list[int]):
        return self.model(self.mx.array([ids]), cache=self.cache)[0, -1, :]

    def argmax(self, logits, mask) -> int:
        mx = self.mx
        if mask is not None:
            from llguidance.mlx import apply_token_bitmask
            logits = apply_token_bitmask(logits.astype(mx.float32), mask)[0]
        tok = mx.argmax(logits)
        mx.eval(tok)
        return int(tok.item())


class Generator:
    def __init__(self, a):
        from huggingface_hub import snapshot_download
        path = Path(a.model)
        if not path.exists():
            path = Path(snapshot_download(a.model, revision=a.revision))
        self.model_dir = path
        self.lm = TorchLM(path) if a.backend == "torch" else MlxLM(path)
        self.tok = self.lm.tokenizer
        self.stops = stop_ids(self.tok, path)
        self.template_args = json.loads(a.chat_template_args) if a.chat_template_args else {}
        self.think_open, self.think_close = a.think_tags or (None, None)
        self.budget = a.reasoning_budget
        self.prefill = a.prefill.encode().decode("unicode_escape") if a.prefill else ""
        self.constrained = a.constrained
        self.ll_tok = None
        if a.constrained:
            import llguidance.hf
            self.ll_tok = llguidance.hf.from_tokenizer(self.tok, eos_token=sorted(self.stops))

    def prompt_ids(self, messages: list[dict]) -> tuple[list[int], bool]:
        text = self.tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False,
                                            **self.template_args) + self.prefill
        thinking = bool(self.think_open) and text.rfind(self.think_open) > text.rfind(self.think_close)
        return self.tok.encode(text, add_special_tokens=False), thinking

    def generate(self, messages, max_tokens: int, schema: dict | None):
        """Yield ("reasoning"|"content", text) deltas, then ("done", usage dict)."""
        ids, thinking = self.prompt_ids(messages)
        matcher, mask = None, None
        if schema is not None and self.constrained:
            from llguidance import LLMatcher
            from llguidance.numpy import allocate_token_bitmask, fill_next_token_bitmask
            matcher = LLMatcher(self.ll_tok, LLMatcher.grammar_from_json_schema(
                schema, defaults={"whitespace_flexible": True}))
            mask = allocate_token_bitmask(1, self.ll_tok.vocab_size)
        logits = self.lm.start(ids)
        out: list[int] = []         # all generated tokens
        seg: list[int] = []         # tokens of the current segment (reasoning or content)
        emitted = ""                # text of `seg` already sent
        n_reason, finish, started = 0, "length", False
        while len(out) < max_tokens:
            m = None
            if matcher is not None and not thinking:
                fill_next_token_bitmask(matcher, mask, 0)
                m = mask
            t = self.lm.argmax(logits, m)
            if matcher is not None and not thinking:
                if t in self.stops or matcher.is_stopped():
                    finish = "stop"
                    break
                matcher.consume_token(t)
            elif t in self.stops:
                finish = "stop"
                break
            out.append(t)
            seg.append(t)
            n_reason += thinking
            text = self.tok.decode(seg, skip_special_tokens=False)
            forced: list[int] = []
            if thinking and self.think_close in text:
                head = text.split(self.think_close)[0]
                if len(head) > len(emitted):
                    yield "reasoning", head[len(emitted):]
                thinking, seg, emitted = False, [], ""
                rest = text.split(self.think_close, 1)[1].lstrip("\n")
                if rest:
                    seg = self.tok.encode(rest, add_special_tokens=False)
            elif thinking and self.budget is not None and n_reason >= self.budget:
                if len(text) > len(emitted) and not text.endswith("�"):
                    yield "reasoning", text[len(emitted):]
                forced = self.tok.encode("\n" + self.think_close + "\n", add_special_tokens=False)
                thinking, seg, emitted = False, [], ""
            elif not text.endswith("�") and len(text) > len(emitted):
                kind = "reasoning" if thinking else "content"
                delta = text[len(emitted):]
                if kind == "content" and not started:
                    delta = delta.lstrip("\n")     # the newline after a closed thinking block
                    started = bool(delta)
                if delta:
                    yield kind, delta
                emitted = text
            if matcher is not None and matcher.is_stopped():
                finish = "stop"
                break
            logits = self.lm.step([t] + forced)
        if seg and not thinking:
            text = self.tok.decode(seg, skip_special_tokens=False)
            delta = text[len(emitted):]
            if not started:
                delta = delta.lstrip("\n")
            if delta:
                yield "content", delta
        yield "done", {"prompt_tokens": len(ids), "completion_tokens": len(out),
                       "completion_tokens_details": {"reasoning_tokens": n_reason},
                       "finish_reason": finish}


def make_handler(gen: Generator, model_id: str):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _json(self, status, obj):
            body = json.dumps(obj).encode()
            self.send_response(status)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/health":
                return self._json(200, {"status": "ok", "model": model_id, "backend": gen.lm.backend,
                                        "dtype": gen.lm.dtype, "constrained": gen.constrained})
            self._json(404, {"error": "not found"})

        def do_POST(self):
            if self.path != "/v1/chat/completions":
                return self._json(404, {"error": "not found"})
            req = json.loads(self.rfile.read(int(self.headers["content-length"])))
            schema = None
            rf = req.get("response_format") or {}
            if rf.get("type") == "json_schema":
                schema = rf["json_schema"]["schema"]
            max_tokens = req.get("max_completion_tokens") or req.get("max_tokens") or 512
            cid, created = f"chatcmpl-{uuid.uuid4().hex}", int(time.time())
            stream = req.get("stream", False)
            include_usage = (req.get("stream_options") or {}).get("include_usage", False)

            def chunk(delta, finish=None, usage=None):
                c = {"id": cid, "object": "chat.completion.chunk", "created": created, "model": model_id,
                     "choices": [] if usage else [{"index": 0, "delta": delta, "finish_reason": finish}]}
                if usage:
                    c["usage"] = usage
                self.wfile.write(f"data: {json.dumps(c)}\n\n".encode())
                self.wfile.flush()

            parts = {"reasoning": "", "content": ""}
            if stream:
                self.send_response(200)
                self.send_header("content-type", "text/event-stream")
                self.end_headers()
                chunk({"role": "assistant"})
            for kind, value in gen.generate(req["messages"], max_tokens, schema):
                if kind == "done":
                    usage = value
                    break
                parts[kind] += value
                if stream:
                    chunk({kind: value})
            finish = usage.pop("finish_reason")
            if stream:
                chunk({}, finish)
                if include_usage:
                    chunk(None, usage=usage)
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
                return
            msg = {"role": "assistant", "content": parts["content"]}
            if parts["reasoning"]:
                msg["reasoning"] = parts["reasoning"]
            self._json(200, {"id": cid, "object": "chat.completion", "created": created, "model": model_id,
                             "choices": [{"index": 0, "message": msg, "finish_reason": finish}],
                             "usage": usage})
    return Handler


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", choices=["torch", "mlx"], required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--chat-template-args")
    ap.add_argument("--think-tags", nargs=2, metavar=("OPEN", "CLOSE"))
    ap.add_argument("--reasoning-budget", type=int)
    ap.add_argument("--prefill")
    ap.add_argument("--constrained", action="store_true")
    a = ap.parse_args()
    gen = Generator(a)
    print(f"serve_ref: {gen.lm.backend}; stop ids {sorted(gen.stops)}", flush=True)
    server = HTTPServer((a.host, a.port), make_handler(gen, a.model))
    print(f"serve_ref: listening on {a.host}:{a.port}", flush=True)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
