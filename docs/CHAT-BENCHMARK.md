# Chat benchmark: Foundation Models against small open models

How well each model does the short, constrained generation sidekick's chat
route ([`POST /v1/chat/completions`](API.md#chat)) is for: titles, tags,
extraction to a JSON Schema, and short instruction following. Public LLM
benchmarks measure reasoning and coding, which is not that work.

The task set and its gold labels are committed, so a later model, or the
same models on another macOS or Mac, is measured the same way.

## Contents

- [What it measures](#what-it-measures)
- [Models and how they ran](#models-and-how-they-ran)
- [Results](#results)
- [Findings](#findings)
- [Recommendation](#recommendation)
- [Running it](#running-it)
- [Limits](#limits)

## What it measures

The task set is in [`fixtures/tasks/chat/`](../fixtures/tasks/chat/): the
prompts, limits and field-matching rules in `tasks.toml`, the items in one
`.jsonl` file per family, and the extraction schemas in `schemas.json`.
Every item records its source and licence: text written for the benchmark
(MIT) or a Wikipedia lead section pinned to a revision (CC BY-SA 4.0, with
attribution). No item contains real personal data.

| Family | Items | Request | Scored by |
|---|---|---|---|
| Titles | 80: 50 assistant conversations, 30 Wikipedia leads | plain text, "at most 6 words, title only" | ROUGE-L and embedding similarity against 3 references; format compliance |
| Tags, fixed vocabulary | 80 texts, 28-tag vocabulary | JSON `{"tags": [...]}`, 1–3 tags | precision, recall and F1 against 1–3 gold tags; top-1; schema validity |
| Tags, free-form | the same 80 texts | plain text, 3–5 lowercase tags | recall of 3–5 gold concepts (with aliases); on-gold and grounded precision; format |
| Extraction | 80 texts, 8 schemas × 10 | JSON matching a JSON Schema | field accuracy; absent fields left out; schema validity; failures |
| Instruction following | 80 prompts, 95 checks | plain text | each check by program; answers where a prompt has one right answer |

Each request is sent the same way to every model: a system message (none
for instruction following) and one user message, temperature 0,
`stream: true` with usage. The JSON families' prompt names the schema and
asks for only the JSON object. A row either adds `response_format:
json_schema` (guided: the model's decoding is constrained to the schema) or
doesn't (prompt-only), so the two differ by the constraint alone.

**Titles** are the softest metric: a title can be good and share few words
with any reference. ROUGE-L and embedding similarity
([bge-small-en-v1.5](MODELS.md) through sidekick's own `/v1/embeddings`)
take the best of the references but one, averaged over which is left out,
and the table gives each reference scored the same way against the other
two as a ceiling: a human-quality title scores about that, not 100. Read
the title rows as "is it on topic", not as a ranking. Format compliance
(one line, at most 6 words, no quotes, prefix, trailing punctuation or
Markdown) is exact.

**Free-form tags** are scored leniently: a tag matches a gold concept when
one's stemmed words contain the other's, against each concept's aliases.
On-gold precision counts tags that match a concept; grounded precision also
counts a tag whose words all appear in the text.

**Extraction** compares each gold field by the rule `tasks.toml` gives it
(names loosely, codes ignoring spaces and case, numbers within 0.005,
lists by F1), and checks that fields the text doesn't state are left out.
A reply is parsed as bare JSON first, then from a code fence or the first
`{...}` block; "bare JSON" counts the first kind. Schema validity is
against the schema as sent.

**Instruction following** checks are `max_words`, `max_sentences`,
`bullet_count`, `language` (detected with lingua among the nine languages
used), `answer_only` (one of the options), `answer_only_number`,
`lowercase`, `starts_with`, `ends_with`, `forbidden_words`, `max_chars` and
`single_line`.

A refusal (the model's guardrails: HTTP 400 `content_filter`) scores zero,
and the refusal rate is reported per family. Every score has a bootstrap
95% confidence interval over items (2,000 resamples), and each open model's
difference from Foundation Models is paired over items.

Latency is measured from the client: time to the first streamed answer
token (after any thinking), total time, and the output tokens the server
reports, each request after one unrecorded warm-up, one model at a time.

## Models and how they ran

Measured on an M1 Max (32 GB) on macOS 27.0 (build 26A428), sidekick 0.16.0
(commit 305c7bf), October 2026. The run files in
[`docs/data/chat-bench/runs/`](data/chat-bench/runs/) hold every reply and
its timing; [`docs/data/chat-bench/results.json`](data/chat-bench/results.json)
holds the scores below with their intervals.

| Row | Model | Served by | JSON |
|---|---|---|---|
| Foundation Models | the on-device model this Mac has: **AFM 3 Core** (`core3`), 4,096-token context, as `/health` reports | sidekickd's `/v1/chat/completions` | guided (`response_format: json_schema`), and again prompt-only |
| MiniCPM5-1B, MiniCPM5-2B | [openbmb/MiniCPM5-1B](https://huggingface.co/openbmb/MiniCPM5-1B) `87179e5`, [openbmb/MiniCPM5-2B](https://huggingface.co/openbmb/MiniCPM5-2B) `f974000` | mlx_lm.server 0.32.2 (MLX 0.32.4, GPU), bf16 weights as published; `enable_thinking: false` | prompt-only |
| SmolLM3-3B | [HuggingFaceTB/SmolLM3-3B](https://huggingface.co/HuggingFaceTB/SmolLM3-3B) `a07cc9a` | mlx_lm.server, bf16; `enable_thinking: false` (no-think) | prompt-only |
| SmolLM2-360M-Instruct | [HuggingFaceTB/SmolLM2-360M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct) `a10cc15` | mlx_lm.server, bf16 | prompt-only |
| K2-Horizon-0.9B | [IFM/K2-Horizon-0.9B](https://huggingface.co/IFM/K2-Horizon-0.9B) `7c75f2f` | its repository's own PyTorch code (transformers 5.20.2, torch 2.14.1, MPS, bf16) behind `tools/chat_bench/serve_ref.py` | prompt-only |
| MiniCPM5-1B, guided | as above | `serve_ref.py` on MLX, every token masked to the schema's grammar by llguidance 1.9.1 | guided, with a prompt-only control on the same server |
| LFM3-1.7B | [LiquidAI/LFM3-1.7B](https://huggingface.co/LiquidAI/LFM3-1.7B) `ac2586a` | its repository's own PyTorch code (transformers 5.20.2, torch 2.14.1, MPS, bf16) behind `serve_ref.py`; thinking off (`enable_thinking: false`), on, and on capped at 512 tokens | prompt-only, and guided (llguidance) with thinking off and on |
| MiniCPM5-2B on PyTorch | as above | `serve_ref.py` on PyTorch (MPS, bf16), the same stack as LFM3-1.7B; `enable_thinking: false` | prompt-only and guided: the like-for-like control for the LFM3 rows |

K2-Horizon is a reasoning model with no non-thinking mode: its chat template
always opens a thinking block, and its card recommends high effort with
32,768 output tokens. It ran three ways: with an empty thinking block
(the reply prefilled with the block's closing tag); at low effort, where it
closes the block almost at once (one token on average); and at high effort
with thinking capped at 512 tokens, after which the harness closes the block
and lets it answer (budget forcing; 24 of 400 requests reached the cap).
Thinking tokens count in its output tokens and latency.

LFM3-1.7B's chat template opens a `<think>` block by default, and
`enable_thinking: false` closes it empty in the generation prompt. Its rows
and the MiniCPM5-2B controls beside them were measured later in October
2026, on the same Mac and macOS, from sidekick 0.17.0, both models on the
same PyTorch stack. MiniCPM5-2B there scores within a point of its MLX row on
every family, so the stack doesn't move the scores; only the latency
differs. LFM3 ran with thinking off; with thinking on, with 4,096 tokens of
headroom over each family's limit for it (5 of 400 requests still ran out
thinking); and with thinking capped at 512 tokens (budget forcing, as for
K2-Horizon). With thinking off it sometimes thinks anyway, without an
opening tag: a draft, a `</think>`, then the answer again (26 of 400
replies). The scorer takes a reply's answer from after its last closing
tag, as a reasoning parser would.

Decoding is greedy for every open model. Foundation Models gets
`temperature: 0`. mlx_lm.server reuses a cached prompt prefix between
requests (the shared system message), which shortens the open models' time
to first token a little; sidekick's Foundation Models sessions don't share
one between unrelated requests.

macOS 26's Foundation Models (a different model and runtime,
[D21](DECISIONS.md#d21--macos-27-real-usage-and-model-facts-typed-errors-the-shim-stays))
is not measured yet: it needs another Mac. See [Running it](#running-it)
for adding that run.

## Results

<!-- chat-bench:begin -->
<!-- Generated by tools/chat_bench/score.py from docs/data/chat-bench/results.json. -->

**Titles**

| Row | ROUGE-L | Embedding similarity | Format OK % | ≤ 6 words % |
|---|---|---|---|---|
| Foundation Models, guided JSON | 48 (44–52) | 0.84 (0.81–0.86) | 99 (96–100) | 99 (96–100) |
| MiniCPM5-1B | 40 (36–44) | 0.83 (0.81–0.85) | 91 (85–96) | 91 (85–96) |
| MiniCPM5-2B | 46 (42–51) | 0.82 (0.80–0.84) | 100 (100–100) | 100 (100–100) |
| MiniCPM5-2B on serve_ref (PyTorch, MPS), control for the LFM3 rows | 46 (42–51) | 0.82 (0.80–0.84) | 100 (100–100) | 100 (100–100) |
| LFM3-1.7B, thinking off | 41 (37–45) | 0.83 (0.82–0.85) | 46 (35–57) | 49 (38–60) |
| LFM3-1.7B, thinking on | 44 (41–48) | 0.84 (0.83–0.85) | 100 (100–100) | 100 (100–100) |
| LFM3-1.7B, thinking capped at 512 tokens | 45 (42–49) | 0.84 (0.83–0.86) | 96 (91–100) | 96 (91–100) |
| SmolLM3-3B, no-think | 37 (33–41) | 0.84 (0.82–0.85) | 36 (26–48) | 42 (31–54) |
| K2-Horizon-0.9B, empty thinking block | 41 (35–46) | 0.81 (0.79–0.84) | 85 (77–92) | 89 (82–95) |
| K2-Horizon-0.9B, low effort, thinking capped at 256 tokens | 41 (35–46) | 0.81 (0.79–0.84) | 81 (72–89) | 88 (80–94) |
| K2-Horizon-0.9B, high effort, thinking capped at 512 tokens | 43 (38–48) | 0.82 (0.80–0.84) | 100 (100–100) | 100 (100–100) |
| SmolLM2-360M-Instruct | 41 (35–48) | 0.81 (0.78–0.84) | 32 (22–42) | 42 (32–54) |

Reference against the other references: ROUGE-L 35 (33–37), embedding similarity 0.82 (0.81–0.83).

**Tags, fixed vocabulary**

| Row | F1 | Precision | Recall | Top-1 % | Schema-valid % | Off-vocabulary tags % |
|---|---|---|---|---|---|---|
| Foundation Models, guided JSON | 71 (63–78) | 68 (60–76) | 80 (72–87) | 84 (75–91) | 86 (78–94) | 0 (0–0) |
| Foundation Models, prompt-only JSON | 74 (67–81) | 76 (69–84) | 77 (70–84) | 90 (82–96) | 90 (84–96) | 6 (2–11) |
| MiniCPM5-1B | 38 (30–46) | 39 (30–49) | 42 (33–51) | 41 (31–51) | 12 (5–20) | 61 (52–70) |
| MiniCPM5-1B, guided JSON (llguidance) | 37 (31–43) | 30 (24–35) | 54 (45–63) | 40 (30–50) | 100 (100–100) | 0 (0–0) |
| MiniCPM5-1B, prompt-only JSON on serve_ref (control for the guided row) | 36 (28–44) | 38 (30–47) | 40 (31–49) | 40 (30–50) | 12 (5–20) | 64 (55–72) |
| MiniCPM5-2B | 76 (70–82) | 78 (71–84) | 80 (72–86) | 82 (74–90) | 98 (94–100) | 2 (0–6) |
| MiniCPM5-2B on serve_ref (PyTorch, MPS), control for the LFM3 rows | 76 (70–82) | 78 (71–84) | 80 (72–86) | 82 (74–90) | 98 (94–100) | 2 (0–6) |
| MiniCPM5-2B on serve_ref (PyTorch, MPS), guided JSON (llguidance) | 77 (71–83) | 79 (72–85) | 80 (74–87) | 82 (74–90) | 100 (100–100) | 0 (0–0) |
| LFM3-1.7B, thinking off | 79 (72–84) | 86 (79–91) | 78 (72–85) | 89 (81–95) | 81 (72–89) | 17 (9–26) |
| LFM3-1.7B, thinking off, guided JSON (llguidance) | 80 (75–84) | 86 (80–91) | 81 (76–87) | 90 (84–96) | 100 (100–100) | 0 (0–0) |
| LFM3-1.7B, thinking on | 89 (85–93) | 92 (88–96) | 91 (86–95) | 98 (94–100) | 100 (100–100) | 0 (0–0) |
| LFM3-1.7B, thinking on, guided JSON (llguidance) | 89 (85–93) | 92 (88–96) | 91 (86–95) | 98 (94–100) | 100 (100–100) | 0 (0–0) |
| LFM3-1.7B, thinking capped at 512 tokens | 92 (89–95) | 94 (89–97) | 94 (90–97) | 99 (96–100) | 99 (96–100) | 1 (0–3) |
| SmolLM3-3B, no-think | 72 (64–79) | 84 (76–92) | 66 (58–74) | 85 (78–92) | 90 (82–96) | 7 (2–13) |
| K2-Horizon-0.9B, empty thinking block | 56 (47–65) | 57 (47–66) | 59 (50–69) | 61 (51–71) | 14 (6–21) | 65 (57–72) |
| K2-Horizon-0.9B, low effort, thinking capped at 256 tokens | 57 (48–65) | 58 (49–68) | 60 (50–69) | 64 (54–74) | 12 (5–20) | 65 (58–72) |
| K2-Horizon-0.9B, high effort, thinking capped at 512 tokens | 89 (85–92) | 89 (84–93) | 94 (90–97) | 92 (86–98) | 95 (90–99) | 2 (0–5) |
| SmolLM2-360M-Instruct | 0 (0–0) | 0 (0–0) | 0 (0–0) | 0 (0–0) | 0 (0–0) | 100 (100–100) |

**Tags, free-form**

| Row | Concept recall | On-gold precision | Grounded % | Format OK % |
|---|---|---|---|---|
| Foundation Models, guided JSON | 57 (52–61) | 68 (63–73) | 94 (91–97) | 86 (78–93) |
| MiniCPM5-1B | 70 (64–76) | 40 (36–45) | 89 (86–91) | 16 (9–25) |
| MiniCPM5-2B | 31 (26–37) | 44 (37–52) | 58 (50–65) | 82 (74–90) |
| MiniCPM5-2B on serve_ref (PyTorch, MPS), control for the LFM3 rows | 33 (27–38) | 46 (39–53) | 61 (54–69) | 78 (68–86) |
| LFM3-1.7B, thinking off | 45 (41–49) | 59 (55–63) | 71 (67–74) | 68 (57–78) |
| LFM3-1.7B, thinking on | 66 (62–70) | 75 (71–79) | 95 (93–97) | 91 (85–98) |
| LFM3-1.7B, thinking capped at 512 tokens | 65 (61–69) | 76 (72–80) | 94 (92–97) | 62 (52–72) |
| SmolLM3-3B, no-think | 55 (51–61) | 59 (53–65) | 96 (94–98) | 52 (41–64) |
| K2-Horizon-0.9B, empty thinking block | 55 (49–60) | 63 (57–69) | 91 (85–95) | 65 (54–75) |
| K2-Horizon-0.9B, low effort, thinking capped at 256 tokens | 55 (49–60) | 63 (57–70) | 91 (86–95) | 65 (54–75) |
| K2-Horizon-0.9B, high effort, thinking capped at 512 tokens | 59 (55–63) | 64 (59–69) | 93 (90–95) | 60 (49–71) |
| SmolLM2-360M-Instruct | 21 (15–27) | 30 (23–37) | 55 (43–65) | 0 (0–0) |

**Extraction to a JSON Schema**

| Row | Field accuracy (item mean) | Field accuracy (all fields) | Absent fields left out % | Exact items % | Schema-valid % | Bare JSON % | Failed % |
|---|---|---|---|---|---|---|---|
| Foundation Models, guided JSON | 65 (60–70) | 62 (58–67) | 95 (89–99) | 14 (6–21) | 100 (100–100) | 100 (100–100) | 0 (0–0) |
| Foundation Models, prompt-only JSON | 79 (75–84) | 78 (74–82) | 99 (95–100) | 30 (20–40) | 98 (94–100) | 0 (0–0) | 0 (0–0) |
| Foundation Models, guided JSON, all properties required | 97 (94–99) | 98 (96–99) | 22 (12–33) | 61 (50–71) | 100 (100–100) | 100 (100–100) | 0 (0–0) |
| MiniCPM5-1B | 86 (81–90) | 86 (82–90) | 26 (13–42) | 36 (25–46) | 88 (79–94) | 0 (0–0) | 0 (0–0) |
| MiniCPM5-1B, guided JSON (llguidance) | 88 (84–92) | 88 (85–92) | 6 (0–15) | 38 (26–49) | 100 (100–100) | 100 (100–100) | 0 (0–0) |
| MiniCPM5-1B, prompt-only JSON on serve_ref (control for the guided row) | 86 (81–90) | 86 (82–91) | 26 (13–42) | 38 (26–49) | 88 (79–94) | 0 (0–0) | 0 (0–0) |
| MiniCPM5-2B | 89 (84–93) | 90 (86–94) | 87 (76–96) | 54 (44–65) | 100 (100–100) | 100 (100–100) | 0 (0–0) |
| MiniCPM5-2B on serve_ref (PyTorch, MPS), control for the LFM3 rows | 89 (84–93) | 90 (86–94) | 87 (76–96) | 55 (44–65) | 100 (100–100) | 100 (100–100) | 0 (0–0) |
| MiniCPM5-2B on serve_ref (PyTorch, MPS), guided JSON (llguidance) | 89 (84–93) | 90 (86–94) | 87 (76–96) | 55 (44–65) | 100 (100–100) | 100 (100–100) | 0 (0–0) |
| LFM3-1.7B, thinking off | 92 (86–96) | 94 (90–97) | 17 (8–27) | 59 (48–69) | 91 (85–98) | 95 (90–99) | 5 (1–10) |
| LFM3-1.7B, thinking off, guided JSON (llguidance) | 97 (94–99) | 98 (96–99) | 14 (6–23) | 60 (49–70) | 100 (100–100) | 100 (100–100) | 0 (0–0) |
| LFM3-1.7B, thinking on | 93 (87–98) | 95 (91–99) | 84 (73–94) | 81 (72–90) | 94 (88–99) | 94 (88–99) | 6 (1–12) |
| LFM3-1.7B, thinking on, guided JSON (llguidance) | 92 (87–98) | 95 (91–98) | 84 (73–94) | 80 (71–89) | 94 (88–99) | 94 (88–99) | 6 (1–12) |
| LFM3-1.7B, thinking capped at 512 tokens | 96 (92–99) | 97 (94–99) | 44 (30–57) | 70 (60–80) | 95 (90–99) | 81 (72–89) | 2 (0–6) |
| SmolLM3-3B, no-think | 92 (88–96) | 94 (90–97) | 46 (29–66) | 59 (48–70) | 86 (79–94) | 0 (0–0) | 2 (0–6) |
| K2-Horizon-0.9B, empty thinking block | 96 (94–98) | 96 (94–98) | 23 (12–37) | 54 (42–65) | 88 (80–94) | 100 (100–100) | 0 (0–0) |
| K2-Horizon-0.9B, low effort, thinking capped at 256 tokens | 96 (94–98) | 96 (94–98) | 22 (11–35) | 52 (41–64) | 86 (79–94) | 100 (100–100) | 0 (0–0) |
| K2-Horizon-0.9B, high effort, thinking capped at 512 tokens | 95 (93–97) | 95 (93–97) | 90 (83–96) | 69 (59–79) | 95 (90–99) | 100 (100–100) | 0 (0–0) |
| SmolLM2-360M-Instruct | 2 (0–5) | 1 (0–2) | 92 (74–100) | 0 (0–0) | 0 (0–0) | 82 (74–90) | 18 (10–26) |

**Short instruction following**

| Row | All checks pass % | Checks pass % | Answer correct % |
|---|---|---|---|
| Foundation Models, guided JSON | 96 (91–100) | 97 (93–100) | 83 (69–97) |
| MiniCPM5-1B | 88 (80–94) | 88 (82–95) | 38 (21–55) |
| MiniCPM5-2B | 98 (94–100) | 98 (95–100) | 90 (76–100) |
| MiniCPM5-2B on serve_ref (PyTorch, MPS), control for the LFM3 rows | 98 (94–100) | 98 (95–100) | 90 (76–100) |
| LFM3-1.7B, thinking off | 80 (70–89) | 83 (75–91) | 69 (52–86) |
| LFM3-1.7B, thinking on | 99 (96–100) | 99 (97–100) | 93 (83–100) |
| LFM3-1.7B, thinking capped at 512 tokens | 98 (94–100) | 98 (95–100) | 93 (83–100) |
| SmolLM3-3B, no-think | 70 (60–80) | 75 (66–84) | 48 (31–66) |
| K2-Horizon-0.9B, empty thinking block | 80 (71–89) | 83 (76–90) | 59 (41–76) |
| K2-Horizon-0.9B, low effort, thinking capped at 256 tokens | 88 (80–94) | 89 (83–95) | 66 (48–83) |
| K2-Horizon-0.9B, high effort, thinking capped at 512 tokens | 95 (90–99) | 96 (91–99) | 93 (83–100) |
| SmolLM2-360M-Instruct | 65 (55–75) | 68 (60–78) | 38 (21–55) |

**Instruction following by check**

| Check | Foundation Models, guided JSON | MiniCPM5-1B | MiniCPM5-2B | MiniCPM5-2B on serve_ref (PyTorch, MPS), control for the LFM3 rows | LFM3-1.7B, thinking off | LFM3-1.7B, thinking on | LFM3-1.7B, thinking capped at 512 tokens | SmolLM3-3B, no-think | K2-Horizon-0.9B, empty thinking block | K2-Horizon-0.9B, low effort, thinking capped at 256 tokens | K2-Horizon-0.9B, high effort, thinking capped at 512 tokens | SmolLM2-360M-Instruct |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `answer_only` | 100 (n=17) | 94 (n=17) | 100 (n=17) | 100 (n=17) | 94 (n=17) | 100 (n=17) | 100 (n=17) | 59 (n=17) | 100 (n=17) | 100 (n=17) | 100 (n=17) | 88 (n=17) |
| `answer_only_number` | 100 (n=8) | 100 (n=8) | 100 (n=8) | 100 (n=8) | 38 (n=8) | 88 (n=8) | 88 (n=8) | 25 (n=8) | 50 (n=8) | 75 (n=8) | 100 (n=8) | 62 (n=8) |
| `bullet_count` | 100 (n=7) | 86 (n=7) | 100 (n=7) | 100 (n=7) | 100 (n=7) | 100 (n=7) | 100 (n=7) | 100 (n=7) | 100 (n=7) | 100 (n=7) | 100 (n=7) | 86 (n=7) |
| `ends_with` | 100 (n=3) | 100 (n=3) | 100 (n=3) | 100 (n=3) | 100 (n=3) | 100 (n=3) | 100 (n=3) | 100 (n=3) | 100 (n=3) | 100 (n=3) | 100 (n=3) | 33 (n=3) |
| `forbidden_words` | 100 (n=4) | 75 (n=4) | 100 (n=4) | 100 (n=4) | 75 (n=4) | 100 (n=4) | 100 (n=4) | 50 (n=4) | 75 (n=4) | 75 (n=4) | 100 (n=4) | 0 (n=4) |
| `language` | 100 (n=18) | 89 (n=18) | 94 (n=18) | 94 (n=18) | 89 (n=18) | 100 (n=18) | 100 (n=18) | 94 (n=18) | 89 (n=18) | 89 (n=18) | 100 (n=18) | 83 (n=18) |
| `lowercase` | 100 (n=4) | 100 (n=4) | 100 (n=4) | 100 (n=4) | 100 (n=4) | 100 (n=4) | 100 (n=4) | 100 (n=4) | 100 (n=4) | 100 (n=4) | 100 (n=4) | 0 (n=4) |
| `max_chars` | 67 (n=3) | 67 (n=3) | 67 (n=3) | 67 (n=3) | 67 (n=3) | 100 (n=3) | 100 (n=3) | 67 (n=3) | 33 (n=3) | 67 (n=3) | 33 (n=3) | 67 (n=3) |
| `max_sentences` | 100 (n=8) | 88 (n=8) | 100 (n=8) | 100 (n=8) | 88 (n=8) | 100 (n=8) | 100 (n=8) | 75 (n=8) | 75 (n=8) | 100 (n=8) | 100 (n=8) | 75 (n=8) |
| `max_words` | 93 (n=15) | 73 (n=15) | 100 (n=15) | 100 (n=15) | 67 (n=15) | 100 (n=15) | 93 (n=15) | 67 (n=15) | 67 (n=15) | 73 (n=15) | 87 (n=15) | 53 (n=15) |
| `single_line` | 100 (n=4) | 100 (n=4) | 100 (n=4) | 100 (n=4) | 100 (n=4) | 100 (n=4) | 100 (n=4) | 100 (n=4) | 100 (n=4) | 100 (n=4) | 100 (n=4) | 100 (n=4) |
| `starts_with` | 75 (n=4) | 100 (n=4) | 100 (n=4) | 100 (n=4) | 100 (n=4) | 100 (n=4) | 100 (n=4) | 100 (n=4) | 100 (n=4) | 100 (n=4) | 100 (n=4) | 75 (n=4) |

**Refused by guardrails** (% of requests; a refusal scores zero above)

| Row | titles | tags_vocab | tags_free | extraction | instructions |
|---|---|---|---|---|---|
| Foundation Models, guided JSON | 1 (0–4) | 14 (6–22) | 2 (0–6) | 0 (0–0) | 0 (0–0) |
| Foundation Models, prompt-only JSON | – | 1 (0–4) | – | 0 (0–0) | – |
| Foundation Models, guided JSON, all properties required | – | – | – | 0 (0–0) | – |
| MiniCPM5-1B | 0 (0–0) | 0 (0–0) | 0 (0–0) | 0 (0–0) | 0 (0–0) |
| MiniCPM5-1B, guided JSON (llguidance) | – | 0 (0–0) | – | 0 (0–0) | – |
| MiniCPM5-1B, prompt-only JSON on serve_ref (control for the guided row) | – | 0 (0–0) | – | 0 (0–0) | – |
| MiniCPM5-2B | 0 (0–0) | 0 (0–0) | 0 (0–0) | 0 (0–0) | 0 (0–0) |
| MiniCPM5-2B on serve_ref (PyTorch, MPS), control for the LFM3 rows | 0 (0–0) | 0 (0–0) | 0 (0–0) | 0 (0–0) | 0 (0–0) |
| MiniCPM5-2B on serve_ref (PyTorch, MPS), guided JSON (llguidance) | – | 0 (0–0) | – | 0 (0–0) | – |
| LFM3-1.7B, thinking off | 0 (0–0) | 0 (0–0) | 0 (0–0) | 0 (0–0) | 0 (0–0) |
| LFM3-1.7B, thinking off, guided JSON (llguidance) | – | 0 (0–0) | – | 0 (0–0) | – |
| LFM3-1.7B, thinking on | 0 (0–0) | 0 (0–0) | 0 (0–0) | 0 (0–0) | 0 (0–0) |
| LFM3-1.7B, thinking on, guided JSON (llguidance) | – | 0 (0–0) | – | 0 (0–0) | – |
| LFM3-1.7B, thinking capped at 512 tokens | 0 (0–0) | 0 (0–0) | 0 (0–0) | 0 (0–0) | 0 (0–0) |
| SmolLM3-3B, no-think | 0 (0–0) | 0 (0–0) | 0 (0–0) | 0 (0–0) | 0 (0–0) |
| K2-Horizon-0.9B, empty thinking block | 0 (0–0) | 0 (0–0) | 0 (0–0) | 0 (0–0) | 0 (0–0) |
| K2-Horizon-0.9B, low effort, thinking capped at 256 tokens | 0 (0–0) | 0 (0–0) | 0 (0–0) | 0 (0–0) | 0 (0–0) |
| K2-Horizon-0.9B, high effort, thinking capped at 512 tokens | 0 (0–0) | 0 (0–0) | 0 (0–0) | 0 (0–0) | 0 (0–0) |
| SmolLM2-360M-Instruct | 0 (0–0) | 0 (0–0) | 0 (0–0) | 0 (0–0) | 0 (0–0) |

- Foundation Models, guided JSON, over the requests it answered: titles rouge_l 49 (45–52); tags_vocab f1 82 (78–86); tags_free concept_recall 58 (54–62); extraction item_field_acc 65 (60–70); instructions all_checks 96 (91–100).
- Foundation Models, prompt-only JSON, over the requests it answered: tags_vocab f1 75 (68–82); extraction item_field_acc 79 (75–84).

**Difference from Foundation Models (paired over items, points)**

| Row | titles (rouge_l) | tags_vocab (f1) | tags_free (concept_recall) | extraction (item_field_acc) | instructions (all_checks) |
|---|---|---|---|---|---|
| Foundation Models, prompt-only JSON | – | +3 (-5 to +11) | – | +15 (+9 to +20) | – |
| Foundation Models, guided JSON, all properties required | – | – | – | +32 (+27 to +37) | – |
| MiniCPM5-1B | -8 (-12 to -4) | -33 (-44 to -22) | +14 (+7 to +20) | +21 (+15 to +27) | -9 (-16 to -2) |
| MiniCPM5-1B, guided JSON (llguidance) | – | -34 (-43 to -24) | – | +23 (+17 to +29) | – |
| MiniCPM5-1B, prompt-only JSON on serve_ref (control for the guided row) | – | -35 (-46 to -24) | – | +21 (+15 to +27) | – |
| MiniCPM5-2B | -1 (-6 to +3) | +5 (-4 to +15) | -25 (-32 to -19) | +24 (+17 to +30) | +1 (-2 to +5) |
| MiniCPM5-2B on serve_ref (PyTorch, MPS), control for the LFM3 rows | -1 (-6 to +3) | +5 (-4 to +15) | -24 (-30 to -18) | +24 (+17 to +30) | +1 (-2 to +5) |
| MiniCPM5-2B on serve_ref (PyTorch, MPS), guided JSON (llguidance) | – | +6 (-3 to +16) | – | +24 (+17 to +30) | – |
| LFM3-1.7B, thinking off | -7 (-11 to -2) | +8 (-1 to +17) | -12 (-17 to -6) | +27 (+19 to +34) | -16 (-25 to -8) |
| LFM3-1.7B, thinking off, guided JSON (llguidance) | – | +8 (-0 to +19) | – | +32 (+27 to +37) | – |
| LFM3-1.7B, thinking on | -3 (-8 to +1) | +18 (+11 to +27) | +9 (+4 to +15) | +28 (+19 to +36) | +2 (-2 to +8) |
| LFM3-1.7B, thinking on, guided JSON (llguidance) | – | +18 (+11 to +27) | – | +28 (+19 to +36) | – |
| LFM3-1.7B, thinking capped at 512 tokens | -3 (-7 to +2) | +21 (+14 to +29) | +8 (+3 to +14) | +31 (+24 to +37) | +1 (-2 to +6) |
| SmolLM3-3B, no-think | -11 (-15 to -6) | +1 (-7 to +10) | -1 (-8 to +6) | +28 (+21 to +34) | -26 (-36 to -16) |
| K2-Horizon-0.9B, empty thinking block | -7 (-13 to -1) | -16 (-27 to -2) | -2 (-9 to +5) | +31 (+26 to +36) | -16 (-25 to -8) |
| K2-Horizon-0.9B, low effort, thinking capped at 256 tokens | -7 (-13 to -2) | -14 (-26 to -2) | -2 (-9 to +5) | +31 (+26 to +36) | -9 (-18 to -1) |
| K2-Horizon-0.9B, high effort, thinking capped at 512 tokens | -5 (-10 to +0) | +18 (+10 to +25) | +2 (-4 to +8) | +30 (+25 to +36) | -1 (-6 to +2) |
| SmolLM2-360M-Instruct | -7 (-13 to -0) | -71 (-78 to -63) | -36 (-43 to -29) | -63 (-68 to -58) | -31 (-41 to -21) |

**Guided minus prompt-only JSON** (same model and items, points)

| Rows | tags_vocab f1 | tags_vocab schema-valid | extraction item_field_acc | extraction schema-valid |
|---|---|---|---|---|
| Foundation Models, guided JSON − Foundation Models, prompt-only JSON | -3 (-11 to +5) | -4 (-14 to +6) | -15 (-20 to -9) | +2 (+0 to +6) |
| Foundation Models, guided JSON, all properties required − Foundation Models, prompt-only JSON | – | – | +17 (+13 to +22) | +2 (+0 to +6) |
| MiniCPM5-1B, guided JSON (llguidance) − MiniCPM5-1B, prompt-only JSON on serve_ref (control for the guided row) | +1 (-6 to +9) | +88 (+80 to +95) | +2 (-1 to +6) | +12 (+6 to +21) |
| MiniCPM5-2B on serve_ref (PyTorch, MPS), guided JSON (llguidance) − MiniCPM5-2B on serve_ref (PyTorch, MPS), control for the LFM3 rows | +1 (+0 to +2) | +2 (+0 to +6) | +0 (+0 to +0) | +0 (+0 to +0) |
| LFM3-1.7B, thinking off, guided JSON (llguidance) − LFM3-1.7B, thinking off | +1 (-2 to +5) | +19 (+11 to +28) | +5 (+1 to +10) | +9 (+2 to +15) |
| LFM3-1.7B, thinking on, guided JSON (llguidance) − LFM3-1.7B, thinking on | +0 (+0 to +0) | +0 (+0 to +0) | -0 (-1 to +0) | +0 (+0 to +0) |

**Latency** (median time to the first answer token / median total, seconds; mean output tokens, thinking included; decode rate is the median of (tokens − 1) / (total − first token))

| Row | titles | tags_vocab | tags_free | extraction | instructions | Decode tok/s |
|---|---|---|---|---|---|---|
| Foundation Models, guided JSON | 0.61 / 0.68; 10 tok | 1.02 / 1.02; 16 tok | 0.66 / 0.77; 16 tok | 1.65 / 1.65; 56 tok | 0.54 / 0.73; 26 tok | 108 |
| Foundation Models, prompt-only JSON | – | 0.72 / 0.96; 26 tok | – | 0.71 / 1.81; 68 tok | – | 79 |
| Foundation Models, guided JSON, all properties required | – | – | – | 2.23 / 2.23; 78 tok | – | – |
| MiniCPM5-1B | 0.10 / 0.15; 8 tok | 0.09 / 0.25; 27 tok | 0.09 / 0.36; 34 tok | 0.09 / 0.60; 71 tok | 0.07 / 0.20; 31 tok | 128 |
| MiniCPM5-1B, guided JSON (llguidance) | – | 0.10 / 0.21; 15 tok | – | 0.11 / 0.57; 63 tok | – | 126 |
| MiniCPM5-1B, prompt-only JSON on serve_ref (control for the guided row) | – | 0.10 / 0.26; 28 tok | – | 0.09 / 0.60; 70 tok | – | 125 |
| MiniCPM5-2B | 0.22 / 0.29; 4 tok | 0.21 / 0.39; 11 tok | 0.21 / 0.31; 8 tok | 0.18 / 1.08; 48 tok | 0.15 / 0.34; 18 tok | 56 |
| MiniCPM5-2B on serve_ref (PyTorch, MPS), control for the LFM3 rows | 0.31 / 0.44; 4 tok | 0.38 / 0.70; 10 tok | 0.28 / 0.50; 7 tok | 0.38 / 1.96; 47 tok | 0.11 / 0.47; 17 tok | 28 |
| MiniCPM5-2B on serve_ref (PyTorch, MPS), guided JSON (llguidance) | – | 0.37 / 0.67; 10 tok | – | 0.37 / 1.97; 47 tok | – | 30 |
| LFM3-1.7B, thinking off | 0.26 / 0.66; 18 tok | 0.30 / 0.53; 16 tok | 0.23 / 0.46; 10 tok | 0.29 / 2.09; 82 tok | 0.10 / 0.80; 39 tok | 33 |
| LFM3-1.7B, thinking off, guided JSON (llguidance) | – | 0.30 / 0.48; 9 tok | – | 0.29 / 2.06; 72 tok | – | 37 |
| LFM3-1.7B, thinking on | 37.02 / 37.26; 1403 tok, 1395 thinking | 16.03 / 16.34; 738 tok, 726 thinking | 40.59 / 40.91; 1626 tok, 1614 thinking | 27.93 / 29.73; 1484 tok, 1425 thinking | 8.85 / 10.35; 495 tok, 466 thinking | 37 |
| LFM3-1.7B, thinking on, guided JSON (llguidance) | – | 15.91 / 16.18; 738 tok, 726 thinking | – | 28.06 / 29.77; 1485 tok, 1425 thinking | – | 36 |
| LFM3-1.7B, thinking capped at 512 tokens | 14.06 / 14.26; 520 tok, 512 thinking | 14.13 / 14.48; 466 tok, 454 thinking | 14.08 / 14.41; 524 tok, 512 thinking | 14.20 / 16.02; 601 tok, 501 thinking | 8.71 / 10.20; 355 tok, 326 thinking | 37 |
| SmolLM3-3B, no-think | 0.28 / 0.54; 12 tok | 0.27 / 0.42; 10 tok | 0.27 / 0.57; 18 tok | 0.23 / 1.69; 69 tok | 0.18 / 0.76; 41 tok | 45 |
| K2-Horizon-0.9B, empty thinking block | 0.14 / 0.29; 7 tok | 0.19 / 0.57; 23 tok | 0.14 / 0.40; 14 tok | 0.18 / 1.45; 70 tok | 0.07 / 0.39; 30 tok | 48 |
| K2-Horizon-0.9B, low effort, thinking capped at 256 tokens | 0.18 / 0.32; 9 tok, 1 thinking | 0.23 / 0.61; 38 tok, 1 thinking | 0.17 / 0.44; 17 tok, 1 thinking | 0.22 / 1.49; 72 tok, 1 thinking | 0.11 / 0.48; 46 tok, 12 thinking | 53 |
| K2-Horizon-0.9B, high effort, thinking capped at 512 tokens | 4.20 / 4.35; 222 tok, 215 thinking | 4.12 / 4.31; 223 tok, 212 thinking | 3.73 / 4.13; 225 tok, 206 thinking | 7.36 / 8.37; 406 tok, 347 thinking | 3.28 / 3.74; 223 tok, 196 thinking | 50 |
| SmolLM2-360M-Instruct | 0.06 / 0.13; 15 tok | 0.06 / 0.42; 56 tok | 0.05 / 0.14; 25 tok | 0.06 / 0.85; 155 tok | 0.04 / 0.17; 54 tok | 175 |

**Repeatability** (share of replies byte-identical between two runs of the row)

- Foundation Models, guided JSON: 84% of 400 (extraction 20%, instructions 100%, tags_free 100%, tags_vocab 100%, titles 100%); 100% with the same values once JSON is parsed

<!-- chat-bench:end -->

## Findings

**Foundation Models' guided JSON leaves out fields the text states.** A
property missing from the schema's `required` list becomes optional in
guided generation, and the model leaves most optional properties out:
extraction field accuracy is 65% guided against 79% prompt-only on the same
prompt (paired difference −15 points, −20 to −9). Guided replies dropped
the email and job title from most contacts and the currency from 20 of 80
items. With every property required, field accuracy rises to 97%, but the
model then fills the fields the text doesn't state ("Not specified",
"00:00", a guessed currency), and only 22% of them stay empty. Neither form
gets both right; prompt-only JSON, parsed out of the code fence the model
puts it in, does best overall for Foundation Models.

**Guided generation trips the guardrails.** Fixed-vocabulary tagging with
the schema (a 28-value enum) was refused on 11 of 80 texts, as "may contain
sensitive content": the leads of Jazz, the Battle of Hastings, Copyright
and Type 2 diabetes among them. The same prompt without the schema was
refused once. Over the texts it answered, guided tagging scores 82 F1.
`tools/chat_bench/guardrails_probe.swift` reran the refused requests
straight through Foundation Models
([results](data/chat-bench/guardrails-probe.jsonl)): as plain text 10 of the 11 tagging
requests answer, and `SystemLanguageModel(guardrails:
.permissiveContentTransformations)` does not stop the guided refusals. It
does stop the three plain-text refusals (a title for a conversation about
crate-training a puppy, and free-form tags for a tenancy-deposit dispute and
a home-network question), which the default guardrails refused as "may
contain unsafe content".

**Constrained decoding buys validity, not accuracy.** On MiniCPM5-1B,
llguidance made every reply schema-valid (+88 points on tags, +12 on
extraction) but left tag F1 (+1, −6 to +9) and extraction field accuracy
(+2, −1 to +6) where they were: its tag errors are wrong choices, not
malformed JSON. Constrained, it also stopped leaving absent fields out
(6% left out, from 26%).

**Every open model but the floor extracts better than Foundation Models.** Field
accuracy is 86–96% for the open models against 65% (guided) and 79%
(prompt-only). The difference is in what they do with fields the text
doesn't state: MiniCPM5-2B and K2-Horizon at high effort leave 87–90% of
them out, MiniCPM5-1B, SmolLM3 and K2-Horizon without thinking fill most of
them in. Counting both, whole items come out exactly right for 69% of
texts with K2-Horizon at high effort, 54–59% with MiniCPM5-2B and SmolLM3,
and 14–30% with Foundation Models.

**Foundation Models' strength is format discipline.** It keeps to the
title format (99%) and passes 96% of the instruction prompts' checks, every
language, `answer_only` and number check among them, with no prompt
tuning. Of the open models only MiniCPM5-2B matches it (100% and 98%), and
K2-Horizon at high effort comes close (100% and 95%). SmolLM3-3B without
thinking ignores length limits (36% of titles in format, 70% of
instruction prompts). SmolLM2-360M is the floor: it echoes the schema back
instead of extracting, and runs on past the tag limit.

**The small models are sensitive to the prompt's wording.** MiniCPM5-2B
often returned the free-form prompt's own words ("lowercase, one, three")
as tags (31% concept recall), where MiniCPM5-1B, with the same prompt,
recalls 70% but ignores the format. MiniCPM5-1B tags from outside the fixed
vocabulary 61% of the time. Another wording might reorder these rows; the
task set fixes one, so every model faces the same one.

**K2-Horizon needs its thinking.** Without it (an empty block, or low
effort, where it closes the block at once) it extracts well (96%) but tags
(56–57 F1) and follows instructions (80–88%) below Foundation Models. At
high effort, capped at 512 thinking tokens, it is the best row on tags
(89 F1, +18 over Foundation Models), extraction (95%, with 90% of absent
fields left out) and instruction answers (93%), at 4–8 s a request on
PyTorch: 200–350 thinking tokens before each answer.

**LFM3-1.7B beats MiniCPM5-2B only with its thinking, at 10–40 s a
request.** Paired with MiniCPM5-2B on the same PyTorch stack:

- Thinking off, it is behind on titles (46% in format, −54 points, −65 to
  −42) and on the instruction prompts (80% pass, −18, −26 to −9). It is
  level on fixed-vocabulary tags (+2 F1, −4 to +10) and extraction (+3, −3
  to +9), and ahead on free-form tag recall (+12, +6 to +19). It fills in
  most fields the text doesn't state: 17% left out, against 87%.
  Constrained decoding makes its JSON valid and lifts extraction to 97%
  (+8 over MiniCPM5-2B guided, +4 to +12), still filling in absent fields.
- Thinking on, it is ahead on fixed-vocabulary tags (89 F1, +13, +6 to
  +21) and free-form tags (66% recall, +33), level on titles, extraction
  (+4, −3 to +11) and instructions (99%), and leaves 84% of absent fields
  out: 81% of extraction items exactly right, the most of any row. It
  thinks for 470–1,600 tokens on average (up to 4,352) before answering.
  Guided generation changes nothing here: the thinking is the same, and the
  answers after it were already valid JSON.
- Capped at 512 thinking tokens, it keeps the tag scores (92 F1, +16; 65%
  recall, +32) and extraction (96%, +7, +2 to +13), at a median 10–16 s a
  request instead of 10–41 s. Almost every request reaches the cap, and it then
  leaves fewer absent fields out (44%).

**Latency.** Foundation Models starts a plain-text reply in 0.5–0.7 s and
returns guided JSON whole in 1.0 s (tags) to 1.7 s (extraction). On the GPU
under MLX, MiniCPM5-1B starts in 0.07–0.10 s and decodes at about 128
tokens/s, MiniCPM5-2B in 0.15–0.22 s at 56 tokens/s, SmolLM3-3B in
0.18–0.28 s at 45 tokens/s. These are GPU numbers on an idle Mac; serving
beside a GPU model means the Neural Engine, where
[#173](https://github.com/ironblock/sidekick/issues/173) measured
MiniCPM5-1B at 29 tokens/s (fp16) and 54 tokens/s (int8). On PyTorch
(MPS), MiniCPM5-2B decodes at 28 tokens/s and LFM3-1.7B at 33–37, so a
reply that thinks for 1,000 tokens takes about 30 s there. Foundation Models'
streamed decode rate isn't comparable: it streams in bursts.

**Repeatability.** Two Foundation Models runs at temperature 0 gave
byte-identical plain-text replies, but only 20% of the guided extraction
replies were byte-identical: the values were all the same, with the JSON
object's keys in another order each run.

## Recommendation

No open model replaces Foundation Models on every family, and none needs
to: the case for one is extraction, where all but the floor beat it.

- **MiniCPM5-2B is the one to serve beside Foundation Models.** It is level
  with Foundation Models on titles (−1 point, −6 to +3), fixed-vocabulary
  tags (+5, −4 to +15) and instruction following (+1, −2 to +5), well ahead
  on extraction (+24, +17 to +30, leaving 87% of absent fields out), keeps
  to formats as well, and refuses nothing. Its weakness is free-form
  tagging with this prompt. It is 2B parameters (about 5 GB at bf16), so on
  the Neural Engine it needs #173's int8 or chunked path, which is
  unmeasured at this size.
- **K2-Horizon-0.9B at high effort is the most accurate model here**, for
  background work that can wait 4–8 s, but its custom architecture is
  PyTorch-only and its thinking multiplies decode time; it is not the first
  model to convert.
- **MiniCPM5-1B, #173's spike model, is not good enough alone**: fine at
  extraction (86%) but poor at fixed-vocabulary tags (38 F1) and at
  answering short questions (38% correct). Constrained decoding would make
  its JSON valid without making it more accurate.
- **SmolLM3-3B** extracts well (92%) but doesn't keep to formats without
  thinking; **SmolLM2-360M** is not usable for this work.

Foundation Models still does better at keeping to a format and an
instruction (titles, length limits, reply language, answer-only replies),
with nothing to download or load. It does worse at extraction, where
guided generation drops optional fields and prompt-only still trails every
open model but the floor, and its guardrails refuse ordinary texts when output
is guided by a schema. For sidekick's chat route this means:

- for extraction, a schema with optional properties loses fields under
  guided generation; making them all required trades that for invented
  values, and prompt-only JSON does better than both;
- guided tags over a fixed vocabulary are refused about one time in seven,
  and the permissive guardrails don't change that;
- an open model needs constrained decoding for valid JSON, but its accuracy
  is the model's own.

## Running it

The harness is [`tools/chat_bench/`](../tools/chat_bench/). Maintainer-run,
one model at a time, never in CI. Its Python environment is pinned in
[`tools/chat_bench/requirements.txt`](../tools/chat_bench/requirements.txt).

```sh
uv venv --python 3.12 .dev-data/chat-bench/venv
uv pip install --python .dev-data/chat-bench/venv/bin/python -r tools/chat_bench/requirements.txt
PY=.dev-data/chat-bench/venv/bin/python

# Foundation Models: a sidekickd with no other models, then its rows
cargo build --release -p sidekick-server --features coreml,onnx
target/release/sidekickd --models-dir <empty dir> --addr 127.0.0.1:8805 --no-warm &
$PY tools/chat_bench/watchdog.py 3600 <log> tools/chat_bench/run.py afm --base-url http://127.0.0.1:8805 --out <runs dir>
$PY tools/chat_bench/watchdog.py 3600 <log> tools/chat_bench/run.py afm-prompt --base-url http://127.0.0.1:8805 --out <runs dir>

# Each open model: run.py starts its server, runs every request, stops it
$PY tools/chat_bench/watchdog.py 7200 <log> tools/chat_bench/run.py minicpm5-1b --port 8811 --out <runs dir>

# The guardrails probe: rerun a row's refused requests straight through Foundation Models
$PY tools/chat_bench/tasks.py refused <runs dir>/afm.json 6 > <dir>/probes.json
xcrun swiftc -O -parse-as-library tools/chat_bench/guardrails_probe.swift -o <dir>/guardrails_probe
<dir>/guardrails_probe <dir>/probes.json

# Score: a sidekickd serving bge-small-en-v1.5 gives the titles' embedding similarity
$PY tools/chat_bench/score.py --runs <runs dir> --embed-url http://127.0.0.1:8806 --doc docs/CHAT-BENCHMARK.md
```

The rows are defined in
[`tools/chat_bench/rows.toml`](../tools/chat_bench/rows.toml).
`watchdog.py` kills a row when swap passes 4 GB or free memory falls under
15%. A run file records the task set's checksums, the request shape, the
model's revision and serving stack, the sidekick commit, the Foundation
Models variant, and the macOS build and chip; it never records the base
URL. To add a run from another Mac or macOS version, run the rows there
with `--suffix` naming the machine (`afm --suffix -macos26` writes
`afm-macos26.json`), put the file beside the committed runs, and score them
together: each run file is a row of the report, labelled from its own
provenance. `--suffix .repeat` writes a second run of a row, which the
scorer compares with the first for repeatability.

## Limits

- **One Mac, one macOS.** The Foundation Models rows are AFM 3 Core on
  macOS 27.0; macOS 26 and other variants are pending.
- **80 items a family.** Intervals are about ±5–10 points; differences
  smaller than that aren't resolved. The paired differences are tighter
  than the separate intervals suggest.
- **The gold labels were written for this benchmark** by one author and
  checked in one independent pass, which found no clear errors. Judgement
  calls are recorded in the items' `notes`, and three inferred fields score
  either way (`either` in `extraction.jsonl`). Most text is English;
  eighteen instruction prompts test other languages.
- **One wording per prompt.** The small models are sensitive to it (see
  [Findings](#findings)); the rows compare models on the same prompt, not
  each at its best prompt.
- **The open models ran on the GPU, in their reference stacks**, not
  through sidekick or on the Neural Engine. This measures the models, not a
  conversion of them (#173's question). K2-Horizon and LFM3-1.7B ran on
  PyTorch, slower than an MLX build would be, so their latency is an upper
  bound.
- **Latency is client-side over local HTTP**, with mlx_lm.server's prompt
  cache on. During the MiniCPM5-1B guided row and its control, another
  process took memory (free memory fell to 32%, 1 GB of swap); their
  latencies may be a little high, and their replies, being greedy, are
  unaffected.
