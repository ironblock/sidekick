# Publishing builds

How sidekick's graded builds are published on Hugging Face (#137), and the
data and tools behind it: the licence audit that decides what may be
published, the checksum ledger that records every build (#136), and the
model-card generator. Nothing here uploads anything; the upload is a
maintainer's step, listed at the end.

## Contents

- [What is published: licence tiers](#what-is-published-licence-tiers)
- [Repositories and their layout](#repositories-and-their-layout)
- [The checksum ledger](#the-checksum-ledger)
- [Model cards](#model-cards)
- [Licence files](#licence-files)
- [Compiled programs, not packages](#compiled-programs-not-packages)
- [Consent for tier 2b: gated repositories](#consent-for-tier-2b-gated-repositories)
- [For fetch](#for-fetch)
- [Publishing a build](#publishing-a-build)

## What is published: licence tiers

Every model in [data/models.toml](data/models.toml) has a verdict in
[data/licences.toml](data/licences.toml): its licence, its base model,
each training dataset with that dataset's licence as its original source
states it (pinned to a revision where the host allows), the terms that
travel with the weights, and a tier.

| tier | what happens | when |
|---|---|---|
| 1 | published | the licence and every training dataset are permissive, and the data is fully disclosed |
| 2a | published, with a notice on the card | training data with non-commercial terms (stated as prominently as a gate would state them), or share-alike, undisclosed or unlicensed data, or synthetic data from an unnamed generator |
| 2b | published behind explicit consent (a gated repository) | terms that must be passed on to everyone who receives the files: today, the Gemma Terms of Use and Prohibited Use Policy |
| 3 | not published; converting it locally stays documented | a licence that forbids redistribution or commercial use, or makes commercial use depend on the user's circumstances |

Two rules decide what counts. A base model's own pretraining data doesn't:
the tier is judged on the data of the training stages that produced these
weights from that base, and the card names the base. And a dataset keeps
its original source's terms when someone re-hosts it under another label.

`tools/licences.py` validates the file, and CI runs it through
`tools/models_table.py --docs --check`. It recomputes the lowest tier each
entry's recorded data allows (travelling terms: 2b; non-commercial,
share-alike, unknown or undisclosed data, or an unnamed generator: 2a;
otherwise 1) and refuses a stated tier that differs, so a verdict can't
drift from its evidence. Tier 3 is the one judgement it can't recompute, so
a tier-3 entry must record the licence terms that put it there.

**Why a gate only for travelling terms.** A gated repository costs every
user something: a Hugging Face account, a click through the terms, and
their username and email address shared with the publisher. That cost is
worth paying when the terms legally must reach every recipient of the
files, as the Gemma Terms of Use require of anyone who redistributes the
model. Training data with non-commercial terms is different: no licence
requires a redistributor to collect anyone's consent, and whether such
terms bind a trained model at all is unsettled. A gate there would buy no
legal protection and cost every user their privacy, so those models are
2a: the card states the datasets and their terms at the top, as
prominently as a gate's terms, says that their reach is unsettled, that
the user decides whether their use is permitted with no assurance from
the publisher, and that the model's own licence is unchanged. (The first
version of this policy gated them as 2b; D50's amendment records the
change.)

No model is tier 1 today: MS MARCO, Natural Questions, SNLI, MultiNLI and
undisclosed corpora are in nearly every model's training data.
LFM2.5-Embedding-350M is tier 3: its licence licenses commercial use only
below a revenue threshold, which a publisher can't vouch for on a user's
behalf. Each model's tier is in licences.toml, and
[MODELS.md](MODELS.md#licences-and-training-data) summarizes them.

## Repositories and their layout

One repository per model per backend, under the
[`sidekickd`](https://huggingface.co/sidekickd) Hugging Face organization:
`sidekickd/<model id>-coreml` and `sidekickd/<model id>-onnx`
([data/publishing.toml](data/publishing.toml)'s `owner`). The model id is
sidekick's (models.toml), so a client finds the repository from the id it
already uses. Uploads use the maintainer's own token, which needs write
access to the organization.

```
sidekickd/bge-m3-zeroshot-v2.0-c-coreml
  README.md              the model card (tools/model_card.py)
  fp16/                  one build: exactly the directory sidekick installs
    classifier.toml        (or manifest.toml)
    tokenizer.json
    model_128.mlmodelc/    one compiled program per bucket
    model_256.mlmodelc/
    ...
  fp16.sha256            every file of fp16/, `shasum -a 256 -c` format
  NOTICE, TERMS.md, ...  where the licence or tier requires them
```

- **Every quantization and variant of a model is a directory** in the same
  repository, named for its variant in the ledger: `fp16/`, `int8/`,
  `fp16-max2048/`. Each is a complete model directory; a user downloads
  one.
- **Buckets and chunks keep their install names**, `model_<len>.mlmodelc`
  and `model_<len>.<chunk>.mlmodelc` (D37), rather than a
  `buckets/<len>/chunk-N` tree: the published tree is the install tree
  (D49), so nothing is renamed between the ledger, the repository and the
  models directory.
- **A shared weights file (D49) is uploaded at every path it occupies.**
  The Hub stores files by content, so one file at three paths should be
  stored once. The first upload checks that (below); if it isn't, the build
  is uploaded with each distinct file once, at its first path, and fetch,
  which already follows the ledger's path list, is unchanged.
- **ONNX repositories hold sidekick's own exports.** Where the model's
  publisher ships an export of its own that grades A, the card lists it as
  an alternative at a pinned revision and sha256
  ([data/publishing.toml](data/publishing.toml)).
- **Builds are never overwritten.** A converter, coremltools or OS change
  that moves a build's checksums is a new ledger entry and a new commit to
  the repository, tagged with the sidekick release that publishes it; the
  old commit stays reachable at its revision, which the old ledger entry
  records.

## The checksum ledger

[data/ledger.json](data/ledger.json) records every build sidekick converts,
published or not, including builds whose licence keeps them off the Hub.
Reconverting a model and comparing it with its entry shows whether a
converter, coremltools, Xcode or macOS change moved it, and which layer.

One entry per build:

| field | what it holds |
|---|---|
| `id` | `<model>/<backend>/<variant>/<digest[:12]>` |
| `model`, `source` | the models.toml id, and `repo@revision` it was converted from |
| `backend`, `variant`, `quantization` | `coreml` or `onnx`; the build's directory name in its repository; `fp32`, `fp16`, `fp16 weights` or `int8` |
| `buckets`, `chunks`, `max_seq_len` | the build's shape, from its programs and manifest |
| `layout` | `shared` when a weights file is held at several paths (D49), else `separate` |
| `digest` | sha256 over every (path, file sha256) pair in path order: the build's identity, however its files are stored |
| `fingerprint` | the build fingerprint tools/mteb_transfer.py records, from the same hashes |
| `produced` | converter and its commit, coremltools, PyTorch, Python, OS, Xcode, machine, conversion date; `null` where nothing recorded it |
| `files` | each distinct file once: sha256, size, and every path that holds it |
| `grades` | the parity reports that graded these exact files: path, chip, OS, letter, report |
| `published` | each upload: repository, revision, date, sidekick release |
| `evidence`, `recorded`, `note` | `hashed` (from the files) or `sha256sums` (from an archived listing); when; anything else |

**Why JSON, not TOML.** The ledger is written by tools, never by hand:
`ledger.py add` and the converters (next) append to it. Python's standard
library reads TOML but can't write it, and the repository's tools are
standard-library only; JSON round-trips exactly, diffs line by line with one
key per line, and is what fetch (Rust, serde) reads most simply.

**Grades by hash.** `ledger.py add --reports DIR` reads parity reports
and attaches the grades of every report that graded this build's files. A
Core ML report records a 12-hex-digit sha256 prefix of each bucket's
weights, and matches when its buckets are exactly this build's with the
same weights; an ONNX report records the full sha256 of the graph, its
external weights files and the tokenizer, and matches when every one is
this build's file at the same path. By default only reports
[data/runs.toml](data/runs.toml) describes count, the ones whose grades are
published; `--any-report` also takes a release candidate's own. So a card
can say which grades belong to the very files it lists, rather than to an
earlier build of the same model.

**Tooling** (`tools/ledger.py`, standard library only):

```sh
python3 tools/ledger.py add <model dir> --reports <reports dir> [--variant fp16-max2048] \
    [--converter-commit SHA --python 3.12.13 --os "macOS 27.0 (26A428)" --xcode "27.1 (27A9275)" \
     --chip "Apple M1 Max" --torch 2.13.0]
python3 tools/ledger.py add --sums <archive>/item.SHA256SUMS --sums-prefix <dir>/ --model ID [--sizes LIST]
python3 tools/ledger.py verify <model dir> [--id BUILD] [--partial]
python3 tools/ledger.py check
python3 tools/ledger.py list [MODEL]
python3 tools/ledger.py published BUILD --repo sidekickd/NAME --revision SHA --sidekick X.Y.Z
```

- `verify` exits non-zero on any difference and names the layer: weights,
  compiled program, compile analytics, graph, tokenizer or manifest, and a
  shared file that split. `--partial` accepts missing paths, for a fetch of
  fewer buckets. It is the check for fetch, for the release gate (a release
  candidate against its entry) and for drift (a reconversion against the
  published entry).
- `check` validates the shape and the references (models.toml, each
  digest and fingerprint against its file list, each entry's buckets
  against its programs); CI runs it through `models_table.py --docs
  --check`.

**What isn't recorded yet.** Builds converted before the ledger existed
have no converter commit, Python version, OS or machine (their notes say
so); coremltools, PyTorch and the conversion date come from the compiled
programs' metadata, and an ONNX graph names the PyTorch that exported it
(`--torch`). The converters should call `ledger.py add` themselves, with
the commit they ran at; that is a follow-up. Every converter can now keep
the `.mlpackage` sources it compiled (`--keep-packages DIR`, beside the
release candidate, not published), so the package hashes #136 asks for
(the weights and spec before compilation) can be taken from them; recording
them in the ledger is a follow-up too.

Some builds were hashed from their directories, others recorded from the
SHA256SUMS and size listings of archived copies (`evidence =
"sha256sums"`), whose local directories were deleted once the archive was
verified; `ledger.py list` shows them all. Builds whose fingerprints match
the MTEB transfer runs in data/mteb_transfer.json tie those runs to
recorded files.

## Model cards

`tools/model_card.py` writes each repository's README.md from committed data
only (standard library):

```sh
python3 tools/model_card.py MODEL [--backend onnx] [--build ID ...] [--ref vX.Y.Z] --out DIR
python3 tools/model_card.py --check    # every card the ledger allows renders, every link resolves (CI)
```

With `--out`, it also writes `<variant>.sha256` per build, a NOTICE where
the licence requires one, and TERMS.md for a tier-2b model. Documents the
licence requires a copy of (licences.toml's `ship_copies`, such as the
Gemma Terms of Use) are fetched from their source at upload.

A card holds only what is particular to its model and its builds.
Everything that is the same for every model lives once in sidekick's docs,
and the card links there, so a correction is made in one place and no card
can drift from it: how fetch works and how to install by hand
([RUNNING.md](RUNNING.md#installing-models)), how to call the model
([API.md](API.md)), what a grade means and the benchmark caveats
([MODELS.md](MODELS.md)), the tier
policy and the gate (this document), and how to reproduce a build
([CONVERTING.md](CONVERTING.md#reproducing-a-published-build)). In order:

| Section | What it holds |
|---|---|
| Front matter | `license` (`other` with `license_name` and `license_link` for a custom licence), `base_model` (the checkpoint converted), `base_model_relation: quantized` (the Hub has no "converted" relation; it uses `quantized` for format conversions such as GGUF at any precision, and the others would claim training that did not happen), `library_name` (`coreml` or `onnx`), `pipeline_tag`, `tags` (with the tier), `datasets` (training sets hosted on the Hub), and for tier 2b the `extra_gated_*` fields below |
| Summary | The model, what kind it is, the backend, and what the repository holds |
| Licence notice | The licence and tier with the model's own reason (2a), or the gate's consent terms (2b), linked to the tier policy |
| Download | One row per build: weights, sequence buckets, max tokens, download size, whether the buckets share one weights file, and each compute path's grade on Apple silicon; then the one `sidekickd fetch` command; for ONNX, the publisher's own export where it grades A |
| Fidelity | One row per compute path and chip: grade, worst cosine (embedders) or worst \|Δp\| (classifiers), and whether the grading report's file hashes match this repository's files |
| Benchmark scores | The original model's headline scores, with their source |
| Training data | Each dataset with its licence and terms, travelling terms, and anything not confirmed from a primary source (collapsed) |
| Original model | The upstream repository, revision, licence, the licence file this repository carries as `LICENSE` (and `NOTICE`) and where it came from, and the base model; what this repository changed (the modified-files notice Apache-2.0 and the Gemma terms ask for) and any other files the licence requires |
| Checksums | Per build, collapsed: its ledger build id, what it was built with, and every distinct file's sha256, size and paths |

Names are written for people: "Neural Engine", not `cpu_and_ne`; "Max
tokens", not `max_seq_len`; "Shared weights: Yes", not `layout =
"shared"`. Machine identifiers (ledger build ids, sha256s, paths, the
fetch command) appear only where a reader copies them, in code
formatting.

**Links are pinned to a release tag.** A card's links into this repository
point at `blob/vX.Y.Z/` (`--ref`, by default the workspace version), not
`main`. A card is published with a sidekick release and its repository
commit is tagged with that release, so it is a versioned artifact, like the
release notes in docs/releases/, which pin their links the same way. The
docs at that tag describe the fetch and the daemon the card's "Needs
sidekick X.Y.Z or later" names, and a heading renamed after the release
can't break a published card. The cost: a card is staged on the release's
version bump and the tag is cut before the repository is made public, or
its links 404 until then. `model_card.py --check` resolves every link
against the working tree, which is what that tag will hold.

## Licence files

Every published repository carries its model's licence text at its root as
`LICENSE`, as MIT ("this permission notice shall be included in all copies")
and Apache-2.0 (Section 4(a), "a copy of this License") require of a
redistributor, and an Apache-2.0 model's upstream NOTICE file as `NOTICE`
(Section 4(d)).

The texts are committed in [data/licence-texts/](data/licence-texts/), byte
for byte as their source publishes them, each with its source pinned to a
revision and its sha256 in `index.toml`; licences.toml names each model's
`licence_text` and `notice_text`, and `tools/licences.py` refuses an MIT or
Apache-2.0 model without one, or a text whose hash moved.
`tools/model_card.py --out` writes them, and the card's "Original model"
table names the file and its source. Where each text comes from, in order:

1. **The model's own repository,** at the pinned revision: Lumma-fev-0.1b's
   LICENSE and NOTICE (the only registry model whose repository has either).
2. **The repository its licence sources cite,** pinned: FlagEmbedding's
   LICENSE for bge-small-en-v1.5 (MIT, with its copyright line), laya's for
   laya-en and laya-typed-decisions, agent-jev's own. None of them has a
   NOTICE file.
3. **The licence's publisher,** for a model whose repository publishes no
   licence file: the Apache Software Foundation's Apache-2.0 text (pinned by
   sha256, since the page has no revisions), or the SPDX License List's MIT
   text. MIT's text begins with a copyright line, and these models publish
   none, so the shipped LICENSE says so in a header and names "the authors
   of <model>" rather than inventing a year or holder.

**Other licences in the registry:**
- **The Gemma Terms of Use** (embeddinggemma-300m) require a copy of the
  terms with every copy and a NOTICE with a prescribed sentence. The NOTICE
  is generated (`notice`); the terms and the Prohibited Use Policy are
  fetched at upload (`ship_copies`), since ai.google.dev pages can't be
  pinned. Committing their text here, like the others, is open.
- **EmbeddingGemma 2** is Apache-2.0 and ships the ASF text; its card's
  Prohibited Use Policy requirement is part of its gate's terms.
- **The LFM Open License v1.0** (lfm2.5-embedding-350m) also requires a
  copy with every redistribution, but the model is tier 3 and isn't
  published.
- **Base models' licences** aren't shipped: a model's own licence is the
  one its weights are distributed under. Most bases are MIT or Apache-2.0
  too; whether a fine-tune must also carry its base's notice is a question
  for the maintainer.

## Compiled programs, not packages

**Decision: ship compiled `.mlmodelc` programs, not `.mlpackage` to compile
on install.**

- **It is what was graded.** sidekick loads `.mlmodelc`. A package compiled
  on the user's Mac yields a program sidekick never graded and whose hash
  nobody can check in advance: the compiled form can legitimately differ
  across Xcode and macOS versions (#136). Shipping the compiled programs
  means the ledger's hashes and the parity grades are about the very files
  a user runs.
- **Compiled programs travel.** The ledger matches parity reports to files
  by hash: eight of its builds, made by the converters on Apple silicon,
  were also graded from the same compiled files on an Intel Mac Pro under
  macOS 26.6.2. That is another architecture, and an older macOS than the
  one the October builds were graded on, and almost certainly built on
  (macOS 27.0). Every one loaded and ran there; the grades that fell there
  are the Intel GPU's numerics (MODELS.md, Intel Macs), not the format. A
  program's floor is its specification version, recorded in its
  `metadata.json` (macOS 15 for sidekick's iOS 18 opset), below sidekick's
  own floor on Apple silicon.
- **Compiling at install buys nothing back.** It needs Xcode's
  `coremlcompiler`, which a stock Mac lacks, or new code in the daemon; it
  doubles the disk while it runs; and it doesn't remove the expensive part,
  the per-device compile on each bucket's first load (D49, #142: 14 s to
  141 s per Qwen3 bucket on the ANE), which happens either way.
- **Core ML can't ship that compile either.** `compileModel` produces the
  same device-independent `.mlmodelc`, and the device-specialized program
  isn't exportable: on the ANE it lives in the system's ANE cache, and a
  cache copied to another home or another executable recompiles there
  (docs/design/coreml-cache.md, "Transplanting a cache"). macOS 27's Core AI framework does
  offer ahead-of-time compilation (`xcrun coreai-build compile`, one
  `.aimodelc` per device architecture), from its own `.aimodel` format;
  #163 evaluates it as a backend beside Core ML. Until then, `.mlmodelc`
  is the right form for Core ML.
- **The risk, and its guard.** Should a future macOS refuse an older
  compiled program, the release gate's regrade on that macOS would show it,
  and the model is reconverted and republished as a new build, or its
  release candidate's packages, kept with `--keep-packages`, are
  recompiled. Shipping both forms would double every download for that
  contingency.

## Consent for tier 2b: gated repositories

**Decision: a tier-2b repository, one whose terms must be passed on to
every recipient (the Gemma models), is a Hugging Face gated repository with
automatic approval.** Access is granted as soon as a signed-in user accepts
the terms; nobody reviews requests, because a reviewer would have nothing
to check. The upload sets `gated="auto"` on the repository; the card's front
matter words the request:

```yaml
extra_gated_heading: "Accept the terms to download embeddinggemma-300m"
extra_gated_prompt: |-
  By requesting access you agree that:

  1. You receive and use these files under the [Gemma Terms of Use](https://ai.google.dev/gemma/terms), and you will give a copy of it to anyone you give them to.
  2. You won't use these files for anything the [Gemma Prohibited Use Policy](https://ai.google.dev/gemma/prohibited_use_policy) prohibits, and you will pass it on to anyone you give them to.
  3. These files are a conversion of [google/embeddinggemma-300m](...), licensed to you under its licence (Gemma Terms of Use), and are provided as is, without warranty of any kind.
  4. Hugging Face shares your account's username and email address with this repository's owner when you accept, and records that you accepted.
extra_gated_fields:
  "I have read the terms above and accept them": checkbox
extra_gated_button_content: "Accept and request access"
```

**What the user agrees to** is generated from the audit, one item per
concern, so it says exactly what applies to that model and nothing else:
- **Non-commercial training data, where a gated model has it too:** which
  datasets, under which terms; that whether such terms bind a trained model
  is unsettled; and that the user decides whether their use is permitted,
  with no assurance from the publisher. (On a tier-2a card the same four
  points are a notice, with nothing to accept.)
- **Terms that travel with the weights** (the Gemma Terms of Use): that
  the user receives the files under those terms and gives a copy to
  anyone they pass the files to; and for a use policy (the Gemma
  Prohibited Use Policy), that they won't use the files for what it
  prohibits, and pass it on.
- **Always:** that the files are a conversion of the named checkpoint, under
  its licence, provided as is; and that Hugging Face shares the user's
  username and email address with the repository's owner and records the
  acceptance. The gate asks for nothing else: no name, company or country.

The same text is published as TERMS.md beside the card, so a client can
show it and record exactly what was accepted. `sidekickd fetch` shows
TERMS.md and downloads nothing more until the user names the model with
`--accept-terms <model>`, and records the sha256 of the terms accepted in
the installed `fetch.toml`. A gated repository also needs a Hugging Face
token from an account that accepted the gate, which fetch uses through the
`hf` CLI; a refused download tells the user to accept the terms on the
repository's page, which fetch can't do for them.

The gate is honest about its limits: it records consent, it doesn't
enforce use, and a copy made after download isn't gated.

## For fetch

`sidekickd fetch <model> [--backend coreml|onnx] [--variant V]
[--buckets 128,512] [--accept-terms <model>]` (D51) relies on this layout:
- **The repository and revision** come from the ledger's latest
  `published` record for the model and backend (`--repo` and `--revision`
  name others); the ledger is built into sidekickd from
  `docs/data/ledger.json`.
- **The build** is one variant's directory, the install tree as is: the
  one asked for, else the one last published, else the only one. Fetch
  first downloads every variant's manifest and TERMS.md; the buckets asked
  for (all by default) become `hf download --include` patterns for that
  variant's manifest, tokenizer and programs.
- **The files land in the Hugging Face cache,** which stores each distinct
  file once as a blob, so a shared weights file (D49) downloads once
  whatever its paths. How fetch downloads (the `hf` CLI or its own
  downloader) is in [RUNNING.md](RUNNING.md#fetching-a-published-build).
- **Every file is verified** against the Hub's hash for it (the sha256 of
  an LFS or Xet file, the Git blob id of any other) and against the
  ledger: the files must be one of the builds it records for that model,
  backend and variant, with the same sha256 and size at every path, and
  none of the buckets fetched may be missing a file. A mismatch installs
  nothing. `ledger.py verify <cache snapshot>/<variant> --partial` makes
  the same check from Python on the cached files. Run on the installed
  directory, it reports fetch's `fetch.toml`, and for a subset of buckets
  the rewritten manifest, as differences.
- **The model directory is hard links to the cached blobs,** every path of
  it, with the manifest rewritten for exactly the buckets fetched and a
  `fetch.toml` recording the repository, commit and every file's sha256.
  Not symlinks into the snapshot: a program whose files link out of its
  directory can run off the ANE without an error
  ([design/coreml-cache.md](design/coreml-cache.md#symlinks-and-the-fallback-loader)),
  and the registry refuses an ANE-served program with a symlinked file.
  Where the filesystem refuses a link (the cache on another volume),
  fetch copies.
- **The tier:** a repository with TERMS.md (tier 2b) needs `--accept-terms
  <model>` first; tier 3 is never published, so there's nothing to fetch.
- **Then it compiles:** for a Core ML build the daemon will serve, fetch
  loads each installed bucket once, so that Core ML's compile cache holds
  it before the first request (D52); `--no-warm` skips this.

## Publishing a build

Not automated yet; for the first uploads:

1. The model's tier is 1, 2a or 2b in licences.toml, and its build passes
   the release gate (graded on every path it serves).
2. `ledger.py add <build> --reports <reports>`: the build is in the ledger
   with its grades, and `ledger.py verify` passes on the directory to be
   uploaded.
3. `model_card.py MODEL --build ID --ref vX.Y.Z --out <staging>`, with the release that publishes it (its links point at that
   tag). It writes the card, the checksum files, `LICENSE` and any
   `NOTICE` ([Licence files](#licence-files)); then add the documents
   `ship_copies` names and the files `ship_files` names (from the source
   repository at its pinned revision).
   Cut the sidekick release before the repository is made public.
4. Upload `<staging>` and the build directory (as `<variant>/`) with the
   Hugging Face CLI, into the `sidekickd` organization with a token that
   has write access to it; for tier 2b, set the
   repository to gated, automatic approval, before the first upload of
   weights.
5. **First upload only:** confirm the Hub stored a shared weights file once
   (the repository's storage use against the sum of distinct file sizes);
   if not, re-upload with each distinct file at its first path only.
6. Tag the repository commit `sidekick-vX.Y.Z`, and record it:
   `ledger.py published ID --repo ... --revision ... --sidekick X.Y.Z`.
   Then run `tools/models_table.py --docs`, which lists the build in
   MODELS.md's [Published builds](MODELS.md#published-builds); CI fails
   until the list matches the ledger.
