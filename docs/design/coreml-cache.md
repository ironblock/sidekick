# Core ML's compile cache: measured behaviour

What Core ML does the first time a process loads a compiled program
(`.mlmodelc`) on a compute path, what it keeps, and what makes a later
load reuse it. sidekick depends on this for first-load latency, disk use,
upgrades and installs; the decision records that rely on it (D46, D48,
D49, D50, D51, D52) cite the sections below.

Everything here comes from one audit, on an M1 Max (32 GB) under macOS
27.0 (build 26A428), with Xcode 27.1 and coremltools 9.0, in October 2026.
Each finding says whether it was **measured** or **inferred**, as the
audit judged it. Apple doesn't document these internals, so any of it
can change with a macOS release; the supported APIs are in their own
section.

## Method

- **Hits and misses come from Core ML's own log lines,** not from timing:
  its bundle cache logs each lookup (`E5BundleCache Lookup … exists=0` or
  `exists=1`), and every ANE compile shows up as a run of Apple's ANE
  compiler service (`ANECompilerService`). Both are read from the unified
  log (`log stream`) around each load.
- **Each experiment had its own home.** Core ML's per-user cache lives
  under the home directory, and `CFFIXED_USER_HOME` points a process at a
  private one, so no experiment saw another's entries.
- **Fresh model paths where the system-wide cache could interfere.** The
  ANE also keeps a cache outside any home (see "Transplanting a cache"),
  which a private home doesn't isolate, so experiments that needed a cold
  ANE load used a fresh copy of the program at a new path.
- **The programs** were synthetic MIL encoders, generated to vary one
  thing at a time (width, layers, length, masks, attention form), and
  copies of bge-small-en-v1.5's compiled buckets. A small Swift probe
  loaded them with `MLModel` and chosen compute units; the signing
  experiment in "The cache key" used sidekickd itself.

## The cache key

| Change against a warmed entry | ANE | GPU / CPU |
|---|---|---|
| Fresh `CFFIXED_USER_HOME` | miss | miss |
| Copy, hard-linked tree, APFS clone or rename to another path | miss | not measured |
| Directory moved away and back; new directory inode with the same file inodes | hit | not measured |
| Same path, a build with new weights (new inodes) | miss | miss |
| Same path, the original files copied back (new inodes) | miss | not measured |
| `weight.bin` rewritten in place with new weights (modification time restored) | miss | **hit, serving stale output** |
| `weight.bin` rewritten in place with the same bytes | miss | hit |
| `weight.bin` birth time changed | miss | hit |
| `weight.bin` modification time, access time, mode or extended attributes changed | hit | hit |
| `model.mil` semantic edit | miss | miss |
| `model.mil` trailing newline, or the same bytes in a new inode | hit | not measured |
| Another executable name (a copy, symlink or hard link to the binary) | miss | miss |
| Same binary in another directory; a rebuilt binary with the same name | hit | hit |
| Same binary, another code-signing identifier | **miss** (the ANE recompiles) | hit |
| Embedded Info.plist with `CFBundleIdentifier` | directory named for the bundle id | same |
| Other compute units | separate entries | separate entries |
| `allowLowPrecisionAccumulationOnGPU`; asynchronous load | hit | hit |
| `MLComputePlan.load` | an entry of its own: a full compile, plus another copy of the weights | same |

Measured. From these:
- **The directory** is
  `~/Library/Caches/<name>/com.apple.e5rt.e5bundlecache/<macOS build>/`,
  where `<name>` is the process's name as invoked, or the bundle
  identifier of an Info.plist embedded in the binary.
- **An entry** is keyed by the program's path, its parsed MIL, the
  identity of its `weight.bin` (inode and birth time), the compute units
  and, on the ANE, the loading binary's code-signing identifier.
- **A moved or rebuilt binary with the same name and signing identifier
  hits.** An earlier account, that a moved binary recompiles, was wrong.
- **Rewriting a weights file in place is unsafe on the GPU and the CPU:**
  the entry still hits and serves the old weights' output. Replace a
  program's files with new ones instead.

**sidekickd's signing identifier.** The linker signs a Rust binary ad hoc
as cargo's name for the build, `sidekickd-<hash>`, and the hash changes
with the package version. Measured with sidekickd itself, on bge-small-
en-v1.5's 128-token bucket on the ANE, two builds a version apart, the
second loading at the same model path and home as the first:

| builds | second build's first request | ANE compiler on the second load |
|---|---|---|
| both signed ad hoc as `io.github.ironblock.sidekickd` | 0.87 s | only bge-small's failed attempt (0.16 s, see "Symlinks and the fallback loader") |
| both with the linker's signature | 3.76 s | that attempt, and a full compile (2.4 s) |

The first load, with no cache, took 4.3 s either way, and every output was
identical. Release archives sign sidekickd that way
(`tools/sign_sidekickd.sh`).

## Transplanting a cache

- **A cache copied to another home, or to another executable's name,**
  hits on the GPU and the CPU, with identical outputs. On the ANE it
  always recompiles. Measured.
- **What an entry holds:**
  - `model.milhash`;
  - `H13C.bundle/H13C.e5`, a manifest with the compiler versions and an
    absolute input path;
  - for the ANE, `main_ane/model.anehash` (129 bytes);
  - for the GPU, a `.mpsgraphpackage` of about 1.0× the weights;
  - for the CPU, a `bnns_program.bnnsir` of about 1.0× the weights.
- **The ANE's program isn't in the home.** There's no `.hwx` there: the
  compiled ANE program lives in the system's ANE cache, kept by Apple's
  `aned` service on the system volume, about 1.0× the weights per compile.
  System Integrity Protection keeps users from removing it, deleting the
  user's cache doesn't free it, and each recompile adds another copy (the
  audit left about 5–8 GB there). Measured.

## Where the compile runs

The ANE compile runs in `ANECompilerService`, a root XPC service, at 94%
of one core and 770 MB resident. `aned` stays at 0%, the loading process
at 2% (splitting the MIL into segments) and the GPU at 0–1%. Measured: the
GPU can't help with the compile.

## Serialization

- Three 6 s ANE compiles took 19.5 s one after another, 19.1 s on threads
  of one process and 18.6 s in separate processes. None overlapped.
- A 2.2 s compile from another process, with another identity, took
  41.9 s: it queued behind a 43 s one.

Measured: ANE compiles are serialized across the whole system. A long
compile delays every app's first ANE load.

## Compile time against length

**bge-small's shape** (attention and MLP): 1.7, 2.7, 3.6, 6.0, 13.8 and
35 s at 128, 256, 512, 1,024, 2,048 and 4,096 tokens. MLP alone compiles
about 15× faster. At 2,048 tokens:

| attention | compile |
|---|---|
| plain (baseline) | 13.8 s |
| fused SDPA | 34.1 s |
| queries in 4 blocks | 26.0 s |
| queries in 16 blocks | 27.2 s |

**Qwen3-Embedding's width at 4,096 tokens, with an attention mask:**

| layers per program | compile |
|---|---|
| 1 | 7.4 s |
| 2 | 16.9 s |
| 4 | 42.6 s |
| 8 | 126 s |

- Compile time is superlinear in the number of layers per program. A
  mask passed as an input costs the same as a constant one.
- With no mask, eight layers compile in 28 s, but predict 2.5× slower.
- Eight one-layer programs compile in about 59 s, against 126 s for one
  eight-layer program, and predict in 1,290 ms against 1,228 ms.

Measured: the slope comes from masked attention at 4,096 tokens per
program, not from length alone. Splitting a model into more programs (more
chunks, D37) is the lever; #164 measures Qwen3-Embedding at 4,096 tokens
that way.

## A hung ANE request

The first prediction of one synthetic program (an eight-layer MLP, width
2,048, FFN 8,192, 128 tokens, 512 MiB of fp16 weights) compiled for the
ANE, then hung for about 100 s and failed with `ANEProgramProcessRequestDirect()
… Request cancelled`. From then on, every ANE prediction on the machine
failed or hung for about 100 s, programs that had run cleanly included,
while compiles, the GPU and the CPU still worked, and the kernel log
repeated `ANE_CheckForPendingRequests … scheduling fast retry`. It
probably needed a reboot to clear. That 100 s request timeout may be what
D46's intermittent failures in the 4,096-token bucket were: inferred.
Anything that measures long ANE buckets should put a timeout on each
prediction and stop at the first hang.

## Chip architectures

Every compiled program carries an `H13C` directory, and the compiler's log
names its "Selected Target Arch". Core AI's documentation says its
ahead-of-time output is one `.aimodelc` per device architecture. Inferred,
strongly: a specialized program is per chip architecture, and doesn't move
across chip generations.

## OS updates

- The cache directory is named for the macOS build (`26A428`), and an
  orphaned directory for an earlier build (`25G83`) remained beside it.
- The machine's Core ML caches held 22.8 GB under the current build,
  14.8 GB of it sidekickd's.
- Core AI's documentation, and an Apple engineer on Apple's developer
  forums, say specialized models are invalidated by an OS update.

Inferred: every program compiles again after an OS update, and
directories named for an older build are safe to delete. Nothing is known
about the system's ANE cache across an update.

## GPU and CPU specialization

- **GPU load:** 0.15–0.24 s cold, 0.04–0.07 s warm.
- **First GPU prediction:** 232–260 ms cold, 90–226 ms warm.
- **Entry sizes:** about 1.0× the weights each on the GPU and the CPU. An
  ANE compile leaves 12 kB in the home and about 1.0× the weights in the
  system's ANE cache. A compute-plan read adds another 1.0×.

Measured. Earlier figures of 3–4 times the weights per bucket, including
3.7 GiB per bucket "on the ANE" (D46), were not reproduced; that ANE
figure was taken from the user's cache, which isn't where the ANE's
programs are.

## Supported APIs

- **Core ML:** `compileModel` produces device-independent output, the same
  for every Mac. An asynchronous load uses the same cache. `MLComputePlan`
  keeps an entry of its own. `specializationStrategy` trades compile time
  against prediction latency. Nothing exports a specialized program.
- **Core AI (macOS 27):**
  - `AIModel.specialize(contentsOf:options:cache:cachePolicy:)`
    specializes a model without loading it;
  - `AIModelCache` adds purge policies, a cache shared by an app group,
    and bookmarks;
  - `xcrun coreai-build compile` compiles ahead of time, one `.aimodelc`
    per device architecture (`deviceArchitectureName` was `h13c` on this
    machine).

  It needs a Core AI model (`.aimodel`) from Apple's coreai-torch tools,
  and the Metal Toolchain, which wasn't installed, so it wasn't run.

From the documentation: beyond loading in the process that will serve a
model, Core ML offers nothing to avoid the first-load compile. macOS 27
supports ahead-of-time compilation through Core AI; #163 evaluates it as
a backend beside Core ML.

Sources:
- https://developer.apple.com/documentation/coreai/managing-model-specialization-and-caching
- https://developer.apple.com/documentation/coreai/compiling-core-ai-models-ahead-of-time
- https://developer.apple.com/documentation/coreai/integrating-on-device-ai-models-in-your-app-with-core-ai
- https://developer.apple.com/forums/thread/831977

## Symlinks and the fallback loader

- **Synthetic programs,** which Core ML's current loader (E5) compiles for
  the ANE, stayed on the ANE, with bit-identical outputs, under every
  symlink variant. Measured.
- **bge-small-en-v1.5 loses the ANE** because its ANE compile on the
  current loader fails (D25's fused attention, a failed attempt of about
  0.17 s on every load) and Core ML falls back to its older loader. The
  ANE service then can't open a link whose target is outside the program
  directory (`Code=2`), so the program runs on the CPU: 28.8 ms against
  3.2 ms per prediction, with different outputs. Measured.
  - A relative link inside the program directory stays on the ANE, as
    does a link for `coremldata.bin` pointing outside. Measured.
  - The likely mechanism is the scope of the ANE service's sandbox
    extension. Inferred.

So the failure is the fallback loader with links that leave the program
directory, not "any symlinked file". sidekick keeps hard links and the
registry's refusal of symlinked programs on the ANE (D51), since any
program that falls back would leave the ANE silently. That programs on
the current loader are unaffected is still to be confirmed on a sidekick
build with explicit attention.
