# sidekick

Small, on-device models for the background work beside a bigger one:
session titles, tags, embeddings, classification, reranking. sidekick runs
them on the parts of a Mac a local LLM or an editor leaves idle (the Neural
Engine for encoders and small open chat models, Apple's Foundation Models
for short constrained generation), so they don't compete with the GPU. On Intel Macs and Linux
the same API serves embedders on the CPU through ONNX Runtime.

Compare every model's fidelity and benchmark scores by chip and path:
[ironblock.github.io/sidekick](https://ironblock.github.io/sidekick/).

The daemon, `sidekickd`, speaks the OpenAI API, plus the vLLM and Cohere
shapes for routes OpenAI doesn't have. Anything that takes an
OpenAI-compatible base URL can use it.

| Route | What it serves |
|---|---|
| `POST /v1/chat/completions` | Apple Foundation Models (Apple silicon, macOS 26+, Apple Intelligence on) and open chat models on the Neural Engine (Apple silicon) |
| `POST /v1/embeddings`, `/v2/embed` | Embedders, on the Neural Engine, GPU or CPU (Core ML) or the CPU (ONNX Runtime) |
| `POST /v1/classify` | Classifiers, including zero-shot and NLI (Core ML, macOS) |
| `POST /v1/rerank`, `/rerank`, `/v2/rerank` | Cross-encoder rerankers (Core ML, macOS) |
| `POST /tokenize`, `/detokenize` | A model's own tokenizer: count tokens, check an input fits |
| `GET /v1/models`, `GET /health` | What this machine serves, and why anything is unavailable |

The full contract of every route is in [docs/API.md](docs/API.md).

## Who it's for

- **Tools that run a local LLM on the GPU** and want embeddings, labels or
  titles without taking GPU time from it.
- **Apps that want on-device embeddings or classification when available,**
  without a hard dependency: probe the daemon, fall back to an in-process
  library or to nothing ([docs/INTEGRATING.md](docs/INTEGRATING.md)).
- **Anyone who wants to know how faithful a model is on each chip:** every
  model is graded against its original on every compute path it serves
  ([docs/MODELS.md](docs/MODELS.md),
  [ironblock.github.io/sidekick](https://ironblock.github.io/sidekick/)).

## Platforms

| Platform | Chat | Embeddings | Classify, rerank |
|---|---|---|---|
| macOS 26+, Apple silicon | yes | Core ML (Neural Engine, GPU, CPU), ONNX | yes |
| macOS, Intel | no | Core ML (GPU, CPU), ONNX | yes |
| Linux x86-64 and arm64 | no | ONNX | no |

Each has a release archive. Minimum OS versions, the bundled ONNX Runtime
and building from source are in [docs/INSTALL.md](docs/INSTALL.md).

## Quickstart

On an Apple silicon Mac. Pick the latest version on the
[Releases page](https://github.com/ironblock/sidekick/releases); other
platforms' archive names are in [docs/INSTALL.md](docs/INSTALL.md).

**1. Install.**

```sh
VERSION=v0.17.0
NAME=sidekick-$VERSION-aarch64-apple-darwin
curl -LO https://github.com/ironblock/sidekick/releases/download/$VERSION/$NAME.tar.gz
curl -LO https://github.com/ironblock/sidekick/releases/download/$VERSION/$NAME.tar.gz.sha256
shasum -a 256 -c $NAME.tar.gz.sha256
tar -xzf $NAME.tar.gz && cd $NAME
```

**2. Get a model.** `sidekickd fetch` downloads one of sidekick's graded
builds from Hugging Face, checks every file against the checksums built
into sidekickd, and installs it in the models directory:

```sh
./sidekickd fetch bge-small-en-v1.5
```

Fetch installs the builds sidekick has published, listed in
[docs/MODELS.md](docs/MODELS.md#published-builds); every other model is
installed by converting it. Some builds ask you to accept terms first;
fetch shows them and tells you how. Every way to install a model is in
[docs/RUNNING.md](docs/RUNNING.md#installing-models).

**3. Start the daemon.** It reads the models directory when it starts, so
start it, or restart it, after installing a model:

```sh
./sidekickd
```

It listens on `127.0.0.1:8790`. To keep it running across logins, install
it as a LaunchAgent ([docs/INSTALL.md](docs/INSTALL.md#running-as-a-service)).

**4. Embed.** In another terminal:

```sh
curl -s localhost:8790/v1/embeddings -H 'content-type: application/json' \
  -d '{"model": "bge-small-en-v1.5", "input": ["How do I reset my password?"]}'
```

Core ML compiles a model for this Mac before serving it: fetch compiles it
when it installs it, and the daemon compiles anything not yet compiled (after
a macOS update, say) in the background when it starts. A request whose
bucket is still compiling waits for it; the rest take milliseconds.

**5. Chat** (needs Apple Intelligence turned on):

```sh
curl -s localhost:8790/v1/chat/completions -H 'content-type: application/json' -d '{
  "model": "apple-fm",
  "messages": [{"role": "user", "content": "Write a five-word title for: fixing the flaky login test"}]
}'
```

`curl -s localhost:8790/health` says what's available and why anything
isn't: a missing capability is a 503 with the reason, never a hang.

### From an OpenAI client

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8790/v1", api_key="unused")
vectors = client.embeddings.create(model="bge-small-en-v1.5", input=["hello"])
```

The key is ignored unless the daemon is configured with one
([docs/CONFIG.md](docs/CONFIG.md)).

## Models

Compare every model's fidelity and benchmark scores by chip and path:
[ironblock.github.io/sidekick](https://ironblock.github.io/sidekick/).

sidekick serves embedders, classifiers (sentiment, zero-shot, NLI) and a
reranker. [docs/MODELS.md](docs/MODELS.md) lists every one, with its
grade on each compute path, its prompts, its licence and its training
data. Some starting points:

| You want | Start with |
|---|---|
| Fast English embeddings | `bge-small-en-v1.5` |
| Multilingual embeddings, smaller vectors on request | `embeddinggemma-300m` |
| Labels you choose per request | `nli-deberta-v3-base` |
| Reordering search results | `ms-marco-minilm-l6-v2` |

## Where next

| To | Read |
|---|---|
| Install on another platform, build from source, run as a service | [docs/INSTALL.md](docs/INSTALL.md) |
| Install models, understand first loads, memory and disk | [docs/RUNNING.md](docs/RUNNING.md) |
| Configure the daemon: every key, flag and environment variable | [docs/CONFIG.md](docs/CONFIG.md) |
| Call the API: every route, field, limit and error | [docs/API.md](docs/API.md) |
| Choose compute units, and the limits of each backend | [docs/BACKENDS.md](docs/BACKENDS.md) |
| Embed sidekick in an app, with or without the daemon | [docs/INTEGRATING.md](docs/INTEGRATING.md) |
| Pick a model, and see how faithful it is | [docs/MODELS.md](docs/MODELS.md) |
| Convert a model yourself | [docs/CONVERTING.md](docs/CONVERTING.md) |
| See why sidekick works the way it does | [docs/DECISIONS.md](docs/DECISIONS.md) |
| See what changed in a release | [docs/releases/](docs/releases/) |

## Contributing

Contributions are welcome. [AGENTS.md](AGENTS.md) has the rules for
commits, branches and releases, and for keeping this public repository
clean; [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) covers the workspace,
building and testing.

## License

MIT; see [LICENSE](LICENSE). Each model has its own licence, which its
entry in [docs/MODELS.md](docs/MODELS.md#licences-and-training-data) and
its Hugging Face repository give.
