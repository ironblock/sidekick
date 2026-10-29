# Working in this repository

Guidelines for anyone — human or AI agent — contributing to sidekick. They
exist to keep a public, general-audience codebase clean, professional, and
self-explanatory.

## This repository is public

Assume every file will be read by someone with **no context** on how it was
built. Write commit messages, comments, and documentation for that reader.
Keep the tone professional and each document self-contained: explain what a
thing is and why, not what it was called while it was being built.

## One home for each fact

Each fact in the docs has one canonical home; every other document links
to it rather than copying it, so a correction is made once and copies
can't drift apart.

## Never commit

- **Local filesystem paths** — home directories, temp/scratch directories, or
  anything machine-specific. Take paths as arguments or use portable examples
  (`~/Library/Application Support/…`, `<data dir>/…`).
- **Secrets** — API keys, tokens, credentials.
- **Personally identifying or machine-specific information** — real names,
  email addresses, hostnames, non-loopback IP addresses, or test fixtures tied
  to a particular machine or account. (`127.0.0.1` / `localhost` is fine; it's
  the documented default bind.)

## No internal planning references

Do not carry ephemeral planning identifiers from a working session into the
repository — milestone or task tags like `M9`, `T.1`, sprint names, or
chat-thread labels. They mean nothing to an outside reader; describe the change
on its own terms instead.

(The `docs/DECISIONS.md` decision log — `D1`, `D2`, … — is the deliberate
exception: it is a maintained, self-explanatory architecture-decision record
meant for external readers, not session scratch. Keep entries self-contained.)

## Commits

Use [Conventional Commits](https://www.conventionalcommits.org/): a
`type(scope): summary` subject, where `type` is one of `feat`, `fix`, `docs`,
`chore`, `refactor`, `test`, `ci`, `perf`, `build`. Write the body for the
external reader — what changed and why, with measured results where relevant.

## Branching and releases

Light, trunk-based git-flow:

- **`main` is the trunk** and is kept in a releasable state.
- **Feature and chore work happens on short-lived branches** named for their
  purpose (`feat/…`, `chore/…`, `fix/…`), which merge back into `main`.
- **Releases are annotated `vX.Y.Z` tags cut from `main`**, following
  [semver](https://semver.org/). Pushing a `v*` tag drives the build-and-
  publish workflow (`.github/workflows/release.yml`).
- **Every release ships published release notes.** Write them in
  `docs/releases/vX.Y.Z.md`, for the reader upgrading from the previous
  release, in the same change that bumps the version. The release workflow
  refuses a tag without them, or one that doesn't match the workspace
  version. It publishes them as the GitHub release's body and ships them in
  the archive as `RELEASE_NOTES.md`. Don't hard-wrap them: GitHub renders
  every newline in a release body as a line break, so keep each paragraph
  and list item on one line.
- Bump `[workspace.package] version` in the root `Cargo.toml` to match the tag
  so `sidekickd --version` stays honest.
- **Run the release gate before tagging.** `tools/release_gate.py` builds
  the release candidate's `sidekickd` and, on a Mac with the published
  models' builds installed, runs each model's task sample through it over
  HTTP: MultiNLI pairs for the nli models, an MTEB slice for the
  embedders, the parity corpora's labelled cases for the other
  classifiers, the reference set for rerankers (its Python environment is
  pinned in `tools/release_gate.requirements.txt`), and a slice of the chat
  benchmark for open chat models (scored in the chat benchmark's
  environment, `--chat-python`). It compares every number
  with the committed baseline, `docs/data/release-gate.json`, and fails on
  drift beyond the tolerances recorded there. Don't tag unless its last
  line reads PASS; put that line in the release PR. A deliberate change to
  a model's numbers (a new conversion, a runtime fix that moves a score)
  updates the baseline with `--update-baseline` in the same PR, and the
  PR says why. Let the gate keep its persistent home
  (`~/Library/Caches/sidekick-release-gate/` by default, `--home` to move
  it): Core ML's ANE programs are cached per home in a system cache that
  can't be deleted, so a fresh home recompiles every ANE model and leaves
  another copy of its weights on disk each time. With the home kept, only
  a new build or a new macOS compiles. `--fresh-home` is for when a cold
  measurement is the point.
