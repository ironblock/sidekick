# Working in this repository

Guidelines for anyone — human or AI agent — contributing to sidekick. They
exist to keep a public, general-audience codebase clean, professional, and
self-explanatory.

## This repository is public

Assume every file will be read by someone with **no context** on how it was
built. Write commit messages, comments, and documentation for that reader.
Keep the tone professional and each document self-contained: explain what a
thing is and why, not what it was called while it was being built.

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
- **Every release ships published release notes.** Do not tag a release
  without them.
- Bump `[workspace.package] version` in the root `Cargo.toml` to match the tag
  so `sidekickd --version` stays honest.
