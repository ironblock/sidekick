#!/usr/bin/env python3
"""Links in sidekick's Markdown docs. Standard library only.

    python3 tools/doclinks.py pin --ref vX.Y.Z SRC OUT

Writes SRC (a Markdown file in this repository) to OUT with every relative
link rewritten to an absolute GitHub URL at REF
(`https://github.com/<repo>/blob/<ref>/<path>`, `tree/` for a directory),
for documents shipped outside the repository: a release archive's
README.md and INTEGRATING.md, whose relative links resolve to nothing once
unpacked. A link to a missing file, or a relative link that survives the
rewrite, is an error. Links within the same file (`#anchor`) are kept.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REPO = "ironblock/sidekick"

# An inline link or image's destination: the `](url)` or `](url "title")`
# that ends it, matched alone because a link's text can span lines.
INLINE = re.compile(r"(\]\()([^)\s]+)((?:\s+\"[^\"]*\")?\))")
# A reference definition: `[id]: url`.
REFDEF = re.compile(r"^(\s{0,3}\[[^\]]+\]:\s*)(\S+)(.*)$")
SCHEME = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*:")


def is_relative(url: str) -> bool:
    """A link that names a path in the repository (not a URL or an anchor)."""
    return not (SCHEME.match(url) or url.startswith("#") or url.startswith("//"))


def links(text: str):
    """Every link outside fenced code: (line number, url)."""
    fence = None
    for n, line in enumerate(text.splitlines(), 1):
        stripped = line.lstrip()
        marker = re.match(r"(`{3,}|~{3,})", stripped)
        if marker:
            if fence is None:
                fence = marker.group(1)[0] * len(marker.group(1))
            elif stripped.startswith(fence):
                fence = None
            continue
        if fence is not None:
            continue
        for m in INLINE.finditer(line):
            yield n, m.group(2)
        m = REFDEF.match(line)
        if m:
            yield n, m.group(2)


def pin(text: str, src: Path, ref: str, repo: str = DEFAULT_REPO, root: Path = ROOT) -> str:
    """`text` (the contents of `src`, a file under `root`) with each relative
    link made absolute at `ref`."""
    base = src.resolve().parent
    errors = []

    def rewrite(url: str, n: int) -> str:
        if not is_relative(url):
            return url
        path, _, frag = url.partition("#")
        target = (base / path).resolve()
        try:
            rel = target.relative_to(root.resolve())
        except ValueError:
            errors.append(f"{src}:{n}: {url} leaves the repository")
            return url
        if not target.exists():
            errors.append(f"{src}:{n}: {url} names a file that doesn't exist")
            return url
        kind = "tree" if target.is_dir() else "blob"
        out = f"https://github.com/{repo}/{kind}/{ref}/{rel.as_posix()}"
        return out + (f"#{frag}" if frag else "")

    out_lines = []
    fence = None
    for n, line in enumerate(text.split("\n"), 1):
        stripped = line.lstrip()
        marker = re.match(r"(`{3,}|~{3,})", stripped)
        if marker:
            if fence is None:
                fence = marker.group(1)[0] * len(marker.group(1))
            elif stripped.startswith(fence):
                fence = None
            out_lines.append(line)
            continue
        if fence is None:
            line = INLINE.sub(lambda m: m.group(1) + rewrite(m.group(2), n) + m.group(3), line)
            line = REFDEF.sub(lambda m: m.group(1) + rewrite(m.group(2), n) + m.group(3), line)
        out_lines.append(line)
    if errors:
        raise ValueError("\n".join(errors))
    out = "\n".join(out_lines)
    left = [f"{src}:{n}: {url}" for n, url in links(out) if is_relative(url)]
    if left:
        raise ValueError("relative links survived the rewrite:\n" + "\n".join(left))
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("pin", help="rewrite relative links to GitHub URLs at a ref")
    p.add_argument("--ref", required=True, help="the tag or commit the links point at")
    p.add_argument("--repo", default=os.environ.get("GITHUB_REPOSITORY", DEFAULT_REPO), help="<owner>/<name>")
    p.add_argument("src", type=Path)
    p.add_argument("out", type=Path)
    args = parser.parse_args(argv)

    if args.command == "pin":
        try:
            text = pin(args.src.read_text(encoding="utf-8"), args.src, args.ref, args.repo)
        except ValueError as e:
            print(f"error: {e}", file=sys.stderr)
            return 1
        args.out.write_text(text, encoding="utf-8")
        print(f"{args.src} -> {args.out}: links pinned to {args.repo}@{args.ref}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
