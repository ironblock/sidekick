"""Stage 1 of the verification corpus: hermetic captures of pinned CLIs.

Runs each world in fixtures/tasks/verification/recipes.toml with the tools
pinned in tools.lock.toml, under that file's fixed environment, then scrubs
and scans every output (tools/tasks/scrub.py). Captured output is written to
`--out`; `--update-lock` also copies it to fixtures/tasks/verification/
captures.jsonl, which is committed beside the recipe, the tool lock and
captures.lock.json (each capture's sha256 and the set's). With the
captures committed, generation needs no tools.

Host tools (git, python3, /bin/sh) are not pinned. The lock records their
versions and which captures depend on each (a world's `host_tools`), and
`--check` fails if a version differs unless --allow-host-drift is given.

    # download the pinned assets (sha256-checked) into a cache directory
    python3 tools/tasks/capture.py fetch --cache <dir>
    # run every recipe and write <out>/captures.jsonl
    python3 tools/tasks/capture.py run --cache <dir> --out <dir> [--update-lock]
    # run again and compare with captures.lock.json, capture by capture
    python3 tools/tasks/capture.py run --cache <dir> --out <dir> --check

`run` builds the world under the recipe's fixed working root (default
/tmp/skv, which it deletes and recreates; pass --root to move it). A fixed
root keeps tool output that wraps on path length identical, and the
scrubber maps it to /work. Run it at low priority (`nice -n 19`): it is
light, but it shares the machine.

A world is a directory under the root, built by `steps` (`write` a file,
`append` to one, `chmod`, or `run` a shell command), optionally starting
from a copy of an earlier world (`from`) and with extra variables (`env`).
A world that `needs` a linked toolchain (Rust) runs only when
`--toolchain NAME=DIR` supplies it; one marked `pending` runs only with
--include-pending. Each capture records `seq`, its place in the run, so the
generator can keep a world's captures in order. Each capture runs one command in
its world and records its stdout, stderr and exit status. `seed_env` names
a variable that gets a fresh value per command (jj's
JJ_RANDOMNESS_SEED), so generated ids are reproducible.
"""

from __future__ import annotations

import argparse
import shutil
import stat
import subprocess
import sys
import tarfile
import tomllib
import urllib.request
import zipfile
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tasks import canon, schema, scrub  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
FIXTURES = REPO / "fixtures" / "tasks" / "verification"
TOOLS_LOCK = FIXTURES / "tools.lock.toml"
RECIPES = FIXTURES / "recipes.toml"
CAPTURES_LOCK = FIXTURES / "captures.lock.json"
CAPTURES = FIXTURES / "captures.jsonl"
CAPTURE_SCHEMA = "sidekick-verification-capture/1"
COMMAND_TIMEOUT_S = 60


def load_toml(path: Path) -> dict:
    with open(path, "rb") as f:
        return tomllib.load(f)


# ---------------------------------------------------------------- fetch


def asset_path(cache: Path, tool: dict) -> Path:
    return cache / tool["name"] / tool["asset"]


def fetch(cache: Path, tools_lock: dict) -> None:
    for tool in tools_lock["tool"]:
        dest = asset_path(cache, tool)
        if dest.exists() and canon.sha256_file(dest) == tool["sha256"]:
            print(f"{tool['name']}: cached, sha256 ok")
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".part")
        print(f"{tool['name']}: downloading {tool['url']}")
        with urllib.request.urlopen(tool["url"], timeout=120) as r, open(tmp, "wb") as f:
            shutil.copyfileobj(r, f)
        got = canon.sha256_file(tmp)
        if got != tool["sha256"]:
            tmp.unlink()
            raise SystemExit(f"{tool['name']}: sha256 {got} does not match the lock's {tool['sha256']}")
        tmp.rename(dest)
        print(f"{tool['name']}: sha256 ok")


def install(cache: Path, root: Path, tools_lock: dict) -> None:
    for tool in tools_lock["tool"]:
        src = asset_path(cache, tool)
        if not src.exists() or canon.sha256_file(src) != tool["sha256"]:
            raise SystemExit(f"{tool['name']}: no verified asset in the cache; run `fetch` first")
        ex = tool["extract"]
        if "tar_tree" in ex:  # a whole tree (a toolchain), under a prefix
            dest = root / tool["install"]
            dest.mkdir(parents=True, exist_ok=True)
            prefix = ex["tar_tree"]
            with tarfile.open(src) as t:
                members = []
                for m in t.getmembers():
                    if not m.name.startswith(prefix) or m.name == prefix.rstrip("/"):
                        continue
                    m.name = m.name[len(prefix):]
                    if m.name:
                        members.append(m)
                t.extractall(dest, members=members, filter="data")
            continue
        if "tar" in ex:
            with tarfile.open(src) as t:
                data = t.extractfile(ex["tar"]).read()
        elif "zip" in ex:
            with zipfile.ZipFile(src) as z:
                data = z.read(ex["zip"])
        else:
            data = src.read_bytes()
        dest = root / tool["install"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        dest.chmod(0o755)
        if "manifest" in tool:
            m = root / tool["manifest"]["path"]
            m.parent.mkdir(parents=True, exist_ok=True)
            m.write_text(tool["manifest"]["content"])


# ---------------------------------------------------------------- run


def link_toolchains(tools_lock: dict, root: Path, given: dict[str, Path]) -> set[str]:
    """Link each `--toolchain NAME=DIR` into the root, refusing a version other
    than the lock's. Returns the names linked."""
    linked = set()
    for tc in tools_lock.get("linked_toolchain", []):
        if tc["name"] not in given:
            continue
        dest = root / tc["link"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.symlink_to(Path(given[tc["name"]]).resolve(), target_is_directory=True)
        out = subprocess.run(["/bin/sh", "-c", tc["version_cmd"].replace("{W}", str(root))],
                             capture_output=True, text=True, timeout=COMMAND_TIMEOUT_S).stdout
        if not out.startswith(tc["expect"]):
            raise SystemExit(f"toolchain {tc['name']}: {out.splitlines()[:1]} is not {tc['expect'].strip()}")
        linked.add(tc["name"])
    return linked


def frame_env(tools_lock: dict, root: Path) -> dict:
    return {k: v.replace("{W}", str(root)) for k, v in tools_lock["frame"]["env"].items()}


def sh(cmd: str, cwd: Path, env: dict) -> subprocess.CompletedProcess:
    return subprocess.run(["/bin/sh", "-c", cmd], cwd=cwd, env=env, stdin=subprocess.DEVNULL,
                          capture_output=True, timeout=COMMAND_TIMEOUT_S)


def _make_writable(path: Path) -> None:
    for p in [path, *path.rglob("*")]:
        try:
            if not p.is_symlink():
                p.chmod(p.stat().st_mode | stat.S_IWUSR | stat.S_IXUSR)
        except OSError:
            pass


def reset_root(root: Path) -> None:
    if root.exists():
        if not (root / ".skv-root").exists():
            raise SystemExit(f"refusing to delete {root}: it was not made by this tool (no .skv-root marker)")
        _make_writable(root)
        shutil.rmtree(root)
    root.mkdir(parents=True)
    (root / ".skv-root").write_text("verification capture root; deleted and rebuilt by tools/tasks/capture.py\n")


def run_world(world: dict, root: Path, env: dict, counters: dict) -> list[dict]:
    wdir = root / world["dir"]
    if "from" in world:
        src = root / counters["dirs"][world["from"]]
        if src != wdir:  # same dir: continue in place, after the earlier world's captures
            if wdir.exists():
                shutil.rmtree(wdir)
            shutil.copytree(src, wdir, symlinks=True)
    wdir.mkdir(parents=True, exist_ok=True)
    env = {**env, **{k: v.replace("{W}", str(root)) for k, v in world.get("env", {}).items()}}
    seed_var = world.get("seed_env")
    seed = counters.setdefault("seed_end", {}).get(world.get("from"), 0)  # a copy continues its source's ids

    def env_for_call():
        nonlocal seed
        e = dict(env)
        if seed_var:
            seed += 1
            e[seed_var] = str(seed)
        return e

    for step in world.get("steps", []):
        if "write" in step:
            p = wdir / step["write"]
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(step["content"])
        elif "append" in step:
            with open(wdir / step["append"], "a") as f:
                f.write(step["content"])
        elif "chmod" in step:
            (wdir / step["chmod"]).chmod(int(step["mode"], 8))
        elif "run" in step:
            r = sh(step["run"], wdir, env_for_call())
            if r.returncode != step.get("exit", 0):
                raise SystemExit(f"world {world['id']}: step {step['run']!r} exited {r.returncode}:\n"
                                 f"{r.stderr.decode(errors='replace')}")
        else:
            raise SystemExit(f"world {world['id']}: unknown step {step}")
    out = []
    for cap in world.get("capture", []):
        r = sh(cap["cmd"], wdir, env_for_call())
        if "exit" in cap and r.returncode != cap["exit"]:
            raise SystemExit(f"capture {cap['id']}: expected exit {cap['exit']}, got {r.returncode}:\n"
                             f"{r.stderr.decode(errors='replace')}")
        counters["seq"] += 1
        out.append({
            "schema": CAPTURE_SCHEMA,
            "id": cap["id"],
            "seq": counters["seq"],
            "world": world["id"],
            "tool": world["tool"],
            "kind": cap["kind"],
            "cmd": cap.get("display", cap["cmd"]),
            "stdout": r.stdout.decode("utf-8"),
            "stderr": r.stderr.decode("utf-8"),
            "exit": r.returncode,
        })
    counters["dirs"][world["id"]] = world["dir"]
    counters["seed_end"][world["id"]] = seed
    return out


def host_versions(tools_lock: dict, env: dict, root: Path, linked: set[str] = frozenset()) -> dict:
    """First line of each host tool's version, and a linked toolchain's full
    version output joined with "; " (rustc -vV names its commit)."""
    out = {}
    for t in tools_lock.get("host_tool", []):
        r = sh(t["version_cmd"], root, env)
        out[t["name"]] = (r.stdout or r.stderr).decode().strip().splitlines()[0] if (r.stdout or r.stderr) else ""
    for tc in tools_lock.get("linked_toolchain", []):
        if tc["name"] in linked:
            r = sh(tc["version_cmd"].replace("{W}", str(root)), root, env)
            out[tc["name"]] = "; ".join(r.stdout.decode().strip().splitlines())
    return out


def run(cache: Path, out_dir: Path, root: Path | None, update_lock: bool, check: bool,
        include_pending: bool = False, allow_host_drift: bool = False,
        toolchains: dict[str, Path] | None = None) -> int:
    tools_lock = load_toml(TOOLS_LOCK)
    recipes = load_toml(RECIPES)
    root = Path(root or recipes["root"])
    reset_root(root)
    for d in ("hm", "tmp", "bin", "xdg", "xcfg", "xcache"):
        (root / d).mkdir(parents=True, exist_ok=True)
    for f in tools_lock["frame"]["file"]:
        (root / f["path"]).write_text(f["content"])
    install(cache, root, tools_lock)
    linked = link_toolchains(tools_lock, root, toolchains or {})
    env = frame_env(tools_lock, root)

    captures = []
    counters = {"dirs": {}, "seq": 0}
    for world in recipes["world"]:
        if world.get("pending") and not include_pending:
            print(f"skipped world {world['id']}: pending ({world['pending']})")
            continue
        missing = set(world.get("needs", [])) - linked
        if missing:
            print(f"skipped world {world['id']}: needs --toolchain {', '.join(sorted(missing))}=<dir>")
            continue
        captures.extend(run_world(world, root, env, counters))

    rules = [(r["pattern"], r["replace"]) for r in recipes.get("scrub", {}).get("rule", [])]
    allow = recipes.get("scan", {}).get("allow", [])
    clock = tuple(recipes.get("scan", {}).get("clock", ["2026-10-01", "2026-10-01"]))
    identity = scrub.machine_identity()
    refused = []
    for c in captures:
        for field in ("cmd", "stdout", "stderr"):
            c[field] = scrub.scrub(c[field], [str(root)], rules)
        text = "\n".join((c["cmd"], c["stdout"], c["stderr"]))
        findings = scrub.unallowed(scrub.scan(text, identity, clock, [str(root)]), allow, c["id"])
        refused.extend((c["id"], f) for f in findings)
    if refused:
        for cid, f in refused:
            print(f"REFUSED {cid}: {f}", file=sys.stderr)
        print(f"{len(refused)} finding(s); nothing written", file=sys.stderr)
        return 1

    for c in captures:
        schema.validate(c, "capture")
    set_sha = canon.write_jsonl(out_dir / "captures.jsonl", captures)
    lock = {
        "schema": "sidekick-verification-captures-lock/1",
        "tools_lock_sha256": canon.sha256_file(TOOLS_LOCK),
        "recipes_sha256": canon.sha256_file(RECIPES),
        "host_tools": host_versions(tools_lock, env, root, linked),
        "host_dependents": host_dependents(recipes, captures),
        "captures": [{"id": c["id"], "sha256": canon.record_sha256(c)} for c in sorted(captures, key=lambda c: c["id"])],
        "set_sha256": set_sha,
    }
    schema.validate(lock, "captures_lock")
    print(f"{len(captures)} captures, set sha256 {set_sha}")
    if update_lock:
        canon.write_json(CAPTURES_LOCK, lock)
        CAPTURES.write_bytes((out_dir / "captures.jsonl").read_bytes())
        print(f"wrote {CAPTURES_LOCK.relative_to(REPO)} and {CAPTURES.relative_to(REPO)}")
    if check:
        return compare_lock(lock, canon.read_json(CAPTURES_LOCK), allow_host_drift)
    return 0


def host_dependents(recipes: dict, captures: list[dict]) -> dict:
    """Which captures' output depends on each host tool, from the worlds'
    `host_tools` declarations."""
    by_world = {w["id"]: w.get("host_tools", []) + w.get("needs", []) for w in recipes["world"]}
    out: dict[str, list[str]] = {}
    for c in sorted(captures, key=lambda c: c["id"]):
        for t in by_world.get(c["world"], []):
            out.setdefault(t, []).append(c["id"])
    return out


def compare_lock(got: dict, want: dict, allow_host_drift: bool = False) -> int:
    bad = 0
    for key in ("tools_lock_sha256", "recipes_sha256"):
        if got[key] != want[key]:
            print(f"MISMATCH {key}: {got[key]} != committed {want[key]}")
            bad += 1
    for name, v in want.get("host_tools", {}).items():
        if got["host_tools"].get(name) != v:
            deps = want.get("host_dependents", {}).get(name, [])
            print(f"{'note' if allow_host_drift else 'MISMATCH'}: host {name} is {got['host_tools'].get(name)!r}, "
                  f"the lock recorded {v!r}; {len(deps)} capture(s) depend on it")
            bad += 0 if allow_host_drift else 1
    g = {c["id"]: c["sha256"] for c in got["captures"]}
    w = {c["id"]: c["sha256"] for c in want["captures"]}
    for cid in sorted(set(g) | set(w)):
        if g.get(cid) != w.get(cid):
            print(f"MISMATCH capture {cid}: {g.get(cid)} != committed {w.get(cid)}")
            bad += 1
    if got["set_sha256"] != want["set_sha256"]:
        print(f"MISMATCH set_sha256: {got['set_sha256']} != committed {want['set_sha256']}")
        bad += 1
    print("captures match the lock" if not bad else f"{bad} mismatch(es)")
    return 1 if bad else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch")
    f.add_argument("--cache", type=Path, required=True)
    r = sub.add_parser("run")
    r.add_argument("--cache", type=Path, required=True)
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--root", type=Path, help="working root (default: recipes.toml's)")
    r.add_argument("--toolchain", action="append", default=[], metavar="NAME=DIR",
                   help="link a toolchain the lock names (rust=<rustup toolchain dir>)")
    r.add_argument("--allow-host-drift", action="store_true",
                   help="with --check, report a different host git, python3 or sh instead of failing")
    r.add_argument("--include-pending", action="store_true",
                   help="also run worlds marked pending (they need toolchains outside the frame's PATH)")
    g = r.add_mutually_exclusive_group()
    g.add_argument("--update-lock", action="store_true")
    g.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "fetch":
        fetch(a.cache, load_toml(TOOLS_LOCK))
        return 0
    toolchains = dict(t.split("=", 1) for t in a.toolchain)
    return run(a.cache, a.out, a.root, a.update_lock, a.check, a.include_pending, a.allow_host_drift,
               {k: Path(v) for k, v in toolchains.items()})


if __name__ == "__main__":
    sys.exit(main())
