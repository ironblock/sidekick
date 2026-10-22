"""The committed recipe, claims and locks are well formed and agree with
each other, and the schema validator rejects malformed records."""

import re
import tempfile
import tomllib
import unittest
from pathlib import Path

from tasks import canon, gen_verification as gen, schema

FIX = Path(__file__).resolve().parents[3] / "fixtures" / "tasks" / "verification"
SHA = re.compile(r"^[0-9a-f]{64}$")


def load(name):
    with open(FIX / name, "rb") as f:
        return tomllib.load(f)


class Canonical(unittest.TestCase):
    def test_lines_are_sorted_compact_and_hashable(self):
        recs = [{"id": "b", "x": "ü"}, {"id": "a", "y": [1, 2]}]
        with tempfile.TemporaryDirectory() as d:
            sha = canon.write_jsonl(Path(d) / "x.jsonl", recs)
            body = (Path(d) / "x.jsonl").read_bytes()
        self.assertEqual(body, '{"id":"a","y":[1,2]}\n{"id":"b","x":"ü"}\n'.encode())
        self.assertEqual(sha, canon.sha256_bytes(body))
        self.assertEqual(canon.record_sha256(recs[1]), canon.sha256_text('{"id":"a","y":[1,2]}\n'))

    def test_sub_seeds_are_stable_and_independent(self):
        self.assertEqual(canon.sub_seed(133, "claim-a"), canon.sub_seed(133, "claim-a"))
        self.assertNotEqual(canon.sub_seed(133, "claim-a"), canon.sub_seed(133, "claim-b"))
        self.assertNotEqual(canon.sub_seed(133, "a", "bc"), canon.sub_seed(133, "ab", "c"))


class Validator(unittest.TestCase):
    CAPTURE = {"schema": "sidekick-verification-capture/1", "id": "x.y", "seq": 1, "world": "w", "tool": "t",
               "kind": "stdout", "cmd": "echo", "stdout": "", "stderr": "", "exit": 0}

    def test_accepts_a_valid_capture(self):
        schema.validate(self.CAPTURE, "capture")

    def test_rejects_missing_extra_and_wrong_values(self):
        bad = dict(self.CAPTURE)
        del bad["exit"]
        self.assertTrue(any("missing 'exit'" in e for e in schema.errors(bad, "capture")))
        self.assertTrue(any("unexpected key" in e for e in schema.errors({**self.CAPTURE, "host": "h"}, "capture")))
        self.assertTrue(any("not one of" in e for e in schema.errors({**self.CAPTURE, "kind": "pdf"}, "capture")))
        self.assertTrue(schema.errors({**self.CAPTURE, "exit": True}, "capture"))
        self.assertTrue(schema.errors({**self.CAPTURE, "id": "Bad Id"}, "capture"))


class ToolsLock(unittest.TestCase):
    def test_every_tool_is_pinned_to_a_release_asset_by_sha256(self):
        lock = load("tools.lock.toml")
        self.assertTrue(lock["tool"])
        for t in lock["tool"]:
            with self.subTest(tool=t["name"]):
                self.assertRegex(t["sha256"], SHA)
                if t["url"].startswith("https://github.com/"):
                    self.assertEqual(t["url"],
                                     f"https://github.com/{t['repo']}/releases/download/{t['tag']}/{t['asset']}")
                else:  # a project's own download site, with its published checksum file
                    self.assertTrue(t["url"].startswith("https://") and t["url"].endswith(t["asset"]))
                    self.assertIn("SHASUMS", t["digest_source"])
                self.assertIn(t["version"], t["tag"])
                self.assertTrue(t["licence"])
                self.assertGreater(t["size"], 0)
        env = lock["frame"]["env"]
        for var in ("HOME", "TMPDIR", "XDG_DATA_HOME", "XDG_CONFIG_HOME", "GIT_CONFIG_GLOBAL", "JJ_CONFIG"):
            self.assertTrue(env[var].startswith("{W}/"), var)
        self.assertEqual(env["TZ"], "UTC")


class CapturesLock(unittest.TestCase):
    def test_validates_and_matches_the_committed_recipe(self):
        lock = canon.read_json(FIX / "captures.lock.json")
        schema.validate(lock, "captures_lock")
        ids = [c["id"] for c in lock["captures"]]
        self.assertEqual(ids, sorted(ids))
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(lock["recipes_sha256"], canon.sha256_file(FIX / "recipes.toml"),
                         "recipes.toml changed: rerun tools/tasks/capture.py run --update-lock")
        self.assertEqual(lock["tools_lock_sha256"], canon.sha256_file(FIX / "tools.lock.toml"))

    def test_committed_captures_match_the_lock(self):
        lock = canon.read_json(FIX / "captures.lock.json")
        self.assertEqual(canon.sha256_file(FIX / "captures.jsonl"), lock["set_sha256"])
        caps = canon.read_jsonl(FIX / "captures.jsonl")
        for c in caps:
            schema.validate(c, "capture")
        self.assertEqual({c["id"]: canon.record_sha256(c) for c in caps},
                         {c["id"]: c["sha256"] for c in lock["captures"]})

    def test_host_tool_dependents_are_recorded(self):
        lock = canon.read_json(FIX / "captures.lock.json")
        self.assertEqual(set(lock["host_dependents"]), set(lock["host_tools"]))
        self.assertIn("gh.g5.git-diff", lock["host_dependents"]["git"])
        self.assertIn("gen.trace.run", lock["host_dependents"]["python3"])

    def test_host_drift_fails_check_unless_allowed(self):
        from tasks import capture
        lock = canon.read_json(FIX / "captures.lock.json")
        drifted = {**lock, "host_tools": {**lock["host_tools"], "git": "git version 0.0.0"}}
        self.assertEqual(capture.compare_lock(drifted, lock), 1)
        self.assertEqual(capture.compare_lock(drifted, lock, allow_host_drift=True), 0)

    def test_every_recipe_capture_is_locked(self):
        recipes = load("recipes.toml")
        lock = canon.read_json(FIX / "captures.lock.json")
        locked = {c["id"] for c in lock["captures"]}
        want = {c["id"] for w in recipes["world"] if not w.get("pending") for c in w.get("capture", [])}
        self.assertEqual(locked, want)


class Claims(unittest.TestCase):
    def setUp(self):
        self.corpus = load("corpus.toml")
        self.claims = gen.load_claims(FIX / "claims.toml", self.corpus["max_per_shape"])

    def test_claims_validate_and_cover_every_type(self):
        self.assertEqual({c["type"] for c in self.claims}, set(gen.TYPES))
        for c in self.claims:
            for p in [*c["probes"]["support"], *c["probes"]["contradict"], *c["probes"].get("mention", [])]:
                re.compile(p)

    def test_claims_name_only_locked_or_pending_captures(self):
        locked = {c["id"] for c in canon.read_json(FIX / "captures.lock.json")["captures"]}
        for c in self.claims:
            missing = gen.needle_captures(c) - locked
            with self.subTest(claim=c["id"]):
                self.assertTrue(not missing or c.get("pending"), missing)

    def test_four_worked_examples_one_per_kind(self):
        worked = [c for c in self.claims if c.get("worked")]
        self.assertEqual(len(worked), 4)

    def test_the_five_ways_of_deciding_what_is_current_are_all_present(self):
        mechs = {c["mechanism"] for c in self.claims}
        self.assertTrue({"sequence", "supersede", "precedence", "version", "timestamp"} <= mechs)
        roles = {c.get("role") for c in self.claims if c["mechanism"] == "supersede"}
        self.assertEqual(roles, {"belief", "reason"})

    def test_fact_cap_is_enforced(self):
        facts = {}
        for c in self.claims:
            facts.setdefault(c["fact"], []).append(c["id"])
        self.assertLessEqual(max(map(len, facts.values())), self.corpus["max_per_fact"])
        text = "\n".join(['[[shape]]', 'id = "s"', 'type = "tool"'] +
                         [f'[[shape.claim]]\nid = "c{i}"\nfact = "same"\ntext = "t"\nprobes.support = ["a"]\n'
                          f'probes.contradict = ["b"]\n[[shape.claim.block]]\ncapture = "x"' for i in range(3)])
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "c.toml").write_text(text)
            with self.assertRaises(SystemExit):
                gen.load_claims(Path(d) / "c.toml", 3, 2)

    def test_version_scoped_notes_are_backed_by_captures(self):
        caps = {c["id"]: c for c in canon.read_jsonl(FIX / "captures.jsonl")}
        scoped = [sc for c in self.claims for sc in c.get("scoped", [])]
        self.assertTrue(scoped)
        for sc in scoped:
            for vb in sc["verified_by"]:
                self.assertRegex(gen.body(caps[vb["capture"]]), re.compile(vb["pattern"], re.M))

    def test_shape_cap_is_enforced(self):
        text = "\n".join(['[[shape]]', 'id = "s"', 'type = "tool"'] +
                         [f'[[shape.claim]]\nid = "c{i}"\nfact = "f{i}"\ntext = "t"\nprobes.support = ["a"]\n'
                          f'probes.contradict = ["b"]\n[[shape.claim.block]]\ncapture = "x"' for i in range(4)])
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "c.toml").write_text(text)
            with self.assertRaises(SystemExit):
                gen.load_claims(Path(d) / "c.toml", 3)


def current_set() -> str:
    corpus = load('corpus.toml')
    return corpus.get('set', f"review-{corpus.get('batch', 2)}")


def derived_nm_kind(t: dict) -> str:
    """nm_kind from the diff alone: a call that SUPPORTED shows with output
    and NOT MENTIONED keeps with none and a nonzero exit."""
    call = re.compile(r"^\[call (c\d+)\] \$ .*\n((?:(?!\[call c\d+\] \$ |\[exit ).*\n)*)\[exit (-?\d+)\]$", re.M)
    s = {m.group(1): m.group(2) for m in call.finditer(gen.expand(t, "s"))}
    for m in call.finditer(gen.expand(t, "nm")):
        if m.group(2) == "" and m.group(3) != "0" and s.get(m.group(1)):
            return "empty-output-nonzero-exit"
    return "removed"


class SetLock(unittest.TestCase):
    def test_validated_batch_2_lock_is_kept(self):
        """Batch 2's lock stays as validated by the host project; it records
        the inputs it was built from (in git history), not today's."""
        lock = canon.read_json(FIX / "verification-review.lock.json")
        schema.validate(lock, "lock")
        self.assertEqual(lock["set_sha256"], "8db0030b10bca4c849dcfb4b54291aef2cec1ea7eb83500abab1f0033ca4803b")
        self.assertEqual(lock["source_commit"], "3c59ef7561eeadacac3a2d479bdae08450c537cf")

    def test_a_historical_set_says_how_to_regenerate_it(self):
        for name, commit in (("review", "3c59ef7"), ("review-3", "81576f9")):
            with self.subTest(set=name), tempfile.TemporaryDirectory() as d, self.assertRaises(SystemExit) as ctx:
                gen.main(["--set", name, "--out", d, "--check"])
            self.assertEqual(str(ctx.exception),
                             f"{name}: historical set, validated at {commit}; check out that commit to regenerate")
        lock = canon.read_json(FIX / "verification-review-3.lock.json")
        schema.validate(lock, "lock")
        self.assertEqual(lock["set_sha256"], "6c93d2fc76fe4b5600172a9b7d6f097bcfce326bf85517c843ec1541fd1f8469")

    def test_review_lock_is_consistent(self):
        lock = canon.read_json(FIX / f"verification-{current_set()}.lock.json")
        schema.validate(lock, "lock")
        ids = [t["id"] for t in lock["triples"]]
        self.assertEqual(ids, sorted(ids))
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(lock["claims_sha256"], canon.sha256_file(FIX / "claims.toml"),
                         "claims.toml changed: regenerate and update the review lock")
        self.assertEqual(lock["corpus_sha256"], canon.sha256_file(FIX / "corpus.toml"))
        self.assertEqual(lock["generator"], gen.GENERATOR)
        self.assertEqual(lock["captures_set_sha256"], canon.read_json(FIX / "captures.lock.json")["set_sha256"])
        corpus = load("corpus.toml")
        claim_ids = {c["id"] for c in gen.load_claims(FIX / "claims.toml", corpus["max_per_shape"])}
        for tid in ids:
            self.assertIn(tid.split(".")[0], claim_ids)

    def test_regenerates_the_committed_review_set(self):
        """`gen_verification.py --check` on the committed captures: the current
        set is rebuilt byte for byte (a few seconds; pure Python), and every
        triple's nm_kind matches the shape of its NOT MENTIONED diff."""
        with tempfile.TemporaryDirectory() as d:
            triples, _, lock = gen.generate(FIX / "captures.jsonl", Path(d), current_set(), FIX / "claims.toml",
                                            FIX / "corpus.toml", FIX / "captures.lock.json")
        self.assertEqual(gen.compare(lock, canon.read_json(FIX / f"verification-{current_set()}.lock.json")), [])
        for t in triples:
            self.assertEqual(t["nm_kind"], derived_nm_kind(t), t["id"])
            at = t["position_at"]
            self.assertEqual(t["config"]["position"], "start" if at < 1 / 3 else "middle" if at < 2 / 3 else "end")


if __name__ == "__main__":
    unittest.main()
