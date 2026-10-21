"""The scrubber and the refusal scan: planted identifiers must be refused,
documented placeholders must pass."""

import tomllib
import unittest
from pathlib import Path

from tasks import scrub

IDENTITY = {"home": "/Users/tester", "user": "tester", "hosts": ["tester-mbp", "tester-mbp.lan"]}
RECIPES = Path(__file__).resolve().parents[3] / "fixtures" / "tasks" / "verification" / "recipes.toml"


def rules(text, **kw):
    return {f.rule for f in scrub.scan(text, IDENTITY, **kw)}


class Scrub(unittest.TestCase):
    def test_maps_both_spellings_of_the_root(self):
        text = "File \"/private/tmp/skv/gen/app.py\", line 2\nConflict worktree: /tmp/skv/gs/G5\n"
        self.assertEqual(scrub.scrub(text, ["/tmp/skv"]),
                         "File \"/work/gen/app.py\", line 2\nConflict worktree: /work/gs/G5\n")

    def test_rules_normalize_durations(self):
        out = scrub.scrub("test result: ok. finished in 0.42s\n", [], [(r"finished in [0-9.]+s", "finished in 0.00s")])
        self.assertEqual(out, "test result: ok. finished in 0.00s\n")


class RefusalScan(unittest.TestCase):
    def test_refuses_planted_paths(self):
        for planted in ("/Users/alice/project/a.py", "/home/bob/.config", "/var/folders/xy/T/tmp1",
                        "/private/var/db/x"):
            with self.subTest(planted=planted):
                self.assertIn("path", rules(f"error at {planted}: nope\n"))

    def test_refuses_the_machines_own_identity(self):
        self.assertIn("home", rules("cwd /Users/tester\n"))
        self.assertIn("user", rules("owner: tester\n"))
        self.assertIn("host", rules("ssh tester-mbp.lan\n"))
        self.assertIn("root", rules("wrote /tmp/skv/x\n", extra_literals=["/tmp/skv"]))

    def test_refuses_hosts_and_addresses(self):
        self.assertIn("ip", rules("connect 10.20.30.40:22\n"))
        self.assertIn("ip", rules("peer fe80:0:0:0:200:f8ff:fe21:67cf\n"))
        self.assertIn("mac", rules("ether 3c:22:fb:11:22:33\n"))

    def test_allows_loopback_documentation_ranges_and_versions(self):
        text = ("listen 127.0.0.1:8080 and 192.0.2.10, 2001:db8::1; jj 0.46.0; revset B::D; "
                "at 12:00:00\n")
        self.assertEqual(rules(text), set())

    def test_refuses_emails_outside_reserved_domains(self):
        self.assertIn("email", rules("Author: Bob <bob@corp.dev.com>\n"))
        self.assertEqual(rules("Author: Test User <test@example.com>, x@db.example.test\n"), set())
        self.assertEqual(rules('"packageManager": "pnpm@9.15.4"\n'), set())

    def test_refuses_dates_off_the_synthetic_clock(self):
        self.assertIn("date", rules("2025-03-04 10:00\n"))
        self.assertEqual(rules("2026-10-01 12:00:00.000 +00:00\n"), set())

    def test_refuses_secrets(self):
        self.assertIn("secret:github", rules("token ghp_" + "a1B2" * 9 + "\n"))
        self.assertIn("secret:aws", rules("AKIAABCDEFGHIJKLMNOP\n"))
        self.assertIn("secret:entropy", rules("key Zq8Lp3Xv9Rt2Mw7Kd4Hs6Bn1Cf5\n"))
        self.assertEqual(rules("commit 43daa37ce0f9716cdf6fb65a29cbcae74a3160ae\n"), set())

    def test_allowlist_needs_a_matching_capture(self):
        found = scrub.scan("mail foo@bar.com\n", IDENTITY)
        allow = [{"capture": "doc.page", "text": "foo@bar.com", "reason": "a placeholder"}]
        self.assertEqual(scrub.unallowed(found, allow, "doc.page"), [])
        self.assertEqual(len(scrub.unallowed(found, allow, "other.page")), 1)

    def test_committed_fixtures_pass_the_scan(self):
        with open(RECIPES, "rb") as f:
            recipes = tomllib.load(f)
        allow = [{**a, "capture": "*"} for a in recipes["scan"]["allow"]]
        clock = tuple(recipes["scan"]["clock"])
        files = sorted(p for p in RECIPES.parent.iterdir() if p.is_file())
        self.assertGreater(len(files), 5)
        for path in files:
            with self.subTest(file=path.name):
                # No machine identity: these files are checked wherever the tests
                # run; the capture stage scanned with the capturing machine's.
                found = scrub.unallowed(scrub.scan(path.read_text(), {}, clock), allow, "*")
                self.assertEqual([str(f) for f in found], [])

    def test_every_committed_allowlist_entry_has_a_reason(self):
        with open(RECIPES, "rb") as f:
            recipes = tomllib.load(f)
        for a in recipes["scan"]["allow"]:
            self.assertTrue(a.get("reason"), a)
            self.assertIn("capture", a, "allowlist entries are scoped to one capture")


if __name__ == "__main__":
    unittest.main()
