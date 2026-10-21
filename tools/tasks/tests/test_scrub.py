"""The scrubber and the refusal scan: planted identifiers must be refused,
documented placeholders must pass."""

import json
import os
import tempfile
import tomllib
import unittest
from unittest import mock
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

    def test_refuses_obfuscated_and_platform_paths(self):
        for planted in ("/users/alice/x", "/Volumes/Work/repo", "/opt/homebrew/bin/jq", "/tmp/501/T/x",
                        '{"path":"\\/Users\\/alice\\/x"}', "file%3A%2F%2F%2FUsers%2Falice%2Fx",
                        "\uff0fUsers\uff0falice"):
            with self.subTest(planted=planted):
                self.assertIn("path", rules(planted + "\n"))

    def test_format_characters_inside_a_path_are_stripped(self):
        for cf in ("​", "‍", "­", "⁠"):  # ZWSP, ZWJ, soft hyphen, word joiner
            with self.subTest(char=hex(ord(cf))):
                self.assertIn("path", rules(f"/U{cf}sers/bob/x\n"))

    def test_short_base64_paths_are_decoded(self):
        import base64
        enc = base64.b64encode(b"/Users/bob/x").decode()
        self.assertEqual(len(enc), 16)
        self.assertIn("path", rules(f"cfg {enc}\n"))
        self.assertIn("path", rules(f"cfg {base64.urlsafe_b64encode(b'/home/bob/notes').decode()}\n"))
        self.assertEqual(rules("change kkmpptxzrspxrzommnulwmwkkqwworpl\n"), set())

    def test_per_user_temporary_directories(self):
        for planted in ("/tmp/claude-501/T/x", "/tmp/501/x", "/tmp/user_1000/x", "/private/tmp/claude-501/x"):
            with self.subTest(planted=planted):
                self.assertIn("path", rules(planted + "\n"))
        self.assertEqual(rules("cache in /tmp/build/x\n"), set())

    def test_a_missing_denylist_file_warns_or_fails(self):
        import contextlib
        import io
        env = {"SIDEKICK_SCAN_DENYLIST": "/nonexistent/deny.txt"}
        with mock.patch.dict(os.environ, env):
            os.environ.pop("CI", None)
            os.environ.pop("SIDEKICK_SCAN_STRICT", None)
            err = io.StringIO()
            with contextlib.redirect_stderr(err):
                self.assertEqual(scrub.denylist(), [])
            self.assertIn("not a file", err.getvalue())
            for strict in ({"CI": "true"}, {"SIDEKICK_SCAN_STRICT": "1"}):
                with mock.patch.dict(os.environ, strict), self.assertRaises(SystemExit):
                    scrub.denylist()

    def test_refuses_hosts_users_and_trailers(self):
        self.assertIn("host", rules("ssh alice-mbp.local\n"))
        self.assertIn("user@host", rules("logged in as alice@mbp\n"))
        self.assertIn("email", rules("mail to bo@b\u00fccher.de\n"))
        self.assertIn("trailer", rules("Signed-off-by: Alice Real <alice@corp.dev>\n"))
        self.assertEqual(rules("Signed-off-by: Test User <test@example.com>\n"), set())

    def test_refuses_more_secrets(self):
        self.assertIn("secret:slack", rules("xoxb-1234567890-abcdefghij\n"))
        self.assertIn("secret:hex-key", rules('api_key = "3fa94e1c0b7d4a2f8e6c5b9a1d0e7f3c"\n'))
        self.assertEqual(rules("commit 3fa94e1c0b7d4a2f8e6c5b9a1d0e7f3c0a1b2c3d\n"), set())

    def test_denylist_from_the_environment(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "deny.txt")
            with open(path, "w") as f:
                f.write("# local names\nProjectFalcon\n")
            with mock.patch.dict(os.environ, {"SIDEKICK_SCAN_DENYLIST": path}):
                found = scrub.scan("notes for projectfalcon\n", scrub.machine_identity())
        self.assertIn("denylist", {f.rule for f in found})

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
                # The machine's identity when run locally; none in CI, where the
                # runner's names are common words (scrub.local_identity).
                found = scrub.unallowed(scrub.scan(path.read_text(), scrub.local_identity(), clock), allow, "*")
                self.assertEqual([str(f) for f in found], [])

    def test_every_allowlist_entry_is_used(self):
        with open(RECIPES, "rb") as f:
            recipes = tomllib.load(f)
        caps = [json.loads(x) for x in (RECIPES.parent / "captures.jsonl").read_text().splitlines()]
        clock = tuple(recipes["scan"]["clock"])
        found = {}
        for c in caps:
            text = "\n".join((c["cmd"], c["stdout"], c["stderr"]))
            for f in scrub.scan(text, {}, clock):
                found.setdefault(f.text, set()).add(c["id"])
        for a in recipes["scan"]["allow"]:
            scope = a["capture"]
            ids = found.get(a["text"], set())
            hit = {i for i in ids if i == scope or (scope.endswith("*") and i.startswith(scope[:-1]))}
            with self.subTest(text=a["text"]):
                self.assertTrue(hit, f"allowlist entry {a['text']!r} ({scope}) matches no finding")

    def test_every_committed_allowlist_entry_has_a_reason(self):
        with open(RECIPES, "rb") as f:
            recipes = tomllib.load(f)
        for a in recipes["scan"]["allow"]:
            self.assertTrue(a.get("reason"), a)
            self.assertIn("capture", a, "allowlist entries are scoped to one capture")


if __name__ == "__main__":
    unittest.main()
