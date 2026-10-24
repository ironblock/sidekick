"""tools/doclinks.py: the repository's link check, and pinning a shipped
document's links to a release."""

import tempfile
import unittest
from pathlib import Path

import doclinks


class PinTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "docs" / "releases").mkdir(parents=True)
        (self.root / "docs" / "API.md").write_text("# API\n")
        (self.root / "LICENSE").write_text("MIT\n")

    def tearDown(self):
        self.tmp.cleanup()

    def pin(self, text, src="README.md"):
        return doclinks.pin(text, self.root / src, "v1.2.3", "o/r", self.root)

    def test_relative_links_become_urls_at_the_ref(self):
        text = (
            "See [the API](docs/API.md#errors), [notes](docs/releases/),\n"
            "[a link whose text\nspans lines](LICENSE \"title\"), [here](#quickstart),\n"
            "[the web](https://example.com/x) and\n"
            "[ref]: docs/API.md\n"
        )
        out = self.pin(text)
        self.assertIn("(https://github.com/o/r/blob/v1.2.3/docs/API.md#errors)", out)
        self.assertIn("(https://github.com/o/r/tree/v1.2.3/docs/releases)", out)
        self.assertIn('(https://github.com/o/r/blob/v1.2.3/LICENSE "title")', out)
        self.assertIn("(#quickstart)", out)
        self.assertIn("(https://example.com/x)", out)
        self.assertIn("[ref]: https://github.com/o/r/blob/v1.2.3/docs/API.md", out)

    def test_links_resolve_from_the_documents_own_directory(self):
        out = self.pin("[back](../LICENSE)", src="docs/API.md")
        self.assertEqual(out, "[back](https://github.com/o/r/blob/v1.2.3/LICENSE)")

    def test_code_is_left_alone(self):
        text = "```md\n[x](not/a/file.md)\n```\n"
        self.assertEqual(self.pin(text), text)

    def test_a_missing_target_is_an_error(self):
        with self.assertRaisesRegex(ValueError, "doesn't exist"):
            self.pin("[gone](docs/GONE.md)")
        with self.assertRaisesRegex(ValueError, "leaves the repository"):
            self.pin("[out](../elsewhere.md)")


class CheckTest(unittest.TestCase):
    def test_anchors_are_githubs(self):
        text = (
            "# Title\n"
            "## D13 — Default bind `127.0.0.1:8790`, `/health` unauthenticated\n"
            "## Prompts (`input_type`)\n"
            "## Prompts (`input_type`)\n"
            "```sh\n# not a heading\n```\n"
            '<a name="kept"></a>\n'
        )
        self.assertEqual(
            doclinks.anchors(text),
            {"title", "d13--default-bind-1270018790-health-unauthenticated", "prompts-input_type",
             "prompts-input_type-1", "kept"},
        )

    def test_broken_files_and_anchors_are_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "docs").mkdir()
            (root / "docs" / "A.md").write_text("# A\n## Here\n")
            (root / "README.md").write_text(
                "[ok](docs/A.md#here) [web](https://example.com/#x) [self](#readme)\n"
                "# README\n[gone](docs/B.md) [bad](docs/A.md#there)\n"
                "```\n[in code](nowhere.md)\n```\n"
            )
            problems = doclinks.check(root)
        self.assertEqual(problems, ["README.md:3: docs/B.md: no such file", "README.md:3: docs/A.md#there: no such anchor"])


if __name__ == "__main__":
    unittest.main()
