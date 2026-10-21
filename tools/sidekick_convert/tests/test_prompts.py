"""Named prompts: the manifest's [prompts] are the checkpoint's
sentence-transformers prompts (docs/DECISIONS.md D45)."""

import json
import tempfile
import tomllib
import unittest
from pathlib import Path

from sidekick_convert import manifest

PROMPTS = {
    "query": "task: search result | query: ",
    "Retrieval-query": "task: search result | query: ",
    "STS": "task: sentence similarity | query: ",
    "Instruct": "Instruct: a \"quoted\" task\nQuery:",
    "two words": "x",
}


def checkpoint(prompts):
    src = Path(tempfile.mkdtemp())
    if prompts is not None:
        (src / "config_sentence_transformers.json").write_text(json.dumps({"prompts": prompts}))
    return src


class PromptsTest(unittest.TestCase):
    def test_the_table_reads_back_as_the_checkpoints_prompts(self):
        table = manifest.prompts_toml(PROMPTS)
        self.assertEqual(tomllib.loads(table)["prompts"], PROMPTS)
        self.assertIn('Retrieval-query = "task: search result | query: "', table)
        self.assertIn('"two words" = "x"', table)

    def test_a_manifest_must_carry_the_checkpoints_prompts_exactly(self):
        src = checkpoint(PROMPTS)
        errors = []
        manifest.check_prompts({"prompts": PROMPTS}, src, errors)
        self.assertEqual(errors, [])
        for wrong in ({}, {**PROMPTS, "STS": "sts: "}, {k: v for k, v in PROMPTS.items() if k != "STS"},
                      {**PROMPTS, "sts": PROMPTS["STS"]}):
            errors = []
            manifest.check_prompts({"prompts": wrong}, src, errors)
            self.assertEqual(len(errors), 1, wrong)
            self.assertIn(manifest.prompts_toml(PROMPTS), errors[0])

    def test_a_checkpoint_without_prompts_leaves_them_to_the_manifest(self):
        for published in (None, {}):
            errors = []
            manifest.check_prompts({"prompts": {"Classification": "classify: "}}, checkpoint(published), errors)
            manifest.check_prompts({}, checkpoint(published), errors)
            self.assertEqual(errors, [])


if __name__ == "__main__":
    unittest.main()
