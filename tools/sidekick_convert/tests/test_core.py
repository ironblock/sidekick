"""The driver's inputs, the tokenizer rule, the manifest rules, and NaN-safe metrics."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from sidekick_convert import core, manifest, metrics, tokenizer


def _corpus_texts():
    import tomllib
    return [c["text"] for c in tomllib.loads(core.PARITY_CORPUS.read_text())["case"] if "text" in c]


def corpus_text():
    return _corpus_texts()[0]


class CalibrationGuard(unittest.TestCase):
    def test_refuses_graded_corpus_texts(self):
        with self.assertRaises(ValueError):
            core.Calibration(["an unrelated calibration text", corpus_text()])

    def test_refuses_prefixed_corpus_texts(self):
        long = next(t for t in _corpus_texts() if len(t) >= 12)
        with self.assertRaises(ValueError):
            core.Calibration(["title: none | text: " + long])

    def test_without_graded_drops_and_keeps(self):
        long = next(t for t in _corpus_texts() if len(t) >= 12)
        cal = core.Calibration.without_graded(["keep me", "query: " + long], report=lambda m: None)
        self.assertEqual(cal.texts, ("keep me",))

    def test_legacy_exemption_is_explicit(self):
        cal = core.Calibration([corpus_text()], legacy_graded="kept for byte identity")
        self.assertEqual(cal.texts, (corpus_text(),))

    def test_accepts_other_texts(self):
        cal = core.Calibration(["an unrelated calibration text"])
        self.assertEqual(cal.texts, ("an unrelated calibration text",))

    def test_calibration_and_evaluation_are_distinct_types(self):
        self.assertFalse(issubclass(core.Calibration, core.Evaluation))
        self.assertFalse(issubclass(core.Evaluation, core.Calibration))


def tiny_tokenizer(padding, truncation):
    from tokenizers import Tokenizer, models, pre_tokenizers
    tok = Tokenizer(models.WordLevel({"[UNK]": 0, "[PAD]": 1, "a": 2, "b": 3}, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    if padding:
        tok.enable_padding(pad_id=1, pad_token="[PAD]", length=8)
    if truncation:
        tok.enable_truncation(max_length=4)
    return tok


class TokenizerRule(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def snapshot(self, name, padding, truncation):
        d = self.tmp / name
        d.mkdir()
        tiny_tokenizer(padding, truncation).save(str(d / "tokenizer.json"))
        return d

    def test_clean_copies_byte_for_byte_without_padding_or_truncation(self):
        src = self.snapshot("plain", False, False)
        out = tokenizer.prepare(src, self.tmp / "out" / "tokenizer.json", mode="clean")
        self.assertEqual(out.read_bytes(), (src / "tokenizer.json").read_bytes())

    def test_clean_strips_padding_and_truncation_deterministically(self):
        src = self.snapshot("padded", True, True)
        a = tokenizer.prepare(src, self.tmp / "a" / "tokenizer.json", mode="clean")
        b = tokenizer.prepare(src, self.tmp / "b" / "tokenizer.json", mode="clean")
        raw = json.loads(a.read_text())
        self.assertIsNone(raw["padding"])
        self.assertIsNone(raw["truncation"])
        self.assertEqual(a.read_bytes(), b.read_bytes())
        plain = self.snapshot("plain", False, False)
        self.assertEqual(a.read_bytes(), (plain / "tokenizer.json").read_bytes())

    def test_verbatim_keeps_padding(self):
        src = self.snapshot("padded", True, True)
        out = tokenizer.prepare(src, self.tmp / "v" / "tokenizer.json", mode="verbatim")
        self.assertEqual(out.read_bytes(), (src / "tokenizer.json").read_bytes())

    def test_expected_sha256_mismatch_fails(self):
        src = self.snapshot("plain", False, False)
        with self.assertRaises(SystemExit):
            tokenizer.prepare(src, self.tmp / "x" / "tokenizer.json", expected_sha256="0" * 64)

    def test_snapshot_revision(self):
        self.assertEqual(tokenizer.snapshot_revision("/c/models--a--b/snapshots/abc123"), "abc123")
        self.assertIsNone(tokenizer.snapshot_revision("/somewhere/else"))


class CaseFeed(unittest.TestCase):
    def test_right_padded_int32_with_token_types(self):
        case = core.Case(ids=[101, 7, 102], ref=np.zeros(1), extra={"token_type_ids": [0, 1, 1]})
        feed = case.feed(6, core.text_ports(token_type_ids=True))
        self.assertEqual(feed["input_ids"].tolist(), [[101, 7, 102, 0, 0, 0]])
        self.assertEqual(feed["attention_mask"].tolist(), [[1, 1, 1, 0, 0, 0]])
        self.assertEqual(feed["token_type_ids"].tolist(), [[0, 1, 1, 0, 0, 0]])
        self.assertTrue(all(a.dtype == np.int32 for a in feed.values()))

    def test_pad_ids(self):
        case = core.Case(ids=[1, 2], ref=np.zeros(1))
        feed = case.feed(4, core.text_ports(), pad_ids=[9, 8, 7, 6])
        self.assertEqual(feed["input_ids"].tolist(), [[1, 2, 9, 8]])

    def test_bucket_of(self):
        self.assertEqual(core.bucket_of(64, [64, 128]), 64)
        self.assertEqual(core.bucket_of(65, [128, 64]), 128)


class ProblemType(unittest.TestCase):
    def cfg(self, **kw):
        return SimpleNamespace(problem_type=kw.get("problem_type"), **{k: v for k, v in kw.items()
                                                                      if k != "problem_type"})

    def test_vllm_rule(self):
        self.assertEqual(manifest.vllm_problem_type(self.cfg(problem_type="regression")), "regression")
        self.assertEqual(manifest.vllm_problem_type(
            self.cfg(sbert_ce_default_activation_function="torch.nn.modules.linear.Identity")), "regression")
        self.assertEqual(manifest.vllm_problem_type(
            self.cfg(sentence_transformers={"activation_fn": "torch.nn.modules.activation.Sigmoid"})),
            "single_label")
        self.assertEqual(manifest.vllm_problem_type(self.cfg()), "single_label")

    def test_classification_rule(self):
        self.assertEqual(manifest.problem_type(self.cfg()), "single_label")
        self.assertEqual(manifest.problem_type(self.cfg(problem_type="multi_label_classification")),
                         "multi_label")


class Metrics(unittest.TestCase):
    def test_non_finite_is_nan(self):
        self.assertTrue(np.isnan(metrics.cosine([1.0, np.nan], [1.0, 0.0])))
        self.assertTrue(np.isnan(metrics.cosine([0.0, 0.0], [1.0, 0.0])))

    def test_worst_keeps_nan(self):
        self.assertTrue(np.isnan(metrics.worst([0.9, float("nan"), 1.0])))
        self.assertTrue(np.isnan(metrics.worst([])))
        self.assertEqual(metrics.worst([0.9, 1.0]), 0.9)

    def test_activate(self):
        p = metrics.activate([0.0, 0.0], "softmax")
        self.assertTrue(np.allclose(p, [0.5, 0.5]))
        self.assertEqual(metrics.activate([2.0], "identity").tolist(), [2.0])


if __name__ == "__main__":
    unittest.main()
