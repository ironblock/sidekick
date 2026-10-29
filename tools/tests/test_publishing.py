"""The publishing tools: the checksum ledger (tools/ledger.py), the licence
audit's validator (tools/licences.py) and the model-card generator
(tools/model_card.py). Standard library only; no model is read.

    python3 -m unittest discover -s tools/tests -t tools -v
"""

import copy
import hashlib
import io
import json
import os
import re
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS))

import ledger  # noqa: E402
import licences  # noqa: E402
import model_card  # noqa: E402
import mteb_transfer  # noqa: E402

SOURCE = "BAAI/bge-small-en-v1.5@5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
MANIFEST = """id = "bge-small-en-v1.5"
backend = "coreml"
artifact = "model_{seq}.mlmodelc"
tokenizer = "tokenizer.json"
dims = 384
buckets = [128, 256]
max_seq_len = 256
"""


def make_build(root, shared=True):
    """A model directory shaped like a two-bucket Core ML install, its
    weights one file hard-linked into both buckets (D49) when `shared`."""
    root = Path(root)
    root.mkdir(parents=True)
    (root / "manifest.toml").write_text(MANIFEST)
    (root / "tokenizer.json").write_text("{}")
    first = None
    for seq in (128, 256):
        prog = root / f"model_{seq}.mlmodelc"
        (prog / "weights").mkdir(parents=True)
        (prog / "model.mil").write_text(f"program {seq}")
        (prog / "coremldata.bin").write_bytes(bytes([seq % 256]) * 8)
        meta = [{"storagePrecision": "Float16", "userDefinedMetadata": {
            "com.github.apple.coremltools.version": "9.0", "com.github.apple.coremltools.conversion_date": "2026-10-01",
            "com.github.apple.coremltools.source": "torch==2.7.0"}}]
        (prog / "metadata.json").write_text(json.dumps(meta))
        weight = prog / "weights" / "weight.bin"
        if shared and first is not None:
            os.link(first, weight)
        else:
            weight.write_bytes(b"weights" * 100 + (b"" if shared else bytes([seq % 256])))
            first = weight
    return root


CHAT_MANIFEST = """id = "minicpm5-2b"
task = "text-generation"
source = { repo = "openbmb/MiniCPM5-2B", revision = "f97400052a43d642bbc6e9975e2397e3ae6a6b52" }
interface = "kv-state-1"
tokenizer = "tokenizer.json"

[decoder]
artifact = "decoder.{chunk}.mlmodelc"
chunks = 2
context_length = 2033

[build]
weights = "palette8"
"""


def make_chat_build(root):
    """A model directory shaped like a two-chunk chat build (D53)."""
    root = Path(root)
    root.mkdir(parents=True)
    (root / "chat.toml").write_text(CHAT_MANIFEST)
    (root / "tokenizer.json").write_text("{}")
    (root / "chat_template.jinja").write_text("{{ messages }}")
    (root / "embedding.safetensors").write_bytes(b"embedding")
    (root / "rope.safetensors").write_bytes(b"rope")
    for i in range(2):
        prog = root / f"decoder.{i}.mlmodelc"
        (prog / "weights").mkdir(parents=True)
        (prog / "model.mil").write_text(f"chunk {i}")
        (prog / "weights" / "weight.bin").write_bytes(bytes([i]) * 64)
    return root


def run(*argv):
    out = io.StringIO()
    old = sys.argv
    sys.argv = ["ledger.py", *argv]
    try:
        with redirect_stdout(out):
            ledger.main()
    finally:
        sys.argv = old
    return out.getvalue()


class LedgerTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.ledger = self.dir / "ledger.json"
        self.build = make_build(self.dir / "build")

    def tearDown(self):
        self.tmp.cleanup()

    def add(self, *extra):
        return run("--ledger", str(self.ledger), "add", str(self.build), "--recorded", "2026-10-23", *extra)

    def entry(self):
        return json.loads(self.ledger.read_text())["builds"][0]

    def test_add_records_a_shared_file_once(self):
        self.add()
        b = self.entry()
        self.assertEqual(b["layout"], "shared")
        self.assertEqual(b["buckets"], [128, 256])
        self.assertEqual(b["source"], SOURCE)
        self.assertEqual(b["produced"]["coremltools"], "9.0")
        self.assertEqual(b["quantization"], "fp16")
        weights = [e for e in b["files"] if e["paths"][0].endswith("weight.bin")]
        self.assertEqual(len(weights), 1)
        self.assertEqual(weights[0]["paths"], ["model_128.mlmodelc/weights/weight.bin",
                                               "model_256.mlmodelc/weights/weight.bin"])
        self.assertTrue(b["id"].startswith("bge-small-en-v1.5/coreml/fp16/"))
        self.assertEqual(ledger.problems(json.loads(self.ledger.read_text()), ledger.load_models()), [])

    def test_separate_layout(self):
        make_build(self.dir / "sep", shared=False)
        run("--ledger", str(self.ledger), "add", str(self.dir / "sep"))
        b = self.entry()
        self.assertEqual(b["layout"], "separate")
        self.assertEqual(sum(1 for e in b["files"] if e["paths"][0].endswith("weight.bin")), 2)

    def test_fingerprint_matches_mteb_transfer(self):
        self.add()
        self.assertEqual(self.entry()["fingerprint"], mteb_transfer.fingerprint(self.build))

    def test_add_is_idempotent(self):
        self.add()
        self.add()
        self.assertEqual(len(json.loads(self.ledger.read_text())["builds"]), 1)

    def test_verify(self):
        self.add()
        self.assertIn("matches", run("--ledger", str(self.ledger), "verify", str(self.build)))
        (self.build / "model_128.mlmodelc" / "model.mil").write_text("changed")
        with self.assertRaises(SystemExit) as e:
            run("--ledger", str(self.ledger), "verify", str(self.build))
        self.assertIn("compiled program: sha256 differs: model_128.mlmodelc/model.mil", str(e.exception))

    def test_verify_names_a_split_shared_file(self):
        self.add()
        w = self.build / "model_256.mlmodelc" / "weights" / "weight.bin"
        w.unlink()
        w.write_bytes(b"other")
        with self.assertRaises(SystemExit) as e:
            run("--ledger", str(self.ledger), "verify", str(self.build))
        self.assertIn("weights: a shared file split", str(e.exception))

    def test_verify_partial(self):
        self.add()
        bid = self.entry()["id"]
        for p in sorted((self.build / "model_256.mlmodelc").rglob("*"), reverse=True):
            p.rmdir() if p.is_dir() else p.unlink()
        (self.build / "model_256.mlmodelc").rmdir()
        with self.assertRaises(SystemExit):
            run("--ledger", str(self.ledger), "verify", str(self.build), "--id", bid)
        self.assertIn("matches", run("--ledger", str(self.ledger), "verify", str(self.build), "--id", bid,
                                     "--partial"))

    def test_sums_give_the_same_build(self):
        self.add()
        hashed = self.entry()
        sums = self.dir / "SHA256SUMS"
        out = subprocess.run(["shasum", "-a", "256"] + [str(p.relative_to(self.dir)) for p in sorted(self.build.rglob("*"))
                                                        if p.is_file()],
                             cwd=self.dir, capture_output=True, text=True, check=True).stdout
        sums.write_text(out.replace("  build/", "  ./build/"))
        other = self.dir / "other.json"
        run("--ledger", str(other), "add", "--sums", str(sums), "--sums-prefix", "build/",
            "--model", "bge-small-en-v1.5", "--variant", "fp16")
        b = json.loads(other.read_text())["builds"][0]
        self.assertEqual(b["id"], hashed["id"])
        self.assertEqual(b["evidence"], "sha256sums")
        self.assertTrue(all(e["size"] is None for e in b["files"]))
        self.assertEqual(ledger.problems(json.loads(other.read_text()), ledger.load_models()), [])

    def test_grades_from_matching_reports(self):
        self.add()
        files = ledger.flatten(self.entry()["files"])
        weights = ledger.bucket_weights(files)
        reports = self.dir / "reports"
        reports.mkdir()
        rec = {"id": "bge-small-en-v1.5", "buckets": [{"bucket": b, "weights": w} for b, w in weights.items()],
               "paths": [{"path": "ane", "letter": "A"}, {"path": "gpu", "letter": "B"}]}
        (reports / "match.json").write_text(json.dumps({"chip": "Apple M1 Max", "macos": "macOS 27.0", "models": [rec]}))
        wrong = copy.deepcopy(rec)
        wrong["buckets"][0]["weights"] = "000000000000"
        (reports / "other.json").write_text(json.dumps({"chip": "Apple M1 Max", "models": [wrong]}))
        grades = ledger.matching_grades("bge-small-en-v1.5", files, [reports], any_report=True)
        self.assertEqual([(g["path"], g["grade"], g["report"]) for g in grades],
                         [("coreml-ane", "A", "match.json"), ("coreml-gpu", "B", "match.json")])
        self.assertEqual(ledger.matching_grades("bge-small-en-v1.5", files, [reports]), [])

    def test_onnx_grades_from_matching_file_hashes(self):
        sha = {name: hashlib.sha256(name.encode()).hexdigest() for name in ("graph", "data", "tok", "manifest")}
        files = {"model.onnx": (sha["graph"], 10), "model.onnx_data": (sha["data"], 20),
                 "tokenizer.json": (sha["tok"], 5), "manifest.toml": (sha["manifest"], 1)}
        reports = self.dir / "reports"
        reports.mkdir()

        def report(name, recorded):
            rec = {"id": "bge-small-en-v1.5", "buckets": [], "paths": [{"path": "onnx-cpu", "letter": "A"}],
                   "files": [{"path": p, "sha256": s} for p, s in recorded.items()]}
            (reports / name).write_text(json.dumps({"chip": "Apple M1 Max", "macos": "macOS 27.0", "models": [rec]}))

        report("match.json", {"model.onnx": sha["graph"], "model.onnx_data": sha["data"], "tokenizer.json": sha["tok"]})
        report("other-weights.json", {"model.onnx": sha["graph"], "model.onnx_data": "0" * 64})
        report("other-path.json", {"int8/model.onnx": sha["graph"]})
        report("no-graph.json", {"tokenizer.json": sha["tok"]})
        (reports / "no-hashes.json").write_text(json.dumps({"chip": "Apple M1 Max", "models": [
            {"id": "bge-small-en-v1.5", "buckets": [], "paths": [{"path": "onnx-cpu", "letter": "A"}]}]}))
        grades = ledger.matching_grades("bge-small-en-v1.5", files, [reports], any_report=True)
        self.assertEqual([(g["path"], g["grade"], g["report"]) for g in grades], [("onnx-cpu", "A", "match.json")])

    def test_chat_build_and_its_d54_grade(self):
        chat = make_chat_build(self.dir / "chat")
        reports = self.dir / "reports"
        reports.mkdir()
        graded = [{"path": rel, "sha256": hashlib.sha256((chat / rel).read_bytes()).hexdigest()}
                  for rel in sorted(str(f.relative_to(chat)) for f in chat.rglob("*") if f.is_file())
                  if rel != "chat.toml"]
        fam = {"positions": 10, "kl_mean": 1e-3, "kl_p99": 1e-2, "top1": 0.99, "grade": "B"}
        report = {"rubric": "D54", "model": "minicpm5-2b", "chip": "Apple M1 Max", "macos": "macOS 27.0 (26A428)",
                  "files": graded, "grade": "B", "stable": True,
                  "paths": {"generation": {"families": {"prose": fam}},
                            "prompt": {"families": {"prose": dict(fam, kl_mean=2e-3)}}}}
        (reports / "causal-lm-minicpm5-2b.json").write_text(json.dumps(report))
        stale = dict(report, files=[dict(graded[0], sha256="0" * 64)] + graded[1:])
        (reports / "stale.json").write_text(json.dumps(stale))
        task = self.dir / "task-check.json"
        task.write_text(json.dumps({"passed": True, "families": {"titles": {
            "metric": "rouge_l", "candidate": 0.45, "reference": 0.44, "difference": 0.01, "passed": True, "n": 20,
            "identical_replies": 0.95, "rule": "ROUGE-L, at most 3 points below"}}}))
        run("--ledger", str(self.ledger), "add", str(chat), "--reports", str(reports), "--any-report",
            "--task-check", str(task))
        b = self.entry()
        self.assertEqual((b["variant"], b["quantization"], b["buckets"], b["chunks"], b["max_seq_len"]),
                         ("palette8", "8-bit palettes", [], 2, 2033))
        self.assertTrue(b["id"].startswith("minicpm5-2b/coreml/palette8/"))
        [g] = b["grades"]
        self.assertEqual((g["path"], g["grade"], g["rubric"], g["report"]), ("coreml-ane", "B", "D54",
                                                                            "causal-lm-minicpm5-2b.json"))
        self.assertEqual(g["families"]["prose"]["kl_mean"], 2e-3)     # the worse of the two decoder paths
        self.assertTrue(g["task_check"]["passed"])
        self.assertEqual(ledger.problems(json.loads(self.ledger.read_text()), ledger.load_models()), [])
        self.assertIn("matches", run("--ledger", str(self.ledger), "verify", str(chat)))
        (chat / "embedding.safetensors").write_bytes(b"other")
        with self.assertRaises(SystemExit) as e:
            run("--ledger", str(self.ledger), "verify", str(chat))
        self.assertIn("host tables: sha256 differs", str(e.exception))

    def test_check_refuses_a_tampered_entry(self):
        self.add()
        data = json.loads(self.ledger.read_text())
        data["builds"][0]["files"][0]["sha256"] = "0" * 64
        found = ledger.problems(data, ledger.load_models())
        self.assertTrue(any("digest doesn't match" in p for p in found), found)

    def test_published(self):
        self.add()
        bid = self.entry()["id"]
        run("--ledger", str(self.ledger), "published", bid, "--repo", "owner/bge-small-en-v1.5-coreml",
            "--revision", "abc", "--sidekick", "0.16.0", "--date", "2026-10-24")
        self.assertEqual(self.entry()["published"][0]["revision"], "abc")
        self.assertEqual(ledger.problems(json.loads(self.ledger.read_text()), ledger.load_models()), [])


class CommittedDataTest(unittest.TestCase):
    def test_ledger_is_valid(self):
        self.assertEqual(ledger.problems(ledger.load(), ledger.load_models()), [])

    def test_licences_are_valid(self):
        self.assertEqual(licences.problems(), [])


def entry(**kw):
    base = {
        "source": "a/b@" + "0" * 40, "licence": "MIT", "licence_sources": ["https://example.org/licence"],
        "base_model": "", "base_model_licence": "", "base_model_licence_sources": [],
        "training_disclosure": "full",
        "training_data": [{"name": "d", "licence": "MIT", "terms": "permissive",
                           "source": "https://example.org/d", "stage": "fine-tune"}],
        "base_pretraining": "", "custom_terms": [], "concerns": [], "tier": "1", "reason": "r",
        "unverified": [], "read": "2026-10-21", "licence_text": "mit.txt",
    }
    base.update(kw)
    return {k: v for k, v in base.items() if v is not None}


class LicencesTest(unittest.TestCase):
    def check(self, e):
        return licences.problems({"schema": licences.SCHEMA, "models": {"m": e}}, {"m": {"source": e["source"]}})

    def test_tier_one(self):
        self.assertEqual(self.check(entry()), [])

    def test_floor(self):
        sa = [{"name": "s", "licence": "CC-BY-SA-4.0", "terms": "share-alike", "source": "https://e.org", "stage": "x"}]
        nc = [{"name": "n", "licence": "CC-BY-NC-4.0", "terms": "non-commercial", "source": "https://e.org", "stage": "x"}]
        self.assertEqual(licences.floor_tier(entry(training_data=sa)), "2a")
        self.assertEqual(licences.floor_tier(entry(training_disclosure="partial")), "2a")
        self.assertEqual(licences.floor_tier(entry(concerns=["unnamed-synthetic-generator"])), "2a")
        # non-commercial training data is disclosed, not gated: only travelling terms are 2b
        self.assertEqual(licences.floor_tier(entry(training_data=nc, concerns=["non-commercial-data"])), "2a")
        self.assertEqual(licences.floor_tier(entry(concerns=["use-policy"])), "2b")
        self.assertEqual(licences.floor_tier(entry(concerns=["custom-terms"])), "2b")
        found = self.check(entry(training_data=nc, concerns=["non-commercial-data"], tier="2b"))
        self.assertTrue(any("puts it at 2a" in p for p in found), found)

    def test_an_unverified_licence_counts_as_unknown_and_is_named(self):
        unv = [{"name": "u", "licence": "unverified", "terms": "unknown", "source": "https://e.org", "stage": "x"}]
        tier, reasons = licences.floor_verdict(entry(training_data=unv))
        self.assertEqual((tier, reasons), ("2a", ["u: licence unverified"]))
        self.assertEqual(licences.floor_verdict(entry()), ("1", []))
        unv[0]["terms"] = "permissive"
        found = self.check(entry(training_data=unv, tier="2a"))
        self.assertTrue(any("unverified licence has unknown terms" in p for p in found), found)

    def test_tier_must_follow_the_data(self):
        sa = [{"name": "s", "licence": "CC-BY-SA-4.0", "terms": "share-alike", "source": "https://e.org", "stage": "x"}]
        found = self.check(entry(training_data=sa, concerns=["copyleft-data"], tier="1"))
        self.assertTrue(any("puts it at 2a" in p for p in found), found)
        self.assertEqual(self.check(entry(training_data=sa, concerns=["copyleft-data"], tier="2a")), [])

    def test_concerns_must_match_the_data(self):
        nc = [{"name": "n", "licence": "CC-BY-NC-4.0", "terms": "non-commercial", "source": "https://e.org", "stage": "x"}]
        found = self.check(entry(training_data=nc, tier="2a"))
        self.assertTrue(any("non-commercial data" in p for p in found), found)

    def test_hub_ids(self):
        d = [{"name": "MS MARCO", "licence": "MS MARCO terms (non-commercial research only)", "terms": "non-commercial",
              "hub": ["microsoft/ms_marco"], "source": "https://github.com/microsoft/msmarco", "stage": "x"},
             {"name": "NQ", "licence": "CC-BY-SA-3.0", "terms": "share-alike", "stage": "x",
              "source": "https://huggingface.co/datasets/google-research-datasets/natural_questions/blob/" + "0" * 40 + "/README.md"}]
        self.assertEqual(licences.hf_datasets(entry(training_data=d)),
                         ["microsoft/ms_marco", "google-research-datasets/natural_questions"])
        d[0]["hub"] = ["not an id"]
        found = self.check(entry(training_data=d, concerns=["non-commercial-data", "copyleft-data"], tier="2a"))
        self.assertTrue(any("hub must be a list" in p for p in found), found)

    def test_tier_three_needs_its_terms(self):
        found = self.check(entry(tier="3", licence="L", licence_text=None))
        self.assertTrue(any("tier 3" in p for p in found), found)
        terms = [{"name": "L", "url": "https://e.org/l", "requires": "a revenue cap"}]
        self.assertEqual(self.check(entry(tier="3", licence="L", licence_text=None, custom_terms=terms,
                                          concerns=["custom-terms"])), [])
        found = self.check(entry(tier="3", licence="L", custom_terms=terms, concerns=["custom-terms"]))
        self.assertTrue(any("ships nothing" in p for p in found), found)

    def test_licence_text(self):
        found = self.check(entry(licence_text=None))
        self.assertTrue(any("set licence_text" in p for p in found), found)
        found = self.check(entry(licence_text="apache-2.0.txt"))
        self.assertTrue(any("is the Apache-2.0 text" in p for p in found), found)
        found = self.check(entry(licence_text="nowhere.txt"))
        self.assertTrue(any("isn't in docs/data/licence-texts/index.toml" in p for p in found), found)
        self.assertEqual(licences.text_problems(), [])

    def test_page_licence_id(self):
        # the comparison page's short id: SPDX ids as they are, custom licences by a one-word name
        self.assertEqual(licences.short_id(entry()), "MIT")
        self.assertEqual(licences.short_id(entry(licence="Gemma Terms of Use")), "Gemma")
        found = self.check(entry(tier="3", licence="Some Custom Licence", licence_text=None,
                                 custom_terms=[{"name": "L", "url": "https://e.org/l", "requires": "x"}],
                                 concerns=["custom-terms"]))
        self.assertTrue(any("no short id" in p for p in found), found)

    def test_licence_url_prefers_the_licence_text(self):
        card = "https://huggingface.co/a/b/blob/" + "0" * 40 + "/README.md"
        text = "https://huggingface.co/a/b/blob/" + "0" * 40 + "/LICENSE"
        self.assertEqual(licences.licence_url(entry(licence_sources=[card, text])), text)
        self.assertEqual(licences.licence_url(entry(licence_sources=[card])), card)

    def test_links_must_be_pinned(self):
        found = self.check(entry(licence_sources=["https://huggingface.co/a/b/blob/main/README.md"]))
        self.assertTrue(any("isn't pinned" in p for p in found), found)


BGE_COREML = "bge-small-en-v1.5/coreml/fp16/f1b85f22e0b4"


class ModelCardTest(unittest.TestCase):
    def test_tier_2a_card(self):
        card = model_card.render("nli-minilm2-l6-h768")[0]
        self.assertTrue(card.startswith("---\nlicense: apache-2.0\n"))
        self.assertIn("library_name: coreml", card)
        self.assertIn("**Licence: Apache-2.0 · publishing tier 2a.**", card)
        self.assertNotIn("extra_gated_prompt", card)
        self.assertIn("## Checksums", card)

    def test_front_matter_is_complete(self):
        card = model_card.render("bge-small-en-v1.5", build_ids=[BGE_COREML])[0]
        front = card.split("---\n")[1]
        for key in ("license: mit", "base_model: BAAI/bge-small-en-v1.5", "base_model_relation: quantized",
                    "library_name: coreml", "pipeline_tag: feature-extraction", "tags:"):
            self.assertIn(key, front)

    def test_card_holds_only_what_is_particular_to_it(self):
        card = model_card.render("bge-small-en-v1.5", build_ids=[BGE_COREML], ref="v9.9.9")[0]
        sections = re.findall(r"^## (.+)$", card, re.M)
        self.assertEqual(sections, ["Download", "Fidelity", "Benchmark scores", "Training data", "Original model",
                                    "Checksums"])
        # one fetch command, nothing general copied in: no install steps, no grade scale, no convert recipe
        self.assertEqual(card.count("```sh"), 1)
        self.assertIn("sidekickd fetch bge-small-en-v1.5\n", card)
        for general in ("hf download", "shasum -a 256 -c ../", "symlink", "0.9999 (indistinguishable",
                        "tools/ledger.py verify", "cpu_and_ne", "max_seq_len", 'layout = "shared"'):
            self.assertNotIn(general, card)
        # the shared weights file once, with its three paths, inside a collapsed block
        self.assertIn("<details><summary><code>fp16/</code>: ledger build <code>" + BGE_COREML, card)
        self.assertIn("<br>`model_256.mlmodelc/weights/weight.bin`", card)
        # docs links pinned to the ref
        links = re.findall(r"\(https://github\.com/ironblock/sidekick/blob/([^/]+)/", card)
        self.assertTrue(links)
        self.assertEqual(set(links), {"v9.9.9"})
        self.assertIn("Needs sidekick 9.9.9 or later.", card)

    def test_grades_of_these_files_are_marked(self):
        card = model_card.render("bge-small-en-v1.5", build_ids=[BGE_COREML])[0]
        self.assertIn("| Neural Engine (default) | Apple M1 Max | macOS 27.0 | **A** | 0.99997 | This build |", card)
        self.assertIn("| GPU | Intel Xeon W-3235 | macOS 26.6.2 | **A** | 0.999999 | Unconfirmed |", card)

    def test_doc_links_resolve(self):
        self.assertEqual(model_card.link_problems(model_card.DOCS), [])

    def test_slug(self):
        self.assertEqual(model_card.slug("D49 — One weights file per chunk, shared by every bucket"),
                         "d49--one-weights-file-per-chunk-shared-by-every-bucket")
        self.assertEqual(model_card.slug("ONNX Runtime on the CPU (`onnx-cpu`)"), "onnx-runtime-on-the-cpu-onnx-cpu")
        self.assertEqual(model_card.slug("Classifiers (`/v1/classify`)"), "classifiers-v1classify")

    def test_non_commercial_data_is_a_notice_not_a_gate(self):
        card = model_card.render("qwen3-embedding-0.6b")[0]
        front = card.split("---\n")[1]
        self.assertIn("licence-tier-2a", front)
        self.assertNotIn("extra_gated", front)
        self.assertIn("- microsoft/ms_marco\n", front)
        self.assertIn("- google-research-datasets/natural_questions\n", front)
        self.assertNotIn("--accept-terms", card)
        self.assertIn("**Training data with non-commercial terms.**", card)
        self.assertIn("**The data:** MS MARCO (non-commercial research only).", card)
        self.assertNotIn("MS MARCO: MS MARCO terms", card)
        self.assertIn("legally unsettled", card)
        self.assertIn("gives no assurance", card)
        self.assertIn("these files are distributed under Apache-2.0, as the original model is", card)
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            model_card.write(Path(tmp) / "q", "qwen3-embedding-0.6b", "onnx", None, None)
            self.assertFalse((Path(tmp) / "q" / "TERMS.md").exists())
            model_card.write(Path(tmp) / "g", "embeddinggemma-300m", "coreml", None, None)
            self.assertTrue((Path(tmp) / "g" / "TERMS.md").exists())

    def test_tier_2b_card_is_gated(self):
        card = model_card.render("embeddinggemma-300m")[0]
        self.assertIn("license: gemma", card)
        self.assertIn("extra_gated_prompt: |-", card)
        self.assertIn("Gemma Prohibited Use Policy", card)
        self.assertIn("--accept-terms embeddinggemma-300m", card)

    def test_onnx_card_lists_the_publishers_export(self):
        card = model_card.render("bge-small-en-v1.5", "onnx")[0]
        self.assertIn("library_name: onnx", card)
        self.assertIn("828e1496d7fabb79cfa4dcd84fa38625c0d3d21da474a00f08db0f559940cf35", card)

    def test_licence_files_written(self):
        texts = TOOLS.parent / "docs" / "data" / "licence-texts"
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            model_card.write(Path(tmp) / "bge", "bge-small-en-v1.5", "coreml", [BGE_COREML], "o")
            self.assertEqual((Path(tmp) / "bge" / "LICENSE").read_bytes(),
                             (texts / "flagopen-flagembedding.LICENSE").read_bytes())
            self.assertFalse((Path(tmp) / "bge" / "NOTICE").exists())
            model_card.write(Path(tmp) / "lumma", "lumma-fev-0.1b", "coreml", None, "o")
            self.assertEqual((Path(tmp) / "lumma" / "NOTICE").read_bytes(),
                             (texts / "frontiersmind-lumma-fev-0.1b.NOTICE").read_bytes())
            model_card.write(Path(tmp) / "roberta", "roberta-large-mnli", "coreml", None, "o")
            mit = (Path(tmp) / "roberta" / "LICENSE").read_text()
            self.assertIn("Copyright (c) the authors of FacebookAI/roberta-large-mnli", mit)
            self.assertNotIn("<copyright holders>", mit)
        card = model_card.render("bge-small-en-v1.5", build_ids=[BGE_COREML])[0]
        self.assertIn("| Licence file | `LICENSE`: the upstream licence text, with its copyright line, from "
                      "[FlagOpen/FlagEmbedding](https://github.com/FlagOpen/FlagEmbedding/blob/", card)

    def test_owner(self):
        self.assertEqual(model_card.owner(), "sidekickd")
        self.assertEqual(model_card.repo_id(model_card.owner(), "bge-small-en-v1.5", "coreml"),
                         "sidekickd/bge-small-en-v1.5-coreml")

    def test_tier_three_is_refused(self):
        with self.assertRaises(SystemExit):
            model_card.render("lfm2.5-embedding-350m")

    def test_every_card_renders(self):
        with redirect_stdout(io.StringIO()):
            model_card.check()


if __name__ == "__main__":
    unittest.main()
