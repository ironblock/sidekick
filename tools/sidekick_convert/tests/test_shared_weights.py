"""One weights file per chunk across buckets (shared_weights.py, D49).

The hard-linking and the program comparison need nothing but the standard
library. The re-export tests build two tiny ML programs with coremltools'
MIL builder (no torch, no download): a weight both buckets hold and a table
of each bucket's own, as a model's positions are. They need coremltools
and Xcode's coremlcompiler, and are skipped without them.
"""

import errno
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from sidekick_convert import shared_weights


def _artifact(root, name, data):
    a = Path(root) / name
    (a / "weights").mkdir(parents=True)
    (a / "weights" / "weight.bin").write_bytes(data)
    return a


class KeepPackages(unittest.TestCase):
    def test_kept_packages_hold_one_copy_of_identical_weights(self):
        with tempfile.TemporaryDirectory() as d:
            root, keep = Path(d), Path(d) / "keep"
            for seq in (128, 256):
                pkg = root / f"model_{seq}.mlpackage"
                (pkg / "Data/com.apple.CoreML/weights").mkdir(parents=True)
                (pkg / "Data/com.apple.CoreML/weights/weight.bin").write_bytes(b"u" * 64)
                (pkg / "Manifest.json").write_text("{}")
                shared_weights.keep_package(pkg, keep, "shared")
                shutil.rmtree(pkg)      # what the converter does next
            shared_weights.keep_package(root / "model_512.mlpackage", None, "separate")   # no directory: nothing
            said = []
            shared_weights.link_kept(keep, log=said.append)
            a, b = (keep / "shared" / f"model_{s}.mlpackage/Data/com.apple.CoreML/weights/weight.bin"
                    for s in (128, 256))
            self.assertTrue(os.path.samefile(a, b))
            self.assertTrue((keep / "shared/model_128.mlpackage/Manifest.json").exists())
            self.assertFalse((keep / "separate").exists())
            self.assertIn("kept 2 .mlpackage source(s)", said[-1])


class LinkIdentical(unittest.TestCase):
    def test_identical_weight_files_become_one_file(self):
        with tempfile.TemporaryDirectory() as d:
            a = _artifact(d, "model_128.mlmodelc", b"w" * 4096)
            b = _artifact(d, "model_256.mlmodelc", b"w" * 4096)
            c = _artifact(d, "model_512.mlmodelc", b"x" * 4096)
            linked, saved = shared_weights.link_identical([a, b, c], log=lambda m: None)
            self.assertEqual((linked, saved), (1, 4096))
            self.assertTrue(os.path.samefile(a / "weights/weight.bin", b / "weights/weight.bin"))
            self.assertFalse(os.path.samefile(a / "weights/weight.bin", c / "weights/weight.bin"))
            self.assertEqual((b / "weights/weight.bin").read_bytes(), b"w" * 4096)
            self.assertEqual(shared_weights.files_bytes(
                [p / "weights/weight.bin" for p in (a, b, c)]), 2 * 4096)
            # already linked: nothing to do
            self.assertEqual(shared_weights.link_identical([a, b, c], log=lambda m: None), (0, 0))

    def test_same_size_different_bytes_stay_apart(self):
        with tempfile.TemporaryDirectory() as d:
            a = _artifact(d, "model_128.mlmodelc", b"a" * 100)
            b = _artifact(d, "model_256.mlmodelc", b"b" * 100)
            self.assertEqual(shared_weights.link_identical([a, b], log=lambda m: None), (0, 0))

    def test_a_refused_link_keeps_the_copy(self):
        with tempfile.TemporaryDirectory() as d:
            a = _artifact(d, "model_128.mlmodelc", b"w" * 64)
            b = _artifact(d, "model_256.mlmodelc", b"w" * 64)
            said = []
            with mock.patch("os.link", side_effect=OSError(errno.EXDEV, "Cross-device link")):
                self.assertEqual(shared_weights.link_identical([a, b], log=said.append), (0, 0))
            self.assertFalse(os.path.samefile(a / "weights/weight.bin", b / "weights/weight.bin"))
            self.assertEqual((b / "weights/weight.bin").read_bytes(), b"w" * 64)
            self.assertIn("keeping the copy", said[0])
            self.assertEqual(sorted(p.name for p in (b / "weights").iterdir()), ["weight.bin"])

    def test_the_command_links_an_installed_model(self):
        with tempfile.TemporaryDirectory() as d:
            a = _artifact(d, "model_128.0.mlmodelc", b"w" * 64)
            b = _artifact(d, "model_256.0.mlmodelc", b"w" * 64)
            (Path(d) / "manifest.toml").write_text("")
            with mock.patch("builtins.print"):
                shared_weights.main(["link", d])
            self.assertTrue(os.path.samefile(a / "weights/weight.bin", b / "weights/weight.bin"))


class ProgramText(unittest.TestCase):
    def test_only_weight_offsets_are_set_aside(self):
        line = ('tensor<fp16, [4]> w = const()[name = string("w"), val = tensor<fp16, [4]>(BLOBFILE(path = '
                'string("@model_path/weights/weight.bin"), offset = uint64({}))))];\n')
        with tempfile.TemporaryDirectory() as d:
            for name, offset in (("a", 64), ("b", 4160)):
                (Path(d) / name).mkdir()
                (Path(d) / name / "model.mil").write_text(line.format(offset))
            (Path(d) / "c").mkdir()
            (Path(d) / "c" / "model.mil").write_text(line.format(64).replace("[4]", "[5]"))
            text = shared_weights.program_text
            self.assertEqual(text(Path(d) / "a"), text(Path(d) / "b"))
            self.assertNotEqual(text(Path(d) / "a"), text(Path(d) / "c"))


def _have_coreml():
    try:
        import coremltools  # noqa: F401
        from coremltools import libmilstoragepython  # noqa: F401
    except ImportError:
        return False
    return shutil.which("xcrun") is not None and subprocess.run(
        ["xcrun", "--find", "coremlcompiler"], capture_output=True).returncode == 0


@unittest.skipUnless(_have_coreml(), "needs coremltools and Xcode's coremlcompiler")
class Reexport(unittest.TestCase):
    """Two buckets of a tiny program: y = (x + table[seq]) @ W."""

    @classmethod
    def setUpClass(cls):
        import coremltools as ct
        from coremltools.converters.mil import Builder as mb
        from sidekick_convert import core
        cls.dir = tempfile.TemporaryDirectory()
        root = Path(cls.dir.name)
        rng = np.random.default_rng(0)
        w = rng.standard_normal((16, 32)).astype(np.float16)
        cls.programs = {}
        for seq in (4, 8):
            table = rng.standard_normal((1, seq, 16)).astype(np.float16)

            @mb.program(input_specs=[mb.TensorSpec(shape=(1, seq, 16), dtype=ct.converters.mil.mil.types.fp16)],
                        opset_version=ct.target.macOS15)
            def prog(x):
                return mb.matmul(x=mb.add(x=x, y=table), y=w, name="y")

            model = ct.convert(prog, convert_to="mlprogram", minimum_deployment_target=ct.target.macOS15,
                               compute_precision=ct.precision.FLOAT16, skip_model_load=True)
            pkg = root / f"model_{seq}.mlpackage"
            model.save(str(pkg))
            compiled = core.compile_mlmodelc(pkg, root / f"compiled_{seq}")
            cls.programs[seq] = shared_weights.Program(seq, None, compiled, pkg, ("x",), "y")
        cls.separate_sizes = {s: (p.compiled / "weights/weight.bin").stat().st_size for s, p in cls.programs.items()}
        group = [cls.programs[s] for s in (4, 8)]
        cls.keep = root / "keep"
        cls.union, cls.union_bytes = shared_weights.build_union(group, root / "union", core.compile_mlmodelc,
                                                                log=lambda m: None, keep=cls.keep)

    @classmethod
    def tearDownClass(cls):
        cls.dir.cleanup()

    def test_every_bucket_compiles_to_one_weights_file(self):
        a, b = (self.union[s] / "weights/weight.bin" for s in (4, 8))
        self.assertTrue(os.path.samefile(a, b))
        self.assertEqual(a.stat().st_size, self.union_bytes)
        # W once, and each bucket's table once
        self.assertLess(self.union_bytes, sum(self.separate_sizes.values()))
        self.assertGreater(self.union_bytes, max(self.separate_sizes.values()))
        for p in self.programs.values():
            self.assertFalse(p.package.exists(), "the separate package is deleted once re-exported")

    def test_kept_packages_are_the_ones_compiled(self):
        for s in (4, 8):
            kept = self.keep / "shared" / f"model_{s}.mlpackage" / "Data/com.apple.CoreML/weights/weight.bin"
            self.assertEqual(kept.read_bytes(), (self.union[s] / "weights/weight.bin").read_bytes())

    def test_programs_are_the_separate_ones_but_for_offsets(self):
        for s in (4, 8):
            self.assertEqual(shared_weights.program_text(self.union[s]),
                             shared_weights.program_text(self.programs[s].compiled))

    def test_outputs_are_bit_identical_in_a_child_process(self):
        for s in (4, 8):
            feed = {"x": np.random.default_rng(s).standard_normal((1, s, 16)).astype(np.float16)}
            union = [shared_weights.Program(s, None, self.union[s], None, ("x",), "y")]
            r = shared_weights.compare([self.programs[s]], union, [feed], "CPU_ONLY")
            self.assertTrue(r["identical"], r)
            self.assertEqual(r["outputs"], 1)


if __name__ == "__main__":
    unittest.main()
