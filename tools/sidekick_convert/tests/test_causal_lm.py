"""The causal-LM converter (causal_lm.py, backbones/llama_causal.py,
techniques/gptq.py), torch only: on a tiny random LlamaForCausalLM saved
like a checkpoint, the rewrites (R1/R2 rotations, the residual 1/K, norm
pre-scales, StableSilu) reproduce transformers' fp32 forward; the chunk
programs called T tokens at a time through the kv-state-1 contract
reproduce the whole-sequence forward; GPTQ stays on its palettes and beats
round to nearest; calibration draws, grading and the committed fixtures and
manifest follow their rules. No Core ML and no download.
"""

import json
import math
import tempfile
import tomllib
import unittest
from pathlib import Path

import numpy as np
import torch

from sidekick_convert import causal_lm as L
from sidekick_convert.backbones import llama_causal as LC
from sidekick_convert.techniques import gptq as G


def tiny_llama(dirname, seed=0, layers=3, **overrides):
    """A random LlamaForCausalLM (hidden 64, 4 heads over 2 KV heads,
    head_dim 16), saved as a checkpoint, with a massive activation's worth
    of range: one channel of the embedding is 40 times the others."""
    from transformers import LlamaConfig, LlamaForCausalLM
    torch.manual_seed(seed)
    cfg = LlamaConfig(**{**dict(vocab_size=128, hidden_size=64, intermediate_size=128, num_hidden_layers=layers,
                                num_attention_heads=4, num_key_value_heads=2, head_dim=16,
                                max_position_embeddings=256, rope_theta=10000.0, rms_norm_eps=1e-6,
                                tie_word_embeddings=False), **overrides})
    model = LlamaForCausalLM(cfg).eval()
    with torch.no_grad():
        for p in model.parameters():
            if p.dim() == 1:
                p.copy_(1.0 + 0.2 * torch.randn_like(p))
        model.model.embed_tokens.weight[:, 5] *= 40.0
    model.save_pretrained(dirname, safe_serialization=True)
    return model


class Rewrites(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_grad_enabled(False)
        cls.dir = tempfile.TemporaryDirectory()
        cls.hf = tiny_llama(cls.dir.name)
        cls.ck = LC.Checkpoint(cls.dir.name)
        cls.ids = [[0] + list(range(5, 40)), [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9, 7, 9]]

    @classmethod
    def tearDownClass(cls):
        cls.dir.cleanup()

    def reference(self, ids):
        return self.hf(torch.tensor([ids])).logits[0].double().numpy()

    def rewritten(self, rot, ranges):
        table = LC.embedding_table(self.ck, rot, ranges.K if ranges else 1)
        hs = LC.stream(self.ck, self.ids, lambda i: LC.build_layer(self.ck, i, rot, ranges), table)
        norm, head = LC.final_norm(self.ck, rot, ranges), LC.head_weight(self.ck, rot)
        return [LC.logits(norm, head, h).double().numpy() for h in hs]

    def assertSameLogits(self, got):
        for ids, g in zip(self.ids, got):
            ref = self.reference(ids)
            cos = (g * ref).sum(1) / np.linalg.norm(g, axis=1) / np.linalg.norm(ref, axis=1)
            self.assertGreater(cos.min(), 0.999999)
            self.assertTrue((g.argmax(1) == ref.argmax(1)).all())

    def test_the_channels_first_model_is_the_checkpoint(self):
        self.assertSameLogits(self.rewritten(None, None))

    def test_rotations_residual_scale_and_norm_prescales_are_exact(self):
        rot = LC.Rotation(seed=3).bind(self.ck)
        ranges = LC.calibrate_ranges(self.ck, self.ids, rot, K=4, report=lambda *_: None)
        self.assertEqual(ranges.K, 4)
        self.assertSameLogits(self.rewritten(rot, ranges))
        forced = LC.Ranges(K=2, norm_scale=[0.25] * (2 * self.ck.layers + 1), norm_max=[], linear_out_max=[])
        self.assertSameLogits(self.rewritten(rot, forced))

    def test_the_rotation_spreads_the_massive_activation(self):
        plain = LC.calibrate_ranges(self.ck, self.ids, None, report=lambda *_: None)
        rotated = LC.calibrate_ranges(self.ck, self.ids, LC.Rotation(seed=0).bind(self.ck), report=lambda *_: None)
        self.assertLess(max(rotated.norm_max), max(plain.norm_max))

    def test_norm_prescales_keep_squares_in_range(self):
        ranges = LC.calibrate_ranges(self.ck, self.ids, None, report=lambda *_: None)
        for m, s in zip(ranges.norm_max, ranges.norm_scale):
            self.assertLessEqual((m * s) ** 2, LC.NORM_SQ_MAX)
            self.assertTrue(s == 1.0 or (2 * m * s) ** 2 > LC.NORM_SQ_MAX)
            self.assertEqual(math.log2(s), round(math.log2(s)))

    def test_hadamard_is_orthogonal_and_seeded(self):
        H = LC.hadamard(64, 5)
        self.assertTrue(torch.allclose(H @ H.T, torch.eye(64, dtype=torch.float64), atol=1e-12))
        self.assertTrue(torch.equal(H, LC.hadamard(64, 5)))
        self.assertFalse(torch.equal(H, LC.hadamard(64, 6)))
        with self.assertRaises(ValueError):
            LC.hadamard(48, 0)

    def test_chunks_called_through_the_contract_are_the_whole_sequence(self):
        """Two chunk programs, T = 8 over C = 32: a 20-token prompt in three
        calls (the last padded with token 0), then one token per call, with
        the host's tables and mask, give the logits of the whole-sequence
        forward at every `sel`."""
        ck, T, C = self.ck, 8, 32
        rot = LC.Rotation(seed=1).bind(ck)
        ranges = LC.calibrate_ranges(ck, self.ids, rot, K=2, report=lambda *_: None)
        layers = [LC.build_layer(ck, i, rot, ranges) for i in range(ck.layers)]
        chunks = [LC.Chunk(layers[:2], ck, T, C),
                  LC.Chunk(layers[2:], ck, T, C, LC.final_norm(ck, rot, ranges), LC.head_weight(ck, rot),
                           head_split=2)]
        table = LC.embedding_table(ck, rot, ranges.K)
        cos, sin = LC.rope_tables(ck.theta, ck.head_dim, range(C))
        seq = self.ids[0][:24]
        whole = self.rewritten(rot, ranges)[0][:24]

        def call(tokens, pos, sel):
            ids = tokens + [0] * (T - len(tokens))
            h = LC.embed(table, ids)
            args = (torch.tensor([pos], dtype=torch.int32), LC.mask_for(pos, T, C),
                    torch.from_numpy(cos[pos:pos + T]).float(), torch.from_numpy(sin[pos:pos + T]).float())
            h = chunks[0](h, *args)
            onehot = torch.zeros(T)
            onehot[sel] = 1.0
            out = chunks[1](h, *args, onehot)
            self.assertEqual(tuple(out.shape), (1, ck.vocab))
            return out[0].double().numpy()

        rows = {}
        for pos in (0, 8, 16):
            piece = seq[pos:min(pos + T, 20)]
            for sel in range(len(piece)):     # every row of the call, as `sel` picks it (rewriting the
                rows[pos + sel] = call(piece, pos, sel)       # same cache positions each time)
        for pos in range(20, 24):
            rows[pos] = call([seq[pos]], pos, 0)
        for pos, row in rows.items():
            ref = whole[pos]
            self.assertGreater(float(row @ ref / np.linalg.norm(row) / np.linalg.norm(ref)), 0.99999, pos)

    def test_refuses_what_kv_state_1_does_not_cover(self):
        with tempfile.TemporaryDirectory() as d:
            tiny_llama(d, layers=1, rope_scaling={"rope_type": "linear", "factor": 2.0})
            with self.assertRaisesRegex(ValueError, "rope_scaling"):
                LC.Checkpoint(d)

    def test_reads_transformers_5_rope_parameters(self):
        with tempfile.TemporaryDirectory() as d:
            tiny_llama(d, layers=1)
            cfg = json.loads((Path(d) / "config.json").read_text())
            cfg.pop("rope_theta", None)
            cfg.pop("rope_scaling", None)
            cfg["rope_parameters"] = {"rope_type": "default", "rope_theta": 5e6}
            (Path(d) / "config.json").write_text(json.dumps(cfg))
            self.assertEqual(LC.Checkpoint(d).theta, 5e6)
            cfg["rope_parameters"] = {"rope_type": "yarn", "rope_theta": 5e6, "factor": 4.0}
            (Path(d) / "config.json").write_text(json.dumps(cfg))
            with self.assertRaisesRegex(ValueError, "rope_parameters"):
                LC.Checkpoint(d)

    def test_logits_at_chosen_positions(self):
        rot = LC.Rotation(seed=0).bind(self.ck)
        h = LC.stream(self.ck, self.ids[:1], lambda i: LC.build_layer(self.ck, i, rot), LC.embedding_table(
            self.ck, rot))[0]
        norm, head = LC.final_norm(self.ck, rot), LC.head_weight(self.ck, rot)
        every = LC.logits(norm, head, h)
        self.assertTrue(torch.allclose(LC.logits(norm, head, h, (3, 9)), every[3:9], atol=1e-6))
        self.assertTrue(torch.allclose(LC.logits(norm, head, h, [2, 7, 30]), every[[2, 7, 30]], atol=1e-6))


class Gptq(unittest.TestCase):
    def setUp(self):
        g = torch.Generator().manual_seed(0)
        self.W = torch.randn(32, 64, generator=g) * 0.05
        X = torch.randn(2000, 64, generator=g) @ (torch.eye(64) + 0.3 * torch.randn(64, 64, generator=g))
        self.H = (X.T @ X / X.shape[0]).double()

    def test_palettes_are_seeded_sorted_and_fp16(self):
        p = G.kmeans_palettes(self.W, group=16, entries=16)
        self.assertEqual(tuple(p.shape), (2, 16))
        self.assertTrue(torch.equal(p, G.kmeans_palettes(self.W, group=16, entries=16)))
        self.assertTrue((p[:, 1:] >= p[:, :-1]).all())
        self.assertTrue(torch.equal(p, p.half().float()))

    def test_gptq_stays_on_the_palettes_and_beats_round_to_nearest(self):
        pal = G.kmeans_palettes(self.W, group=16, entries=16)
        for act_order in (False, True):
            grid = G.Palette(self.W, pal, 16)
            codes, out = G.gptq(self.W, self.H, grid, act_order=act_order)
            Wq = G.dequantize(codes, out, 16)
            for r in range(32):
                self.assertTrue(set(Wq[r].tolist()) <= set(pal[r // 16].tolist()))
            rtn = G.output_error(G.Palette(self.W, pal, 16).round_to_nearest(self.W), self.W, self.H)
            self.assertLess(G.output_error(Wq, self.W, self.H), rtn)

    def test_hessian_reads_channels_first_inputs(self):
        x = torch.randn(1, 8, 1, 5)
        h = G.Hessian()
        h.add(x)
        X = x.view(8, 5).double()
        self.assertTrue(torch.allclose(h.mean(), X @ X.T / 5))


class Draws(unittest.TestCase):
    def fixture(self, lengths):
        cal = [{"id": f"d{i}"} for i in range(len(lengths))]
        ids = {f"d{i}": [1000 * (i + 1) + j for j in range(n)] for i, n in enumerate(lengths)}
        return L.Fixture(cal, [], {"calibration": ids, "evaluation": {}}, "", Path("."))

    def test_draws_are_seeded_full_and_start_at_documents(self):
        fx = self.fixture([30, 7, 12, 50, 9, 3, 40, 22, 18, 5])
        starts = {v[0] for v in fx.ids["calibration"].values()}
        a, used = L.draw(fx, 0, 64, 16, [-1])
        self.assertEqual((a, used), L.draw(fx, 0, 64, 16, [-1]))
        self.assertNotEqual(a, L.draw(fx, 1, 64, 16, [-1])[0])
        owner = {t: d for d, ids in fx.ids["calibration"].items() for t in ids}
        self.assertLessEqual({owner[t] for s in a for t in s if t != -1}, used)
        self.assertEqual([len(s) for s in a], [16] * 4)
        self.assertTrue(all(s[0] in starts for s in a))

    def test_the_stability_draw_reads_only_documents_the_build_draw_did_not(self):
        fx = self.fixture([30, 7, 12, 50, 9, 3, 40, 22, 18, 5, 33, 41, 26, 17, 60, 8])
        job = L.Job(model_id="m", src=".", work=".", recipe=L.Recipe(tokens=64, sequence=16, draw_seed=0,
                                                                      stability_seed=1))
        saved = L._separator
        L._separator = lambda job: [-1]
        try:
            _, used0 = L.draw(fx, 0, 64, 16, [-1])
            first = L.job_draw(job, fx, 0)
            second = L.job_draw(job, fx, 1)
        finally:
            L._separator = saved
        self.assertEqual(first, L.draw(fx, 0, 64, 16, [-1])[0])
        owner = {t: d for d, ids in fx.ids["calibration"].items() for t in ids}
        read = {owner[t] for s in second for t in s if t != -1}
        self.assertFalse(read & used0)

    def test_a_draw_too_large_for_the_fixture_fails(self):
        with self.assertRaises(ValueError):
            L.draw(self.fixture([10, 10]), 0, 64, 16, [-1])
        with self.assertRaises(ValueError):
            L.draw(self.fixture([100]), 0, 60, 16, [-1])


class Grading(unittest.TestCase):
    def rows(self, n=200, v=50, seed=0):
        return np.random.default_rng(seed).normal(size=(n, v)) * 3

    def test_letters_follow_p1_cosine_and_top1(self):
        ref = self.rows()
        self.assertEqual(L.grade({"t": ref}, {"t": ref})["grade"], "A")
        noisy = ref + np.random.default_rng(1).normal(size=ref.shape) * 0.05
        g = L.grade({"t": noisy}, {"t": ref})
        self.assertEqual(g["grade_cos_p1"], L.letter(g["cos_p1"], L.cosine_scale()))
        self.assertEqual(g["grade"], L.worse(g["grade_cos_p1"], g["grade_top1"]))

    def test_a_wide_flip_caps_the_grade_at_c(self):
        ref = self.rows(v=10)
        ref[0] = 0.0
        ref[0, 3], ref[0, 4] = 5.0, 5.0 - L.WIDE_MARGIN - 0.01        # fp32 prefers token 3, clearly
        got = ref.copy()
        got[0, 4] = 5.5                                                 # the build flips it
        g = L.grade({"t": got}, {"t": ref})
        self.assertEqual(g["wide_flips"], 1)
        self.assertEqual(g["grade"], "C")
        near = ref.copy()
        near[0, 4] = 4.999                                             # a near-tie flip isn't capped
        near_ref = near.copy()
        near[0, 4] = 5.001
        self.assertEqual(L.grade({"t": near}, {"t": near_ref})["wide_flips"], 0)

    def test_the_wide_margin_is_compared_unrounded(self):
        ref = self.rows(v=10)
        ref[0] = 0.0
        ref[0, 3], ref[0, 4] = 5.0, 5.0 - L.WIDE_MARGIN - 1e-6        # rounded to 4 places it would be narrow
        got = ref.copy()
        got[0, 4] = 5.5
        self.assertEqual(L.grade({"t": got}, {"t": ref})["wide_flips"], 1)

    def test_families_are_graded_on_their_own_texts(self):
        ref = {"a": self.rows(seed=0), "b": self.rows(seed=1)}
        got = {"a": ref["a"], "b": ref["b"] + np.random.default_rng(2).normal(size=ref["b"].shape) * 0.3}
        g = L.grade(got, ref, {"a": "repository", "b": "prose"})
        self.assertEqual(g["families"]["repository"]["grade"], "A")
        self.assertEqual(g["families"]["prose"]["positions"], 200)
        self.assertEqual(g["families"]["prose"]["grade"], L._grade({"b": got["b"]}, {"b": ref["b"]})["grade"])

    def test_non_finite_output_is_f(self):
        ref = self.rows()
        got = ref.copy()
        got[5, 2] = np.nan
        self.assertEqual(L.grade({"t": got}, {"t": ref})["grade"], "F")


class WorkDirectory(unittest.TestCase):
    """The converter's cache of intermediates, and its install."""

    def test_an_intermediate_is_reused_only_for_its_key(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "ranges-0.json"
            self.assertFalse(L.fresh(path, {"a": 1}))
            path.write_text("{}")
            self.assertFalse(L.fresh(path, {"a": 1}))         # made, but from unknown inputs
            L._stamp(path, {"a": 1})
            self.assertTrue(L.fresh(path, {"a": 1}))
            self.assertFalse(L.fresh(path, {"a": 2}))
            L._discard(path)
            self.assertFalse(path.exists() or L._key_path(path).exists())

    def test_keys_follow_the_inputs_they_depend_on(self):
        job = L.Job(model_id="m", src=".", work=".", recipe=L.Recipe())
        job.inputs = {"revision": "r", "config_sha256": "c", "tokenizer_sha256": "t", "calibration_sha256": "k",
                      "evaluation_sha256": "e", "manifest_sha256": "m",
                      "versions": {"torch": "1", "transformers": "2", "scikit_learn": "3", "coremltools": "4"}}
        a = L.keys(job, 0)
        job.inputs = {**job.inputs, "evaluation_sha256": "e2"}
        b = L.keys(job, 0)
        self.assertNotEqual(a["references"], b["references"])
        self.assertNotEqual(a["screen"], b["screen"])
        self.assertEqual(a["gptq"], b["gptq"])                 # GPTQ doesn't read the evaluation set
        job.inputs = {**job.inputs, "calibration_sha256": "k2"}
        self.assertNotEqual(L.keys(job, 0)["gptq"], a["gptq"])
        self.assertNotEqual(L.keys(job, 1)["ranges"]["disjoint_from"], a["ranges"]["disjoint_from"])
        self.assertNotEqual(a["stage"], L.keys(job, 1)["stage"])

    def test_install_is_atomic_and_keeps_unchanged_programs(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            stage, dest = d / "stage-0", d / "models" / "m"
            for i, content in enumerate((b"zero", b"one")):
                (stage / f"decoder.{i}.mlmodelc" / "weights").mkdir(parents=True)
                (stage / f"decoder.{i}.mlmodelc" / "weights" / "weight.bin").write_bytes(content)
            m = {"decoder": {"chunks": 2}}
            job = L.Job(model_id="m", src=".", work=str(d), recipe=L.Recipe())
            saved = L.install_host
            L.install_host = lambda job, m, where, fixture, summaries: (where / "chat.toml").write_text("new")
            try:
                L.install(job, m, stage, dest, None, [])
                inode = (dest / "decoder.0.mlmodelc" / "weights" / "weight.bin").stat().st_ino
                (stage / "decoder.1.mlmodelc" / "weights" / "weight.bin").write_bytes(b"changed")
                L.install(job, m, stage, dest, None, [])
                self.assertEqual((dest / "decoder.0.mlmodelc" / "weights" / "weight.bin").stat().st_ino, inode)
                self.assertEqual((dest / "decoder.1.mlmodelc" / "weights" / "weight.bin").read_bytes(), b"changed")

                def fail(job, m, where, fixture, summaries):
                    raise OSError("disk full")
                L.install_host = fail
                (stage / "decoder.0.mlmodelc" / "weights" / "weight.bin").write_bytes(b"newer")
                with self.assertRaises(OSError):
                    L.install(job, m, stage, dest, None, [])
                self.assertEqual((dest / "decoder.0.mlmodelc" / "weights" / "weight.bin").read_bytes(), b"zero")
                self.assertEqual((dest / "chat.toml").read_text(), "new")
            finally:
                L.install_host = saved

    def test_chat_ids_must_be_token_ids(self):
        class Template:
            def apply_chat_template(self, messages, **kwargs):
                return {"input_ids": [1, 2], "attention_mask": [1, 1]}      # a BatchEncoding-like mapping

        tok = L.Tokens.__new__(L.Tokens)
        tok.hf, tok.template_kwargs = Template(), {}
        with self.assertRaises(TypeError):
            tok.chat([{"role": "user", "content": "hi"}], True)


class Committed(unittest.TestCase):
    """The committed fixtures and manifest, without the checkpoint."""

    def test_fixture_hashes_match_the_token_ids(self):
        for f in L.FIXTURES.glob("*.tokens.json"):
            rec = json.loads(f.read_text())
            self.assertEqual(rec["calibration_sha256"], L.sha256(L.CALIBRATION), f.name)
            self.assertEqual(rec["evaluation_sha256"], L.sha256(L.EVALUATION), f.name)
            cal = {e["id"] for e in L.read_jsonl(L.CALIBRATION)}
            self.assertEqual(set(rec["calibration"]), cal)
            for ids in list(rec["calibration"].values()) + list(rec["evaluation"].values()):
                self.assertTrue(ids and all(isinstance(i, int) for i in ids))

    def test_every_document_records_its_source_and_licence(self):
        for e in L.read_jsonl(L.CALIBRATION) + L.read_jsonl(L.EVALUATION):
            self.assertIn(e["licence"], ("MIT", "CC-BY-SA-4.0"), e["id"])
            self.assertTrue(e["source"], e["id"])
            if e["licence"] == "CC-BY-SA-4.0":
                self.assertIn("oldid=", e["source"])
                self.assertIn("Wikipedia contributors", e["attribution"])

    def test_the_manifest_template_is_consistent(self):
        m = tomllib.loads(L.manifest_path("minicpm5-2b").read_text())
        d = m["decoder"]
        self.assertEqual(d["context_length"], d["cache_capacity"] - d["tokens_per_call"] + 1)
        self.assertEqual(d["chunks"], len(d["layers"]))
        self.assertLessEqual(d["cache_capacity"], L.ANE_SEQ_CAP)
        self.assertNotIn("build", m)

    def test_the_build_table_appends_valid_toml(self):
        text = L.manifest_path("minicpm5-2b").read_text()
        job = L.Job(model_id="minicpm5-2b", src=".", work=".", recipe=L.Recipe(fp16_projections=("7.down",)),
                    inputs={"versions": {"torch": "2.13.0", "coremltools": "9.0"}})
        fx = L.Fixture([], [], {}, "ab" * 32, Path("."))
        s = {"ane": 10, "gpu": 0, "cpu": 0, "unassigned": 5, "total": 15, "off": {}}
        placement = "\n[placement]\ncompute_units = \"cpu_and_ne\"\n\n[placement.chunks.0]\n" + "\n".join(
            __import__("sidekick_convert.plan", fromlist=["_counts"])._counts(s)) + "\n"
        m = tomllib.loads(text.rstrip("\n") + "\n" + L.build_toml(job, {"K": 2}, fx, placement))
        self.assertEqual(m["build"]["residual_scale"], 2)
        self.assertEqual(m["build"]["fp16_projections"], ["7.down"])
        self.assertEqual(m["build"]["calibration"]["tokens"], 40960)
        self.assertEqual(m["placement"]["chunks"]["0"]["ane"], 10)
        self.assertEqual(m["build"]["versions"]["coremltools"], "9.0")

    def test_the_evaluation_set_has_long_held_out_prose(self):
        rec = json.loads((L.FIXTURES / "minicpm5-2b.tokens.json").read_text())
        fx = L.Fixture(L.read_jsonl(L.CALIBRATION), L.read_jsonl(L.EVALUATION), rec, "", Path("."))
        texts = fx.teacher_forced
        prose = [t for t in texts if t.family == "prose"]
        self.assertEqual(len(prose), 2)
        context = tomllib.loads(L.manifest_path("minicpm5-2b").read_text())["decoder"]["context_length"]
        for t in prose:
            self.assertGreater(len(t.ids), 1900)
            self.assertLessEqual(len(t.ids), context)
            self.assertLess(len(t.ids) - 1 - t.rows[-1], 8)          # graded to the end
        calibrated = {L.wikipedia_title(e["source"]) for e in fx.calibration}
        for e in fx.evaluation:
            self.assertNotIn(L.wikipedia_title(e["source"]) or "-", calibrated)


if __name__ == "__main__":
    unittest.main()
