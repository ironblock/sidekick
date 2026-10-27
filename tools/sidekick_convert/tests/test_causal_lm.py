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
        a = L.draw(fx, 0, 64, 16, [-1])
        self.assertEqual(a, L.draw(fx, 0, 64, 16, [-1]))
        self.assertNotEqual(a, L.draw(fx, 1, 64, 16, [-1]))
        self.assertEqual([len(s) for s in a], [16] * 4)
        self.assertTrue(all(s[0] in starts for s in a))

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

    def test_non_finite_output_is_f(self):
        ref = self.rows()
        got = ref.copy()
        got[5, 2] = np.nan
        self.assertEqual(L.grade({"t": got}, {"t": ref})["grade"], "F")


class Committed(unittest.TestCase):
    """The committed fixtures and manifest, without the checkpoint."""

    def test_fixture_hashes_match_the_token_ids(self):
        for f in L.FIXTURES.glob("*.tokens.json"):
            rec = json.loads(f.read_text())
            self.assertEqual(rec["calibration_sha256"], L.sha256(L.CALIBRATION), f.name)
            self.assertEqual(rec["evaluation_sha256"], L.sha256(L.EVALUATION), f.name)
            cal = {e["id"] for e in L.read_jsonl(L.CALIBRATION)}
            self.assertEqual(set(rec["calibration"]), cal)

    def test_every_calibration_document_records_its_source_and_licence(self):
        for e in L.read_jsonl(L.CALIBRATION):
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
        job = L.Job(model_id="minicpm5-2b", src=".", work=".", recipe=L.Recipe(fp16_projections=("7.down",)))
        fx = L.Fixture([], [], {}, "ab" * 32, Path("."))
        s = {"ane": 10, "gpu": 0, "cpu": 0, "unassigned": 5, "total": 15, "off": {}}
        placement = "\n[placement]\ncompute_units = \"cpu_and_ne\"\n\n[placement.chunks.0]\n" + "\n".join(
            __import__("sidekick_convert.plan", fromlist=["_counts"])._counts(s)) + "\n"
        m = tomllib.loads(text.rstrip("\n") + "\n" + L.build_toml(job, {"K": 2}, fx, placement))
        self.assertEqual(m["build"]["residual_scale"], 2)
        self.assertEqual(m["build"]["fp16_projections"], ["7.down"])
        self.assertEqual(m["build"]["calibration"]["tokens"], 40960)
        self.assertEqual(m["placement"]["chunks"]["0"]["ane"], 10)


if __name__ == "__main__":
    unittest.main()
