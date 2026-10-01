"""ModernBERT backbone rewrites and checks, on a tiny random model."""

import copy
import unittest

import torch

from sidekick_convert.backbones import modernbert
from sidekick_convert.techniques import attention, masks


def tiny(**kw):
    from transformers import ModernBertConfig, ModernBertModel
    cfg = dict(vocab_size=100, hidden_size=64, num_attention_heads=4, num_hidden_layers=3,
               intermediate_size=96, local_attention=8, global_attn_every_n_layers=3,
               max_position_embeddings=64, attn_implementation="eager", pad_token_id=0,
               bos_token_id=1, eos_token_id=2, cls_token_id=1, sep_token_id=2)
    cfg.update(kw)
    torch.manual_seed(0)
    return ModernBertModel(ModernBertConfig(**cfg)).eval()


def backbone(model):
    return modernbert.ModernBertBackbone(family="modernbert", model=model, config=model.config,
                                         special_ids=(1, 2))


def run(model, ids):
    t = torch.tensor([ids])
    with torch.no_grad():
        return model(input_ids=t, attention_mask=torch.ones_like(t)).last_hidden_state


IDS = [1, 5, 9, 17, 33, 2, 40, 41, 42, 43, 44, 2]


class ConfigGuard(unittest.TestCase):
    def test_matching_file_passes(self):
        m = tiny()
        modernbert.verify_config(m, {"global_rope_theta": 160000.0, "local_rope_theta": 10000.0})

    def test_rope_parameters_mismatch_is_caught(self):
        m = tiny()
        raw = {"rope_parameters": {"full_attention": {"rope_theta": 160000.0},
                                   "sliding_attention": {"rope_theta": 160000.0}}}
        with self.assertRaises(SystemExit):
            modernbert.verify_config(m, raw)

    def test_resolve_config_builds_what_the_file_says(self):
        raw = {"rope_parameters": {"full_attention": {"rope_theta": 160000.0},
                                   "sliding_attention": {"rope_theta": 160000.0}}}
        m = tiny()
        cfg = modernbert.resolve_config(copy.deepcopy(m.config), raw)
        from transformers import ModernBertModel
        modernbert.verify_config(ModernBertModel(cfg), raw)

    def test_layer_types_are_checked(self):
        with self.assertRaises(SystemExit):
            modernbert.verify_config(tiny(), {"layer_types": ["sliding_attention"] * 3})


class Rewrites(unittest.TestCase):
    def test_residual_rewrite_is_exact(self):
        m = tiny()
        want = run(m, IDS)
        for k in (2, 4):
            b = backbone(copy.deepcopy(m))
            modernbert.residual_rewrite(b, k)
            torch.testing.assert_close(run(b.model, IDS), want, rtol=1e-4, atol=1e-5)

    def test_twice_gelu_is_exact(self):
        m = tiny(hidden_activation="gelu")
        want = run(m, IDS)
        b = backbone(copy.deepcopy(m))
        modernbert.twice_gelu(b)
        torch.testing.assert_close(run(b.model, IDS), want, rtol=1e-5, atol=1e-6)
        self.assertIn("gelu", b.forbid_ops)

    def test_split_maxima(self):
        fixed, scaled = modernbert.split_maxima({(0, "Wqkv"): 10.0, (0, "attn.Wo"): 50.0,
                                                 (1, "Wi"): 20.0, (1, "mlp.Wo"): 70.0})
        self.assertEqual((fixed, scaled), (20.0, 70.0))


class HeadLayer(unittest.TestCase):
    def test_written_out_encoder_layer_matches_pytorch(self):
        torch.manual_seed(3)
        layer = torch.nn.TransformerEncoderLayer(32, 4, 64, dropout=0.0, batch_first=True, norm_first=True).eval()
        x = torch.randn(1, 10, 32)
        am = torch.tensor([[1] * 7 + [0] * 3])
        with torch.no_grad():
            want = layer(x, src_key_padding_mask=am == 0)
            got = attention.transformer_encoder_layer(layer, x, masks.key_padding(am), 10)
            got_mm = attention.transformer_encoder_layer(layer, x, masks.key_padding(am), 10, softmax="matmul")
        torch.testing.assert_close(got[:, :7], want[:, :7], rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(got_mm[:, :7], want[:, :7], rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
