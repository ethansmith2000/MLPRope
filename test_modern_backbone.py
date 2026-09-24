"""Correctness and nesting tests for the bundled modern backbone."""

from __future__ import annotations

import json
import tempfile
import unittest
from argparse import Namespace

import torch

from train_gpt import load_config, make_model
from transformer import GeGLU, SwiGLU, Transformer, count_parameters


def _cli(path: str) -> Namespace:
    return Namespace(
        override_json=path,
        pos_variant=None,
        attn_impl=None,
        max_train_steps=None,
        dry_run=False,
        print_model=False,
    )


def _load(payload: dict):
    with tempfile.NamedTemporaryFile("w", suffix=".json") as handle:
        json.dump(payload, handle)
        handle.flush()
        return load_config(_cli(handle.name))


class ModernBackboneTest(unittest.TestCase):
    def test_explicit_controlled_variant_is_bit_exact_default(self):
        common = {
            "dim": 32,
            "depth": 2,
            "heads": 4,
            "ff_mult": 2,
            "vocab_size": 64,
            "max_seq_len": 16,
            "paired_initialization_seed": 123,
        }
        default = Transformer(**common).eval()
        explicit = Transformer(**common, backbone_variant="controlled").eval()
        self.assertIsInstance(default.blocks[0].ff, GeGLU)
        self.assertEqual(default.state_dict().keys(), explicit.state_dict().keys())
        for name, value in default.state_dict().items():
            torch.testing.assert_close(value, explicit.state_dict()[name], rtol=0, atol=0)
        tokens = torch.randint(0, 64, (2, 12))
        torch.testing.assert_close(default(tokens), explicit(tokens), rtol=0, atol=0)

    def test_modern_bundle_has_declared_structure_and_tied_output(self):
        model = Transformer(
            dim=32,
            depth=2,
            heads=4,
            ff_mult=4,
            vocab_size=64,
            max_seq_len=16,
            qk_norm_mode="method_aware_rms",
            paired_initialization_seed=123,
            backbone_variant="modern",
        )
        self.assertIsInstance(model.in_proj, torch.nn.Identity)
        self.assertIsInstance(model.blocks[0].norm1, torch.nn.RMSNorm)
        self.assertIsInstance(model.blocks[0].norm2, torch.nn.RMSNorm)
        self.assertEqual(model.blocks[0].norm1.eps, 1e-6)
        self.assertIsInstance(model.blocks[0].ff, SwiGLU)
        self.assertEqual(model.blocks[0].ff.proj_out.in_features, 128)
        for block in model.blocks:
            self.assertIsNone(block.attn.to_q.bias)
            self.assertIsNone(block.attn.to_k.bias)
            self.assertIsNone(block.attn.to_v.bias)
            self.assertIsNone(block.attn.to_out.bias)
            self.assertIsNone(block.ff.proj_in.bias)
            self.assertIsNone(block.ff.proj_out.bias)
        output_parameters = list(model.out_proj.named_parameters())
        self.assertEqual([name for name, _ in output_parameters], ["norm.weight"])
        self.assertIs(output_parameters[0][1], model.out_proj.norm.weight)
        self.assertFalse(
            any("lm_head" in name for name, _ in model.named_parameters())
        )
        counts = count_parameters(model)
        self.assertEqual(counts["embeddings"], 64 * 32)
        self.assertEqual(counts["lm_head"], 32)

        tokens = torch.randint(0, 64, (2, 12))
        targets = torch.randint(0, 64, (2, 12))
        loss, logits = model(tokens, targets, return_logits=True)
        self.assertEqual(logits.shape, (2, 12, 64))
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(model.token_embedding.weight.grad)
        self.assertTrue(torch.isfinite(model.token_embedding.weight.grad).all())

    def test_modern_rope_and_scalar_share_exact_initialization(self):
        common = {
            "dim": 32,
            "depth": 2,
            "heads": 4,
            "ff_mult": 4,
            "vocab_size": 64,
            "max_seq_len": 16,
            "qk_norm_mode": "method_aware_rms",
            "paired_initialization_seed": 123,
            "backbone_variant": "modern",
        }
        rope = Transformer(**common)
        scalar = Transformer(
            **common,
            qk_preprojection_config={"enabled": True},
        )
        scalar_state = scalar.state_dict()
        shared = [
            name
            for name in rope.state_dict()
            if name in scalar_state and "qk_preprojection" not in name
        ]
        self.assertGreater(len(shared), 20)
        for name in shared:
            torch.testing.assert_close(
                rope.state_dict()[name], scalar_state[name], rtol=0, atol=0
            )

    def test_modern_low_rank_readouts_start_as_exact_scalar_function(self):
        common = {
            "dim": 32,
            "depth": 2,
            "heads": 4,
            "ff_mult": 4,
            "vocab_size": 64,
            "max_seq_len": 16,
            "qk_norm_mode": "method_aware_rms",
            "paired_initialization_seed": 123,
            "backbone_variant": "modern",
        }
        scalar = Transformer(
            **common,
            qk_preprojection_config={"enabled": True},
        )
        tokens = torch.randint(0, 64, (2, 12))
        scalar_logits = scalar(tokens)
        scalar_state = scalar.state_dict()
        for rank, multiplier in ((8, 6.367487169620489), (16, 2.449489742783178)):
            readout = Transformer(
                **common,
                qk_preprojection_config={
                    "enabled": True,
                    "mode": "low_rank_qk_residual",
                    "rank": rank,
                    "readout_lr_multiplier": multiplier,
                },
            )
            readout_state = readout.state_dict()
            for name, value in scalar_state.items():
                if name in readout_state:
                    torch.testing.assert_close(
                        value, readout_state[name], rtol=0, atol=0
                    )
            for block in readout.blocks:
                adapter = block.attn.qk_preprojection
                self.assertTrue(torch.count_nonzero(adapter.q_up.weight) == 0)
                self.assertTrue(torch.count_nonzero(adapter.k_up.weight) == 0)
            torch.testing.assert_close(
                scalar_logits, readout(tokens), rtol=0, atol=0
            )

    def test_config_resolves_modern_width_and_rejects_hybrids(self):
        config = _load(
            {
                "backbone_variant": "modern",
                "qk_norm_mode": "method_aware_rms",
            }
        )
        self.assertEqual(config.ff_hidden_dim, 2048)
        model = make_model(config, 64)
        self.assertEqual(model.backbone_variant, "modern")
        self.assertEqual(model.blocks[0].ff.proj_out.in_features, 2048)

        for payload in (
            {"backbone_variant": "unknown"},
            {"backbone_variant": "modern", "qk_projection_bias": True},
            {
                "backbone_variant": "modern",
                "ff_widened_hidden_dim": 4096,
                "ff_widened_layers": [0],
            },
        ):
            with self.subTest(payload=payload):
                with tempfile.NamedTemporaryFile("w", suffix=".json") as handle:
                    json.dump(payload, handle)
                    handle.flush()
                    with self.assertRaises(ValueError):
                        load_config(_cli(handle.name))


if __name__ == "__main__":
    unittest.main()
