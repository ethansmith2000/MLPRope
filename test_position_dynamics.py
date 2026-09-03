"""Safety and equivalence tests for attention-local pre-Q/K position."""

from __future__ import annotations

import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import torch

from position import (
    QKPreprojectionPosition,
    normalize_qk_preprojection_config,
)
from train_gpt import load_config
from transformer import Attention, Transformer, count_parameters


def _cli(path: str) -> Namespace:
    return Namespace(
        override_json=path,
        pos_variant=None,
        attn_impl=None,
        max_train_steps=None,
        dry_run=False,
        print_model=False,
    )


class QKPreprojectionTest(unittest.TestCase):
    def test_preprojection_is_tied_and_does_not_modify_v_input(self):
        config = normalize_qk_preprojection_config(
            {
                "enabled": True,
                "gate_init": 0.3,
                "learnable_gate": False,
            },
            model_dim=8,
            rope_theta=10_000.0,
        )
        attention = Attention(
            8,
            2,
            use_rope=False,
            max_seq_len=12,
            qk_norm=False,
            qk_config={"enabled": False},
            logit_bias_config={"enabled": False},
            qk_preprojection_config=config,
        ).eval()
        values = torch.randn(2, 6, 8)
        seen = {}

        def capture(name):
            def hook(_module, args):
                seen[name] = args[0].detach().clone()

            return hook

        handles = [
            attention.to_q.register_forward_pre_hook(capture("q")),
            attention.to_k.register_forward_pre_hook(capture("k")),
            attention.to_v.register_forward_pre_hook(capture("v")),
        ]
        try:
            attention(values)
        finally:
            for handle in handles:
                handle.remove()

        positional = attention.qk_preprojection(6, dtype=values.dtype)[None]
        torch.testing.assert_close(seen["q"], values + positional)
        torch.testing.assert_close(seen["k"], values + positional)
        torch.testing.assert_close(seen["v"], values)
        torch.testing.assert_close(
            attention.to_q(seen["q"]),
            attention.to_q(values) + attention.to_q(positional),
            atol=1e-6,
            rtol=1e-6,
        )

    def test_preprojection_requires_full_model_width(self):
        with self.assertRaisesRegex(ValueError, "basis_dim=model_dim"):
            normalize_qk_preprojection_config(
                {"basis_dim": 4},
                model_dim=8,
                rope_theta=10_000.0,
            )

    def test_gate_reset_restores_configured_anchor(self):
        config = normalize_qk_preprojection_config(
            {"enabled": True, "gate_init": 0.25},
            model_dim=8,
            rope_theta=10_000.0,
        )
        module = QKPreprojectionPosition(config, model_dim=8, extent=16)
        with torch.no_grad():
            module.gate.fill_(4.0)
        module.reset_output_parameters()
        self.assertEqual(module.gate.item(), 0.25)


class IntegratedPreprojectionTest(unittest.TestCase):
    @staticmethod
    def _model(**updates):
        config = {
            "dim": 16,
            "depth": 1,
            "heads": 2,
            "ff_mult": 2,
            "vocab_size": 32,
            "max_seq_len": 16,
            "attn_impl": "sdpa",
            "qk_config": {"enabled": False},
            "logit_bias_config": {"enabled": False},
            "paired_initialization_seed": 17,
        }
        config.update(updates)
        return Transformer(**config)

    def test_preprojection_combines_with_additive_qk_channel(self):
        additive = {
            "enabled": True,
            "feature_map": "mlp",
            "sharing": "per_head",
            "apply": "add",
            "rank": 4,
            "mlp_hidden": 12,
        }
        combined = self._model(
            qk_config=additive,
            qk_preprojection_config={"enabled": True},
        ).eval()
        attention = combined.blocks[0].attn
        self.assertIsNotNone(attention.qk_preprojection)
        self.assertIsNotNone(attention.qk_position)
        self.assertEqual(attention.qk_position.application, "additive")
        self.assertFalse(attention.multiplicative_rope)
        output = combined(torch.randint(0, 32, (2, 10)))
        self.assertEqual(output.shape, (2, 10, 32))
        counts = count_parameters(combined)
        self.assertGreater(counts["qk_preprojection_params"], 0)
        self.assertGreater(counts["qk_position_params"], 0)

    def test_derived_custom_config_round_trips(self):
        with tempfile.TemporaryDirectory() as directory:
            first_path = Path(directory) / "first.json"
            first_path.write_text(
                json.dumps(
                    {
                        "use_rope": False,
                        "qk_preprojection": {"enabled": True},
                    }
                )
            )
            first = load_config(_cli(str(first_path)))
            self.assertEqual(first.pos_variant, "custom")

            saved_path = Path(directory) / "training_config.json"
            saved_path.write_text(json.dumps(vars(first)))
            restored = load_config(_cli(str(saved_path)))
            self.assertEqual(restored.pos_variant, "custom")
            self.assertEqual(restored.qk_preprojection, first.qk_preprojection)
            self.assertEqual(restored.run_name, first.run_name)

    def test_removed_mechanisms_fail_with_migration_message(self):
        removed = {
            "position_gain": {"enabled": True},
            "rotary_clock": {"enabled": True},
            "rope_frequency": {"mode": "content"},
        }
        with tempfile.TemporaryDirectory() as directory:
            for key, value in removed.items():
                with self.subTest(key=key):
                    path = Path(directory) / f"{key}.json"
                    path.write_text(json.dumps({key: value}))
                    with self.assertRaisesRegex(ValueError, "removed|only fixed"):
                        load_config(_cli(str(path)))


if __name__ == "__main__":
    unittest.main()
