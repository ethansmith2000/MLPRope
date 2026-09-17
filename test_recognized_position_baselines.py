"""Correctness tests for the static positional baselines used in Phase 57."""

from __future__ import annotations

import json
import tempfile
import unittest
from argparse import Namespace
from unittest import mock

import torch

import transformer as transformer_module
from train_gpt import load_config, make_model
from transformer import Attention, Transformer, count_parameters, standard_alibi_slopes


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


class AlibiBaselineTest(unittest.TestCase):
    def test_power_of_two_slopes_match_the_standard_eight_head_schedule(self):
        expected = torch.tensor(
            [0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125, 0.00390625]
        )
        torch.testing.assert_close(standard_alibi_slopes(8), expected, rtol=0, atol=0)

    def test_causal_bias_penalizes_older_keys_by_head_slope(self):
        attention = Attention(
            16,
            2,
            use_rope=False,
            use_alibi=True,
            attn_impl="flex",
            max_seq_len=8,
        )
        captured = {}

        def fake_flex(q, k, v, **kwargs):
            captured.update(kwargs)
            return v

        values = torch.randn(1, 2, 6, 8)
        with (
            mock.patch.object(attention, "_block_mask", return_value="causal"),
            mock.patch.object(transformer_module, "_flex_attention_call", fake_flex),
        ):
            returned = attention._flex_attention(values, values, values)
        self.assertIs(returned, values)
        score_mod = captured["score_mod"]
        score = torch.tensor(2.0)
        torch.testing.assert_close(score_mod(score, 0, 0, 5, 5), score)
        torch.testing.assert_close(
            score_mod(score, 0, 1, 5, 2),
            score - 3 * standard_alibi_slopes(2)[1],
        )


class PartialRopeBaselineTest(unittest.TestCase):
    def test_only_the_leading_selected_width_is_rotated(self):
        attention = Attention(
            16,
            2,
            use_rope=True,
            rope_fraction=0.5,
            max_seq_len=8,
        )
        q = torch.randn(1, 2, 6, 8)
        k = torch.randn(1, 2, 6, 8)
        rotated_q, rotated_k = attention._apply_rope(q, k)
        self.assertEqual(attention.rotary_dim, 4)
        torch.testing.assert_close(rotated_q[..., 4:], q[..., 4:], rtol=0, atol=0)
        torch.testing.assert_close(rotated_k[..., 4:], k[..., 4:], rtol=0, atol=0)
        self.assertFalse(torch.equal(rotated_q[..., 1:, :4], q[..., 1:, :4]))
        self.assertFalse(torch.equal(rotated_k[..., 1:, :4], k[..., 1:, :4]))

    def test_explicit_full_fraction_preserves_the_default_model_exactly(self):
        common = {
            "dim": 16,
            "depth": 2,
            "heads": 2,
            "ff_mult": 2,
            "vocab_size": 64,
            "max_seq_len": 16,
            "paired_initialization_seed": 123,
        }
        default = Transformer(**common).eval()
        explicit = Transformer(**common, rope_fraction=1.0).eval()
        self.assertEqual(default.state_dict().keys(), explicit.state_dict().keys())
        for name, value in default.state_dict().items():
            torch.testing.assert_close(value, explicit.state_dict()[name], rtol=0, atol=0)
        tokens = torch.randint(0, 64, (2, 12))
        torch.testing.assert_close(default(tokens), explicit(tokens), rtol=0, atol=0)


class LearnedAbsoluteBaselineTest(unittest.TestCase):
    def test_parameter_count_forward_and_gradient(self):
        model = Transformer(
            dim=16,
            depth=2,
            heads=2,
            ff_mult=2,
            vocab_size=64,
            max_seq_len=16,
            use_rope=False,
            use_learned_absolute_position=True,
            paired_initialization_seed=123,
        )
        counts = count_parameters(model)
        self.assertEqual(counts["absolute_position_params"], 16 * 16)
        self.assertEqual(counts["position_params"], 16 * 16)
        tokens = torch.randint(0, 64, (2, 12))
        targets = torch.randint(0, 64, (2, 12))
        loss = model(tokens, targets)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        gradient = model.absolute_position_embedding.grad
        self.assertIsNotNone(gradient)
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(gradient[:12].abs().sum().item(), 0.0)

    def test_optional_table_does_not_perturb_shared_initialization(self):
        common = {
            "dim": 16,
            "depth": 2,
            "heads": 2,
            "ff_mult": 2,
            "vocab_size": 64,
            "max_seq_len": 16,
            "use_rope": False,
            "paired_initialization_seed": 123,
        }
        reference = Transformer(**common)
        candidate = Transformer(**common, use_learned_absolute_position=True)
        candidate_state = candidate.state_dict()
        for name, value in reference.state_dict().items():
            torch.testing.assert_close(value, candidate_state[name], rtol=0, atol=0)


class RecognizedBaselineConfigTest(unittest.TestCase):
    def test_each_new_baseline_round_trips_into_the_model(self):
        partial = _load({"rope_fraction": 0.25})
        self.assertEqual(partial.rope_fraction, 0.25)
        self.assertEqual(make_model(partial, 64).blocks[0].attn.rotary_dim, 24)

        alibi = _load({"use_rope": False, "use_alibi": True, "attn_impl": "flex"})
        self.assertTrue(make_model(alibi, 64).blocks[0].attn.use_alibi)

        learned = _load(
            {"use_rope": False, "use_learned_absolute_position": True}
        )
        self.assertIsNotNone(make_model(learned, 64).absolute_position_embedding)

    def test_invalid_combinations_are_rejected(self):
        invalid_payloads = (
            {"use_rope": False, "rope_fraction": 0.25},
            {"use_rope": True, "use_alibi": True, "attn_impl": "flex"},
            {"use_rope": False, "use_alibi": True, "attn_impl": "sdpa"},
            {"use_rope": True, "use_learned_absolute_position": True},
            {
                "use_rope": False,
                "use_learned_absolute_position": True,
                "input_sinusoid": {"enabled": True},
            },
        )
        for payload in invalid_payloads:
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                _load(payload)


if __name__ == "__main__":
    unittest.main()
