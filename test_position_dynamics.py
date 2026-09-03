"""Safety and equivalence tests for the new dynamic position primitives."""

from __future__ import annotations

import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import torch

from position import (
    PositionGain,
    QKPreprojectionPosition,
    RotaryClockController,
    build_rope_frequencies,
    normalize_qk_preprojection_config,
    normalize_position_gain_config,
    normalize_rotary_clock_config,
)
from position.temporal import CausalControlMapper, CausalEMA
from position.clock import _exclusive_associative_sum
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


class CausalControlMapperTest(unittest.TestCase):
    def test_debiased_ema_matches_streaming_and_preserves_constants(self):
        torch.manual_seed(0)
        ema = CausalEMA(5, decay_init=0.9)
        values = torch.randn(3, 17, 5)
        expected = ema(values)
        state = ema.initial_state(3, device=values.device)
        steps = []
        for token in values.unbind(dim=1):
            output, state = ema.step(token, state)
            steps.append(output)
        torch.testing.assert_close(
            torch.stack(steps, dim=1),
            expected,
            atol=2e-6,
            rtol=2e-6,
        )

        constant = torch.randn(3, 1, 5).expand(-1, 23, -1).clone()
        torch.testing.assert_close(
            ema(constant),
            constant,
            atol=2e-6,
            rtol=2e-6,
        )

    def test_debiased_ema_is_prefix_causal_and_decay_learns(self):
        torch.manual_seed(8)
        ema = CausalEMA(4, decay_init=0.8)
        values = torch.randn(2, 13, 4, requires_grad=True)
        altered = values.detach().clone()
        altered[:, 7:] = torch.randn_like(altered[:, 7:])
        torch.testing.assert_close(ema(values)[:, :7], ema(altered)[:, :7])
        ema(values).square().sum().backward()
        self.assertIsNotNone(ema.decay_logit.grad)
        self.assertGreater(ema.decay_logit.grad.abs().sum().item(), 0)

    def test_causal_convolution_matches_incremental_execution(self):
        torch.manual_seed(1)
        mapper = CausalControlMapper(
            input_dim=8,
            output_dim=3,
            mapper="low_rank_silu",
            rank=5,
            temporal="causal_conv",
            kernel_size=3,
        )
        torch.nn.init.normal_(mapper.output.weight, std=0.2)
        torch.nn.init.normal_(mapper.output.bias, std=0.1)
        values = torch.randn(2, 7, 8)

        expected = mapper(values)
        state = mapper.initial_state(2, device=values.device, dtype=values.dtype)
        steps = []
        for token in values.unbind(dim=1):
            output, state = mapper.step(token, state)
            steps.append(output)
        actual = torch.stack(steps, dim=1)
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)

    def test_causal_convolution_has_no_future_leakage(self):
        torch.manual_seed(2)
        mapper = CausalControlMapper(
            input_dim=8,
            output_dim=2,
            mapper="low_rank_silu",
            rank=4,
            temporal="causal_conv",
            kernel_size=4,
        )
        torch.nn.init.normal_(mapper.output.weight, std=0.2)
        values = torch.randn(2, 9, 8)
        altered = values.clone()
        altered[:, 5:] = torch.randn_like(altered[:, 5:])
        torch.testing.assert_close(mapper(values)[:, :5], mapper(altered)[:, :5])

    def test_ema_mapper_matches_incremental_execution(self):
        torch.manual_seed(9)
        mapper = CausalControlMapper(
            input_dim=8,
            output_dim=3,
            mapper="low_rank_silu",
            rank=5,
            temporal="ema",
            kernel_size=1,
            ema_decay_init=0.85,
        )
        torch.nn.init.normal_(mapper.output.weight, std=0.2)
        values = torch.randn(2, 19, 8)
        expected = mapper(values)
        state = mapper.initial_state(2, device=values.device, dtype=values.dtype)
        steps = []
        for token in values.unbind(dim=1):
            output, state = mapper.step(token, state)
            steps.append(output)
        torch.testing.assert_close(
            torch.stack(steps, dim=1),
            expected,
            atol=3e-6,
            rtol=3e-6,
        )


class RotaryClockTest(unittest.TestCase):
    @staticmethod
    def _config(**updates):
        config = {
            "enabled": True,
            "head_coupling": "per_head",
            "mapper": "low_rank_silu",
            "rank": 6,
            "temporal": "causal_conv",
            "kernel_size": 3,
            "speed_bound": 0.2,
        }
        config.update(updates)
        return normalize_rotary_clock_config(config)

    @staticmethod
    def _clock(config):
        return RotaryClockController(
            model_dim=12,
            heads=3,
            pair_dim=2,
            inverse_frequency=build_rope_frequencies(4, 10_000.0),
            config=config,
        )

    def test_zero_output_is_exact_standard_rope_clock(self):
        clock = self._clock(self._config())
        values = torch.randn(2, 8, 12)
        speed = clock.speed(values)
        coordinates = clock.coordinates(values)
        expected = torch.arange(8, dtype=torch.float32)[None, :, None]

        torch.testing.assert_close(speed, torch.ones_like(speed), rtol=0, atol=0)
        torch.testing.assert_close(
            coordinates,
            expected.expand(2, 8, 3),
            rtol=0,
            atol=0,
        )
        self.assertEqual(torch.count_nonzero(clock.phase_delta(values)).item(), 0)

    def test_associative_scan_matches_prefix_sum_and_long_anchor(self):
        torch.manual_seed(13)
        for length in (1, 2, 3, 17, 1_024):
            speed = 0.8 + 0.4 * torch.rand(2, length, 3)
            expected = speed.cumsum(dim=1) - speed
            torch.testing.assert_close(
                _exclusive_associative_sum(speed),
                expected,
                atol=1e-4,
                rtol=1e-6,
            )
        ones = torch.ones(2, 1_024, 3)
        expected_anchor = torch.arange(1_024, dtype=torch.float32)[None, :, None]
        torch.testing.assert_close(
            _exclusive_associative_sum(ones),
            expected_anchor.expand_as(ones),
            atol=0,
            rtol=0,
        )

    def test_speed_is_bounded_monotone_and_spectrally_locked(self):
        torch.manual_seed(3)
        clock = self._clock(self._config(speed_bound=0.15))
        torch.nn.init.normal_(clock.controller.output.weight, std=0.5)
        values = torch.randn(2, 10, 12)
        speed = clock.speed(values)
        coordinates = clock.coordinates(values)
        phase = clock.phase_delta(values)

        self.assertGreaterEqual(speed.min().item(), 0.85)
        self.assertLessEqual(speed.max().item(), 1.15)
        self.assertTrue(torch.all(coordinates[:, 1:] > coordinates[:, :-1]))
        # Every frequency plane receives the same scalar displacement times its
        # fixed base omega; there are no independently learned frequencies.
        displacement = coordinates.permute(0, 2, 1) - torch.arange(10)[None, None]
        expected = displacement[..., None] * clock.inverse_frequency
        torch.testing.assert_close(phase, expected)

    def test_clock_is_prefix_causal_and_matches_incremental_execution(self):
        torch.manual_seed(4)
        clock = self._clock(self._config())
        torch.nn.init.normal_(clock.controller.output.weight, std=0.3)
        values = torch.randn(2, 9, 12)
        altered = values.clone()
        altered[:, 6:] = torch.randn_like(altered[:, 6:])

        expected = clock.phase_delta(values)
        torch.testing.assert_close(
            expected[:, :, :6],
            clock.phase_delta(altered)[:, :, :6],
        )

        state = clock.initial_state(2, device=values.device, dtype=values.dtype)
        steps = []
        for token in values.unbind(dim=1):
            phase, state = clock.step_phase_delta(token, state)
            steps.append(phase)
        actual = torch.cat(steps, dim=2)
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)

    def test_clock_anchor_learns_and_preserves_frequency_precision(self):
        clock = self._clock(self._config(temporal="pointwise", kernel_size=1))
        values = torch.randn(2, 7, 12, requires_grad=True)
        clock.phase_delta(values).sum().backward()
        self.assertGreater(clock.controller.output.weight.grad.abs().sum().item(), 0)

        expected_frequency = clock.inverse_frequency.clone()
        clock.half()
        self.assertEqual(clock.inverse_frequency.dtype, torch.float32)
        torch.testing.assert_close(clock.inverse_frequency, expected_frequency)

    def test_invalid_clock_configs_fail_early(self):
        with self.assertRaisesRegex(ValueError, "strictly inside"):
            normalize_rotary_clock_config({"speed_bound": 1.0})
        with self.assertRaisesRegex(ValueError, "kernel_size >= 2"):
            normalize_rotary_clock_config(
                {
                    "temporal": "causal_conv",
                    "mapper": "low_rank_silu",
                    "kernel_size": 1,
                }
            )
        with self.assertRaisesRegex(ValueError, "requires mapper"):
            normalize_rotary_clock_config(
                {"temporal": "causal_conv", "mapper": "linear"}
            )
        with self.assertRaisesRegex(ValueError, "requires mapper"):
            normalize_rotary_clock_config(
                {"temporal": "ema", "mapper": "linear"}
            )
        with self.assertRaisesRegex(ValueError, "ema_decay_init"):
            normalize_rotary_clock_config(
                {"temporal": "ema", "ema_decay_init": 1.0}
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


class PositionGainTest(unittest.TestCase):
    @staticmethod
    def _config(target="both", **updates):
        config = {
            "enabled": True,
            "target": target,
            "basis_dim": 16,
            "scalars": ["normalized_position", "log_position"],
            "mapper": "linear",
            "log_gain_bound": 0.7,
        }
        config.update(updates)
        return normalize_position_gain_config(
            config,
            heads=3,
            head_dim=8,
            rope_theta=10_000.0,
            normalization_extent=16,
        )

    def test_exact_unit_anchor_prefix_stability_and_bf16(self):
        module = PositionGain(
            self._config(),
            heads=3,
            head_dim=8,
            extent=16,
        )
        short = module(6, dtype=torch.float32)
        long = module(12, dtype=torch.float32)
        torch.testing.assert_close(short.q, torch.ones_like(short.q), rtol=0, atol=0)
        torch.testing.assert_close(short.k, torch.ones_like(short.k), rtol=0, atol=0)
        torch.testing.assert_close(short.q, long.q[:, :, :6], rtol=0, atol=0)
        torch.testing.assert_close(short.k, long.k[:, :, :6], rtol=0, atol=0)
        module.to(dtype=torch.bfloat16)
        bf16 = module(6, dtype=torch.bfloat16)
        torch.testing.assert_close(
            bf16.q,
            torch.ones_like(bf16.q),
            rtol=0,
            atol=0,
        )

    def test_active_readouts_learn_and_inactive_branch_has_no_parameters(self):
        for target in ("q", "k", "both"):
            module = PositionGain(
                self._config(target),
                heads=3,
                head_dim=8,
                extent=16,
            )
            output = module(7, dtype=torch.float32)
            (output.q.sum() + output.k.sum()).backward()
            if target in {"q", "both"}:
                self.assertIsNotNone(module.q_readout.weight.grad)
                self.assertGreater(module.q_readout.weight.grad.abs().sum().item(), 0)
            else:
                self.assertIsNone(module.q_readout)
                torch.testing.assert_close(output.q, torch.ones_like(output.q))
            if target in {"k", "both"}:
                self.assertIsNotNone(module.k_readout.weight.grad)
                self.assertGreater(module.k_readout.weight.grad.abs().sum().item(), 0)
            else:
                self.assertIsNone(module.k_readout)
                torch.testing.assert_close(output.k, torch.ones_like(output.k))

    def test_q_gain_scales_rows_and_k_gain_scales_key_columns(self):
        torch.manual_seed(9)
        q = torch.randn(2, 3, 6, 8)
        k = torch.randn(2, 3, 6, 8)
        baseline = torch.einsum("bhid,bhjd->bhij", q, k)
        for target in ("q", "k"):
            module = PositionGain(
                self._config(target),
                heads=3,
                head_dim=8,
                extent=16,
            )
            readout = module.q_readout if target == "q" else module.k_readout
            torch.nn.init.normal_(readout.weight, std=0.2)
            gains = module(6, dtype=q.dtype)
            actual = torch.einsum(
                "bhid,bhjd->bhij",
                q * gains.q,
                k * gains.k,
            )
            expected = (
                baseline * gains.q.squeeze(-1)[..., :, None]
                if target == "q"
                else baseline * gains.k.squeeze(-1)[..., None, :]
            )
            torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
            active = gains.q if target == "q" else gains.k
            self.assertGreaterEqual(active.min().item(), torch.exp(torch.tensor(-0.7)).item())
            self.assertLessEqual(active.max().item(), torch.exp(torch.tensor(0.7)).item())

    def test_invalid_config_fails_early(self):
        with self.assertRaisesRegex(ValueError, "target"):
            self._config("neither")
        with self.assertRaisesRegex(ValueError, "finite and positive"):
            self._config(log_gain_bound=0.0)

    def test_state_dict_round_trip_preserves_nontrivial_gains(self):
        torch.manual_seed(12)
        config = self._config("both")
        source = PositionGain(config, heads=3, head_dim=8, extent=16)
        torch.nn.init.normal_(source.q_readout.weight, std=0.2)
        torch.nn.init.normal_(source.k_readout.weight, std=0.2)
        expected = source(11, dtype=torch.float32)

        restored = PositionGain(config, heads=3, head_dim=8, extent=16)
        restored.load_state_dict(source.state_dict())
        actual = restored(11, dtype=torch.float32)
        torch.testing.assert_close(actual.q, expected.q)
        torch.testing.assert_close(actual.k, expected.k)


class IntegratedDynamicPositionTest(unittest.TestCase):
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

    def test_zero_clock_matches_fixed_rope_model(self):
        baseline = self._model().eval()
        candidate = self._model(
            rotary_clock_config={
                "enabled": True,
                "mapper": "low_rank_silu",
                "rank": 4,
                "temporal": "causal_conv",
                "kernel_size": 3,
                "speed_bound": 0.2,
            }
        ).eval()
        candidate.load_state_dict(baseline.state_dict(), strict=False)
        input_ids = torch.randint(0, 32, (2, 10))
        torch.testing.assert_close(candidate(input_ids), baseline(input_ids))

        counts = count_parameters(candidate)
        self.assertGreater(counts["rotary_clock_params"], 0)
        self.assertEqual(counts["qk_preprojection_params"], 0)

    def test_zero_position_gain_matches_fixed_rope_model(self):
        baseline = self._model().eval()
        candidate = self._model(
            position_gain_config={
                "enabled": True,
                "target": "both",
                "basis_dim": 16,
                "mapper": "linear",
                "log_gain_bound": 1.0,
            }
        ).eval()
        candidate.load_state_dict(baseline.state_dict(), strict=False)
        input_ids = torch.randint(0, 32, (2, 10))
        torch.testing.assert_close(
            candidate(input_ids),
            baseline(input_ids),
            rtol=0,
            atol=0,
        )
        counts = count_parameters(candidate)
        self.assertGreater(counts["position_gain_params"], 0)
        self.assertEqual(counts["qk_position_params"], 0)

    def test_clock_attention_has_no_future_content_leakage(self):
        attention = Attention(
            12,
            3,
            max_seq_len=12,
            qk_config={"enabled": False},
            logit_bias_config={"enabled": False},
            rotary_clock_config={
                "enabled": True,
                "rank": 5,
                "temporal": "causal_conv",
                "kernel_size": 3,
                "speed_bound": 0.2,
            },
        ).eval()
        torch.nn.init.normal_(attention.rotary_clock.controller.output.weight, std=0.2)
        values = torch.randn(2, 9, 12)
        altered = values.clone()
        altered[:, 5:] = torch.randn_like(altered[:, 5:])
        torch.testing.assert_close(
            attention(values)[:, :5],
            attention(altered)[:, :5],
            atol=1e-6,
            rtol=1e-6,
        )

    def test_first_stage_interventions_are_isolated(self):
        with self.assertRaisesRegex(ValueError, "isolated first-stage"):
            self._model(
                qk_preprojection_config={"enabled": True},
                rotary_clock_config={"enabled": True},
            )
        with self.assertRaisesRegex(ValueError, "isolated first-stage"):
            self._model(
                position_gain_config={"enabled": True},
                rotary_clock_config={"enabled": True},
            )
        with self.assertRaisesRegex(ValueError, "erase scalar position gains"):
            self._model(
                position_gain_config={"enabled": True},
                post_position_qk_norm=True,
            )

    def test_preprojection_combines_only_with_additive_qk_channel(self):
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

        rotary = dict(additive, apply="phase_residual")
        with self.assertRaisesRegex(ValueError, "only be combined with an additive"):
            self._model(
                qk_config=rotary,
                qk_preprojection_config={"enabled": True},
            )

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

            gain_path = Path(directory) / "gain.json"
            gain_path.write_text(
                json.dumps(
                    {
                        "position_gain": {
                            "enabled": True,
                            "target": "q",
                            "basis_dim": 16,
                            "log_gain_bound": 0.7,
                        }
                    }
                )
            )
            gain = load_config(_cli(str(gain_path)))
            self.assertEqual(gain.pos_variant, "custom")
            saved_gain_path = Path(directory) / "gain_training_config.json"
            saved_gain_path.write_text(json.dumps(vars(gain)))
            restored_gain = load_config(_cli(str(saved_gain_path)))
            self.assertEqual(restored_gain.position_gain, gain.position_gain)
            self.assertEqual(restored_gain.run_name, gain.run_name)


if __name__ == "__main__":
    unittest.main()
