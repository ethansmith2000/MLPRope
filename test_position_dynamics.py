"""Safety and equivalence tests for attention-local pre-Q/K position."""

from __future__ import annotations

import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest import mock

import torch

from position import (
    InterventionOptimizationMonitor,
    QK_PREPROJECTION_MODES,
    QKPreprojectionPosition,
    collect_intervention_parameter_groups,
    normalize_qk_preprojection_config,
)
from train_gpt import load_config, make_optimizer
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


def _config(
    *,
    model_dim: int = 8,
    gate_init: float = 1.0,
    learnable_gate: bool = True,
) -> dict:
    return normalize_qk_preprojection_config(
        {
            "enabled": True,
            "mode": "tied_scalar",
            "gate_init": gate_init,
            "learnable_gate": learnable_gate,
        },
        model_dim=model_dim,
        rope_theta=10_000.0,
    )


class QKPreprojectionFormulaTest(unittest.TestCase):
    def test_tied_scalar_preserves_formula_and_v_input(self):
        config = _config(gate_init=0.3, learnable_gate=False)
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

        positional = attention.qk_preprojection(6, dtype=values.dtype)
        torch.testing.assert_close(positional.q, positional.k)
        torch.testing.assert_close(seen["q"], values + positional.q[None])
        torch.testing.assert_close(seen["k"], values + positional.k[None])
        torch.testing.assert_close(seen["v"], values)
        torch.testing.assert_close(
            attention.to_q(seen["q"]),
            attention.to_q(values) + attention.to_q(positional.q[None]),
            atol=1e-6,
            rtol=1e-6,
        )


class QKPreprojectionTest(unittest.TestCase):
    def test_only_tied_scalar_is_active_and_anchor_is_exact(self):
        self.assertEqual(QK_PREPROJECTION_MODES, {"tied_scalar"})
        module = QKPreprojectionPosition(
            _config(),
            model_dim=8,
            extent=16,
        )
        output = module(11, dtype=torch.float32)
        expected = module.basis(11)
        torch.testing.assert_close(output.q, expected, rtol=0, atol=0)
        torch.testing.assert_close(output.k, expected, rtol=0, atol=0)
        self.assertEqual(sum(p.numel() for p in module.parameters()), 1)

    def test_gate_receives_gradient_and_reset_restores_anchor(self):
        module = QKPreprojectionPosition(
            _config(gate_init=0.25),
            model_dim=8,
            extent=16,
        )
        loss = (module(9, dtype=torch.float32).q * torch.randn(9, 8)).sum()
        loss.backward()
        self.assertGreater(module.gate.grad.abs().item(), 0)
        with torch.no_grad():
            module.gate.fill_(4.0)
        module.reset_output_parameters()
        self.assertEqual(module.gate.item(), 0.25)

    def test_fixed_gate_state_and_fp32_cast_behavior(self):
        fixed = QKPreprojectionPosition(
            _config(gate_init=0.3, learnable_gate=False),
            model_dim=8,
            extent=1024,
        )
        self.assertEqual(sum(p.numel() for p in fixed.parameters()), 0)
        self.assertEqual(set(fixed.state_dict()), {"fixed_gate"})
        reference = fixed(1024, dtype=torch.bfloat16).q
        fixed.bfloat16()
        self.assertEqual(fixed.basis.basis.dtype, torch.float32)
        self.assertEqual(fixed.fixed_gate.dtype, torch.float32)
        torch.testing.assert_close(
            fixed(1024, dtype=torch.bfloat16).q,
            reference,
            rtol=0,
            atol=0,
        )

    def test_state_dict_round_trip(self):
        source = QKPreprojectionPosition(_config(), model_dim=8, extent=16)
        with torch.no_grad():
            source.gate.fill_(0.7)
        target = QKPreprojectionPosition(_config(), model_dim=8, extent=16)
        target.load_state_dict(source.state_dict(), strict=True)
        torch.testing.assert_close(
            target(9, dtype=torch.float32).q,
            source(9, dtype=torch.float32).q,
        )

    def test_config_rejects_invalid_active_values(self):
        with self.assertRaisesRegex(ValueError, "basis_dim=model_dim"):
            normalize_qk_preprojection_config(
                {"basis_dim": 4}, model_dim=8, rope_theta=10_000.0
            )
        with self.assertRaisesRegex(TypeError, "learnable_gate"):
            normalize_qk_preprojection_config(
                {"learnable_gate": 1}, model_dim=8, rope_theta=10_000.0
            )
        with self.assertRaisesRegex(ValueError, "even model_dim"):
            normalize_qk_preprojection_config(
                {}, model_dim=7, rope_theta=10_000.0
            )

    def test_historical_modes_fail_enabled_and_canonicalize_disabled(self):
        removed_modes = {
            "tied_smooth_amplitude",
            "tied_smooth_direct_amplitude",
            "tied_smooth_polar",
            "split_scalar",
            "split_smooth_polar",
            "split_pair_amplitude",
            "split_pair_polar",
        }
        for mode in sorted(removed_modes):
            with self.subTest(mode=mode):
                with self.assertRaisesRegex(ValueError, "removed.*Phase"):
                    normalize_qk_preprojection_config(
                        {"enabled": True, "mode": mode, "smooth_rank": 4},
                        model_dim=8,
                        rope_theta=10_000.0,
                    )
                normalized = normalize_qk_preprojection_config(
                    {"enabled": False, "mode": mode, "smooth_rank": 4},
                    model_dim=8,
                    rope_theta=10_000.0,
                )
                self.assertEqual(normalized["mode"], "tied_scalar")
                self.assertNotIn("smooth_rank", normalized)

    def test_historical_frequency_fails_enabled_and_is_dropped_disabled(self):
        with self.assertRaisesRegex(ValueError, "frequency.*removed"):
            normalize_qk_preprojection_config(
                {
                    "enabled": True,
                    "frequency": {"mode": "learned_horizon"},
                },
                model_dim=8,
                rope_theta=10_000.0,
            )
        normalized = normalize_qk_preprojection_config(
            {
                "enabled": False,
                "frequency": {"mode": "learned_horizon"},
            },
            model_dim=8,
            rope_theta=10_000.0,
        )
        self.assertNotIn("frequency", normalized)


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
        self.assertTrue(attention.multiplicative_rope)
        output = combined(torch.randint(0, 32, (2, 10)))
        self.assertEqual(output.shape, (2, 10, 32))
        counts = count_parameters(combined)
        self.assertEqual(counts["qk_preprojection_params"], 1)
        self.assertGreater(counts["qk_position_params"], 0)

    def test_diagnostics_report_gate_and_qknorm_mixture(self):
        model = self._model(
            qk_norm_mode="method_aware_rms",
            qk_preprojection_config={"enabled": True},
        )
        metrics, profiles = model.position_diagnostics(
            sequence_length=8,
            input_ids=torch.randint(0, 32, (2, 8)),
        )
        prefix = "position/layer_00/qk_preprojection"
        self.assertEqual(metrics[f"{prefix}/gate"], 1.0)
        self.assertEqual(metrics[f"{prefix}/input_qk_diff_rms"], 0.0)
        self.assertGreater(
            metrics[f"{prefix}/input_mixture_position_energy_fraction"], 0.0
        )
        self.assertGreater(
            metrics[f"{prefix}/projected_q_mixture_position_to_content_rms_ratio"],
            0.0,
        )
        self.assertLessEqual(
            metrics[f"{prefix}/normalized_q_cosine_to_content"], 1.0
        )
        self.assertFalse(
            any("frequency" in key or "amplitude_factor" in key for key in profiles)
        )

    def test_active_mode_round_trip_and_finite_backward(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "carrier.json"
            path.write_text(
                json.dumps(
                    {
                        "use_rope": True,
                        "qk_preprojection": {"enabled": True},
                    }
                )
            )
            config = load_config(_cli(str(path)))
            self.assertEqual(config.qk_preprojection["mode"], "tied_scalar")
            self.assertNotIn("frequency", config.qk_preprojection)
            self.assertFalse(hasattr(config, "frequency_lr_multiplier"))
            restored_path = Path(directory) / "resolved.json"
            restored_path.write_text(json.dumps(vars(config)))
            restored = load_config(_cli(str(restored_path)))
            self.assertEqual(restored.qk_preprojection, config.qk_preprojection)
            self.assertEqual(restored.run_name, config.run_name)

        model = self._model(qk_preprojection_config={"enabled": True})
        ids = torch.randint(0, 32, (2, 10))
        loss = model(ids, torch.randint(0, 32, (2, 10)))
        loss.backward()
        self.assertTrue(torch.isfinite(loss).item())
        gate = model.blocks[0].attn.qk_preprojection.gate
        self.assertIsNotNone(gate.grad)
        self.assertTrue(torch.isfinite(gate.grad).item())

    def test_standard_rope_has_no_trainable_frequency_intervention(self):
        model = self._model(depth=3, use_rope=True)
        self.assertFalse(any("frequency" in name for name, _ in model.named_parameters()))
        self.assertNotIn("qk_preprojection_frequency_params", count_parameters(model))

    def test_position_lr_multiplier_has_its_own_optimizer_group(self):
        model = self._model(qk_preprojection_config={"enabled": True})
        optimizer_args = Namespace(
            optimizer="adamw",
            exclude_position_from_decay=True,
            position_lr_multiplier=0.25,
            weight_decay=0.1,
            learning_rate=3.0e-4,
            beta1=0.9,
            beta2=0.98,
        )
        with mock.patch("torch.cuda.is_available", return_value=False):
            optimizer = make_optimizer(optimizer_args, model)
        gate = model.blocks[0].attn.qk_preprojection.gate
        group = next(
            group
            for group in optimizer.param_groups
            if any(parameter is gate for parameter in group["params"])
        )
        self.assertEqual(group["group_name"], "position")
        self.assertAlmostEqual(group["lr"], 7.5e-5)
        self.assertEqual(group["weight_decay"], 0.0)

    def test_optimizer_monitor_tracks_static_carrier_function_step(self):
        model = self._model(qk_preprojection_config={"enabled": True})
        optimizer_args = Namespace(
            optimizer="adamw",
            exclude_position_from_decay=False,
            position_lr_multiplier=1.0,
            weight_decay=0.1,
            learning_rate=3.0e-4,
            beta1=0.9,
            beta2=0.98,
        )
        with mock.patch("torch.cuda.is_available", return_value=False):
            optimizer = make_optimizer(optimizer_args, model)
        ids = torch.randint(0, 32, (2, 11))
        model(ids, torch.randint(0, 32, (2, 11))).backward()
        monitor = InterventionOptimizationMonitor(
            collect_intervention_parameter_groups(model), reference_length=16
        )
        sample = monitor.capture_before_clip(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        monitor.capture_after_clip(sample)
        optimizer.step()
        metrics = monitor.capture_after_step(sample, optimizer)
        prefix = "optimization/pre_qk_sinusoid_adapter"
        self.assertGreater(metrics[f"{prefix}/raw_gradient/l2"], 0)
        self.assertGreater(metrics[f"{prefix}/parameter_update/l2"], 0)
        self.assertGreater(metrics[f"{prefix}/carrier_function_step/rms"], 0)
        self.assertGreater(
            metrics[f"{prefix}/carrier_function_to_parameter_update_rms_ratio"],
            0,
        )

    def test_removed_mechanisms_fail_with_migration_message(self):
        removed = {
            "position_gain": {"enabled": True},
            "rotary_clock": {"enabled": True},
            "qk_preprojection": {
                "enabled": True,
                "mode": "tied_smooth_amplitude",
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            for key, value in removed.items():
                with self.subTest(key=key):
                    path = Path(directory) / f"{key}.json"
                    path.write_text(json.dumps({key: value}))
                    with self.assertRaisesRegex(ValueError, "removed|fixed"):
                        load_config(_cli(str(path)))

            path = Path(directory) / "rope-frequency.json"
            path.write_text(json.dumps({"rope_frequency": {"mode": "learned_log"}}))
            with self.assertRaisesRegex(ValueError, "RoPE frequency"):
                load_config(_cli(str(path)))


if __name__ == "__main__":
    unittest.main()
