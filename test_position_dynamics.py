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
    InputSinusoidPosition,
    InterventionOptimizationMonitor,
    QK_PREPROJECTION_MODES,
    QKPreprojectionPosition,
    collect_intervention_parameter_groups,
    normalize_input_sinusoid_config,
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
        torch.testing.assert_close(positional.q_input, positional.k_input)
        torch.testing.assert_close(seen["q"], values + positional.q_input[None])
        torch.testing.assert_close(seen["k"], values + positional.k_input[None])
        torch.testing.assert_close(seen["v"], values)
        torch.testing.assert_close(
            attention.to_q(seen["q"]),
            attention.to_q(values) + attention.to_q(positional.q_input[None]),
            atol=1e-6,
            rtol=1e-6,
        )


class QKPreprojectionTest(unittest.TestCase):
    def test_only_tied_scalar_is_active_and_anchor_is_exact(self):
        self.assertEqual(
            QK_PREPROJECTION_MODES,
            {
                "tied_scalar",
                "low_rank_premap",
                "low_rank_qk_replace",
                "low_rank_qk_residual",
            },
        )
        module = QKPreprojectionPosition(
            _config(),
            model_dim=8,
            extent=16,
        )
        output = module(11, dtype=torch.float32)
        expected = module.basis(11)
        torch.testing.assert_close(output.q_input, expected, rtol=0, atol=0)
        torch.testing.assert_close(output.k_input, expected, rtol=0, atol=0)
        self.assertIsNone(output.q_projected)
        self.assertIsNone(output.k_projected)
        self.assertEqual(sum(p.numel() for p in module.parameters()), 1)

    def test_gate_receives_gradient_and_reset_restores_anchor(self):
        module = QKPreprojectionPosition(
            _config(gate_init=0.25),
            model_dim=8,
            extent=16,
        )
        loss = (
            module(9, dtype=torch.float32).q_input * torch.randn(9, 8)
        ).sum()
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
        reference = fixed(1024, dtype=torch.bfloat16).q_input
        fixed.bfloat16()
        self.assertEqual(fixed.basis.basis.dtype, torch.float32)
        self.assertEqual(fixed.fixed_gate.dtype, torch.float32)
        torch.testing.assert_close(
            fixed(1024, dtype=torch.bfloat16).q_input,
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
            target(9, dtype=torch.float32).q_input,
            source(9, dtype=torch.float32).q_input,
        )

    def test_low_rank_modes_have_exact_nested_anchors(self):
        basis_dim = 8
        rank = 2
        expected_counts = {
            "low_rank_premap": 1 + 2 * basis_dim * rank,
            "low_rank_qk_replace": 3 * basis_dim * rank,
            "low_rank_qk_residual": 1 + 3 * basis_dim * rank,
        }
        for mode, expected_count in expected_counts.items():
            with self.subTest(mode=mode):
                config = normalize_qk_preprojection_config(
                    {"enabled": True, "mode": mode, "rank": rank},
                    model_dim=basis_dim,
                    rope_theta=10_000.0,
                )
                module = QKPreprojectionPosition(
                    config,
                    model_dim=basis_dim,
                    extent=16,
                )
                module.reset_output_parameters()
                output = module(11, dtype=torch.float32)
                basis = module.basis(11)
                self.assertEqual(
                    sum(parameter.numel() for parameter in module.parameters()),
                    expected_count,
                )
                if mode == "low_rank_qk_replace":
                    self.assertIsNone(output.q_input)
                    self.assertIsNone(output.k_input)
                else:
                    torch.testing.assert_close(output.q_input, basis, rtol=0, atol=0)
                    torch.testing.assert_close(output.k_input, basis, rtol=0, atol=0)
                if mode == "low_rank_premap":
                    self.assertIsNone(output.q_projected)
                    self.assertIsNone(output.k_projected)
                else:
                    torch.testing.assert_close(
                        output.q_projected,
                        torch.zeros_like(basis),
                        rtol=0,
                        atol=0,
                    )
                    torch.testing.assert_close(
                        output.k_projected,
                        torch.zeros_like(basis),
                        rtol=0,
                        atol=0,
                    )

    def test_low_rank_readouts_receive_live_initial_gradients(self):
        for mode in (
            "low_rank_premap",
            "low_rank_qk_replace",
            "low_rank_qk_residual",
        ):
            with self.subTest(mode=mode):
                config = normalize_qk_preprojection_config(
                    {"enabled": True, "mode": mode, "rank": 2},
                    model_dim=8,
                    rope_theta=10_000.0,
                )
                module = QKPreprojectionPosition(config, model_dim=8, extent=16)
                module.reset_output_parameters()
                output = module(9, dtype=torch.float32)
                loss = sum(
                    (value * torch.randn_like(value)).sum()
                    for value in output.carrier_tensors()
                )
                loss.backward()
                output_modules = (
                    (module.shared_up,)
                    if mode == "low_rank_premap"
                    else (module.q_up, module.k_up)
                )
                self.assertTrue(
                    all(
                        readout.weight.grad is not None
                        and readout.weight.grad.abs().sum().item() > 0
                        for readout in output_modules
                    )
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
        with self.assertRaisesRegex(ValueError, "gate_sharing"):
            normalize_qk_preprojection_config(
                {"gate_sharing": "per_head"},
                model_dim=8,
                rope_theta=10_000.0,
            )
        with self.assertRaisesRegex(TypeError, "active_layers"):
            normalize_qk_preprojection_config(
                {"active_layers": [0, "1"]},
                model_dim=8,
                rope_theta=10_000.0,
            )
        with self.assertRaisesRegex(TypeError, "rank"):
            normalize_qk_preprojection_config(
                {"rank": 2.0}, model_dim=8, rope_theta=10_000.0
            )
        with self.assertRaisesRegex(ValueError, "rank"):
            normalize_qk_preprojection_config(
                {"enabled": True, "mode": "low_rank_premap", "rank": 9},
                model_dim=8,
                rope_theta=10_000.0,
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


class InputSinusoidTest(unittest.TestCase):
    @staticmethod
    def _config(**updates):
        raw = {"enabled": True, **updates}
        return normalize_input_sinusoid_config(
            raw,
            model_dim=8,
            rope_theta=10_000.0,
        )

    def test_exact_anchor_gradient_and_fp32_gate(self):
        module = InputSinusoidPosition(
            self._config(gate_init=1.0),
            model_dim=8,
            extent=16,
        )
        torch.testing.assert_close(
            module(11, dtype=torch.float32),
            module.basis(11),
            rtol=0,
            atol=0,
        )
        module(9, dtype=torch.float32).square().sum().backward()
        self.assertGreater(module.gate.grad.abs().item(), 0)
        module.bfloat16()
        self.assertEqual(module.gate.dtype, torch.float32)
        self.assertEqual(module.basis.basis.dtype, torch.float32)

    def test_integrated_input_is_added_once_after_input_projection(self):
        model = IntegratedPreprojectionTest._model(
            input_sinusoid_config={"enabled": True, "learnable_gate": False},
        ).eval()
        input_ids = torch.randint(0, 32, (2, 10))
        seen = {}

        def capture(_module, args):
            seen["block_input"] = args[0].detach().clone()

        handle = model.blocks[0].register_forward_pre_hook(capture)
        try:
            model(input_ids)
        finally:
            handle.remove()
        content = model.in_proj(model.token_embedding(input_ids))
        carrier = model.input_sinusoid(10, dtype=content.dtype)
        torch.testing.assert_close(seen["block_input"], content + carrier[None])
        counts = count_parameters(model)
        self.assertEqual(counts["input_sinusoid_params"], 0)

    def test_config_round_trip_and_optimizer_monitor(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.json"
            path.write_text(
                json.dumps(
                    {
                        "use_rope": True,
                        "input_sinusoid": {"enabled": True},
                    }
                )
            )
            config = load_config(_cli(str(path)))
            self.assertTrue(config.input_sinusoid["enabled"])
            restored_path = Path(directory) / "resolved.json"
            restored_path.write_text(json.dumps(vars(config)))
            restored = load_config(_cli(str(restored_path)))
            self.assertEqual(restored.input_sinusoid, config.input_sinusoid)

        model = IntegratedPreprojectionTest._model(
            input_sinusoid_config={"enabled": True},
        )
        optimizer_args = Namespace(
            optimizer="adamw",
            exclude_position_from_decay=True,
            position_lr_multiplier=0.5,
            weight_decay=0.1,
            learning_rate=3.0e-4,
            beta1=0.9,
            beta2=0.98,
        )
        with mock.patch("torch.cuda.is_available", return_value=False):
            optimizer = make_optimizer(optimizer_args, model)
        ids = torch.randint(0, 32, (2, 10))
        model(ids, torch.randint(0, 32, (2, 10))).backward()
        monitor = InterventionOptimizationMonitor(
            collect_intervention_parameter_groups(model), reference_length=16
        )
        sample = monitor.capture_before_clip(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        monitor.capture_after_clip(sample)
        optimizer.step()
        metrics = monitor.capture_after_step(sample, optimizer)
        prefix = "optimization/input_sinusoid"
        self.assertGreater(metrics[f"{prefix}/raw_gradient/l2"], 0)
        self.assertGreater(metrics[f"{prefix}/carrier_function_step/rms"], 0)


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

    def test_low_rank_modes_match_their_intended_initial_models(self):
        ids = torch.randint(0, 32, (2, 10))
        rope = self._model(use_rope=True).eval()
        tied = self._model(
            use_rope=True,
            qk_preprojection_config={"enabled": True, "mode": "tied_scalar"},
        ).eval()
        replacement = self._model(
            use_rope=True,
            qk_preprojection_config={
                "enabled": True,
                "mode": "low_rank_qk_replace",
                "rank": 4,
            },
        ).eval()
        premap = self._model(
            use_rope=True,
            qk_preprojection_config={
                "enabled": True,
                "mode": "low_rank_premap",
                "rank": 4,
            },
        ).eval()
        residual = self._model(
            use_rope=True,
            qk_preprojection_config={
                "enabled": True,
                "mode": "low_rank_qk_residual",
                "rank": 4,
            },
        ).eval()
        torch.testing.assert_close(replacement(ids), rope(ids), rtol=0, atol=0)
        torch.testing.assert_close(premap(ids), tied(ids), rtol=0, atol=0)
        torch.testing.assert_close(residual(ids), tied(ids), rtol=0, atol=0)

    def test_low_rank_modes_have_finite_end_to_end_gradients(self):
        ids = torch.randint(0, 32, (2, 10))
        targets = torch.randint(0, 32, (2, 10))
        for mode in (
            "low_rank_premap",
            "low_rank_qk_replace",
            "low_rank_qk_residual",
        ):
            with self.subTest(mode=mode):
                model = self._model(
                    use_rope=True,
                    qk_norm_mode="method_aware_rms",
                    qk_preprojection_config={
                        "enabled": True,
                        "mode": mode,
                        "rank": 4,
                    },
                )
                loss = model(ids, targets)
                loss.backward()
                adapter = model.blocks[0].attn.qk_preprojection
                readouts = (
                    (adapter.shared_up,)
                    if mode == "low_rank_premap"
                    else (adapter.q_up, adapter.k_up)
                )
                self.assertTrue(torch.isfinite(loss).item())
                self.assertTrue(
                    all(
                        readout.weight.grad is not None
                        and torch.isfinite(readout.weight.grad).all().item()
                        and readout.weight.grad.abs().sum().item() > 0
                        for readout in readouts
                    )
                )

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

    def test_nope_resolved_config_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nope.json"
            path.write_text(json.dumps({"use_rope": False}))
            config = load_config(_cli(str(path)))
            self.assertEqual(config.pos_variant, "none")
            resolved_path = Path(directory) / "resolved-nope.json"
            resolved_path.write_text(json.dumps(vars(config)))
            restored = load_config(_cli(str(resolved_path)))
            self.assertEqual(restored.pos_variant, "none")
            self.assertFalse(restored.use_rope)

    def test_global_gate_is_one_parameter_shared_across_layers(self):
        model = self._model(
            depth=3,
            qk_preprojection_config={
                "enabled": True,
                "gate_sharing": "global",
            },
        )
        gates = [block.attn.qk_preprojection.gate for block in model.blocks]
        self.assertTrue(all(gate is gates[0] for gate in gates[1:]))
        self.assertEqual(count_parameters(model)["qk_preprojection_params"], 1)
        ids = torch.randint(0, 32, (2, 10))
        model(ids, torch.randint(0, 32, (2, 10))).backward()
        self.assertIsNotNone(gates[0].grad)
        self.assertTrue(torch.isfinite(gates[0].grad).item())

    def test_active_layers_limit_repeated_carrier(self):
        model = self._model(
            depth=3,
            qk_preprojection_config={
                "enabled": True,
                "active_layers": [0],
            },
        )
        self.assertIsNotNone(model.blocks[0].attn.qk_preprojection)
        self.assertIsNone(model.blocks[1].attn.qk_preprojection)
        self.assertIsNone(model.blocks[2].attn.qk_preprojection)
        self.assertEqual(count_parameters(model)["qk_preprojection_params"], 1)
        with self.assertRaisesRegex(ValueError, "outside model depth"):
            self._model(
                depth=3,
                qk_preprojection_config={
                    "enabled": True,
                    "active_layers": [3],
                },
            )

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
