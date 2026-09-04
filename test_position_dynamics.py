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
    SinusoidFrequencyBank,
    collect_intervention_parameter_groups,
    normalize_qk_preprojection_config,
    normalize_sinusoid_frequency_config,
)
from train_gpt import (
    carrier_frequency_gradient_clip_groups,
    load_config,
    make_optimizer,
)
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
    mode: str = "tied_scalar",
    *,
    model_dim: int = 8,
    gate_init: float = 1.0,
    learnable_gate: bool = True,
    smooth_rank: int = 2,
) -> dict:
    return normalize_qk_preprojection_config(
        {
            "enabled": True,
            "mode": mode,
            "gate_init": gate_init,
            "learnable_gate": learnable_gate,
            "smooth_rank": smooth_rank,
        },
        model_dim=model_dim,
        rope_theta=10_000.0,
    )


class QKPreprojectionFormulaTest(unittest.TestCase):
    def test_tied_scalar_preserves_original_formula_and_v_input(self):
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


class SinusoidFrequencyBankTest(unittest.TestCase):
    def _bank(self, mode: str, *, dimension: int = 8, reference_length: int = 16):
        config = normalize_sinusoid_frequency_config(
            {"mode": mode},
            default_reference_length=reference_length,
        )
        return SinusoidFrequencyBank(dimension, 10_000.0, config)

    def test_both_learned_parameterizations_start_at_exact_fixed_frequency(self):
        fixed = self._bank("fixed").frequencies()
        for mode in ("learned_log", "learned_horizon"):
            with self.subTest(mode=mode):
                bank = self._bank(mode)
                torch.testing.assert_close(bank.frequencies(), fixed, rtol=0, atol=0)
                self.assertEqual(bank.coordinate.dtype, torch.float32)

    def test_horizon_coordinate_has_position_normalized_phase_gradient(self):
        reference_length = 16
        bank = self._bank("learned_horizon", reference_length=reference_length)
        position = reference_length - 1
        angle = position * bank.frequencies()[2]
        gradient = torch.autograd.grad(angle, bank.coordinate)[0]
        expected = torch.zeros_like(gradient)
        expected[2] = position / reference_length
        torch.testing.assert_close(gradient, expected, rtol=0, atol=0)
        self.assertLessEqual(gradient.abs().max().item(), 1.0)

    def test_endpoint_phase_jacobian_exposes_parameterization_conditioning(self):
        log_bank = self._bank("learned_log", reference_length=16)
        horizon_bank = self._bank("learned_horizon", reference_length=16)
        torch.testing.assert_close(
            log_bank.endpoint_phase_coordinate_jacobian(),
            16 * log_bank.frequencies(),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            horizon_bank.endpoint_phase_coordinate_jacobian(),
            torch.ones(4),
            rtol=0,
            atol=0,
        )

    def test_log_frequency_stays_positive_and_both_modes_receive_gradients(self):
        for mode in ("learned_log", "learned_horizon"):
            with self.subTest(mode=mode):
                bank = self._bank(mode)
                if mode == "learned_log":
                    with torch.no_grad():
                        bank.coordinate.copy_(torch.tensor([-2.0, -0.5, 0.5, 2.0]))
                    self.assertTrue(bool((bank.frequencies() > 0).all()))
                bank.frequencies().square().sum().backward()
                self.assertIsNotNone(bank.coordinate.grad)
                self.assertTrue(torch.isfinite(bank.coordinate.grad).all().item())

    def test_frequency_parameters_survive_explicit_bf16_conversion(self):
        bank = self._bank("learned_log")
        with torch.no_grad():
            bank.coordinate.copy_(torch.tensor([0.1, -0.2, 0.3, -0.4]))
        expected = bank.coordinate.detach().clone()
        bank.bfloat16()
        self.assertEqual(bank.base_frequency.dtype, torch.float32)
        self.assertEqual(bank.coordinate.dtype, torch.float32)
        torch.testing.assert_close(bank.coordinate, expected, rtol=0, atol=0)


class QKPreprojectionTest(unittest.TestCase):
    def test_all_modes_share_the_exact_unit_anchor(self):
        modules = {
            mode: QKPreprojectionPosition(
                _config(mode),
                model_dim=8,
                extent=16,
            )
            for mode in sorted(QK_PREPROJECTION_MODES)
        }
        reference = modules["tied_scalar"](11, dtype=torch.float32)
        for mode, module in modules.items():
            with self.subTest(mode=mode):
                output = module(11, dtype=torch.float32)
                torch.testing.assert_close(output.q, reference.q, rtol=0, atol=0)
                torch.testing.assert_close(output.k, reference.k, rtol=0, atol=0)

    def test_shared_basis_override_accepts_a_longer_cached_prefix(self):
        module = QKPreprojectionPosition(
            _config("tied_scalar"),
            model_dim=8,
            extent=16,
        )
        cached = module.basis(12)
        expected = module(11, dtype=torch.float32)
        actual = module(11, dtype=torch.float32, basis_override=cached)
        torch.testing.assert_close(actual.q, expected.q, rtol=0, atol=0)
        torch.testing.assert_close(actual.k, expected.k, rtol=0, atol=0)

    def test_smooth_amplitude_is_tied_and_has_no_scale_gauge(self):
        module = QKPreprojectionPosition(
            _config("tied_smooth_amplitude", smooth_rank=2),
            model_dim=8,
            extent=16,
        )
        torch.testing.assert_close(
            module.smooth_amplitude_basis.T @ module.smooth_amplitude_basis,
            4 * torch.eye(2),
            atol=1e-6,
            rtol=1e-6,
        )
        torch.testing.assert_close(
            module.smooth_amplitude_basis.sum(dim=0),
            torch.zeros(2),
            atol=1e-6,
            rtol=0,
        )
        torch.testing.assert_close(
            module.smooth_amplitude_basis.square().mean(dim=0),
            torch.ones(2),
            atol=1e-6,
            rtol=1e-6,
        )
        with torch.no_grad():
            module.log_amplitude_coordinates.copy_(torch.tensor([0.4, -0.2]))
        q_delta, k_delta = module.log_amplitude_deltas()
        torch.testing.assert_close(q_delta, k_delta, rtol=0, atol=0)
        torch.testing.assert_close(
            q_delta.mean(),
            torch.tensor(0.0),
            atol=1e-7,
            rtol=0,
        )

        output = module(9, dtype=torch.float32)
        torch.testing.assert_close(output.q, output.k, rtol=0, atol=0)

    def test_smooth_amplitude_coordinates_receive_gradients(self):
        torch.manual_seed(4)
        module = QKPreprojectionPosition(
            _config("tied_smooth_amplitude", smooth_rank=2),
            model_dim=8,
            extent=16,
        )
        output = module(9, dtype=torch.float32)
        loss = (output.q * torch.randn_like(output.q)).sum()
        loss.backward()
        for parameter in (module.gate, module.log_amplitude_coordinates):
            self.assertIsNotNone(parameter.grad)
            self.assertGreater(parameter.grad.abs().sum().item(), 0)

    def test_mode_parameter_counts_are_nested_and_exact(self):
        expected = {
            "tied_scalar": 1,
            "tied_smooth_amplitude": 3,
        }
        for mode, count in expected.items():
            with self.subTest(mode=mode):
                module = QKPreprojectionPosition(
                    _config(mode),
                    model_dim=8,
                    extent=16,
                )
                self.assertEqual(sum(p.numel() for p in module.parameters()), count)
        tied = QKPreprojectionPosition(
            _config("tied_scalar", learnable_gate=False),
            model_dim=8,
            extent=16,
        )
        self.assertEqual(sum(p.numel() for p in tied.parameters()), 0)
        self.assertEqual(set(tied.state_dict()), {"fixed_gate"})
        tied_learnable = QKPreprojectionPosition(
            _config("tied_scalar", learnable_gate=True),
            model_dim=8,
            extent=16,
        )
        self.assertEqual(set(tied_learnable.state_dict()), {"gate"})

    def test_reset_restores_every_anchor_parameter(self):
        module = QKPreprojectionPosition(
            _config("tied_smooth_amplitude", gate_init=0.25),
            model_dim=8,
            extent=16,
        )
        with torch.no_grad():
            for parameter in module.parameters():
                parameter.fill_(4.0)
        module.reset_output_parameters()
        self.assertEqual(module.gate.item(), 0.25)
        self.assertEqual(module.log_amplitude_coordinates.count_nonzero(), 0)

    def test_smooth_fp32_carrier_survives_bf16_module_conversion(self):
        module = QKPreprojectionPosition(
            _config("tied_smooth_amplitude"),
            model_dim=8,
            extent=1024,
        )
        with torch.no_grad():
            module.log_amplitude_coordinates.copy_(torch.tensor([0.1, -0.2]))
        reference = module(1024, dtype=torch.float32).q.to(torch.bfloat16)
        parameter_ids = {name: id(parameter) for name, parameter in module.named_parameters()}
        module.bfloat16()
        self.assertEqual(module.basis.basis.dtype, torch.float32)
        self.assertEqual(module.smooth_amplitude_basis.dtype, torch.float32)
        self.assertEqual(
            {name: id(parameter) for name, parameter in module.named_parameters()},
            parameter_ids,
        )
        for parameter in module.parameters():
            self.assertEqual(parameter.dtype, torch.float32)
        actual = module(1024, dtype=torch.bfloat16).q
        self.assertEqual(actual.dtype, torch.bfloat16)
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)

    def test_state_dict_round_trip_for_every_mode(self):
        torch.manual_seed(3)
        for mode in sorted(QK_PREPROJECTION_MODES):
            with self.subTest(mode=mode):
                module = QKPreprojectionPosition(
                    _config(mode),
                    model_dim=8,
                    extent=16,
                )
                with torch.no_grad():
                    for parameter in module.parameters():
                        parameter.add_(torch.randn_like(parameter) * 0.1)
                clone = QKPreprojectionPosition(
                    _config(mode),
                    model_dim=8,
                    extent=16,
                )
                clone.load_state_dict(module.state_dict(), strict=True)
                expected = module(9, dtype=torch.float32)
                actual = clone(9, dtype=torch.float32)
                torch.testing.assert_close(actual.q, expected.q)
                torch.testing.assert_close(actual.k, expected.k)

    def test_config_rejects_invalid_width_mode_and_scalars(self):
        with self.assertRaisesRegex(ValueError, "basis_dim=model_dim"):
            normalize_qk_preprojection_config(
                {"basis_dim": 4},
                model_dim=8,
                rope_theta=10_000.0,
            )
        with self.assertRaisesRegex(ValueError, "mode"):
            _config("per_head")
        with self.assertRaisesRegex(TypeError, "learnable_gate"):
            normalize_qk_preprojection_config(
                {"learnable_gate": 1},
                model_dim=8,
                rope_theta=10_000.0,
            )
        with self.assertRaisesRegex(ValueError, "smooth_rank"):
            _config("tied_smooth_amplitude", smooth_rank=4)
        with self.assertRaisesRegex(ValueError, "even model_dim"):
            normalize_qk_preprojection_config(
                {},
                model_dim=7,
                rope_theta=10_000.0,
            )

    def test_removed_modes_reject_enabled_but_canonicalize_disabled(self):
        removed_modes = {
            "tied_smooth_polar",
            "split_scalar",
            "split_smooth_polar",
            "split_pair_amplitude",
            "split_pair_polar",
        }
        for mode in sorted(removed_modes):
            with self.subTest(mode=mode):
                with self.assertRaisesRegex(ValueError, "removed.*Phase 33/35"):
                    _config(mode)
                normalized = normalize_qk_preprojection_config(
                    {"enabled": False, "mode": mode},
                    model_dim=8,
                    rope_theta=10_000.0,
                )
                self.assertEqual(normalized["mode"], "tied_scalar")


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
            qk_preprojection_config={
                "enabled": True,
                "mode": "tied_smooth_amplitude",
                "smooth_rank": 2,
            },
        ).eval()
        attention = combined.blocks[0].attn
        self.assertIsNotNone(attention.qk_preprojection)
        self.assertIsNotNone(attention.qk_position)
        self.assertEqual(attention.qk_position.application, "additive")
        self.assertTrue(attention.multiplicative_rope)
        output = combined(torch.randint(0, 32, (2, 10)))
        self.assertEqual(output.shape, (2, 10, 32))
        counts = count_parameters(combined)
        self.assertEqual(counts["qk_preprojection_params"], 3)
        self.assertGreater(counts["qk_position_params"], 0)

    def test_diagnostics_report_adapter_health(self):
        model = self._model(
            qk_preprojection_config={
                "enabled": True,
                "mode": "tied_smooth_amplitude",
                "smooth_rank": 2,
            }
        )
        metrics, _ = model.position_diagnostics(sequence_length=8)
        prefix = "position/layer_00/qk_preprojection"
        self.assertEqual(metrics[f"{prefix}/gate_q"], 1.0)
        self.assertEqual(metrics[f"{prefix}/gate_k"], 1.0)
        self.assertEqual(metrics[f"{prefix}/input_qk_diff_rms"], 0.0)
        self.assertEqual(metrics[f"{prefix}/log_amplitude_q_rms"], 0.0)
        self.assertNotIn(f"{prefix}/phase_q_rms", metrics)

    def test_modes_round_trip_and_receive_distinct_run_names(self):
        names = set()
        with tempfile.TemporaryDirectory() as directory:
            for mode in sorted(QK_PREPROJECTION_MODES):
                path = Path(directory) / f"{mode}.json"
                path.write_text(
                    json.dumps(
                        {
                            "use_rope": True,
                            "qk_preprojection": {
                                "enabled": True,
                                "mode": mode,
                            },
                        }
                    )
                )
                config = load_config(_cli(str(path)))
                self.assertEqual(config.qk_preprojection["mode"], mode)
                names.add(config.run_name)
                saved_path = Path(directory) / f"{mode}-resolved.json"
                saved_path.write_text(json.dumps(vars(config)))
                restored = load_config(_cli(str(saved_path)))
                self.assertEqual(restored.qk_preprojection, config.qk_preprojection)
                self.assertEqual(restored.run_name, config.run_name)
        self.assertEqual(len(names), len(QK_PREPROJECTION_MODES))

    def test_all_modes_have_finite_forward_backward(self):
        torch.manual_seed(9)
        ids = torch.randint(0, 32, (2, 10))
        targets = torch.randint(0, 32, (2, 10))
        for mode in sorted(QK_PREPROJECTION_MODES):
            with self.subTest(mode=mode):
                model = self._model(
                    qk_preprojection_config={"enabled": True, "mode": mode}
                )
                loss = model(ids, targets)
                loss.backward()
                self.assertTrue(torch.isfinite(loss).item())
                adapter = model.blocks[0].attn.qk_preprojection
                for parameter in adapter.parameters():
                    self.assertIsNotNone(parameter.grad)
                    self.assertTrue(torch.isfinite(parameter.grad).all().item())

    def test_standard_rope_has_no_trainable_intervention(self):
        model = self._model(depth=3, use_rope=True)
        self.assertFalse(hasattr(model, "rope_frequency"))
        self.assertFalse(any("rope_frequency" in name for name, _ in model.named_parameters()))
        self.assertNotIn("rope_frequency_params", count_parameters(model))

    def test_shared_carrier_frequency_is_exact_anchor_and_global(self):
        fixed_config = {
            "enabled": True,
            "mode": "tied_scalar",
        }
        fixed = self._model(
            depth=3,
            qk_preprojection_config=fixed_config,
        ).eval()
        for frequency_mode in ("learned_log", "learned_horizon"):
            with self.subTest(frequency_mode=frequency_mode):
                learned_config = {
                    **fixed_config,
                    "frequency": {"mode": frequency_mode, "reference_length": 16},
                }
                learned = self._model(
                    depth=3,
                    qk_preprojection_config=learned_config,
                ).eval()
                ids = torch.randint(0, 32, (2, 11))
                torch.testing.assert_close(learned(ids), fixed(ids), rtol=0, atol=0)
                self.assertEqual(
                    learned.qk_preprojection_frequency.coordinate.numel(),
                    8,
                )
                counts = count_parameters(learned)
                self.assertEqual(counts["qk_preprojection_params"], 3)
                self.assertEqual(counts["qk_preprojection_frequency_params"], 8)

    def test_shared_frequency_coordinates_receive_finite_model_gradients(self):
        ids = torch.randint(0, 32, (2, 11))
        targets = torch.randint(0, 32, (2, 11))
        model = self._model(
            qk_preprojection_config={
                "enabled": True,
                "mode": "tied_scalar",
                "frequency": {
                    "mode": "learned_horizon",
                    "reference_length": 16,
                },
            }
        )
        model(ids, targets).backward()
        bank = model.qk_preprojection_frequency
        self.assertIsNotNone(bank.coordinate.grad)
        self.assertTrue(torch.isfinite(bank.coordinate.grad).all().item())
        self.assertGreater(bank.coordinate.grad.abs().max().item(), 0)

    def test_frequency_diagnostics_preserve_full_shared_spectrum(self):
        model = self._model(
            qk_preprojection_config={
                "enabled": True,
                "mode": "tied_scalar",
                "frequency": {"mode": "learned_horizon", "reference_length": 16},
            },
        )
        metrics, profiles = model.position_diagnostics(sequence_length=8)
        self.assertEqual(
            metrics["position/shared_qkpre_frequency/endpoint_phase_delta_rms"],
            0.0,
        )
        self.assertEqual(
            metrics[
                "position/shared_qkpre_frequency/"
                "endpoint_phase_coordinate_jacobian_abs_max"
            ],
            1.0,
        )
        self.assertEqual(profiles["shared_qkpre_frequency/frequency"].numel(), 8)

    def test_frequency_configs_round_trip_with_explicit_reference_horizon(self):
        payload = {
            "training_length": 64,
            "model_position_extent": 64,
            "evaluation_lengths": [64],
            "qk_preprojection": {
                "enabled": True,
                "mode": "tied_scalar",
                "frequency": {"mode": "learned_horizon"},
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "frequency.json"
            path.write_text(json.dumps(payload))
            config = load_config(_cli(str(path)))
            self.assertFalse(hasattr(config, "rope_frequency"))
            self.assertEqual(
                config.qk_preprojection["frequency"]["reference_length"],
                64,
            )
            resolved = Path(directory) / "resolved.json"
            resolved.write_text(json.dumps(vars(config)))
            restored = load_config(_cli(str(resolved)))
            self.assertEqual(restored.qk_preprojection, config.qk_preprojection)

    def test_frequency_banks_have_no_decay_and_independent_clip_groups(self):
        model = self._model(
            qk_preprojection_config={
                "enabled": True,
                "mode": "tied_scalar",
                "frequency": {
                    "mode": "learned_horizon",
                    "reference_length": 16,
                    "max_grad_norm": 0.5,
                },
            },
        )
        optimizer_args = Namespace(
            optimizer="adamw",
            exclude_position_from_decay=False,
            weight_decay=0.1,
            learning_rate=3.0e-4,
            beta1=0.9,
            beta2=0.98,
        )
        with mock.patch("torch.cuda.is_available", return_value=False):
            optimizer = make_optimizer(optimizer_args, model)
        decay_by_id = {
            id(parameter): group["weight_decay"]
            for group in optimizer.param_groups
            for parameter in group["params"]
        }
        self.assertEqual(
            decay_by_id[id(model.qk_preprojection_frequency.coordinate)],
            0.0,
        )
        qkpre_gate = model.blocks[0].attn.qk_preprojection.gate
        self.assertEqual(decay_by_id[id(qkpre_gate)], 0.1)

        clip_groups = carrier_frequency_gradient_clip_groups(model)
        self.assertEqual(len(clip_groups), 1)
        self.assertEqual({max_norm for _, max_norm in clip_groups}, {0.5})
        self.assertEqual(
            {id(parameters[0]) for parameters, _ in clip_groups},
            {id(model.qk_preprojection_frequency.coordinate)},
        )

    def test_optimizer_monitor_separates_gradient_update_and_function_step(self):
        model = self._model(
            qk_preprojection_config={
                "enabled": True,
                "mode": "tied_scalar",
                "frequency": {
                    "mode": "learned_log",
                    "reference_length": 16,
                },
            },
        )
        optimizer_args = Namespace(
            optimizer="adamw",
            exclude_position_from_decay=False,
            weight_decay=0.1,
            learning_rate=3.0e-4,
            beta1=0.9,
            beta2=0.98,
        )
        with mock.patch("torch.cuda.is_available", return_value=False):
            optimizer = make_optimizer(optimizer_args, model)
        ids = torch.randint(0, 32, (2, 11))
        targets = torch.randint(0, 32, (2, 11))
        model(ids, targets).backward()
        monitor = InterventionOptimizationMonitor(
            collect_intervention_parameter_groups(model),
            reference_length=16,
        )
        sample = monitor.capture_before_clip(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        monitor.capture_after_clip(sample)
        optimizer.step()
        metrics = monitor.capture_after_step(sample, optimizer)
        prefix = "optimization/pre_qk_sinusoid_frequency"
        self.assertGreater(metrics[f"{prefix}/raw_gradient/l2"], 0)
        self.assertGreater(metrics[f"{prefix}/parameter_update/l2"], 0)
        self.assertGreater(metrics[f"{prefix}/carrier_function_step/rms"], 0)
        self.assertGreater(
            metrics[f"{prefix}/endpoint_phase_coordinate_jacobian/abs_max"],
            1.0,
        )
        self.assertIsNotNone(metrics[f"{prefix}/descent_update_gradient_cosine"])
        adapter_prefix = "optimization/pre_qk_sinusoid_adapter"
        self.assertGreater(
            metrics[f"{adapter_prefix}/carrier_function_step/rms"],
            0,
        )
        self.assertGreater(
            metrics[
                f"{adapter_prefix}/"
                "carrier_function_to_parameter_update_rms_ratio"
            ],
            0,
        )

    def test_removed_mechanisms_fail_with_migration_message(self):
        removed = {
            "position_gain": {"enabled": True},
            "rotary_clock": {"enabled": True},
            "qk_preprojection": {
                "enabled": True,
                "mode": "split_pair_polar",
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            for key, value in removed.items():
                with self.subTest(key=key):
                    path = Path(directory) / f"{key}.json"
                    path.write_text(json.dumps({key: value}))
                    with self.assertRaisesRegex(ValueError, "removed|only fixed"):
                        load_config(_cli(str(path)))

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "content-frequency.json"
            path.write_text(json.dumps({"rope_frequency": {"mode": "learned_log"}}))
            with self.assertRaisesRegex(ValueError, "RoPE frequency"):
                load_config(_cli(str(path)))


if __name__ == "__main__":
    unittest.main()
