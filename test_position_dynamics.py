"""Safety and equivalence tests for attention-local pre-Q/K position."""

from __future__ import annotations

import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import torch

from position import (
    QK_PREPROJECTION_MODES,
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


def _config(
    mode: str = "tied_scalar",
    *,
    model_dim: int = 8,
    gate_init: float = 1.0,
    learnable_gate: bool = True,
) -> dict:
    return normalize_qk_preprojection_config(
        {
            "enabled": True,
            "mode": mode,
            "gate_init": gate_init,
            "learnable_gate": learnable_gate,
        },
        model_dim=model_dim,
        rope_theta=10_000.0,
    )


class QKPreprojectionTest(unittest.TestCase):
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

    def test_split_scalar_can_separate_q_and_k(self):
        module = QKPreprojectionPosition(
            _config("split_scalar"),
            model_dim=8,
            extent=16,
        )
        with torch.no_grad():
            module.q_gate.fill_(1.25)
            module.k_gate.fill_(0.75)
        output = module(9, dtype=torch.float32)
        self.assertFalse(torch.allclose(output.q, output.k))
        loss = output.q.square().sum() + 2.0 * output.k.square().sum()
        loss.backward()
        self.assertGreater(module.q_gate.grad.abs().item(), 0)
        self.assertGreater(module.k_gate.grad.abs().item(), 0)
        self.assertNotEqual(module.q_gate.grad.item(), module.k_gate.grad.item())

    def test_pair_polar_rotation_matches_direct_trigonometry(self):
        module = QKPreprojectionPosition(
            _config("split_pair_polar"),
            model_dim=8,
            extent=16,
        )
        q_phase = torch.tensor([0.3, -0.2, 0.7, -0.5])
        k_phase = torch.tensor([-0.1, 0.4, -0.6, 0.2])
        with torch.no_grad():
            module.q_gate.fill_(1.2)
            module.k_gate.fill_(0.8)
            module.q_phase.copy_(q_phase)
            module.k_phase.copy_(k_phase)
        output = module(7, dtype=torch.float32)
        base = module.basis(7).reshape(7, 4, 2)

        def expected(gain, phase):
            real, imag = base.unbind(dim=-1)
            return torch.stack(
                (
                    gain * (real * phase.cos() - imag * phase.sin()),
                    gain * (real * phase.sin() + imag * phase.cos()),
                ),
                dim=-1,
            ).flatten(-2)

        torch.testing.assert_close(output.q, expected(1.2, q_phase))
        torch.testing.assert_close(output.k, expected(0.8, k_phase))

    def test_pairwise_amplitude_factorization_has_no_scale_gauge(self):
        module = QKPreprojectionPosition(
            _config("split_pair_amplitude"),
            model_dim=8,
            extent=16,
        )
        coordinates = torch.tensor([0.4, -0.7, 0.2])
        with torch.no_grad():
            module.q_gate.fill_(2.0)
            module.q_log_amplitude_coordinates.copy_(coordinates)
        q_delta, _ = module.log_amplitude_deltas()
        torch.testing.assert_close(q_delta.sum(), torch.tensor(0.0), atol=1e-7, rtol=0)
        gram = module.zero_sum_basis.T @ module.zero_sum_basis
        torch.testing.assert_close(gram, torch.eye(3), atol=1e-6, rtol=1e-6)
        effective_amplitude = module.q_gate.float() * q_delta.exp()
        geometric_mean = effective_amplitude.log().mean().exp()
        torch.testing.assert_close(
            geometric_mean,
            module.q_gate.float(),
            atol=1e-6,
            rtol=1e-6,
        )

    def test_pair_parameters_receive_distinct_qk_gradients(self):
        torch.manual_seed(0)
        module = QKPreprojectionPosition(
            _config("split_pair_polar"),
            model_dim=8,
            extent=16,
        )
        output = module(9, dtype=torch.float32)
        q_probe = torch.randn_like(output.q)
        k_probe = torch.randn_like(output.k)
        loss = (output.q * q_probe).sum() + (output.k * k_probe).sum()
        loss.backward()
        for parameter in (
            module.q_gate,
            module.k_gate,
            module.q_log_amplitude_coordinates,
            module.k_log_amplitude_coordinates,
            module.q_phase,
            module.k_phase,
        ):
            self.assertIsNotNone(parameter.grad)
            self.assertGreater(parameter.grad.abs().sum().item(), 0)
        self.assertFalse(
            torch.allclose(module.q_phase.grad, module.k_phase.grad)
        )

    def test_mode_parameter_counts_are_nested_and_exact(self):
        expected = {
            "tied_scalar": 1,
            "split_scalar": 2,
            "split_pair_amplitude": 8,
            "split_pair_polar": 16,
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
            _config("split_pair_polar", gate_init=0.25),
            model_dim=8,
            extent=16,
        )
        with torch.no_grad():
            for parameter in module.parameters():
                parameter.fill_(4.0)
        module.reset_output_parameters()
        self.assertEqual(module.q_gate.item(), 0.25)
        self.assertEqual(module.k_gate.item(), 0.25)
        self.assertEqual(module.q_log_amplitude_coordinates.count_nonzero(), 0)
        self.assertEqual(module.k_log_amplitude_coordinates.count_nonzero(), 0)
        self.assertEqual(module.q_phase.count_nonzero(), 0)
        self.assertEqual(module.k_phase.count_nonzero(), 0)

    def test_pairwise_fp32_carrier_survives_bf16_module_conversion(self):
        module = QKPreprojectionPosition(
            _config("split_pair_polar"),
            model_dim=8,
            extent=1024,
        )
        with torch.no_grad():
            module.q_phase.copy_(torch.tensor([0.1, -0.2, 0.3, -0.4]))
        reference = module(1024, dtype=torch.float32).q.to(torch.bfloat16)
        parameter_ids = {name: id(parameter) for name, parameter in module.named_parameters()}
        module.bfloat16()
        self.assertEqual(module.basis.basis.dtype, torch.float32)
        self.assertEqual(module.zero_sum_basis.dtype, torch.float32)
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
        with self.assertRaisesRegex(ValueError, "even model_dim"):
            normalize_qk_preprojection_config(
                {},
                model_dim=7,
                rope_theta=10_000.0,
            )


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
                "mode": "split_pair_polar",
            },
        ).eval()
        attention = combined.blocks[0].attn
        self.assertIsNotNone(attention.qk_preprojection)
        self.assertIsNotNone(attention.qk_position)
        self.assertEqual(attention.qk_position.application, "additive")
        self.assertFalse(attention.multiplicative_rope)
        output = combined(torch.randint(0, 32, (2, 10)))
        self.assertEqual(output.shape, (2, 10, 32))
        counts = count_parameters(combined)
        self.assertEqual(counts["qk_preprojection_params"], 32)
        self.assertGreater(counts["qk_position_params"], 0)

    def test_diagnostics_report_adapter_health(self):
        model = self._model(
            qk_preprojection_config={
                "enabled": True,
                "mode": "split_pair_polar",
            }
        )
        metrics, _ = model.position_diagnostics(sequence_length=8)
        prefix = "position/layer_00/qk_preprojection"
        self.assertEqual(metrics[f"{prefix}/gate_q"], 1.0)
        self.assertEqual(metrics[f"{prefix}/gate_k"], 1.0)
        self.assertEqual(metrics[f"{prefix}/input_qk_diff_rms"], 0.0)
        self.assertEqual(metrics[f"{prefix}/log_amplitude_q_rms"], 0.0)
        self.assertEqual(metrics[f"{prefix}/phase_q_rms"], 0.0)

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
