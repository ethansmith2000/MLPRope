"""Characterization and v2 foundation tests for position channels."""

from __future__ import annotations

import json
import math
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import torch

from position import (
    FrozenFourierBasis,
    apply_rotary,
    build_qk_position_channel,
    build_rope_cache,
    compose_phase,
    interleaved_fourier_basis,
    normalize_position_config_v2,
    rotate_half,
    upgrade_legacy_position_config,
)
from position.config import detect_channel_schema
from train_gpt import load_config
from transformer import Attention, Transformer, count_parameters


FEATURE_MAPS = (
    "identity",
    "add_rope",
    "linear",
    "low_rank",
    "bottleneck_mlp",
    "mlp",
)
SHARING_MODES = ("shared_head", "per_head", "full_dim")
ADDITIVE_SINUSOID_MAPS = ("identity", "add_rope", "low_rank", "bottleneck_mlp", "mlp")
SWEEP_CONFIG_DIR = Path(__file__).resolve().parent / "sweep_configs"
HEAD_COUPLINGS = (
    "shared_head",
    "per_head_independent",
    "per_head_joint",
)
QK_COUPLINGS = (
    "shared",
    "shared_trunk_separate_readouts",
    "separate",
)


def _cli(path):
    return Namespace(
        override_json=path,
        pos_variant=None,
        attn_impl=None,
        max_train_steps=None,
        dry_run=False,
        print_model=False,
    )


def _v1_qk(feature_map, sharing, apply, *, rank=4, mlp_hidden=12, enabled=True):
    return {
        "enabled": enabled,
        "feature_map": feature_map,
        "sharing": sharing,
        "apply": apply,
        "rank": rank,
        "mlp_hidden": mlp_hidden,
    }


def _build_qk(v1_or_v2, *, heads=4, head_dim=8, extent=16, rope_theta=10_000.0):
    return build_qk_position_channel(
        v1_or_v2,
        heads=heads,
        head_dim=head_dim,
        model_dim=heads * head_dim,
        extent=extent,
        rope_theta=rope_theta,
    )


class BasisAndRotaryTest(unittest.TestCase):
    def test_interleaved_basis_first_positions(self):
        basis = interleaved_fourier_basis(4, 4, 10_000.0)
        self.assertEqual(basis.shape, (4, 4))
        torch.testing.assert_close(
            basis[0],
            torch.tensor([1.0, 0.0, 1.0, 0.0]),
        )
        # pos=1, freq0 angle=1, freq1 angle=0.01
        expected = torch.tensor([
            math.cos(1.0),
            math.sin(1.0),
            math.cos(0.01),
            math.sin(0.01),
        ])
        torch.testing.assert_close(basis[1], expected, atol=1e-6, rtol=1e-6)

        module = FrozenFourierBasis(4, 4, 10_000.0)
        torch.testing.assert_close(module(2), basis[:2])

    def test_rotary_no_delta_and_nonzero_delta(self):
        head_dim = 8
        seq = 5
        sin, cos = build_rope_cache(seq, head_dim, 10_000.0)
        torch.manual_seed(0)
        q = torch.randn(2, 3, seq, head_dim)
        k = torch.randn(2, 3, seq, head_dim)

        # Manual no-delta reference.
        half = head_dim // 2
        sin_b = sin.to(q.dtype)[None, None, :, :]
        cos_b = cos.to(q.dtype)[None, None, :, :]

        def rotate(x, s, c):
            x1, x2 = x[..., :half], x[..., half:]
            return torch.cat([x1 * c - x2 * s, x1 * s + x2 * c], dim=-1)

        q_ref, k_ref = rotate(q, sin_b, cos_b), rotate(k, sin_b, cos_b)
        q_out = rotate_half(q, sin_b, cos_b)
        k_out = rotate_half(k, sin_b, cos_b)
        torch.testing.assert_close(q_out, q_ref)
        torch.testing.assert_close(k_out, k_ref)

        delta = torch.randn(3, seq, half)
        composed_sin, composed_cos = compose_phase(sin_b, cos_b, delta[None])
        q_delta = rotate_half(q, composed_sin, composed_cos)
        # Algebraic R(theta+delta) via complex multiply on first pair.
        # Nonzero delta must change the rotated result.
        self.assertFalse(torch.allclose(q_delta, q_out))

    def test_fixed_position_tables_survive_explicit_half_conversion(self):
        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                attention = Attention(32, 4, max_seq_len=1024)
                rope_sin = attention.rope_sin.clone()
                rope_cos = attention.rope_cos.clone()
                inverse_frequency = attention.rope_inverse_frequency.clone()
                attention.to(dtype=dtype)
                self.assertEqual(attention.rope_sin.dtype, torch.float32)
                self.assertEqual(attention.rope_cos.dtype, torch.float32)
                torch.testing.assert_close(attention.rope_sin, rope_sin)
                torch.testing.assert_close(attention.rope_cos, rope_cos)
                self.assertEqual(
                    attention.rope_inverse_frequency.dtype,
                    torch.float32,
                )
                torch.testing.assert_close(
                    attention.rope_inverse_frequency,
                    inverse_frequency,
                )

                basis = FrozenFourierBasis(1024, 8, 10_000.0)
                basis_reference = basis.basis.clone()
                basis.to(dtype=dtype)
                self.assertEqual(basis.basis.dtype, torch.float32)
                torch.testing.assert_close(basis.basis, basis_reference)

                channel = _build_qk(
                    _v1_qk("identity", "per_head", "add"),
                    extent=1024,
                )
                references = {
                    name: getattr(channel, name).clone()
                    for name in ("base_sin", "base_cos", "base_angle")
                }
                channel.to(dtype=dtype)
                for name, reference in references.items():
                    value = getattr(channel, name)
                    self.assertEqual(value.dtype, torch.float32)
                    torch.testing.assert_close(value, reference)
                self.assertEqual(channel.pipeline.basis.basis.dtype, torch.float32)

    def test_bf16_dynamic_phase_is_composed_in_fp32(self):
        sequence_length = 1024
        head_dim = 8
        half = head_dim // 2
        sin, cos = build_rope_cache(sequence_length, head_dim, 10_000.0)
        q = torch.zeros(
            1,
            1,
            sequence_length,
            head_dim,
            dtype=torch.bfloat16,
        )
        q[..., :half] = 1
        delta = torch.linspace(
            -1.0,
            1.0,
            sequence_length,
            dtype=torch.bfloat16,
        )[None, :, None].expand(1, -1, half)
        q_out, _ = apply_rotary(
            q,
            q,
            sin,
            cos,
            q_phase_delta=delta,
            k_phase_delta=delta,
        )

        delta_fp32 = delta.float()[None]
        sin_fp32 = sin[None, None]
        cos_fp32 = cos[None, None]
        expected_sin = (
            sin_fp32 * delta_fp32.cos()
            + cos_fp32 * delta_fp32.sin()
        ).to(torch.bfloat16)
        expected_cos = (
            cos_fp32 * delta_fp32.cos()
            - sin_fp32 * delta_fp32.sin()
        ).to(torch.bfloat16)
        expected = torch.cat((expected_cos, expected_sin), dim=-1)
        torch.testing.assert_close(q_out, expected, rtol=0, atol=0)


class PositionChannelTest(unittest.TestCase):
    def test_qk_phase_zero_init_and_add_sinusoid_init(self):
        for sharing in SHARING_MODES:
            for feature_map in FEATURE_MAPS:
                with self.subTest(sharing=sharing, feature_map=feature_map, apply="phase"):
                    phase = _build_qk(
                        _v1_qk(feature_map, sharing, "phase_residual")
                    )
                    output = phase(9)
                    self.assertEqual(output.q.shape, (4, 9, 4))
                    self.assertEqual(torch.count_nonzero(output.q).item(), 0)
                    torch.testing.assert_close(output.q, output.k)

                with self.subTest(sharing=sharing, feature_map=feature_map, apply="add"):
                    add = _build_qk(_v1_qk(feature_map, sharing, "add"))
                    output = add(9)
                    self.assertEqual(output.q.shape, (4, 9, 8))
                    if feature_map in ADDITIVE_SINUSOID_MAPS:
                        self.assertGreater(torch.count_nonzero(output.q).item(), 0)

    def test_phase_residual_matches_rope_at_init(self):
        common = {
            "dim": 32,
            "depth": 1,
            "heads": 4,
            "ff_mult": 2,
            "vocab_size": 64,
            "max_seq_len": 16,
            "attn_impl": "sdpa",
            "logit_bias_config": {"enabled": False},
        }
        baseline = Transformer(**common, qk_config={"enabled": False}).eval()
        input_ids = torch.randint(0, 64, (2, 8))
        expected = baseline(input_ids)

        candidate = Transformer(
            **common,
            qk_config=_v1_qk("mlp", "per_head", "phase_residual", rank=4, mlp_hidden=12),
        ).eval()
        candidate.load_state_dict(baseline.state_dict(), strict=False)
        actual = candidate(input_ids)
        torch.testing.assert_close(actual, expected)

    def test_additive_skips_multiplicative_rope(self):
        common = {
            "dim": 32,
            "depth": 1,
            "heads": 4,
            "ff_mult": 2,
            "vocab_size": 64,
            "max_seq_len": 16,
            "attn_impl": "sdpa",
            "logit_bias_config": {"enabled": False},
        }
        rope = Transformer(**common, qk_config={"enabled": False}).eval()
        additive = Transformer(
            **common,
            qk_config=_v1_qk("identity", "per_head", "add"),
        ).eval()
        self.assertTrue(rope.blocks[0].attn.multiplicative_rope)
        self.assertFalse(additive.blocks[0].attn.multiplicative_rope)

        additive.load_state_dict(rope.state_dict(), strict=False)
        input_ids = torch.randint(0, 64, (2, 8))
        self.assertFalse(torch.allclose(rope(input_ids), additive(input_ids)))

    def test_additive_residual_maps_match_identity_at_init(self):
        for feature_map in ("low_rank", "bottleneck_mlp", "mlp"):
            with self.subTest(feature_map=feature_map):
                identity = _build_qk(_v1_qk("identity", "per_head", "add"))
                residual = _build_qk(_v1_qk(feature_map, "per_head", "add"))
                torch.testing.assert_close(identity(11).q, residual(11).q)

    def test_tiny_parameter_count_fixtures(self):
        common = dict(
            dim=32,
            depth=1,
            heads=4,
            ff_mult=2,
            vocab_size=64,
            max_seq_len=16,
            attn_impl="sdpa",
            logit_bias_config={"enabled": False},
        )
        rope = count_parameters(Transformer(**common, qk_config={"enabled": False}))
        phase = count_parameters(
            Transformer(
                **common,
                qk_config=_v1_qk("mlp", "per_head", "phase_residual", rank=4, mlp_hidden=12),
            )
        )
        self.assertEqual(rope["total"], 15936)
        self.assertEqual(phase["qk_position_params"], 992)
        self.assertEqual(phase["total"], 16928)
        self.assertEqual(rope["logit_bias_params"], 0)


class QKCouplingTest(unittest.TestCase):
    def _v2_qk(self, *, application, head_coupling, qk_coupling, mapper_kind="identity"):
        residual = mapper_kind in {"low_rank", "bottleneck_mlp", "mlp"}
        heads = 4
        head_dim = 8
        model_dim = heads * head_dim
        basis_dim = model_dim if head_coupling == "per_head_joint" else head_dim
        return {
            "enabled": True,
            "application": application,
            "geometry": "free" if application == "additive" else "phase",
            "input": {
                "kind": "frozen_fourier",
                "basis_dim": basis_dim,
                "theta": None,
                "scalars": [],
            },
            "mapper": {
                "kind": mapper_kind,
                "residual": residual,
                "rank": 4,
                "hidden_dim": 12,
            },
            "qk_coupling": qk_coupling,
            "head_coupling": head_coupling,
        }

    def test_coupling_shapes_init_and_independence(self):
        for application in ("additive", "rotary"):
            for head_coupling in HEAD_COUPLINGS:
                for qk_coupling in QK_COUPLINGS:
                    with self.subTest(
                        application=application,
                        head_coupling=head_coupling,
                        qk_coupling=qk_coupling,
                    ):
                        cfg = self._v2_qk(
                            application=application,
                            head_coupling=head_coupling,
                            qk_coupling=qk_coupling,
                            mapper_kind="mlp",
                        )
                        channel = _build_qk(cfg)
                        output = channel(9)
                        if application == "additive":
                            self.assertEqual(output.q.shape, (4, 9, 8))
                        else:
                            self.assertEqual(output.q.shape, (4, 9, 4))
                            self.assertEqual(torch.count_nonzero(output.q).item(), 0)
                            self.assertEqual(torch.count_nonzero(output.k).item(), 0)
                        torch.testing.assert_close(output.q, output.k)

                        if qk_coupling == "separate":
                            q_ids = {id(p) for p in channel.q_pipeline.parameters()}
                            k_ids = {id(p) for p in channel.k_pipeline.parameters()}
                            self.assertTrue(q_ids.isdisjoint(k_ids))

    def test_separate_modes_receive_distinct_gradients(self):
        for qk_coupling in ("shared_trunk_separate_readouts", "separate"):
            with self.subTest(qk_coupling=qk_coupling):
                cfg = self._v2_qk(
                    application="additive",
                    head_coupling="per_head_independent",
                    qk_coupling=qk_coupling,
                    mapper_kind="linear",
                )
                channel = _build_qk(cfg)
                output = channel(8)
                loss = output.q.pow(2).sum() - output.k.pow(2).sum()
                loss.backward()
                if qk_coupling == "shared_trunk_separate_readouts":
                    self.assertIsNotNone(channel.q_add_readout.weight.grad)
                    self.assertIsNotNone(channel.k_add_readout.weight.grad)
                    self.assertFalse(
                        torch.allclose(
                            channel.q_add_readout.weight.grad,
                            channel.k_add_readout.weight.grad,
                        )
                    )
                else:
                    q_grad = next(channel.q_pipeline.parameters()).grad
                    k_grad = next(channel.k_pipeline.parameters()).grad
                    self.assertIsNotNone(q_grad)
                    self.assertIsNotNone(k_grad)
                    self.assertFalse(torch.allclose(q_grad, k_grad))

    def test_parameter_count_ordering(self):
        counts = {}
        for qk_coupling in QK_COUPLINGS:
            cfg = self._v2_qk(
                application="rotary",
                head_coupling="per_head_independent",
                qk_coupling=qk_coupling,
                mapper_kind="mlp",
            )
            channel = _build_qk(cfg)
            counts[qk_coupling] = sum(p.numel() for p in channel.parameters())
        self.assertLess(counts["shared"], counts["shared_trunk_separate_readouts"])
        self.assertLess(counts["shared_trunk_separate_readouts"], counts["separate"])


class PositionConfigTest(unittest.TestCase):
    def _load(self, overrides):
        with tempfile.NamedTemporaryFile("w", suffix=".json") as config_file:
            json.dump(overrides, config_file)
            config_file.flush()
            return load_config(_cli(config_file.name))

    def test_qk_only_keeps_sdpa(self):
        config = self._load(
            {
                "qk": {
                    "enabled": True,
                    "feature_map": "mlp",
                    "sharing": "shared_head",
                    "apply": "phase_residual",
                },
            }
        )
        self.assertTrue(config.qk["enabled"])
        self.assertFalse(config.logit_bias["enabled"])
        self.assertEqual(config.attn_impl, "sdpa")
        self.assertEqual(config.pos_variant, "custom")
        self.assertEqual(config.qk["application"], "rotary")
        self.assertEqual(config.qk["head_coupling"], "shared_head")

    def test_carrier_hypernetwork_schema_round_trip_and_rejections(self):
        base = {
            "enabled": True,
            "application": "additive",
            "geometry": "amplitude_phase",
            "input": {
                "kind": "frozen_fourier",
                "basis_dim": 8,
                "theta": None,
                "scalars": [],
            },
            "mapper": {
                "kind": "identity",
                "residual": False,
                "rank": 4,
                "hidden_dim": 12,
            },
            "output": {
                "learn_amplitude": False,
                "learn_phase": False,
            },
            "conditioning": {
                "kind": "carrier_hypernetwork",
                "source": "dedicated",
                "input_mode": "content_position",
                "network": "swiglu_mlp",
                "components": "phase",
                "target": "both",
                "coupling": "separate",
                "head_coupling": "shared_head",
                "hidden_dim": 16,
            },
            "qk_coupling": "shared",
            "head_coupling": "per_head_independent",
        }
        normalized = normalize_position_config_v2(
            "qk",
            base,
            model_dim=32,
            heads=4,
            rope_theta=10_000.0,
        )
        self.assertEqual(
            normalized["conditioning"]["input_mode"], "content_position"
        )
        self.assertEqual(normalized["conditioning"]["network"], "swiglu_mlp")
        self.assertEqual(normalized["conditioning"]["coupling"], "separate")
        self.assertEqual(
            normalize_position_config_v2(
                "qk",
                json.loads(json.dumps(normalized)),
                model_dim=32,
                heads=4,
                rope_theta=10_000.0,
            ),
            normalized,
        )

        invalid_values = {
            "input_mode": "cross_token",
            "input_normalization": "layer_norm",
            "network": "relu_mlp",
            "components": "gain_only",
            "head_coupling": "per_head_joint",
        }
        for key, value in invalid_values.items():
            with self.subTest(key=key), self.assertRaises(ValueError):
                invalid = json.loads(json.dumps(base))
                invalid["conditioning"][key] = value
                normalize_position_config_v2(
                    "qk",
                    invalid,
                    model_dim=32,
                    heads=4,
                    rope_theta=10_000.0,
                )

        normalized_inputs = json.loads(json.dumps(base))
        normalized_inputs["conditioning"]["input_normalization"] = "modality_rms"
        normalized_inputs["conditioning"]["learnable_input_gains"] = True
        normalized_with_gains = normalize_position_config_v2(
            "qk",
            normalized_inputs,
            model_dim=32,
            heads=4,
            rope_theta=10_000.0,
        )
        self.assertTrue(
            normalized_with_gains["conditioning"]["learnable_input_gains"]
        )
        invalid_gains = json.loads(json.dumps(base))
        invalid_gains["conditioning"]["learnable_input_gains"] = True
        with self.assertRaises(ValueError):
            normalize_position_config_v2(
                "qk",
                invalid_gains,
                model_dim=32,
                heads=4,
                rope_theta=10_000.0,
            )
        invalid_nonhyper = json.loads(json.dumps(normalized_inputs))
        invalid_nonhyper["conditioning"]["kind"] = "none"
        with self.assertRaises(ValueError):
            normalize_position_config_v2(
                "qk",
                invalid_nonhyper,
                model_dim=32,
                heads=4,
                rope_theta=10_000.0,
            )

        dynamic = json.loads(json.dumps(base))
        dynamic["output"]["amplitude_parameterization"] = "softplus"
        dynamic["conditioning"]["components"] = "amplitude_phase"
        normalized_dynamic = normalize_position_config_v2(
            "qk",
            dynamic,
            model_dim=32,
            heads=4,
            rope_theta=10_000.0,
        )
        self.assertEqual(
            normalized_dynamic["conditioning"]["components"],
            "amplitude_phase",
        )
        signed_dynamic = json.loads(json.dumps(dynamic))
        signed_dynamic["output"]["amplitude_init"] = 1.0
        signed_dynamic["output"]["amplitude_parameterization"] = "signed"
        normalized_signed = normalize_position_config_v2(
            "qk",
            signed_dynamic,
            model_dim=32,
            heads=4,
            rope_theta=10_000.0,
        )
        self.assertEqual(
            normalized_signed["output"]["amplitude_parameterization"],
            "signed",
        )
        asymmetric = json.loads(json.dumps(signed_dynamic))
        asymmetric["output"]["parameter_source"] = "direct"
        asymmetric["conditioning"]["target"] = "q"
        asymmetric["conditioning"]["static_complement"] = True
        asymmetric["qk_coupling"] = "shared_trunk_separate_readouts"
        normalized_asymmetric = normalize_position_config_v2(
            "qk",
            asymmetric,
            model_dim=32,
            heads=4,
            rope_theta=10_000.0,
        )
        self.assertTrue(
            normalized_asymmetric["conditioning"]["static_complement"]
        )
        for key, value in (
            ("target", "both"),
            ("parameter_source", "mapped"),
            ("qk_coupling", "shared"),
        ):
            with self.subTest(asymmetric_key=key), self.assertRaises(ValueError):
                invalid = json.loads(json.dumps(asymmetric))
                if key == "target":
                    invalid["conditioning"][key] = value
                elif key == "parameter_source":
                    invalid["output"][key] = value
                else:
                    invalid[key] = value
                normalize_position_config_v2(
                    "qk",
                    invalid,
                    model_dim=32,
                    heads=4,
                    rope_theta=10_000.0,
                )
        with self.assertRaises(ValueError):
            invalid = json.loads(json.dumps(dynamic))
            invalid["output"]["amplitude_parameterization"] = "bounded_sigmoid"
            normalize_position_config_v2(
                "qk",
                invalid,
                model_dim=32,
                heads=4,
                rope_theta=10_000.0,
            )

        for mutation in ("learn_amplitude", "learn_phase"):
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                invalid = json.loads(json.dumps(dynamic))
                invalid["output"][mutation] = True
                normalize_position_config_v2(
                    "qk",
                    invalid,
                    model_dim=32,
                    heads=4,
                    rope_theta=10_000.0,
                )

        direct = json.loads(json.dumps(base))
        direct["conditioning"] = {"kind": "none"}
        direct["output"] = {
            "parameter_source": "direct",
            "amplitude_parameterization": "softplus",
            "learn_amplitude": True,
            "learn_phase": True,
        }
        normalized_direct = normalize_position_config_v2(
            "qk",
            direct,
            model_dim=32,
            heads=4,
            rope_theta=10_000.0,
        )
        self.assertEqual(normalized_direct["output"]["parameter_source"], "direct")

    def test_addrope_run_tag(self):
        config = self._load(
            {
                "qk": {
                    "enabled": True,
                    "feature_map": "identity",
                    "apply": "add",
                },
            }
        )
        self.assertEqual(config.qk["application"], "additive")
        self.assertTrue(config.run_name.startswith("qk-add-identity"))

    def test_legacy_feature_map_upgrade_table(self):
        cases = [
            ("identity", "add", "identity", False, "additive", "free"),
            ("add_rope", "add", "euclidean_affine", False, "additive", "free"),
            ("linear", "phase_residual", "linear", False, "rotary", "phase"),
            ("low_rank", "phase_residual", "low_rank", True, "rotary", "phase"),
            ("bottleneck_mlp", "add", "bottleneck_mlp", True, "additive", "free"),
            ("mlp", "phase_residual", "mlp", True, "rotary", "phase"),
        ]
        for feature_map, apply, mapper, residual, application, geometry in cases:
            with self.subTest(feature_map=feature_map, apply=apply):
                upgraded = upgrade_legacy_position_config(
                    "qk",
                    _v1_qk(feature_map, "per_head", apply),
                    model_dim=32,
                    heads=4,
                    rope_theta=10_000.0,
                )
                self.assertEqual(upgraded["mapper"]["kind"], mapper)
                self.assertEqual(upgraded["mapper"]["residual"], residual)
                self.assertEqual(upgraded["application"], application)
                self.assertEqual(upgraded["geometry"], geometry)
                self.assertEqual(upgraded["qk_coupling"], "shared")

    def test_sharing_upgrade(self):
        mapping = {
            "shared_head": "shared_head",
            "per_head": "per_head_independent",
            "full_dim": "per_head_joint",
        }
        for sharing, head_coupling in mapping.items():
            upgraded = upgrade_legacy_position_config(
                "qk",
                _v1_qk("linear", sharing, "phase_residual"),
                model_dim=32,
                heads=4,
                rope_theta=10_000.0,
            )
            self.assertEqual(upgraded["head_coupling"], head_coupling)
            expected_basis = 32 if sharing == "full_dim" else 8
            self.assertEqual(upgraded["input"]["basis_dim"], expected_basis)

    def test_rejects_mixed_keys_and_unsupported_modes(self):
        with self.assertRaises(ValueError):
            detect_channel_schema(
                "qk",
                {"feature_map": "linear", "application": "rotary"},
            )
        with self.assertRaises(ValueError):
            normalize_position_config_v2(
                "qk",
                {
                    "enabled": True,
                    "application": "additive",
                    "geometry": "phase",
                    "input": {"kind": "frozen_fourier", "basis_dim": 8, "theta": None, "scalars": []},
                    "mapper": {"kind": "identity", "residual": False, "rank": 4, "hidden_dim": 12},
                    "qk_coupling": "shared",
                    "head_coupling": "per_head_independent",
                },
                model_dim=32,
                heads=4,
                rope_theta=10_000.0,
            )
        with self.assertRaises(ValueError):
            normalize_position_config_v2(
                "qk",
                {
                    "enabled": True,
                    "application": "rotary",
                    "geometry": "phase",
                    "input": {
                        "kind": "learned_fourier",
                        "basis_dim": 8,
                        "theta": None,
                        "scalars": [],
                    },
                    "mapper": {
                        "kind": "identity",
                        "residual": False,
                        "rank": 4,
                        "hidden_dim": 12,
                    },
                    "qk_coupling": "shared",
                    "head_coupling": "per_head_independent",
                },
                model_dim=32,
                heads=4,
                rope_theta=10_000.0,
            )

    def test_json_round_trip_canonical(self):
        upgraded = upgrade_legacy_position_config(
            "qk",
            _v1_qk("mlp", "full_dim", "phase_residual"),
            model_dim=32,
            heads=4,
            rope_theta=10_000.0,
        )
        payload = json.loads(json.dumps(upgraded))
        again = normalize_position_config_v2(
            "qk",
            payload,
            model_dim=32,
            heads=4,
            rope_theta=10_000.0,
        )
        self.assertEqual(again, upgraded)

    def test_all_sweep_configs_load(self):
        paths = sorted(SWEEP_CONFIG_DIR.glob("*.json"))
        self.assertEqual(len(paths), 8)
        for path in paths:
            with self.subTest(config=path.name):
                config = load_config(_cli(str(path)))
                self.assertEqual(config.position_schema_version, 2)
                model = Transformer(
                    dim=config.hidden_size,
                    depth=1,
                    heads=config.n_head,
                    ff_mult=2,
                    vocab_size=128,
                    max_seq_len=32,
                    qk_config=config.qk,
                    logit_bias_config=config.logit_bias,
                    attn_impl=config.attn_impl,
                    rel_extent=min(32, config.rel_extent or config.block_size),
                )
                counts = count_parameters(model)
                self.assertGreaterEqual(counts["total"], counts["position_params"])


class CompatibilityTest(unittest.TestCase):
    def test_state_dict_adapter_shared_phase(self):
        common = {
            "dim": 32,
            "depth": 1,
            "heads": 4,
            "ff_mult": 2,
            "vocab_size": 64,
            "max_seq_len": 16,
            "attn_impl": "sdpa",
            "logit_bias_config": {"enabled": False},
            "qk_config": _v1_qk("linear", "per_head", "phase_residual"),
        }
        model = Transformer(**common).eval()
        # Synthesize a legacy-named state dict from the live v2 module.
        state = model.state_dict()
        legacy = {}
        for key, value in state.items():
            legacy_key = (
                key.replace(".pipeline.mapper.", ".features.")
                .replace(".phase_head.weight", ".output_weight")
                .replace(".phase_head.bias", ".output_bias")
            )
            legacy[legacy_key] = value
        clone = Transformer(**common).eval()
        with self.assertWarns(UserWarning):
            missing = clone.load_state_dict(legacy, strict=True)
        self.assertEqual(len(getattr(missing, "missing_keys", [])), 0)
        input_ids = torch.randint(0, 64, (2, 8))
        torch.testing.assert_close(model(input_ids), clone(input_ids))

    def test_optimizer_step_after_adapted_load(self):
        common = {
            "dim": 32,
            "depth": 1,
            "heads": 4,
            "ff_mult": 2,
            "vocab_size": 64,
            "max_seq_len": 16,
            "attn_impl": "sdpa",
            "logit_bias_config": {"enabled": False},
            "qk_config": _v1_qk("mlp", "per_head", "phase_residual", rank=4, mlp_hidden=12),
        }
        model = Transformer(**common)
        state = {
            key.replace(".pipeline.mapper.", ".features.")
            .replace(".phase_head.weight", ".output_weight")
            .replace(".phase_head.bias", ".output_bias"): value
            for key, value in model.state_dict().items()
        }
        restored = Transformer(**common)
        restored.load_state_dict(state, strict=True)
        optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-3)
        input_ids = torch.randint(0, 64, (2, 8))
        targets = torch.randint(0, 64, (2, 8))
        loss = restored(input_ids, targets)
        loss.backward()
        optimizer.step()
        self.assertTrue(torch.isfinite(loss).item())

    def test_qk_diagnostic_keys(self):
        model = Transformer(
            dim=32,
            depth=1,
            heads=4,
            ff_mult=2,
            vocab_size=64,
            max_seq_len=16,
            attn_impl="sdpa",
            qk_config=_v1_qk("identity", "per_head", "add"),
            logit_bias_config={"enabled": False},
        )
        metrics, _ = model.position_diagnostics(sequence_length=8)
        self.assertIn("position/layer_00/qk/q/rms", metrics)
        self.assertIn("position/layer_00/qk/qk_diff_rms", metrics)
        self.assertEqual(metrics["position/layer_00/qk/qk_diff_rms"], 0.0)


class SpectralCarrierComponentsTest(unittest.TestCase):
    """Narrow relativity-preserving carrier readouts (slope / position offset)."""

    HEADS = 4
    HEAD_DIM = 8
    PAIR_DIM = HEAD_DIM // 2

    def _cfg(
        self,
        components,
        *,
        amplitude_init=1.0,
        offset_parameterization="tanh",
        target="both",
    ):
        return {
            "enabled": True,
            "application": "additive",
            "geometry": "amplitude_phase",
            "input": {
                "kind": "frozen_fourier",
                "basis_dim": 8,
                "theta": None,
                "scalars": [],
            },
            "output": {
                "parameter_source": "direct",
                "learn_amplitude": False,
                "learn_phase": False,
                "amplitude_init": amplitude_init,
                "amplitude_parameterization": "signed",
            },
            "conditioning": {
                "kind": "carrier_hypernetwork",
                "source": "dedicated",
                "input_mode": "content_position",
                "network": "silu_mlp",
                "components": components,
                "target": target,
                "coupling": "shared_trunk_separate_readouts",
                "head_coupling": "per_head_independent",
                "hidden_dim": 16,
                "offset_parameterization": offset_parameterization,
            },
            "qk_coupling": "shared_trunk_separate_readouts",
            "head_coupling": "per_head_independent",
        }

    def _channel(self, components, **kwargs):  # noqa: D401
        normalized = normalize_position_config_v2(
            "qk",
            self._cfg(components, **kwargs),
            model_dim=self.HEADS * self.HEAD_DIM,
            heads=self.HEADS,
            rope_theta=10_000.0,
        )
        return _build_qk(
            normalized,
            heads=self.HEADS,
            head_dim=self.HEAD_DIM,
        )

    def _content(self, length, *, seed=0):
        generator = torch.Generator().manual_seed(seed)
        content = torch.randn(
            2,
            self.HEADS,
            length,
            self.HEAD_DIM,
            generator=generator,
        )
        inverse_rms = torch.rsqrt(content.square().mean(-1, keepdim=True) + 1e-6)
        return content * inverse_rms

    def test_readout_width_is_one_scalar_per_component(self):
        PAIR = self.PAIR_DIM
        expected = {
            "amplitude_slope": (2, (1, 1)),
            "position_offset": (1, (1,)),
            "slope_offset": (3, (1, 1, 1)),
            # Mixed modes narrow exactly one branch.
            "amplitude_offset": (PAIR + 1, (PAIR, 1)),
            "slope_phase": (2 + PAIR, (1, 1, PAIR)),
        }
        for components, (width, widths) in expected.items():
            with self.subTest(components=components):
                channel = self._channel(components)
                hyper = channel.carrier_hypernetwork
                self.assertTrue(hyper.spectral)
                self.assertEqual(hyper.component_widths, widths)
                self.assertEqual(hyper.q_readout.weight.shape[-1], width)
                # Strictly narrower than the free per-frequency readout.
                self.assertLess(width, self.PAIR_DIM * 2)

    def test_asymmetric_narrow_targets_use_exact_component_width(self):
        length = 6
        content = self._content(length)
        for components in (
            "amplitude_slope",
            "position_offset",
            "slope_offset",
            "amplitude_offset",
            "slope_phase",
        ):
            for target in ("q", "k"):
                with self.subTest(components=components, target=target):
                    channel = self._channel(components, target=target)
                    output = channel(
                        length,
                        q_content=content,
                        k_content=content,
                    )
                    self.assertEqual(output.q.shape, content.shape)
                    self.assertEqual(output.k.shape, content.shape)

    def test_zero_readout_recovers_exact_rope_anchor(self):
        length = 6
        content = self._content(length)
        other = self._content(length, seed=7)
        for components in (
            "amplitude_slope",
            "position_offset",
            "slope_offset",
            "amplitude_offset",
            "slope_phase",
        ):
            with self.subTest(components=components):
                channel = self._channel(components)
                # amplitude_init=1 with zeroed readouts must be exactly
                # 1 * cis(omega * p), independent of the content fed in.
                expected = torch.cat(
                    (
                        channel.base_cos[:length],
                        channel.base_sin[:length],
                    ),
                    dim=-1,
                )
                dynamic = channel(
                    length,
                    q_content=content,
                    k_content=content,
                )
                torch.testing.assert_close(
                    dynamic.q,
                    expected.expand_as(dynamic.q),
                )
                torch.testing.assert_close(
                    dynamic.k,
                    expected.expand_as(dynamic.k),
                )
                shifted = channel(length, q_content=other, k_content=other)
                torch.testing.assert_close(shifted.q, dynamic.q)

    def test_position_offset_is_a_pure_position_shift(self):
        """phase = omega * m must equal evaluating the carrier at p + m."""
        shift = 3.0
        self._check_shift("tanh", math.atanh(shift / 8.0), shift)

    def _check_shift(self, parameterization, raw_value, shift):
        channel = self._channel(
            "position_offset",
            offset_parameterization=parameterization,
        )
        hyper = channel.carrier_hypernetwork
        omega = hyper.spectral_omega
        raw = torch.full((1, self.HEADS, 5, 1), raw_value)
        phase = hyper._parse(raw).phase
        expected = shift * omega
        torch.testing.assert_close(
            phase,
            expected.expand_as(phase),
            atol=1e-5,
            rtol=1e-4,
        )
        # A shift of m advances the carrier angle by exactly omega*m, i.e. it is
        # the carrier evaluated at position p+m -- so the logit still depends
        # only on the difference of shifted positions.
        positions = torch.arange(5, dtype=torch.float32)[:, None]
        base = positions * omega[None, :]
        torch.testing.assert_close(
            base + phase[0, 0],
            (positions + shift) * omega[None, :],
            atol=1e-5,
            rtol=1e-4,
        )

    def test_amplitude_slope_tilts_across_log_frequency(self):
        channel = self._channel("amplitude_slope")
        hyper = channel.carrier_hypernetwork
        tilt = hyper.spectral_tilt
        # Zero mean, unit scale, and monotone decreasing in frequency index
        # (omega decreases with index for the standard RoPE schedule).
        self.assertAlmostEqual(float(tilt.mean()), 0.0, places=5)
        self.assertAlmostEqual(float(tilt.std(unbiased=False)), 1.0, places=5)
        self.assertTrue(bool((tilt[1:] < tilt[:-1]).all()))
        raw = torch.zeros(1, self.HEADS, 3, 2)
        raw[..., 1] = 0.5  # slope only
        amplitude = hyper._parse(raw).amplitude
        torch.testing.assert_close(
            amplitude,
            (0.5 * tilt).expand_as(amplitude),
            atol=1e-5,
            rtol=1e-4,
        )
        raw2 = torch.zeros(1, self.HEADS, 3, 2)
        raw2[..., 0] = 0.25  # gain only -> flat across frequency
        flat = hyper._parse(raw2).amplitude
        self.assertTrue(
            bool((flat - flat[..., :1]).abs().max() < 1e-6)
        )

    def test_gradients_reach_narrow_readouts_and_differ_across_qk(self):
        length = 6
        content = self._content(length, seed=1)
        channel = self._channel("slope_offset")
        with torch.no_grad():
            for readout in (
                channel.carrier_hypernetwork.q_readout,
                channel.carrier_hypernetwork.k_readout,
            ):
                readout.weight.normal_(std=0.02)
        output = channel(length, q_content=content, k_content=content)
        (output.q.pow(2).sum() - output.k.pow(2).sum()).backward()
        q_grad = channel.carrier_hypernetwork.q_readout.weight.grad
        k_grad = channel.carrier_hypernetwork.k_readout.weight.grad
        self.assertIsNotNone(q_grad)
        self.assertIsNotNone(k_grad)
        self.assertFalse(torch.allclose(q_grad, k_grad))

    def test_readout_head_mixing_keeps_anchor_and_mixes_heads(self):
        length = 6
        content = self._content(length, seed=3)
        cfg = self._cfg("amplitude_phase")
        cfg["conditioning"]["readout_head_mixing"] = True
        normalized = normalize_position_config_v2(
            "qk",
            cfg,
            model_dim=self.HEADS * self.HEAD_DIM,
            heads=self.HEADS,
            rope_theta=10_000.0,
        )
        channel = _build_qk(normalized, heads=self.HEADS, head_dim=self.HEAD_DIM)
        hyper = channel.carrier_hypernetwork
        self.assertTrue(hyper.readout_head_mixing)
        # One dense [groups*hidden, groups*out] map rather than a batch of
        # per-head matrices.
        self.assertEqual(hyper.q_readout.weight.shape[0], 1)
        self.assertEqual(
            hyper.q_readout.weight.shape[-1],
            self.HEADS * hyper.readout_output_dim,
        )
        # Still starts at the exact carrier.
        expected = torch.cat(
            (channel.base_cos[:length], channel.base_sin[:length]), dim=-1
        )
        out = channel(length, q_content=content, k_content=content)
        torch.testing.assert_close(out.q, expected.expand_as(out.q))

        # With mixing on, perturbing one head's trunk features must be able to
        # change another head's output.
        with torch.no_grad():
            hyper.q_readout.weight.normal_(std=0.05)
        mixed = channel(length, q_content=content, k_content=content)
        self.assertFalse(torch.allclose(mixed.q, expected.expand_as(mixed.q)))

    def test_lowrank_head_mixing_anchor_and_rank_independent_scale(self):
        length = 6
        content = self._content(length, seed=5)
        for rank in (2, 4):
            with self.subTest(rank=rank):
                cfg = self._cfg("amplitude_phase")
                cfg["conditioning"]["readout_head_mixing"] = "lowrank"
                cfg["conditioning"]["readout_mix_rank"] = rank
                cfg["conditioning"]["readout_mix_alpha"] = 4.0
                normalized = normalize_position_config_v2(
                    "qk", cfg,
                    model_dim=self.HEADS * self.HEAD_DIM,
                    heads=self.HEADS, rope_theta=10_000.0,
                )
                channel = _build_qk(
                    normalized, heads=self.HEADS, head_dim=self.HEAD_DIM
                )
                readout = channel.carrier_hypernetwork.q_readout
                # LoRA convention: up is zero, down is random with a fan-in
                # that does not depend on rank.
                self.assertEqual(readout.rank, rank)
                self.assertTrue(bool((readout.up == 0).all()))
                self.assertFalse(bool((readout.down == 0).all()))
                self.assertAlmostEqual(readout.scale, 4.0 / rank, places=6)
                # Down-matrix scale must be rank-independent.
                self.assertAlmostEqual(
                    float(readout.down.std()),
                    (2.0 / (self.HEADS * 16)) ** 0.5,
                    delta=0.05,
                )
                # Exact carrier anchor despite the extra path.
                expected = torch.cat(
                    (channel.base_cos[:length], channel.base_sin[:length]),
                    dim=-1,
                )
                out = channel(length, q_content=content, k_content=content)
                torch.testing.assert_close(out.q, expected.expand_as(out.q))
                # reset_output_parameters must restore the anchor too.
                with torch.no_grad():
                    readout.up.normal_(std=0.1)
                channel.carrier_hypernetwork.reset_output_parameters()
                reset = channel(length, q_content=content, k_content=content)
                torch.testing.assert_close(reset.q, expected.expand_as(reset.q))

    def test_lowrank_mixing_is_cheaper_than_dense(self):
        def count(mode):
            cfg = self._cfg("amplitude_phase")
            cfg["conditioning"]["readout_head_mixing"] = mode
            cfg["conditioning"]["readout_mix_rank"] = 4
            normalized = normalize_position_config_v2(
                "qk", cfg,
                model_dim=self.HEADS * self.HEAD_DIM,
                heads=self.HEADS, rope_theta=10_000.0,
            )
            channel = _build_qk(
                normalized, heads=self.HEADS, head_dim=self.HEAD_DIM
            )
            return sum(p.numel() for p in channel.parameters())

        self.assertLess(count("none"), count("lowrank"))
        self.assertLess(count("lowrank"), count("dense"))

    def test_schema_rejections(self):
        with self.assertRaises(ValueError):
            bad = self._cfg("amplitude_slope")
            bad["output"]["learn_amplitude"] = True
            normalize_position_config_v2(
                "qk",
                bad,
                model_dim=self.HEADS * self.HEAD_DIM,
                heads=self.HEADS,
                rope_theta=10_000.0,
            )
        with self.assertRaises(ValueError):
            bad = self._cfg("position_offset")
            bad["output"]["learn_phase"] = True
            normalize_position_config_v2(
                "qk",
                bad,
                model_dim=self.HEADS * self.HEAD_DIM,
                heads=self.HEADS,
                rope_theta=10_000.0,
            )
        with self.assertRaises(ValueError):
            bad = self._cfg("slope_offset")
            bad["conditioning"]["offset_bound"] = 0.0
            normalize_position_config_v2(
                "qk",
                bad,
                model_dim=self.HEADS * self.HEAD_DIM,
                heads=self.HEADS,
                rope_theta=10_000.0,
            )
        with self.assertRaises(ValueError):
            bad = self._cfg("amplitude_slope")
            bad["application"] = "rotary"
            bad["geometry"] = "phase"
            normalize_position_config_v2(
                "qk",
                bad,
                model_dim=self.HEADS * self.HEAD_DIM,
                heads=self.HEADS,
                rope_theta=10_000.0,
            )


if __name__ == "__main__":
    unittest.main()
