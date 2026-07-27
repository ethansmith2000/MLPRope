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
    build_logit_bias_channel,
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
from transformer import Transformer, count_parameters


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


def _v1_logit(feature_map, sharing, *, rank=4, mlp_hidden=12, enabled=True):
    return {
        "enabled": enabled,
        "feature_map": feature_map,
        "sharing": sharing,
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


def _build_logit(v1_or_v2, *, heads=4, head_dim=8, extent=16, rope_theta=10_000.0):
    return build_logit_bias_channel(
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


class PositionChannelTest(unittest.TestCase):
    def test_logit_contract_and_zero_init(self):
        for sharing in SHARING_MODES:
            for feature_map in FEATURE_MAPS:
                with self.subTest(sharing=sharing, feature_map=feature_map):
                    channel = _build_logit(_v1_logit(feature_map, sharing))
                    curves = channel()
                    self.assertEqual(curves.shape, (4, 16))
                    self.assertEqual(torch.count_nonzero(curves).item(), 0)

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

    def test_low_rank_and_bottleneck_have_fixed_feature_shape(self):
        low_rank = _build_logit(_v1_logit("low_rank", "per_head", rank=3))
        bottleneck = _build_logit(_v1_logit("bottleneck_mlp", "per_head", rank=3))
        self.assertEqual(low_rank.pipeline(dtype=None).shape, (4, 16, 8))
        self.assertEqual(bottleneck.pipeline(dtype=None).shape, (4, 16, 8))
        self.assertEqual(low_rank.pipeline.mapper.up.shape, (4, 3, 8))
        self.assertEqual(bottleneck.pipeline.mapper.up.shape, (4, 3, 8))

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
        logit = count_parameters(
            Transformer(
                dim=32,
                depth=1,
                heads=4,
                ff_mult=2,
                vocab_size=64,
                max_seq_len=16,
                attn_impl="flex",
                qk_config={"enabled": False},
                logit_bias_config=_v1_logit("linear", "per_head"),
            )
        )
        self.assertEqual(rope["total"], 15936)
        self.assertEqual(phase["qk_position_params"], 992)
        self.assertEqual(phase["total"], 16928)
        self.assertEqual(logit["logit_bias_params"], 324)
        self.assertEqual(logit["total"], 16260)


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

    def test_legacy_preset_expands_to_logit_channel(self):
        config = self._load({"pos_variant": "low_rank", "pos_rank": 7})
        self.assertFalse(config.qk["enabled"])
        self.assertTrue(config.logit_bias["enabled"])
        self.assertEqual(config.logit_bias["mapper"]["kind"], "low_rank")
        self.assertEqual(config.logit_bias["mapper"]["rank"], 7)
        self.assertEqual(config.attn_impl, "flex")
        self.assertEqual(config.position_schema_version, 2)
        self.assertEqual(config.position_source_schema, 1)

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
                "components": "log_gain_phase",
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

    def test_explicit_channels_can_enable_both(self):
        config = self._load(
            {
                "qk": {"enabled": True, "feature_map": "linear"},
                "logit_bias": {
                    "enabled": True,
                    "feature_map": "bottleneck_mlp",
                },
            }
        )
        self.assertTrue(config.qk["enabled"])
        self.assertTrue(config.logit_bias["enabled"])
        self.assertEqual(config.attn_impl, "flex")
        self.assertIn("+", config.run_name)

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
                "logit_bias",
                _v1_logit("linear", sharing),
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
        self.assertEqual(len(paths), 14)
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

    def test_logit_diagnostic_keys(self):
        model = Transformer(
            dim=32,
            depth=2,
            heads=4,
            ff_mult=2,
            vocab_size=64,
            max_seq_len=16,
            attn_impl="flex",
            qk_config={"enabled": False},
            logit_bias_config=_v1_logit("linear", "per_head"),
        )
        metrics, profiles = model.position_diagnostics(sequence_length=8)
        self.assertIn("position/layer_00/bias_mean", metrics)
        self.assertIn("position/layer_00/bias_std", metrics)
        self.assertIn("position/layer_00/bias_abs_max", metrics)
        self.assertIn("layer_00", profiles)

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


if __name__ == "__main__":
    unittest.main()
