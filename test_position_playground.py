"""Coverage for the fully configurable position experimentation playground."""

from __future__ import annotations

import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import torch

from position.channels import GroupedContentConditioner
from position import (
    FeatureMapper,
    build_position_basis,
    build_qk_position_channel,
    interleaved_fourier_basis,
    normalize_attention_write_config,
    normalize_position_config_v2,
    normalize_residual_stream_config,
)
from train_gpt import RechunkedTokenDataset, load_config
from transformer import (
    Attention,
    Transformer,
    TransformerBlock,
    suggest_matched_baselines,
)


def qk_config(
    application: str,
    geometry: str,
    *,
    basis_kind: str = "frozen_fourier",
    mapper_kind: str = "identity",
    qk_coupling: str = "shared",
    head_coupling: str = "per_head_independent",
    conditioning: str = "none",
    scalars: list[str] | None = None,
    amplitude_init: float = 0.1,
    scale_init: float = 1.0,
    learn_amplitude: bool = True,
    learn_phase: bool = True,
) -> dict:
    scalars = list(scalars or [])
    basis_dim = 32 if head_coupling == "per_head_joint" else 8
    return normalize_position_config_v2(
        "qk",
        {
            "enabled": True,
            "application": application,
            "geometry": geometry,
            "input": {
                "kind": basis_kind,
                "basis_dim": basis_dim,
                "theta": None,
                "scalars": scalars,
            },
            "mapper": {
                "kind": mapper_kind,
                "residual": mapper_kind
                in {"low_rank", "bottleneck_mlp", "mlp"},
                "rank": 4,
                "hidden_dim": 12,
            },
            "output": {
                "amplitude_init": amplitude_init,
                "scale_init": scale_init,
                "learn_amplitude": learn_amplitude,
                "learn_phase": learn_phase,
            },
            "conditioning": {
                "kind": conditioning,
                "hidden_dim": 12,
            },
            "qk_coupling": qk_coupling,
            "head_coupling": head_coupling,
        },
        model_dim=32,
        heads=4,
        rope_theta=10_000.0,
    )


def logit_config(
    kind: str,
    *,
    position_mode: str = "relative_only",
    pair_rank: int = 16,
) -> dict:
    return normalize_position_config_v2(
        "logit_bias",
        {
            "enabled": True,
            "application": "logit_bias",
            "geometry": "scalar_curve",
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
            "conditioning": {
                "kind": kind,
                "num_profiles": 4,
                "router_hidden_dim": 8,
                "num_frequencies": 3,
                "gate_init": 0.0,
                "pair_rank": pair_rank,
                "position_mode": position_mode,
            },
            "head_coupling": "per_head_independent",
        },
        model_dim=32,
        heads=4,
        rope_theta=10_000.0,
    )


class PositionBasisTest(unittest.TestCase):
    def test_learned_bases_match_frozen_at_initialization(self):
        expected = interleaved_fourier_basis(16, 8, 10_000.0)
        for kind in (
            "learned_temperature_fourier",
            "learned_frequency_fourier",
        ):
            with self.subTest(kind=kind):
                basis = build_position_basis(
                    kind=kind,
                    extent=16,
                    basis_dim=8,
                    theta=10_000.0,
                )
                torch.testing.assert_close(basis(16), expected)
                basis(16).sum().backward()
                parameter = next(basis.parameters())
                self.assertIsNotNone(parameter.grad)

    def test_scalar_features_extend_mapper_input(self):
        basis = build_position_basis(
            kind="frozen_fourier",
            extent=16,
            basis_dim=8,
            theta=10_000.0,
            scalars=["position", "normalized_position", "log_position"],
        )
        output = basis(5)
        self.assertEqual(output.shape, (5, 11))
        self.assertEqual(output[4, 8].item(), 4.0)
        self.assertAlmostEqual(output[4, 9].item(), 4.0 / 15.0)

    def test_scalar_normalization_can_follow_training_not_model_extent(self):
        basis = build_position_basis(
            kind="frozen_fourier",
            extent=16,
            basis_dim=8,
            theta=10_000.0,
            scalars=["normalized_position", "log_position"],
            normalization_extent=8,
        )
        output = basis(16)
        self.assertAlmostEqual(output[7, 8].item(), 1.0)
        self.assertGreater(output[15, 8].item(), 2.0)
        self.assertGreater(output[15, 9].item(), 1.0)

    def test_decoupled_basis_width_with_linear_mapper(self):
        config = qk_config(
            "additive",
            "free",
            mapper_kind="linear",
            scalars=["normalized_position"],
        )
        channel = build_qk_position_channel(
            config,
            heads=4,
            head_dim=8,
            model_dim=32,
            extent=16,
            rope_theta=10_000.0,
        )
        self.assertEqual(channel(7).q.shape, (4, 7, 8))

    def test_non_residual_bottleneck_has_live_initial_output(self):
        mapper = FeatureMapper(
            kind="bottleneck_mlp",
            groups=2,
            input_dim=8,
            output_dim=6,
            residual=False,
            rank=4,
            hidden_dim=12,
        )
        output = mapper(torch.randn(2, 5, 8))
        self.assertGreater(torch.count_nonzero(output).item(), 0)


class GeometryAndContentTest(unittest.TestCase):
    def _channel(self, config: dict):
        return build_qk_position_channel(
            config,
            heads=4,
            head_dim=8,
            model_dim=32,
            extent=16,
            rope_theta=10_000.0,
        )

    def test_canonical_amplitude_phase_addrope(self):
        config = qk_config(
            "additive",
            "amplitude_phase",
            qk_coupling="shared_trunk_separate_readouts",
            amplitude_init=0.125,
        )
        output = self._channel(config)(9)
        self.assertEqual(output.q.shape, (4, 9, 8))
        torch.testing.assert_close(output.q, output.k)
        q_pairs = torch.stack(
            (output.q[..., :4], output.q[..., 4:]),
            dim=-1,
        )
        amplitude = q_pairs.pow(2).sum(dim=-1).sqrt()
        torch.testing.assert_close(
            amplitude,
            torch.full_like(amplitude, 0.125),
        )

    def test_addrope_components_can_be_isolated(self):
        configs = {
            "fixed": qk_config(
                "additive",
                "amplitude_phase",
                amplitude_init=0.125,
                learn_amplitude=False,
                learn_phase=False,
            ),
            "amplitude": qk_config(
                "additive",
                "amplitude_phase",
                amplitude_init=0.125,
                learn_amplitude=True,
                learn_phase=False,
            ),
            "phase": qk_config(
                "additive",
                "amplitude_phase",
                amplitude_init=0.125,
                learn_amplitude=False,
                learn_phase=True,
            ),
            "combined": qk_config(
                "additive",
                "amplitude_phase",
                amplitude_init=0.125,
            ),
        }
        channels = {
            name: self._channel(config)
            for name, config in configs.items()
        }
        fixed = channels["fixed"]
        self.assertIsNone(fixed.pipeline)
        self.assertEqual(sum(p.numel() for p in fixed.parameters()), 0)
        self.assertEqual(fixed.summarize(9)["qk_frequency_diff_rms"], 0.0)
        self.assertIsNotNone(channels["amplitude"].amplitude_head)
        self.assertIsNone(channels["amplitude"].phase_head)
        self.assertIsNone(channels["phase"].amplitude_head)
        self.assertIsNotNone(channels["phase"].phase_head)
        self.assertIsNotNone(channels["combined"].amplitude_head)
        self.assertIsNotNone(channels["combined"].phase_head)

        for channel in channels.values():
            output = channel(9)
            torch.testing.assert_close(output.q, fixed(9).q)
            torch.testing.assert_close(output.q, output.k)

    def test_projected_phase_and_scaled_phase_init(self):
        projected = self._channel(
            qk_config("rotary", "projected_phase")
        )(9)
        self.assertEqual(projected.q.shape, (4, 9, 4))
        self.assertEqual(torch.count_nonzero(projected.q).item(), 0)

        scaled = self._channel(
            qk_config("rotary", "scaled_phase", scale_init=1.0)
        )(9)
        self.assertEqual(torch.count_nonzero(scaled.q).item(), 0)
        torch.testing.assert_close(
            scaled.q_scale,
            torch.ones_like(scaled.q_scale),
        )

    def test_local_conditioners_are_initially_controlled(self):
        base_config = qk_config(
            "additive",
            "free",
            mapper_kind="linear",
            qk_coupling="shared_trunk_separate_readouts",
        )
        for conditioning in ("local_residual", "content_gate"):
            with self.subTest(conditioning=conditioning):
                conditioned_config = qk_config(
                    "additive",
                    "free",
                    mapper_kind="linear",
                    qk_coupling="shared_trunk_separate_readouts",
                    conditioning=conditioning,
                )
                torch.manual_seed(3)
                base = self._channel(base_config)
                torch.manual_seed(3)
                conditioned = self._channel(conditioned_config)
                content_q = torch.randn(2, 4, 8, 8)
                content_k = torch.randn(2, 4, 8, 8)
                expected = base(8)
                actual = conditioned(
                    8,
                    q_content=content_q,
                    k_content=content_k,
                )
                torch.testing.assert_close(
                    actual.q,
                    expected.q[None].expand(2, -1, -1, -1),
                )
                loss = actual.q.square().sum() - actual.k.square().sum()
                loss.backward()
                conditioner = conditioned.q_conditioner
                self.assertTrue(
                    any(
                        parameter.grad is not None
                        for parameter in conditioner.parameters()
                    )
                )

    def test_local_conditioners_remain_bounded_at_extreme_logits(self):
        content = torch.zeros(2, 1, 3, 4)
        base = torch.ones(1, 3, 2)

        content_gate = GroupedContentConditioner(
            kind="content_gate",
            groups=1,
            content_dim=4,
            output_dim=2,
            hidden_dim=4,
            gate_init=0.0,
        )
        with torch.no_grad():
            content_gate.gate_bias.fill_(1000.0)
        gated = content_gate(base, content)
        self.assertTrue(torch.all(gated >= 0.0))
        self.assertTrue(torch.all(gated <= 2.0))

        local_residual = GroupedContentConditioner(
            kind="local_residual",
            groups=1,
            content_dim=4,
            output_dim=2,
            hidden_dim=4,
            gate_init=0.0,
        )
        with torch.no_grad():
            local_residual.up_bias.fill_(1000.0)
        corrected = local_residual(base, content)
        self.assertTrue(torch.all(corrected >= base))
        self.assertTrue(torch.all(corrected <= base + 1.0))

    def test_amplitude_phase_conditioning_preserves_latent_synthesis(self):
        config = qk_config(
            "additive",
            "amplitude_phase",
            qk_coupling="shared_trunk_separate_readouts",
            conditioning="local_residual",
        )
        channel = self._channel(config)
        content_q = torch.randn(2, 4, 8, 8)
        content_k = torch.randn(2, 4, 8, 8)
        output = channel(
            8,
            q_content=content_q,
            k_content=content_k,
        )
        self.assertEqual(output.q.shape, (2, 4, 8, 8))
        # Conditioning modules operate on pair amplitudes/phases, not on the
        # synthesized x/y coordinates.
        self.assertIsNotNone(channel.q_amplitude_conditioner)
        self.assertEqual(channel.q_conditioner.output_dim, 4)

    def test_full_model_scaled_rotary_matches_rope_at_init(self):
        common = dict(
            dim=32,
            depth=1,
            heads=4,
            ff_mult=2,
            vocab_size=64,
            max_seq_len=16,
            logit_bias_config={"enabled": False},
        )
        torch.manual_seed(7)
        baseline = Transformer(**common, qk_config={"enabled": False}).eval()
        torch.manual_seed(7)
        scaled = Transformer(
            **common,
            qk_config=qk_config("rotary", "scaled_phase"),
        ).eval()
        scaled.load_state_dict(baseline.state_dict(), strict=False)
        ids = torch.randint(0, 64, (2, 8))
        torch.testing.assert_close(baseline(ids), scaled(ids))

    def test_content_conditioned_full_model_keeps_four_dimensions(self):
        config = qk_config(
            "additive",
            "free",
            mapper_kind="linear",
            qk_coupling="shared_trunk_separate_readouts",
            conditioning="content_gate",
        )
        model = Transformer(
            dim=32,
            depth=1,
            heads=4,
            ff_mult=2,
            vocab_size=64,
            max_seq_len=16,
            qk_config=config,
            logit_bias_config={"enabled": False},
        )
        attention = model.blocks[0].attn
        captured = {}

        def hook(_module, _inputs, output):
            captured["shape"] = output.shape

        handle = attention.register_forward_hook(hook)
        output = model(torch.randint(0, 64, (2, 8)))
        handle.remove()
        self.assertEqual(captured["shape"], (2, 8, 32))
        self.assertEqual(output.shape, (2, 8, 64))


class ResidualAndWriteTest(unittest.TestCase):
    def _common(self):
        return dict(
            dim=32,
            depth=2,
            heads=4,
            ff_mult=2,
            vocab_size=64,
            max_seq_len=16,
            qk_config={"enabled": False},
            logit_bias_config={"enabled": False},
        )

    def test_zero_gated_residual_stream_is_exact_null(self):
        config = normalize_residual_stream_config(
            {
                "enabled": True,
                "placement": "both",
                "source": "position_basis",
                "gate_init": 0.0,
            },
            model_dim=32,
            heads=4,
            rope_theta=10_000.0,
        )
        torch.manual_seed(11)
        baseline = Transformer(**self._common()).eval()
        torch.manual_seed(11)
        candidate = Transformer(
            **self._common(),
            residual_stream_config=config,
        ).eval()
        candidate.load_state_dict(baseline.state_dict(), strict=False)
        ids = torch.randint(0, 64, (2, 8))
        torch.testing.assert_close(baseline(ids), candidate(ids))

    def test_learned_absolute_and_functional_streams(self):
        for source in ("learned_absolute", "position_basis"):
            with self.subTest(source=source):
                config = normalize_residual_stream_config(
                    {
                        "enabled": True,
                        "placement": "per_layer",
                        "source": source,
                        "gate_init": 0.1,
                        "layer_shared": True,
                    },
                    model_dim=32,
                    heads=4,
                    rope_theta=10_000.0,
                )
                model = Transformer(
                    **self._common(),
                    residual_stream_config=config,
                )
                output = model(torch.randint(0, 64, (2, 8)))
                self.assertEqual(output.shape, (2, 8, 64))

    def test_rope_can_be_disabled_for_residual_only_and_no_pe_controls(self):
        ids = torch.randint(0, 64, (2, 8))
        no_position = Transformer(
            **self._common(),
            use_rope=False,
        ).eval()
        self.assertFalse(no_position.blocks[0].attn.multiplicative_rope)
        self.assertEqual(no_position(ids).shape, (2, 8, 64))

        residual_config = normalize_residual_stream_config(
            {
                "enabled": True,
                "placement": "input",
                "source": "position_basis",
                "gate_init": 1.0,
            },
            model_dim=32,
            heads=4,
            rope_theta=10_000.0,
        )
        residual_only = Transformer(
            **self._common(),
            use_rope=False,
            residual_stream_config=residual_config,
        ).eval()
        self.assertFalse(residual_only.blocks[0].attn.multiplicative_rope)
        self.assertEqual(residual_only(ids).shape, (2, 8, 64))

    def test_rope_disable_rejects_rotary_qk_channel(self):
        common = self._common()
        common["qk_config"] = qk_config("rotary", "phase")
        with self.assertRaisesRegex(ValueError, "rotary Q/K"):
            Transformer(
                **common,
                use_rope=False,
            )

    def test_attention_write_modes_and_zero_gate(self):
        ids = torch.randint(0, 64, (2, 8))
        for mode in ("key_position", "relative_offset"):
            with self.subTest(mode=mode):
                config = normalize_attention_write_config(
                    {
                        "enabled": True,
                        "mode": mode,
                        "gate_init": 0.0,
                    },
                    model_dim=32,
                    heads=4,
                    rope_theta=10_000.0,
                )
                torch.manual_seed(13)
                baseline = Transformer(**self._common()).eval()
                torch.manual_seed(13)
                candidate = Transformer(
                    **self._common(),
                    attention_write_config=config,
                ).eval()
                candidate.load_state_dict(baseline.state_dict(), strict=False)
                torch.testing.assert_close(
                    baseline(ids),
                    candidate(ids),
                    atol=1e-6,
                    rtol=1e-5,
                )


class InklingAndConfigTest(unittest.TestCase):
    def _load_payload(self, payload: dict):
        with tempfile.NamedTemporaryFile("w", suffix=".json") as file:
            json.dump(payload, file)
            file.flush()
            return load_config(
                Namespace(
                    override_json=file.name,
                    pos_variant=None,
                    attn_impl=None,
                    max_train_steps=None,
                    dry_run=False,
                    print_model=False,
                )
            )

    def test_inkling_banks_are_zero_effect_with_live_routing(self):
        for kind in ("inkling_table", "inkling_cosnet"):
            with self.subTest(kind=kind):
                model = Transformer(
                    dim=32,
                    depth=1,
                    heads=4,
                    ff_mult=2,
                    vocab_size=64,
                    max_seq_len=16,
                    qk_config={"enabled": False},
                    logit_bias_config=logit_config(kind),
                    attn_impl="flex",
                )
                channel = model.blocks[0].attn.logit_bias
                query = torch.randn(2, 4, 8, 8)
                bias = channel(query=query)
                self.assertEqual(bias.shape, (2, 4, 8, 16))
                self.assertEqual(torch.count_nonzero(bias).item(), 0)
                summary = channel.routing_summary()
                self.assertIn("routing_entropy_mean", summary)
                self.assertGreater(summary["routing_entropy_mean"], 0)

    def test_pairwise_low_rank_modes_have_exact_zero_effect_gate(self):
        for mode in (
            "relative_only",
            "query_absolute",
            "full_absolute",
        ):
            with self.subTest(mode=mode):
                model = Transformer(
                    dim=32,
                    depth=1,
                    heads=4,
                    ff_mult=2,
                    vocab_size=64,
                    max_seq_len=16,
                    qk_config={"enabled": False},
                    logit_bias_config=logit_config(
                        "pairwise_low_rank",
                        position_mode=mode,
                        pair_rank=4,
                    ),
                    attn_impl="flex",
                )
                channel = model.blocks[0].attn.logit_bias
                query = torch.randn(2, 4, 8, 8)
                key = torch.randn(2, 4, 8, 8)
                base, factors = channel.prepare(query=query, key=key)
                self.assertEqual(torch.count_nonzero(base).item(), 0)
                query_factor, key_factor, distance_factor, gate = factors
                self.assertEqual(query_factor.shape, (2, 4, 8, 4))
                self.assertEqual(key_factor.shape, (2, 4, 8, 4))
                self.assertEqual(distance_factor.shape, (4, 16, 4))
                self.assertEqual(torch.count_nonzero(gate).item(), 0)
                for factor in (
                    query_factor,
                    key_factor,
                    distance_factor,
                ):
                    torch.testing.assert_close(
                        factor.float().square().mean(dim=-1),
                        torch.ones_like(factor[..., 0].float()),
                        atol=2e-4,
                        rtol=2e-4,
                    )

    def test_pairwise_relative_mode_is_translation_invariant(self):
        channels = {}
        for mode in ("relative_only", "query_absolute"):
            model = Transformer(
                dim=32,
                depth=1,
                heads=4,
                ff_mult=2,
                vocab_size=64,
                max_seq_len=16,
                qk_config={"enabled": False},
                logit_bias_config=logit_config(
                    "pairwise_low_rank",
                    position_mode=mode,
                    pair_rank=4,
                ),
                attn_impl="flex",
            )
            channels[mode] = model.blocks[0].attn.logit_bias

        content = torch.ones(1, 4, 8, 8)
        relative_factors = channels["relative_only"].prepare(
            query=content,
            key=content,
        )[1]
        absolute_factors = channels["query_absolute"].prepare(
            query=content,
            key=content,
        )[1]

        def interaction(factors, query_idx, key_idx):
            query_factor, key_factor, distance_factor, _ = factors
            distance = query_idx - key_idx
            return (
                query_factor[0, :, query_idx]
                * key_factor[0, :, key_idx]
                * distance_factor[:, distance]
            ).sum(dim=-1)

        torch.testing.assert_close(
            interaction(relative_factors, 2, 1),
            interaction(relative_factors, 3, 2),
        )
        self.assertFalse(
            torch.allclose(
                interaction(absolute_factors, 2, 1),
                interaction(absolute_factors, 3, 2),
            )
        )

    def test_pairwise_gate_receives_gradient_at_zero_initialization(self):
        model = Transformer(
            dim=32,
            depth=1,
            heads=4,
            ff_mult=2,
            vocab_size=64,
            max_seq_len=16,
            qk_config={"enabled": False},
            logit_bias_config=logit_config(
                "pairwise_low_rank",
                pair_rank=4,
            ),
            attn_impl="flex",
        )
        channel = model.blocks[0].attn.logit_bias
        query = torch.randn(2, 4, 8, 8)
        key = torch.randn(2, 4, 8, 8)
        _, factors = channel.prepare(query=query, key=key)
        query_factor, key_factor, distance_factor, gate = factors
        raw = (
            query_factor[:, :, :, None, :]
            * key_factor[:, :, None, :, :]
            * distance_factor[
                :,
                (
                    torch.arange(8)[:, None]
                    - torch.arange(8)[None, :]
                ).clamp_min(0),
            ][None]
        ).sum(dim=-1)
        (gate[None, :, None, None] * raw.detach().square()).sum().backward()
        self.assertGreater(
            channel.pairwise.gate.grad.abs().sum().item(),
            0.0,
        )

    def test_inkling_presets_resolve_and_force_flex(self):
        for preset in ("inkling_table", "inkling_cosnet"):
            args = Namespace(
                override_json=None,
                pos_variant=preset,
                attn_impl=None,
                max_train_steps=None,
                dry_run=False,
                print_model=False,
            )
            config = load_config(args)
            self.assertEqual(
                config.logit_bias["conditioning"]["kind"],
                preset,
            )
            self.assertEqual(config.attn_impl, "flex")
            self.assertEqual(config.position_source_schema, 2)

    def test_enabled_only_override_can_disable_v2_preset(self):
        config = self._load_payload(
            {
                "pos_variant": "inkling_table",
                "logit_bias": {"enabled": False},
            }
        )
        self.assertFalse(config.logit_bias["enabled"])
        self.assertEqual(config.attn_impl, "sdpa")

    def test_no_explicit_position_config_disables_rope(self):
        config = self._load_payload({"use_rope": False})
        self.assertFalse(config.use_rope)
        self.assertEqual(config.pos_variant, "none")
        self.assertEqual(config.position_source_schema, 2)

    def test_training_model_and_evaluation_lengths_are_independent(self):
        config = self._load_payload(
            {
                "training_length": 16,
                "model_position_extent": 64,
                "evaluation_lengths": [32, 64],
                "scalar_normalization_extent": 16,
            }
        )
        self.assertEqual(config.block_size, 16)
        self.assertEqual(config.training_length, 16)
        self.assertEqual(config.model_position_extent, 64)
        self.assertEqual(config.evaluation_lengths, [16, 32, 64])
        self.assertEqual(config.scalar_normalization_extent, 16)

        with self.assertRaisesRegex(ValueError, "must cover"):
            self._load_payload(
                {
                    "training_length": 16,
                    "model_position_extent": 32,
                    "evaluation_lengths": [64],
                }
            )

    def test_validation_tokens_can_be_rechunked_to_longer_contexts(self):
        source = [
            {"input_ids": list(range(0, 4))},
            {"input_ids": list(range(4, 8))},
            {"input_ids": list(range(8, 12))},
        ]
        rechunked = RechunkedTokenDataset(source, 6)
        self.assertEqual(len(rechunked), 2)
        self.assertEqual(rechunked[0]["input_ids"], list(range(0, 6)))
        self.assertEqual(rechunked[1]["input_ids"], list(range(6, 12)))

    def test_native_v2_auto_tags_hash_all_behavior_fields(self):
        base = {
            "hidden_size": 32,
            "n_head": 4,
            "depth": 1,
            "block_size": 16,
            "qk": {
                "enabled": True,
                "application": "additive",
                "geometry": "amplitude_phase",
                "output": {"amplitude_init": 0.1},
            },
        }
        first = self._load_payload(base)
        changed = json.loads(json.dumps(base))
        changed["qk"]["output"]["amplitude_init"] = 0.2
        second = self._load_payload(changed)
        self.assertNotEqual(first.run_name, second.run_name)
        self.assertRegex(first.run_name, r"-c[0-9a-f]{10}-e16-h32d1$")

    def test_flex_rejects_fullgraph_outer_compile(self):
        with self.assertRaisesRegex(ValueError, "compile_fullgraph"):
            self._load_payload(
                {
                    "pos_variant": "inkling_table",
                    "compile": True,
                    "compile_fullgraph": True,
                }
            )

    def test_aux_configs_round_trip_through_training_loader(self):
        payload = {
            "hidden_size": 32,
            "n_head": 4,
            "depth": 1,
            "block_size": 16,
            "residual_stream": {
                "enabled": True,
                "placement": "per_layer",
                "source": "learned_absolute",
                "gate_init": 0.0,
            },
            "attention_write": {
                "enabled": True,
                "mode": "relative_offset",
                "gate_init": 0.0,
            },
        }
        with tempfile.NamedTemporaryFile("w", suffix=".json") as file:
            json.dump(payload, file)
            file.flush()
            config = load_config(
                Namespace(
                    override_json=file.name,
                    pos_variant=None,
                    attn_impl=None,
                    max_train_steps=None,
                    dry_run=False,
                    print_model=False,
                )
            )
        self.assertTrue(config.residual_stream["enabled"])
        self.assertTrue(config.attention_write["enabled"])
        self.assertEqual(config.position_source_schema, 2)

    def test_parameter_matched_ffn_recommendation(self):
        result = suggest_matched_baselines(
            {
                "hidden_size": 32,
                "depth": 2,
                "ff_mult": 2,
                "ff_hidden_dim": None,
            },
            position_params=10_000,
            align_multiple=8,
        )
        self.assertGreater(result["matched_ff_hidden_dim"], 64)
        self.assertGreaterEqual(result["matched_ff_added_params"], 10_000)

    def test_every_historical_and_phase2_config_loads(self):
        paths = sorted(
            Path(__file__).resolve().parent.joinpath("sweep_configs").glob(
                "**/*.json"
            )
        )
        self.assertGreaterEqual(len(paths), 24)
        for path in paths:
            with self.subTest(path=path.name):
                config = load_config(
                    Namespace(
                        override_json=str(path),
                        pos_variant=None,
                        attn_impl=None,
                        max_train_steps=None,
                        dry_run=False,
                        print_model=False,
                    )
                )
                self.assertEqual(config.position_schema_version, 2)


class PositionalApiCompatibilityTest(unittest.TestCase):
    def test_existing_optional_argument_order_is_unchanged(self):
        attention = Attention(
            32,
            4,
            True,
            True,
            10_000.0,
            16,
            True,
            None,
            {"enabled": False},
            {"enabled": False},
            "sdpa",
        )
        self.assertEqual(attention.attn_impl, "sdpa")

        block = TransformerBlock(
            32,
            4,
            64,
            True,
            True,
            10_000.0,
            16,
            True,
            None,
            {"enabled": False},
            {"enabled": False},
            "sdpa",
        )
        self.assertEqual(block.attn.attn_impl, "sdpa")

        model = Transformer(
            32,
            1,
            4,
            2,
            64,
            16,
            False,
            True,
            10_000.0,
            True,
            None,
            {"enabled": False},
            {"enabled": False},
            "sdpa",
        )
        self.assertEqual(model.blocks[0].attn.attn_impl, "sdpa")


if __name__ == "__main__":
    unittest.main()
