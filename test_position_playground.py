"""Coverage for the fully configurable position experimentation playground."""

from __future__ import annotations

import json
import os
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
    exp_with_identity_grad,
    interleaved_fourier_basis,
    normalize_attention_write_config,
    normalize_position_config_v2,
    normalize_residual_stream_config,
)
from train_gpt import RechunkedTokenDataset, load_config, make_model
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
    conditioning_source: str = "qk",
    conditioning_target: str = "both",
    conditioning_coupling: str = "shared_trunk_separate_readouts",
    conditioning_static_complement: bool = False,
    phase_bound: float = 0.25,
    conditioning_input_mode: str = "content",
    conditioning_network: str = "linear",
    conditioning_components: str = "phase",
    conditioning_head_coupling: str = "per_head_independent",
    scalars: list[str] | None = None,
    amplitude_init: float = 0.1,
    amplitude_parameterization: str = "signed",
    parameter_source: str = "mapped",
    scale_init: float = 1.0,
    learn_amplitude: bool = True,
    learn_phase: bool = True,
    mapper_residual: bool | None = None,
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
                "residual": (
                    mapper_kind in {"low_rank", "bottleneck_mlp", "mlp"}
                    if mapper_residual is None
                    else mapper_residual
                ),
                "rank": 4,
                "hidden_dim": 12,
            },
            "output": {
                "parameter_source": parameter_source,
                "amplitude_init": amplitude_init,
                "amplitude_parameterization": amplitude_parameterization,
                "scale_init": scale_init,
                "learn_amplitude": learn_amplitude,
                "learn_phase": learn_phase,
            },
            "conditioning": {
                "kind": conditioning,
                "source": conditioning_source,
                "target": conditioning_target,
                "coupling": conditioning_coupling,
                "static_complement": conditioning_static_complement,
                "phase_bound": phase_bound,
                "hidden_dim": 12,
                "input_mode": conditioning_input_mode,
                "network": conditioning_network,
                "components": conditioning_components,
                "head_coupling": conditioning_head_coupling,
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
                "source": "qk",
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
    def test_exp_parameterization_uses_exp_forward_identity_backward(self):
        value = torch.tensor([-3.0, 0.0, 3.0], requires_grad=True)
        output = exp_with_identity_grad(value)
        torch.testing.assert_close(output, value.detach().exp())
        output.sum().backward()
        torch.testing.assert_close(value.grad, torch.ones_like(value))

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

    def test_linear_residual_mapper_honors_residual_flag(self):
        mapper = FeatureMapper(
            kind="linear",
            groups=1,
            input_dim=8,
            output_dim=8,
            residual=True,
            rank=4,
            hidden_dim=12,
        )
        with torch.no_grad():
            mapper.weight.zero_()
            mapper.bias.zero_()
        features = torch.randn(1, 5, 8)
        torch.testing.assert_close(mapper(features), features)


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

    def test_carrier_hypernetwork_preserves_addrope_anchor_and_gets_gradient(self):
        config = qk_config(
            "additive",
            "amplitude_phase",
            conditioning="carrier_hypernetwork",
            conditioning_source="dedicated",
            conditioning_coupling="shared_trunk_separate_readouts",
            conditioning_input_mode="content_position",
            conditioning_network="silu_mlp",
            conditioning_components="log_gain_phase",
            learn_amplitude=False,
            learn_phase=False,
            amplitude_init=0.2,
        )
        channel = self._channel(config)
        self.assertIsNone(channel.pipeline)
        content = torch.randn(2, 4, 9, 8)
        output = channel(9, q_content=content, k_content=content)
        expected = (
            0.2
            * torch.cat(
                (channel.base_cos[:9], channel.base_sin[:9]),
                dim=-1,
            )[None]
            .expand(4, -1, -1)
        )
        torch.testing.assert_close(output.q, expected[None].expand(2, -1, -1, -1))
        torch.testing.assert_close(output.k, output.q)
        self.assertEqual(torch.count_nonzero(output.q_log_gain_delta).item(), 0)
        self.assertEqual(torch.count_nonzero(output.q_hyper_phase_delta).item(), 0)
        summary = channel.summarize(
            9,
            q_content=content,
            k_content=content,
        )
        self.assertEqual(summary["hyper_log_gain_delta_q/rms"], 0.0)
        self.assertEqual(summary["hyper_phase_delta_k/p95_abs"], 0.0)
        self.assertEqual(summary["hyper_effective_gain_q/max"], 1.0)

        (output.q.sum() + output.k.sum()).backward()
        q_readout = channel.carrier_hypernetwork.q_readout
        k_readout = channel.carrier_hypernetwork.k_readout
        self.assertGreater(q_readout.weight.grad.abs().sum().item(), 0)
        self.assertGreater(k_readout.weight.grad.abs().sum().item(), 0)

    def test_carrier_hypernetwork_rotary_phase_and_gain_are_exact_nulls(self):
        config = qk_config(
            "rotary",
            "phase",
            conditioning="carrier_hypernetwork",
            conditioning_source="dedicated",
            conditioning_coupling="shared",
            conditioning_input_mode="content",
            conditioning_network="linear",
            conditioning_components="log_gain_phase",
            conditioning_head_coupling="shared_head",
            learn_phase=False,
        )
        channel = self._channel(config)
        self.assertIsNone(channel.pipeline)
        content = torch.randn(2, 4, 9, 8)
        output = channel(9, q_content=content, k_content=content)
        torch.testing.assert_close(output.q, torch.zeros_like(output.q))
        torch.testing.assert_close(output.k, output.q)
        torch.testing.assert_close(output.q_scale, torch.ones_like(output.q_scale))
        torch.testing.assert_close(output.k_scale, output.q_scale)
        self.assertEqual(output.q.shape, (2, 4, 9, 4))

    def test_carrier_hypernetwork_axes_and_asymmetric_targets(self):
        for input_mode in ("content", "position", "content_position"):
            for network in ("linear", "silu_mlp", "swiglu_mlp"):
                with self.subTest(input_mode=input_mode, network=network):
                    config = qk_config(
                        "additive",
                        "amplitude_phase",
                        conditioning="carrier_hypernetwork",
                        conditioning_source="dedicated",
                        conditioning_target="q",
                        conditioning_coupling="separate",
                        conditioning_input_mode=input_mode,
                        conditioning_network=network,
                        conditioning_components="phase",
                        learn_amplitude=False,
                        learn_phase=False,
                    )
                    channel = self._channel(config)
                    content = (
                        None
                        if input_mode == "position"
                        else torch.randn(2, 4, 7, 8)
                    )
                    output = channel(
                        7,
                        q_content=content,
                        k_content=content,
                    )
                    self.assertEqual(output.q.shape[-3:], (4, 7, 8))
                    self.assertEqual(output.k.shape[-3:], (4, 7, 8))
                    self.assertEqual(
                        torch.count_nonzero(output.k_hyper_phase_delta).item(),
                        0,
                    )

    def test_shared_carrier_hypernetwork_requires_shared_content(self):
        config = qk_config(
            "rotary",
            "phase",
            conditioning="carrier_hypernetwork",
            conditioning_source="dedicated",
            conditioning_coupling="shared",
            conditioning_input_mode="content",
            learn_phase=False,
        )
        with self.assertRaisesRegex(ValueError, "shared Q/K"):
            Attention(
                32,
                4,
                qk_config=config,
                position_content_coupling="separate",
            )
        attention = Attention(
            32,
            4,
            qk_config=config,
            position_content_coupling="shared",
            qk_norm_mode="method_aware_rms",
            attn_impl="sdpa",
        )
        output = attention(torch.randn(2, 7, 32))
        self.assertEqual(output.shape, (2, 7, 32))

    def test_position_only_carrier_hypernetwork_needs_no_content_projector(self):
        config = qk_config(
            "rotary",
            "phase",
            conditioning="carrier_hypernetwork",
            conditioning_source="dedicated",
            conditioning_coupling="shared_trunk_separate_readouts",
            conditioning_input_mode="position",
            conditioning_network="swiglu_mlp",
            learn_phase=False,
        )
        attention = Attention(
            32,
            4,
            qk_config=config,
            qk_norm_mode="method_aware_rms",
            attn_impl="sdpa",
        )
        self.assertIsNone(attention.position_content)
        x = torch.randn(2, 7, 32)
        self.assertEqual(attention(x).shape, x.shape)
        summary = attention.qk_position_summary_from_input(x)
        self.assertEqual(summary["hyper_phase_delta_q/rms"], 0.0)

    def test_additive_carrier_hypernetwork_is_composed_before_qk_rms(self):
        config = qk_config(
            "additive",
            "amplitude_phase",
            conditioning="carrier_hypernetwork",
            conditioning_source="dedicated",
            conditioning_coupling="shared",
            conditioning_input_mode="content",
            learn_amplitude=False,
            learn_phase=False,
        )
        attention = Attention(
            32,
            4,
            qk_config=config,
            position_content_coupling="shared",
            qk_norm_mode="method_aware_rms",
            attn_impl="sdpa",
        )
        captured = {}

        def capture_q_input(_module, args):
            captured["q_input"] = args[0]

        handle = attention.q_norm.register_forward_pre_hook(capture_q_input)
        x = torch.randn(2, 7, 32)
        attention(x)
        handle.remove()
        projected = attention._split_heads(attention.to_q(x))
        content, _ = attention.position_content(x)
        addend = attention.qk_position(
            7,
            dtype=projected.dtype,
            q_content=content,
            k_content=content,
        ).q
        torch.testing.assert_close(captured["q_input"], projected + addend)

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

    def test_direct_addrope_uses_only_per_frequency_parameters(self):
        config = qk_config(
            "additive",
            "amplitude_phase",
            qk_coupling="shared_trunk_separate_readouts",
            parameter_source="direct",
            amplitude_parameterization="softplus",
            amplitude_init=0.3,
        )
        channel = self._channel(config)
        self.assertIsNone(channel.pipeline)
        self.assertIsNone(channel.q_amplitude_head)
        self.assertEqual(channel.q_direct_amplitude_raw.shape, (4, 4))
        self.assertEqual(channel.q_direct_phase.shape, (4, 4))

        output = channel(9)
        expected = 0.3 * torch.cat(
            (channel.base_cos[:9], channel.base_sin[:9]),
            dim=-1,
        )[None].expand(4, -1, -1)
        torch.testing.assert_close(output.q, expected)
        torch.testing.assert_close(output.k, expected)

        output.q.square().mean().backward()
        self.assertGreater(
            channel.q_direct_amplitude_raw.grad.abs().sum().item(),
            0,
        )
        self.assertIsNone(channel.k_direct_amplitude_raw.grad)

    def test_dynamic_addrope_replaces_static_mapper_at_exact_softplus_anchor(self):
        config = qk_config(
            "additive",
            "amplitude_phase",
            qk_coupling="shared_trunk_separate_readouts",
            conditioning="carrier_hypernetwork",
            conditioning_source="dedicated",
            conditioning_input_mode="content",
            conditioning_components="amplitude_phase",
            amplitude_parameterization="softplus",
            amplitude_init=0.3,
            learn_amplitude=False,
            learn_phase=False,
        )
        channel = self._channel(config)
        self.assertIsNone(channel.pipeline)
        self.assertIsNone(channel.q_amplitude_head)
        self.assertIsNone(channel.q_direct_amplitude_raw)
        content = torch.randn(2, 4, 9, 8)
        output = channel(
            9,
            q_content=content,
            k_content=content,
        )
        expected = 0.3 * torch.cat(
            (channel.base_cos[:9], channel.base_sin[:9]),
            dim=-1,
        )[None, None].expand(2, 4, -1, -1)
        torch.testing.assert_close(output.q, expected)
        torch.testing.assert_close(output.k, expected)
        self.assertIsNone(output.q_log_gain_delta)
        torch.testing.assert_close(
            output.q_amplitude_delta,
            torch.zeros_like(output.q_amplitude_delta),
        )

        output.q.square().mean().backward()
        self.assertGreater(
            channel.carrier_hypernetwork.q_readout.weight.grad.abs().sum().item(),
            0,
        )

    def test_unit_anchor_hyperaddrope_has_raw_scale_and_phase_gradients(self):
        config = qk_config(
            "additive",
            "amplitude_phase",
            qk_coupling="shared_trunk_separate_readouts",
            conditioning="carrier_hypernetwork",
            conditioning_source="dedicated",
            conditioning_input_mode="content",
            conditioning_components="amplitude_phase",
            amplitude_parameterization="signed",
            amplitude_init=1.0,
            learn_amplitude=False,
            learn_phase=False,
        )
        channel = self._channel(config)
        content = torch.randn(2, 4, 9, 8)
        output = channel(9, q_content=content, k_content=content)
        expected = torch.cat(
            (channel.base_cos[:9], channel.base_sin[:9]),
            dim=-1,
        )[None, None].expand(2, 4, -1, -1)
        torch.testing.assert_close(output.q, expected)
        torch.testing.assert_close(output.k, expected)

        target = torch.randn_like(output.q)
        (output.q * target).sum().backward()
        gradient = channel.carrier_hypernetwork.q_readout.weight.grad
        scale_gradient, phase_gradient = gradient.split(4, dim=-1)
        self.assertGreater(scale_gradient.abs().sum().item(), 0)
        self.assertGreater(phase_gradient.abs().sum().item(), 0)

    def test_unit_anchor_shared_content_keeps_separate_qk_readouts(self):
        config = qk_config(
            "additive",
            "amplitude_phase",
            qk_coupling="shared_trunk_separate_readouts",
            conditioning="carrier_hypernetwork",
            conditioning_source="dedicated",
            conditioning_input_mode="content",
            conditioning_components="amplitude_phase",
            amplitude_parameterization="signed",
            amplitude_init=1.0,
            learn_amplitude=False,
            learn_phase=False,
        )
        attention = Attention(
            32,
            4,
            qk_config=config,
            position_content_coupling="shared",
            qk_norm_mode="method_aware_rms",
            attn_impl="sdpa",
        )
        self.assertIs(
            attention.position_content.q_projection,
            attention.position_content.k_projection,
        )
        hyper = attention.qk_position.carrier_hypernetwork
        self.assertIsNot(hyper.q_readout, hyper.k_readout)
        output = attention(torch.randint(0, 1, (2, 7, 32)).float())
        self.assertEqual(output.shape, (2, 7, 32))

    def test_asymmetric_dynamic_addrope_uses_learned_static_complement(self):
        content = torch.randn(2, 4, 9, 8)
        for dynamic_target in ("q", "k"):
            with self.subTest(dynamic_target=dynamic_target):
                config = qk_config(
                    "additive",
                    "amplitude_phase",
                    qk_coupling="shared_trunk_separate_readouts",
                    conditioning="carrier_hypernetwork",
                    conditioning_source="dedicated",
                    conditioning_target=dynamic_target,
                    conditioning_static_complement=True,
                    conditioning_input_mode="content_position",
                    conditioning_network="silu_mlp",
                    conditioning_components="amplitude_phase",
                    amplitude_parameterization="signed",
                    amplitude_init=1.0,
                    parameter_source="direct",
                    learn_amplitude=False,
                    learn_phase=False,
                )
                channel = self._channel(config)
                static_target = "k" if dynamic_target == "q" else "q"
                self.assertIsNone(
                    getattr(channel, f"{dynamic_target}_direct_amplitude_raw")
                )
                self.assertIsNotNone(
                    getattr(channel, f"{static_target}_direct_amplitude_raw")
                )
                self.assertIsNone(
                    getattr(
                        channel.carrier_hypernetwork,
                        f"{static_target}_readout",
                    )
                )
                dynamic_readout = getattr(
                    channel.carrier_hypernetwork,
                    f"{dynamic_target}_readout",
                )

                output = channel(9, q_content=content, k_content=content)
                static_expected = torch.cat(
                    (channel.base_cos[:9], channel.base_sin[:9]),
                    dim=-1,
                )[None].expand(4, -1, -1)
                dynamic_expected = static_expected[None].expand(2, -1, -1, -1)
                torch.testing.assert_close(
                    getattr(output, dynamic_target),
                    dynamic_expected,
                )
                torch.testing.assert_close(
                    getattr(output, static_target),
                    static_expected,
                )
                metrics = channel.summarize(
                    9,
                    q_content=content,
                    k_content=content,
                )
                self.assertIn(
                    f"hyper_amplitude_delta_{dynamic_target}/rms",
                    metrics,
                )
                self.assertNotIn(
                    f"hyper_amplitude_delta_{static_target}/rms",
                    metrics,
                )

                loss = (
                    output.q * torch.randn_like(output.q)
                    + output.k * torch.randn_like(output.k)
                ).sum()
                loss.backward()
                self.assertGreater(
                    dynamic_readout.weight.grad.abs().sum().item(),
                    0,
                )
                self.assertGreater(
                    getattr(
                        channel,
                        f"{static_target}_direct_amplitude_raw",
                    ).grad.abs().sum().item(),
                    0,
                )
                self.assertGreater(
                    getattr(
                        channel,
                        f"{static_target}_direct_phase",
                    ).grad.abs().sum().item(),
                    0,
                )

    def test_hyperaddrope_accepts_separate_normalized_content_projections(self):
        config = qk_config(
            "additive",
            "amplitude_phase",
            qk_coupling="shared_trunk_separate_readouts",
            conditioning="carrier_hypernetwork",
            conditioning_source="dedicated",
            conditioning_input_mode="content_position",
            conditioning_network="silu_mlp",
            conditioning_components="amplitude_phase",
            amplitude_parameterization="signed",
            amplitude_init=1.0,
            parameter_source="direct",
            learn_amplitude=False,
            learn_phase=False,
        )
        attention = Attention(
            32,
            4,
            qk_config=config,
            position_content_coupling="separate",
            qk_norm_mode="method_aware_rms",
            attn_impl="sdpa",
        )
        self.assertIsNot(
            attention.position_content.q_projection,
            attention.position_content.k_projection,
        )
        q_content, k_content = attention.position_content(
            torch.randn(2, 9, 32)
        )
        self.assertFalse(torch.equal(q_content, k_content))
        torch.testing.assert_close(
            q_content.square().mean(dim=-1),
            torch.ones_like(q_content[..., 0]),
            atol=2e-5,
            rtol=2e-5,
        )
        torch.testing.assert_close(
            k_content.square().mean(dim=-1),
            torch.ones_like(k_content[..., 0]),
            atol=2e-5,
            rtol=2e-5,
        )

    def test_phase_only_hyperrope_starts_at_standard_rope_and_gets_gradient(self):
        config = qk_config(
            "rotary",
            "phase",
            qk_coupling="shared_trunk_separate_readouts",
            conditioning="carrier_hypernetwork",
            conditioning_source="dedicated",
            conditioning_input_mode="content_position",
            conditioning_network="silu_mlp",
            conditioning_components="phase",
            learn_phase=False,
        )
        channel = self._channel(config)
        content = torch.randn(2, 4, 9, 8)
        output = channel(9, q_content=content, k_content=content)
        torch.testing.assert_close(output.q, torch.zeros_like(output.q))
        torch.testing.assert_close(output.k, torch.zeros_like(output.k))
        target = torch.randn_like(output.q)
        (output.q * target).sum().backward()
        self.assertGreater(
            channel.carrier_hypernetwork.q_readout.weight.grad.abs().sum().item(),
            0,
        )

    def test_free_residual_additive_mapper_preserves_basis_skip(self):
        config = qk_config(
            "additive",
            "free",
            mapper_kind="linear",
            mapper_residual=True,
            qk_coupling="shared_trunk_separate_readouts",
        )
        channel = self._channel(config)
        with torch.no_grad():
            channel.pipeline.mapper.weight.zero_()
            channel.pipeline.mapper.bias.zero_()
        output = channel(9)
        basis = channel.pipeline.basis(9)
        expected = basis.unsqueeze(0).expand(4, -1, -1)
        torch.testing.assert_close(output.q, expected)
        torch.testing.assert_close(output.k, expected)

    def test_pair_normalized_additive_geometry_has_fixed_pair_radius(self):
        config = qk_config(
            "additive",
            "pair_normalized",
            mapper_kind="linear",
            qk_coupling="shared_trunk_separate_readouts",
            amplitude_init=0.3,
        )
        output = self._channel(config)(9)
        for values in (output.q, output.k):
            pair_radius = (
                values[..., :4].square() + values[..., 4:].square()
            ).sqrt()
            torch.testing.assert_close(
                pair_radius,
                torch.full_like(pair_radius, 0.3),
                atol=2e-5,
                rtol=2e-5,
            )

    def test_phase_rotation_conditioning_starts_as_exact_anchor(self):
        config = qk_config(
            "additive",
            "pair_normalized",
            mapper_kind="linear",
            qk_coupling="shared_trunk_separate_readouts",
            conditioning="phase_rotation",
            conditioning_source="residual",
            amplitude_init=0.3,
        )
        channel = self._channel(config)
        content = torch.randn(2, 4, 9, 32)
        output = channel(9, q_content=content, k_content=content)
        raw = channel.pipeline(9)
        expected_q = channel._normalize_additive_pairs(
            channel._apply_add_readout(raw, channel.q_add_readout)
        )
        expected_k = channel._normalize_additive_pairs(
            channel._apply_add_readout(raw, channel.k_add_readout)
        )
        torch.testing.assert_close(
            output.q,
            expected_q.unsqueeze(0).expand(2, -1, -1, -1),
        )
        torch.testing.assert_close(
            output.k,
            expected_k.unsqueeze(0).expand(2, -1, -1, -1),
        )

    def test_phase_rotation_is_linear_and_preserves_pair_radius(self):
        config = qk_config(
            "additive",
            "pair_normalized",
            mapper_kind="linear",
            qk_coupling="shared_trunk_separate_readouts",
            conditioning="phase_rotation",
            conditioning_source="residual",
            conditioning_target="both",
            conditioning_coupling="shared_trunk_separate_readouts",
            phase_bound=0.25,
            amplitude_init=0.3,
        )
        channel = self._channel(config)
        conditioner = channel.phase_rotation_conditioner
        with torch.no_grad():
            conditioner.q_up.normal_(std=0.2)
            conditioner.k_up.normal_(std=0.2)
        content = torch.randn(2, 4, 9, 32)
        output = channel(9, q_content=content, k_content=content)
        for branch, values in (("q", output.q), ("k", output.k)):
            phase = conditioner.phase(content, branch)
            self.assertTrue(torch.isfinite(phase).all())
            pair_radius = (
                values[..., :4].square() + values[..., 4:].square()
            ).sqrt()
            torch.testing.assert_close(
                pair_radius,
                torch.full_like(pair_radius, 0.3),
                atol=2e-5,
                rtol=2e-5,
            )

    def test_phase_rotation_targeting_and_zero_init_gradients(self):
        config = qk_config(
            "additive",
            "pair_normalized",
            mapper_kind="linear",
            qk_coupling="shared_trunk_separate_readouts",
            conditioning="phase_rotation",
            conditioning_source="residual",
            conditioning_target="q",
            amplitude_init=0.3,
        )
        channel = self._channel(config)
        content = torch.randn(2, 4, 9, 32)
        output = channel(9, q_content=content, k_content=content)
        probe = torch.randn_like(output.q)
        (output.q * probe).sum().backward()
        conditioner = channel.phase_rotation_conditioner
        self.assertIsNotNone(conditioner.q_up.grad)
        self.assertGreater(conditioner.q_up.grad.abs().sum().item(), 0)
        self.assertIsNone(conditioner.k_up)
        self.assertEqual(output.k.ndim, 3)

    def test_additive_content_phase_starts_at_carrier_anchor_and_opens(self):
        config = qk_config(
            "additive",
            "amplitude_phase",
            qk_coupling="shared_trunk_separate_readouts",
            conditioning="additive_phase",
            conditioning_source="dedicated",
            conditioning_coupling="shared_trunk_separate_readouts",
            amplitude_init=0.3,
        )
        channel = self._channel(config)
        content = torch.randn(2, 4, 9, 8)
        output = channel(9, q_content=content, k_content=content)
        expected = torch.cat(
            (
                0.3 * channel.base_cos[:9],
                0.3 * channel.base_sin[:9],
            ),
            dim=-1,
        )[None, None].expand(2, 4, -1, -1)
        torch.testing.assert_close(output.q, expected)
        torch.testing.assert_close(output.k, expected)

        probe = torch.randn_like(output.q)
        (output.q * probe).sum().backward()
        actuator = channel.content_actuator
        self.assertGreater(actuator.q_up.grad.abs().sum().item(), 0)

    def test_rope_content_phase_and_adaptive_gain_are_exact_nulls(self):
        content = torch.randn(2, 4, 9, 8)
        phase_channel = self._channel(
            qk_config(
                "rotary",
                "phase",
                conditioning="rope_phase",
                conditioning_source="dedicated",
            )
        )
        phase = phase_channel(9, q_content=content, k_content=content)
        self.assertEqual(torch.count_nonzero(phase.q).item(), 0)
        self.assertEqual(torch.count_nonzero(phase.k).item(), 0)

        gain_channel = self._channel(
            qk_config(
                "rotary",
                "phase",
                conditioning="adaptive_gain",
                conditioning_source="dedicated",
            )
        )
        gain = gain_channel(9, q_content=content, k_content=content)
        torch.testing.assert_close(gain.q_gain, torch.ones_like(gain.q_gain))
        torch.testing.assert_close(gain.k_gain, torch.ones_like(gain.k_gain))

    def test_dedicated_position_content_is_low_rank_and_configurable(self):
        config = qk_config(
            "rotary",
            "phase",
            conditioning="rope_phase",
            conditioning_source="dedicated",
        )
        for coupling in ("shared", "separate"):
            with self.subTest(coupling=coupling):
                attention = Attention(
                    32,
                    4,
                    qk_config=config,
                    position_content_dim=6,
                    position_content_coupling=coupling,
                    qk_norm_mode="method_aware_rms",
                )
                q_content, k_content = attention.position_content(
                    torch.randn(2, 7, 32)
                )
                self.assertEqual(q_content.shape, (2, 4, 7, 6))
                torch.testing.assert_close(
                    q_content.square().mean(-1),
                    torch.ones_like(q_content[..., 0]),
                    atol=2e-5,
                    rtol=2e-5,
                )
                if coupling == "shared":
                    torch.testing.assert_close(q_content, k_content)
                else:
                    self.assertIsNot(
                        attention.position_content.q_projection,
                        attention.position_content.k_projection,
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

        unit_pair = self._channel(qk_config("rotary", "unit_pair"))(9)
        self.assertEqual(unit_pair.q.shape, (4, 9, 4))
        torch.testing.assert_close(
            unit_pair.q,
            torch.zeros_like(unit_pair.q),
            atol=1e-6,
            rtol=0,
        )

        scaled = self._channel(
            qk_config("rotary", "scaled_phase", scale_init=1.0)
        )(9)
        self.assertEqual(torch.count_nonzero(scaled.q).item(), 0)
        torch.testing.assert_close(
            scaled.q_scale,
            torch.ones_like(scaled.q_scale),
        )

    def test_bounded_amplitude_and_additive_gain_control_branch_rms(self):
        config = qk_config(
            "additive",
            "amplitude_phase",
            qk_coupling="shared_trunk_separate_readouts",
            amplitude_init=0.3,
        )
        config["output"].update(
            {
                "amplitude_parameterization": "bounded_sigmoid",
                "amplitude_max": 1.0,
                "additive_normalization": "rms",
                "additive_gain_init": 0.2,
                "additive_gain_max": 0.5,
            }
        )
        channel = self._channel(config)
        with torch.no_grad():
            channel.q_amplitude_head.bias.fill_(1000.0)
            channel.k_amplitude_head.bias.fill_(-1000.0)
        output = channel(8)
        torch.testing.assert_close(
            output.q.float().square().mean(dim=-1).sqrt(),
            torch.full_like(output.q[..., 0], 0.2),
            atol=2e-5,
            rtol=2e-5,
        )
        self.assertLessEqual(output.k.abs().max().item(), 0.2)

    def test_bounded_rotary_scale_limits_pair_anisotropy(self):
        config = qk_config("rotary", "scaled_phase")
        config["output"].update(
            {"scale_parameterization": "bounded_log", "scale_max": 2.0}
        )
        channel = self._channel(config)
        with torch.no_grad():
            channel.scale_head.bias.fill_(1000.0)
        output = channel(8)
        self.assertLessEqual(output.q_scale.max().item(), 2.0)
        self.assertGreaterEqual(output.k_scale.min().item(), 0.5)

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

    def test_scaled_sigmoid_gate_starts_at_one_and_stays_bounded(self):
        content = torch.randn(2, 1, 3, 4)
        base = torch.ones(1, 3, 2)
        conditioner = GroupedContentConditioner(
            kind="content_gate",
            groups=1,
            content_dim=4,
            output_dim=2,
            hidden_dim=4,
            gate_init=1.0,
            activation="scaled_sigmoid",
        )
        expected = base.unsqueeze(0).expand(2, -1, -1, -1)
        torch.testing.assert_close(conditioner(base, content), expected)
        with torch.no_grad():
            conditioner.gate_bias.fill_(1000.0)
        self.assertLessEqual(conditioner(base, content).max().item(), 2.0)

    def test_residual_content_source_uses_dedicated_conditioners(self):
        config = qk_config(
            "additive",
            "amplitude_phase",
            qk_coupling="shared_trunk_separate_readouts",
            conditioning="content_gate",
        )
        config["conditioning"].update(
            {
                "source": "residual",
                "activation": "scaled_sigmoid",
                "gate_init": 1.0,
            }
        )
        channel = self._channel(config)
        self.assertEqual(channel.q_conditioner.content_dim, 32)
        self.assertIsNot(channel.q_conditioner, channel.k_conditioner)
        residual = torch.randn(2, 4, 8, 32)
        output = channel(8, q_content=residual, k_content=residual)
        self.assertEqual(output.q.shape, (2, 4, 8, 8))

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

    def test_real_content_position_diagnostics_report_mixture(self):
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
        ).eval()
        ids = torch.randint(0, 64, (1, 8))
        metrics, _ = model.position_diagnostics(
            sequence_length=8,
            input_ids=ids,
        )
        self.assertIn(
            "position/layer_00/qk/addend_q_to_q_ratio_p95",
            metrics,
        )
        self.assertIn(
            "position/layer_00/qk/q_content_combined_cosine_mean",
            metrics,
        )


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

    def test_post_position_qk_norm_is_parameter_free_unit_rms(self):
        attention = Attention(
            32,
            4,
            max_seq_len=16,
            post_position_qk_norm=True,
        )
        value = torch.randn(2, 4, 8, 8) * 7.0 + 3.0
        normalized = attention._unit_rms(value)
        torch.testing.assert_close(
            normalized.float().square().mean(dim=-1),
            torch.ones_like(normalized[..., 0].float()),
            atol=2e-6,
            rtol=2e-6,
        )
        self.assertEqual(
            sum(p.numel() for p in attention.parameters()),
            sum(
                p.numel()
                for p in Attention(32, 4, max_seq_len=16).parameters()
            ),
        )

    def test_static_qk_channel_does_not_build_or_request_content(self):
        config = qk_config(
            "additive",
            "amplitude_phase",
            learn_amplitude=False,
            learn_phase=False,
        )
        config["conditioning"]["source"] = "dedicated"
        attention = Attention(
            32,
            4,
            max_seq_len=16,
            qk_config=config,
            qk_norm_mode="method_aware_rms",
        )
        self.assertIsNone(attention.position_content)
        output = attention(torch.randn(2, 8, 32))
        self.assertEqual(output.shape, (2, 8, 32))

    def test_rope_disable_rejects_rotary_qk_channel(self):
        common = self._common()
        common["qk_config"] = qk_config("rotary", "phase")
        with self.assertRaisesRegex(ValueError, "rotary Q/K"):
            Transformer(
                **common,
                use_rope=False,
            )

    def test_method_aware_rms_normalizes_after_additive_position(self):
        config = qk_config(
            "additive",
            "free",
            qk_coupling="shared",
        )
        attention = Attention(
            32,
            4,
            max_seq_len=16,
            qk_config=config,
            qk_norm_mode="method_aware_rms",
        )
        captured = {}
        def capture_q_input(_module, args):
            captured["q_input"] = args[0]

        handle = attention.q_norm.register_forward_pre_hook(capture_q_input)
        x = torch.randn(2, 7, 32)
        attention(x)
        handle.remove()
        projected = attention._split_heads(attention.to_q(x))
        addend = attention.qk_position(7, dtype=projected.dtype).q[None]
        torch.testing.assert_close(captured["q_input"], projected + addend)
        self.assertIsInstance(attention.q_norm, torch.nn.RMSNorm)

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

    def test_query_position_write_is_exact_null_with_live_final_gradient(self):
        config = normalize_attention_write_config(
            {
                "enabled": True,
                "mode": "query_position",
            },
            model_dim=32,
            heads=4,
            rope_theta=10_000.0,
        )
        torch.manual_seed(17)
        baseline = Transformer(**self._common()).eval()
        torch.manual_seed(17)
        candidate = Transformer(
            **self._common(),
            attention_write_config=config,
        ).eval()
        candidate.load_state_dict(baseline.state_dict(), strict=False)
        ids = torch.randint(0, 64, (2, 8))
        torch.testing.assert_close(baseline(ids), candidate(ids))

        candidate(ids).sum().backward()
        channel = candidate.blocks[0].attn.position_write
        self.assertIsNone(channel.gate)
        self.assertGreater(
            channel.query_projection.weight.grad.abs().sum().item(),
            0,
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

    def test_post_position_qk_norm_round_trips(self):
        config = self._load_payload({"post_position_qk_norm": True})
        self.assertTrue(config.post_position_qk_norm)
        model = make_model(config, 64)
        self.assertTrue(model.blocks[0].attn.post_position_qk_norm)

    def test_training_model_and_evaluation_lengths_are_independent(self):
        config = self._load_payload(
            {
                "training_length": 16,
                "model_position_extent": 64,
                "evaluation_lengths": [32, 64],
                "scalar_normalization_extent": 16,
                "per_device_train_batch_size": 8,
                "per_device_eval_batch_size": 2,
                "gradient_accumulation_steps": 4,
                "gradient_checkpointing": True,
                "checkpointing_steps": 5000,
                "resume_from_checkpoint": "auto",
                "save_final_model": True,
                "compile_mode": "max-autotune-no-cudagraphs",
            }
        )
        self.assertEqual(config.block_size, 16)
        self.assertEqual(config.training_length, 16)
        self.assertEqual(config.model_position_extent, 64)
        self.assertEqual(config.evaluation_lengths, [16, 32, 64])
        self.assertEqual(config.scalar_normalization_extent, 16)
        self.assertEqual(config.per_device_train_batch_size, 8)
        self.assertEqual(config.per_device_eval_batch_size, 2)
        self.assertEqual(config.gradient_accumulation_steps, 4)
        self.assertTrue(config.gradient_checkpointing)
        self.assertEqual(config.checkpointing_steps, 5000)
        self.assertEqual(config.resume_from_checkpoint, "auto")
        self.assertTrue(config.save_final_model)
        self.assertEqual(config.compile_mode, "max-autotune-no-cudagraphs")

        with self.assertRaisesRegex(ValueError, "must cover"):
            self._load_payload(
                {
                    "training_length": 16,
                    "model_position_extent": 32,
                    "evaluation_lengths": [64],
                }
            )

        for key in (
            "per_device_train_batch_size",
            "per_device_eval_batch_size",
            "gradient_accumulation_steps",
            "checkpointing_steps",
        ):
            with self.subTest(key=key):
                with self.assertRaisesRegex(ValueError, "must be positive"):
                    self._load_payload({key: 0})

    def test_50k_scale_configuration_round_trips(self):
        config = self._load_payload(
            {
                "hidden_size": 1024,
                "depth": 12,
                "n_head": 16,
                "training_length": 1024,
                "model_position_extent": 4096,
                "evaluation_lengths": [1024, 2048, 4096],
                "per_device_train_batch_size": 8,
                "per_device_eval_batch_size": 1,
                "gradient_accumulation_steps": 4,
                "gradient_checkpointing": True,
                "max_train_steps": 50000,
                "num_warmup_steps": 1000,
                "checkpointing_steps": 5000,
                "resume_from_checkpoint": "auto",
                "save_final_model": True,
            }
        )
        self.assertEqual(config.hidden_size, 1024)
        self.assertEqual(config.depth, 12)
        self.assertEqual(config.per_device_train_batch_size, 8)
        self.assertEqual(config.per_device_eval_batch_size, 1)
        self.assertEqual(
            config.per_device_train_batch_size
            * config.gradient_accumulation_steps,
            32,
        )
        self.assertEqual(config.evaluation_lengths, [1024, 2048, 4096])
        self.assertEqual(config.checkpointing_steps, 5000)
        self.assertEqual(config.resume_from_checkpoint, "auto")
        self.assertTrue(config.save_final_model)

    @unittest.skipUnless(
        torch.cuda.is_available() and os.environ.get("RUN_CUDA_TESTS") == "1",
        "requires an explicitly claimed CUDA device",
    )
    def test_gradient_checkpointed_compiled_forward_backward(self):
        config = self._load_payload(
            {
                "hidden_size": 32,
                "depth": 2,
                "n_head": 4,
                "training_length": 16,
                "model_position_extent": 16,
                "gradient_checkpointing": True,
                "compile": True,
            }
        )
        model = torch.compile(make_model(config, 64).cuda(), mode="default")
        input_ids = torch.randint(0, 64, (2, 15), device="cuda")
        loss = model(input_ids=input_ids, targets=input_ids)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))

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
