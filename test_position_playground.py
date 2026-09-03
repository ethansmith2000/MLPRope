"""Coverage for the fully configurable position experimentation playground."""

from __future__ import annotations

import json
import math
import os
import tempfile
import unittest
from argparse import Namespace
from unittest import mock
from pathlib import Path
from types import SimpleNamespace

import torch

from position.channels import GroupedContentConditioner
from position import (
    FeatureMapper,
    build_position_basis,
    build_qk_position_channel,
    exp_with_identity_grad,
    interleaved_fourier_basis,
    normalize_position_config_v2,
)
from train_gpt import RechunkedTokenDataset, evaluate, load_config, make_model
from transformer import (
    Attention,
    Transformer,
    TransformerBlock,
    count_parameters,
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
    conditioning_source: str = "dedicated",
    conditioning_target: str = "both",
    conditioning_coupling: str = "shared_trunk_separate_readouts",
    conditioning_static_complement: bool = False,
    phase_bound: float = 0.25,
    conditioning_input_mode: str = "content",
    conditioning_input_normalization: str = "none",
    conditioning_learnable_input_gains: bool = False,
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
                "input_normalization": conditioning_input_normalization,
                "learnable_input_gains": conditioning_learnable_input_gains,
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


class PositionBasisTest(unittest.TestCase):
    def test_exp_parameterization_uses_exp_forward_identity_backward(self):
        value = torch.tensor([-3.0, 0.0, 3.0], requires_grad=True)
        output = exp_with_identity_grad(value)
        torch.testing.assert_close(output, value.detach().exp())
        output.sum().backward()
        torch.testing.assert_close(value.grad, torch.ones_like(value))

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
            "additive",
            "amplitude_phase",
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
            "additive",
            "amplitude_phase",
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

    def test_position_only_carrier_is_content_invariant(self):
        """Behavioral form of the phase-19 severance guarantee.

        The existing structural test asserts the content projector is never
        built; this one asserts the carrier *values* cannot depend on content
        even if a caller supplies some, with the hypernetwork deliberately
        randomized away from its zero-initialized (trivially invariant) state.
        """
        config = qk_config(
            "additive",
            "amplitude_phase",
            conditioning="carrier_hypernetwork",
            conditioning_source="dedicated",
            conditioning_coupling="shared_trunk_separate_readouts",
            conditioning_input_mode="position",
            conditioning_network="silu_mlp",
            conditioning_components="amplitude_phase",
            learn_amplitude=False,
            learn_phase=False,
            amplitude_init=1.0,
        )
        anchor = self._channel(config)
        channel = self._channel(config)
        torch.manual_seed(7)
        with torch.no_grad():
            for parameter in channel.parameters():
                torch.nn.init.normal_(parameter, std=0.05)
        content_a = torch.randn(2, 4, 7, 8)
        content_b = torch.randn(2, 4, 7, 8)
        base = channel(7)
        from_a = channel(7, q_content=content_a, k_content=content_a)
        from_b = channel(7, q_content=content_b, k_content=content_b)
        for attr in ("q", "k"):
            randomized = getattr(base, attr)
            zeroed = getattr(anchor(7), attr)
            self.assertFalse(
                torch.equal(randomized, zeroed),
                f"randomized hypernetwork left {attr} on the anchor; "
                "the invariance check below would be vacuous",
            )
            output_a = getattr(from_a, attr)
            output_b = getattr(from_b, attr)
            self.assertTrue(torch.equal(output_a, output_b))
            self.assertTrue(
                torch.equal(output_a, randomized.expand_as(output_a))
            )

    def test_paired_initialization_matches_every_shared_tensor(self):
        common = {
            "dim": 32,
            "depth": 2,
            "heads": 4,
            "ff_mult": 2,
            "vocab_size": 64,
            "max_seq_len": 16,
            "attn_impl": "sdpa",
            "logit_bias_config": {"enabled": False},
            "paired_initialization_seed": 456,
        }
        position_config = qk_config(
            "additive",
            "amplitude_phase",
            conditioning="carrier_hypernetwork",
            conditioning_source="dedicated",
            conditioning_input_mode="position",
            conditioning_network="silu_mlp",
            conditioning_components="amplitude_phase",
            parameter_source="direct",
            amplitude_init=1.0,
            learn_amplitude=False,
            learn_phase=False,
        )
        torch.manual_seed(1)
        baseline = Transformer(**common, qk_config={"enabled": False})
        torch.manual_seed(999)
        candidate = Transformer(**common, qk_config=position_config)
        baseline_state = baseline.state_dict()
        candidate_state = candidate.state_dict()
        shared = {
            name
            for name, value in baseline_state.items()
            if name in candidate_state and candidate_state[name].shape == value.shape
        }
        self.assertGreater(len(shared), 20)
        for name in shared:
            with self.subTest(name=name):
                torch.testing.assert_close(
                    baseline_state[name],
                    candidate_state[name],
                    rtol=0,
                    atol=0,
                )








    def test_layer_selective_ffn_widening(self):
        common = {
            "dim": 32,
            "depth": 4,
            "heads": 4,
            "ff_mult": 2,
            "vocab_size": 64,
            "max_seq_len": 16,
            "qk_config": {"enabled": False},
            "logit_bias_config": {"enabled": False},
        }
        baseline = Transformer(**common)
        widened = Transformer(
            **common,
            ff_widened_hidden_dim=128,
            ff_widened_layers=[1, 3],
        )
        self.assertEqual(
            [block.ff.proj_out.in_features for block in widened.blocks],
            [64, 128, 64, 128],
        )
        expected_added = 2 * (128 - 64) * (3 * 32 + 2)
        self.assertEqual(
            count_parameters(widened)["non_embed"]
            - count_parameters(baseline)["non_embed"],
            expected_added,
        )

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

    def test_hypernetwork_modality_rms_and_learnable_gains(self):
        config = qk_config(
            "additive",
            "amplitude_phase",
            qk_coupling="shared_trunk_separate_readouts",
            conditioning="carrier_hypernetwork",
            conditioning_source="dedicated",
            conditioning_input_mode="content_position",
            conditioning_input_normalization="modality_rms",
            conditioning_learnable_input_gains=True,
            conditioning_components="amplitude_phase",
            amplitude_parameterization="signed",
            amplitude_init=1.0,
            parameter_source="direct",
            learn_amplitude=False,
            learn_phase=False,
        )
        channel = self._channel(config)
        hyper = channel.carrier_hypernetwork
        self.assertEqual(hyper.content_input_gain.item(), 1.0)
        self.assertEqual(hyper.position_input_gain.item(), 1.0)
        content = torch.randn(2, 4, 9, 8) * 7.0
        position = channel._hyper_position_features(9, dtype=content.dtype)
        inputs = hyper._inputs(content, position)
        content_inputs = inputs[..., :8]
        position_inputs = inputs[..., 8:]
        torch.testing.assert_close(
            content_inputs.square().mean(dim=-1),
            torch.ones_like(content_inputs[..., 0]),
            rtol=2e-5,
            atol=2e-5,
        )
        torch.testing.assert_close(
            position_inputs.square().mean(dim=-1),
            torch.ones_like(position_inputs[..., 0]),
            rtol=2e-5,
            atol=2e-5,
        )
        with torch.no_grad():
            hyper.content_input_gain.fill_(2.0)
            hyper.position_input_gain.fill_(0.5)
        scaled = hyper._inputs(content, position)
        torch.testing.assert_close(scaled[..., :8], content_inputs * 2.0)
        torch.testing.assert_close(scaled[..., 8:], position_inputs * 0.5)

    def test_hyperaddrope_output_geometries_preserve_anchor_and_gradients(self):
        modes = {
            "amplitude": (False, False),
            "phase": (False, False),
            "amplitude_phase": (False, False),
            "cartesian": (False, False),
        }
        content = torch.randn(2, 4, 9, 8)
        for components, (learn_amplitude, learn_phase) in modes.items():
            with self.subTest(components=components):
                config = qk_config(
                    "additive",
                    "amplitude_phase",
                    qk_coupling="shared_trunk_separate_readouts",
                    conditioning="carrier_hypernetwork",
                    conditioning_source="dedicated",
                    conditioning_input_mode="content_position",
                    conditioning_components=components,
                    amplitude_parameterization="signed",
                    amplitude_init=1.0,
                    parameter_source="direct",
                    learn_amplitude=learn_amplitude,
                    learn_phase=learn_phase,
                )
                channel = self._channel(config)
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
                for chunk in gradient.split(4, dim=-1):
                    self.assertGreater(chunk.abs().sum().item(), 0)

    def test_cartesian_residual_multiplies_base_carrier(self):
        config = qk_config(
            "additive",
            "amplitude_phase",
            qk_coupling="shared_trunk_separate_readouts",
            conditioning="carrier_hypernetwork",
            conditioning_source="dedicated",
            conditioning_input_mode="content_position",
            conditioning_components="cartesian",
            amplitude_init=1.0,
            parameter_source="direct",
            learn_amplitude=False,
            learn_phase=False,
        )
        channel = self._channel(config)
        hyper = channel.carrier_hypernetwork
        with torch.no_grad():
            hyper.q_readout.bias[:, :4].fill_(0.2)
            hyper.q_readout.bias[:, 4:].fill_(-0.1)
        content = torch.randn(2, 4, 9, 8)
        output = channel(9, q_content=content, k_content=content)
        base_cos = channel.base_cos[:9]
        base_sin = channel.base_sin[:9]
        expected = torch.cat(
            (1.2 * base_cos + 0.1 * base_sin,
             1.2 * base_sin - 0.1 * base_cos),
            dim=-1,
        )[None, None].expand(2, 4, -1, -1)
        torch.testing.assert_close(output.q, expected)

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
            conditioning_source="dedicated",
            amplitude_init=0.3,
        )
        channel = self._channel(config)
        content = torch.randn(2, 4, 9, 8)
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
            conditioning_source="dedicated",
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
        content = torch.randn(2, 4, 9, 8)
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
            conditioning_source="dedicated",
            conditioning_target="q",
            amplitude_init=0.3,
        )
        channel = self._channel(config)
        content = torch.randn(2, 4, 9, 8)
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

    def test_additive_adaptive_gain_is_an_exact_null_at_initialization(self):
        content = torch.randn(2, 4, 9, 8)
        gain_channel = self._channel(
            qk_config(
                "additive",
                "amplitude_phase",
                conditioning="adaptive_gain",
                conditioning_source="dedicated",
            )
        )
        gain = gain_channel(9, q_content=content, k_content=content)
        torch.testing.assert_close(gain.q_gain, torch.ones_like(gain.q_gain))
        torch.testing.assert_close(gain.k_gain, torch.ones_like(gain.k_gain))

    def test_dedicated_position_content_is_low_rank_and_configurable(self):
        torch.manual_seed(0)
        config = qk_config(
            "additive",
            "amplitude_phase",
            conditioning="additive_phase",
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



    def test_rope_can_be_disabled_for_no_explicit_position_control(self):
        ids = torch.randint(0, 64, (2, 8))
        no_position = Transformer(
            **self._common(),
            use_rope=False,
        ).eval()
        self.assertFalse(no_position.blocks[0].attn.multiplicative_rope)
        self.assertEqual(no_position(ids).shape, (2, 8, 64))

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

    def test_method_aware_diagnostics_reference_raw_projection(self):
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
        with torch.no_grad():
            attention.to_q.weight.mul_(0.05)
        x = torch.randn(2, 7, 32)
        projected = attention._split_heads(attention.to_q(x))
        addend = attention.qk_position(7, dtype=projected.dtype).q[None]
        expected = (
            addend.float().square().mean().sqrt()
            / projected.float().square().mean().sqrt()
        ).item()
        summary = attention.qk_position_summary_from_input(x)
        self.assertAlmostEqual(
            summary["addend_q_to_q_rms_ratio"],
            expected,
            places=5,
        )




class EvaluationProtocolTest(unittest.TestCase):
    def test_final_evaluation_uses_disjoint_window_and_saves_details(self):
        class TinyModel(torch.nn.Module):
            def forward(self, input_ids, targets):
                return targets.float().mean()

            def position_diagnostics(self, **_kwargs):
                return {}, {}

        class TinyAccelerator:
            is_main_process = True

            @staticmethod
            def gather_for_metrics(value):
                return value

            @staticmethod
            def unwrap_model(model):
                return model

        batches = [
            {"input_ids": torch.arange(start, start + 4).reshape(1, 4)}
            for start in range(6)
        ]
        with tempfile.TemporaryDirectory() as output_dir:
            args = SimpleNamespace(
                training_length=4,
                validation_start_batch=1,
                final_validation_start_batch=3,
                num_validation_batches=2,
                num_final_validation_batches=2,
                save_evaluation_details=True,
                output_dir=output_dir,
                with_tracking=False,
            )
            model = TinyModel()
            with mock.patch("train_gpt.logger.info"):
                development = evaluate(
                    args,
                    model,
                    {4: batches},
                    TinyAccelerator(),
                    10,
                )
                final = evaluate(
                    args,
                    model,
                    {4: batches},
                    TinyAccelerator(),
                    10,
                    final_evaluation=True,
                )
            self.assertAlmostEqual(development["eval_loss"], 3.5)
            self.assertAlmostEqual(final["eval_loss"], 5.5)
            detail_path = (
                Path(output_dir)
                / "evaluation_details"
                / "step_00000010_context_000004.json"
            )
            details = json.loads(detail_path.read_text())
            self.assertEqual(details["evaluation_start_batch"], 3)
            self.assertEqual(details["losses"], [5.0, 6.0])
            rows = [
                json.loads(line)
                for line in (Path(output_dir) / "metrics.jsonl").read_text().splitlines()
            ]
            self.assertEqual(
                [row["evaluation_kind"] for row in rows],
                ["development", "final_holdout"],
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
                "paired_initialization_seed": 77,
                "ff_widened_hidden_dim": 128,
                "ff_widened_layers": [0, 2],
                "num_final_validation_batches": 128,
                "validation_start_batch": 4,
                "final_validation_start_batch": 512,
                "save_evaluation_details": True,
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
        self.assertEqual(config.paired_initialization_seed, 77)
        self.assertEqual(config.ff_widened_hidden_dim, 128)
        self.assertEqual(config.ff_widened_layers, [0, 2])
        self.assertEqual(config.num_final_validation_batches, 128)
        self.assertEqual(config.validation_start_batch, 4)
        self.assertEqual(config.final_validation_start_batch, 512)
        self.assertTrue(config.save_evaluation_details)
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
                    "attn_impl": "flex",
                    "compile": True,
                    "compile_fullgraph": True,
                }
            )

    def test_removed_aux_configs_fail_with_migration_message(self):
        for key in ("residual_stream", "attention_write"):
            with self.subTest(key=key):
                with self.assertRaisesRegex(ValueError, "removed"):
                    self._load_payload({key: {"enabled": True}})

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
