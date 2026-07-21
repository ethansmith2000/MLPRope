import json
import tempfile
import unittest
from argparse import Namespace

import torch

from train_gpt import load_config
from transformer import LogitBiasChannel, QKPositionChannel, Transformer


FEATURE_MAPS = (
    "identity",
    "add_rope",
    "linear",
    "low_rank",
    "bottleneck_mlp",
    "mlp",
)
SHARING_MODES = ("shared_head", "per_head", "full_dim")
# Feature maps that start as (scaled) sinusoids for true AddRoPE.
ADDROPE_SINUSOID_MAPS = ("identity", "add_rope", "low_rank", "bottleneck_mlp", "mlp")


class PositionChannelTest(unittest.TestCase):
    def test_logit_contract_and_zero_init(self):
        for sharing in SHARING_MODES:
            for feature_map in FEATURE_MAPS:
                with self.subTest(sharing=sharing, feature_map=feature_map):
                    channel = LogitBiasChannel(
                        feature_map=feature_map,
                        sharing=sharing,
                        heads=4,
                        head_dim=8,
                        extent=16,
                        theta=10_000.0,
                        rank=4,
                        mlp_hidden=12,
                    )
                    curves = channel()
                    self.assertEqual(curves.shape, (4, 16))
                    self.assertEqual(torch.count_nonzero(curves).item(), 0)

    def test_qk_phase_zero_init_and_add_sinusoid_init(self):
        for sharing in SHARING_MODES:
            for feature_map in FEATURE_MAPS:
                with self.subTest(sharing=sharing, feature_map=feature_map, apply="phase"):
                    phase = QKPositionChannel(
                        feature_map=feature_map,
                        sharing=sharing,
                        apply="phase_residual",
                        heads=4,
                        head_dim=8,
                        extent=16,
                        theta=10_000.0,
                        rank=4,
                        mlp_hidden=12,
                    )
                    output = phase(9)
                    self.assertEqual(output.shape, (4, 9, 4))
                    self.assertEqual(torch.count_nonzero(output).item(), 0)

                with self.subTest(sharing=sharing, feature_map=feature_map, apply="add"):
                    add = QKPositionChannel(
                        feature_map=feature_map,
                        sharing=sharing,
                        apply="add",
                        heads=4,
                        head_dim=8,
                        extent=16,
                        theta=10_000.0,
                        rank=4,
                        mlp_hidden=12,
                    )
                    output = add(9)
                    self.assertEqual(output.shape, (4, 9, 8))
                    # True AddRoPE uses the feature map as the addend; sinusoid
                    # and residual maps are non-zero at initialization.
                    if feature_map in ADDROPE_SINUSOID_MAPS:
                        self.assertGreater(torch.count_nonzero(output).item(), 0)

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
        baseline = Transformer(
            **common,
            qk_config={"enabled": False},
        ).eval()
        input_ids = torch.randint(0, 64, (2, 8))
        expected = baseline(input_ids)

        candidate = Transformer(
            **common,
            qk_config={
                "enabled": True,
                "feature_map": "mlp",
                "sharing": "per_head",
                "apply": "phase_residual",
                "rank": 4,
                "mlp_hidden": 12,
            },
        ).eval()
        candidate.load_state_dict(baseline.state_dict(), strict=False)
        actual = candidate(input_ids)
        torch.testing.assert_close(actual, expected)

    def test_addrope_does_not_apply_multiplicative_rope(self):
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
        addrope = Transformer(
            **common,
            qk_config={
                "enabled": True,
                "feature_map": "identity",
                "sharing": "per_head",
                "apply": "add",
                "rank": 4,
                "mlp_hidden": 12,
            },
        ).eval()
        self.assertTrue(rope.blocks[0].attn.multiplicative_rope)
        self.assertFalse(addrope.blocks[0].attn.multiplicative_rope)

        addrope.load_state_dict(rope.state_dict(), strict=False)
        input_ids = torch.randint(0, 64, (2, 8))
        rope_out = rope(input_ids)
        add_out = addrope(input_ids)
        self.assertFalse(torch.allclose(rope_out, add_out))

    def test_addrope_residual_maps_match_identity_at_init(self):
        for feature_map in ("low_rank", "bottleneck_mlp", "mlp"):
            with self.subTest(feature_map=feature_map):
                identity = QKPositionChannel(
                    feature_map="identity",
                    sharing="per_head",
                    apply="add",
                    heads=4,
                    head_dim=8,
                    extent=16,
                    theta=10_000.0,
                    rank=4,
                    mlp_hidden=12,
                )
                residual = QKPositionChannel(
                    feature_map=feature_map,
                    sharing="per_head",
                    apply="add",
                    heads=4,
                    head_dim=8,
                    extent=16,
                    theta=10_000.0,
                    rank=4,
                    mlp_hidden=12,
                )
                torch.testing.assert_close(identity(11), residual(11))

    def test_low_rank_and_bottleneck_have_fixed_feature_shape(self):
        low_rank = LogitBiasChannel(
            feature_map="low_rank",
            sharing="per_head",
            heads=4,
            head_dim=8,
            extent=16,
            theta=10_000.0,
            rank=3,
            mlp_hidden=12,
        )
        bottleneck = LogitBiasChannel(
            feature_map="bottleneck_mlp",
            sharing="per_head",
            heads=4,
            head_dim=8,
            extent=16,
            theta=10_000.0,
            rank=3,
            mlp_hidden=12,
        )
        self.assertEqual(low_rank.features().shape, (4, 16, 8))
        self.assertEqual(bottleneck.features().shape, (4, 16, 8))
        self.assertEqual(low_rank.features.up.shape, (4, 3, 8))
        self.assertEqual(bottleneck.features.up.shape, (4, 3, 8))


class PositionConfigTest(unittest.TestCase):
    @staticmethod
    def _cli(path):
        return Namespace(
            override_json=path,
            pos_variant=None,
            attn_impl=None,
            max_train_steps=None,
            dry_run=False,
            print_model=False,
        )

    def _load(self, overrides):
        with tempfile.NamedTemporaryFile("w", suffix=".json") as config_file:
            json.dump(overrides, config_file)
            config_file.flush()
            return load_config(self._cli(config_file.name))

    def test_legacy_preset_expands_to_logit_channel(self):
        config = self._load({"pos_variant": "low_rank", "pos_rank": 7})
        self.assertFalse(config.qk["enabled"])
        self.assertTrue(config.logit_bias["enabled"])
        self.assertEqual(config.logit_bias["feature_map"], "low_rank")
        self.assertEqual(config.logit_bias["rank"], 7)
        self.assertEqual(config.attn_impl, "flex")

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
        self.assertEqual(config.qk["apply"], "add")
        self.assertTrue(config.run_name.startswith("qk-add-identity"))


if __name__ == "__main__":
    unittest.main()
