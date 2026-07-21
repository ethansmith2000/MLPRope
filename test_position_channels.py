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

    def test_qk_contract_and_zero_init(self):
        for sharing in SHARING_MODES:
            for feature_map in FEATURE_MAPS:
                for apply in ("add", "phase_residual"):
                    with self.subTest(
                        sharing=sharing,
                        feature_map=feature_map,
                        apply=apply,
                    ):
                        channel = QKPositionChannel(
                            feature_map=feature_map,
                            sharing=sharing,
                            apply=apply,
                            heads=4,
                            head_dim=8,
                            extent=16,
                            theta=10_000.0,
                            rank=4,
                            mlp_hidden=12,
                        )
                        output = channel(9)
                        output_dim = 8 if apply == "add" else 4
                        self.assertEqual(output.shape, (4, 9, output_dim))
                        self.assertEqual(torch.count_nonzero(output).item(), 0)

    def test_qk_residuals_match_rope_at_init(self):
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

        for apply in ("add", "phase_residual"):
            with self.subTest(apply=apply):
                candidate = Transformer(
                    **common,
                    qk_config={
                        "enabled": True,
                        "feature_map": "mlp",
                        "sharing": "per_head",
                        "apply": apply,
                        "rank": 4,
                        "mlp_hidden": 12,
                    },
                ).eval()
                candidate.load_state_dict(baseline.state_dict(), strict=False)
                actual = candidate(input_ids)
                torch.testing.assert_close(actual, expected)

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


if __name__ == "__main__":
    unittest.main()
