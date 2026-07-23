"""Composable positional encoding channels for MLPRope experiments."""

from position.basis import FrozenFourierBasis, interleaved_fourier_basis
from position.channels import (
    LogitBiasChannel,
    PositionChannel,
    QKPositionChannel,
    QKPositionOutput,
    adapt_legacy_position_state_dict,
    build_logit_bias_channel,
    build_qk_position_channel,
    count_position_parameters,
    load_position_compatible_state_dict,
)
from position.config import (
    POSITION_PRESETS,
    POSITION_SCHEMA_VERSION,
    POSITION_VARIANTS,
    V1_CHANNEL_DEFAULTS,
    V2_CHANNEL_DEFAULTS,
    channel_theta,
    deep_merge,
    ensure_channel_v2,
    legacy_position_run_tag,
    normalize_position_config_v2,
    resolve_channel_config,
    upgrade_legacy_position_config,
    v2_position_run_tag,
)
from position.mappers import FeatureMapper, build_mapper
from position.rotary import apply_rotary, build_rope_cache, compose_phase, rotate_half

__all__ = [
    "FrozenFourierBasis",
    "FeatureMapper",
    "LogitBiasChannel",
    "POSITION_PRESETS",
    "POSITION_SCHEMA_VERSION",
    "POSITION_VARIANTS",
    "PositionChannel",
    "QKPositionChannel",
    "QKPositionOutput",
    "V1_CHANNEL_DEFAULTS",
    "V2_CHANNEL_DEFAULTS",
    "adapt_legacy_position_state_dict",
    "apply_rotary",
    "build_logit_bias_channel",
    "build_mapper",
    "build_qk_position_channel",
    "build_rope_cache",
    "channel_theta",
    "compose_phase",
    "count_position_parameters",
    "deep_merge",
    "ensure_channel_v2",
    "interleaved_fourier_basis",
    "legacy_position_run_tag",
    "load_position_compatible_state_dict",
    "normalize_position_config_v2",
    "resolve_channel_config",
    "rotate_half",
    "upgrade_legacy_position_config",
    "v2_position_run_tag",
]
