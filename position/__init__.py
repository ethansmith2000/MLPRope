"""Composable positional encoding channels for MLPRope experiments."""

from position.autograd import exp_with_identity_grad
from position.basis import (
    FrozenFourierBasis,
    build_position_basis,
    interleaved_fourier_basis,
)
from position.channels import (
    PositionChannel,
    QKPositionChannel,
    QKPositionOutput,
    adapt_legacy_position_state_dict,
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
    normalize_logit_bias_config,
    normalize_position_config_v2,
    normalize_position_content_config,
    resolve_channel_config,
    upgrade_legacy_position_config,
    v2_position_run_tag,
)
from position.mappers import FeatureMapper, build_mapper
from position.optimization import (
    InterventionOptimizationMonitor,
    collect_intervention_parameter_groups,
    intervention_optimization_due,
)
from position.preprojection import (
    QK_PREPROJECTION_DEFAULTS,
    QK_PREPROJECTION_MODES,
    QKPreprojectionPosition,
    QKPreprojectionOutput,
    normalize_qk_preprojection_config,
)
from position.rotary import (
    apply_rotary,
    build_rope_cache,
    build_rope_frequencies,
    rotate_half,
)

__all__ = [
    "FrozenFourierBasis",
    "FeatureMapper",
    "InterventionOptimizationMonitor",
    "POSITION_PRESETS",
    "POSITION_SCHEMA_VERSION",
    "POSITION_VARIANTS",
    "PositionChannel",
    "QKPositionChannel",
    "QKPositionOutput",
    "V1_CHANNEL_DEFAULTS",
    "V2_CHANNEL_DEFAULTS",
    "QK_PREPROJECTION_DEFAULTS",
    "QK_PREPROJECTION_MODES",
    "QKPreprojectionPosition",
    "QKPreprojectionOutput",
    "adapt_legacy_position_state_dict",
    "apply_rotary",
    "build_mapper",
    "build_qk_position_channel",
    "build_position_basis",
    "build_rope_cache",
    "build_rope_frequencies",
    "channel_theta",
    "collect_intervention_parameter_groups",
    "count_position_parameters",
    "deep_merge",
    "ensure_channel_v2",
    "exp_with_identity_grad",
    "interleaved_fourier_basis",
    "intervention_optimization_due",
    "legacy_position_run_tag",
    "load_position_compatible_state_dict",
    "normalize_logit_bias_config",
    "normalize_position_config_v2",
    "normalize_position_content_config",
    "normalize_qk_preprojection_config",
    "resolve_channel_config",
    "rotate_half",
    "upgrade_legacy_position_config",
    "v2_position_run_tag",
]
