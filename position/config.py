"""Canonical v2 position-channel configuration and v1 upgrade helpers."""

from __future__ import annotations

import copy
from typing import Any, Literal

POSITION_SCHEMA_VERSION = 2

Application = Literal["additive", "rotary", "logit_bias"]
Geometry = Literal[
    "free",
    "pair_normalized",
    "amplitude_phase",
    "phase",
    "projected_phase",
    "unit_pair",
    "scaled_phase",
    "scalar_curve",
]
InputKind = Literal[
    "frozen_fourier",
    "learned_temperature_fourier",
    "learned_frequency_fourier",
]
MapperKind = Literal[
    "identity",
    "euclidean_affine",
    "linear",
    "low_rank",
    "bottleneck_mlp",
    "mlp",
]
QKCoupling = Literal["shared", "shared_trunk_separate_readouts", "separate"]
HeadCoupling = Literal["shared_head", "per_head_independent", "per_head_joint"]

V1_FEATURE_MAPS = {
    "identity",
    "add_rope",
    "linear",
    "low_rank",
    "bottleneck_mlp",
    "mlp",
}
V1_SHARING_MODES = {"shared_head", "per_head", "full_dim"}
V1_QK_APPLY_MODES = {"add", "phase_residual"}

V1_QK_KEYS = {
    "enabled",
    "feature_map",
    "sharing",
    "apply",
    "rank",
    "mlp_hidden",
}
V1_LOGIT_KEYS = {
    "enabled",
    "feature_map",
    "sharing",
    "rank",
    "mlp_hidden",
}
V2_COMMON_KEYS = {
    "enabled",
    "application",
    "geometry",
    "input",
    "mapper",
    "output",
    "conditioning",
    "head_coupling",
}
V2_QK_KEYS = V2_COMMON_KEYS | {"qk_coupling"}
V2_LOGIT_KEYS = V2_COMMON_KEYS
V2_INPUT_KEYS = {
    "kind",
    "basis_dim",
    "theta",
    "scalars",
    "normalization_extent",
}
V2_MAPPER_KEYS = {"kind", "residual", "rank", "hidden_dim"}
V2_OUTPUT_KEYS = {
    "amplitude_init",
    "amplitude_max",
    "amplitude_parameterization",
    "learn_amplitude",
    "learn_phase",
    "phase_scale",
    "additive_normalization",
    "additive_gain_init",
    "additive_gain_max",
    "learn_additive_gain",
    "scale_init",
    "scale_max",
    "scale_parameterization",
}
V2_CONDITIONING_KEYS = {
    "kind",
    "source",
    "activation",
    "hidden_dim",
    "input_mode",
    "network",
    "components",
    "head_coupling",
    "gate_init",
    "target",
    "coupling",
    "phase_bound",
    "pair_rank",
    "position_mode",
    "num_profiles",
    "router_hidden_dim",
    "profile_init_std",
    "num_frequencies",
}

POSITION_CONTENT_COUPLINGS = {"shared", "separate"}

# Legacy v1 names that must not appear on a v2 channel (except enabled).
V1_ONLY_KEYS = {"feature_map", "sharing", "apply", "rank", "mlp_hidden"}
# v2 axis keys that must not appear on a v1 channel.
V2_ONLY_KEYS = {
    "application",
    "geometry",
    "input",
    "mapper",
    "output",
    "conditioning",
    "qk_coupling",
    "head_coupling",
}

POSITION_ONLY_VARIANTS = {
    "add_rope",
    "linear",
    "low_rank",
    "bottleneck_mlp",
    "mlp_rope",
}
CONTENT_CONDITIONED_VARIANTS = {"inkling_table", "inkling_cosnet"}
POSITION_VARIANTS = {"rope"} | POSITION_ONLY_VARIANTS | CONTENT_CONDITIONED_VARIANTS

POSITION_PRESETS = {
    "rope": {},
    "add_rope": {
        "logit_bias": {"enabled": True, "feature_map": "add_rope"},
    },
    "linear": {
        "logit_bias": {"enabled": True, "feature_map": "linear"},
    },
    "low_rank": {
        "logit_bias": {"enabled": True, "feature_map": "low_rank"},
    },
    "bottleneck_mlp": {
        "logit_bias": {"enabled": True, "feature_map": "bottleneck_mlp"},
    },
    "mlp_rope": {
        "logit_bias": {"enabled": True, "feature_map": "mlp"},
    },
    "inkling_table": {
        "logit_bias": {
            "enabled": True,
            "application": "logit_bias",
            "geometry": "scalar_curve",
            "conditioning": {"kind": "inkling_table"},
        },
    },
    "inkling_cosnet": {
        "logit_bias": {
            "enabled": True,
            "application": "logit_bias",
            "geometry": "scalar_curve",
            "conditioning": {"kind": "inkling_cosnet"},
        },
    },
}

V1_CHANNEL_DEFAULTS = {
    "qk": {
        "enabled": False,
        "feature_map": "identity",
        "sharing": "per_head",
        "apply": "phase_residual",
        "rank": 32,
        "mlp_hidden": 128,
    },
    "logit_bias": {
        "enabled": False,
        "feature_map": "identity",
        "sharing": "per_head",
        "rank": 32,
        "mlp_hidden": 128,
    },
}

V2_CHANNEL_DEFAULTS = {
    "qk": {
        "enabled": False,
        "application": "rotary",
        "geometry": "phase",
        "input": {
            "kind": "frozen_fourier",
            "basis_dim": None,
            "theta": None,
            "scalars": [],
            "normalization_extent": None,
        },
        "mapper": {
            "kind": "identity",
            "residual": False,
            "rank": 32,
            "hidden_dim": 128,
        },
        "output": {
            "amplitude_init": 0.1,
            "amplitude_max": 1.0,
            "amplitude_parameterization": "signed",
            "learn_amplitude": True,
            "learn_phase": True,
            "phase_scale": 1.0,
            "additive_normalization": "none",
            "additive_gain_init": 0.1,
            "additive_gain_max": 1.0,
            "learn_additive_gain": True,
            "scale_init": 1.0,
            "scale_max": 4.0,
            "scale_parameterization": "exp",
        },
        "conditioning": {
            "kind": "none",
            "source": "qk",
            "activation": "tanh",
            "hidden_dim": 64,
            "input_mode": "content",
            "network": "linear",
            "components": "phase",
            "head_coupling": "per_head_independent",
            "gate_init": 0.0,
            "target": "both",
            "coupling": "shared_trunk_separate_readouts",
            "phase_bound": 0.25,
            "pair_rank": 16,
            "position_mode": "relative_only",
            "num_profiles": 8,
            "router_hidden_dim": 64,
            "profile_init_std": 0.02,
            "num_frequencies": 16,
        },
        "qk_coupling": "shared",
        "head_coupling": "per_head_independent",
    },
    "logit_bias": {
        "enabled": False,
        "application": "logit_bias",
        "geometry": "scalar_curve",
        "input": {
            "kind": "frozen_fourier",
            "basis_dim": None,
            "theta": None,
            "scalars": [],
            "normalization_extent": None,
        },
        "mapper": {
            "kind": "identity",
            "residual": False,
            "rank": 32,
            "hidden_dim": 128,
        },
        "output": {
            "amplitude_init": 0.1,
            "amplitude_max": 1.0,
            "amplitude_parameterization": "signed",
            "learn_amplitude": True,
            "learn_phase": True,
            "phase_scale": 1.0,
            "additive_normalization": "none",
            "additive_gain_init": 0.1,
            "additive_gain_max": 1.0,
            "learn_additive_gain": True,
            "scale_init": 1.0,
            "scale_max": 4.0,
            "scale_parameterization": "exp",
        },
        "conditioning": {
            "kind": "none",
            "source": "qk",
            "activation": "tanh",
            "hidden_dim": 64,
            "input_mode": "content",
            "network": "linear",
            "components": "phase",
            "head_coupling": "per_head_independent",
            "gate_init": 0.0,
            "target": "both",
            "coupling": "shared_trunk_separate_readouts",
            "phase_bound": 0.25,
            "pair_rank": 16,
            "position_mode": "relative_only",
            "num_profiles": 8,
            "router_hidden_dim": 64,
            "profile_init_std": 0.02,
            "num_frequencies": 16,
        },
        "head_coupling": "per_head_independent",
    },
}

RESIDUAL_STREAM_DEFAULTS = {
    "enabled": False,
    "placement": "input",
    "source": "position_basis",
    "input": {
        "kind": "frozen_fourier",
        "basis_dim": None,
        "theta": None,
        "scalars": [],
    },
    "mapper": {
        "kind": "identity",
        "residual": False,
        "rank": 32,
        "hidden_dim": 128,
    },
    "gate_init": 0.0,
    "layer_shared": False,
}

ATTENTION_WRITE_DEFAULTS = {
    "enabled": False,
    "mode": "key_position",
    "input": {
        "kind": "frozen_fourier",
        "basis_dim": None,
        "theta": None,
        "scalars": [],
    },
    "mapper": {
        "kind": "identity",
        "residual": False,
        "rank": 32,
        "hidden_dim": 128,
    },
    "head_coupling": "per_head_independent",
    "gate_init": 0.0,
}


def normalize_position_content_config(
    content_dim: int = 64,
    coupling: str = "separate",
) -> dict:
    """Validate the dedicated low-rank content stream used by position modules."""
    content_dim = int(content_dim)
    if content_dim <= 0:
        raise ValueError("position_content_dim must be positive")
    if coupling not in POSITION_CONTENT_COUPLINGS:
        raise ValueError(
            "position_content_coupling must be 'shared' or 'separate'"
        )
    return {"dim": content_dim, "coupling": coupling}

_FEATURE_MAP_TO_MAPPER = {
    "identity": ("identity", False),
    "add_rope": ("euclidean_affine", False),
    "linear": ("linear", False),
    "low_rank": ("low_rank", True),
    "bottleneck_mlp": ("bottleneck_mlp", True),
    "mlp": ("mlp", True),
}

_SHARING_TO_HEAD_COUPLING = {
    "shared_head": "shared_head",
    "per_head": "per_head_independent",
    "full_dim": "per_head_joint",
}

_HEAD_COUPLING_TO_SHARING = {
    value: key for key, value in _SHARING_TO_HEAD_COUPLING.items()
}

_MAPPER_TO_FEATURE_MAP = {
    "identity": "identity",
    "euclidean_affine": "add_rope",
    "linear": "linear",
    "low_rank": "low_rank",
    "bottleneck_mlp": "bottleneck_mlp",
    "mlp": "mlp",
}


def deep_merge(base: dict, updates: dict) -> dict:
    """Recursively merge nested config dictionaries without aliasing defaults."""
    merged = copy.deepcopy(base)
    for key, value in updates.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _require_dict(name: str, value: Any) -> dict:
    if not isinstance(value, dict):
        raise TypeError(f"{name} must be a dict, got {type(value).__name__}")
    return value


def detect_channel_schema(channel_name: str, channel: dict) -> Literal[1, 2]:
    """Return 1 or 2. Reject mixed v1/v2 axis keys."""
    channel = _require_dict(channel_name, channel)
    keys = set(channel)
    has_v1 = bool(keys & V1_ONLY_KEYS)
    has_v2 = bool(keys & V2_ONLY_KEYS)
    if has_v1 and has_v2:
        mixed = sorted((keys & V1_ONLY_KEYS) | (keys & V2_ONLY_KEYS))
        raise ValueError(
            f"{channel_name} mixes legacy v1 and v2 keys {mixed}. "
            "Provide either a legacy channel (feature_map/sharing/...) or a v2 "
            "channel (application/geometry/mapper/...), not both."
        )
    if has_v2:
        return 2
    return 1


def _legacy_feature_map_to_mapper(feature_map: str) -> tuple[str, bool]:
    if feature_map not in _FEATURE_MAP_TO_MAPPER:
        raise ValueError(
            f"Unknown legacy feature_map {feature_map!r}; expected one of "
            f"{sorted(V1_FEATURE_MAPS)}"
        )
    return _FEATURE_MAP_TO_MAPPER[feature_map]


def resolve_basis_dim(
    head_coupling: str,
    *,
    model_dim: int,
    heads: int,
    head_dim: int | None = None,
) -> int:
    if head_dim is None:
        if model_dim % heads != 0:
            raise ValueError("model_dim must be divisible by heads.")
        head_dim = model_dim // heads
    if head_coupling == "per_head_joint":
        return int(model_dim)
    if head_coupling in {"shared_head", "per_head_independent"}:
        return int(head_dim)
    raise ValueError(
        f"Unknown head_coupling {head_coupling!r}; expected one of "
        f"{sorted(_SHARING_TO_HEAD_COUPLING.values())}"
    )


def upgrade_legacy_position_config(
    channel_name: str,
    raw_config: dict,
    *,
    model_dim: int,
    heads: int,
    rope_theta: float,
) -> dict:
    """Upgrade a complete legacy v1 channel dict to canonical v2."""
    raw_config = _require_dict(channel_name, raw_config)
    schema = detect_channel_schema(channel_name, raw_config)
    if schema != 1:
        raise ValueError(
            f"upgrade_legacy_position_config expects a v1 {channel_name} config."
        )

    allowed = V1_QK_KEYS if channel_name == "qk" else V1_LOGIT_KEYS
    unknown = set(raw_config) - allowed
    if unknown:
        raise ValueError(
            f"Unknown legacy {channel_name} config keys: {sorted(unknown)}"
        )

    defaults = V1_CHANNEL_DEFAULTS[channel_name]
    normalized = deep_merge(defaults, raw_config)
    if not isinstance(normalized["enabled"], bool):
        raise TypeError(f"{channel_name}.enabled must be a boolean")
    if normalized["feature_map"] not in V1_FEATURE_MAPS:
        raise ValueError(
            f"{channel_name}.feature_map must be one of {sorted(V1_FEATURE_MAPS)}, "
            f"got {normalized['feature_map']!r}"
        )
    if normalized["sharing"] not in V1_SHARING_MODES:
        raise ValueError(
            f"{channel_name}.sharing must be one of {sorted(V1_SHARING_MODES)}, "
            f"got {normalized['sharing']!r}"
        )
    if channel_name == "qk" and normalized["apply"] not in V1_QK_APPLY_MODES:
        raise ValueError(
            "qk.apply must be 'add' or 'phase_residual', "
            f"got {normalized['apply']!r}"
        )
    for key in ("rank", "mlp_hidden"):
        normalized[key] = int(normalized[key])
        if normalized[key] <= 0:
            raise ValueError(f"{channel_name}.{key} must be positive")

    mapper_kind, residual = _legacy_feature_map_to_mapper(normalized["feature_map"])
    head_coupling = _SHARING_TO_HEAD_COUPLING[normalized["sharing"]]
    basis_dim = resolve_basis_dim(
        head_coupling,
        model_dim=model_dim,
        heads=heads,
    )

    if channel_name == "qk":
        if normalized["apply"] == "add":
            application, geometry = "additive", "free"
        else:
            application, geometry = "rotary", "phase"
        upgraded = {
            "enabled": normalized["enabled"],
            "application": application,
            "geometry": geometry,
            "input": {
                "kind": "frozen_fourier",
                "basis_dim": basis_dim,
                "theta": None,
                "scalars": [],
            },
            "mapper": {
                "kind": mapper_kind,
                "residual": residual,
                "rank": normalized["rank"],
                "hidden_dim": normalized["mlp_hidden"],
            },
            "qk_coupling": "shared",
            "head_coupling": head_coupling,
        }
    else:
        upgraded = {
            "enabled": normalized["enabled"],
            "application": "logit_bias",
            "geometry": "scalar_curve",
            "input": {
                "kind": "frozen_fourier",
                "basis_dim": basis_dim,
                "theta": None,
                "scalars": [],
            },
            "mapper": {
                "kind": mapper_kind,
                "residual": residual,
                "rank": normalized["rank"],
                "hidden_dim": normalized["mlp_hidden"],
            },
            "head_coupling": head_coupling,
        }

    # Validate through the v2 normalizer (also fills any nested defaults).
    return normalize_position_config_v2(
        channel_name,
        upgraded,
        model_dim=model_dim,
        heads=heads,
        rope_theta=rope_theta,
    )


def normalize_position_config_v2(
    channel_name: str,
    raw_config: dict,
    *,
    model_dim: int,
    heads: int,
    rope_theta: float,
) -> dict:
    """Validate and resolve a v2 channel config to a JSON-safe canonical dict."""
    if channel_name not in {"qk", "logit_bias"}:
        raise ValueError(f"Unknown channel name: {channel_name!r}")
    raw_config = _require_dict(channel_name, raw_config)
    schema = detect_channel_schema(channel_name, raw_config)
    # enabled-only / empty dicts are schema-1 by detection; callers should upgrade.
    # Accept them here only when merging onto v2 defaults (no legacy axis keys).
    if schema == 1 and (set(raw_config) & V1_ONLY_KEYS):
        raise ValueError(
            f"normalize_position_config_v2 expects a v2 {channel_name} config."
        )

    allowed = V2_QK_KEYS if channel_name == "qk" else V2_LOGIT_KEYS
    unknown = set(raw_config) - allowed
    if unknown:
        raise ValueError(f"Unknown {channel_name} config keys: {sorted(unknown)}")
    if channel_name == "logit_bias" and "qk_coupling" in raw_config:
        raise ValueError("qk_coupling is invalid on the logit_bias channel.")

    normalized = deep_merge(V2_CHANNEL_DEFAULTS[channel_name], raw_config)
    if not isinstance(normalized["enabled"], bool):
        raise TypeError(f"{channel_name}.enabled must be a boolean")

    application = normalized["application"]
    geometry = normalized["geometry"]
    if channel_name == "qk":
        allowed_pairs = {
            ("additive", "free"),
            ("additive", "pair_normalized"),
            ("additive", "amplitude_phase"),
            ("rotary", "phase"),
            ("rotary", "projected_phase"),
            ("rotary", "unit_pair"),
            ("rotary", "scaled_phase"),
        }
        if (application, geometry) not in allowed_pairs:
            raise ValueError(
                f"Unsupported qk application/geometry pair "
                f"application={application!r}, geometry={geometry!r}. "
                f"Allowed pairs are {sorted(allowed_pairs)}."
            )
    else:
        if application != "logit_bias" or geometry != "scalar_curve":
            raise ValueError(
                f"Unsupported logit_bias application/geometry pair "
                f"application={application!r}, geometry={geometry!r}. "
                "This refactor ships only (logit_bias, scalar_curve)."
            )

    head_coupling = normalized["head_coupling"]
    if head_coupling not in _HEAD_COUPLING_TO_SHARING:
        raise ValueError(
            f"{channel_name}.head_coupling must be one of "
            f"{sorted(_HEAD_COUPLING_TO_SHARING)}, got {head_coupling!r}"
        )

    if channel_name == "qk":
        qk_coupling = normalized["qk_coupling"]
        allowed_coupling = {
            "shared",
            "shared_trunk_separate_readouts",
            "separate",
        }
        if qk_coupling not in allowed_coupling:
            raise ValueError(
                f"qk.qk_coupling must be one of {sorted(allowed_coupling)}, "
                f"got {qk_coupling!r}"
            )

    input_cfg = _require_dict(f"{channel_name}.input", normalized["input"])
    unknown_input = set(input_cfg) - V2_INPUT_KEYS
    if unknown_input:
        raise ValueError(
            f"Unknown {channel_name}.input keys: {sorted(unknown_input)}"
        )
    input_kind = input_cfg.get("kind", "frozen_fourier")
    allowed_input_kinds = {
        "frozen_fourier",
        "learned_temperature_fourier",
        "learned_frequency_fourier",
    }
    if input_kind not in allowed_input_kinds:
        raise ValueError(
            f"{channel_name}.input.kind={input_kind!r} is unsupported; "
            f"expected one of {sorted(allowed_input_kinds)}."
        )
    scalars = input_cfg.get("scalars", [])
    if scalars is None:
        scalars = []
    if not isinstance(scalars, list):
        raise TypeError(f"{channel_name}.input.scalars must be a list")
    allowed_scalars = {"position", "normalized_position", "log_position"}
    unknown_scalars = set(scalars) - allowed_scalars
    if unknown_scalars:
        raise ValueError(
            f"{channel_name}.input.scalars contains unsupported values "
            f"{sorted(unknown_scalars)}; expected a subset of "
            f"{sorted(allowed_scalars)}."
        )
    if len(set(scalars)) != len(scalars):
        raise ValueError(f"{channel_name}.input.scalars must not contain duplicates")
    theta = input_cfg.get("theta", None)
    if theta is not None:
        theta = float(theta)
        if theta <= 0:
            raise ValueError(f"{channel_name}.input.theta must be positive or null")
    # null inherits the model rope theta at construction time; store null.
    input_cfg = {
        "kind": input_kind,
        "basis_dim": input_cfg.get("basis_dim", None),
        "theta": theta,
        "scalars": scalars,
        "normalization_extent": input_cfg.get("normalization_extent"),
    }
    if input_cfg["normalization_extent"] is not None:
        input_cfg["normalization_extent"] = int(
            input_cfg["normalization_extent"]
        )
        if input_cfg["normalization_extent"] <= 0:
            raise ValueError(
                f"{channel_name}.input.normalization_extent must be positive"
            )

    mapper = _require_dict(f"{channel_name}.mapper", normalized["mapper"])
    unknown_mapper = set(mapper) - V2_MAPPER_KEYS
    if unknown_mapper:
        raise ValueError(
            f"Unknown {channel_name}.mapper keys: {sorted(unknown_mapper)}"
        )
    mapper_kind = mapper.get("kind", "identity")
    allowed_mappers = set(_MAPPER_TO_FEATURE_MAP)
    if mapper_kind not in allowed_mappers:
        raise ValueError(
            f"{channel_name}.mapper.kind must be one of {sorted(allowed_mappers)}, "
            f"got {mapper_kind!r}"
        )
    residual = bool(mapper.get("residual", False))
    rank = int(mapper.get("rank", 32))
    hidden_dim = int(mapper.get("hidden_dim", 128))
    if rank <= 0:
        raise ValueError(f"{channel_name}.mapper.rank must be positive")
    if hidden_dim <= 0:
        raise ValueError(f"{channel_name}.mapper.hidden_dim must be positive")

    if model_dim % heads != 0:
        raise ValueError("model_dim must be divisible by heads.")
    head_dim = model_dim // heads
    if head_dim % 2 != 0:
        raise ValueError("head_dim must be even for Fourier position channels.")

    basis_dim = input_cfg["basis_dim"]
    if basis_dim is None:
        basis_dim = resolve_basis_dim(
            head_coupling,
            model_dim=model_dim,
            heads=heads,
            head_dim=head_dim,
        )
    else:
        basis_dim = int(basis_dim)
    if basis_dim <= 0 or basis_dim % 2 != 0:
        raise ValueError(
            f"{channel_name}.input.basis_dim must be a positive even integer, "
            f"got {basis_dim}"
        )
    default_output_dim = resolve_basis_dim(
        head_coupling,
        model_dim=model_dim,
        heads=heads,
        head_dim=head_dim,
    )
    input_cfg["basis_dim"] = basis_dim

    # identity / residual mappers require matching input/output widths.
    mapper_output_dim = default_output_dim
    mapper_input_dim = basis_dim + len(scalars)
    output_preview = normalized["output"]
    fixed_position_pipeline = channel_name == "qk" and (
        (
            application == "additive"
            and geometry == "amplitude_phase"
            and not output_preview.get("learn_amplitude", True)
            and not output_preview.get("learn_phase", True)
        )
        or (
            application == "rotary"
            and geometry == "phase"
            and not output_preview.get("learn_phase", True)
        )
    )
    if (
        not fixed_position_pipeline
        and mapper_kind in {"identity", "euclidean_affine"}
        and mapper_input_dim != mapper_output_dim
    ):
        raise ValueError(
            f"{channel_name}: {mapper_kind} mapper requires matching input/output "
            f"dimensions ({mapper_input_dim} vs {mapper_output_dim})."
        )
    if (
        not fixed_position_pipeline
        and residual
        and mapper_input_dim != mapper_output_dim
    ):
        raise ValueError(
            f"{channel_name}: residual mapper requires matching input/output "
            f"dimensions ({mapper_input_dim} vs {mapper_output_dim})."
        )
    if mapper_kind in {"identity", "euclidean_affine"} and residual:
        raise ValueError(
            f"{channel_name}: mapper.kind={mapper_kind!r} does not support residual=true."
        )

    output_cfg = _require_dict(f"{channel_name}.output", normalized["output"])
    unknown_output = set(output_cfg) - V2_OUTPUT_KEYS
    if unknown_output:
        raise ValueError(
            f"Unknown {channel_name}.output keys: {sorted(unknown_output)}"
        )
    amplitude_init = float(output_cfg.get("amplitude_init", 0.1))
    if amplitude_init < 0:
        raise ValueError(f"{channel_name}.output.amplitude_init must be non-negative")
    amplitude_parameterization = output_cfg.get(
        "amplitude_parameterization", "signed"
    )
    if amplitude_parameterization not in {
        "signed",
        "softplus",
        "bounded_sigmoid",
    }:
        raise ValueError(
            f"{channel_name}.output.amplitude_parameterization must be "
            "'signed', 'softplus', or 'bounded_sigmoid'"
        )
    amplitude_max = float(output_cfg.get("amplitude_max", 1.0))
    if (
        amplitude_parameterization == "bounded_sigmoid"
        and amplitude_max <= amplitude_init
    ):
        raise ValueError(
            f"{channel_name}.output.amplitude_max must exceed amplitude_init"
        )
    learn_amplitude = output_cfg.get("learn_amplitude", True)
    learn_phase = output_cfg.get("learn_phase", True)
    if not isinstance(learn_amplitude, bool):
        raise TypeError(f"{channel_name}.output.learn_amplitude must be a boolean")
    if not isinstance(learn_phase, bool):
        raise TypeError(f"{channel_name}.output.learn_phase must be a boolean")
    if (
        (not learn_amplitude or not learn_phase)
        and not (
            channel_name == "qk"
            and (
                (application == "additive" and geometry == "amplitude_phase")
                or (
                    application == "rotary"
                    and geometry == "phase"
                    and normalized["conditioning"].get("kind")
                    in {"adaptive_gain", "rope_phase", "carrier_hypernetwork"}
                    and learn_amplitude
                )
            )
        )
    ):
        raise ValueError(
            f"{channel_name}.output learn_amplitude/learn_phase controls require "
            "qk application='additive', geometry='amplitude_phase'"
        )
    phase_scale = float(output_cfg.get("phase_scale", 1.0))
    if phase_scale <= 0:
        raise ValueError(f"{channel_name}.output.phase_scale must be positive")
    scale_init = float(output_cfg.get("scale_init", 1.0))
    if scale_init <= 0:
        raise ValueError(f"{channel_name}.output.scale_init must be positive")
    scale_max = float(output_cfg.get("scale_max", 4.0))
    scale_parameterization = output_cfg.get("scale_parameterization", "exp")
    if scale_parameterization == "bounded_log" and scale_max <= 1:
        raise ValueError(f"{channel_name}.output.scale_max must exceed 1")
    if scale_parameterization not in {"exp", "linear", "bounded_log"}:
        raise ValueError(
            f"{channel_name}.output.scale_parameterization must be "
            "'exp', 'linear', or 'bounded_log'"
        )
    additive_normalization = output_cfg.get("additive_normalization", "none")
    if additive_normalization not in {"none", "rms"}:
        raise ValueError(
            f"{channel_name}.output.additive_normalization must be 'none' or 'rms'"
        )
    additive_gain_init = float(output_cfg.get("additive_gain_init", 0.1))
    additive_gain_max = float(output_cfg.get("additive_gain_max", 1.0))
    if not 0 < additive_gain_init < additive_gain_max:
        raise ValueError(
            f"{channel_name}.output requires "
            "0 < additive_gain_init < additive_gain_max"
        )
    learn_additive_gain = output_cfg.get("learn_additive_gain", True)
    if not isinstance(learn_additive_gain, bool):
        raise TypeError(
            f"{channel_name}.output.learn_additive_gain must be a boolean"
        )
    if additive_normalization != "none" and not (
        channel_name == "qk" and application == "additive"
    ):
        raise ValueError(
            f"{channel_name}.output.additive_normalization requires "
            "qk application='additive'"
        )

    conditioning_cfg = _require_dict(
        f"{channel_name}.conditioning", normalized["conditioning"]
    )
    unknown_conditioning = set(conditioning_cfg) - V2_CONDITIONING_KEYS
    if unknown_conditioning:
        raise ValueError(
            f"Unknown {channel_name}.conditioning keys: "
            f"{sorted(unknown_conditioning)}"
        )
    conditioning_kind = conditioning_cfg.get("kind", "none")
    raw_conditioning = raw_config.get("conditioning", {})
    default_content_source = (
        "dedicated"
        if conditioning_kind in {
            "adaptive_gain",
            "additive_phase",
            "rope_phase",
            "carrier_hypernetwork",
        }
        else "qk"
    )
    allowed_conditioning = (
        {
            "none",
            "local_residual",
            "content_gate",
            "phase_rotation",
            "adaptive_gain",
            "additive_phase",
            "rope_phase",
            "carrier_hypernetwork",
        }
        if channel_name == "qk"
        else {"none", "inkling_table", "inkling_cosnet", "pairwise_low_rank"}
    )
    if conditioning_kind not in allowed_conditioning:
        raise ValueError(
            f"{channel_name}.conditioning.kind={conditioning_kind!r} is "
            f"unsupported; expected one of {sorted(allowed_conditioning)}."
        )
    conditioning = {
        "kind": conditioning_kind,
        "source": raw_conditioning.get("source", default_content_source),
        "activation": conditioning_cfg.get("activation", "tanh"),
        "hidden_dim": int(conditioning_cfg.get("hidden_dim", 64)),
        "input_mode": conditioning_cfg.get("input_mode", "content"),
        "network": conditioning_cfg.get("network", "linear"),
        "components": conditioning_cfg.get("components", "phase"),
        "head_coupling": conditioning_cfg.get(
            "head_coupling", "per_head_independent"
        ),
        "gate_init": float(conditioning_cfg.get("gate_init", 0.0)),
        "target": conditioning_cfg.get("target", "both"),
        "coupling": conditioning_cfg.get(
            "coupling", "shared_trunk_separate_readouts"
        ),
        "phase_bound": float(conditioning_cfg.get("phase_bound", 0.25)),
        "pair_rank": int(conditioning_cfg.get("pair_rank", 16)),
        "position_mode": conditioning_cfg.get(
            "position_mode", "relative_only"
        ),
        "num_profiles": int(conditioning_cfg.get("num_profiles", 8)),
        "router_hidden_dim": int(
            conditioning_cfg.get("router_hidden_dim", 64)
        ),
        "profile_init_std": float(
            conditioning_cfg.get("profile_init_std", 0.02)
        ),
        "num_frequencies": int(conditioning_cfg.get("num_frequencies", 16)),
    }
    if conditioning["source"] not in {"dedicated", "qk", "residual"}:
        raise ValueError(
            f"{channel_name}.conditioning.source must be 'dedicated'; "
            "'qk' and 'residual' are accepted only for legacy configs"
        )
    if channel_name != "qk" and conditioning["source"] == "residual":
        raise ValueError(
            f"{channel_name}.conditioning.source='residual' is only supported "
            "for the Q/K channel"
        )
    if conditioning["target"] not in {"q", "k", "both"}:
        raise ValueError(
            f"{channel_name}.conditioning.target must be 'q', 'k', or 'both'"
        )
    if conditioning["coupling"] not in {
        "shared",
        "shared_trunk_separate_readouts",
        *({"separate"} if conditioning_kind == "carrier_hypernetwork" else set()),
    }:
        raise ValueError(
            f"{channel_name}.conditioning.coupling must be 'shared' or "
            "'shared_trunk_separate_readouts'"
            + (
                ", or 'separate' for carrier_hypernetwork"
                if conditioning_kind == "carrier_hypernetwork"
                else ""
            )
        )
    if conditioning["input_mode"] not in {
        "content",
        "position",
        "content_position",
    }:
        raise ValueError(
            f"{channel_name}.conditioning.input_mode must be 'content', "
            "'position', or 'content_position'"
        )
    if conditioning["network"] not in {"linear", "silu_mlp", "swiglu_mlp"}:
        raise ValueError(
            f"{channel_name}.conditioning.network must be 'linear', "
            "'silu_mlp', or 'swiglu_mlp'"
        )
    if conditioning["components"] not in {"phase", "log_gain_phase"}:
        raise ValueError(
            f"{channel_name}.conditioning.components must be 'phase' or "
            "'log_gain_phase'"
        )
    if conditioning["head_coupling"] not in {
        "shared_head",
        "per_head_independent",
    }:
        raise ValueError(
            f"{channel_name}.conditioning.head_coupling must be 'shared_head' "
            "or 'per_head_independent'"
        )
    if conditioning["phase_bound"] <= 0:
        raise ValueError(
            f"{channel_name}.conditioning.phase_bound must be positive"
        )
    if conditioning["activation"] not in {
        "tanh",
        "gelu",
        "linear",
        "scaled_sigmoid",
    }:
        raise ValueError(
            f"{channel_name}.conditioning.activation is unsupported"
        )
    if (
        conditioning_kind == "content_gate"
        and conditioning["activation"] not in {"tanh", "scaled_sigmoid"}
    ):
        raise ValueError(
            f"{channel_name}: content_gate activation must be "
            "'tanh' or 'scaled_sigmoid'"
        )
    if (
        conditioning_kind == "local_residual"
        and conditioning["activation"] == "scaled_sigmoid"
    ):
        raise ValueError(
            f"{channel_name}: local_residual does not support scaled_sigmoid"
        )
    if conditioning_kind == "phase_rotation" and not (
        channel_name == "qk"
        and application == "additive"
        and geometry == "pair_normalized"
    ):
        raise ValueError(
            f"{channel_name}: phase_rotation conditioning requires "
            "application='additive', geometry='pair_normalized'"
        )
    if conditioning_kind == "adaptive_gain" and channel_name != "qk":
        raise ValueError("adaptive_gain conditioning is only valid for Q/K")
    if conditioning_kind == "additive_phase" and not (
        channel_name == "qk"
        and application == "additive"
        and geometry == "amplitude_phase"
    ):
        raise ValueError(
            "additive_phase conditioning requires additive amplitude_phase Q/K"
        )
    if conditioning_kind == "rope_phase" and not (
        channel_name == "qk" and application == "rotary"
    ):
        raise ValueError("rope_phase conditioning requires rotary Q/K")
    if conditioning_kind == "carrier_hypernetwork":
        valid_carrier = channel_name == "qk" and (
            (application == "additive" and geometry == "amplitude_phase")
            or (
                application == "rotary"
                and geometry in {"phase", "scaled_phase"}
            )
        )
        if not valid_carrier:
            raise ValueError(
                "carrier_hypernetwork requires additive amplitude_phase or "
                "rotary phase/scaled_phase Q/K"
            )
        if conditioning["source"] != "dedicated":
            raise ValueError(
                "carrier_hypernetwork uses the dedicated normalized content stream"
            )
    if (
        conditioning_kind == "content_gate"
        and conditioning["activation"] == "scaled_sigmoid"
        and not 0 < conditioning["gate_init"] < 2
    ):
        raise ValueError(
            f"{channel_name}: scaled_sigmoid gate_init must lie in (0, 2)"
        )
    for key in (
        "hidden_dim",
        "num_profiles",
        "router_hidden_dim",
        "num_frequencies",
        "pair_rank",
    ):
        if conditioning[key] <= 0:
            raise ValueError(f"{channel_name}.conditioning.{key} must be positive")
    if conditioning["profile_init_std"] <= 0:
        raise ValueError(
            f"{channel_name}.conditioning.profile_init_std must be positive"
        )
    if conditioning["position_mode"] not in {
        "relative_only",
        "query_absolute",
        "full_absolute",
    }:
        raise ValueError(
            f"{channel_name}.conditioning.position_mode must be "
            "'relative_only', 'query_absolute', or 'full_absolute'"
        )
    if (
        conditioning_kind != "none"
        and geometry == "amplitude_phase"
        and conditioning_kind not in {"additive_phase", "carrier_hypernetwork"}
        and (not learn_amplitude or not learn_phase)
    ):
        raise ValueError(
            f"{channel_name}: content conditioning requires both amplitude and "
            "phase learning to be enabled"
        )

    result = {
        "enabled": normalized["enabled"],
        "application": application,
        "geometry": geometry,
        "input": input_cfg,
        "mapper": {
            "kind": mapper_kind,
            "residual": residual,
            "rank": rank,
            "hidden_dim": hidden_dim,
        },
        "output": {
            "amplitude_init": amplitude_init,
            "amplitude_max": amplitude_max,
            "amplitude_parameterization": amplitude_parameterization,
            "learn_amplitude": learn_amplitude,
            "learn_phase": learn_phase,
            "phase_scale": phase_scale,
            "additive_normalization": additive_normalization,
            "additive_gain_init": additive_gain_init,
            "additive_gain_max": additive_gain_max,
            "learn_additive_gain": learn_additive_gain,
            "scale_init": scale_init,
            "scale_max": scale_max,
            "scale_parameterization": scale_parameterization,
        },
        "conditioning": conditioning,
        "head_coupling": head_coupling,
    }
    if channel_name == "qk":
        result["qk_coupling"] = normalized["qk_coupling"]
    # rope_theta is available for builders; unused beyond validation here.
    _ = rope_theta
    return result


def resolve_channel_config(
    channel_name: str,
    *,
    preset_fragment: dict | None,
    override: dict | None,
    model_dim: int,
    heads: int,
    rope_theta: float,
) -> tuple[dict, Literal[1, 2]]:
    """Resolve a channel from preset + override into canonical v2.

    Returns ``(canonical_v2, source_schema)``.
    """
    override = dict(override or {})
    preset_fragment = dict(preset_fragment or {})
    if not override and not preset_fragment:
        upgraded = upgrade_legacy_position_config(
            channel_name,
            copy.deepcopy(V1_CHANNEL_DEFAULTS[channel_name]),
            model_dim=model_dim,
            heads=heads,
            rope_theta=rope_theta,
        )
        return upgraded, 1

    preset_schema = (
        detect_channel_schema(channel_name, preset_fragment)
        if preset_fragment
        else None
    )
    override_schema = (
        detect_channel_schema(channel_name, override)
        if override
        else None
    )
    # Empty/enabled-only overrides are structurally ambiguous. Inherit the
    # preset schema so {"enabled": false} can disable a native-v2 preset.
    override_has_axes = bool(set(override) - {"enabled"})
    if preset_schema == 2 and not override_has_axes:
        schema = 2
    elif override_schema is not None:
        schema = override_schema
    elif preset_schema is not None:
        schema = preset_schema
    else:
        schema = 1

    if schema == 1:
        raw = deep_merge(V1_CHANNEL_DEFAULTS[channel_name], preset_fragment)
        raw = deep_merge(raw, override)
        return (
            upgrade_legacy_position_config(
                channel_name,
                raw,
                model_dim=model_dim,
                heads=heads,
                rope_theta=rope_theta,
            ),
            1,
        )

    # v2 override path: upgrade any v1 preset fragment first, then merge.
    base = copy.deepcopy(V2_CHANNEL_DEFAULTS[channel_name])
    if preset_fragment:
        if preset_schema == 1:
            preset_v2 = upgrade_legacy_position_config(
                channel_name,
                deep_merge(V1_CHANNEL_DEFAULTS[channel_name], preset_fragment),
                model_dim=model_dim,
                heads=heads,
                rope_theta=rope_theta,
            )
        else:
            preset_v2 = normalize_position_config_v2(
                channel_name,
                deep_merge(V2_CHANNEL_DEFAULTS[channel_name], preset_fragment),
                model_dim=model_dim,
                heads=heads,
                rope_theta=rope_theta,
            )
        base = deep_merge(base, preset_v2)
    merged = deep_merge(base, override)
    return (
        normalize_position_config_v2(
            channel_name,
            merged,
            model_dim=model_dim,
            heads=heads,
            rope_theta=rope_theta,
        ),
        2,
    )


def channel_theta(channel: dict, rope_theta: float) -> float:
    theta = channel["input"]["theta"]
    return float(rope_theta if theta is None else theta)


def legacy_position_run_tag(
    *,
    qk_v1: dict,
    logit_v1: dict,
    attn_impl: str,
) -> str:
    """Stable auto-tag for source-schema v1 configs (historical names)."""
    qk = qk_v1
    logit = logit_v1
    if not qk["enabled"] and not logit["enabled"]:
        return "rope-flex" if attn_impl == "flex" else "rope"

    tags = []
    for channel_name, channel in (("qk", qk), ("logit", logit)):
        if not channel["enabled"]:
            continue
        parts = [channel_name]
        if channel_name == "qk":
            parts.append(
                "phase" if channel["apply"] == "phase_residual" else "add"
            )
        parts.extend((channel["feature_map"], channel["sharing"]))
        if channel["feature_map"] in {"low_rank", "bottleneck_mlp"}:
            parts.append(f"r{channel['rank']}")
        elif channel["feature_map"] == "mlp":
            parts.append(f"m{channel['mlp_hidden']}")
        tags.append("-".join(parts))
    return "+".join(tags)


def v2_position_run_tag(
    *,
    qk: dict,
    logit_bias: dict,
    attn_impl: str,
) -> str:
    """Auto-tag for native v2 configs."""
    if not qk["enabled"] and not logit_bias["enabled"]:
        return "rope-flex" if attn_impl == "flex" else "rope"

    tags = []
    for channel_name, channel in (("qk", qk), ("logit", logit_bias)):
        if not channel["enabled"]:
            continue
        parts = [channel_name, channel["application"], channel["geometry"]]
        input_cfg = channel["input"]
        if input_cfg["kind"] != "frozen_fourier":
            parts.append(input_cfg["kind"].replace("_fourier", ""))
        if input_cfg["scalars"]:
            parts.append("scalars-" + "_".join(input_cfg["scalars"]))
        parts.append(channel["mapper"]["kind"])
        if channel_name == "qk":
            parts.append(channel["qk_coupling"])
        conditioning = channel["conditioning"]["kind"]
        if conditioning != "none":
            parts.append(conditioning)
        parts.append(channel["head_coupling"])
        mapper = channel["mapper"]
        if mapper["kind"] in {"low_rank", "bottleneck_mlp"}:
            parts.append(f"r{mapper['rank']}")
        elif mapper["kind"] == "mlp":
            parts.append(f"m{mapper['hidden_dim']}")
        tags.append("-".join(parts))
    return "+".join(tags)


def v2_to_legacy_tag_fields(channel_name: str, channel: dict) -> dict:
    """Best-effort reverse map used only when comparing tag semantics."""
    feature_map = _MAPPER_TO_FEATURE_MAP[channel["mapper"]["kind"]]
    sharing = _HEAD_COUPLING_TO_SHARING[channel["head_coupling"]]
    legacy = {
        "enabled": channel["enabled"],
        "feature_map": feature_map,
        "sharing": sharing,
        "rank": channel["mapper"]["rank"],
        "mlp_hidden": channel["mapper"]["hidden_dim"],
    }
    if channel_name == "qk":
        if channel["application"] == "additive":
            legacy["apply"] = "add"
        else:
            legacy["apply"] = "phase_residual"
    return legacy


def ensure_channel_v2(
    channel_name: str,
    channel: dict | None,
    *,
    model_dim: int,
    heads: int,
    rope_theta: float,
) -> dict:
    """Accept v1 or v2 channel dicts and return canonical v2."""
    channel = dict(channel or {})
    if not channel or detect_channel_schema(channel_name, channel) == 1:
        return upgrade_legacy_position_config(
            channel_name,
            channel,
            model_dim=model_dim,
            heads=heads,
            rope_theta=rope_theta,
        )
    return normalize_position_config_v2(
        channel_name,
        channel,
        model_dim=model_dim,
        heads=heads,
        rope_theta=rope_theta,
    )


def normalize_residual_stream_config(
    raw_config: dict | None,
    *,
    model_dim: int,
    heads: int,
    rope_theta: float,
) -> dict:
    """Normalize residual-stream absolute-position injection settings."""
    raw = _require_dict("residual_stream", dict(raw_config or {}))
    allowed = set(RESIDUAL_STREAM_DEFAULTS)
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"Unknown residual_stream keys: {sorted(unknown)}")
    normalized = deep_merge(RESIDUAL_STREAM_DEFAULTS, raw)
    if not isinstance(normalized["enabled"], bool):
        raise TypeError("residual_stream.enabled must be a boolean")
    placement = normalized["placement"]
    if placement not in {"input", "per_layer", "both"}:
        raise ValueError(
            "residual_stream.placement must be 'input', 'per_layer', or 'both'"
        )
    source = normalized["source"]
    if source not in {"position_basis", "learned_absolute"}:
        raise ValueError(
            "residual_stream.source must be 'position_basis' or 'learned_absolute'"
        )
    if not isinstance(normalized["layer_shared"], bool):
        raise TypeError("residual_stream.layer_shared must be a boolean")

    # Reuse the strict basis/mapper normalization with a joint model-dim output.
    probe = normalize_position_config_v2(
        "qk",
        {
            "enabled": True,
            "application": "additive",
            "geometry": "free",
            "input": normalized["input"],
            "mapper": normalized["mapper"],
            "qk_coupling": "shared",
            "head_coupling": "per_head_joint",
        },
        model_dim=model_dim,
        heads=heads,
        rope_theta=rope_theta,
    )
    return {
        "enabled": normalized["enabled"],
        "placement": placement,
        "source": source,
        "input": probe["input"],
        "mapper": probe["mapper"],
        "gate_init": float(normalized["gate_init"]),
        "layer_shared": normalized["layer_shared"],
    }


def normalize_attention_write_config(
    raw_config: dict | None,
    *,
    model_dim: int,
    heads: int,
    rope_theta: float,
) -> dict:
    """Normalize attended key-position / relative-offset write settings."""
    raw = _require_dict("attention_write", dict(raw_config or {}))
    allowed = set(ATTENTION_WRITE_DEFAULTS)
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"Unknown attention_write keys: {sorted(unknown)}")
    normalized = deep_merge(ATTENTION_WRITE_DEFAULTS, raw)
    if not isinstance(normalized["enabled"], bool):
        raise TypeError("attention_write.enabled must be a boolean")
    mode = normalized["mode"]
    if mode not in {"key_position", "relative_offset", "query_position"}:
        raise ValueError(
            "attention_write.mode must be 'key_position', 'relative_offset', "
            "or 'query_position'"
        )
    head_coupling = normalized["head_coupling"]
    probe = normalize_position_config_v2(
        "qk",
        {
            "enabled": True,
            "application": "additive",
            "geometry": "free",
            "input": normalized["input"],
            "mapper": normalized["mapper"],
            "qk_coupling": "shared",
            "head_coupling": head_coupling,
        },
        model_dim=model_dim,
        heads=heads,
        rope_theta=rope_theta,
    )
    if mode == "relative_offset" and probe["input"]["scalars"]:
        raise ValueError(
            "attention_write relative_offset mode requires pure paired Fourier "
            "features; scalar inputs cannot be translated by the Fourier identity."
        )
    return {
        "enabled": normalized["enabled"],
        "mode": mode,
        "input": probe["input"],
        "mapper": probe["mapper"],
        "head_coupling": head_coupling,
        "gate_init": float(normalized["gate_init"]),
    }
