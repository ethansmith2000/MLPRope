"""Canonical v2 position-channel configuration and v1 upgrade helpers."""

from __future__ import annotations

import copy
from typing import Any, Literal

POSITION_SCHEMA_VERSION = 2

Application = Literal["additive", "rotary", "logit_bias"]
Geometry = Literal["free", "phase", "scalar_curve"]
InputKind = Literal["frozen_fourier"]
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
    "head_coupling",
}
V2_QK_KEYS = V2_COMMON_KEYS | {"qk_coupling"}
V2_LOGIT_KEYS = V2_COMMON_KEYS
V2_INPUT_KEYS = {"kind", "basis_dim", "theta", "scalars"}
V2_MAPPER_KEYS = {"kind", "residual", "rank", "hidden_dim"}

# Legacy v1 names that must not appear on a v2 channel (except enabled).
V1_ONLY_KEYS = {"feature_map", "sharing", "apply", "rank", "mlp_hidden"}
# v2 axis keys that must not appear on a v1 channel.
V2_ONLY_KEYS = {
    "application",
    "geometry",
    "input",
    "mapper",
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
        },
        "mapper": {
            "kind": "identity",
            "residual": False,
            "rank": 32,
            "hidden_dim": 128,
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
        },
        "mapper": {
            "kind": "identity",
            "residual": False,
            "rank": 32,
            "hidden_dim": 128,
        },
        "head_coupling": "per_head_independent",
    },
}

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
        allowed_pairs = {("additive", "free"), ("rotary", "phase")}
        if (application, geometry) not in allowed_pairs:
            raise ValueError(
                f"Unsupported qk application/geometry pair "
                f"application={application!r}, geometry={geometry!r}. "
                "This refactor ships only (additive, free) and (rotary, phase)."
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
    if input_cfg.get("kind", "frozen_fourier") != "frozen_fourier":
        raise ValueError(
            f"{channel_name}.input.kind={input_cfg.get('kind')!r} is unsupported; "
            "this refactor ships only kind='frozen_fourier'."
        )
    scalars = input_cfg.get("scalars", [])
    if scalars not in ([], None):
        raise ValueError(
            f"{channel_name}.input.scalars must be empty in this refactor; "
            f"got {scalars!r}"
        )
    theta = input_cfg.get("theta", None)
    if theta is not None:
        theta = float(theta)
        if theta <= 0:
            raise ValueError(f"{channel_name}.input.theta must be positive or null")
    # null inherits the model rope theta at construction time; store null.
    input_cfg = {
        "kind": "frozen_fourier",
        "basis_dim": input_cfg.get("basis_dim", None),
        "theta": theta,
        "scalars": [],
    }

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
    expected = resolve_basis_dim(
        head_coupling,
        model_dim=model_dim,
        heads=heads,
        head_dim=head_dim,
    )
    if basis_dim != expected:
        raise ValueError(
            f"{channel_name}.input.basis_dim={basis_dim} is incompatible with "
            f"head_coupling={head_coupling!r}; expected {expected} in this refactor."
        )
    input_cfg["basis_dim"] = basis_dim

    # identity / residual mappers require matching input/output widths.
    mapper_output_dim = head_dim if head_coupling != "per_head_joint" else model_dim
    mapper_input_dim = basis_dim
    if mapper_kind == "identity" and mapper_input_dim != mapper_output_dim:
        raise ValueError(
            f"{channel_name}: identity mapper requires matching input/output "
            f"dimensions ({mapper_input_dim} vs {mapper_output_dim})."
        )
    if residual and mapper_input_dim != mapper_output_dim:
        raise ValueError(
            f"{channel_name}: residual mapper requires matching input/output "
            f"dimensions ({mapper_input_dim} vs {mapper_output_dim})."
        )
    if mapper_kind in {"identity", "euclidean_affine"} and residual:
        raise ValueError(
            f"{channel_name}: mapper.kind={mapper_kind!r} does not support residual=true."
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

    # If only preset is present, treat as v1.
    probe = override if override else preset_fragment
    schema = detect_channel_schema(channel_name, probe)

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
        preset_schema = detect_channel_schema(channel_name, preset_fragment)
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
        parts.append(channel["mapper"]["kind"])
        if channel_name == "qk":
            parts.append(channel["qk_coupling"])
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
