import math
from typing import Literal

import torch
import torch.nn.functional as F
from torch.nn.attention.flex_attention import (
    BlockMask,
    create_block_mask,
    flex_attention as _flex_attention_op,
)

# Inductor's default FlexAttention tiles need ~120KiB SMEM; this GPU reports a
# 101376 cap, so pin small tiles. Compile flex alone (not nested under the outer
# model compile) while allowing graph breaks in the outer model.
_FLEX_KERNEL_OPTIONS = {
    "BLOCK_M": 32,
    "BLOCK_N": 32,
    "BLOCK_M1": 32,
    "BLOCK_N1": 32,
    "BLOCK_M2": 32,
    "BLOCK_N2": 32,
    "num_stages": 2,
    "num_warps": 4,
}
_compiled_flex_attention = None


def _flex_attention_call(*args, **kwargs):
    global _compiled_flex_attention
    kwargs.setdefault("kernel_options", _FLEX_KERNEL_OPTIONS)
    if _compiled_flex_attention is None:
        # Prefer default over max-autotune: the latter enables CUDAGraphs, which
        # fail the training fast-path ("previous outputs still require backward")
        # and step time balloons (~1s -> 5s). Pinned kernel_options already keep
        # tiles under the SMEM limit.
        _compiled_flex_attention = torch.compile(
            _flex_attention_op,
            mode="default",
            fullgraph=True,
        )
    return _compiled_flex_attention(*args, **kwargs)


PositionVariant = Literal[
    "rope",
    "add_rope",
    "linear",
    "low_rank",
    "bottleneck_mlp",
    "mlp_rope",
    "inkling_table",
    "inkling_cosnet",
]
FeatureMapName = Literal[
    "identity",
    "add_rope",
    "linear",
    "low_rank",
    "bottleneck_mlp",
    "mlp",
]
SharingMode = Literal["shared_head", "per_head", "full_dim"]
QKApply = Literal["add", "phase_residual"]
AttentionImpl = Literal["sdpa", "flex"]

FEATURE_MAPS = {
    "identity",
    "add_rope",
    "linear",
    "low_rank",
    "bottleneck_mlp",
    "mlp",
}
SHARING_MODES = {"shared_head", "per_head", "full_dim"}
QK_APPLY_MODES = {"add", "phase_residual"}
POSITION_ONLY_VARIANTS = {
    "add_rope",
    "linear",
    "low_rank",
    "bottleneck_mlp",
    "mlp_rope",
}
CONTENT_CONDITIONED_VARIANTS = {"inkling_table", "inkling_cosnet"}
POSITION_VARIANTS = {"rope"} | POSITION_ONLY_VARIANTS | CONTENT_CONDITIONED_VARIANTS


def causal_mask(batch_idx, head_idx, query_idx, key_value_idx):
    del batch_idx, head_idx
    return query_idx >= key_value_idx


def sinusoidal_basis(
    extent: int,
    feature_dim: int,
    theta: float,
) -> torch.Tensor:
    """Return interleaved cosine/sine features for non-negative distances."""
    if feature_dim % 2 != 0:
        raise ValueError("Position-bias feature_dim must be even.")
    half = feature_dim // 2
    frequencies = torch.arange(half, dtype=torch.float32)
    inverse_frequencies = 1.0 / (theta ** (frequencies / half))
    distances = torch.arange(extent, dtype=torch.float32)
    angles = torch.outer(distances, inverse_frequencies)
    return torch.stack((angles.cos(), angles.sin()), dim=-1).flatten(-2)


class PositionChannel(torch.nn.Module):
    """Marker base class for position-channel parameter accounting and resets."""

    def reset_output_parameters(self) -> None:
        raise NotImplementedError


class PositionFeatureMap(torch.nn.Module):
    """Map a sinusoidal basis to [heads, extent, head_dim] features."""

    def __init__(
        self,
        *,
        name: FeatureMapName,
        sharing: SharingMode,
        heads: int,
        head_dim: int,
        extent: int,
        theta: float,
        rank: int,
        mlp_hidden: int,
    ):
        super().__init__()
        if name not in FEATURE_MAPS:
            raise ValueError(f"Unknown position feature_map: {name!r}")
        if sharing not in SHARING_MODES:
            raise ValueError(f"Unknown position sharing mode: {sharing!r}")
        if rank <= 0:
            raise ValueError("position rank must be positive.")
        if mlp_hidden <= 0:
            raise ValueError("position mlp_hidden must be positive.")

        self.name = name
        self.sharing = sharing
        self.heads = heads
        self.head_dim = head_dim
        self.extent = extent
        self.feature_dim = heads * head_dim if sharing == "full_dim" else head_dim
        self.groups = heads if sharing == "per_head" else 1
        self.hidden_dim = rank if name in {"low_rank", "bottleneck_mlp"} else mlp_hidden
        self.register_buffer(
            "basis",
            sinusoidal_basis(extent, self.feature_dim, theta),
            persistent=False,
        )

        if name == "add_rope":
            self.scale = torch.nn.Parameter(
                torch.zeros(self.groups, self.feature_dim)
            )
            self.offset = torch.nn.Parameter(
                torch.zeros(self.groups, self.feature_dim)
            )
        elif name == "linear":
            self.weight = torch.nn.Parameter(
                torch.empty(self.groups, self.feature_dim, self.feature_dim)
            )
            self.bias = torch.nn.Parameter(
                torch.zeros(self.groups, self.feature_dim)
            )
        elif name in {"low_rank", "bottleneck_mlp", "mlp"}:
            self.down = torch.nn.Parameter(
                torch.empty(self.groups, self.feature_dim, self.hidden_dim)
            )
            self.down_bias = torch.nn.Parameter(
                torch.zeros(self.groups, self.hidden_dim)
            )
            self.up = torch.nn.Parameter(
                torch.zeros(self.groups, self.hidden_dim, self.feature_dim)
            )
            self.up_bias = torch.nn.Parameter(
                torch.zeros(self.groups, self.feature_dim)
            )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.name == "add_rope":
            torch.nn.init.zeros_(self.scale)
            torch.nn.init.zeros_(self.offset)
        elif self.name == "linear":
            for weight in self.weight:
                torch.nn.init.xavier_normal_(weight)
            torch.nn.init.zeros_(self.bias)
        elif self.name in {"low_rank", "bottleneck_mlp", "mlp"}:
            for weight in self.down:
                torch.nn.init.xavier_normal_(weight)
            torch.nn.init.zeros_(self.down_bias)
            # A zero residual branch makes the feature map exactly identity.
            torch.nn.init.zeros_(self.up)
            torch.nn.init.zeros_(self.up_bias)

    def forward(self, *, dtype: torch.dtype | None = None) -> torch.Tensor:
        basis = self.basis if dtype is None else self.basis.to(dtype=dtype)
        grouped_basis = basis.unsqueeze(0).expand(self.groups, -1, -1)

        if self.name == "identity":
            mapped = grouped_basis
        elif self.name == "add_rope":
            mapped = (
                grouped_basis * (1.0 + self.scale[:, None, :])
                + self.offset[:, None, :]
            )
        elif self.name == "linear":
            mapped = torch.einsum(
                "grd,gde->gre", grouped_basis, self.weight
            )
            mapped = mapped + self.bias[:, None, :]
        else:
            hidden = torch.einsum(
                "grd,gdk->grk", grouped_basis, self.down
            )
            hidden = hidden + self.down_bias[:, None, :]
            if self.name in {"bottleneck_mlp", "mlp"}:
                hidden = F.gelu(hidden)
            delta = torch.einsum("grk,gkd->grd", hidden, self.up)
            mapped = grouped_basis + delta + self.up_bias[:, None, :]

        if self.sharing == "per_head":
            return mapped
        if self.sharing == "shared_head":
            return mapped.expand(self.heads, -1, -1)
        # A full-dimension map can mix frequencies across heads before reshape.
        return (
            mapped.squeeze(0)
            .reshape(self.extent, self.heads, self.head_dim)
            .permute(1, 0, 2)
            .contiguous()
        )


class LogitBiasChannel(PositionChannel):
    """Position-only scalar logit curves with a fixed [heads, extent] contract."""

    def __init__(
        self,
        *,
        feature_map: FeatureMapName,
        sharing: SharingMode,
        heads: int,
        head_dim: int,
        extent: int,
        theta: float,
        rank: int,
        mlp_hidden: int,
    ):
        super().__init__()
        self.heads = heads
        self.extent = extent
        self.sharing = sharing
        self.features = PositionFeatureMap(
            name=feature_map,
            sharing=sharing,
            heads=heads,
            head_dim=head_dim,
            extent=extent,
            theta=theta,
            rank=rank,
            mlp_hidden=mlp_hidden,
        )
        readout_groups = 1 if sharing == "shared_head" else heads
        self.readout = torch.nn.Parameter(
            torch.zeros(readout_groups, head_dim)
        )
        self.readout_bias = torch.nn.Parameter(torch.zeros(readout_groups))

    def forward(self, *, dtype: torch.dtype | None = None) -> torch.Tensor:
        features = self.features(dtype=dtype)
        if self.sharing == "shared_head":
            curve = torch.einsum("rd,d->r", features[0], self.readout[0])
            curve = curve + self.readout_bias[0]
            return curve.unsqueeze(0).expand(self.heads, -1)
        curves = torch.einsum("hrd,hd->hr", features, self.readout)
        return curves + self.readout_bias[:, None]

    def reset_output_parameters(self) -> None:
        torch.nn.init.zeros_(self.readout)
        torch.nn.init.zeros_(self.readout_bias)


class QKPositionChannel(PositionChannel):
    """Vector or phase residual composed with the standard RoPE Q/K path."""

    def __init__(
        self,
        *,
        feature_map: FeatureMapName,
        sharing: SharingMode,
        apply: QKApply,
        heads: int,
        head_dim: int,
        extent: int,
        theta: float,
        rank: int,
        mlp_hidden: int,
    ):
        super().__init__()
        if apply not in QK_APPLY_MODES:
            raise ValueError(f"Unknown Q/K position apply mode: {apply!r}")
        self.heads = heads
        self.head_dim = head_dim
        self.extent = extent
        self.sharing = sharing
        self.apply = apply
        self.features = PositionFeatureMap(
            name=feature_map,
            sharing=sharing,
            heads=heads,
            head_dim=head_dim,
            extent=extent,
            theta=theta,
            rank=rank,
            mlp_hidden=mlp_hidden,
        )
        output_dim = head_dim if apply == "add" else head_dim // 2
        output_groups = 1 if sharing == "shared_head" else heads
        self.output_weight = torch.nn.Parameter(
            torch.zeros(output_groups, head_dim, output_dim)
        )
        self.output_bias = torch.nn.Parameter(
            torch.zeros(output_groups, output_dim)
        )

    def forward(
        self,
        sequence_length: int,
        *,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        if sequence_length > self.extent:
            raise ValueError(
                f"Sequence length {sequence_length} exceeds Q/K position extent "
                f"{self.extent}."
            )
        features = self.features(dtype=dtype)[:, :sequence_length]
        if self.sharing == "shared_head":
            output = torch.einsum(
                "rd,do->ro", features[0], self.output_weight[0]
            )
            output = output + self.output_bias[0]
            return output.unsqueeze(0).expand(self.heads, -1, -1)
        output = torch.einsum(
            "hrd,hdo->hro", features, self.output_weight
        )
        return output + self.output_bias[:, None, :]

    def reset_output_parameters(self) -> None:
        # Additive delta=0; phase delta=0 => cos(delta)=1, sin(delta)=0.
        torch.nn.init.zeros_(self.output_weight)
        torch.nn.init.zeros_(self.output_bias)


class Attention(torch.nn.Module):
    def __init__(
        self,
        dim,
        heads,
        is_causal=True,
        use_rope=True,
        rope_theta=10000.0,
        max_seq_len=2048,
        qk_norm=True,
        rel_extent: int | None = None,
        qk_config: dict | None = None,
        logit_bias_config: dict | None = None,
        attn_impl: AttentionImpl = "sdpa",
    ):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.is_causal = is_causal
        self.use_rope = use_rope
        self.head_dim = dim // heads
        self.rope_theta = rope_theta
        self.max_seq_len = max_seq_len
        self.rel_extent = rel_extent or max_seq_len
        self.attn_impl = attn_impl
        self.qk_config = dict(qk_config or {})
        self.logit_bias_config = dict(logit_bias_config or {})
        self._prepared_block_mask: BlockMask | None = None
        self._prepared_query_length: int | None = None
        # Stable score_mod closure target — updated each flex forward, not redefined.
        self._flex_bias_curves: torch.Tensor | None = None

        if dim % heads != 0:
            raise ValueError("dim must be divisible by heads.")
        if not use_rope:
            raise ValueError("These experiments keep RoPE enabled as the geometric prior.")
        if self.head_dim % 2 != 0:
            raise ValueError("RoPE requires an even head dimension.")
        if attn_impl not in ("sdpa", "flex"):
            raise ValueError(f"Unknown attn_impl: {attn_impl!r}")
        if self.logit_bias_config.get("enabled", False) and attn_impl != "flex":
            raise ValueError(
                "The logit-bias channel requires attn_impl='flex'. "
                "Q/K-only position channels may use SDPA."
            )
        if self.rel_extent <= 0:
            raise ValueError("rel_extent must be positive.")

        # Split projections match the other experimental Transformer.
        self.to_q = torch.nn.Linear(dim, dim, bias=False)
        self.to_k = torch.nn.Linear(dim, dim, bias=False)
        self.to_v = torch.nn.Linear(dim, dim, bias=False)
        self.to_out = torch.nn.Linear(dim, dim, bias=True)

        self.qk_position = None
        if self.qk_config.get("enabled", False):
            self.qk_position = QKPositionChannel(
                feature_map=self.qk_config["feature_map"],
                sharing=self.qk_config["sharing"],
                apply=self.qk_config["apply"],
                heads=heads,
                head_dim=self.head_dim,
                extent=max_seq_len,
                theta=rope_theta,
                rank=self.qk_config["rank"],
                mlp_hidden=self.qk_config["mlp_hidden"],
            )
        self.logit_bias = None
        if self.logit_bias_config.get("enabled", False):
            self.logit_bias = LogitBiasChannel(
                feature_map=self.logit_bias_config["feature_map"],
                sharing=self.logit_bias_config["sharing"],
                heads=heads,
                head_dim=self.head_dim,
                extent=self.rel_extent,
                theta=rope_theta,
                rank=self.logit_bias_config["rank"],
                mlp_hidden=self.logit_bias_config["mlp_hidden"],
            )

        if qk_norm:
            self.q_norm = torch.nn.LayerNorm(self.head_dim)
            self.k_norm = torch.nn.LayerNorm(self.head_dim)
        else:
            self.q_norm = torch.nn.Identity()
            self.k_norm = torch.nn.Identity()

        half = self.head_dim // 2
        freqs = torch.arange(half, dtype=torch.float32)
        inv_freq = 1.0 / (self.rope_theta ** (freqs / half))
        positions = torch.arange(max_seq_len, dtype=torch.float32)
        angles = torch.outer(positions, inv_freq)
        self.register_buffer("rope_sin", angles.sin(), persistent=False)
        self.register_buffer("rope_cos", angles.cos(), persistent=False)

    def _apply_rope(self, q, k, phase_delta=None):
        seq_len = q.shape[-2]
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds RoPE cache length {self.max_seq_len}"
            )
        half = q.shape[-1] // 2
        sin = self.rope_sin[:seq_len].to(dtype=q.dtype)[None, None, :, :]
        cos = self.rope_cos[:seq_len].to(dtype=q.dtype)[None, None, :, :]
        if phase_delta is not None:
            delta = phase_delta.to(dtype=q.dtype)[None, :, :, :]
            delta_sin = delta.sin()
            delta_cos = delta.cos()
            # R(theta + delta) = R(theta) R(delta).
            sin, cos = (
                sin * delta_cos + cos * delta_sin,
                cos * delta_cos - sin * delta_sin,
            )

        def rotate(x):
            x1, x2 = x[..., :half], x[..., half:]
            return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)

        return rotate(q), rotate(k)

    def _split_heads(self, x):
        batch, sequence, _ = x.shape
        return x.view(batch, sequence, self.heads, self.head_dim).transpose(1, 2)

    def prepare_flex_mask(
        self,
        query_length: int,
        device: torch.device | str,
    ) -> None:
        if self.attn_impl != "flex":
            return
        self._prepared_block_mask = create_block_mask(
            causal_mask,
            B=None,
            H=None,
            Q_LEN=query_length,
            KV_LEN=query_length,
            device=device,
        )
        self._prepared_query_length = query_length

    def _block_mask(self, q: torch.Tensor) -> BlockMask:
        query_length = q.shape[-2]
        if (
            self._prepared_block_mask is not None
            and self._prepared_query_length == query_length
        ):
            return self._prepared_block_mask
        return create_block_mask(
            causal_mask,
            B=None,
            H=None,
            Q_LEN=query_length,
            KV_LEN=query_length,
            device=q.device,
        )

    def _position_bias_score_mod(self, score, batch_idx, head_idx, query_idx, key_value_idx):
        del batch_idx
        bias_curves = self._flex_bias_curves
        distance = query_idx - key_value_idx
        in_range = (distance >= 0) & (distance < self.rel_extent)
        distance = distance.clamp(0, self.rel_extent - 1)
        bias = bias_curves[head_idx, distance]
        return score + torch.where(in_range, bias, 0.0)

    @torch.compiler.disable
    def _flex_attention(self, q, k, v):
        block_mask = self._block_mask(q)
        if self.logit_bias is None:
            return _flex_attention_call(q, k, v, block_mask=block_mask)

        self._flex_bias_curves = self.logit_bias(dtype=q.dtype)
        return _flex_attention_call(
            q,
            k,
            v,
            score_mod=self._position_bias_score_mod,
            block_mask=block_mask,
        )

    def forward(self, x):
        q = self._split_heads(self.to_q(x))
        k = self._split_heads(self.to_k(x))
        v = self._split_heads(self.to_v(x))

        q = self.q_norm(q)
        k = self.k_norm(k)
        phase_delta = None
        if self.qk_position is not None:
            position_output = self.qk_position(q.shape[-2], dtype=q.dtype)
            if self.qk_position.apply == "add":
                position_output = position_output[None, :, :, :]
                q = q + position_output
                k = k + position_output
            else:
                phase_delta = position_output
        q, k = self._apply_rope(q, k, phase_delta=phase_delta)
        if self.attn_impl == "flex":
            attn = self._flex_attention(q, k, v)
        else:
            attn = F.scaled_dot_product_attention(
                q,
                k,
                v,
                is_causal=self.is_causal,
            )
        attn = attn.transpose(1, 2).contiguous().view(x.shape[0], x.shape[1], -1)
        return self.to_out(attn)

    @torch.no_grad()
    def logit_bias_curves(self) -> torch.Tensor | None:
        if self.logit_bias is None:
            return None
        parameter = next(self.logit_bias.parameters())
        return self.logit_bias(dtype=parameter.dtype).float().detach()


class GeGLU(torch.nn.Module):
    def __init__(
        self,
        dim,
        hidden_dim=None,
        align_multiple=64,
    ):
        super().__init__()
        hidden_dim = hidden_dim or math.ceil(dim * 8 / 3)
        if align_multiple is not None and align_multiple > 1:
            hidden_dim = math.ceil(hidden_dim / align_multiple) * align_multiple
        self.proj_in = torch.nn.Linear(dim, hidden_dim * 2, bias=True)
        self.proj_out = torch.nn.Linear(hidden_dim, dim, bias=True)
        self.act = torch.nn.GELU()

    def forward(self, x):
        value, gate = self.proj_in(x).chunk(2, dim=-1)
        return self.proj_out(value * self.act(gate))


class TransformerBlock(torch.nn.Module):
    def __init__(
        self,
        dim,
        heads,
        ff_hidden_dim,
        is_causal=True,
        use_rope=True,
        rope_theta=10000.0,
        max_seq_len=2048,
        qk_norm=True,
        rel_extent: int | None = None,
        qk_config: dict | None = None,
        logit_bias_config: dict | None = None,
        attn_impl: AttentionImpl = "sdpa",
    ):
        super().__init__()
        self.attn = Attention(
            dim,
            heads,
            is_causal=is_causal,
            use_rope=use_rope,
            rope_theta=rope_theta,
            max_seq_len=max_seq_len,
            qk_norm=qk_norm,
            rel_extent=rel_extent,
            qk_config=qk_config,
            logit_bias_config=logit_bias_config,
            attn_impl=attn_impl,
        )
        self.ff = GeGLU(dim, hidden_dim=ff_hidden_dim)
        self.norm1 = torch.nn.LayerNorm(dim)
        self.norm2 = torch.nn.LayerNorm(dim)

    def forward(self, x):
        x = self.attn(self.norm1(x)) + x
        x = self.ff(self.norm2(x)) + x
        return x


class Transformer(torch.nn.Module):
    def __init__(
        self,
        dim,
        depth,
        heads,
        ff_mult,
        vocab_size,
        max_seq_len,
        gradient_checkpointing=False,
        use_rope=True,
        rope_theta=10000.0,
        qk_norm=True,
        rel_extent: int | None = None,
        qk_config: dict | None = None,
        logit_bias_config: dict | None = None,
        attn_impl: AttentionImpl = "sdpa",
    ):
        super().__init__()

        self.token_embedding = torch.nn.Embedding(vocab_size, dim)
        self.gradient_checkpointing = gradient_checkpointing
        self.blocks = torch.nn.ModuleList([
            TransformerBlock(
                dim,
                heads,
                dim * ff_mult,
                is_causal=True,
                use_rope=use_rope,
                rope_theta=rope_theta,
                max_seq_len=max_seq_len,
                qk_norm=qk_norm,
                rel_extent=rel_extent,
                qk_config=qk_config,
                logit_bias_config=logit_bias_config,
                attn_impl=attn_impl,
            )
            for _ in range(depth)
        ])
        self.in_proj = torch.nn.Sequential(
            torch.nn.LayerNorm(dim),
            torch.nn.Linear(dim, dim, bias=True),
        )
        self.out_proj = torch.nn.Sequential(
            torch.nn.LayerNorm(dim),
            torch.nn.Linear(dim, vocab_size, bias=True),
        )
        self._init_weights(dim)

    def _init_weights(self, dim):
        embed_std = dim ** -0.5
        lm_head_std = 0.02
        with torch.no_grad():
            for module in self.modules():
                if isinstance(module, torch.nn.Embedding):
                    torch.nn.init.normal_(module.weight, mean=0.0, std=embed_std)
                elif isinstance(module, torch.nn.Linear):
                    torch.nn.init.xavier_normal_(module.weight)
                    if module.bias is not None:
                        torch.nn.init.zeros_(module.bias)
                elif isinstance(module, torch.nn.LayerNorm):
                    torch.nn.init.ones_(module.weight)
                    torch.nn.init.zeros_(module.bias)
            lm_head = self.out_proj[1]
            torch.nn.init.normal_(lm_head.weight, mean=0.0, std=lm_head_std)
            if lm_head.bias is not None:
                torch.nn.init.zeros_(lm_head.bias)
            for module in self.modules():
                if isinstance(module, PositionChannel):
                    module.reset_output_parameters()

    def prepare_flex_masks(
        self,
        query_length: int,
        device: torch.device | str,
    ) -> None:
        for block in self.blocks:
            block.attn.prepare_flex_mask(query_length, device)

    @torch.no_grad()
    def position_diagnostics(
        self,
    ) -> tuple[dict[str, float], dict[str, torch.Tensor]]:
        """Return scalar bias statistics and selected [heads, extent] profiles."""
        metrics: dict[str, float] = {}
        profiles: dict[str, torch.Tensor] = {}
        selected_layers = {0, len(self.blocks) // 2, len(self.blocks) - 1}
        for layer_idx, block in enumerate(self.blocks):
            curves = block.attn.logit_bias_curves()
            if curves is None:
                continue
            curves_cpu = curves.cpu()
            prefix = f"position/layer_{layer_idx:02d}"
            metrics[f"{prefix}/bias_mean"] = curves_cpu.mean().item()
            metrics[f"{prefix}/bias_std"] = curves_cpu.std().item()
            metrics[f"{prefix}/bias_abs_max"] = curves_cpu.abs().max().item()
            if layer_idx in selected_layers:
                profiles[f"layer_{layer_idx:02d}"] = curves_cpu
        return metrics, profiles

    def forward(self, input_ids, targets=None, *, return_logits: bool = False):
        """Training path returns loss only so torch.compile need not keep vocab logits live."""
        x = self.in_proj(self.token_embedding(input_ids))
        for block in self.blocks:
            if self.gradient_checkpointing:
                x = torch.utils.checkpoint.checkpoint(
                    block,
                    x,
                    preserve_rng_state=False,
                    use_reentrant=False,
                    determinism_check="none",
                )
            else:
                x = block(x)
        logits = self.out_proj(x)
        if targets is None:
            return logits
        # CE in fp32 under bf16 autocast; default path drops logits from the return.
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]).float(),
            targets.reshape(-1),
        )
        if return_logits:
            return loss, logits
        return loss

    def resize_token_embeddings(self, new_size: int):
        old_weight = self.token_embedding.weight
        new_emb = torch.nn.Embedding(
            new_size,
            old_weight.shape[1],
            device=old_weight.device,
            dtype=old_weight.dtype,
        )
        with torch.no_grad():
            num_tokens = min(old_weight.shape[0], new_size)
            new_emb.weight[:num_tokens] = old_weight[:num_tokens]
        self.token_embedding = new_emb
        return self.token_embedding


def count_parameters(model: torch.nn.Module) -> dict[str, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    embed = sum(parameter.numel() for parameter in model.token_embedding.parameters())
    head = sum(parameter.numel() for parameter in model.out_proj.parameters())
    qk_position = sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if ".qk_position." in name
    )
    logit_bias = sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if ".logit_bias." in name
    )
    position = qk_position + logit_bias
    return {
        "total": total,
        "embeddings": embed,
        "lm_head": head,
        "position_params": position,
        "qk_position_params": qk_position,
        "logit_bias_params": logit_bias,
        "non_embed": total - embed - head,
    }


def suggest_matched_baselines(cfg: dict) -> dict:
    """Placeholder for position-variant parameter/wallclock matching."""
    del cfg
    raise NotImplementedError(
        "Matched-baseline helper deferred. Match pos_rank / pos_mlp_hidden or "
        "override ff_mult manually for now."
    )
