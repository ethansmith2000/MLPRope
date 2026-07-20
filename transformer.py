import math
from typing import Literal

import torch
import torch.nn.functional as F
from torch.nn.attention.flex_attention import (
    BlockMask,
    create_block_mask,
    flex_attention as _flex_attention,
)

# Fuse FlexAttention kernels even when the outer module isn't compiled yet.
flex_attention = torch.compile(_flex_attention)


PositionVariant = Literal[
    "rope",
    "add_rope",
    "linear",
    "low_rank",
    "mlp_rope",
    "inkling_table",
    "inkling_cosnet",
]
AttentionImpl = Literal["sdpa", "flex"]

POSITION_ONLY_VARIANTS = {"add_rope", "linear", "low_rank", "mlp_rope"}
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


class PositionBias(torch.nn.Module):
    """Base interface for position-only per-head relative-bias curves."""

    def __init__(
        self,
        heads: int,
        feature_dim: int,
        rel_extent: int,
        theta: float,
    ):
        super().__init__()
        self.heads = heads
        self.feature_dim = feature_dim
        self.rel_extent = rel_extent
        self.register_buffer(
            "basis",
            sinusoidal_basis(rel_extent, feature_dim, theta),
            persistent=False,
        )

    def _basis(self) -> torch.Tensor:
        parameter = next(self.parameters())
        return self.basis.to(dtype=parameter.dtype)

    def reset_output_parameters(self) -> None:
        raise NotImplementedError


class AddRoPEBias(PositionBias):
    """Per-head affine weighting of the fixed sinusoidal frequency basis."""

    def __init__(self, heads, feature_dim, rel_extent, theta):
        super().__init__(heads, feature_dim, rel_extent, theta)
        self.scale = torch.nn.Parameter(torch.zeros(heads, feature_dim))
        self.offset = torch.nn.Parameter(torch.zeros(heads, feature_dim))

    def forward(self) -> torch.Tensor:
        basis = self._basis()
        curves = torch.einsum("rd,hd->hr", basis, self.scale)
        curves = curves + self.offset.mean(dim=-1, keepdim=True)
        return curves / math.sqrt(self.feature_dim)

    def reset_output_parameters(self) -> None:
        torch.nn.init.zeros_(self.scale)
        torch.nn.init.zeros_(self.offset)


class LinearPositionBias(PositionBias):
    """Per-head full linear transform of the sinusoidal basis."""

    def __init__(self, heads, feature_dim, rel_extent, theta):
        super().__init__(heads, feature_dim, rel_extent, theta)
        self.weight = torch.nn.Parameter(
            torch.empty(heads, feature_dim, feature_dim)
        )
        self.bias = torch.nn.Parameter(torch.zeros(heads, feature_dim))
        self.readout = torch.nn.Parameter(torch.zeros(heads, feature_dim))
        self.readout_bias = torch.nn.Parameter(torch.zeros(heads))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for weight in self.weight:
            torch.nn.init.xavier_normal_(weight)
        torch.nn.init.zeros_(self.bias)
        self.reset_output_parameters()

    def forward(self) -> torch.Tensor:
        basis = self._basis()
        transformed = torch.einsum("rd,hde->hre", basis, self.weight)
        transformed = transformed + self.bias[:, None, :]
        curves = torch.einsum("hre,he->hr", transformed, self.readout)
        return curves + self.readout_bias[:, None]

    def reset_output_parameters(self) -> None:
        torch.nn.init.zeros_(self.readout)
        torch.nn.init.zeros_(self.readout_bias)


class LowRankPositionBias(PositionBias):
    """Per-head nonlinear feature_dim -> rank -> scalar bias mapping."""

    def __init__(self, heads, feature_dim, rel_extent, theta, rank):
        super().__init__(heads, feature_dim, rel_extent, theta)
        if rank <= 0:
            raise ValueError("pos_rank must be positive.")
        self.rank = rank
        self.down = torch.nn.Parameter(torch.empty(heads, feature_dim, rank))
        self.down_bias = torch.nn.Parameter(torch.zeros(heads, rank))
        self.up = torch.nn.Parameter(torch.zeros(heads, rank))
        self.up_bias = torch.nn.Parameter(torch.zeros(heads))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for weight in self.down:
            torch.nn.init.xavier_normal_(weight)
        torch.nn.init.zeros_(self.down_bias)
        self.reset_output_parameters()

    def forward(self) -> torch.Tensor:
        basis = self._basis()
        hidden = torch.einsum("rd,hdk->hrk", basis, self.down)
        hidden = F.gelu(hidden + self.down_bias[:, None, :])
        curves = torch.einsum("hrk,hk->hr", hidden, self.up)
        return curves + self.up_bias[:, None]

    def reset_output_parameters(self) -> None:
        torch.nn.init.zeros_(self.up)
        torch.nn.init.zeros_(self.up_bias)


class MLPRoPEBias(PositionBias):
    """Per-head MLP with a sinusoidal residual and zero-initialized readout."""

    def __init__(
        self,
        heads,
        feature_dim,
        rel_extent,
        theta,
        hidden_dim,
    ):
        super().__init__(heads, feature_dim, rel_extent, theta)
        if hidden_dim <= 0:
            raise ValueError("pos_mlp_hidden must be positive.")
        self.hidden_dim = hidden_dim
        self.in_weight = torch.nn.Parameter(
            torch.empty(heads, feature_dim, hidden_dim)
        )
        self.in_bias = torch.nn.Parameter(torch.zeros(heads, hidden_dim))
        self.out_weight = torch.nn.Parameter(
            torch.empty(heads, hidden_dim, feature_dim)
        )
        self.out_bias = torch.nn.Parameter(torch.zeros(heads, feature_dim))
        self.readout = torch.nn.Parameter(torch.zeros(heads, feature_dim))
        self.readout_bias = torch.nn.Parameter(torch.zeros(heads))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for weight in self.in_weight:
            torch.nn.init.xavier_normal_(weight)
        for weight in self.out_weight:
            # Keep the nonlinear branch a small perturbation of the raw basis.
            torch.nn.init.normal_(weight, mean=0.0, std=1.0e-3)
        torch.nn.init.zeros_(self.in_bias)
        torch.nn.init.zeros_(self.out_bias)
        self.reset_output_parameters()

    def forward(self) -> torch.Tensor:
        basis = self._basis()
        hidden = torch.einsum("rd,hdm->hrm", basis, self.in_weight)
        hidden = F.gelu(hidden + self.in_bias[:, None, :])
        delta = torch.einsum("hrm,hmd->hrd", hidden, self.out_weight)
        enriched = basis[None, :, :] + delta + self.out_bias[:, None, :]
        curves = torch.einsum("hrd,hd->hr", enriched, self.readout)
        return curves + self.readout_bias[:, None]

    def reset_output_parameters(self) -> None:
        torch.nn.init.zeros_(self.readout)
        torch.nn.init.zeros_(self.readout_bias)


def make_position_bias(
    variant: PositionVariant,
    heads: int,
    feature_dim: int,
    rel_extent: int,
    theta: float,
    rank: int,
    mlp_hidden: int,
) -> PositionBias | None:
    common = (heads, feature_dim, rel_extent, theta)
    if variant == "rope":
        return None
    if variant == "add_rope":
        return AddRoPEBias(*common)
    if variant == "linear":
        return LinearPositionBias(*common)
    if variant == "low_rank":
        return LowRankPositionBias(*common, rank=rank)
    if variant == "mlp_rope":
        return MLPRoPEBias(*common, hidden_dim=mlp_hidden)
    if variant in CONTENT_CONDITIONED_VARIANTS:
        raise NotImplementedError(
            f"pos_variant={variant!r} is scaffolded for Phase 2 but not implemented."
        )
    raise ValueError(f"Unknown pos_variant: {variant!r}")


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
        pos_variant: PositionVariant = "rope",
        rel_extent: int | None = None,
        pos_rank: int = 32,
        pos_mlp_hidden: int = 128,
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
        self.pos_variant = pos_variant
        self.rel_extent = rel_extent or max_seq_len
        self.attn_impl = attn_impl
        self._prepared_block_mask: BlockMask | None = None
        self._prepared_query_length: int | None = None

        if dim % heads != 0:
            raise ValueError("dim must be divisible by heads.")
        if not use_rope:
            raise ValueError("These experiments keep RoPE enabled as the geometric prior.")
        if self.head_dim % 2 != 0:
            raise ValueError("RoPE requires an even head dimension.")
        if pos_variant not in POSITION_VARIANTS:
            raise ValueError(f"Unknown pos_variant: {pos_variant!r}")
        if attn_impl not in ("sdpa", "flex"):
            raise ValueError(f"Unknown attn_impl: {attn_impl!r}")
        if pos_variant != "rope" and attn_impl != "flex":
            raise ValueError(
                "Learned position-bias variants require attn_impl='flex'. "
                "Use pos_variant='rope' for the SDPA baseline."
            )
        if self.rel_extent <= 0:
            raise ValueError("rel_extent must be positive.")

        # Split projections match the other experimental Transformer.
        self.to_q = torch.nn.Linear(dim, dim, bias=False)
        self.to_k = torch.nn.Linear(dim, dim, bias=False)
        self.to_v = torch.nn.Linear(dim, dim, bias=False)
        self.to_out = torch.nn.Linear(dim, dim, bias=True)

        self.position_bias = make_position_bias(
            pos_variant,
            heads,
            self.head_dim,
            self.rel_extent,
            rope_theta,
            pos_rank,
            pos_mlp_hidden,
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

    def _apply_rope(self, q, k):
        seq_len = q.shape[-2]
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds RoPE cache length {self.max_seq_len}"
            )
        half = q.shape[-1] // 2
        sin = self.rope_sin[:seq_len].to(dtype=q.dtype)[None, None, :, :]
        cos = self.rope_cos[:seq_len].to(dtype=q.dtype)[None, None, :, :]

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

    def _flex_attention(self, q, k, v):
        block_mask = self._block_mask(q)
        if self.position_bias is None:
            return flex_attention(q, k, v, block_mask=block_mask)

        bias_curves = self.position_bias().to(dtype=q.dtype)
        rel_extent = self.rel_extent

        def score_mod(score, batch_idx, head_idx, query_idx, key_value_idx):
            del batch_idx
            distance = query_idx - key_value_idx
            in_range = (distance >= 0) & (distance < rel_extent)
            distance = distance.clamp(0, rel_extent - 1)
            bias = bias_curves[head_idx, distance]
            return score + torch.where(in_range, bias, 0.0)

        return flex_attention(
            q,
            k,
            v,
            score_mod=score_mod,
            block_mask=block_mask,
        )

    def forward(self, x):
        q = self._split_heads(self.to_q(x))
        k = self._split_heads(self.to_k(x))
        v = self._split_heads(self.to_v(x))

        q = self.q_norm(q)
        k = self.k_norm(k)
        q, k = self._apply_rope(q, k)
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
        pos_variant: PositionVariant = "rope",
        rel_extent: int | None = None,
        pos_rank: int = 32,
        pos_mlp_hidden: int = 128,
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
            pos_variant=pos_variant,
            rel_extent=rel_extent,
            pos_rank=pos_rank,
            pos_mlp_hidden=pos_mlp_hidden,
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
        pos_variant: PositionVariant = "rope",
        rel_extent: int | None = None,
        pos_rank: int = 32,
        pos_mlp_hidden: int = 128,
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
                pos_variant=pos_variant,
                rel_extent=rel_extent,
                pos_rank=pos_rank,
                pos_mlp_hidden=pos_mlp_hidden,
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
                if isinstance(module, PositionBias):
                    module.reset_output_parameters()

    def prepare_flex_masks(
        self,
        query_length: int,
        device: torch.device | str,
    ) -> None:
        for block in self.blocks:
            block.attn.prepare_flex_mask(query_length, device)

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
    position = sum(
        parameter.numel()
        for module in model.modules()
        if isinstance(module, PositionBias)
        for parameter in module.parameters(recurse=False)
    )
    return {
        "total": total,
        "embeddings": embed,
        "lm_head": head,
        "position_bias": position,
        "non_embed": total - embed - head,
    }


def suggest_matched_baselines(cfg: dict) -> dict:
    """Placeholder for position-variant parameter/wallclock matching."""
    del cfg
    raise NotImplementedError(
        "Matched-baseline helper deferred. Match pos_rank / pos_mlp_hidden or "
        "override ff_mult manually for now."
    )
