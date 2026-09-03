import hashlib
import math
from typing import Literal

import torch
import torch.nn.functional as F
from torch.nn.attention.flex_attention import (
    BlockMask,
    create_block_mask,
    flex_attention as _flex_attention_op,
)

from position import (
    PositionChannel,
    QKPreprojectionPosition,
    adapt_legacy_position_state_dict,
    apply_rotary,
    build_qk_position_channel,
    build_rope_cache,
    build_rope_frequencies,
    count_position_parameters,
    ensure_channel_v2,
    interleaved_fourier_basis,
    normalize_logit_bias_config,
    normalize_qk_preprojection_config,
    normalize_position_content_config,
)
from position.precision import PreserveFP32BuffersMixin

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


AttentionImpl = Literal["sdpa", "flex"]

# Re-export the interleaved Fourier helper under its historical name.
sinusoidal_basis = interleaved_fourier_basis


def causal_mask(batch_idx, head_idx, query_idx, key_value_idx):
    del batch_idx, head_idx
    return query_idx >= key_value_idx


class PositionContentProjection(torch.nn.Module):
    """Dedicated normalized low-rank content for positional mechanisms."""

    def __init__(
        self,
        model_dim: int,
        content_dim: int,
        heads: int,
        coupling: str,
    ):
        super().__init__()
        self.heads = heads
        self.coupling = coupling
        self.q_projection = torch.nn.Linear(model_dim, content_dim, bias=False)
        self.k_projection = (
            self.q_projection
            if coupling == "shared"
            else torch.nn.Linear(model_dim, content_dim, bias=False)
        )

    @staticmethod
    def _unit_rms(value: torch.Tensor) -> torch.Tensor:
        scale = torch.rsqrt(
            value.float().square().mean(dim=-1, keepdim=True) + 1e-6
        )
        return value * scale.to(dtype=value.dtype)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        q_content = self._unit_rms(self.q_projection(x))
        k_content = (
            q_content
            if self.coupling == "shared"
            else self._unit_rms(self.k_projection(x))
        )
        return (
            q_content[:, None].expand(-1, self.heads, -1, -1).contiguous(),
            k_content[:, None].expand(-1, self.heads, -1, -1).contiguous(),
        )


class Attention(PreserveFP32BuffersMixin, torch.nn.Module):
    _fp32_buffer_names = (
        "rope_sin",
        "rope_cos",
        "rope_inverse_frequency",
    )

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
        post_position_qk_norm: bool = False,
        position_content_dim: int = 64,
        position_content_coupling: str = "separate",
        qk_norm_mode: str = "legacy_layernorm",
        qk_preprojection_config: dict | None = None,
    ):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.is_causal = is_causal
        self.use_rope = use_rope
        self.head_dim = dim // heads
        self.rope_theta = rope_theta
        self.qk_preprojection_config = normalize_qk_preprojection_config(
            qk_preprojection_config,
            model_dim=dim,
            rope_theta=rope_theta,
        )
        self.max_seq_len = max_seq_len
        self.rel_extent = rel_extent or max_seq_len
        self.attn_impl = attn_impl
        self.post_position_qk_norm = bool(post_position_qk_norm)
        if qk_norm_mode not in {"legacy_layernorm", "method_aware_rms"}:
            raise ValueError(
                "qk_norm_mode must be 'legacy_layernorm' or 'method_aware_rms'"
            )
        if qk_norm_mode == "method_aware_rms" and self.post_position_qk_norm:
            raise ValueError(
                "method_aware_rms already applies one Q/K normalization; "
                "post_position_qk_norm must be false"
            )
        self.qk_norm_mode = qk_norm_mode
        self.position_content_config = normalize_position_content_config(
            position_content_dim,
            position_content_coupling,
        )
        if dim % heads != 0:
            raise ValueError("dim must be divisible by heads.")
        if self.head_dim % 2 != 0:
            raise ValueError("Position channels require an even head dimension.")
        if attn_impl not in ("sdpa", "flex"):
            raise ValueError(f"Unknown attn_impl: {attn_impl!r}")
        if self.rel_extent <= 0:
            raise ValueError("rel_extent must be positive.")

        # Accept v1 or v2 channel dicts; store canonical v2 on the module.
        self.qk_config = ensure_channel_v2(
            "qk",
            qk_config,
            model_dim=dim,
            heads=heads,
            rope_theta=rope_theta,
        )
        # The relative logit-bias channel was removed; only {"enabled": false}
        # remains valid for archived-config compatibility.
        self.logit_bias_config = normalize_logit_bias_config(logit_bias_config)
        qk_conditioning = self.qk_config["conditioning"]
        if (
            qk_conditioning["kind"] == "carrier_hypernetwork"
            and qk_conditioning["input_mode"]
            in {"content", "content_position"}
            and qk_conditioning["coupling"] == "shared"
            and qk_conditioning["target"] == "both"
            and self.position_content_config["coupling"] != "shared"
        ):
            raise ValueError(
                "A shared Q/K carrier hypernetwork requires "
                "position_content_coupling='shared'"
            )
        self._prepared_block_mask: BlockMask | None = None
        self._prepared_query_length: int | None = None

        # Split projections match the other experimental Transformer.
        self.to_q = torch.nn.Linear(dim, dim, bias=False)
        self.to_k = torch.nn.Linear(dim, dim, bias=False)
        self.to_v = torch.nn.Linear(dim, dim, bias=False)
        self.to_out = torch.nn.Linear(dim, dim, bias=True)

        conditioning_configs = (self.qk_config["conditioning"],)

        def uses_content(config: dict) -> bool:
            return config["kind"] != "none" and not (
                config["kind"] == "carrier_hypernetwork"
                and config["input_mode"] == "position"
            )

        uses_dedicated_content = any(
            uses_content(cfg) for cfg in conditioning_configs
        )
        self.position_content = (
            PositionContentProjection(
                dim,
                self.position_content_config["dim"],
                heads,
                self.position_content_config["coupling"],
            )
            if uses_dedicated_content
            else None
        )

        self.qk_position = build_qk_position_channel(
            self.qk_config,
            heads=heads,
            head_dim=self.head_dim,
            model_dim=dim,
            content_dim=self.position_content_config["dim"],
            extent=max_seq_len,
            rope_theta=rope_theta,
        )
        self.qk_preprojection = (
            QKPreprojectionPosition(
                self.qk_preprojection_config,
                model_dim=dim,
                extent=max_seq_len,
            )
            if self.qk_preprojection_config["enabled"]
            else None
        )
        # Additive Fourier Q/K replaces multiplicative RoPE. Explicit
        # use_rope=False enables the no-explicit-PE control.
        self.multiplicative_rope = bool(use_rope) and self.qk_position is None

        if qk_norm:
            norm_type = (
                torch.nn.RMSNorm
                if qk_norm_mode == "method_aware_rms"
                else torch.nn.LayerNorm
            )
            self.q_norm = norm_type(self.head_dim, eps=1e-6)
            self.k_norm = norm_type(self.head_dim, eps=1e-6)
        else:
            self.q_norm = torch.nn.Identity()
            self.k_norm = torch.nn.Identity()

        rope_sin, rope_cos = build_rope_cache(
            max_seq_len,
            self.head_dim,
            self.rope_theta,
        )
        self.register_buffer("rope_sin", rope_sin, persistent=False)
        self.register_buffer("rope_cos", rope_cos, persistent=False)
        self.register_buffer(
            "rope_inverse_frequency",
            build_rope_frequencies(self.head_dim, self.rope_theta),
            persistent=False,
        )
    def _apply_rope(self, q, k):
        return apply_rotary(
            q,
            k,
            self.rope_sin,
            self.rope_cos,
        )

    def _split_heads(self, x):
        batch, sequence, _ = x.shape
        return x.view(batch, sequence, self.heads, self.head_dim).transpose(1, 2)

    @staticmethod
    def _unit_rms(value: torch.Tensor) -> torch.Tensor:
        scale = torch.rsqrt(
            value.float().square().mean(dim=-1, keepdim=True) + 1e-6
        )
        return value * scale.to(dtype=value.dtype)

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

    @torch.compiler.disable
    def _flex_attention(self, q, k, v):
        return _flex_attention_call(q, k, v, block_mask=self._block_mask(q))

    def forward(self, x):
        qk_input = x
        if self.qk_preprojection is not None:
            qk_input = x + self.qk_preprojection(
                x.shape[1],
                dtype=x.dtype,
            )[None, :, :]
        q_projected = self._split_heads(self.to_q(qk_input))
        k_projected = self._split_heads(self.to_k(qk_input))
        v = self._split_heads(self.to_v(x))
        q_normed = self.q_norm(q_projected)
        k_normed = self.k_norm(k_projected)
        dedicated_q_content = None
        dedicated_k_content = None
        if self.position_content is not None:
            dedicated_q_content, dedicated_k_content = self.position_content(x)

        def content_for(config: dict) -> tuple[torch.Tensor, torch.Tensor]:
            del config  # every conditioning source is the dedicated projection
            if dedicated_q_content is None or dedicated_k_content is None:
                raise ValueError(
                    "Dedicated positional content was requested but not built"
                )
            return dedicated_q_content, dedicated_k_content

        position_q_content = position_k_content = None
        if self.qk_position is not None:
            qk_conditioning = self.qk_config["conditioning"]
            if qk_conditioning["kind"] != "none" and not (
                qk_conditioning["kind"] == "carrier_hypernetwork"
                and qk_conditioning["input_mode"] == "position"
            ):
                position_q_content, position_k_content = content_for(
                    self.qk_config
                )
        q_gain = None
        k_gain = None
        position_output = None
        if self.qk_position is not None:
            position_output = self.qk_position(
                q_projected.shape[-2],
                dtype=q_projected.dtype,
                q_content=position_q_content,
                k_content=position_k_content,
            )
            q_gain = position_output.q_gain
            k_gain = position_output.k_gain

        if self.qk_norm_mode == "method_aware_rms":
            q = q_projected
            k = k_projected
            if position_output is not None:
                q_addend = (
                    position_output.q[None]
                    if position_output.q.ndim == 3
                    else position_output.q
                )
                k_addend = (
                    position_output.k[None]
                    if position_output.k.ndim == 3
                    else position_output.k
                )
                q = q + q_addend
                k = k + k_addend
                q = self.q_norm(q)
                k = self.k_norm(k)
            else:
                q = q_normed
                k = k_normed
        else:
            q = q_normed
            k = k_normed
            if position_output is not None:
                q_addend = (
                    position_output.q[None]
                    if position_output.q.ndim == 3
                    else position_output.q
                )
                k_addend = (
                    position_output.k[None]
                    if position_output.k.ndim == 3
                    else position_output.k
                )
                q = q + q_addend
                k = k + k_addend
        if self.multiplicative_rope:
            q, k = self._apply_rope(q, k)
        if q_gain is not None:
            q = q * q_gain.to(dtype=q.dtype)
        if k_gain is not None:
            k = k * k_gain.to(dtype=k.dtype)
        if self.post_position_qk_norm:
            q = self._unit_rms(q)
            k = self._unit_rms(k)
        if self.attn_impl == "flex":
            attn = self._flex_attention(q, k, v)
        else:
            attn = F.scaled_dot_product_attention(
                q,
                k,
                v,
                is_causal=self.is_causal,
            )
        content_attn = attn
        content_attn = (
            content_attn.transpose(1, 2)
            .contiguous()
            .view(x.shape[0], x.shape[1], -1)
        )
        output = self.to_out(content_attn)
        return output

    @torch.no_grad()
    def qk_position_summary(
        self,
        sequence_length: int,
        *,
        q_ref: torch.Tensor | None = None,
        k_ref: torch.Tensor | None = None,
    ) -> dict[str, float] | None:
        if self.qk_position is None:
            return None
        parameter = next(self.qk_position.parameters(), None)
        dtype = parameter.dtype if parameter is not None else None
        return self.qk_position.summarize(
            sequence_length,
            dtype=dtype,
            q_ref=q_ref,
            k_ref=k_ref,
        )

    @torch.no_grad()
    def qk_position_summary_from_input(
        self,
        x: torch.Tensor,
    ) -> dict[str, float] | None:
        if self.qk_position is None:
            return None
        q_projected = self._split_heads(self.to_q(x))
        k_projected = self._split_heads(self.to_k(x))
        q = self.q_norm(q_projected)
        k = self.k_norm(k_projected)
        conditioning = self.qk_position.conditioning_config
        if conditioning["kind"] == "none" or (
            conditioning["kind"] == "carrier_hypernetwork"
            and conditioning["input_mode"] == "position"
        ):
            q_content = k_content = None
        else:
            if self.position_content is None:
                raise ValueError("Dedicated positional content projector is missing")
            q_content, k_content = self.position_content(x)
        parameter = next(self.qk_position.parameters(), None)
        dtype = parameter.dtype if parameter is not None else x.dtype
        diagnostic_q_ref = q
        diagnostic_k_ref = k
        if self.qk_norm_mode == "method_aware_rms":
            # Method-aware additive attention composes position with the raw
            # projections and normalizes only afterward. Ratios and cosines
            # must therefore use the raw projections as their reference.
            diagnostic_q_ref = q_projected
            diagnostic_k_ref = k_projected
        summary = self.qk_position.summarize(
            x.shape[1],
            dtype=dtype,
            q_ref=diagnostic_q_ref,
            k_ref=diagnostic_k_ref,
            q_content=q_content,
            k_content=k_content,
        )
        position = self.qk_position(
            x.shape[1],
            dtype=dtype,
            q_content=q_content,
            k_content=k_content,
        )
        q_add = position.q[None] if position.q.ndim == 3 else position.q
        k_add = position.k[None] if position.k.ndim == 3 else position.k
        if self.qk_norm_mode == "method_aware_rms":
            final_q = self.q_norm(q_projected + q_add)
            final_k = self.k_norm(k_projected + k_add)
        else:
            final_q, final_k = q + q_add, k + k_add
        if position.q_gain is not None:
            final_q = final_q * position.q_gain
            final_k = final_k * position.k_gain
        summary["final_q_rms"] = (
            final_q.detach().float().square().mean().sqrt().item()
        )
        summary["final_k_rms"] = (
            final_k.detach().float().square().mean().sqrt().item()
        )
        return summary


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
        post_position_qk_norm: bool = False,
        position_content_dim: int = 64,
        position_content_coupling: str = "separate",
        qk_norm_mode: str = "legacy_layernorm",
        qk_preprojection_config: dict | None = None,
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
            post_position_qk_norm=post_position_qk_norm,
            position_content_dim=position_content_dim,
            position_content_coupling=position_content_coupling,
            qk_norm_mode=qk_norm_mode,
            qk_preprojection_config=qk_preprojection_config,
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
        ff_hidden_dim=None,
        post_position_qk_norm: bool = False,
        position_content_dim: int = 64,
        position_content_coupling: str = "separate",
        qk_norm_mode: str = "legacy_layernorm",
        paired_initialization_seed: int | None = None,
        ff_widened_hidden_dim: int | None = None,
        ff_widened_layers: list[int] | tuple[int, ...] | None = None,
        qk_preprojection_config: dict | None = None,
    ):
        super().__init__()

        self.token_embedding = torch.nn.Embedding(vocab_size, dim)
        self.gradient_checkpointing = gradient_checkpointing
        base_ff_hidden_dim = ff_hidden_dim or dim * ff_mult
        widened_layers = set(ff_widened_layers or ())
        if any(layer_idx < 0 or layer_idx >= depth for layer_idx in widened_layers):
            raise ValueError("ff_widened_layers must contain valid layer indices")
        if widened_layers and ff_widened_hidden_dim is None:
            raise ValueError(
                "ff_widened_hidden_dim is required when ff_widened_layers is set"
            )
        if ff_widened_hidden_dim is not None and ff_widened_hidden_dim <= 0:
            raise ValueError("ff_widened_hidden_dim must be positive")
        self.blocks = torch.nn.ModuleList([
            TransformerBlock(
                dim,
                heads,
                (
                    ff_widened_hidden_dim
                    if layer_idx in widened_layers
                    else base_ff_hidden_dim
                ),
                is_causal=True,
                use_rope=use_rope,
                rope_theta=rope_theta,
                max_seq_len=max_seq_len,
                qk_norm=qk_norm,
                rel_extent=rel_extent,
                qk_config=qk_config,
                logit_bias_config=logit_bias_config,
                attn_impl=attn_impl,
                post_position_qk_norm=post_position_qk_norm,
                position_content_dim=position_content_dim,
                position_content_coupling=position_content_coupling,
                qk_norm_mode=qk_norm_mode,
                qk_preprojection_config=qk_preprojection_config,
            )
            for layer_idx in range(depth)
        ])
        self.in_proj = torch.nn.Sequential(
            torch.nn.LayerNorm(dim),
            torch.nn.Linear(dim, dim, bias=True),
        )
        self.out_proj = torch.nn.Sequential(
            torch.nn.LayerNorm(dim),
            torch.nn.Linear(dim, vocab_size, bias=True),
        )
        self._init_weights(dim, paired_initialization_seed)

    @staticmethod
    def _named_generator(
        seed: int | None,
        parameter_name: str,
    ) -> torch.Generator | None:
        """Return a stable per-parameter RNG for paired comparisons."""
        if seed is None:
            return None
        payload = f"{int(seed)}:{parameter_name}".encode("utf-8")
        digest = hashlib.sha256(payload).digest()
        parameter_seed = int.from_bytes(digest[:8], "big", signed=False)
        return torch.Generator().manual_seed(parameter_seed)

    def _init_weights(
        self,
        dim,
        paired_initialization_seed: int | None = None,
    ):
        embed_std = dim ** -0.5
        lm_head_std = 0.02
        with torch.no_grad():
            for module_name, module in self.named_modules():
                if isinstance(module, torch.nn.Embedding):
                    torch.nn.init.normal_(
                        module.weight,
                        mean=0.0,
                        std=embed_std,
                        generator=self._named_generator(
                            paired_initialization_seed,
                            f"{module_name}.weight",
                        ),
                    )
                elif isinstance(module, torch.nn.Linear):
                    torch.nn.init.xavier_normal_(
                        module.weight,
                        generator=self._named_generator(
                            paired_initialization_seed,
                            f"{module_name}.weight",
                        ),
                    )
                    if module.bias is not None:
                        torch.nn.init.zeros_(module.bias)
                elif isinstance(module, torch.nn.LayerNorm):
                    torch.nn.init.ones_(module.weight)
                    torch.nn.init.zeros_(module.bias)
            lm_head = self.out_proj[1]
            torch.nn.init.normal_(
                lm_head.weight,
                mean=0.0,
                std=lm_head_std,
                generator=self._named_generator(
                    paired_initialization_seed,
                    "out_proj.1.lm_head_weight",
                ),
            )
            if lm_head.bias is not None:
                torch.nn.init.zeros_(lm_head.bias)
            for module in self.modules():
                if isinstance(module, PositionChannel):
                    module.reset_output_parameters()
                elif isinstance(module, QKPreprojectionPosition):
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
        *,
        sequence_length: int | None = None,
        input_ids: torch.Tensor | None = None,
    ) -> tuple[dict[str, float], dict[str, torch.Tensor]]:
        """Return scalar position statistics and selected logit profiles."""
        metrics: dict[str, float] = {}
        profiles: dict[str, torch.Tensor] = {}
        seq_len = sequence_length
        diagnostic_x = None
        if input_ids is not None:
            diagnostic_x = self.in_proj(self.token_embedding(input_ids))
        for layer_idx, block in enumerate(self.blocks):
            actual_qk_summary = None
            normalized_diagnostic_x = None
            if diagnostic_x is not None:
                normalized_diagnostic_x = block.norm1(diagnostic_x)
                actual_qk_summary = block.attn.qk_position_summary_from_input(
                    normalized_diagnostic_x
                )
            if block.attn.qk_preprojection is not None:
                preprojection = block.attn.qk_preprojection
                pre_prefix = f"position/layer_{layer_idx:02d}/qk_preprojection"
                gate = preprojection.gate_value().detach().float()
                metrics[f"{pre_prefix}/gate"] = gate.item()
                if seq_len is not None:
                    positional_input = preprojection(
                        seq_len,
                        dtype=torch.float32,
                    ).detach()
                    metrics[f"{pre_prefix}/input_rms"] = (
                        positional_input.square().mean().sqrt().item()
                    )
                    q_branch = block.attn.to_q(positional_input).detach().float()
                    k_branch = block.attn.to_k(positional_input).detach().float()
                    metrics[f"{pre_prefix}/projected_q_rms"] = (
                        q_branch.square().mean().sqrt().item()
                    )
                    metrics[f"{pre_prefix}/projected_k_rms"] = (
                        k_branch.square().mean().sqrt().item()
                    )
            if diagnostic_x is not None:
                diagnostic_x = block(diagnostic_x)
            if seq_len is None:
                continue
            qk_summary = (
                actual_qk_summary
                if actual_qk_summary is not None
                else block.attn.qk_position_summary(seq_len)
            )
            if qk_summary is None:
                continue
            prefix = f"position/layer_{layer_idx:02d}/qk"
            for key, value in qk_summary.items():
                metrics[f"{prefix}/{key}"] = value
        return metrics, profiles

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        adapted = adapt_legacy_position_state_dict(state_dict)
        return super().load_state_dict(adapted, strict=strict, assign=assign)

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
    position_counts = count_position_parameters(model)
    return {
        "total": total,
        "embeddings": embed,
        "lm_head": head,
        "position_params": position_counts["position_params"],
        "qk_position_params": position_counts["qk_position_params"],
        "qk_preprojection_params": position_counts["qk_preprojection_params"],
        "logit_bias_params": position_counts["logit_bias_params"],
        "non_embed": total - embed - head,
    }


def suggest_matched_baselines(
    cfg: dict,
    *,
    position_params: int = 0,
    align_multiple: int = 64,
) -> dict[str, int]:
    """Recommend a wider GeGLU hidden width spending the position budget.

    A GeGLU hidden unit contributes ``3*dim + 2`` parameters per layer:
    two input projections (including biases) and one output projection.
    """
    dim = int(cfg["hidden_size"])
    depth = int(cfg["depth"])
    current_hidden = int(
        cfg.get("ff_hidden_dim") or dim * int(cfg["ff_mult"])
    )
    per_hidden = 3 * dim + 2
    extra_hidden = math.ceil(max(int(position_params), 0) / max(depth * per_hidden, 1))
    target = current_hidden + extra_hidden
    if align_multiple > 1:
        target = math.ceil(target / align_multiple) * align_multiple
    added = (target - current_hidden) * per_hidden * depth
    return {
        "current_ff_hidden_dim": current_hidden,
        "matched_ff_hidden_dim": target,
        "matched_ff_added_params": added,
        "position_param_target": int(position_params),
    }


# Public re-exports for tests and checkpoint helpers.
from position.channels import QKPositionChannel  # noqa: E402,F401
