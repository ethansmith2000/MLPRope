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
    adapt_legacy_position_state_dict,
    apply_rotary,
    build_attention_position_write_channel,
    build_logit_bias_channel,
    build_qk_position_channel,
    build_residual_position_channel,
    build_rope_cache,
    count_position_parameters,
    ensure_channel_v2,
    interleaved_fourier_basis,
    normalize_attention_write_config,
    normalize_residual_stream_config,
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


AttentionImpl = Literal["sdpa", "flex"]

# Re-export the interleaved Fourier helper under its historical name.
sinusoidal_basis = interleaved_fourier_basis


def causal_mask(batch_idx, head_idx, query_idx, key_value_idx):
    del batch_idx, head_idx
    return query_idx >= key_value_idx


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
        attention_write_config: dict | None = None,
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
        self.logit_bias_config = ensure_channel_v2(
            "logit_bias",
            logit_bias_config,
            model_dim=dim,
            heads=heads,
            rope_theta=rope_theta,
        )
        self.attention_write_config = normalize_attention_write_config(
            attention_write_config,
            model_dim=dim,
            heads=heads,
            rope_theta=rope_theta,
        )
        self._prepared_block_mask: BlockMask | None = None
        self._prepared_query_length: int | None = None
        # Stable score_mod closure target — updated each flex forward, not redefined.
        self._flex_bias_curves: torch.Tensor | None = None

        if self.logit_bias_config.get("enabled", False) and attn_impl != "flex":
            raise ValueError(
                "The logit-bias channel requires attn_impl='flex'. "
                "Q/K-only position channels may use SDPA."
            )

        # Split projections match the other experimental Transformer.
        self.to_q = torch.nn.Linear(dim, dim, bias=False)
        self.to_k = torch.nn.Linear(dim, dim, bias=False)
        self.to_v = torch.nn.Linear(dim, dim, bias=False)
        self.to_out = torch.nn.Linear(dim, dim, bias=True)

        self.qk_position = build_qk_position_channel(
            self.qk_config,
            heads=heads,
            head_dim=self.head_dim,
            model_dim=dim,
            extent=max_seq_len,
            rope_theta=rope_theta,
        )
        # Additive Fourier Q/K replaces multiplicative RoPE. Rotary/phase and the
        # logit-only / baseline paths keep it. use_rope=False is only valid
        # together with qk.application="additive".
        self.multiplicative_rope = bool(use_rope) and not (
            self.qk_position is not None
            and not self.qk_position.uses_multiplicative_rope
        )
        if not self.multiplicative_rope and self.qk_position is None:
            raise ValueError(
                "Disable multiplicative RoPE only when qk.application='additive' "
                "(additive Fourier Q/K) supplies position."
            )
        if not use_rope and (
            self.qk_position is None
            or self.qk_position.application != "additive"
        ):
            raise ValueError(
                "use_rope=False requires an enabled additive Q/K channel "
                "(qk.application='additive')."
            )
        self.logit_bias = build_logit_bias_channel(
            self.logit_bias_config,
            heads=heads,
            head_dim=self.head_dim,
            model_dim=dim,
            extent=self.rel_extent,
            rope_theta=rope_theta,
        )
        self.position_write = build_attention_position_write_channel(
            self.attention_write_config,
            heads=heads,
            head_dim=self.head_dim,
            model_dim=dim,
            extent=max_seq_len,
            rope_theta=rope_theta,
        )

        if qk_norm:
            self.q_norm = torch.nn.LayerNorm(self.head_dim)
            self.k_norm = torch.nn.LayerNorm(self.head_dim)
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

    def _apply_rope(
        self,
        q,
        k,
        *,
        q_phase_delta=None,
        k_phase_delta=None,
        q_scale=None,
        k_scale=None,
        phase_delta=None,
    ):
        if phase_delta is not None:
            if q_phase_delta is not None or k_phase_delta is not None:
                raise ValueError(
                    "Pass either phase_delta or q_phase_delta/k_phase_delta, not both."
                )
            q_phase_delta = phase_delta
            k_phase_delta = phase_delta
        return apply_rotary(
            q,
            k,
            self.rope_sin,
            self.rope_cos,
            q_phase_delta=q_phase_delta,
            k_phase_delta=k_phase_delta,
            q_scale=q_scale,
            k_scale=k_scale,
        )

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
        bias_curves = self._flex_bias_curves
        distance = query_idx - key_value_idx
        in_range = (distance >= 0) & (distance < self.rel_extent)
        distance = distance.clamp(0, self.rel_extent - 1)
        if bias_curves.ndim == 2:
            bias = bias_curves[head_idx, distance]
        else:
            bias = bias_curves[batch_idx, head_idx, query_idx, distance]
        return score + torch.where(in_range, bias, 0.0)

    @torch.compiler.disable
    def _flex_attention(self, q, k, v, query_content):
        block_mask = self._block_mask(q)
        if self.logit_bias is None:
            return _flex_attention_call(q, k, v, block_mask=block_mask)

        self._flex_bias_curves = self.logit_bias(
            dtype=q.dtype,
            query=query_content,
        )
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
        q_content = q
        k_content = k
        q_phase_delta = None
        k_phase_delta = None
        q_scale = None
        k_scale = None
        if self.qk_position is not None:
            position_output = self.qk_position(
                q.shape[-2],
                dtype=q.dtype,
                q_content=q_content,
                k_content=k_content,
            )
            if position_output.application == "additive":
                # Additive Fourier Q/K: q' = q + e_q(p), no R(θ)q.
                q_addend = (
                    position_output.q[None, :, :, :]
                    if position_output.q.ndim == 3
                    else position_output.q
                )
                k_addend = (
                    position_output.k[None, :, :, :]
                    if position_output.k.ndim == 3
                    else position_output.k
                )
                q = q + q_addend
                k = k + k_addend
            else:
                q_phase_delta = position_output.q
                k_phase_delta = position_output.k
                q_scale = position_output.q_scale
                k_scale = position_output.k_scale
        if self.multiplicative_rope:
            q, k = self._apply_rope(
                q,
                k,
                q_phase_delta=q_phase_delta,
                k_phase_delta=k_phase_delta,
                q_scale=q_scale,
                k_scale=k_scale,
            )
        if self.position_write is not None:
            position_values = self.position_write.position_values(
                v.shape[-2],
                dtype=v.dtype,
            )
            position_values = position_values[None].expand(
                v.shape[0], -1, -1, -1
            )
            v = torch.cat((v, position_values), dim=-1)
        if self.attn_impl == "flex":
            attn = self._flex_attention(q, k, v, q_content)
        else:
            attn = F.scaled_dot_product_attention(
                q,
                k,
                v,
                is_causal=self.is_causal,
            )
        if self.position_write is None:
            content_attn = attn
            position_output = None
        else:
            content_attn = attn[..., : self.head_dim]
            position_summary = attn[..., self.head_dim :]
            position_output = self.position_write(position_summary)
        content_attn = (
            content_attn.transpose(1, 2)
            .contiguous()
            .view(x.shape[0], x.shape[1], -1)
        )
        output = self.to_out(content_attn)
        if position_output is not None:
            output = output + position_output
        return output

    @torch.no_grad()
    def logit_bias_curves(self) -> torch.Tensor | None:
        if self.logit_bias is None:
            return None
        parameter = next(self.logit_bias.parameters())
        return self.logit_bias.base_curves(
            dtype=parameter.dtype
        ).float().detach()

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
        attention_write_config: dict | None = None,
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
            attention_write_config=attention_write_config,
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
        ff_hidden_dim=None,
        residual_stream_config: dict | None = None,
        attention_write_config: dict | None = None,
    ):
        super().__init__()

        self.token_embedding = torch.nn.Embedding(vocab_size, dim)
        self.gradient_checkpointing = gradient_checkpointing
        self.residual_stream_config = normalize_residual_stream_config(
            residual_stream_config,
            model_dim=dim,
            heads=heads,
            rope_theta=rope_theta,
        )
        placement = self.residual_stream_config["placement"]
        self.residual_position_input = None
        if self.residual_stream_config["enabled"] and placement in {"input", "both"}:
            self.residual_position_input = build_residual_position_channel(
                self.residual_stream_config,
                model_dim=dim,
                heads=heads,
                extent=max_seq_len,
                rope_theta=rope_theta,
            )
        self.residual_position_layer_shared = bool(
            self.residual_stream_config["layer_shared"]
        )
        self.residual_position_layers = torch.nn.ModuleList()
        if self.residual_stream_config["enabled"] and placement in {
            "per_layer",
            "both",
        }:
            layer_count = 1 if self.residual_position_layer_shared else depth
            self.residual_position_layers.extend(
                build_residual_position_channel(
                    self.residual_stream_config,
                    model_dim=dim,
                    heads=heads,
                    extent=max_seq_len,
                    rope_theta=rope_theta,
                )
                for _ in range(layer_count)
            )
        self.blocks = torch.nn.ModuleList([
            TransformerBlock(
                dim,
                heads,
                ff_hidden_dim or dim * ff_mult,
                is_causal=True,
                use_rope=use_rope,
                rope_theta=rope_theta,
                max_seq_len=max_seq_len,
                qk_norm=qk_norm,
                rel_extent=rel_extent,
                qk_config=qk_config,
                logit_bias_config=logit_bias_config,
                attention_write_config=attention_write_config,
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
        *,
        sequence_length: int | None = None,
    ) -> tuple[dict[str, float], dict[str, torch.Tensor]]:
        """Return scalar position statistics and selected logit profiles."""
        metrics: dict[str, float] = {}
        profiles: dict[str, torch.Tensor] = {}
        selected_layers = {0, len(self.blocks) // 2, len(self.blocks) - 1}
        seq_len = sequence_length
        for layer_idx, block in enumerate(self.blocks):
            curves = block.attn.logit_bias_curves()
            if curves is not None:
                curves_cpu = curves.cpu()
                prefix = f"position/layer_{layer_idx:02d}"
                metrics[f"{prefix}/bias_mean"] = curves_cpu.mean().item()
                metrics[f"{prefix}/bias_std"] = curves_cpu.std().item()
                metrics[f"{prefix}/bias_abs_max"] = curves_cpu.abs().max().item()
                frequencies = (
                    block.attn.logit_bias.pipeline.basis.frequencies()
                    .detach()
                    .float()
                )
                metrics[f"{prefix}/frequency_min"] = frequencies.min().item()
                metrics[f"{prefix}/frequency_max"] = frequencies.max().item()
                if layer_idx in selected_layers:
                    profiles[f"layer_{layer_idx:02d}"] = curves_cpu
                routing = block.attn.logit_bias.routing_summary()
                for key, value in routing.items():
                    metrics[f"{prefix}/{key}"] = value

            if block.attn.position_write is not None:
                gate = block.attn.position_write.gate.detach().float()
                prefix = f"position/layer_{layer_idx:02d}/attention_write"
                metrics[f"{prefix}/gate_mean"] = gate.mean().item()
                metrics[f"{prefix}/gate_abs_max"] = gate.abs().max().item()

            if seq_len is None:
                continue
            qk_summary = block.attn.qk_position_summary(seq_len)
            if qk_summary is None:
                continue
            prefix = f"position/layer_{layer_idx:02d}/qk"
            for key, value in qk_summary.items():
                metrics[f"{prefix}/{key}"] = value
        if self.residual_position_input is not None:
            gate = self.residual_position_input.gate.detach().float()
            metrics["position/residual_stream/input_gate"] = gate.item()
        for layer_idx, channel in enumerate(self.residual_position_layers):
            gate = channel.gate.detach().float()
            metrics[
                f"position/residual_stream/layer_{layer_idx:02d}_gate"
            ] = gate.item()
        return metrics, profiles

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        adapted = adapt_legacy_position_state_dict(state_dict)
        return super().load_state_dict(adapted, strict=strict, assign=assign)

    def forward(self, input_ids, targets=None, *, return_logits: bool = False):
        """Training path returns loss only so torch.compile need not keep vocab logits live."""
        x = self.in_proj(self.token_embedding(input_ids))
        if self.residual_position_input is not None:
            x = x + self.residual_position_input(
                x.shape[1],
                dtype=x.dtype,
            )[None, :, :]
        for layer_idx, block in enumerate(self.blocks):
            if self.residual_position_layers:
                position_idx = (
                    0 if self.residual_position_layer_shared else layer_idx
                )
                x = x + self.residual_position_layers[position_idx](
                    x.shape[1],
                    dtype=x.dtype,
                )[None, :, :]
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
        "logit_bias_params": position_counts["logit_bias_params"],
        "attention_write_params": position_counts["attention_write_params"],
        "residual_stream_params": position_counts["residual_stream_params"],
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
from position.channels import LogitBiasChannel, QKPositionChannel  # noqa: E402,F401
