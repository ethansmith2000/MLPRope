# MLPRope Handoff Brief

_Last updated: 2026-07-29_

## What this project is

**MLPRope** is a research codebase for systematically exploring positional encoding in a small GPT-style transformer. The goal is not just "beat RoPE" but to build a **configurable position playground** where application site, geometry, Fourier basis, mappers, Q/K coupling, head coupling, content conditioning, residual-stream PE, attention writes, and relative logit biases can be mixed and compared under controlled conditions.

Training target: OpenWebText with GPT-2 tokenizer. Default small model: **h768/d8** (8 layers, 8 heads). Larger scale gates used **h1024/d12**. Eval loss at sequence length 1024 is the primary metric unless extrapolation is explicitly requested.

## Start here (docs)

| Path | Purpose |
| --- | --- |
| [`EXPERIMENT_JOURNAL.md`](EXPERIMENT_JOURNAL.md) | **Living experiment record** — read the tail first (2026-07-27 onward) |
| [`POSITION_CONFIG.md`](POSITION_CONFIG.md) | Full v2 schema reference |
| [`axes.md`](axes.md) | Original research axes / experiment menu |
| [`position_embedding_experiments.md`](position_embedding_experiments.md) | Early design notes |
| [`GPU_QUEUEING.md`](/workspace/GPU_QUEUEING.md) | **Required** for launching GPU jobs on this machine |
| [`launch_position_bias.sh`](launch_position_bias.sh) | Sweep launcher; set `EXPERIMENT_FAMILY` |
| [`position_results.py`](position_results.py) | Pull/summarize local run metrics |

## Code map (implementation)

| Module | Role |
| --- | --- |
| `position/config.py` | v2 schema, validation, v1 upgrade |
| `position/basis.py` | Fourier / learned-frequency inputs |
| `position/mappers.py` | identity, affine, linear, low-rank, MLP mappers |
| `position/channels.py` | Q/K channel, `CarrierHypernetwork`, logit/residual/write channels |
| `position/rotary.py` | RoPE, phase composition, radial scaling |
| `transformer.py` | Attention, SDPA vs FlexAttention dispatch |
| `train_gpt.py` | Training entrypoint |
| `test_position_playground.py`, `test_position_channels.py` | CPU tests |
| `scripts/position_v2_cuda_smoke.py` | GPU smoke tests |

Key classes: `QKPositionChannel` and `CarrierHypernetwork` in `position/channels.py`.

## Architecture in one paragraph

Position is factored into independent channels (Q/K, logit bias, residual stream, attention writes). The v2 Q/K channel separates **application** (`additive` vs `rotary`), **geometry** (`amplitude_phase`, `phase`, `free_direct`, etc.), **input basis**, **mapper**, **Q/K coupling**, and **head coupling**. Content conditioning can target Q/K carriers via a **carrier hypernetwork** that modulates amplitude and phase while starting at a null anchor. Logit-bias channels require **FlexAttention** (slower); pure Q/K carrier methods can use **fused SDPA** (much faster).

## What we've learned (high level)

### Pruned or deprioritized

- **Relative logit biases**: Can win quality (best overall historically ~4.06 at h768/d8) but require FlexAttention and are ~2× slower than SDPA. Not the current focus for throughput-sensitive work.
- **Content conditioning in logit-bias / Euclidean additive space**: Many runs collapsed or were unstable; largely shelved.
- **Rank-32 logit at scale**: Extrapolation advantage did not survive 50k at h1024/d12; pruned.
- **Full Q/K Fourier basis at scale**: Basis-16 tied/beats full basis at 50k with fewer params; basis-32 pruned.
- **Asymmetric dynamic-Q/static-K carriers, phase-only HyperRoPE**: Underperformed symmetric HyperAddRoPE.
- **Separate Q/K content projections and separate hypernetwork trunks**: Marginal at 5k; shared content + shared trunk + separate Q/K readouts is the default.
- **Shared Q/K readout or shared conditioner across heads**: Clearly worse.

### Current winners / active direction

The project pivoted from FlexAttention+logit stacks toward **SDPA-compatible content-conditioned AddRoPE**:

1. **Mapped AddRoPE** (static position mapper → amplitude/phase): strong, simple baseline.
2. **Unit-anchored HyperAddRoPE** (dynamic): content (+ optionally position) predicts
   `(1 + scale) * cis(ωp + phase_delta)` with zero init → exact `cis(ωp)` anchor.
   - Best 5k: content+position SiLU (~4.29 vs mapped AddRoPE ~4.34).
   - Best 30k: content+position SiLU **3.571** vs mapped AddRoPE **3.586** vs standard RoPE **3.627**.
3. Structural default: **shared normalized content projection**, **shared trunk**, **per-head conditioner**, **separate Q/K readouts**, `content_dim=64`, `trunk_width=64` (cheap; wider cells only marginally better but cost 4–5× positional params).

### Important confound we fixed

Early hypernetwork runs had a **gauge confound**: static learned amplitude/phase **and** dynamic log-gain/phase both modulated the carrier, with `exp_with_identity_grad` allowing huge effective gains (100×+). Fixed by:
- `output.parameter_source=direct` (no position mapper when using dynamic hypernetwork)
- `conditioning.components=amplitude_phase` (not `log_gain_phase` for additive)
- Unit anchor: `amplitude_init=1`, `signed` amplitude, zero hypernetwork outputs

See journal entries **2026-07-28 — AddRoPE gauge correction** and **Unit-anchor HyperAddRoPE screen**.

## Completed experiment families (recent)

| Family | What it tested |
| --- | --- |
| `phase9_unit_hyper` | Unit-anchor HyperAddRoPE screen (10 cells) |
| `phase9_carrier_followup` | Shared/separate content, asymmetric Q/K, phase-only HyperRoPE |
| `phase9_hyper_30k` | 30k promotion: RoPE, mapped AddRoPE, HyperAddRoPE SiLU/linear |
| `phase9_qk_independence` | 2×2 content/trunk sharing grid |
| `phase9_hyper_capacity` | Shared readout, shared head, content 128, trunk 128/256 |

Output roots under `model-output/position_bias_<family>/`.

## Where we're at now

**HyperAddRoPE is the lead candidate.** Mapped AddRoPE remains the static control. Standard RoPE is the throughput baseline.

The capacity screen (`phase9_hyper_capacity`) finished. Findings:
- Control (c64/h64): 4.2905
- Best tie-band: content-128/trunk-256 at 4.2810, but **6.35M positional params** vs 1.53M for c64/h64
- Structural mistakes (shared Q/K readout, shared head) are clearly worse

**Not yet run / under active discussion** — a micro-screen of normalization and output-geometry tweaks:
- Modality-wise RMS on hypernetwork inputs + learnable content/position scalar gains (init 1)
- Amplitude parameterization: keep `1+s` (signed/raw) vs exact-one softplus vs Cartesian complex residual `(1+u)+iv`
- Content-conditioned **frequency multiplier**: `cis((1 + pred_freq_mult) * (ωp + φ))`
- Whether amplitude + frequency + phase all dynamic is redundant vs partial dynamic + static vector
- Amplitude-only / phase-only isolation for HyperAddRoPE (deferred earlier)

User preference: keep **c128/h64** as cheap default; sweep structural/normalization axes rather than more capacity dimensions. Prefer **`1+s` over softplus** for amplitude. Checkpoint saving off by default; eval at **1024 only** unless extrapolation is explicit.

## How to run experiments

```bash
cd /workspace/MLPRope

# CPU tests
/venv/main/bin/python -m unittest test_position_channels test_position_playground -v

# GPU smoke
gpu-claim run --owner mlprope --job position-smoke --wait -- \
  /venv/main/bin/python -u scripts/position_v2_cuda_smoke.py

# Launch a sweep (example)
export EXPERIMENT_FAMILY=phase9_hyper_capacity
export WITH_TRACKING=true   # user typically uses W&B
export SUBMIT_JOBS=true
export PARALLEL=true
./launch_position_bias.sh
```

Always use `gpu-claim run --owner mlprope --job <name> --wait --` per `/workspace/GPU_QUEUEING.md`. Long sweeps should go under **supervisor** (see `/opt/supervisor-scripts/mlprope-capacity.sh` as a template) to survive terminal disconnects.

## Pull results

```bash
/venv/main/bin/python position_results.py \
  --root model-output/position_bias_phase9_hyper_capacity \
  --format table
```

## Suggested next steps for incoming agent

1. Read journal tail (2026-07-28 entries) and confirm latest run metrics with `position_results.py`.
2. Implement + launch the **micro-screen** discussed above (normalization, amplitude geometry, optional freq multiplier) as a new `EXPERIMENT_FAMILY` in `launch_position_bias.sh`.
3. Include controls: mapped AddRoPE, current signed-polar HyperAddRoPE (c64/h64), standard RoPE.
4. If a micro-screen winner is clear and cheap, consider **50k h1024/d12** promotion (batch 8, accum 4, compile default, no checkpointing) — but user wants parameter-matched FFN control before promoting high-param variants.
5. Do **not** re-open logit-bias or Euclidean content-conditioning paths without strong reason.
6. Update `EXPERIMENT_JOURNAL.md` after each screen (dated entry, don't rewrite history).

## Open design questions (from user)

- Should hypernetwork inputs use **modality-wise RMS** with learnable per-modality gains?
- Is **`1+s` amplitude** sufficient, or worth Cartesian `(1+u)+iv` complex residual?
- **Frequency multiplier** vs amplitude+phase: redundant or complementary?
- Mix of **static learned vectors** (per head/freq) + **dynamic** content predictions for different components?
- QKNorm timing: additive carriers benefit from norm **after** position add; rotary **before** rotation (`qk_norm_mode=method_aware_rms`).

## Transcript

Full prior conversation (including code walkthroughs and design rationale):  
`/root/.cursor/projects/workspace/agent-transcripts/76683046-c950-4db2-8490-e97bb84a9e67/76683046-c950-4db2-8490-e97bb84a9e67.jsonl`
