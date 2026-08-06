# MLPRope Handoff Brief

_Last updated: 2026-08-05. Written for an incoming agent who has not seen this
codebase. Read the "Audit brief" section before trusting anything else here._

The current decision-driven roadmap is
[`CONSOLIDATED_RESEARCH_PLAN.md`](CONSOLIDATED_RESEARCH_PLAN.md). It supersedes
older "where to pick up" ordering when they conflict; locked protocol files
remain authoritative for experiments already begun.

## What this project is

**MLPRope** is a research codebase for systematically exploring positional
encoding in a small GPT-style transformer. It is a **configurable position
playground**: application site, geometry, Fourier basis, mappers, Q/K coupling,
head coupling, content conditioning, residual-stream PE, attention writes, and
relative logit biases can be mixed and compared under controlled conditions.

Training target: OpenWebText with the GPT-2 tokenizer. Two model sizes are in
use: **h768/d8** (8 heads) and **h1024/d12** (8 heads). Eval loss at sequence
length 1024 is the primary metric. Length extrapolation is explicitly **not**
being optimized for right now.

## The headline screening result

> In the phase-18 seed-123 screen, a **learned per-head positional profile,
> conditioned on position only**, beat standard RoPE by `0.0421` at h1024/d12 /
> 30k steps, for `1.71M` positional parameters and a reported 4.7% throughput
> cost. Under that run's throughput estimate it was the only tested variant
> still ahead after charging compute.

This is not yet a confirmed multi-seed headline. Phase 18 was unpaired and used
a repeatedly observed 25-batch development window; its throughput values were
single-run measurements on a shared box. Content conditioning, cross-head
mixing, and extra positional capacity appeared to lose on a compute-adjusted
basis, but the close negative iso-wallclock verdicts are below measured
cross-day throughput variation. See the phase-19 status and audit brief below.

## Current standing

h1024/d12, seed 123, 30k steps, SDPA, 25-batch development evaluation at 1024,
lr `4e-4`, betas `0.95/0.999`, batch 8:

| Cell | Loss | Gap vs RoPE | Tokens/s | Position params | Iso-wallclock vs RoPE |
| --- | ---: | ---: | ---: | ---: | ---: |
| wide trunk 256 | **3.29126** | 0.0573 | 71,075 | 11,501,568 | -0.0055 |
| free control (content+position) | 3.30178 | 0.0467 | 75,008 | 4,073,472 | -0.0014 |
| **position-only** | 3.30639 | 0.0421 | 85,285 | 1,714,176 | **+0.0289** |
| standard RoPE | 3.34852 | — | 89,536 | 0 | — |

h768/d8, same protocol, lr `3e-4`, betas `0.9/0.98` (phase17):

| Cell | Loss | Gap vs RoPE | Tokens/s |
| --- | ---: | ---: | ---: |
| wide trunk 256 | 3.55872 | 0.0667 | 140,679 |
| dense head mixing | 3.56074 | 0.0647 | 153,100 |
| slope + free phase | 3.57243 | 0.0530 | 151,630 |
| free control | 3.57480 | 0.0506 | 154,745 |
| position-only | 3.57865 | 0.0467 | 169,405 |
| standard RoPE | 3.62539 | — | 180,198 |

## Three scaling facts that should shape any new work

**1. Width appears to erode the positional advantage less than tokens do.** Going h768/d8 ->
h1024/d12 (2.7x non-embedding parameters) costs the gap only `8-14%`. Going 5k
-> 30k steps (6x tokens) **halves** it (`0.127 -> 0.0506`). So the shrinking-gap
concern may be more about training duration than model size. This is
corroborated by the older phase6 gate, where the logit stack held `0.094` over
RoPE at both sizes, but phase17/18 used different optimizer recipes and the
comparison remains confounded.

**2. Content conditioning's measured increment is ~`0.004`, but unresolved.** Position-only
sits `0.0038` behind full content+position at h768 and `0.0046` at h1024 -- near
the replication floor and not growing. Both values are single-seed, unpaired
contrasts; partial phase-19 cross-seed trajectories vary by more than this gap.
The economical working hypothesis is that the line is about a learned per-head
positional profile rather than content conditioning, but phase 19 must resolve
that statement.

**3. 5k screening is unreliable, not merely noisy.** No ordering measured inside
the conditioned group at 5k survived to 30k. Treat 5k differences under `~0.02`
as uninformative about longer horizons. Same-seed reruns replicate to `~0.0015`;
cross-seed spread is `0.002-0.027`.

## Audit brief — read this before trusting the above

A fresh pair of eyes is genuinely wanted here. The following are the places the
previous agent considers weakest, plus mistakes actually made during the work
that suggest what kinds of errors to look for.

### Known errors made during this work (all corrected, but indicative)

- **An invalid low-rank bound.** The first probe of the relative-logit matrix
  SVD'd `Toeplitz(b)` with hard zeros above the diagonal, reporting rank 64-250.
  That measured the causal mask, not the curve -- attention applies the mask
  itself, so entries above the diagonal are free. Retracted in
  `CONCAT_QK_POSITION.md`.
- **A mechanism claim refuted at longer horizon.** Phase15 concluded dense
  cross-head mixing was feature sharing rather than capacity, because it beat a
  parameter-matched wide-trunk control by `0.0125` at 5k. At 30k the wide trunk
  **wins**. The effect was real; the explanation was wrong.
- **A near-miss on iso-wallclock.** Using the 5k-30k *average* loss slope rather
  than the *local* slope at 30k would have reversed the conclusion about whether
  conditioned arms beat RoPE per unit compute. Always use the local slope from
  the last two eval points.
- **A silently ignored config key.** The launcher emitted `angular_rank` into
  every generated config after the key had been removed from the schema; a
  string-escaping bug in an earlier edit had also silently no-op'd the intended
  change. Caught by the config-loading test, not by inspection.
- **An omitted interrupted confirmation.** Phase 19 was locked and launched on
  2026-08-02, but only one run completed and its status was not added to the
  journal or the earlier handoff. The omission was caught by an independent
  artifact audit on 2026-08-05. The exact inventory is now recorded below; the
  reason the launch was terminated is not established by the surviving logs.

### Specific things worth checking

1. **Is "position-only" actually position-only?** Confirm
   `conditioning.input_mode == "position"` genuinely severs the content path and
   that `PositionContentProjection` contributes nothing to those runs. This is
   the load-bearing claim; verify it in the code, not the config.
2. **Throughput numbers.** `tokens_per_second` is read from the final logged
   value per run and includes warmup. The iso-wallclock table is only as good as
   these. A clean steady-state benchmark would strengthen or overturn the
   ordering, and the position-only margin (`+0.0289`) is the number most worth
   re-measuring.
3. **Single seed everywhere.** Every result in phases 11-18 is seed 123. The
   `~0.0015` replication floor comes from same-seed reruns of nominally identical
   configs, not from a proper seed study. Cross-seed spread was `0.002-0.027` in
   the older two-seed data. Anything claimed under `~0.01` deserves a second seed.
   Phase 19 locked a paired three-seed confirmation, but only one of its 15 runs
   completed; see "Where to pick up."
4. **The phase18 optimizer confound.** h1024 used lr `4e-4` / betas `0.95/0.999`;
   h768 used `3e-4` / `0.9/0.98`. Some of the `8-14%` gap erosion attributed to
   width could be optimizer. One h768 run at the h1024 recipe closes this.
5. **The gap-halving extrapolation** rests on two points on the token axis
   (5k, 30k) at one model size. It is used to argue the advantage may approach
   noise at ~1M steps. That is an extrapolation, not a measurement.
6. **Anchor correctness.** Every conditioning mode claims an exact
   `amplitude_init * cis(omega*p)` anchor at zero-init. Tests assert this
   (`test_position_channels.py`), but the gauge confound found in July
   (effective gains reaching 122x) came from exactly this area, so it is worth
   independent verification for the modes currently in use.
7. **Eval protocol.** Phase-17/18 "final" loss is the last evaluation on the
   same 25-batch development window used throughout training, not a large
   disjoint holdout. There was no early stopping, so this does not create
   training leakage, and all arms share the window. Phase 19 introduced a
   disjoint 1,024-example holdout; its only completed model has development loss
   `3.30964` and holdout loss `3.43016`, illustrating why absolute values from
   the two windows must not be mixed.

### Questions worth forming an independent view on

- Is the compute-adjusted framing the right one, or is it flattering
  position-only by charging the heavier arms for throughput while ignoring that
  positional parameters are a one-time cost?
- Is `0.042` at h1024 large enough to matter, given it halves per 6x tokens?
- Has the project's breadth-first search over ~20 axes actually been more
  informative than a narrow, well-powered study of two or three would have been?

## Architecture in one paragraph

Position is factored into independent channels (Q/K, logit bias, residual
stream, attention writes). The v2 Q/K channel separates **application**
(`additive` vs `rotary`), **geometry** (`amplitude_phase`, `phase`, ...),
**input basis**, **mapper**, **Q/K coupling**, and **head coupling**.
Conditioning targets Q/K carriers via a **carrier hypernetwork** that modulates
amplitude and phase from a null anchor: with zeroed readouts the channel is
exactly `amplitude_init * cis(omega*p)`. Logit-bias channels require
**FlexAttention** (~1.9x slower); Q/K carrier methods run on fused SDPA.

## Repo navigation

| Path | What it is |
| --- | --- |
| [`EXPERIMENT_JOURNAL.md`](EXPERIMENT_JOURNAL.md) | Living record, ~2,400 lines. **Read the tail from 2026-07-31.** Dated entries, never rewritten |
| [`POSITION_CONFIG.md`](POSITION_CONFIG.md) | v2 schema reference plus a status table of removed / deprecated / live axes with effect sizes |
| [`CONCAT_QK_POSITION.md`](CONCAT_QK_POSITION.md) | Why relative logit biases cannot be moved onto fused SDPA. Contains a retraction worth reading as a worked example |
| [`axes.md`](axes.md) | Original research axes, largely historical |
| [`CONFIRMATION_PROTOCOL.md`](CONFIRMATION_PROTOCOL.md) | locked phase-19 paired h1024 confirmation; currently incomplete |
| [`CONSOLIDATED_RESEARCH_PLAN.md`](CONSOLIDATED_RESEARCH_PLAN.md) | current objectives, evidence rules, execution order, gates, and conditional methods backlog |
| [`INDEPENDENT_REVIEW_BRIEF.md`](INDEPENDENT_REVIEW_BRIEF.md) | read-only prompts for code, evidence, methods, and synthesis reviews |
| [`/workspace/GPU_QUEUEING.md`](/workspace/GPU_QUEUEING.md) | **Required** before launching any GPU job. Shared box |

| Module | Role |
| --- | --- |
| `position/config.py` | v2 schema, validation, v1 upgrade. Rejects unknown keys |
| `position/channels.py` | `QKPositionChannel`, `CarrierHypernetwork`, `_MixedReadout`. ~3k lines, the core |
| `position/basis.py`, `mappers.py`, `rotary.py` | Inputs, mappers, RoPE helpers |
| `transformer.py` | Attention, SDPA vs FlexAttention dispatch, `PerHeadRMSNorm` |
| `train_gpt.py` | Training entrypoint, `make_optimizer`, `POSITION_DECAY_EXEMPT` |
| `launch_position_bias.sh` | Sweep launcher, ~880 lines after the 2026-07-31 prune |
| `position_results.py` | Pull and summarize local run metrics |
| `test_position_channels.py`, `test_position_playground.py` | 121 passing CPU tests and 1 skipped test after the frequency-controller work |
| `scripts/position_v2_cuda_smoke.py` | GPU smoke tests, eager and compiled |

The learned-frequency work has its own compact trail:

| Path | What it is |
| --- | --- |
| `ROPE_FREQUENCY_ROADMAP.md` | hypotheses, staging, and current interpretation for phases 20-23 |
| `ROPE_DYNAMIC_FREQUENCY_PROTOCOL.md` | locked phase-23 protocol, results, diagnostics, and the remaining bounded-vs-free question |
| `model-output/position_bias_phase20_rope_frequency/FREQUENCY_RESULTS.md` | direct static-frequency screen |
| `model-output/position_bias_phase21_rope_parameterization/PARAMETERIZATION_RESULTS.md` | 5k static-parameterization screen |
| `model-output/position_bias_phase22_rope_additive_30k/ADDITIVE_30K_RESULTS.md` | paired 30k additive-frequency confirmation |
| `model-output/position_bias_phase23_dynamic_frequency/DYNAMIC_FREQUENCY_RESULTS.md` | bounded token-conditioned controller screen |

Results live in `model-output/position_bias_<family>/<run>/metrics.jsonl`.
Generated configs live in `sweep_configs/<family>/`. A test asserts every config
on disk still loads, which is how schema drift gets caught.

## Settled negatives

**Closed by mechanism** -- these get *worse* with scale, not better. Removed
from the codebase:

- **Content-conditioned frequency multipliers.** `4.5369` / `4.7788` vs a
  `4.2840` control. Content-dependent `omega` makes the logit depend on absolute
  position with error growing as `m*p`; a multiplier bounded by `epsilon` needs
  `epsilon < 1e-4` to keep drift under 0.1 rad at L=1024. The *normalized* form
  (divide by `p`) is translation-preserving and survives as `position_offset`.
- **Relative logit biases on fused SDPA.** Learned biases are log-shaped
  per-head decay profiles: not spectrally compact (DCT rank-16 leaves 18-26%
  error), not low-rank, and the closed form that fits them is not separable.
  Materializing `[H, L, L]` is `137 GB` at h=64/L=32k. See
  `CONCAT_QK_POSITION.md`.

**Lost at 5k only** -- kept in the codebase, default off, could plausibly flip
at longer horizons. Effect sizes in `POSITION_CONFIG.md`: per-head QKNorm,
`offset_parameterization` raw/softplus, isolated narrow readouts, position-only
warps, low-rank cross-head mixing.

**Ruled out with no effect:** weight decay on zero-anchored position parameters
(all arms within `0.0007` of decayed counterparts). `exclude_position_from_decay`
exists and is the correct default but changes nothing measurable.

## Structural findings worth keeping

- The carrier gain lives almost entirely in the **amplitude** branch
  (amplitude-only `4.3024` vs full `4.2840`, phase-only `4.4348` -- worse than
  RoPE). Angular conditioning only helps alongside amplitude.
- Amplitude compresses to **2 scalars** nearly losslessly (`+0.0035`); the
  angular branch is genuinely **~rank 30 of 48** (SVD of trained readouts) and
  does not compress.
- **Effective angular rank falls with depth** (`~18` at layer 0 to `~9-12` at
  layer 7), matching the independent finding that the fitted decay `tau` of the
  logit curves *grows* with depth (`7.2 -> 12.7 -> 37.9`).
- Extra capacity that is a no-op at init can still hurt, via optimization rather
  than expressivity: per-head QKNorm splits one gradient-averaged parameter into
  eight noisier ones; low-rank factorizations replace one zero-init matrix with a
  zero-times-random product. Both were identical at step 0 and worse at step 5000.

## How to run

```bash
cd /workspace/MLPRope

# CPU tests -- run these first; they catch schema drift and null-anchor errors
/venv/main/bin/python -m unittest test_position_channels test_position_playground

# GPU smoke, eager and compiled
gpu-claim run --owner mlprope --job position-smoke --wait -- \
  /venv/main/bin/python -u scripts/position_v2_cuda_smoke.py

# Validate a config without training
/venv/main/bin/python train_gpt.py --override_json <cfg.json> --dry_run

# Launch a sweep
export EXPERIMENT_FAMILY=phase18_scale_h1024
export WITH_TRACKING=true SUBMIT_JOBS=true PARALLEL=true
./launch_position_bias.sh

# Pull results
/venv/main/bin/python position_results.py \
  model-output/position_bias_phase18_scale_h1024 --format markdown
```

**Always** go through `gpu-claim` per `/workspace/GPU_QUEUEING.md` -- other
projects share this box and will collide with you otherwise. Long sweeps belong
under **supervisor** (`/opt/supervisor-scripts/mlprope-phase18-scale-h1024.sh`
is the current template) so they survive terminal disconnects.

Model size, lr, and betas are family-scoped near the top of
`launch_position_bias.sh`, so a new family can change them without disturbing
the settings older families' results were produced under.

## Launcher state

Pruned 2026-07-31 from 3,008 lines to ~880. Live families: `phase11_spectral`,
`phase13_decomp_qknorm`, `phase15_decay_mixing`, `phase16_cheap_mixing`,
`phase17_gate_30k`, `phase18_scale_h1024`. Historical families (phase1-phase10,
phase12, phase14) were removed; their generated configs remain under
`sweep_configs/` and the generating blocks are recoverable from git history.
Nineteen configs referencing deleted modes were removed and need a git checkout
to re-run.

## Where to pick up

Ordered by current information value after the independent audit.

1. **Complete the phase-19 headline gate.** Phase 19 locked 15 paired h1024/d12
   runs across three seeds and a disjoint 1,024-example holdout. Only
   `position-only/seed123` completed. Five other jobs began but were interrupted
   (`content-position` reached 29,350/29,491/29,988; `standard-rope/seed456`
   reached 21,449; `mapped-addrope/seed456` reached 1,307), and nine never began.
   The protocol used no intermediate checkpoints, so none of the interrupted
   work is resumable. The first decisive tranche is standard RoPE seeds
   123/456/789 plus position-only seeds 456/789; the completed position-only
   seed123 can be reused. If that confirms the headline, finish the mapped,
   content, and matched-FFN contrasts required by the locked protocol.
2. **Clean throughput benchmark.** Measure the four h1024 phase-18 arms on the
   same GPU, without eval/profiling/tracking in the timed region, with
   interleaved repetitions. Position-only's raw margin is probably robust, but
   the exact 4.7% cost and the negative free/wide iso-wallclock conclusions are
   not.
3. **De-confound the scale comparison.** Run both standard RoPE and
   position-only at h768/d8 under the h1024 recipe (lr `4e-4`, betas
   `0.95/0.999`, 8 heads). A single run cannot isolate an optimizer effect on a
   two-arm loss gap.
4. **Cheap phase-23 checkpoint ablations.** Zero, sequence-mean, and
   token-shuffle controller outputs on the saved three-seed checkpoints before
   training another frequency variant. These require inference GPU time but no
   retraining. They distinguish endpoint reliance on same-token conditioning
   from a quasi-static schedule; they do not by themselves prove what caused a
   training-time improvement.
5. **Then decide whether to stop and write up.** The free-vs-bounded controller
   question is conceptually real, but existing low-rank outputs were mostly in
   tanh's near-linear region, so it ranks below the headline confirmation.

Phase 20 subsequently tested direct static base-RoPE frequency learning across
three paired seeds. Layer-shared and layer-head schedules each improved 1024
loss by about `0.0019`, while head-specific frequencies added nothing over a
layer-shared schedule. This is reproducible but below the `0.01` materiality
gate. The subsequent static and bounded-dynamic questions are recorded in
`ROPE_FREQUENCY_ROADMAP.md`; both screens are now complete.

Phases 21-22 resolved the static parameterization question. Free additive
frequency (`omega = omega0 + u`) appeared promising at 5k (`-0.011992` mean,
all three seeds favorable), but at 30k it fell to `-0.001593` with mixed signs
(`-0.004342`, `+0.004510`, `-0.004947`). Full identity-backward exp, softplus,
and bounded-log parameterizations were already below the 5k materiality gate.
At 30k, additive frequencies were finite but not benign-looking: `11-12%` were
negative and p95 extra phase at position 1024 reached `42-45` radians. There was
no numerical collapse, but describing this only as "sane spectra" understates
the winding and sign reversal. Together with the null loss result, this is
evidence for a poorly identified/flat angular direction rather than a promising
schedule. Static learned base frequencies are closed at h768/d8.

Phase 23 tested that distinct dynamic hypothesis using the normalized residual,
Q/K-shared per-head outputs, and an extra phase limited by `t/1024` times one
radian. Full linear was worse than fixed RoPE (`+0.003057` mean). Rank-32 linear
and rank-32 SiLU were favorable in all three seeds but only by `-0.001893` and
`-0.002063`; neither approached the locked `-0.01` promotion gate. The low-rank
controllers were not broadly pinned at the limit (maximum per-layer raw RMS
`0.27-0.40`, maximum per-layer phase p95 `0.32-0.42` rad), although rare token/
pair extrema did approach the one-radian bound. Per the locked protocol, Q/K
separation, head sharing, source/rank sweeps, and the phase-residual control were
not opened.

The remaining methodological question is whether the one-radian `tanh` trust
region was justified. It was a conservative safety choice, not a property of
rotary phase. Before new training, use checkpoint ablations to determine whether
same-token controller alignment matters. If a later focused reopening is
justified, retain the best rank-32 SiLU mapper and the same three seeds and
change only the output map. The cleanest alternative is a free
horizon-normalized phase,
`delta_phase = (t / 1024) * raw(norm_x)`, with diagnostics for raw magnitude and
phase winding. A rational squash or clamp-with-STE can test optimizer behavior,
but both retain an arbitrary output boundary and therefore do not answer the
more fundamental free-vs-bounded question.

Do **not** re-open logit-bias/SDPA reformulations or the old raw multiplicative
content-conditioned frequency arms without new information. A free
horizon-normalized phase is not identical to those old arms: its controller
gradient is scaled by `t/1024`, and it changes phase additively rather than
multiplying each base frequency. It still breaks joint-shift equivariance and
must not be described as a translation-relative RoPE schedule.

## Transcript

Full prior conversation, including code walkthroughs, design rationale, the
reasoning behind the settled negatives, and the corrections listed in the audit
brief:
`/root/.cursor/projects/workspace/agent-transcripts/76683046-c950-4db2-8490-e97bb84a9e67/76683046-c950-4db2-8490-e97bb84a9e67.jsonl`
