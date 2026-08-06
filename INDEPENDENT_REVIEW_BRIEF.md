# Independent review brief for MLPRope

_Prepared 2026-08-05. This is a read-only review request. Do not edit files,
launch training, or treat the project's own interpretations as ground truth._

## Purpose

We want an independent assessment of a positional-encoding research codebase:

1. implementation correctness;
2. validity and strength of the experimental findings;
3. conceptual interpretation of the mechanisms and null results;
4. the smallest high-value next steps, including whether the right action is to
   stop experimenting and write up.

Please look actively for errors and confounds, but do not assume that something
must be wrong merely because results are messy or negative. Separate verified
facts, plausible interpretations, open questions, and speculation.

## Research scope

The repository is `/workspace/MLPRope`. It explores positional mechanisms in a
small GPT-style transformer trained on OpenWebText with the GPT-2 tokenizer.
The current primary endpoint is held-out loss at context 1024. Length
extrapolation is not a current target, so please do not make 2048/4096 evaluation
the default recommendation unless it tests a mechanism relevant at 1024.

Two broad findings currently motivate the work:

- A learned position-only additive carrier has beaten standard RoPE in the
  existing h768/d8 and h1024/d12 experiments, though the headline h1024 evidence
  and throughput adjustment have important replication/confounding concerns.
- Directly learned static RoPE frequencies and a bounded token-conditioned phase
  controller have produced only small or null effects under paired multi-seed
  tests. The bounded controller used an arguably arbitrary one-radian `tanh`
  trust region, so free versus bounded output remains open.

These statements summarize reported evidence; they are not conclusions you are
being asked to endorse.

## Suggested reading order

Start with:

1. `HANDOFF.md` -- project overview and an explicit audit brief. Treat it as a
   map and a list of claims to verify, not as evidence.
2. `POSITION_CONFIG.md` -- live/deprecated configuration surface and historical
   effect-size summary.
3. `ROPE_FREQUENCY_ROADMAP.md` -- hypotheses and current interpretation for
   phases 20-23.
4. `ROPE_DYNAMIC_FREQUENCY_PROTOCOL.md` -- phase-23 mechanism, locked protocol,
   results, diagnostics, and the bounded/free issue.

For implementation review, inspect at least:

- `position/frequency.py`
- `position/rotary.py`
- `transformer.py`
- `train_gpt.py`
- `position/config.py`
- `test_position_playground.py`
- `test_position_channels.py`
- `scripts/rope_frequency_parameterization_cuda_smoke.py`

For the recent learned-frequency evidence, inspect rather than merely trusting
the Markdown summaries:

- `sweep_configs/phase20_rope_frequency/`
- `sweep_configs/phase21_rope_parameterization/`
- `sweep_configs/phase22_rope_additive_30k/`
- `sweep_configs/phase23_dynamic_frequency/`
- `prepare_rope_frequency_screen.py`
- `prepare_rope_frequency_parameterization_screen.py`
- `prepare_rope_frequency_additive_30k.py`
- `prepare_rope_dynamic_frequency_screen.py`
- `analyze_rope_frequency_screen.py`
- `analyze_rope_frequency_parameterization_screen.py`
- `analyze_rope_frequency_additive_30k.py`
- `analyze_rope_dynamic_frequency_screen.py`
- `model-output/position_bias_phase20_rope_frequency/`
- `model-output/position_bias_phase21_rope_parameterization/`
- `model-output/position_bias_phase22_rope_additive_30k/`
- `model-output/position_bias_phase23_dynamic_frequency/`

The result directories contain configs, `metrics.jsonl`, final evaluation
details, checkpoints, profiles, Markdown summaries, and machine-readable
analysis JSON. `EXPERIMENT_JOURNAL.md` is the longer historical record; read its
recent relevant sections after forming an initial view, since it contains many
retired hypotheses and the authors' evolving interpretations.

## Common instructions for every reviewer

- Cite concrete evidence using file paths and line numbers or exact result
  artifacts. If you run a read-only command, include enough of it to reproduce
  the observation.
- Check semantic behavior in code, not just configuration names. In particular,
  distinguish frequency changes, horizon-scaled phase changes, and
  position-independent phase residuals.
- Check the actual train/eval configuration, seed pairing, initialization, data
  ordering, holdout ranges, step counts, optimizer schedules, parameter counts,
  and throughput measurement.
- Treat 5k screens as failure/materiality screens. Historically, differences
  around `0.01` at 5k have changed or vanished by 30k.
- Do not infer a mechanism from an ordering when capacity, optimization, or
  throughput is an equally viable explanation.
- Treat null and negative results as evidence. Do not recommend a large grid
  merely because more axes exist.
- For every proposed experiment, state the falsifiable prediction, necessary
  controls, primary metric, minimum meaningful effect, and stopping rule.
- Note any conclusion that depends on a subjective research value choice rather
  than the evidence.

## Review A: implementation and numerical correctness

Please perform a red-team code audit. Inspect implementation and tests before
reading the reported loss ordering if practical.

Questions to answer include:

1. Are RoPE positions, frequency tables, angle multiplication, and trigonometry
   genuinely fp32 under autocast and after `.half()`/`.bfloat16()` conversion,
   with casting only when applying the rotation?
2. Do all learned-frequency parameterizations have the documented forward maps,
   gradients, shapes, and exact fixed-RoPE anchors?
3. For token-conditioned controllers, is the input truly the normalized
   residual vector at the same token, with no future-token or unintended scalar
   norm path? Are Q/K sharing and per-head dimensionality implemented as claimed?
4. Does zeroing the final projection survive the model-wide initializer, and do
   all intended layers receive useful gradients?
5. Is the current formula actually
   `phase=t*omega_base+(t/reference_length)*bound*tanh(raw)`? Are bounds and fp32
   arithmetic applied at the intended point?
6. Are controller parameters counted, checkpointed, optimized, and excluded
   from weight decay as documented? Could parameter-group logic silently omit or
   duplicate anything?
7. Do diagnostics measure the semantic quantities their names imply? Are there
   missing diagnostics that could reverse an interpretation?
8. Can reused fixed controls from phase 20 be compared fairly with phase 23
   after intervening code changes? Verify exact relevant configs, paired
   initialization, data/eval slices, and fixed-model behavior.

Deliver a severity-ranked list of confirmed bugs, plausible risks, and checks
that passed. Do not propose architectural extensions unless they follow from a
specific correctness finding.

## Review B: experimental design and statistical claims

Please audit whether the evidence supports the reported claims.

Questions to answer include:

1. Which comparisons are genuinely paired, replicated, and evaluated on
   disjoint data? Which reuse controls or share enough experimental state to
   create dependence?
2. Are the analysis scripts computing signs, means, confidence intervals, and
   throughput summaries correctly from the intended final evaluations?
3. Does the `0.01` promotion threshold make sense relative to observed
   same-seed repeatability, across-seed variance, multiple comparisons, and the
   historical 5k-to-30k instability?
4. Which findings are durable enough to state, which should be called screening
   results, and which are below resolution?
5. Is the compute-adjusted headline about the position-only carrier supported by
   the throughput measurement and local learning-curve interpolation?
6. What is the smallest replication or benchmark set that most improves the
   credibility of the project?

Return a claim table with columns: claim, evidence, status (`supported`,
`qualified`, `unsupported`, or `contradicted`), and the cheapest decisive check.

## Review C: methods and mechanism

Please form an independent technical view rather than extending the current
roadmap by default.

Focus on:

1. What capabilities are unique to the successful additive position carrier,
   versus strict rotary transformations, learned static spectra, relative logit
   biases, or token-conditioned phase?
2. Which apparent degrees of freedom are absorbable into Q/K projections,
   redundant under symmetries, or identifiable only up to phase wrapping?
3. Is a token-conditioned horizon-scaled phase a coherent hypothesis at context
   1024? What inductive bias, if any, supports a one-radian bound?
4. Compare these output maps without assuming one must win:
   - bounded `tanh`;
   - free `(t/L_ref)*raw`;
   - rationally bounded `raw/sqrt(1+raw^2)`;
   - hard clamp with straight-through backward;
   - a penalty or learned scale instead of a hard bound.
   Discuss forward expressivity, gradient geometry, identifiability, possible
   phase winding, and behavior outside the reference context separately.
5. Do the phase-20 to phase-23 null/small results substantially weaken the
   learned-frequency hypothesis, or do they leave a sharply different untested
   formulation? State what observation would discriminate these views.
6. Are there simpler explanations for the observed low-rank advantage over the
   full linear controller, such as optimization or implicit regularization?

Recommend no more than three next experiments. It is acceptable, and useful, to
recommend stopping this line.

## Review D: synthesis and research prioritization

Read the other reviews if available, resolve disagreements against the actual
artifacts, and rank the next actions by expected information value rather than
novelty.

Compare at least:

- a focused free-vs-bounded phase-controller control;
- a second-seed h1024 replication of the main position-only result;
- a clean steady-state throughput benchmark;
- an optimizer deconfound for the width comparison;
- stopping experiments and producing a careful write-up.

Return a short decision memo with assumptions, expected cost, what each action
could change in the project's conclusions, and a recommended order. Preserve
meaningful dissent rather than forcing consensus.

## Requested response format

1. Executive assessment in at most ten sentences.
2. Verified observations with evidence.
3. Bugs or confounds, severity-ranked.
4. Claims that need qualification.
5. Mechanistic interpretation and credible alternatives.
6. Zero to three recommended experiments, each with prediction, controls,
   endpoint, materiality threshold, and stopping rule.
7. Questions for the project owner.

