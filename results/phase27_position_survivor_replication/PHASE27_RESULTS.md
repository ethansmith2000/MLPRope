# Phase-27 position-survivor replication

Seed 123 comes from the frozen Phase-26 r3 screen; seeds 456 and 789
are fresh Phase-27 runs with an identical core source and protocol.
Deltas are candidate minus reference; negative is better.

| Arm | Seed 123 | Seed 456 | Seed 789 | Mean | Median target tok/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| rope-fixed | 4.598100 | 4.550260 | 4.587592 | 4.578651 | 187,326 |
| qkpre-rope | 4.516569 | 4.491122 | 4.520895 | 4.509529 | 185,365 |
| posgain-qk | 4.560987 | 4.534077 | 4.568245 | 4.554436 | 180,293 |

| Contrast | Seed 123 | Seed 456 | Seed 789 | Mean | All wins? | Gate? |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| qkpre-rope_vs_rope-fixed | -0.081531 | -0.059138 | -0.066697 | -0.069122 | True | True |
| posgain-qk_vs_rope-fixed | -0.037113 | -0.016184 | -0.019347 | -0.024215 | True | True |
| qkpre-rope_vs_posgain-qk | -0.044417 | -0.042955 | -0.047350 | -0.044907 | True | True |

The JSON companion contains per-seed paired-example confidence
intervals, seed dispersion, protocol fingerprints, throughput, and
mechanism-health diagnostics.
