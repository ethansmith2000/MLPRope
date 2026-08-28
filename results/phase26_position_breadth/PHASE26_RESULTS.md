# Phase-26 dynamic-position breadth screen

This is a paired seed-123 mechanism screen, not a multi-seed result.
Loss uses the disjoint 256-example holdout beginning at validation
batch 2,048. Deltas are candidate minus its direct control; negative
is better. Only clear survivors should receive seed replication.

| Arm | Final loss | Target tok/s | Direct control | Delta | Triage |
| --- | ---: | ---: | --- | ---: | --- |
| rope-fixed | 4.598100 | 180,613 | — | — | control |
| nope | 4.972125 | 187,228 | rope-fixed | +0.374026 | prune |
| addrope-a10 | 4.430811 | 164,390 | rope-fixed | -0.167289 | survive |
| posgain-q | 4.568053 | 175,870 | rope-fixed | -0.030047 | survive |
| posgain-k | 4.569350 | 177,803 | rope-fixed | -0.028750 | survive |
| posgain-qk | 4.560987 | 180,293 | rope-fixed | -0.037113 | survive |
| qkpre-nope | 4.624839 | 182,694 | nope | -0.347287 | survive |
| qkpre-rope | 4.516569 | 185,365 | rope-fixed | -0.081531 | survive |
| clock-pointwise | 4.593802 | 169,347 | rope-fixed | -0.004298 | unresolved |
| clock-causalconv | 4.593422 | 174,904 | rope-fixed | -0.004678 | unresolved |

The JSON companion contains paired-example confidence intervals
and layer-aggregated mechanism-health diagnostics.
