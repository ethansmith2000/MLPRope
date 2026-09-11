# Phase 51: trained carrier structure

This is a weight-only diagnostic. It does not establish that the positional
mean causes the loss improvement; the registered checkpoint counterfactual
evaluations test that separately.

The reported fraction is
`||mean_p C(p)||^2 / mean_p ||C(p)||^2` over the 1,023 positions actually
entering attention for each 1,024-token language-model block.

| Seed | Q direct mean-energy range | K direct mean-energy range | Q scalar/direct RMS range | K scalar/direct RMS range |
|---:|---:|---:|---:|---:|
| 123 | 96.54%--99.01% | 98.76%--99.52% | 0.00156--0.03724 | 0.00135--0.02312 |
| 456 | 94.65%--99.52% | 97.88%--99.56% | 0.00179--0.03464 | 0.00149--0.02132 |
| 789 | 97.24%--99.22% | 98.72%--99.46% | 0.00124--0.03523 | 0.00119--0.02469 |

## Interpretation

A direct carrier dominated by its positional mean is close to a learned
constant Q/K vector before QK normalization and RoPE. RoPE rotates such a
vector by position, so it is not functionally position-free: its pure
position--position numerator is a relative Toeplitz kernel
`b_q^T R(r-p) b_k`. It can also create content--position cross terms.

The very small scalar/direct RMS ratios show that the mature rank-32
extension is structurally dominated by its dedicated projected branch.
They do not by themselves prove that the scalar anchor is causally
unnecessary, because QK normalization and all downstream hidden states
change jointly when either branch is removed.

## Decision use

- If positional-mean-only evaluation preserves the gain and the
  mean-removed component does not, train a matched constant-carrier/QK-bias
  control before claiming that a rich Fourier readout is responsible.
- If mean removal preserves the gain, retain the Fourier interpretation and
  characterize the centered component.
- If scalar removal is neutral only at evaluation but a no-anchor training
  control loses, interpret the scalar as optimization scaffolding.
