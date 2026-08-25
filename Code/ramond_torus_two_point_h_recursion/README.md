# Ramond torus two-point h-recursion check

This folder implements the `q1 -> 0` cross-check without an SCA PBW block
calculation.  It compares

1. the enlarged Ramond torus two-point block reconstructed from its Ramond
   internal-weight poles, and
2. the `q1 = 0` restriction of the independently computed double-Virasoro
   theta block.

The generic point is

\[
b=\frac75,\qquad
(P_1,P_2,P_3)=\left(\frac{11}{23},\frac{13}{29},\frac{17}{31}\right),
\]

with `f=0`, `eta=eta'=+`, and positive tube signs.

## Algorithm

The Ramond momenta are uniformized by

\[
t=\beta_2+\beta_3,\qquad
a=\beta_2^2-\beta_3^2,
\]

so every coefficient is a rational function of `t`.  Both signs of each
degenerate Ramond momentum and both sheets of the neighboring momentum are
included.  Reflection through a negative pole changes the fusion label by
`eta -> -eta` and shifts the degenerate momentum by
`beta_prime -> -beta_prime`.  The inverse null norm uses the even lattice
`p+q in 2 Z`, as required by the two-ground-state BPZ convention of the main
notes.

The recurrence is evaluated first in the standard necklace variables, where
the simultaneous-large-weight seed is

\[
4\prod_{n\geq1}\frac{(1+Q^n)^2}{1-Q^n}.
\]

The formal change to the theta plumbing frame is extracted from the closed
`L_-1` global block:

\[
\widetilde q_2=q_2\frac{X}{Y},\qquad
\widetilde q_3=q_3Y,\qquad
\widehat{\mathbb F}^{\theta}
=X^{H+1/16}Y^aD^{h_1}\widehat{\mathbb F}^{\rm necklace}.
\]

The internal check `Q = q2 q3 X` holds through total level 10 with maximum
coefficient error `7.95e-35`.

## Total-level-10 result

The command

```sh
python3 python/ramond_torus_two_point_h_recursion/check_h_recursion.py \
  --cutoff 10 --precision 40 \
  --output python/ramond_torus_two_point_h_recursion/results.json
```

compares all 66 coefficients with `N2+N3 <= 10`.

| Quantity | Result |
| --- | ---: |
| Coefficients compared | 66 |
| Maximum absolute difference | `3.81629e-3` |
| Maximum relative difference | `5.49088e-7` |
| Worst levels | `(9,1)` |
| Seed extraction | `1.2601 s` |
| Ramond h-recursion | `85.3064 s` |
| Necklace-to-theta conversion | `0.0340 s` |
| Total runtime | `86.6100 s` |

The largest absolute difference occurs at `(5,5)`, where the coefficient is
about `3.05e5`; its relative difference is `1.25e-8`.  The worst relative
difference occurs at `(9,1)`.  These discrepancies are consistent with the
conditioning error already present in the numerical branching-coefficient
side.

## Files

- `check_h_recursion.py`: standalone h-recursion and comparison.
- `results.json`: all 66 coefficient pairs, diagnostics, and timings.

No SCA PBW coefficient enters this calculation.
