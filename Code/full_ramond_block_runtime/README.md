# Ramond block q-expansion runtime benchmark

## Authoritative calculation

`compute_q_expansion.py` computes the coefficient-by-coefficient,
three-variable q-expansion of

\[
\widehat{\mathbb F}^{(+,+)}_0(q_1,q_2,q_3)
\]

through total plumbing level 10. This is not a fixed-q evaluation and is not
the `q_1 -> 0` degeneration. The output retains the parity of every `eta_i`,
so it can be evaluated for all eight tube-sign assignments.

An exponent `(e1,e2,e3)` in the JSON denotes

\[
q_1^{e_1/2}q_2^{e_2/2}q_3^{e_3/2},
\]

and only terms with `e1+e2+e3 <= 20` are retained. The generic branching
point is

\[
b=\frac75,\qquad
(P_1,P_2,P_3)=\left(\frac{11}{23},\frac{13}{29},\frac{17}{31}\right),
\]

with `f=0` and `eta=eta'=+` for the two three-point structures.

## Algorithm

The code constructs all required `L_{+1}` NS and `L_{-1}` Ramond branch-state
decompositions, then closes the first `L_1` Ward identity using the directly
computed low-level branching coefficients as boundary data. For each branch
triple, its primary propagation level is subtracted first. If that shift is
`s`, each ordinary Virasoro block is computed only through descendant level
`10-s`. The two Virasoro series are then multiplied with a combined cutoff,
so no term above total level 10 survives.

The ordinary Virasoro blocks are coefficient dictionaries produced by the CCY
central-charge recursion. The universal large-c vacuum series is computed
once and its square is multiplied into the final double-Virasoro sum. No
generic PBW conformal-block sum is used. A representative ordinary Virasoro
series was checked coefficientwise against its direct descendant sum through
level 2; the maximum absolute error was `1.70e-15`.

## Adaptive Virasoro cutoffs

There are 388 branch-label triples and 776 ordinary Virasoro blocks.

| Descendant cutoff | Virasoro blocks |
| ---: | ---: |
| 10 | 8 |
| 9 | 32 |
| 8 | 56 |
| 7 | 64 |
| 6 | 64 |
| 5 | 80 |
| 4 | 88 |
| 3 | 80 |
| 2 | 128 |
| 1 | 112 |
| 0 | 64 |

## Level-10 runtime

The timed command was

```sh
python3 python/full_ramond_block_runtime/compute_q_expansion.py \
  --cutoff 10 \
  --json python/full_ramond_block_runtime/level10_q_expansion.json
```

It ran serially with Python 3.14.3 on macOS arm64.

| Stage | Seconds |
| --- | ---: |
| Branch action decompositions | 75.0996 |
| Four branching Ward systems | 3.2103 |
| Formal large-c vacuum series | 0.0193 |
| 776 adaptive Virasoro c-recursions | 4.4920 |
| Formal double-Virasoro assembly | 0.0848 |
| Low-level validation | 0.0009 |
| Cold total | 82.9134 |

With the branching coefficients already available, producing the complete
q-expansion takes `4.5961` seconds. The output contains 1,012 nonzero
coefficients after separating the `eta_i` parities.

As a numerical evaluation check only, setting
`(q1,q2,q3)=(0.019,0.023,0.029)` and all tube signs to `+1` gives

\[
4.119743437973190-1.06\times10^{-15}i.
\]

The imaginary part is roundoff. Comparing the independently generated
level-4 expansion with the restriction of the level-10 expansion gives a
maximum scaled coefficient difference of `1.18e-7`; this is governed by the
conditioning of the high-level branching Ward system. The largest Ward
residual in the level-10 run is `7.83e-10`.

## Files

- `compute_q_expansion.py`: formal q-series implementation.
- `level10_q_expansion.json`: all coefficients, parity labels, timings, and
  diagnostics.
- `compute_full_block.py` and `level10_results.json`: the earlier fixed-q
  diagnostic. These do not constitute a q-expansion and their runtime should
  not be quoted as the q-expansion runtime.
