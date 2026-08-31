# Ramond block q-expansion runtime benchmark

## Physical-block boundary audit, 2026-08-30

The certified enlarged series below must not be confused with a physical
fixed-lift block or with a nonchiral partition function. The corrected
`nsrr_double_virasoro_block.py` now:

- divides the four supported auxiliary-star characters and reconstructs
  parity components before taking the **ordinary** lift sum;
- supplies the opposite-HJS-sign nullspace explicitly from the independent
  PBW oracle (default cost cap: physical level 3);
- obtains the odd form from the exact Ramond ground-partner Ward identity
  `F_1 = -i e_(001) star F_0`, applied to the certified even-form series.
  The old odd-form branching-grid extension is not used: its level-1/2
  coefficient interchanges the HJS signs relative to the physical PBW block.

Thus the supported data still use the branching recursion and the product
of two ordinary Virasoro c-recursions. Missing star channels are not claimed
to be a second double-Virasoro determination. The new level-3 certificate
checks both primary parities, both form parities and all HJS sign pairs.
See `Data Set/nsrr_nsnsns_human_note_repair_20260830/README.md` from the
repository root. The nonchiral NSRR comparison is guarded until its Ramond
ground projector and marked spin-lift dictionary are established.

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

Passing `--direct-pbw-check` additionally builds the physical NS--R--R block
from the Human-Note PBW Gram matrices and Ward identities, sews it with the
auxiliary Majorana block, and compares it coefficientwise with this production
series.  It leaves all three plumbing variables unrestricted.  Passing
`--primary-parity 1` performs the same computation for an intrinsically odd NS
primary.  The `L_{\pm1}` recursion matrices are unchanged; every direct
boundary anchor is instead recomputed using the selected primary parity.

Passing `--branching-mp-dps 50` evaluates the free-field action
decompositions with `mpmath` and solves each finite Ward grid by
multiprecision residual refinement around a binary64 QR factorization. The
directly computed low-state PBW anchors are deliberately left in their
independently certified binary64 implementation. This option is useful once
conditioning, rather than algebraic signs, limits the coefficient
comparison.

The current Human Note defines the enlarged and auxiliary boxes with the
first-tube signs

\[
\widehat{\mathbb F}:\ (-1)^{A+\mathsf A},
\qquad
\mathbb F_{\mathsf F}:\ (-1)^{\mathsf A}.
\]

On a double-Virasoro branch primary, `A+mathsf_A = 2*n_1 (mod 2)`.  The
production branch sum therefore includes `(-1)^(2*n_1)`, and the auxiliary
PBW sum includes `(-1)^mathsf_A` directly.  The comparison is now simply
`Fhat = F_F star_R F_PBW`: there is no post-processing frame adapter and no
extra sign in the physical PBW oracle.

The level-six certification command is

```sh
python3 Code/full_ramond_block_runtime/compute_q_expansion.py \
  --cutoff 6 \
  --direct-pbw-check \
  --json /tmp/type0b_full_q6_pbw_double_virasoro.json
```

It checks 1,120 parity-resolved entries through `e1+e2+e3 <= 12`: 280 are
nonzero production coefficients, and the remaining required zero components
are checked as well.  The maximum absolute and scaled relative discrepancy is
`9.4132e-10`; the worst entry is `(e1,e2,e3)=(10,0,2)`, parity-component
index `0`.  Adding `--primary-parity 1` checks the same 1,120 entries and gives
a maximum discrepancy of `4.131e-10`.  These values are obtained using the
current Human-note definitions directly, not by converting the output after
the calculation.

## Level-seven multiprecision certification

The unrestricted level-seven comparison is

```sh
python3 Code/full_ramond_block_runtime/compute_q_expansion.py \
  --cutoff 7 \
  --branching-mp-dps 50 \
  --direct-pbw-check \
  --primary-parity 0 \
  --json /tmp/type0b_full_q7_p0_mp50.json
```

It checks all 1,632 parity-resolved entries through
`e1+e2+e3 <= 14`. The maximum absolute discrepancy is `2.0533e-10`, and
the maximum scaled relative discrepancy is `9.6783e-11`. Repeating the
command with `--primary-parity 1` gives `6.0907e-11` and `3.1944e-11`,
respectively. Thus both intrinsic NS parities satisfy the strict `1e-9`
maximum-absolute-error standard at level seven. The corresponding binary64
Ward solve had errors of order `2e-8`; their disappearance under precision
refinement identifies conditioning, not a parity sign, as the source of that
earlier discrepancy.

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
python3 Code/full_ramond_block_runtime/compute_q_expansion.py \
  --cutoff 10 \
  --json Code/full_ramond_block_runtime/level10_q_expansion.json
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
- `../double_virasoro/nsrr/nsrr_genus2_block.py`: independent PBW and
  auxiliary-Majorana comparison oracle.
- `../ramond_branching_recursion/`: Yuchen's production branching grid and
  direct boundary code.
- `level10_q_expansion.json`: all coefficients, parity labels, timings, and
  diagnostics.
- `compute_full_block.py` and `level10_results.json`: the earlier fixed-q
  diagnostic. These do not constitute a q-expansion and their runtime should
  not be quoted as the q-expansion runtime.
