# Vertex transport and the physical NS sewing sign: directed low-level tests

The corrected double-Virasoro candidate now passes direct physical PBW
comparisons through total level 3. It evaluates the human SCA vertex and
applies the contour transport explicitly, rather than using the native
production Ward evaluator. Both `f=0,1` and `eta eta'=+/-1` were tested at
40-digit precision, with maximum scaled error below `7.4e-28` (240 parity
components in each of four cases). The repaired local Virasoro Ward
identity has absolute residual below `4.1e-33`.

Production headers and executables were not changed. The experimental
executable defaults to `--vertex transported --ns-branch-sign keep`.
The unphased results below remain negative controls, not the final status.

## Required vertex and matching sewing convention

For the PBW arguments in the manuscript, the full transported vertex is

\[
\hat\rho_f^{(\eta)}=
(-i)^{A\bmod2}
(-1)^{f(A+B+|\alpha|)+A\mathsf A+(B+|\alpha|+p_1)(\mathsf C+\mathsf c)}
\rho_{\mathsf F}\rho_f^{(\eta)}.
\]

Here A and B count physical G modes; alpha is the physical second-slot
Ramond ground label. For an even NS primary, the extra phase beyond the
literal Koszul product is `(-i)^(A mod 2) (-1)^[f(A+B+|alpha|)]`.
For a general primary parity, the transport relative to the literal
Koszul product includes the additional `(-1)^[p1(mathsfB+mathsfb)]`;
the displayed full formula incorporates it using auxiliary parity conservation.
The full pipeline comparisons in this experiment use `p1=0`.

The helper `vertex_transport` in the experimental `anchors.hpp` implements
this map. The Ramond ground-basis conversion remains separate and unchanged.
The sign label of the human form is eta; the production control instead
receives its native `(-1)^f eta` label. The corrected path does not apply
that native label conversion a second time.

Using this same transported vertex at both ends produces
`(-i)^(A mod 2) (-i)^(A' mod 2)=(-1)^A` under the physical NS inverse Gram
contraction. Consequently, a PBW formula written using **this** vertex
needs the original `(-1)^(A+mathsfA)` sewing factor. Removing `(-1)^A`
works for the literal, unphased PBW vertex, but that literal vertex is not
the Virasoro intertwiner for the retained embeddings. One cannot identify
these two vertex conventions without the transport.

The corrected double-Virasoro sum retains `(-1)^(2 n1)` and the existing
primary norms; no CCY recursion, Schottky seed, or auxiliary divisor was
changed. `results/corrected_summary.json` records the corrected comparisons,
commands, timings and source hashes. `results/vertex_ward_transported.json`
records the local Ward check before and after transport.

## Definitions tested

The literal vertex is the human SCA form times the auxiliary fermion form,
with the Koszul sign from moving the physical operators past the auxiliary
operators. There is no `(-i)^(A mod 2)` contour phase. Physical Ward values
are evaluated using `ScaWard`, including the conversion between the native
Ramond ground basis and the human `w+`, `w-` basis.

`literal_pbw.cpp` explicitly assembles the enlarged PBW vertices, contracts
physical inverse Gram matrices, and uses the total theta sign times
`(-1)^mathsf{A}`. It then divides by the direct auxiliary factor and compares
the recovered physical coefficients with direct physical PBW. It does not
construct its numerator by convolving a precomputed physical block.
For the inserted case, equal middle levels select `Theta psi_0`. This
experiment uses the production normalization `Q Theta psi`; the same Q
appears in the auxiliary divisor and cancels in the recovered block.

`ramond_unphased` is a separate double-Virasoro candidate. Its boundary
vertices use the literal form; the Virasoro embeddings, primary bases,
norms, branching Ward recurrences, CCY recursion, Schottky factors, and
auxiliary recovery otherwise retain their production definitions. The
`--ns-branch-sign keep` candidate retains the displayed `(-1)^(2 n1)` in
the double-Virasoro sum. The `remove` candidate tests deleting that sign
as well. **The PBW word parity A is not interchangeable with 2 n1.**
Neither candidate passes; changing this branch sign alone is insufficient.
`--vertex production --ns-branch-sign keep` is the control calculation.

## Scope and results

Parameters are `b=7/5`, `P1=11/23`, `P2=13/29`, `P3=17/31`, `p=f=0`,
`eta=1`, and `eta'=+1` or `-1`. All arithmetic uses 40-digit working
precision. The comparison includes every coefficient with
`r1+r2+r3 <=3`, where `r1` is half-integral and `r2,r3` are integral:
30 monomials and 240 parity components per sector. The physical PBW
reference executable computes an independent-edge level-3 box; the
comparison retains only the requested total-level-3 subset.

| Calculation vs direct physical PBW | eta eta'=+1 | eta eta'=-1 |
| --- | ---: | ---: |
| Literal enlarged PBW, recovered: maximum scaled error | 6.4e-41 | 8.9e-41 |
| Production double Virasoro: maximum scaled error | 6.8e-28 | 7.4e-28 |
| Unphased double Virasoro, original branch sign | fails at level 1/2 | fails at level 1/2 |
| Unphased double Virasoro, branch sign also removed | fails at level 1/2 | fails at level 1/2 |

The scale is `max(1, abs(candidate), abs(reference))`; zero components
are included. Detailed differences are in `results/comparison.json`.
Both unphased double-Virasoro candidates have 40 failing components per
sector at tolerance 1e-20. Sector residuals alone do not detect this error.

For the coefficient of `q1^(1/2) eta1 eta2` at `q2=q3=0`:

| Calculation | eta eta'=+1 | eta eta'=-1 |
| --- | ---: | ---: |
| Direct physical PBW | +0.0056379732767405754 | -0.0561291561773283953 |
| Unphased DV, branch sign kept | -0.0056379732767405754 | +0.0561291561773283953 |
| Unphased DV, branch sign removed | +1.0056379732767405754 | -1.0561291561773283953 |

## Local reason the old Virasoro reduction cannot simply be reused

`vertex_probe.cpp` independently expands the states in the free-field/PBW
basis and compares

    hatrho(L_-1^(i) v_0, v_(1/4)^0, v_(1/4)^0)
      = (h_(1,0)^(i) + h_(2,1/4)^(i) - h_(3,1/4)^(i))
        hatrho(v_0, v_(1/4)^0, v_(1/4)^0).

The production vertex satisfies this identity to about 1e-32. The unphased
vertex has absolute residual approximately 0.206469 for **each** Virasoro
copy. Its imaginary part is nonzero, while the expected value is real.
Thus the unphased tensor-product form is not the Virasoro three-point
intertwiner for the retained embeddings and local contour conventions.
This is a convention incompatibility, not a numerical-precision issue.
See `results/vertex_ward.json` for the actual and expected values.

These negative controls establish that deleting the sign alone does not
give the Virasoro intertwiner used by CCY. The explicit transport above
repairs the double-Virasoro candidate, while preserving the existing
embeddings, primary bases, and norms.

## Reproduce

From this directory:

```sh
make -j2
python3 check_transport.py
```

Build the physical reference with `make -C ../.. bin/pbw` if needed.
`results/corrected_summary.json` is the authoritative final result.
`python3 run_probes.py` reproduces the earlier definition check and negative
controls; their separate verdicts are in `results/summary.json`. No high-level benchmarks are run. The initial
level-1 exploratory outputs are retained separately; the corrected level-3 summary
is the authoritative final comparison.
