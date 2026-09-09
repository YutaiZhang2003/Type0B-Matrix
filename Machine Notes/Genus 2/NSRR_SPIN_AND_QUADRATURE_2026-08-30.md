# NSRR/all-NS comparison: checks of factors 2 and 3

This note checks (2) identification of the numerator spin against the
physical fixed-spin free denominator, and (3) momentum-quadrature
convergence. It does not change the Human Note, the checked PBW/double
Virasoro or c-recursion kernels, or the proposed NSRR vertex contraction.
The NSRR quantity is still an explicitly hypothetical trial.

The point is the same original period matrix as before,

\[
\Omega_{\rm original}(0.60)=
\begin{pmatrix}i&0.60+i/2\\0.60+i/2&i\end{pmatrix},\qquad b=1.4.
\]

All values omit the same cosmological prefactor. The Weyl-quotient exponent
is \(\kappa=1+2(b+b^{-1})^2=9.940408163265307\).

## 2. What the independent spin control establishes

### The physical denominators and their marked transport pass

The source characteristic is `[11|00]` in the re-marked NS-at-infinity
chart; the target characteristic is `[00|00]`. The saved composed
symplectic matrix transports the former to the latter. The charged-boson
period extraction independently reproduces the marked periods after the
explicit integer branches

\[
B_{\rm source}=\begin{pmatrix}0&0\\0&1\end{pmatrix},\qquad
B_{\rm target}=\begin{pmatrix}-1&-1\\-1&0\end{pmatrix}.
\]

The residuals are `2.53e-13` and `6.33e-11`, respectively. Independent
charge-lattice sewing reproduces

| Same-frame physical free factor | Value |
|---|---:|
| Source, `[11|00]` | 0.5754095923717206 |
| Target, `[00|00]` | 0.5638924570404817 |

These are physical scalar plus real Majorana factors, not the auxiliary
fermion used by the double-Virasoro calculation. The scalar and fermion
are evaluated in the same plumbing frame. Their ratio predicts the raw
Liouville frame ratio `1.2226013135427836` if the fixed-spin modular
comparison is correct. Raw partition functions need not be equal.
The direct free modular/frame check at this point has relative residual
`1.72e-11`. Repeating the spin-basis control over all five saved surfaces
gives period residual at most `1.31e-9` and the complex free-basis identity
error at most `5.56e-16`.

### The numerator's old spin label does not pass the free control

The all-NS numerator calls the checked block evaluator with geometric
lifts `(+,-,+)`. The old helper labels this `[00|00]`. That helper is
also used by the old sewing-spin audit, so agreement of those two labels
is not an independent identification of the physical spin.

Let \(D_\lambda\) be the *unfiltered* physical free-fermion Fredholm
determinant. Independent bosonization verifies

\[
D_\lambda^2=P(q)\theta[00|\beta_{\rm charge}](\Omega_{\rm charge}),
\quad
\beta_{\rm charge}=(\lambda_0\lambda_\infty<0,
                    \lambda_1\lambda_\infty<0).
\]

The characteristic must then be transported through the specified integer
period branch. In this target chart, unfiltered lifts `(-,+,+)` give
`[00|00]`; unfiltered lifts `(+,-,+)` give `[00|11]`.

Crucially, the legacy free block carrying the same quadratic parity sign
as the Human-Note block is **not** this unfiltered determinant. In the
ordered list of lifts `(+++)`, `(+-+)`, `(-++)`, `(--+)`, it satisfies

\[
\mathbf F_{\rm legacy}=U\mathbf D,\qquad
U=\tfrac12\mathbf1\mathbf1^T-I_4,\qquad U^2=I_4.
\]

The complex identity is checked to `2.23e-16` at the target and
`2.23e-16` at the source. Thus assigning a single spin label to the
filtered expression is not justified by the unfiltered determinant map.

| Target free control | Value of full scalar + Majorana factor |
|---|---:|
| Selected **filtered** `(+,-,+)` | 0.5372515435695151 |
| Unfiltered `(+,-,+)`, `[00|11]` | 0.6148619404480470 |
| Unfiltered `(-,+,+)`, desired `[00|00]` | 0.5638924570404816 |

There is also an exact lowest-level obstruction to repairing this by a
permutation of real lifts. In the unfiltered free block, the three
coefficients of \(\sqrt{q_0q_1},\sqrt{q_0q_\infty},
\sqrt{q_1q_\infty}\) have signs
\((\lambda_0\lambda_1,\lambda_0\lambda_\infty,
\lambda_1\lambda_\infty)\), whose product is +1. The quadratic
Human-Note sign flips all three, giving product -1. No assignment of
three real signs changes that invariant.

This does **not** prove that the Human Note or its checked chiral blocks
are wrong. Nor does the free identity authorize applying \(U\) to the
Liouville numerator or deleting the quadratic signs. It shows that a
physical fixed-spin assembly needs a separately derived local spin-frame
and nonchiral-pairing dictionary. In particular, replacing only the old
free denominator cannot establish that the numerator has the new
denominator's spin.

For scale only: changing that denominator alone multiplies the target
quotient by `0.6181109034664737`. This is not a proposed correction, and
must not be fitted to the few-percent residual.

### The actual Liouville lift dependence is non-negligible

Three dominant N5 momentum nodes are reevaluated with the unchanged R16
all-NS code at all four literal lifts. The selected lift reproduces the
saved node. At the largest node, index 31, the relative changes from
`(+,-,+)` are `+6.1342%`, `0`, `+2.6690%`, and `-1.4997%` in the lift
order above. These are pointwise sensitivity checks, not new integrated
partitions or a physical choice of spin.

For the NSRR trial, all four literal lifts give the same integrated value
at N5/L3. By contrast, independent free sewing at fixed source alpha=11
gives different answers for the two even beta choices: `0.57540959237`
for beta=00 and `0.51261962100` for beta=11. The odd free characteristics
vanish because of free-fermion zero modes. These free statements are not
claims about interacting Liouville odd-spin partition functions.
Lift invariance of the trial does not by itself prove a spin average or
an error: the physical Ramond projector/lift dictionary is still absent.
Already at ground level the trial sums to `(E+O)^2` for every real lift:
changing a Ramond lift just exchanges the contributing form-parity/sign
components. This algebraic invariance cannot select the desired marked
spin by itself.

**Factor 2 is therefore not cleared.** The fixed-spin denominators pass,
but the actual numerator-to-spin identification is not established, and
the old label-only check is insufficient. The next analytic task is a
spin-frame/nonchiral assembly derivation, retaining the Human Note's
graded pairing, quadratic signs, odd coefficient phase and decomposition
sign.

## 3. Fixed-block-order momentum convergence

The new numerical sweep uses all-NS `R=16`, N=6 and 7, and the unchanged
NSRR trial at `L=3`, N=6. It retains the previous channel-specific
five-point momentum envelopes, quadrature measure, b, coefficients,
plumbing coordinates, and free denominators. Thus changes across N are
integration effects, not a simultaneous change of block order.

Each quadrature node is a fresh subprocess. A process-local memoization
wrapper caches only literal one-module Ward actions; three-module Ward
systems and blocks are recalculated. The cached and uncached N5 NSRR
control agree bitwise. All-NS saved-node reproduction is exact; NSRR
reproduction differs by `2.22e-16` relatively.

All **775 new nodes completed**, in about 33.4 minutes on four local
workers. The figures below use the independently checked physical free
denominators on both sides, but remain **diagnostic Qs**, because the
Liouville numerator spin identification discussed above is unresolved.

| Nodes per momentum axis N | All-NS R16, Q × 10^7 | NSRR trial L3, Q × 10^7 |
|---|---:|---:|
| 5 (saved reference) | 1.949426255892 | 1.877478955644 |
| 6 (new) | 1.953560371080 | 1.877511197998 |
| 7 (new, all-NS only) | 1.954698300856 | — |

The corresponding raw finest-grid values are
`Z_NSNSNS(N7)=6.574598832292943e-10` and
`Z_NSRR_trial(N6)=7.720704252824786e-10`.

- All-NS N5→6: **+0.2120683035%**.
- All-NS N6→7: **+0.0582490202%**.
- NSRR N4→5, saved fixed-L3 control: **+0.0716310532%**.
- NSRR N5→6: **+0.0017173218%**.

The finest-grid diagnostic comparison is

\[
\frac{Q_{\rm NSRR,trial}(L3,N6)}{Q_{\rm NSNSNS,diagnostic}(R16,N7)}-1
=-0.03948798790269625,
\]

or **−3.9487987903%**. The old N5/N5 value was −3.690691%. Thus the
measured integration corrections do not close the mismatch; their net
effect slightly increases it. The observed convergence strongly disfavors
momentum quadrature as the explanation of the remaining few-percent gap.
No extrapolation or fitted rescaling is included.

Successive quadrature differences are convergence observations, not
rigorous tail bounds.

The plot `omega_comparison_refined_point.png` overlays the new central
point on the previous five-point curve versus
`t=Re Omega_original,12`. Only t=.60 is newly refined; the other four
points remain the old N5 results. Both the plot and its numerical data
explicitly retain this distinction and the unresolved-spin warning.

## Verification and conclusion

- All 775 node identities, coordinates, quadrature weights and metadata
  are validated. An independent 50-digit decimal summation gives the same
  final binary64 totals. An independently implemented quadrature rule
  agrees to `6.67e-16` relatively or better.
- Three NSRR nodes, indices 0, 43 and 215, are reconstructed directly
  from stored blocks, recomputed structure coefficients, the explicit
  primary weights and the two odd i phases plus the decomposition sign.
  The maximum relative discrepancy is `2.00e-15`.
- The largest NSRR Ward residual is `1.57e-12`; the largest ground/half
  level analytic error is `3.55e-13`. There are 864 explicit mixed-sign
  PBW completion calls. All four literal-lift integrals remain identical.
- Neither all-NS grid has a global resummation failure. The maximum
  occupation is 21 and the largest reported last-shell fraction is
  `1.078e-8`.
- **43 regression tests pass.** The eight protected kernels match their
  prior manifest, and the Human Note matches the pre-run fingerprint.

**Conclusion:** factor 3 is numerically small at the tested accuracy;
factor 2 is not cleared. The next priority is the physical spin-frame and
nonchiral assembly at the boundary of the checked block code. The free
control identifies an actual gap in that dictionary, but it does not yet
prove a particular correction that removes the full −3.95% residual.

## Reproducibility

New scripts:

- `Code/genus_2/check_nsrr_spin_quadrature.py`: frozen numerical design,
  isolated workers and reduction.
- `Code/genus_2/audit_nsrr_comparison_spin_basis.py`: independent spin
  transport, free determinant/charge-lattice checks and sign obstruction.
- `Code/genus_2/probe_nsnsns_comparison_lifts.py`: actual Liouville lift
  sensitivity on three dominant reference nodes.
- `Code/genus_2/test_check_nsrr_spin_quadrature.py`: regression tests.
- `Code/genus_2/verify_nsrr_spin_quadrature.py`: independent integration,
  quadrature-weight and selected-node contraction checks.
- `Code/genus_2/plot_nsrr_spin_quadrature.py`: reproducible Omega plot,
  preserving which points have actually been refined.

New data directory: `Data Set/nsrr_spin_quadrature_t060_20260830/`.
The config freezes both reference configs, reference summary digests,
implementation hashes, and all eight protected-kernel hashes. Archived
results and the Human Note are not edited.

Completed outputs include `quadrature_summary.json`, `verification.json`,
`spin_basis_audit.json`, `spin_basis_fivepoint.json`,
`target_lift_sensitivity.json`, `plot_data.json`, and the SVG/PNG plot.
The separately saved `target_N6_complete.json` and
`target_N7_complete.json` were interim completed-grid snapshots; the
authoritative final numerical reduction is `quadrature_summary.json`.
