# Numerical NSRR block and factorized-sign trial

This records the user's requested numerical test of the proposed
`(-1)^f` conformal-block sewing formula. It follows the separate
`NSRR_NONCHIRAL_SEWING_DERIVATION_2026-08-30.md`.

The low-order **chiral blocks have been computed**, including the
opposite-sign components. The integrated result below is an explicitly
specified factorization trial, **not a certified fixed-spin partition**.
The physical Ramond state restriction is not inserted or inferred.

## Formula actually evaluated

Let `(E,O)=(C_even,C_odd)` be the generic-b BRY-normalized coefficients.
The additional vertex hypothesis used for this test is

\[
c_+=E/2,\quad c_-=O/2,\qquad
t_{f;\eta\zeta}=i^f c_\eta\delta_{\eta\zeta}.
\]

The additional antiholomorphic hypothesis is coefficientwise conjugation
of the chiral block at real continuum momenta. Thus the trial is

\[
\begin{aligned}
Z_{\rm trial}
={}&\int_{P_i\geq0}\prod_{i=1}^{3}\frac{dP_i}{\pi}
 \left|\prod_i q_i^{h_i}\right|^2
 \sum_{f=0,1}\sum_{\eta,\eta'=\pm1}
 (-1)^f(i^f c_\eta)(i^f c_{\eta'})
 \mathbb F_f^{(\eta,\eta')}(q)
 \overline{\mathbb F_f^{(\eta,\eta')}(q)}.
\end{aligned}
\]

The code multiplies the two vertex coefficients and the explicit sewing
sign separately. It does not replace the vertex product by an absolute
square. Algebraically the two `i^f` factors cancel `(-1)^f`, leaving
`c_eta c_eta' |F|^2` in this particular ansatz.

Neither the diagonal `t` matrix nor the conjugate-block identification
has been established as the full physical NSRR dictionary. In particular,
the `i` in the definition of `rho_1^(eta)` does not itself prove that the
additional vertex factor above is required. These choices are explicit
trial inputs, not conclusions of the grading derivation.

Two controls are retained in every shard:

- omit the explicit `(-1)^f` while keeping the same vertex coefficients;
- retain that sign and the coefficients, but use the formal
  same-chiral-convention block evaluated at `conjugate(q)` for the
  antiholomorphic factor, instead of conjugating the coefficients too.

## Method and scope

- `b=1.4`, hence `kappa=1+2(b+1/b)^2=9.940408163265307`.
- Saved marked surfaces with original
  `Omega_11=Omega_22=i`, `Omega_12=t+i/2`, at
  `t=0.52,0.56,0.60,0.64,0.68`.
- The re-marked NS-at-infinity plumbing is used, with all edge data
  reversed consistently between geometry and Human-Note slot order.
- Total **chiral** descendant cutoffs `L=0,1/2,1,3/2,2`.
  Each holomorphic and antiholomorphic factor is truncated separately.
- Momentum quadrature orders `N=2,3`, with `8+27=35` independent nodes.
- All eight channels `(f,eta,eta')` are retained. All four representatives
  `(lambda_0,lambda_1,lambda_infinity)=(+/-,+/-,+)` are evaluated.
- Equal-sign components use the checked branching recursion and the
  product of two ordinary Virasoro c-recursions.
- The singular auxiliary star quotient cannot recover opposite-sign
  components. They use the adapter's **explicit PBW diagnostic
  completion**, capped at level 2. This is not a pure-double-Virasoro
  computation of every channel. There are four such completions per
  momentum node, 140 in the integrated run.
- The common cosmological factor is omitted, as in the previous toys.
  No extra Ramond multiplicity or fitted normalization is introduced.

## Actual chiral blocks at an unequal-momentum probe

For Human slots `(P_NS,P_R at 1,P_R at 0)=(0.31,0.43,0.57)`, `t=0.60`,
and literal lifts `(+,+,+)`, the level-2 blocks are the following matrices
in sign-index order `(eta,eta')=(+,-)`. Primary powers are **not** included:

\[
\mathbb F_0\simeq
\begin{pmatrix}
1.91014498+0.01810671i & 0.01083333+0.02087844i\\
0.01083333+0.02087844i & 1.90545794+0.01843021i
\end{pmatrix},
\]

\[
\mathbb F_1\simeq
\begin{pmatrix}
0.00450539-0.00187585i & 0.01740272-1.90877497i\\
0.01740272-1.90877497i & -0.15529922+0.04767016i
\end{pmatrix}.
\]

The opposite-sign odd block is large because its exact ground term is
`-2i`, whereas the equal-sign odd blocks have zero ground term at these
lifts. The probe and every integration node pass the independent exact
ground and first-NS-descendant formulas in the derivation note.

Full probe data, including every cutoff, are saved in
`Data Set/nsrr_factorized_sign_trial_L2_N3_20260830/probe_blocks_t060.json`.

## Five-point integrated trial

The table uses `N=3`, `L=2`, and literal lifts `(+,+,+)`.
Define a reference-normalized diagnostic

\[
Q_{\rm trial,ref}(t)
=\frac{Z_{\rm trial}(t)}
 {[Z_{X+\psi,[11|00]}^{\rm source\ plumbing}(t)]^\kappa}.
\]

The denominator is independently recomputed in the same plumbing frame
using the marked characteristic `[11|00]`. The trial's lift-to-physical-spin
dictionary is not established, so this ratio is **not** labelled physical
`Q_NSrr`.

| t | Z_trial | Q_trial,ref | L=1 to 2 change | N=2 to 3 change |
|---:|---:|---:|---:|---:|
| 0.52 | 8.13961605e-10 | 2.93425521e-7 | 1.338% | 3.847% |
| 0.56 | 7.96925697e-10 | 2.33769523e-7 | 1.194% | 3.818% |
| 0.60 | 7.65456200e-10 | 1.86142681e-7 | 0.948% | 3.760% |
| 0.64 | 7.23910837e-10 | 1.47808590e-7 | 0.671% | 3.676% |
| 0.68 | 6.77186343e-10 | 1.16886465e-7 | 0.431% | 3.570% |

The changes are diagnostics, not rigorous truncation-error bounds.
Opposite-sign components account for **47.87–47.93%** of the level-2
trial. Omitting them cannot be called a small accuracy reduction.

![NSRR factorized sign trial](</Users/yutaizhang/Desktop/Type0B-Matrix/Data Set/nsrr_factorized_sign_trial_L2_N3_20260830/nsrr_factorized_sign_trial.png>)

## What the sign and spin controls tell us

At `t=0.60`, omitting the explicit sewing sign gives

\[
Z_{\rm no\ sewing\ sign}=3.08615126\times10^{-11},
\qquad Z_{\rm trial}/Z_{\rm no\ sewing\ sign}=24.80294.
\]

This is a large effect, not a small numerical uncertainty. On the saved
run the formal same-chiral-convention antiholomorphic control agrees with
this odd-sign-flipped control up to numerical precision. The two
antiholomorphic identifications therefore cannot be interchanged while
leaving the vertex phases unchanged. Their difference must ultimately be
settled by transporting the physical BPZ-dual vertex, not by selecting
the result that looks closer to another channel.

There is a second, decisive limitation: for every cutoff and quadrature
grid in this run, `Z_trial` is identical for all four lift representatives
tested. Thus this diagonal coupling ansatz has removed the lift dependence
from the sewn sum. It does **not** establish the required fixed-spin
NSRR partition. This observation does not invalidate the computed chiral
blocks or the universal grading sign; it identifies a limitation of the
additional nonchiral assembly hypotheses.

At zero descendant level and `(+,+,+)` lifts, the ansatz gives
`Z_trial/|primary|^2=(E+O)^2=4 d_+^2` before momentum integration. This
is an exact diagnostic of the chosen ansatz, not a proposed multiplicity
for the physical Ramond ground-state sum. We do not rescale it to fit.

The next useful step is therefore the physical Ramond coupling/spin
assembly, rather than increasing numerical precision of this same ansatz.
No NSNSNS modular-agreement claim is made from the trial table.

## Validation and reproducibility

Files:

- `Code/genus_2/nsrr_factorized_sign_trial.py`: explicit trial assembler,
  one subprocess per momentum node, immutable-provenance shards.
- `Code/genus_2/test_nsrr_factorized_sign_trial.py`: nine new tests.
- `Code/genus_2/audit_nsrr_factorized_sign_trial.py`: independent
  coefficient/primary/quadrature reconstruction, probe and plot.
- `Data Set/nsrr_factorized_sign_trial_L2_N3_20260830/`: configuration,
  35 shards, summary, verification, CSV, probe blocks, SVG and PNG.

The independent audit reconstructs the weighted terms after analytically
cancelling the two vertex phases against the sewing sign, without calling
the assembler's contraction function. Maximum term discrepancy is
`6.60e-16`; quadrature reassembly discrepancy is `6.90e-19`, both relative
to the stated trial scales. Ground normalization checks agree to
`2.53e-14`. The exact ground/half-level checks across the integration nodes
have maximum scaled error `4.26e-14`.

All **39 relevant tests pass**. They include equal-sign double-Virasoro
versus independent PBW comparisons, opposite-sign vertex-exchange
symmetry, coefficient-phase and sewing-sign checks, all prior grading
checks, and the independent free-factor tests. All eight protected-kernel
hashes match their manifest. No Human Note, protected kernel, previous
partition output, or plumbing parameter was edited.

Run command:

```sh
env PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  PYTHONPATH=Code:Code/genus_2:Code/c_Recursion:Code/full_ramond_block_runtime:Code/double_virasoro/nsrr \
  python3 Code/genus_2/nsrr_factorized_sign_trial.py run \
  --geometry 'Data Set/nsnsns_recompute_fivepoint_R16_N5_20260830/source_geometry_audit.json' \
  --output-dir 'Data Set/nsrr_factorized_sign_trial_L2_N3_20260830' --workers 2
```

The JSON fields `physical_Z`, `physical_Q`, and
`physical_Ramond_projector` intentionally remain null. Computed values
are stored under the explicitly named trial fields instead.
