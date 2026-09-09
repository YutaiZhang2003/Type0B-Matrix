# Accuracy refinement of the unchanged NSRR factorized sewing trial

This continues `NSRR_FACTORIZED_SIGN_TRIAL_2026-08-30.md`. The user requested
higher numerical accuracy after the preliminary normalization-only comparison
with the saved NSNSNS numerator gave differences of 3.67–5.94 percent.

## Scope and invariant assumptions

The refinement changes **only numerical accuracy**, not the trial's physical
interpretation. The vertex ansatz remains

\[
t_{f;\eta\zeta}=i^f c_\eta\delta_{\eta\zeta},\qquad
(c_+,c_-)=(C_{\rm even},C_{\rm odd})/2,
\]

with the explicit `(-1)^f` sewing sign and the additional hypothesis
`Ftilde = conjugate(F)`. Coefficients at the two vertices are multiplied,
not absolute-squared. The original contraction function is reused verbatim.
The physical Ramond projector and lift/spin dictionary remain unestablished;
all physical partition/modular-agreement output fields remain null.

Unchanged inputs are `b=1.4`, the five surfaces with
`Omega_original,11=Omega_original,22=i` and
`Omega_original,12=t+i/2`, `t=0.52,0.56,0.60,0.64,0.68`, and the saved
NS-at-infinity source plumbing. The common cosmological factor is omitted
on both numerator sides. No fitted constant or Ramond multiplicity is used.

The original runner, previous output, Human Note, and all eight protected
PBW/double-Virasoro/branching/c-recursion kernels are left untouched.

## Numerical design

- Chiral total level raised from `L=2` to `L=3`. Every half-level from zero
  through three is retained; holomorphic and antiholomorphic factors are
  each truncated at that level.
- Momentum quadrature orders `N=3,4,5`: `27+64+125=216` independent nodes,
  two subprocess workers, a fresh process per node to bound recursion caches.
- Each node supplies all five surfaces, four lift representatives, and all
  eight channels `(f,eta,eta')`.
- Equal-sign channels use the checked branching recursion and product of
  two ordinary Virasoro c-recursions. Opposite signs use the same explicit
  PBW diagnostic completion, now capped at level three. This is not a pure
  double-Virasoro evaluation of every channel.
- The complete old `N=3,L<=2` node data are checked against their new
  counterparts; this distinguishes an accuracy change from an implementation
  or convention change.
- Source geometry/free factors are freshly checked before dispatch.
  The final comparison also recomputes both free factors at oscillator
  cutoffs 32 and 40.

The unequal-momentum `L=3` benchmark took 9.80 seconds and had branching Ward
residual `1.95e-14` or smaller. The four equal-sign channels at the newly added
levels are independently checked against the protected PBW oracle.

## Comparison definition

\[
Q_{R,\rm trial}=Z_{R,\rm trial}^{\rm pl}/(Z_{\rm free,R}^{\rm pl})^\kappa,
\qquad
Q_{N,\rm diagnostic}=Z_{N,\rm saved}^{\rm pl}/(Z_{\rm free,N}^{\rm pl})^\kappa,
\quad \kappa=1+2(b+b^{-1})^2=9.940408163265307.
\]

The saved all-NS numerator is unchanged (`R=16,N=5`). Its former filtered
free denominator is replaced by the independently computed fixed-spin free
factor in its own plumbing frame. Neither numerator's fixed-spin physical
identification is certified by this replacement. Reported differences
`Q_R/Q_N-1` are therefore diagnostics, not established modular errors.

The desired accuracy comparisons are `L=2 -> 3` at fixed `N=5`, and
`N=3 -> 4 -> 5` at fixed `L=3`. Differences between successive approximants
are not rigorous error bounds.

## Execution and reproducibility

New files:

- `Code/genus_2/refine_nsrr_factorized_sign_trial.py`
- `Code/genus_2/test_refine_nsrr_factorized_sign_trial.py`
- `Code/genus_2/audit_nsrr_trial_refinement.py`
- `Data Set/nsrr_factorized_sign_trial_L3_N5_20260830/`

```sh
env PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  PYTHONPATH=Code:Code/genus_2:Code/c_Recursion:Code/full_ramond_block_runtime:Code/double_virasoro/nsrr \
  python3 Code/genus_2/refine_nsrr_factorized_sign_trial.py run \
  --baseline-dir 'Data Set/nsrr_factorized_sign_trial_L2_N3_20260830' \
  --output-dir 'Data Set/nsrr_factorized_sign_trial_L3_N5_20260830' \
  --orders 3 4 5 --max-level 3 --workers 2

env PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  PYTHONPATH=Code:Code/genus_2:Code/c_Recursion:Code/full_ramond_block_runtime:Code/double_virasoro/nsrr \
  python3 Code/genus_2/audit_nsrr_trial_refinement.py \
  --run-dir 'Data Set/nsrr_factorized_sign_trial_L3_N5_20260830' \
  --all-ns-summary 'Data Set/nsnsns_recompute_fivepoint_R16_N5_20260830/summary.json'
```

The runner validates existing shards on resume and cancels queued work if
a node fails. It refuses to resume with changed numerical settings, changed
trial assumptions, or changed implementation/kernel hashes.

## Results

All **216 nodes completed**, in 23.37 minutes from configuration creation to
the saved summary. The run contains 864 explicitly requested mixed-sign PBW
completions. The table uses `L=3,N=5` and the representative lifts `(+,+,+)`.

| t | Z_NSrr_trial | Q_NSrr_trial | Q_NSNSNS_diagnostic | Q_R/Q_N - 1 |
|---:|---:|---:|---:|---:|
| 0.52 | 8.20134812e-10 | 2.95650905e-7 | 3.04607896e-7 | -2.94050% |
| 0.56 | 8.03285997e-10 | 2.35635248e-7 | 2.43492191e-7 | -3.22677% |
| 0.60 | 7.72057167e-10 | 1.87747896e-7 | 1.94942626e-7 | -3.69069% |
| 0.64 | 7.30645990e-10 | 1.49183778e-7 | 1.55888544e-7 | -4.30100% |
| 0.68 | 6.83871409e-10 | 1.18040348e-7 | 1.24269955e-7 | -5.01296% |

The refinement reduces the original diagnostic difference of 3.67–5.94%
to 2.94–5.01%, with no fitted normalization.

### Separate convergence checks

All entries below are signed changes `(new/old-1) * 100`.

| t | N=3 to 4 at L=3 | N=4 to 5 at L=3 | L=2 to 3 at N=5 |
|---:|---:|---:|---:|
| 0.52 | +0.85199% | +0.06624% | -0.15830% |
| 0.56 | +0.85962% | +0.06808% | -0.12850% |
| 0.60 | +0.87407% | +0.07163% | -0.08290% |
| 0.64 | +0.89383% | +0.07667% | -0.04031% |
| 0.68 | +0.91701% | +0.08286% | -0.01332% |

Thus the remaining few-percent diagnostic difference is substantially larger
than the **observed final NSRR accuracy shifts**. This is not a rigorous error
bound, and it does not certify the NSNSNS momentum quadrature or either
numerator's physical spin identification. It does indicate that merely
raising the NSRR descendant order is unlikely to eliminate the whole gap.

For context, the already saved all-NS `R=12 -> 16` scan changes its numerator
by at most `6.93e-9` relatively at fixed `N=5`; those `R=16` numerators match
the fresh saved all-NS reference exactly. That check is in
`Data Set/nsrr_nsnsns_target_R8_R12_R16_N5_20260830/summary.json` and controls
the block recursion order, not momentum quadrature or nonchiral assembly.

![Refined NSRR trial comparison](</Users/yutaizhang/Desktop/Type0B-Matrix/Data Set/nsrr_factorized_sign_trial_L3_N5_20260830/nsrr_refinement_comparison.png>)

### Verification

- **48 regression tests pass**, including nine new refinement tests and
  the protected-kernel hash check.
- All 27 shared `N=3` nodes reproduce every archived `L<=2` block, across
  five points and four lifts, with maximum scaled block error `4.93e-14`;
  maximum node-total relative error is `1.82e-14`.
- Independent primary-power and term reconstruction has maximum relative
  error `7.82e-15`; momentum reassembly agrees to `9.97e-16`.
- Exact ground normalization agrees to `9.16e-14`. Across all nodes,
  ground/half-level coefficient checks agree to `1.40e-13`, and the maximum
  branching Ward residual is `7.06e-13`.
- At the largest `N=5` momentum endpoint, 30- versus 45-digit structure
  constants differ by `8.46e-14` relatively. Equal-sign double-Virasoro
  coefficients through level three match independent PBW evaluation with
  maximum scaled error `1.53e-12`.
- All eight protected kernel hashes remain identical to the manifest.
  Geometry provenance and all five target plumbing charts match the saved
  all-NS numerator. Free oscillator cutoffs 32 and 40 agree within the
  explicit audit tolerance, with no meaningful change in displayed values.

The previous spin limitation persists: the integrated trial's maximum
relative spread across the four lift representatives is `2.55e-16`.
The formal same-chiral-convention antiholomorphic control still agrees with
the odd-sign-flipped control to `1.36e-15`, scaled by the trial. Refining
numerics has not established a physical Ramond projection or resolved this
antiholomorphic dictionary. Those questions remain separate from the checked
chiral-block computation.

Saved outputs include `summary.json`, `fivepoint_trial.csv`,
`verification.json`, `extreme_momentum_probe.json`, `comparison.json`,
`comparison.csv`, and the SVG/PNG figure. No previous data were overwritten.
