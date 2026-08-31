# Moduli dependence of the Q_L cross-channel comparison

The path is `Omega_11=Omega_22=i`, `Omega_12=t+i/2`, at `b=1.4` and
`kappa=1+2(b+1/b)^2=9.940408163265307`. The horizontal coordinate is the
original marked `t`, not the off-diagonal entry after fundamental-domain
reduction. All previously verified characteristic changes are included.

The cross-channel observable is the anomaly-cancelled quotient `Q_L`, not
either partition function separately. In each channel, numerator and free
denominator must be computed in that channel's same local plumbing frame:

\[
Q_L=\frac{Z_L}{(Z_{\mathrm{free}})^\kappa},
\qquad
\Delta_Q(t)=\frac{Q_L^{\mathrm{NSRR,trial}}}
                         {Q_L^{\mathrm{NSNSNS}}}-1.
\]

The two quotients are evaluated at corresponding modularly related period
matrices with transported spin structures. They use the fixed-spin free
factors in both native frames, not the legacy target factor. Differences
between individual partition functions are not independent cross-channel
tests. The former free-factor panel is removed from the comparison; its
data are retained only as internal normalization diagnostics. Neither the
Human Note nor a block implementation was changed.

## Common-cutoff result

All five numerator pairs use momentum quadrature `N=5`, NSRR total chiral
level `L=3`, and all-NS recursion order `R=16` (twice-level units). These
are saved, provenance-validated numerators; no new Liouville integration
was needed. The previous free-factor convergence and marking checks are
internal validation of the denominators, not additional observables.

| t | Quotient difference Delta_Q |
|---:|---:|
| 0.52 | -2.940499% |
| 0.56 | -3.226774% |
| 0.60 | -3.690691% |
| 0.64 | -4.301000% |
| 0.68 | -5.012963% |

The computed common-cutoff quotient discrepancy increases in magnitude on
the sampled path. It is not constant at this numerical accuracy. This
curve alone does not distinguish residual numerical error from an error
in the trial nonchiral assembly.

At `t=0.60` the separately saved refinement to source `N=6`, target `N=7`
gives `Delta_Q=-3.948799%`. This is plotted as a diamond and is not spliced
into the common-`N=5` curve. A uniformly refined five-point numerator scan
has not been performed in this step. Quadrature errors are not rigorously
bounded, and the physical NSRR nonchiral assembly remains a trial.

## Validation and artifacts

Mode 32 to 40 changes the fixed free factors by at most `5.6e-16` and the
legacy factor by `6.7e-16`. All free factors reproduce their saved values
to `6.7e-16`. The correctly transported fundamental-domain free invariant
agrees to `4.90e-10`, consistent with the saved geometric precision.

All 51 tests pass: the archived plot/free/spin suites plus three Q_L-only
presentation tests. The Q_L-only driver
verifies the saved provenance and all eight protected kernel hashes and
checks both the common-cutoff and refined quotient ratios. It does not
recompute or change any partition function.

- Current driver: `Code/genus_2/plot_nsrr_ql_moduli.py`.
- Presentation tests: `Code/genus_2/test_plot_nsrr_ql_moduli.py`.
- Data: `Data Set/nsrr_moduli_difference_20260830/summary.json` and
  `moduli_differences.csv` (archived, including auxiliary internal columns).
- Current figure: `Data Set/nsrr_moduli_difference_20260830/ql_moduli_difference.svg`
  and `ql_moduli_difference.png`.

![Q_L relative difference versus the original modulus](</Users/yutaizhang/Desktop/Type0B-Matrix/Data Set/nsrr_moduli_difference_20260830/ql_moduli_difference.png>)
