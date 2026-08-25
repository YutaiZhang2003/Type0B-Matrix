# Sphere five-point equal-energy physical finite part

The signed energies are
\[
(k_1,k_2,k_3,k_4,k_5)=(4\omega,-\omega,-\omega,-\omega,-\omega).
\]
In Xi's conventions,
\[
\mu^3\mathcal A^{\rm tree}_{1\to4}=4i\omega^5Q(\omega).
\]

The physical calculation is now defined directly at
\(\omega+i\epsilon/4\).  The moduli integral is split into an excised bulk,
ten analytically continued face collars, and fifteen double-primitive
corners.  No fit or continuation in \(\omega\) is used.

No direct physical five-point amplitude table in this directory is frozen or
ready for a matrix-model comparison.  In particular, the
`physical_iepsilon_pilot*` artifacts remain smoke tests.  The old
`worldsheet_fit_frozen.json` and `matrix_model_comparison.json` are
quarantined because they use only the original five-point convergent-ray fit.

Files:

- `physical_finite_part_smoke_refined.json`: block-order-2 collar smoke test,
  explicitly marked unfrozen.
- `physical_finite_part_block3_smoke.json`: low-statistics block-order check.
- `physical_finite_part_momentum3_smoke.json` and
  `physical_finite_part_momentum4_smoke.json`: very low momentum-grid diagnostics.
- `physical_finite_part_momentum6_c_smoke.json` through
  `physical_finite_part_momentum12_c_smoke.json`: endpoint-panel convergence
  diagnostics using the independently checked c-recursion accelerator.

The collar smoke test is stable within its moduli error, while the momentum
diagnostics are not yet converged.  The promotion gates are therefore higher
endpoint-resolved momentum order, block order, moduli statistics, three-radius
stability, and an \(\epsilon\to0^+\) sequence.  Only after those gates pass
will a worldsheet curve be frozen and the comparison script enabled.

## Exploratory convergent-ray continuation

The analytic-continuation program has also been revived as an independent
worldsheet calculation.  It does not alter the unfrozen status of the direct
physical finite-part artifacts above and does not call the matrix-comparison
script.

On \(\omega=it\), the positive-real Liouville-momentum representation reaches
its first DOZZ endpoint pinch at \(t=2/5\).  The continued calculation adds

\[
-2i\mathop{\rm Res}_{P=(5/2)\omega-i}
\]

on every internal edge whose cherry contains the incoming operator and one
outgoing operator.  The remaining momentum integrals use the
threshold-adapted map \(P=6u^{5/4}\).

At momentum order 20 and Sobol power 9, the reduced amplitudes are

| \(t\) | level 4 | level 6 | level 8 |
|---:|---:|---:|---:|
| 0.42 | -0.217426 | -0.217993 | -0.218869 |
| 0.46 | -0.134689 | -0.135030 | -0.135604 |
| 0.48 | -0.072961 | -0.073141 | -0.073468 |
| 0.49 | -0.036089 | -0.036178 | -0.036353 |

The range through \(t=0.48\) is the validated first-wall extension.  The
\(t=0.49\) row is retained as a near-wall diagnostic because a second pole
family pinches at \(t=1/2\).

The corresponding worldsheet-only files are
`worldsheet_continued_level4.json`, `worldsheet_continued_level6.json`, and
`worldsheet_continued_level8.json`; their convergence audit is collected in
`worldsheet_continued_summary.json`.  The proposed generalization to
arbitrary sphere multiplicity is in
`../../sphere_n_point_momentum_integration.md`.

## Frozen imaginary-ray fit and downstream comparison

The analytic-continuation branch is now post-processed directly on
\(\omega=it\).  A points-only table is mechanically extracted from the
eighteen-point audit, and the target-free program
`../../sphere_five_point_imaginary_ray_fit.py` assumes

\[
Q_4(it)=a+bt+ct^2.
\]

Its primary unweighted fit uses the seventeen validated points with
\(t\leq0.48\), including the three residue-corrected points.  It gives

\[
(a,b,c)=(2.00575458,-12.03227258,16.03986983).
\]

The \(t=0.49\) near-second-wall diagnostic is excluded from the primary fit;
including it gives the separately recorded sensitivity fit
\((2.00823417,-12.05012319,16.07011571)\).

The points-only input and fit are
`worldsheet_imaginary_ray_points_frozen.json` and
`worldsheet_imaginary_ray_fit_frozen.json`.  Neither contains a matrix-model
coefficient.  Only afterward does
`../../sphere_five_point_matrix_comparison.py` verify the frozen points hash
and write `matrix_model_comparison_imaginary_ray.json` and the comparison
figure.  This is a valid imaginary-energy analytic-continuation workflow
under the declared degree-two ansatz; it remains distinct from the unfinished
direct physical-\(i\epsilon\) finite-part program.
