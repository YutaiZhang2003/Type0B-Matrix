# Fixed complex-frequency production comparison

The production target is the amplitude at the specified complex energies
`omega_a = E_a + i epsilon nu_a`, with `Omega = sum_a omega_a`.
An epsilon-to-zero extrapolation is not required for comparison with the
matrix prediction evaluated at those same complex energies. This supersedes
any earlier proposal to require that limit before reporting a comparison.

The first comparison may use a preliminary worldsheet estimate. Report its
sampling uncertainty and existing diagnostics, then study block truncation,
momentum integration, collar dependence and sampling convergence as follow-up
work. The first estimate is not a certified numerical answer. Finite complex
energies also do not remove the need for the correct analytic-continuation
sheet, contour residues when crossings occur, or boundary finite parts.

BRY, [arXiv:2201.05621](https://arxiv.org/pdf/2201.05621), section 4.2.2,
already compares worldsheet and matrix amplitudes at finite complex energies.
Its equations (2.9)-(2.13) give the basis and energy dictionary. In particular,
`T=(R+L)/2`; the perturbatively decoupled, identical matrix sectors predict
`A_T = 2^(-n) A_R` for `1->n`. This is a prediction for the all-tachyon
amplitude, without assuming equality among worldsheet NS/R diagrams.

For the coefficient of `delta(E) mu_F^(-(n-1))`, the implemented predictions
for `n=2,3,4,5` are

```
A_T = i Omega prod_a(omega_a) prod_{j=1}^{n-2}(j + 2i Omega)
A_R = 2^n A_T.
```

The higher-point polynomials agree with the frozen c=1 reference files
`sphere_five_point_matrix_comparison.py` and
`sphere_six_point_matrix_comparison.py` after doubling each energy and
including the factor 1/2 from the energy delta function. The reference files
are not modified or imported by the worldsheet calculation.

For the existing five-point integral `I5`, the literal all-NS prefactor is
`(i/64) g_s^5 C_S2 delta(E) I5`. Applying BRY (4.14),
`g_s=4/(pi mu_F)` and `C_S2=pi/g_s^2`, gives
`A_T,worldsheet = i I5/pi^2` after stripping `delta(E) mu_F^-3`.
The postprocessor compares this with the all-tachyon matrix prediction.
It does not multiply the worldsheet result by sixteen and call it a
calculated all-right-mode amplitude.

From the repository root, produce a prediction now:

```sh
python3 Code/higher_point_amplitude_attempts/complex_frequency_comparison/compare.py \
  --config Code/config/type0b_ns_five_tachyon_batched_c_recursion_cluster.json \
  --output /tmp/type0b-complex-prediction.json
```

Add `--summary /path/to/summary.json` to compare the completed cluster
reduction. The config SHA-256 must match the summary. All configured collar
radii are reported without fitting or selecting against the matrix value.
Real and imaginary sampling errors are rotated by the normalization; no
unavailable systematic error or covariance is inferred.

**Multiplicity:** five external points mean `1->4`. The running Type-0B
pipeline computes that process. The matrix predictor also supports `1->5`,
but a six-point Type-0B worldsheet integrator and its normalization are not
implemented by this postprocessor. The bosonic c=1 six-point integrator is
not a Type-0B result.
