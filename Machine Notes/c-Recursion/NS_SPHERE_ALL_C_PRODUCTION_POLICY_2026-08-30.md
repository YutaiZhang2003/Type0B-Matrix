# NS sphere amplitudes: all-c production policy

30 August 2026. This supersedes the proposed `|q_ell|<0.3` h/c switch
for the NS four- and five-point amplitude computations. The human note is
unchanged.

## Computational route

Use component-aware central-charge recursion for every block, including
external `G_{-1/2} V` components, in every selected sphere chart. Select a
well-conditioned chart as before, but never change the recursion backend
as a function of the elliptic nome or plane plumbing coordinates.

The existing local OPE approximations and polynomial-subtraction layer
remain in place. For the five-point forest this means numerically evaluating

    F - chi_1 P_1 - chi_2 P_2 + chi_1 chi_2 P_12

and adding the existing analytic face/corner finite parts. This change does
not alter counterterm degrees, momenta, central-charge prescriptions,
integration contours, precision, block cutoffs, or quadrature settings.
Choosing c-recursion is not itself a convergence certificate for an
integrated amplitude; those independent checks are still required.

The distinction motivating the policy is the regular term of the recursion:
the fixed-weight large-c seed is the component-aware global `osp(1|2)`
block. The proposed pillow h-recursion with general upper cap components
instead needs its additional weight-polynomial regular seed. We do not
assume that the bottom-component infinite product supplies that seed.
This policy does not assert that h-recursion is mathematically invalid or
cannot be useful in a separately validated research calculation.

## Implementation boundary

- The shared four-/five-point kernels and sphere multipoint correlator
  default to `block_backend="c"`.
- Amplitude CLI backend selectors accept only `c`; four-point drivers
  without a selector explicitly construct c-recursive kernels.
- The five-point cluster loader rejects h/hybrid production configurations.
  Existing historical configs and result files are preserved.
- The analytic four-point routing audit requires c-recursion in every
  checked frame and fails on an h-recursive frame.
- Explicit h/hybrid Python selections and comparison tests remain available
  for research. Historical script/class names containing `hybrid` are kept
  for compatibility, not as a statement of current production routing.
- Legacy hybrid-threshold arguments are inactive under the c-only route.
  An elliptic-coordinate re-expansion of c-derived coefficients is still
  c-recursion; the choice of coordinate alone does not identify the backend.

The current bounded-memory c-coefficient caches are preserved. No cluster
job was submitted and no production amplitude was recomputed for this
policy change.

## Regression coverage

Run from the repository root:

```sh
PYTHONPATH=Code/c_Recursion:Code/higher_point_amplitude_attempts/type0b_ns_five_tachyon:Code python3 -m unittest test_ns_amplitude_c_only_policy test_bry_one_to_three test_type0b_sphere_four_point_hybrid test_sphere_multipoint test_ns_multipoint_c_recursion test_run_type0b_ns_five_tachyon_cluster test_fivepoint_runtime test_type0b_ns_five_tachyon_domain test_type0b_ns_five_tachyon test_sphere_four_point test_type0b_sphere_four_point_continuation test_ns_multipoint_h_recursion
```

The policy tests cover all public kernel defaults, CLI rejection of h/hybrid,
the c-only collar reference, and bulk/corner routing. The existing suites
exercise component blocks, subtraction identities, primary factorization,
chart/spin transport, cache reuse, and cluster planning/reduction without
submitting jobs.

Verification result: **145 distinct tests passed**, run in separate groups.
The legacy default-hybrid test was changed to request hybrid explicitly;
its h/c comparison assertions are retained. Python compilation and scoped
`git diff --check` passed. The human-note TeX and PDF hashes matched their
values at the start of this change.
