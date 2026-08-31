# Type-0B sphere four-point review manifest

This directory is the review entry point for the current Type-0B NS sphere
four-tachyon computation.  The source files remain in their original
locations so there is only one copy of each implementation.

The default evaluator now uses the complete continued Liouville contour in
the certified tilted complex-energy chamber. It must pass a pointwise
crossing-frame gate before it is allowed to integrate. The old equal
pure-imaginary fixed-contour calculation is retained only as an explicitly
labelled diagnostic because it omits crossed poles. All paths return the
reduced BRY quantity before the overall string normalization.

## Start here

1. [`../evaluate_type0b_sphere_four_point_hybrid.py`](../evaluate_type0b_sphere_four_point_hybrid.py)
   is the command-line entry point and records every numerical setting in
   JSON.
2. [`../type0b_sphere_four_point_hybrid.py`](../type0b_sphere_four_point_hybrid.py)
   is the amplitude kernel, convergence audit, Liouville residue ledger,
   crossing atlas, finite-part experiments, and QMC integration code.
3. [`../test_type0b_sphere_four_point_hybrid.py`](../test_type0b_sphere_four_point_hybrid.py)
   contains the focused regression tests.

The main kernel is intentionally presented without claiming that every path
inside it is production-ready. The complete-contour path is a candidate until
its crossing gate passes at converged settings. The direct equal-energy
finite-part functions remain experimental.

## Runtime dependency closure

The four-point kernel directly uses the following local modules:

- [`../sphere_four_point.py`](../sphere_four_point.py): the BRY
  `G,H,J` correlators, momentum integral, and four-point h-recursive wrapper.
- [`../ns_multipoint_h_recursion.py`](../ns_multipoint_h_recursion.py): the
  fixed-difference NS sphere h recursion.
- [`../ns_multipoint_c_recursion.py`](../ns_multipoint_c_recursion.py): the NS
  sphere c recursion used in the collars.
- [`../superconformal_blocks.py`](../superconformal_blocks.py): elliptic nome,
  finite-c four-point block, and series operations.
- [`../super_liouville_structure_constants.py`](../super_liouville_structure_constants.py):
  NS and twisted-NS super-Liouville three-point constants.
- [`../ns_global_osp_block.py`](../ns_global_osp_block.py): global
  `osp(1|2)` seed data.
- [`../ns_recursion_recipe.py`](../ns_recursion_recipe.py): shared NS null
  weights, fusion polynomials, and residue kernels.
- [`../../bosonic_c1_one_to_n_reference/reference_implementation/plumbing/sphere_five_point_liouville.py`](../../bosonic_c1_one_to_n_reference/reference_implementation/plumbing/sphere_five_point_liouville.py):
  projective-point and Mobius-frame utilities reused by the four-point atlas.
- [`../../higher_point_amplitude_attempts/type0b_ns_five_tachyon/type0b_ns_five_tachyon.py`](../../higher_point_amplitude_attempts/type0b_ns_five_tachyon/type0b_ns_five_tachyon.py):
  the positive-half-contour pole enumeration reused by the continued path.

The last two imports are much larger than the pieces actually used.  They are
listed explicitly because hiding them would make the review incomplete.  A
later cleanup should move the Mobius utilities and contour-pole ledger into
small shared modules; that refactor is not part of the present numerical
result.

## Reference and validation code

These files are useful for comparison but are not imported by the default
evaluator:

- [`../bry_one_to_three.py`](../bry_one_to_three.py): the direct BRY
  subtraction implementation and matrix-model comparison infrastructure.
- [`../test_sphere_four_point.py`](../test_sphere_four_point.py): lower-level
  four-point block and correlator tests.
- [`../compare_ns_sphere_c_h_recursion.py`](../compare_ns_sphere_c_h_recursion.py):
  direct c-versus-h recursion comparison.

The two historical checked-in outputs are fixed-contour diagnostics, not
certified amplitudes:

- [`../results/type0b_sphere_four_point_pure_imaginary_t06_folded_hybrid.json`](../results/type0b_sphere_four_point_pure_imaginary_t06_folded_hybrid.json)
- [`../results/type0b_sphere_four_point_pure_imaginary_t06_folded_hybrid_p20.json`](../results/type0b_sphere_four_point_pure_imaginary_t06_folded_hybrid_p20.json)

## Reproduce the focused checks

From the repository root:

```bash
PYTHONPATH=Code/c_Recursion python3 -m unittest \
  Code/c_Recursion/test_type0b_sphere_four_point_hybrid.py
```

Run the crossing-gated continued candidate with:

```bash
PYTHONPATH=Code/c_Recursion python3 \
  Code/c_Recursion/evaluate_type0b_sphere_four_point_hybrid.py
```

The numerical convergence axes are independent:

- `--twice-level`: superconformal-block truncation;
- `--momentum-order` and `--momentum-maximum`: Liouville momentum quadrature;
- `--sobol-power` and `--replicates`: moduli-space sampling;
- `--corner-radius`: h/c-recursion routing boundary.

The reproducible wall-1 through wall-4 power-counting certificate is exposed
as `certify_residue_convergent_ray_rectangle()` in the main kernel and tested
by `test_large_wall_four_domain_has_uniform_residue_convergence`. It certifies
the larger rectangle `x in (0.965,1.055)`, `t in (1.185,1.265)` for both the
continuum and every residue stratum. It does not certify the present numerical
evaluation of the order-three/four residues or crossing.

## Suggested review order inside the main kernel

1. `audit_four_point_convergence`: endpoint and crossed-residue power count.
2. `audit_four_point_crossing`: mandatory pointwise channel comparison.
3. `Type0BSphereFourPointHybrid.__init__`: conventions and chamber guard.
4. `_pco_terms` and `density_components`: BRY picture-changing combination.
5. `_selected_backend` and the block builders: h/c-recursion routing.
6. `integrate_subtraction_free_four_point`: continued-chamber atlas integral.
7. `folded_unit_disk_density`: the fixed-diagnostic crossing patch.
8. The functions containing `continued_finite_part`: experimental analytic
   continuation and meromorphic finite-part alternatives.
