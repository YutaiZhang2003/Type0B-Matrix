# Verification report

Date: 2026-08-23

This report records checks performed **after** the screening-charge
derivation in `../r_branching_free_field.tex`.  The formula-side code uses
the literal finite external-colour path function at generic momentum.  The
comparison side
assembles the simultaneous `Vir x Vir` primary directly.  Both use the same
NS-R-R Ward functional, audited separately below; the grid is therefore an
exact path/normalization/phase check and is not an independent proof of the
Jack-Selberg average or of the all-level screening interpolation lemma.

## Direct first-crossed matrix elements

The corrected free-field trial is independent of the PBW/Ward oracle:

```sh
env PYTHONDONTWRITEBYTECODE=1 python3 \
  'Machine Notes/R branching coefficient/Code/direct_matrix_element_trial.py'
```

It evaluates the literal two-spin-field Pfaffian and Selberg integral for
`(0,3/4,3/4)`.  Exact symbolic reduction gives

```text
N=2, eta=-: H/(d2*d3) residual zero
N=3, eta=+: K/(d2*d3) residual zero
```

These are genuine contour restrictions.  They do not constitute the five
same-parity nodes needed for generic degree-four interpolation.

## Coverage through branch onset level 7/2

The positive labels with individual onset at most `7/2` are

```text
NS: n = 0, 1/2, 1
R:  n = 1/4, 3/4, 5/4
```

The audit covers all 27 triples, both Ramond-copy masters `epsilon2=0,1`,
both chiral forms `eta=+1,-1`, and two exact rational momentum samples.
After each master agrees, it restores all four `(epsilon3,f)` choices.

Command:

```sh
env PYTHONDONTWRITEBYTECODE=1 python3 \
  'Machine Notes/R branching coefficient/Code/verify_level_7_2.py'
```

Result:

```text
PASS: 216 exact path/PBW master comparisons;
864 exact discrete reductions; elapsed=3996.2s
```

Every residual was simplified exactly to zero.  In particular the maximal
triple `(1,5/4,5/4)` passed at both exact samples.

## Direct Ramond raw norms

Command:

```sh
env PYTHONDONTWRITEBYTECODE=1 python3 \
  'python 2/ramond_three_point_grid/compute_grid.py' --norm-check
```

Result: the direct finite Gram contractions agree with the factorized norm
at `n=1/4,3/4,5/4`, for both copies and both exact samples.  The symbolic
`n=1/4,3/4` checks and all 12 sampled checks have zero residual.  This is
the range used by the level-`7/2` branching audit.  The all-level product
norm remains a conjecture; the main note's unconditional formula instead
uses the exact finite Gram contraction.

## First crossed Ramond kernel

Command:

```sh
env PYTHONDONTWRITEBYTECODE=1 python3 \
  'python 2/ramond_three_point_grid/certify_master_ell_ansatz.py'
```

Result:

```text
exact hard-master formulas: residual=0 for all four masters at 2 exact samples
crossed identity: PASS
H irreducible over Q; H+iK and H-iK irreducible over Q(i)
single-product ansatz: 52/108 masters pass, 56/108 fail
```

Thus the first crossed component at `(0,3/4,3/4)` is genuinely the
two-colour quadratic kernel `H`, not a scalar four-ell product on a reflected
sheet.

## Generalized Ramond Ward signs

Command:

```sh
env PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH='Code/PBW_c_recursion_double_virasoro crosscheck' \
python3 \
  'Code/PBW_c_recursion_double_virasoro crosscheck/audit_ramond_pbw_generalized_ward.py'
```

The four Ramond ground normalizations, the `(2,1)` and `(1,2)` null vectors,
their Gram matrices, and the generalized parity anchor pass.  The audit also
exposes two literal-formula issues in the current human note rather than
hiding them:

- the inverse-norm product uses the `p+q` even sublattice;
- for NS-primary parity `p_phi`, the direct contraction is
  `-P((-1)^p_phi eta)` in the printed polynomial convention.

The branching calculation uses the corresponding effective chiral label
`eta_eff=(-1)^p_phi eta`; its remaining common sign is fixed by the declared
Ramond ground-state phase convention.
