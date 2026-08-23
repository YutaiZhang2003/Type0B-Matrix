# Ramond branching coefficient

This directory contains the free-field derivation of the normalized
`(NS,R,R)` branching coefficient in the conventions of
`Human Notes/SCblock.tex`.

The order of work is intentionally:

1. reconstruct the Hadasz--Jaskolski NS screening-charge argument;
2. derive the Ramond spin-field screening integral and its closed finite
   evaluation;
3. only then compare with the independent PBW/Ward calculation through
   branching onset level `7/2`.

The target coefficient is

```text
Bhat_f^(eta)(P1,P2,P3;n1,n2,n3;epsilon2,epsilon3)
```

with

```text
n1 in Z/2,
n2,n3 in Z/2 + 1/4.
```

Here `eta` labels the two chiral `(NS,R,R)` superconformal structures and
`epsilon2,epsilon3` label the two copies of each Ramond
`Vir x Vir` summand.  These discrete labels cannot be suppressed before a
Ramond zero-mode basis has been fixed.

Main artifacts:

- `r_branching_free_field.tex` and `r_branching_free_field.pdf`: derivation,
  finite Pfaffian/Jack-Selberg formula, normalized coefficient, and low-level
  consequences;
- `Code/screening_kernel.py`: the literal callable
  `branching_square(b,p1,p2,p3,n1,n2,n3,epsilon2,epsilon3,f,eta)` and the
  finite colour-path evaluation of its pole-cleared Ramond kernel.  The
  checked-in transition tables cover exactly the requested range
  `n1 in {0,1/2,1}`, `n2,n3 in {1/4,3/4,5/4}`;
- `Code/verify_level_7_2.py`: exact comparison with direct simultaneous
  primaries through branch onset level `7/2`;
- `Code/verification_report.md`: commands, scope, results, and caveats.
- `DIRECT_MATRIX_ELEMENT_TRIAL.md` and
  `Code/direct_matrix_element_trial.py`: the corrected, genuinely
  free-field calculation of the first crossed state on the two- and
  three-screening neutrality planes.

Status correction: equation `literal-A` in the older PDF is a finite
PBW/Ward reconstruction, not by itself a screening-charge derivation.  The
direct matrix element and the strongest presently justified screening
result are recorded in `DIRECT_MATRIX_ELEMENT_TRIAL.md`.  A complete
generic-momentum Ramond formula is not yet established.

Current residue-recursion status: the ordinary scalar Selberg argument is
complete because endpoint power counting fixes the Gamma divisor and a
collision residue fixes the one remaining normalization recursively.  The
Ramond descendant integral is instead vector-valued.  Its two presently
proved endpoint residues contain complementary rank-one projectors, so
neither residue by itself determines the full integral.  A complete proof
must first identify the finite contour-orbit space, prove its degree/pole
bounds, compute a full-rank collection of collision maps together with the
inversion/crossing connection matrix, and fix the result from the `N=0,1`
base vectors.  Only after this Selberg problem is solved can the separate
same-parity neutrality-node interpolation reconstruct generic Liouville
momenta.  The projectors are therefore part of the evaluation algorithm,
not part of the definition of the branching coefficient.

The finite external-colour path sum `literal-A` is an exact PBW/Ward oracle,
not a derivation of the screening integral.  The direct contour calculation
currently establishes two nontrivial restrictions of the first crossed
state: the `N=2, eta=-` integral produces `H/(d2*d3)`, and the natural
`N=3, eta=+` integral produces `K/(d2*d3)`.  Dividing by the NS-like
four-ell factor therefore leaves a momentum-dependent kernel; it is not a
field-normalization constant such as `Omega_k(alpha)`.  Generic-momentum
interpolation, closure of the all-level contour-orbit recurrence, and the
compact all-level Ramond norm product remain separate conjectural steps.
