# Ramond branching-coefficient check

Run

```bash
python3 -u check_ramond_branching.py
```

The script constructs the Ramond branch states from the two free-fermion
strings, converts every component to the abstract SCA PBW basis, builds the
Ramond Gram matrices from the commutation relations, and projects the even
copy on the Ramond Whittaker vector.

It checks all states with branch labels
`n = +/-1/4, +/-3/4` symbolically, including both parity copies.  It also
constructs the complete 30-component even states at `n = +/-5/4` and checks
their projections at the exact rational samples

- `b = 3/2, P = 2/5`, and
- `b = 5/3, P = 7/10`.

The comparison includes the conversion from the published coefficient's
block-coordinate normalization to the literal state convention
`L_1 |N> = (1/2) |N-1>` used in the notes.  At the branch onset
`N = 2 n^2 - 1/8`, the squared coefficient is therefore multiplied by
`4^(-N)`.

To compute the remaining finite NS--R--R restriction, run

```bash
python3 -u compute_ramond_kappa.py
```

This second check constructs the enlarged three-point matrix element for
the boundary family `(n_1,n_2,n_3)=(0,1/4,n)`.  It reduces the physical
factor with the NS--R--R Ward identities and the auxiliary factor with the
two-spin-field kernel.  All 16 choices of the two branch parities, physical
form parity, and Ramond chiral structure are checked symbolically at
`n=1/4,3/4`; all 16 are checked at two exact rational momentum samples at
`n=5/4`.  The result is

```text
P_2 in the numerator -> eta*(-1)^(2*n-1/2)*P_2
kappa^2 = eta*(-1)^(epsilon_3+2*n-1/2)*i^(1-f)/2.
```

Squaring removes the independent sign choice of each branch highest state.

For explicit low-state listings, see
`LOW_RAMOND_DOUBLE_VIRASORO_PRIMARIES.md` and run
`enumerate_ramond_double_virasoro_primaries.py`.  The symbolic enumeration
covers `n=+/-1/4,+/-3/4` and both parity copies; `n=+/-5/4` is available
after exact `Q,P` specialization.

The action of the physical SCA mode `L_1` in this branching basis is derived
in `PHYSICAL_L1_BRANCH_REDUCTION.md` and implemented by
`decompose_physical_l1.py`.  It includes the symbolic `|n|=3/4` reduction and
the first genuine descendant expansion at `n=5/4`.  The generalized exact
reducer also verifies at `n=7/4` that the image closes in the level-four
descendant space of `W_(3/4)` alone.

For the explicit PBW computation of the NS--R--R branching coefficient, the
ground and first excited closed formulas, the first crossed two-channel
coefficient, and the resulting all-label path-sum conjecture, see
`PBW_RAMOND_BRANCHING_COEFFICIENTS.md`.

To divide out the scalar Hadasz--Jaskolski screening/ell candidate in the
first two-excited Ramond case and verify the exact remainder, run

```bash
python3 -u strip_literature_ell_factor.py
```
