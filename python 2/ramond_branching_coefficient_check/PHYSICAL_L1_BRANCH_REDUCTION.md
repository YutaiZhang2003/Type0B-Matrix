# Physical L1 reduction in the Ramond branching basis

## Which L1 is meant

There are three different modes in the enlarged module.  A double-Virasoro
primary obeys

```text
L_m^(1) W_n^epsilon = L_m^(2) W_n^epsilon = 0,   m > 0.
```

The physical super-Virasoro mode `1_F x L_m` is not either of these modes.
Since

```text
L_m^(1) + L_m^(2) = L_m + L_m^F,
```

its action on a branching primary satisfies the useful identity

```text
L_1 W_n^epsilon = -L_1^F W_n^epsilon.
```

Here and below an undecorated `L_1` means the physical SCA mode.

## General triangular reduction

The Ramond branch with label `n` starts at total enlarged-module level

```text
N_n = 2 n^2 - 1/8.
```

Therefore `L_1 W_n^epsilon` lies at level `N_n-1` and has the unique generic
expansion

```text
L_1 W_n^epsilon
  = sum C(n;n',A,B)
      L_-A^(1) L_-B^(2) W_n'^epsilon,

N_n' + |A| + |B| = N_n - 1.
```

The parity copy `epsilon` is unchanged because `L_1` is even.  The code
constructs every allowed column in the common auxiliary-fermion x physical
Fock basis and solves this finite linear system exactly.  This is triangular
in the branch-onset level.

## First step: n = 3 sigma/4

For `sigma=+1,-1` and the state normalization in
`LOW_RAMOND_DOUBLE_VIRASORO_PRIMARIES.md`, the answer is

```text
L_1 W_(3 sigma/4)^(0)
  = (1/4) W_(-sigma/4)^(0),

L_1 W_(3 sigma/4)^(1)
  = (1/2) W_(-sigma/4)^(1).
```

This target is at level zero, so no lowering operators occur yet.
In the PBW calculation the factor `3/2` in
`[L_1,G_-1]=(3/2)G_0` is essential; a direct free-field calculation checks
the result independently.

## First genuine descendant step: n = 5/4

Put `x=2bP` and `Q=b+b^(-1)`.  For either parity copy,

```text
L_1 W_(5/4)^epsilon
  = A L_-2^(2) W_(1/4)^epsilon
    + B (L_-1^(2))^2 W_(1/4)^epsilon
    + C L_-1^(1)L_-1^(2) W_(1/4)^epsilon
    + D L_-2^(1) W_(1/4)^epsilon
    + E (L_-1^(1))^2 W_(1/4)^epsilon,
```

where

```text
A = -(x+b^2+6)/[(x+b^2+4)(x+2b^2+1)],

B = 2(2x+3b^2+7)
    /[(x+b^2+2)(x+b^2+4)(x+2b^2+1)],

C = 8b^2/[(x+b^2+2)(x+2b^2+1)],

D = -b^2(x+6b^2+1)
    /[(x+b^2+2)(x+4b^2+1)],

E = 2b^4(2x+7b^2+3)
    /[(x+b^2+2)(x+2b^2+1)(x+4b^2+1)].
```

The complete level-two target basis also contains descendants of
`W_(-1/4)` and `W_(+/-3/4)`.  Their coefficients vanish in the exact solve.
The `n=-5/4` expression follows in the reflected convention by sending
`P -> -P` and `W_(1/4) -> W_(-1/4)`.

The formulas hold at generic momentum and coupling.  A zero denominator
signals a degenerate module where this Verma-basis decomposition must be
replaced by a quotient or a limiting prescription.

## Next example and selection conjecture

The preceding cases suggest, for positive Ramond labels,

```text
L_1 W_n^epsilon belongs only to the W_(n-1)^epsilon module,
at relative descendant level 4n-3.
```

For `n=7/4`, the target is the 20-dimensional level-four descendant space of
`W_(3/4)^epsilon`.  The exact reducer closes in that space at both
`(b,P)=(3/2,2/5)` and `(5/3,7/10)`; at the first point all 20 coefficients
are nonzero and agree for the two parity copies.  The other lower branch
modules account for 102 additional target-level basis vectors, none of which
is required.  This supports the all-label rule but is not yet a symbolic
proof at `n=7/4`.

## Commands

The onset calculation is the default:

```bash
python3 "python 2/ramond_branching_coefficient_check/decompose_physical_l1.py"
```

Include the level-two reduction, symbolically or at an exact value of `b`:

```bash
python3 "python 2/ramond_branching_coefficient_check/decompose_physical_l1.py" \
  --five-quarters

python3 "python 2/ramond_branching_coefficient_check/decompose_physical_l1.py" \
  --five-quarters --b 3/2

python3 "python 2/ramond_branching_coefficient_check/decompose_physical_l1.py" \
  --higher-label 7/4 --b 3/2 --momentum 2/5
```
