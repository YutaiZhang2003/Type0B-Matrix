# Directed theta checks for the plumbing-graph notes

The authoritative results are `theta_summary.json`, `theta_all_ns_results.json`,
and `theta_ramond_local_results.json`. No draft or production source is changed.

Reproduce from this directory:

```sh
make -f theta_makefile all
python3 theta_tests.py
python3 theta_all_ns.py
./theta_ramond_local theta_ramond_local_results.json
```

## NS–R–R theta and split theta

`theta_validate.cpp` uses the explicitly transported vertex from the preceding
isolated experiment, both Virasoro CCY recursions, and the Schottky vacuum.
The comparison independently constructs physical SCA PBW coefficients and the
enlarged auxiliary-Fock/SCA-PBW contraction. No enlarged reference numerator is
constructed by multiplying already known physical and fermion blocks.

The parameters are `b=7/5`, `P=(11/23,13/29,17/31)`, with 40-digit arithmetic,
total level at most 3, NS primary parity `p=0,1`, form parity `f=0,1`, and all
four endpoint-sign pairs. Each of these 16 cases retains all eight parity
components and all eight plumbing-spin evaluations. These are closed blocks
with no external punctures. There are 30 diagonal monomials (240 components)
per case.

The inserted full series retains independent `q_L,q_R` powers, with cutoff
`a+l+r+d<=6` in `q1^(a/2) qL^l qR^r q3^(d/2)` and even `d`. This gives 130
monomials (1,040 components) per inserted case. The explicit middle Fock action
is `Q Theta psi_(l-r)`. Both the numerator and the auxiliary factor include
this same `Q`, which cancels on recovery.

| Comparison | Components | Maximum scaled error |
|---|---:|---:|
| Recovered double Virasoro vs direct physical PBW | 3,840 | `7.32e-28` |
| Restored double-Virasoro numerator vs direct enlarged PBW | 3,840 | `2.20e-27` |
| Explicit graph convolution vs direct enlarged PBW | 3,840 | `7.73e-41` |
| Full split double-Virasoro numerator vs direct split PBW | 8,320 | `7.06e-26` |
| Full split graph convolution vs direct split PBW | 8,320 | `6.29e-41` |
| Opposite endpoint signs, no insertion: enlarged PBW vanishes | 1,920 | `4.34e-40` |

All 136 comparisons pass tolerance `1e-20`. The suite also checks the
sector identity, literal enlarged recovery, all plumbing-spin evaluations,
and that the split diagonal equals the zero-mode calculation. The diagonal
identities agree exactly in the saved arithmetic. Sixteen fresh processes
took about 3.61 seconds together; this is a validation-suite time, not a
high-level block benchmark.

The graph kernel is independently obtained by permuting the six half-edges,
then compared to the polarized theta quadratic form in all 64 parity pairs.
Its associativity is checked for all 512 parity triples.

## Local Ramond transport identity

`theta_ramond_local.cpp` directly evaluates the physical Ward form in the
human ground-state basis. It verifies
`rho(x,Jy,Jz)=-i eta (-1)^parity(y) rho(x,y,z)` in 864 cases, including both NS
primary parities, both form parities, both signs, NS primary/`G_-1/2`/`L_-1`
states, and ground/`G_-1`/`L_-1` Ramond states with both ground parities.
The maximum scaled error is `2.90e-41` at 40 digits. This test addresses the
local identity used in the arbitrary-graph loop-sector argument.

## All-NS local vertex and theta block

`theta_all_ns.py` implements the current human-note local convention
`(-1)^(B delta3+p1 A+p2 C) rho_F rho_a`, explicitly. The historical
generic-primary-parity audit used the different provisional crossing
`(-1)^((p2+B) delta3)`; its reported discrepancy does not apply here.

The current physical Ward forms obey
`rho_a^(p)=(-1)^(p1 A+p2 C) rho_a^(0)`, so the current enlarged form is
independent of intrinsic primary parity term by term. Fresh exact rational
checks pass for all 8 primary-parity tuples and all 8 branch-label tuples
`n_i in {0,1/2}` (64 comparisons), followed by all three `L_-1` Ward identities
in both Virasoro copies (384 checks).

A fresh all-NS closed theta comparison with even primaries passes for 35
coefficients through total level 2: maximum scaled error `4.67e-15`. All
eight spin lifts and both vertex-parity sectors pass evaluation checks.
This older all-NS backend uses complex-double series arithmetic, with
40-digit branching-polynomial evaluation; it is not a 40-digit end-to-end
test. No physical PBW result is used to generate its double-Virasoro series.

## External-state expansion

`external_state.cpp` separately tests the BPZ/inverse-Gram expansion added to
the notes. Build with `make -f external_state_makefile`, then run
`./external_state external_state_results.json`.

It reconstructs every tensor-PBW state at NS levels 0, 1/2, 1 and R levels
0, 1: 21 independent inputs, including all four Ramond ground tensors and
all 12 level-one Ramond tensors. The 169 matrix entries of the normalized
branching-basis Gram agree with the independently known Virasoro Gram
entries (1 for a primary, `2h` for an `L_-1` descendant), with maximum
scaled error `1.27e-30`. Reconstruction by BPZ overlap followed by these
inverse Virasoro Grams has error at most `1.29e-30`, at 40 digits.

Branching vectors are obtained from `FreeField` primary/generator actions;
their overlaps are computed in the physical SCA PBW times auxiliary basis.
A coordinate conversion into that PBW basis is used only to evaluate the
overlaps. The final expansion coefficients are calculated by the prescribed
BPZ/inverse-Gram formula, not by fitting an input to the branching basis.

These checks do not test full correlators with external insertions, and do
not by themselves prove identities at arbitrary genus.
