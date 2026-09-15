# Direct enlarged torus two-point block with two NS external primaries

The two spheres each have an external NS primary at infinity and internal
Ramond punctures at 1 and 0. Glue the 1 punctures with q2 and the 0 punctures
with q3. This is the frame obtained by removing the NS tube from the theta
graph. The two NS external weights are independent. Converting to an annular
necklace frame changes coordinates and normalization factors, not an
identically vanishing block. Do not identify the torus nome with q2*q3 in
this pants frame without a coordinate conversion.

The external auxiliary fields are both the identity. Parameters are b=7/5,
NS external momenta (11/23,19/37), internal R momenta (13/29,17/31), and
external intrinsic parities zero. The first three-point sign is eta=+1;
both choices eta'=+1,-1 and both form parities f=0,1 are evaluated.

## Calculation

The executable builds tensor-product states on each internal edge, constructs
their full Gram matrices at each total level and total parity, inverts those
matrices at 136-bit MPC precision, and contracts the two enlarged vertex
matrices. Physical and auxiliary Ward evaluators are reused from the existing
C++ implementation. The two vertices use different NS external weights.

The physical basis uses the native Gram convention and the existing ground
phases to convert the physical Ward forms. Auxiliary norms are
(-1)^(number of oscillators + auxiliary ground parity). Each enlarged vertex
includes the tensor Koszul sign (-1)^(physical parity on edge 2 times
auxiliary parity on edge 3). The final sewing includes
(-1)^(total edge-2 parity times total edge-3 parity).

There is no convolution, no sector projection, and no imposed vanishing
identity in the calculation. Both contributions to each Ramond ground
doublet are explicitly summed.

## Results

Independent cutoffs are 0 <= level2,level3 <= 3, including the (3,3) corner.
There are 16 monomials and four formal spin-parity slots in each sign/form
sector. A slot index e2+2*e3 multiplies eta2^e2 eta3^e3.

| Form parity | Maximum absolute coefficient, eta eta'=-1 |
|---|---:|
| f=0 | 2.5714e-38 |
| f=1 | 4.5918e-39 |

The coefficients of 1 and eta2*eta3 in the f=0, eta eta'=+1 result agree
to working precision. Some values of either coefficient are:

| level2 | level3 | Coefficient |
|---:|---:|---:|
| 0 | 0 | 2 |
| 1 | 0 | 0.809229423287000619 |
| 0 | 1 | 0.895261130342204643 |
| 1 | 1 | 5.50578189036800744 |
| 2 | 0 | 0.576095621098942024 |
| 3 | 3 | 739.297712501228475 |

For f=1, the corresponding nonzero slots are i*eta2 and -i*eta3 with
the same listed coefficients. The internal elapsed time for the entire
calculation, including both f values and both sign choices, is 0.134917 s;
compilation is excluded.

## Ground-state explanation

With leading plumbing powers suppressed, the even-form ground contribution
is the explicit sum over physical ground parity a and auxiliary ground parity b:

    sum_{a,b=0,1} (eta*eta')^a (eta2*eta3)^(a+b)
      = (1+eta*eta') (1+eta2*eta3).

It vanishes when eta*eta'=-1. The physical ground coefficient alone is
1+eta*eta'*eta2*eta3; it is not identically zero in the negative sector.
At eta2*eta3=-1 it equals 2. The auxiliary ground coefficient is
1+eta2*eta3. The direct descendant calculation confirms the same cancellation
in the enlarged block through the stated cutoff.

## Reproduction

From the repository root:

```sh
clang++ -I C++/include -I/opt/homebrew/include -O3 -std=c++17 \
  -Wall -Wextra -Wpedantic \
  C++/experiments/torus_ns2_enlarged_2026-09-14/check.cpp \
  -L/opt/homebrew/lib -lmpc -lmpfr -lgmpxx -lgmp \
  -o C++/experiments/torus_ns2_enlarged_2026-09-14/check
C++/experiments/torus_ns2_enlarged_2026-09-14/check \
  C++/experiments/torus_ns2_enlarged_2026-09-14/result.json
```

Full coefficients are retained in `result.json`; `provenance.json` records
the source and executable hashes. Production code and manuscript were not
modified for this check.
