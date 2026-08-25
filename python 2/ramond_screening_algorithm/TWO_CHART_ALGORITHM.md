# Exact two-chart Ramond algorithm: proved steps and open step

This file states the strongest algorithm currently justified by the code. It
does not construct `v_n`, `W_n`, or a super-Virasoro PBW expansion, and it
does not hide the missing complementary Coulomb vertex behind a fitted
kernel.

## 1. Clear the known leg denominators

For branch labels

```text
n1 in Z/2,  n2,n3 in Z/2+1/4,
```

the denominator-cleared three-point function has degree at most

```text
D = (2 n1)^2 + (2 n2)^2 + (2 n3)^2 - 1/2
```

in any one external momentum. This is an integer. Hence `D+1` distinct
values determine one charge-chart polynomial exactly.

The function `momentum_degree_bound` in `two_chart_interpolation.py` computes
this integer. For `(0,3/4,3/4)`, it returns `D=4`.

## 2. The raw signed-path identity does not eliminate reflection

For the special 2016 Ramond branch representatives, the raw free-Fock
coefficients obey

```text
coeff_path[-n,P] = (-1)^g coeff_path[+n,-P].
```

Here `g` is the final physical Ramond ground label.  Equivalently, endpoint
`Z_0=diag(1,-1)` records the sign on the raw path boundary.  The finite audit
in `signed_chart_reduction.py` checks every path coefficient of both copies at
`n=1/4,3/4,5/4,7/4`: `4+8+16+32=60` exact coefficients.

This coefficient statement cannot be promoted to

```text
W_{-n}^epsilon(P) = (1_F tensor Z_0) W_n^epsilon(-P).
```

The two sides are converted to the abstract SCA module by different
free-field transition maps.  The first exact obstruction occurs at
`n=3/4`.  With

```text
d = 4 P^2 - 6 P Q + 2 Q^2 + 1,
```

the endpoint-only proposal misses two `L_-1` components in each copy.  For
`epsilon=0` the residuals are `4/d` and
`2 sqrt(2)(-1-i)/d`; for `epsilon=1` they are
`4(1+i)/d` and `4 sqrt(2)/d`.  Its first signed zero-screening value also
has nonzero residuals.  The executable exact certificate is
`reflection/audit_signed_state_obstruction.py`.

Therefore a negative SCA branch and every nonzero signed Coulomb value still
require the 2013 reflection operator.  Reflected representatives may be used
to infer homogeneous fusion zeros, as in the 2013 proof, because no signed
normalization is asserted there.  Endpoint `Z_0` remains useful only for raw
path and ground-boundary bookkeeping.

### 2.1 The eight zero charts do not leave only two constants

Let `sigma=(sigma1,sigma2,sigma3)` be one of the eight reflection choices
and put

```text
m_sigma = 2*(sigma1*n1 + sigma2*n2 + sigma3*n3).
```

At a charge-neutral point with `N=r+s` screenings, the 2013 `Psi_-A`
counting argument proves a zero when `m_sigma>0` and `N<m_sigma`, with the
required parity.  Opposite charts obey `m_(-sigma)=-m_sigma`, so only one
member of each global-sign pair contributes.  Summed over all eight charts,
the number of scalar zero rows is exactly

```text
D = (2*n1)^2 + (2*n2)^2 + (2*n3)^2 - 1/2.
```

For fixed form parity the denominator-cleared Ramond answer is a vector of
two degree-`D` polynomials.  It has `2*(D+1)` coefficients.  Each charge
chart supplies one covector on this pair, hence one scalar equation per
zero, not two.  The zero system consequently has nullity at least `D+2`.

One might also grant every saturation-plane value `N=m_sigma`.  This is a
strictly stronger assumption than the 2013 proof, which does not evaluate
mixed reflected correlators.  It is still insufficient.  The eight ground
covectors lie on two independent lines, labelled by
`(sigma1*sigma2)*(sigma1*sigma3)=+/-1`.  Within either line the constraint
matrix is an ordinary Vandermonde matrix, so its exact rank is the number of
distinct nodes, capped at `D+1`.  The audit finds

```text
labels                 unknowns   zero rank   zero+saturation rank   nullity
(0,3/4,3/4)               10          4                 9                1
(1,5/4,5/4)               34         16                25                9
(3/2,7/4,7/4)             68         33                46               22
```

For the balanced family
`(k/2,(2k+1)/4,(2k+1)/4)`, the degree is `3*k^2+2*k`, whereas the number
of saturation rows is only linear in `k`.  Thus the residual nullity grows
quadratically; there is no all-level two-constant reconstruction from the
deficient and saturation planes.

If a genuine reflected-vertex oracle supplied nonzero values on a full
two-dimensional lattice of interior nodes, then one could assemble enough
rows and a dense exact solve would cost `O(D^3)=O(S^6)`.  This statement is
conditional: the reflected oscillators mix bosonic currents and fermions,
so those interior values are not ordinary ground-resolved Pfaffians and are
not furnished by the 2013 zero proof.

The executable construction contains no Ward or target branching values:

```bash
python3 -m python.ramond_screening_algorithm.eight_chart_constraint_audit
```

## 3. Reconstruct two analytic Coulomb forms independently

Choose a charge chart with signs `s_i=+/-1` and solve

```text
Q/2 + s1 P1 + s2 P2 + s3 P3 = -r b - s/b
```

for `P1`. The native positive chart and the signed chart must each be
evaluated on their own nodes. To keep the Ramond structure fixed, one safe
starting value is the natural all-positive count
`N_0=2(|n1|+|n2|+|n3|)`. Keep its screening parity,

```text
N_j = N_0 + 2 j,  j=0,...,D.
```

At every node perform these operations anew:

1. keep the consecutive `chi` strings as contour insertions;
2. resolve only the finite Ramond zero-mode boundary space;
3. evaluate the auxiliary and physical fermions by ordinary or bordered
   Pfaffians;
4. apply the exact Selberg functional to the resulting symmetric screening
   insertion;
5. clear the known one-leg denominators.

The two resulting lists are separately interpolated. No value from one chart
may be copied to the other, and a closed formula inferred from the target
coefficient is not an admissible node callback.

For the hard case, `pfaffian/native_hard_screening.py` is a genuine positive
node evaluator.  It uses literal chi paths, both native ground-resolved
fermion kernels, the bosonic screening weight, and exact Selberg averages.
The independent audit checks both form parities at two rational samples for
every `N=0,1,2,3`; the natural maximal-screening node `N=3` is included.
The remaining hard interpolation nodes `N=5,7,9,11` have not been evaluated:
the direct symbolic expansion already stalls at `N=5`.  The full signed
callback is absent because it needs the reflected SCA vertex, not endpoint
`Z_0`.

After analytic reconstruction, each chart object is an SCA trilinear form.
The generic NS--R--R trilinear-form space is two-dimensional, so its
coordinates on `rho_f^+` and `rho_f^-` are fixed by its Ramond ground matrix.
This is the only point at which the constant two-by-two ground change is
used. Applying that ground matrix directly to excited same-plane Majorana
Pfaffians is incorrect.

Once two genuine SCA forms have been obtained, their ground matrices may be
ordered as `(canonical, canonical-with-boundary-Z)`.  The phrase
`canonical-with-boundary-Z` specifies the ground coordinates of the second
form; it is not a prescription for producing that form at excited level.
`boundary_side="right"` means that the reflected third-leg form has the
right-boundary ground matrix.  Reflecting the second leg gives `ZK` or `ZJ`
instead of `KZ` or `JZ`; since `ZJ=-JZ`, `boundary_side="left"` reverses the
second column in the odd form.

Explicitly, with `C=-(1-i)/sqrt(2)`, rows ordered by `eta=(+,-)`, the two
right-boundary maps are

```text
M_0 = 1/2 [[1+i, 1-i], [1-i, 1+i]],
M_1^(right) = 1/2 [[C(1-i), -C(1+i)], [C(1+i), -C(1-i)]],
M_1^(left)  = 1/2 [[C(1-i),  C(1+i)], [C(1+i),  C(1-i)]].
```

Their determinants are respectively `i` and a nonzero phase, so the inverse
contains no momentum-dependent solve. The code constructs and inverts these
matrices exactly.

`two_chart_interpolation.py` implements the neutrality nodes, exact
interpolation, and the nonsingular ground-space maps. Its built-in test makes
five independent callback calls in each chart and reconstructs two unrelated
degree-four polynomials exactly. This tests the orchestration; the callbacks
in that unit test are intentionally small algebraic mocks.

## 4. Hard finite certificate

For the first non-factorized case define

```text
E_j = Q+2P_j,
d_j = E_j^2+Q E_j+1,
M_j = [[d_j,E_j],[E_j,1]],
L   = (Q/2+P1+P2+P3)(-Q/2+P1-P2-P3).
```

The entries of `M_j` are the stripped one-leg fusion polynomials
`ell(E_j,3)`, `ell(E_j,2)`, and `ell(E_j,1)`. The finite two-leg construction
is

```text
K_23 = M_2 o M_3 + [[0,1],[1,0]],
H_C  = (1,L) K_23 (1,L)^T.
```

Here `o` is entrywise multiplication and the last matrix is the universal
zero-mode exchange. The construction contains no occurrence of the known
expanded hard polynomial. Only afterwards does
`hard_two_chart_certificate.py` import the independent expanded state-level
answer and prove the symbolic residual to be zero. It also reconstructs all
four hard phases,

```text
R_0^+ = -(1+i) K/(d_2 d_3),
R_0^- = -(1-i) H_C/(d_2 d_3),
R_1^eta = i sqrt(2) eta R_0^eta.
```

This is a noncircular algebraic certificate for the finite fusion-polynomial
kernel.  Separately, the natural positive `N=3` hard node now has a genuine
native Pfaffian--Selberg evaluation.  This is still not a native five-node
Coulomb reconstruction: the positive `N=5,7,9,11` evaluations and the
reflected Ramond vertex needed for the signed chart have not been implemented
independently. The older
`hard_complementary_pair_multiplier` is explicitly `H`-calibrated and cannot
serve as that callback.

## 5. Exact complexity

Let `S` be linear in the absolute branch labels. Then

```text
D = O(S^2),     K_ext = O(S),     N_max = N_0+2D = O(S^2).
```

There are at most sixteen ground sectors. One ground-resolved Pfaffian costs

```text
O((K_ext+N)^3)
```

field operations and `O((K_ext+N)^2)` memory.

Suppose, in addition, that the symmetric remainder at `N` screenings has a
proved Schur-width bound `w`. Put

```text
M = binomial(N+w,w).
```

The compound-Vandermonde reconstruction uses exactly `M` black-box Pfaffian
values and `O(M^2 w^3)` subsequent field operations. For both charts and all
nodes the conservative bounds are

```text
O((D+1) M (K_ext+N_max)^3 + (D+1) M^2 w^3)
```

time and

```text
O((K_ext+N_max)^2 + M)
```

working memory, apart from exact coefficient sizes. At fixed `w` this is
polynomial. More explicitly, with `N_max=O(S^2)`, the two displayed time
terms are `O(S^(2w+8))` and `O(S^(4w+2))`.

No uniform all-level bound on `w` has been proved for the complementary
Ramond vertex. If `w=O(S)`, then `M=exp(O(S log S))`; polynomial all-level
complexity does not follow. This is the precise remaining complexity
question.

There is an exact obstruction to setting `w=0` at every interpolation node.
The natural node `N=N0` is the rectangular external--screening determinant
and gives `C*Delta^2`.  At the first surplus node of
`(0,1/4,1/4)`, namely `N=3`, the polynomial is not divisible by
`Delta^2` and has Schur support

```text
(2,1), (1,1,1), (2,2), (2,1,1).
```

For `(0,3/4,3/4)`, the exact cleared one-variable degrees are `4` at
`N=3` and `6` at `N=5`; the latter is strictly below the degree `8` of
`Delta_5^2`, so divisibility is impossible.  The raw Schur rectangle at
the last hard interpolation node has `N=11`, width bound `12`, and
`binomial(23,12)=1,352,078` labels.  This is already far from a bounded
width callback.

The smallest surplus polynomial is nevertheless exactly the standard BFL
`(1,1,0)` spin polynomial up to the constant `(1-i)/2`.  Therefore the
audit excludes persistence of the determinant/width-zero formula, but does
not prove that no single Uglov, BFL, or holonomic integral recurrence exists.
No such arbitrary-label recurrence is currently known or implemented.

```bash
python3 -m python.ramond_screening_algorithm.pfaffian.audit_surplus_screening_width
```

All calculations can be made over primes `p=1 mod 8`, using the four
embeddings of `i` and `sqrt(2)`, followed by CRT and rational reconstruction.
Therefore the method has no floating-point accuracy loss even when the
operation count is large.

## 6. Why dual reflection does not solve the complexity problem

At grade `l`, stack the generic intertwining equations as

```text
A_-(l) R_l = B_l,
```

where `B_l` contains lower reflection blocks. To compute only a scalar
functional `phi R_l v`, one may solve the transposed system

```text
A_-(l)^T y = phi^T
```

and return `y^T B_l v`. This applies reflection implicitly and avoids
materializing all columns of `R_l` when only one output functional is needed.
It is an exact useful memory optimization.

It is not a fixed-size recurrence. The vector `y` still lives in the grade
`l` free-Fock block, whose dimension is `exp(O(sqrt(l)))`; exact worst-case
linear algebra remains cubic in that dimension. Since a `W_n` endpoint has
`l=O(n^2)`, no polynomial bound in `n` follows.  The raw coefficient identity
in Section 2 reduces path enumeration, but it does not remove this reflection
problem for signed SCA values.

## 7. Independent audit status

The following checks do not use the known buggy auxiliary Virasoro evaluator:

* raw signed-path coefficient identity: 60 exact endpoints through `W_7/4`,
  including all 16 endpoints of `W_5/4`;
* signed-state obstruction: the four nonzero abstract `L_-1` residuals at
  `n=3/4` and two nonzero signed zero-screening residuals are checked exactly;
* native Majorana kernel: 2048 canonical and 2048 boundary-`Z` mode
  strings, exhaustively covering all subsets of `(3,2,1)` on both Ramond
  legs, plus 16 ground-map entries; the boundary-`Z` checks concern only the
  raw ground/path kernel;
* native auxiliary kernel: all 512 distinct auxiliary endpoints occurring in
  the 432 stored restrictions agree with literal native Wick contraction;
* native positive hard screening: 16 exact comparisons, namely two rational
  samples, `N=0,1,2,3`, and both form parities; the natural `N=3` node passes;
* hard finite kernel: symbolic residual zero for `K` and the irreducible `H`,
  with exact `f=0,1` phases.

The old auxiliary evaluator is wrong from mode two because it sorted negative
Virasoro modes without their commutators. Therefore old stored comparisons
which contain mode two or higher are not valid evidence. A full corrected
grid audit through NS level 1 and Ramond level `5/4` is still separate work;
the present algorithm must not be described as having passed all those
values until that audit completes.

Run the current independent checks with

```bash
python3 -m python.ramond_screening_algorithm.signed_chart_reduction
python3 -m python.ramond_screening_algorithm.reflection.audit_native_spin_kernel
python3 -m python.ramond_screening_algorithm.reflection.audit_signed_state_obstruction
python3 -m python.ramond_screening_algorithm.pfaffian.auxiliary_ising_kernel
python3 -m python.ramond_screening_algorithm.pfaffian.audit_native_hard_screening
python3 -m python.ramond_screening_algorithm.pfaffian.hard_two_chart_certificate
python3 -m python.ramond_screening_algorithm.two_chart_interpolation
```
