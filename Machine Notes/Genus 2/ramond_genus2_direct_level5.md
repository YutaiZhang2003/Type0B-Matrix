# Genus-two Ramond theta block: direct truncation through level five

Date: 2026-08-08

## Scope and convention

This is a direct finite-level computation of the chiral genus-two `NRR`
theta block.  It does not use a Kac determinant, a regular seed, or a
Zamolodchikov recursion to construct the coefficients.

The ordered trinion is

\[
 \rho_{RR}(\eta_1,\xi_N,\eta_2\mid 1),
\]

with one NS module and two generic long-R modules.  The R ground basis is

\[
 e_+=w^+,
 \qquad
 e_-=G_0w^+,
 \qquad
 B_R^{(0)}=\operatorname{diag}(1,h_R-c/24).
\]

At each of the two vertices the even HJS form sign is `+`, and the ordered
plumbing lifts are `(xi_N,xi_R1,xi_R2)=(+,+,+)`.  The two R edges are sewn
with the identity, and the leading coefficient `1+sign_L sign_R=2` is
divided out.  No `G_0`, parity, GSO, or nonchiral spin-sum insertion is
included.  Thus this is one normalized chiral block, not a physical
genus-two partition function.  The infinity-half-edge sign uses the same
determinant-one BPZ lift and Koszul contraction ledger as the ordered NS
theta computation; changing plumbing lifts selects other chiral components.

The numerical benchmark is

\[
 c=37.25,
 \qquad h_N=0.73,
 \qquad (\beta_1,\beta_2)=(0.67,0.83),
 \qquad h_{R,a}=\frac{c}{24}-\beta_a^2,
\]

so

\[
 (h_{R,1},h_{R,2})
 =(1.10318333333333,\ 0.863183333333333).
\]

Write

\[
 \mathcal F_\Theta^{NRR}
 =\sum_{\ell_N,\ell_1,\ell_2}
 D_{\ell_N,\ell_1,\ell_2}
 q_N^{\ell_N}q_1^{\ell_1}q_2^{\ell_2}.
\]

Here \(\ell_N\in\tfrac12\mathbb Z_{\geq0}\), while
\(\ell_1,\ell_2\in\mathbb Z_{\geq0}\).  The cutoff is the total physical
level

\[
 \ell_N+\ell_1+\ell_2\leq5.
\]

For the equal-form identity sewing used here, all half-integral-total-level
coefficients cancel.  The largest numerical remainder among the 35 such
coefficients through level five is `3.11e-15`; they are zero at the accuracy
of this computation.

## Direct result through level two

The previously requested low-level block is

\[
\begin{aligned}
 \mathcal F_\Theta^{NRR,[2]}
={}&1\\
 &+0.164452054794521\,q_N
  +0.396205062642791\,q_1
  +0.135945631637545\,q_2\\
 &+0.118109371891463\,q_N^2
  +0.0705842032831438\,q_Nq_1
  +0.206721162190762\,q_Nq_2\\
 &+0.265908249357633\,q_1^2
  +1.55132652614021\,q_1q_2
  +0.0893735696177088\,q_2^2
  +O(\text{total level}>2).
\end{aligned}
\]

Equivalently, the six new total-level-two coefficients are

| \((\ell_N,\ell_1,\ell_2)\) | direct coefficient |
|---:|---:|
| `(0,0,2)` | `0.0893735696177088` |
| `(0,1,1)` | `1.55132652614021` |
| `(0,2,0)` | `0.265908249357633` |
| `(1,0,1)` | `0.206721162190762` |
| `(1,1,0)` | `0.0705842032831438` |
| `(2,0,0)` | `0.118109371891463` |

## New levels three through five

### Total level three

| \((\ell_N,\ell_1,\ell_2)\) | direct coefficient |
|---:|---:|
| `(0,0,3)` | `0.0688636300044048` |
| `(0,1,2)` | `0.314825146372786` |
| `(0,2,1)` | `0.855135721991051` |
| `(0,3,0)` | `0.429135698944309` |
| `(1,0,2)` | `0.379537718484080` |
| `(1,1,1)` | `0.255118834881003` |
| `(1,2,0)` | `0.415272191342697` |
| `(2,0,1)` | `0.415792735781165` |
| `(2,1,0)` | `0.155972356800660` |
| `(3,0,0)` | `0.0951211138617139` |

### Total level four

| \((\ell_N,\ell_1,\ell_2)\) | direct coefficient |
|---:|---:|
| `(0,0,4)` | `0.0569081178434706` |
| `(0,1,3)` | `0.161952243719599` |
| `(0,2,2)` | `3.56879944580329` |
| `(0,3,1)` | `0.512009689801185` |
| `(0,4,0)` | `1.47703652538740` |
| `(1,0,3)` | `0.574497191655241` |
| `(1,1,2)` | `0.478728292782345` |
| `(1,2,1)` | `0.152343014582104` |
| `(1,3,0)` | `1.85177932665688` |
| `(2,0,2)` | `2.29451303724158` |
| `(2,1,1)` | `1.00263814579737` |
| `(2,2,0)` | `1.35043139809936` |
| `(3,0,1)` | `0.654809304673767` |
| `(3,1,0)` | `0.0713704693144673` |
| `(4,0,0)` | `0.0810414157840814` |

### Total level five

| \((\ell_N,\ell_1,\ell_2)\) | direct coefficient |
|---:|---:|
| `(0,0,5)` | `0.0489463134665771` |
| `(0,1,4)` | `0.125370993906238` |
| `(0,2,3)` | `0.720406975251532` |
| `(0,3,2)` | `2.04140502732324` |
| `(0,4,1)` | `0.894948799488017` |
| `(0,5,0)` | `4.95244233570810` |
| `(1,0,4)` | `0.785803662011062` |
| `(1,1,3)` | `0.687753497456086` |
| `(1,2,2)` | `0.586896402011898` |
| `(1,3,1)` | `0.799611845010740` |
| `(1,4,0)` | `12.4638614359077` |
| `(2,0,3)` | `7.14394501300375` |
| `(2,1,2)` | `2.90837803250245` |
| `(2,2,1)` | `1.89335348857570` |
| `(2,3,0)` | `18.0867400551341` |
| `(3,0,2)` | `7.24313871167835` |
| `(3,1,1)` | `1.06997374604141` |
| `(3,2,0)` | `0.184292624555106` |
| `(4,0,1)` | `0.915046075677301` |
| `(4,1,0)` | `0.0506115762371577` |
| `(5,0,0)` | `0.0713680188557200` |

There are 91 multidegrees through total level five: 56 integral-total-level
coefficients listed above (including level zero) and 35 half-integral terms
that cancel in this sector.

## Power-counting status

The edgewise super-Virasoro Gram--Schmidt step is now explicit in
`ramond_channel_c_recursion.tex`.  At fixed beta, every nonzero Ramond mode
has order-`c` squared norm, while `G_0` acts only in the finite ground fibre.
If `nu_R(I)` counts the nonzero modes in a canonical PBW word, the diagonal
Gram entry is `kappa_I c**nu_R(I)`, an off-diagonal entry with the same mode
count is at most `O(c**(nu_R(I)-1))`, and an entry with unequal mode counts is
at most `O(c**min(nu_R(I),nu_R(J)))`.  Ordering first by `nu_R` therefore
gives a triangular orthogonal basis: lower-count admixtures are `O(1)`,
equal-count admixtures are `O(1/c)`, and the orthogonal norm retains its
diagonal leading power.

This completes the module-by-module Gram power count used by each Ramond
propagator.  It does **not** yet determine the genus-two regular seed.  For
that step the two trinion tensors must be reduced simultaneously in the
edgewise orthogonal bases, with the `G_0` fibre kept open and the fermionic
half-edges placed in canonical Koszul order.  Source contractions on one
edge can then compensate an unpaired oscillator on another edge, so the
sphere suppression argument cannot be applied independently edge by edge.

## Comparison with the current recursion formula

The comparison must be analytic.  A finite displacement from a Kac pole is
only a convergence diagnostic and is not used as the reported test.

The direct answer supports the edge-local pole part but does **not** yet
support a closed scalar genus-two Ramond recursion.  At level one the
comparison can be made algebraically because, at fixed beta,

\[
 h_R=\frac c{24}-\beta^2
\]

makes every entry of the level-one R Gram matrix affine in \(c\).  If
\(v\) is its null vector at \(c=c_*\), then

\[
 \underset{c=c_*}{\operatorname{Res}}B_R(c)^{-1}
 =\frac{vv^T}{v^TB'_R(c_*)v}.
\]

This is an algebraic inverse-Gram residue; no `epsilon` or Laurent fit is
present.

For the first Ramond Kac channel `(r,s)=(2,1)` on edge 1, direct Laurent
factorization of the coefficient \(D_{0,1,0}\) gives

\[
 \underset{c=c^R_{2,1}(\beta_1)}{\operatorname{Res}}
 D_{0,1,0}
 =-
 \frac{A^{(\beta),R}_{2,1}}
 {\partial_c\beta_{2,1}}
 P^+_{2,1}(\beta_2,h_N)^2.
\]

The two algebraic comparisons are

| branch | direct inverse-Gram residue | published kernel | relative error |
|---:|---:|---:|---:|
| `+1` | `5.606999999999999 + 3.875399731061870 i` | `5.607 + 3.875399731061872 i` | `2.35e-16` |
| `-1` | `5.606999999999999 - 3.875399731061870 i` | `5.607 - 3.875399731061872 i` | `2.35e-16` |

Thus the fixed-beta Jacobian, inverse null norm, and the two local R-R-NS
fusion factors match the direct Ward/Gram computation to machine accuracy.

This local equality must not be promoted into a scalar higher-level tail.
Beginning at the next shifted level, the correct residue has the form

\[
 \mathbf P_L\,
 \mathbf F_\Theta(c_*,\beta_1')\,
 \mathbf P_R,
\]

with the two-dimensional `G_0` fibre and its analytic branch transport kept
open.  The current repository has the local matrix formula but does not yet
have an executable all-level NRR component ledger or an independently
derived simultaneous large-c NRR seed.  Consequently there is not yet a
second, closed genus-two R evaluator to compare against the 56 direct
coefficients through level five.  Defining the missing seed as `direct minus
pole sum` at every target coefficient would force machine agreement by
construction and would not be an independent check; this report does not do
that.

The acceptance criterion for the completed evaluator is coefficient-wise
agreement with the direct ledger at a generic nonresonant point, with a
maximum scale-normalized residual of order `1e-14` or below.  The local
level-one certificate above already satisfies that criterion.

## Independent checks

The implementation performs the following checks.

1. With the NS state restricted to its bottom primary or top component, the
   new all-descendant trilinear form agrees with the pre-existing independent
   two-R-leg Ward matrix for every state through combined R level four.
2. The NS `L_-1` coefficient equals the exact scaling derivative
   `h_R1-h_N-h_R2`.
3. The normalized ground contraction is exactly one.
4. The six level-two and six selected level-five coefficients are retained
   as regression anchors.
5. The first fixed-beta Ramond Kac residue agrees algebraically, without a
   pole displacement, on both quadratic branches at `2.35e-16` relative
   error.
6. All equal-form half-integral-total-level coefficients cancel through the
   requested cutoff.

Reproduce with

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=Code \
  python3 Code/ramond_genus2_direct.py --max-total-level 5

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=Code \
  python3 -m unittest Code/test_ramond_genus2_direct.py -v
```

The implementation is in `Code/ramond_genus2_direct.py`; its regression
tests are in `Code/test_ramond_genus2_direct.py`.

## All-NS calibration (in case “R” meant the recursive coefficient)

The existing all-NS theta block was also rerun at

\[
 c=37.25,
 \qquad (h_0,h_1,h_\infty)=(0.73,0.91,1.17),
 \qquad (\xi_0,\xi_1,\xi_\infty)=(1,1,1).
\]

Through total level two, all 35 direct coefficients agree with the NS
Zamolodchikov recursion; the maximum error is `5.55e-16`.  Through total
level five, all 286 coefficients agree; the maximum absolute error is
`4.26e-14`, at twice-level `(0,8,2)`, and the maximum scale-normalized error
is `2.08e-14`.  This is the control example of the requested analytic
direct-versus-recursive matching: both evaluators are complete and no Kac
pole regulator is used.
