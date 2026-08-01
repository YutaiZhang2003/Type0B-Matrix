# Current Genus-Two c=1 Computation

Status: 2026-07-18.  This is the short reviewer-facing map of the code that is
actually used for the genus-two modular pilot.

**Xi scalar-normalization correction (2026-07-21).** The production kernel
now converts the dimensionless plumbing momentum `p=sqrt(alpha') k` to Xi's
per-volume measure `prod_I dk_I/(2 pi)` and uses the matching compact zero
mode `2 pi R_phys`. At genus two,

```text
Z_X^Xi = Z_X^p/(4 pi^2 alpha'),
I_2^Xi/I_2^old = 1/(2 pi sqrt(alpha')).
```

Thus the local Xi density is the historical dimensionless density divided by
`2 pi` at `alpha'=1`; all normalized radius shapes, channel ratios, and
relative errors are unchanged. The release exporter migrates archived values
by an explicit convention identifier.

**c=1 sphere-topology correction (2026-07-21; applied in production v3).**
The multiplier `alpha'/pi` inherited the critical
string sphere metric `K_S2=8*pi/alpha'`. The c=1 timelike zero mode instead
gives

```text
K_tilde_S2 = 2/sqrt(alpha'),
Khat_S2^c1 = 2*pi*sqrt(alpha')*K_tilde_S2 = 4*pi.
```

With `C_T2=1`, separating sewing implies `C_g=(K_S2)^(1-g)`. Therefore

```text
C_g^c1/C_g^crit = (2/alpha')^(g-1).
```

The genus-one vacuum is unchanged, while genus two acquires
`Lambda_top=2/alpha'`. At `alpha'=1`, the final c=1 multiplier is consequently
`2/pi`. The production kernel now applies this final coefficient exactly once.
The archived 6,846-node table retains its original convention identifier and
is migrated exactly by `1/pi` when exported to v3. This notice supersedes the older statements
below that the genus-one anchor certified the absolute genus-two
normalization: that anchor fixes the local critical/scalar/ghost bridge, but
cannot see an Euler-characteristic sphere constant at genus one.

Here “genus one is unchanged” means that there is no multiplicative
topology factor in the torus vacuum integrand. A constant change in the
`g_s`--`mu` dictionary can replace `log(mu)` by `log(mu)+constant`; this only
changes the scheme-dependent additive part of `F_1`, not the universal
coefficient multiplying `log(mu)`.

For the production prescription and present numerical verdict without the
historical pilot record, start with
[`genus2_free_energy_procedure.md`](genus2_free_energy_procedure.md).

## 1. Quantity currently evaluated

At a period matrix

```text
Omega = X + i Y in H_2,
```

the code evaluates the locally normalized coefficient of
`|dOmega_11 dOmega_12 dOmega_22|^2`,

```text
I_2^Xi(Omega;R)
  = 2^24 (2 pi) R Theta_R^(2)(Omega)
      / [det(Y)^13 |chi10_product(Omega)|^2]
      * Z_L^a(Omega) / [Z_X,p^a(Omega)]^25.               (1)
```

Here `chi10_product` is the raw product of the ten even theta constants
squared.  The factor `2^24` converts this raw convention to a nonchiral
Mumford form with unit separating residue.  The label `a` is the theta or
glasses plumbing frame selected for that point.

The Liouville and noncompact-scalar partition functions are evaluated in the
same plumbing frame.  If

```text
Z_X,p^a = W_a Z_X,p^B,          Z_L^a = W_a^25 Z_L^B,
```

then their quotient in (1) is frame independent.  No separately guessed Weyl
factor is used in the production path.

The currently implemented c=1 integration kernel and Monte Carlo
quantity are

```text
K_2^c1(Omega;R) = 2 I_2^Xi(Omega;R)/pi,
F_2(R)/g_s^2 = (1/2) integral_F2(coarse) d^3X d^3Y K_2^c1(Omega;R). (2)
```

Here `alpha'=1`; `1/pi` is the critical-string sphere coefficient after the
string-note six-form conversion. The sphere-normalized c=1 kernel that has now
been pinned down, without matrix-model input, is

```text
K_2^c1(Omega;R) = 2 I_2^Xi(Omega;R)/pi = 2 K_2^note(Omega;R).       (2a)
```

Equation (2a) is the v3 production convention. The historical
`K_2^note=I_2^Xi/pi` convention is accepted only through an explicit migration
path.

## 2. Production pipeline

| Layer | Main code | Role |
|---|---|---|
| Fundamental-domain samples | `genus2_siegel_fundamental_domain.py` | Samples the coarse Gottschling domain from `d^6Omega/det(Y)^3`; uses `Vol(F_2)=pi^3/270`. |
| Plumbing atlas | `genus2_plumbing_atlas.py`, `genus2_hybrid_period_map.py`, `genus2_holomorphic_period_table.py`, `plumbing_algorithms.py` | Searches symplectic markings, takes seeds from the leading formula or a reference table, uses holomorphic-form inversion in the bulk and adaptive Schottky inversion in long cusps, and chooses the usable chart with the smallest `max |q_e|`. |
| Theta Liouville block | `ccy_genus2_block.py` | CCY central-charge recursion in the theta graph. |
| Glasses Liouville block | `ccy_genus2_glasses_block.py` | The corresponding recursion in the separating/glasses graph. |
| Momentum integrals | `liouville_genus2_ccy.py`, `liouville_genus2_glasses.py`, `liouville_momentum_quadrature.py` | Performs the three internal Liouville momentum integrals. |
| Free scalar plumbing | `free_boson_plumbing.py` | Schottky primitive-word Heisenberg product with `n>=1`, multiplied by the two-loop Gaussian `det(Y)^(-1/2)`. |
| Compact scalar | `genus2_c1_string_integrand.py` | Evaluates the genus-two radius-`R` lattice sum and Xi's target zero mode `2 pi R_phys`. |
| Ghost/Mumford form | `genus2_bc_ghost.py`, `genus2_c1_string_integrand.py` | Supplies the critical matter-ghost coefficient and assembles (1). |
| Normalization ledger | `genus2_integrand_normalization.py` | Keeps the cusp-form, CFT-state, stack, BRY topology, and Polyakov factors separate. |
| BRY/Xi convention audit | `audit_bry_xi_convention_map.py` | Proves the unit local Liouville map, records the coupling and complex-measure changes, and isolates the remaining full-partition bridge. |
| Modular integration | `monte_carlo_integrate_genus2_c1.py` | Evaluates every sampled node at low/high truncation and forms the estimator. |
| Batch assembly | `assemble_genus2_c1_mc_pilot.py` | Combines completed CSV batches without reevaluating the CFT. |
| Cutoff refinement | `refine_genus2_c1_mc_nodes.py`, `assemble_genus2_c1_refined_pilot.py` | Reuses certified saved charts, separates block and quadrature steps, and assembles the highest validated order per node. |

Raw sewing parameters are never matched across channels.  Each theta/glasses
comparison first matches the period matrix, including the relevant symplectic
marking, and certifies the finite-`q` inverse-period residual.

### 2.1 The direct finite-`q` map from plumbing parameters to `Omega`

The period map is not obtained by retaining only the leading logarithms of the
plumbing parameters.  For either topology, the bulk representation is

```text
(q1,q2,q3; marked cycles) -> normalized holomorphic one-forms -> Omega(q).
```

The implementation is `solve_theta_collocation` and
`glasses_collocation_period_matrix` in `plumbing_algorithms.py`.  The common
construction, and the topology-dependent identifications, are as follows.

#### Common two-sphere construction

Start with two copies `S_s` of the three-punctured sphere, with punctures at
`z_s=0,1,infinity` and local coordinates

```text
u_0=z,             u_1=z-1,             u_infinity=1/z.
```

On each sphere a candidate differential is expanded through Laurent order
`N` as

```text
omega_J^(s) = f_J^(s)(z_s) dz_s,

f_J^(s)(z) = sum_(n=1)^N a_(s,0,n) z^(-n)
           + sum_(n=1)^N a_(s,1,n) (z-1)^(-n)
           - sum_(n=2)^N a_(s,infinity,n) z^(n-2).
```

The infinity sum starts at `n=2`: its missing `n=1` term would be
`du_infinity/u_infinity=-dz/z`, which is already represented by the simple
poles.  On every sewn collar, local coordinates satisfy `u_L u_R=q_e` and a
one-form must obey

```text
f_L(z_L) = f_R(z_R) (dz_R/dz_L).
```

This equality is imposed at an overdetermined set of equally spaced points on
each seam circle.  The default balanced radius is `r_e=sqrt(|q_e|)`, subject to
`|q_e|<r_e<1`.  If `M_N(q)` is the resulting seam matrix and `P` is the exact
`A`-period matrix of the Laurent basis, the two normalized forms are obtained
from the constrained problem

```text
minimize ||M_N(q) C||_2,        subject to P C = identity_2.
```

The columns are scaled first, the constraint null space is constructed by an
SVD, and the least-squares correction is made only inside that null space.
Thus the `A`-normalization is imposed as a constraint rather than repaired
after the solve.  The marked period matrix is then

```text
Omega_(J I) = integral_(B_I) omega_J.
```

The `B`-period code integrates the Laurent modes by elementary antiderivatives;
the remaining glasses simple-pole path term is evaluated by controlled
high-order quadrature, and all terms are combined with compensated summation.
Together with constrained `A`-normalization and nonlinear finite-`q`
refinement, this removes the cancellation mechanism that contaminated the
early pointwise one-form calculation and produced discrepancies at the
several-percent level.  The leading plumbing formula is now used only to
recognize and seed a chart; it is not the accepted finite-`q` period map.

#### Theta identification

For the theta graph all three tubes join the two different spheres:

```text
0_1 <-> 0_2 with q_0,
1_1 <-> 1_2 with q_1,
infinity_1 <-> infinity_2 with q_infinity.
```

In the global `z` coordinates used by the collocation matrix, the three maps
on the left seam circle are

```text
zero:       z_R = q_0/z_L,
one:        z_R = 1 + q_1/(z_L-1),
infinity:   z_R = 1/(q_infinity z_L).
```

The two `A`-cycles are the positively oriented loops around the zero and one
seams on sphere 1.  The infinity seam is the reference seam for the two
`B`-cycles: on sphere 1 a path goes from the infinity seam to the zero or one
seam, crosses that tube, and returns from the corresponding seam to the
infinity seam on sphere 2.  These paths are evaluated by
`theta_b_periods`.  The two off-diagonal entries can initially differ by an
integral `B`-cycle branch; that integer is removed before their numerical
symmetry error is measured and before they are averaged.

#### Glasses identification

The same two three-punctured spheres and the same one-form ansatz are used, but
the pairing of the punctures is different:

```text
0_1 <-> infinity_1 with q_L,       (left nonseparating handle)
0_2 <-> infinity_2 with q_R,       (right nonseparating handle)
1_1 <-> 1_2        with q_B.       (separating bridge)
```

For a handle, a point on the zero collar is paired on the same sphere by
`z_R=z_L/q_L` or `z_R=z_L/q_R`, with derivative `1/q_L` or `1/q_R`.  Across the
bridge,

```text
z_2 = 1 + q_B/(z_1-1),       dz_2/dz_1 = -q_B/(z_1-1)^2.
```

The two `A`-cycles are the handle circles, one on each sphere.  Each `B`-cycle
runs through its annulus along a continuous logarithmic spiral
`z(t)=z_outer exp(t log q_I)`, with `I=L,R`.  The bridge matching couples the
two normalized forms and thereby produces the off-diagonal entry of `Omega`;
it is not discarded or replaced by a theta identification.  The fast
production helper also reevaluates the seam residual on a distinct, denser
angular grid.

The topology dependence can therefore be summarized without ambiguity:

| item | theta frame | glasses frame |
|---|---|---|
| building blocks | two three-punctured spheres | the same two three-punctured spheres |
| tube pairing | three cross-sphere pairings | two same-sphere handles plus one cross-sphere bridge |
| `A`-cycles | zero and one seam loops | left and right handle loops |
| `B`-cycles | infinity-to-zero/one paths across the graph | logarithmic paths through the two handle annuli |
| public forward routine | `solve_theta_collocation` | `glasses_collocation_period_matrix` |

#### Truncation certificate and use in the inverse atlas

For a fixed `q`, the period matrix is recomputed at two Laurent orders.  The
atlas chooses `(N_low,N_high)` adaptively from `q_max=max_e |q_e|`:

| `q_max` range | theta orders | glasses orders |
|---|---:|---:|
| `q_max <= 0.2` | `20 -> 24` | `20 -> 28` |
| `0.2 < q_max <= 0.3` | `24 -> 32` | `24 -> 32` |
| `q_max > 0.3` | `32 -> 40` | `32 -> 40` |

Theta uses four seam samples per retained order and glasses uses six.  A chart
is accepted only after checking the inverse-period residual, the low/high
basis movement, the seam residual, the period-matrix symmetry, and finiteness.
For `Omega -> q`, the leading plumbing formula supplies a nonlinear seed until
the reference table has been generated.  Afterwards, nearest table rows in the
same topology and marking supply additional seeds.  `_refine_collocation_inverse`
then varies `q` against the normalized-form map, determines the symmetric
integral period branch, and returns the finite-`q` solution.  The handle
variables in the glasses inverse are logarithmic, while its bridge variable is
optimized directly.  A table lookup is only an initial value: every atlas
candidate is freshly refined and certified at the requested `Omega`.

#### Adaptive holomorphic/Schottky region split

`genus2_hybrid_period_map.py` assigns every geometrically valid theta or
glasses chart to a numerical route.  The default regions are

| Region | Test | Preferred computation |
|---|---|---|
| Schottky cusp | `min |q_e| < 10^-12` | Multiprecision Schottky cross-ratio sum |
| Conditioning transition | `min |q_e| < 10^-10` or standard-disk clearance `< 0.02` | Try both adaptive representations |
| Two-method overlap | `max |q_e| <= 0.16` | Compute both and enforce agreement |
| Holomorphic bulk | all remaining valid charts | Normalized-form boundary collocation |

Collocation raises the Laurent basis through at most order 72.  Schottky raises
the word cutoff from the configured minimum through order 9 by default, and
uses two successive word steps to estimate a safety-factored geometric tail.
When both methods pass, their period matrices must agree modulo a symmetric
integral B-period shift within the requested agreement bar (default `10^-6`).
The result with the smaller certified error estimate is used.  If the preferred
method fails, the expensive alternative is attempted; if neither reaches the
bar, the point fails loudly rather than silently accepting an inaccurate map.

This removes the old scalar-q coverage hole.  It does not make an overlapping
sewing-disk configuration valid: such a chart is rejected and the atlas must
choose another marking or pants decomposition.  The current finite symplectic
search still does not constitute a mathematical proof that every moduli point
has been found; it reports `uncovered-at-current-search-settings` when its
search must be enlarged.

The holomorphic-form map remains the overlap calibration reference.  The
reference table stores, for each theta or glasses point,

```text
q, Omega_hol, Laurent order, seam samples, basis movement,
seam residual, symmetry error, and plumbing-geometry margin,
```

and, where evaluated, the Schottky word length, the discrepancy
`||Omega_Sch-Omega_hol||`, the last word-order movement, and the Schottky
symmetry error.  Adjacent reference rows may then be promoted to a
six-real-dimensional validity cell only after boundary and interior probes
pass.  Each cell records ranges in all three `log|q_e|` and all three phases,
the separate boundary/interior validation counts, the SHA-256 identity of the
reference table, the minimum geometry margin, the maximum observed errors, and
a safety factor.

At runtime `genus2_calibrated_schottky.py` can strengthen a Schottky result if
the complete query lies inside such a cell and

```text
safety_factor * max(reference discrepancy, word step, symmetry error)
    <= requested period tolerance.
```

The word step and symmetry bound are checked again at the query.  Outside the
saved cells, the adaptive word-tail certificate remains available in a long
cusp, so a calibration table is an additional guarantee rather than a coverage
gate.  Table or leading data provide inverse seeds; the final forward map is
certified by the same regional policy.

A focused fixed-`q` glasses benchmark presently has seam residual
`1.0e-15`, symmetry error `5.5e-17`, and relative disagreement
`1.9e-10` with the independent word-length-eight Schottky evaluation.  This
checks that the glasses map is an actual implementation, not merely a theta
result relabeled in a second channel.

There is also an internal large-`q` glasses stress case with `q_max=0.3411`.
Against an order-56 target, adaptive certification raises the basis from 32 to
52 and obtains target residual `9.10e-11`, basis movement `2.09e-7`, seam
residual `7.43e-7`, and symmetry error `3.82e-10`, passing the `1e-6`
criterion.  Since the target is generated by the same solver at a still higher
order, this is a convergence stress test rather than an independent period-map
validation.

One geometric limitation remains explicit.  The collocation entry points check
`|q_e|<r_e<1`, but do not yet enforce every pairwise disjointness inequality
among the three excised disks on a sphere.  Therefore a large-`q` evaluation is
not a certified plumbing chart merely because the linear solve converges; the
disk geometry and basis convergence must both be checked.  This matters in the
still-untested `q_max` approximately `0.4--0.6` region.

### 2.2 Fresh `Omega -> q -> Omega` round-trip audit

The ten-point audit in `audit_omega_q_omega_roundtrip.py` starts from fresh
period matrices sampled with the invariant Siegel measure (seed `20260718`).
For each point it searches both theta and glasses markings to symplectic depth
three, retains two candidates per topology, obtains `q` with the inverse just
described, and reevaluates `Omega(q)` eight Laurent orders above the inverse
basis.  It then removes the symmetric integral-period branch, applies the exact
integer inverse of the selected symplectic matrix, and compares in the original
marking:

```text
epsilon_F = ||Omega_returned-Omega_start||_F / ||Omega_start||_F.
```

All ten points passed at depth three.  Eight selected theta charts and two
selected glasses charts; all ten used the normalized-form collocation map, and
all ten validation branches agreed with the branch saved by the inverse.  In
this pre-table run every inverse started from the leading plumbing formula; no
Schottky-derived inverse seed was used.

| point | chart | `q_max` | inverse -> validation order | `epsilon_F` |
|---:|---|---:|---:|---:|
| 0 | theta | 0.043955 | `24 -> 32` | `2.227e-14` |
| 1 | glasses | 0.142064 | `28 -> 36` | `5.406e-14` |
| 2 | theta | 0.038822 | `24 -> 32` | `1.692e-14` |
| 3 | theta | 0.153088 | `24 -> 32` | `1.027e-12` |
| 4 | theta | 0.026733 | `24 -> 32` | `3.249e-14` |
| 5 | theta | 0.218443 | `32 -> 40` | `8.169e-13` |
| 6 | theta | 0.024788 | `24 -> 32` | `1.403e-14` |
| 7 | theta | 0.009813 | `24 -> 32` | `1.098e-11` |
| 8 | theta | 0.077072 | `24 -> 32` | `9.949e-12` |
| 9 | glasses | 0.164182 | `28 -> 36` | `6.117e-13` |

The relative Frobenius error has minimum `1.40e-14`, median `3.33e-13`,
90th percentile `1.01e-11`, and maximum `1.10e-11`.  The largest absolute
matrix-entry error is `4.83e-11`.  The saved reviewer table, full matrices and
parameters, and machine summary are in
`results/genus2_omega_q_omega_roundtrip_10pt/`.

This is strong evidence for the branch handling, symplectic bookkeeping,
finite-`q` optimization, and Laurent-order convergence on the tested sample.
It is not a fully independent second derivation: the validation raises the
order but uses the same normalized-holomorphic-form solver family as the
inverse.  Also, the sample covers only
`0.0098 <= q_max <= 0.2184`.  It includes a useful long-tube point but does not
yet validate the deliberately difficult `q_max` approximately `0.4--0.6`
region; that requires a separate geometrically admissible large-`q` suite.

## 3. Liouville and scalar inputs

At `b=1`, `c_L=25`, and `h(P)=1+P^2`, the theta wrapper computes

```text
Z_L^theta = integral_(P_i>=0) dP1 dP2 dP3 / pi^3
  C(P1,P2,P3)^2
  |q1^h1 q2^h2 q3^h3 F_theta^(N)|^2.                    (3)
```

The glasses wrapper computes

```text
Z_L^glasses = integral_(P_L,P_R,P_B>=0) dP_L dP_R dP_B / pi^3
  C(P_L,P_B,P_L) C(P_R,P_B,P_R)
  |q_L^hL q_R^hR q_B^hB F_glasses^(N)|^2.              (4)
```

The physical path uses `propagator_shift=0`.  Thus the primary sewing factors
are literally `q^h`, as in the plumbing derivation; threshold or Casimir
shifts are not inserted.  The default BRY/Yin structure constants omit a
separate cosmological prefactor.

For the scalar, the code uses

```text
Z_X,p^a = det(Im Omega_a)^(-1/2)
        * product_[primitive gamma] product_[n>=1]
          |1-k_gamma^n|^(-2).                           (5)
```

The primitive-word and oscillator products are truncated numerically, with
separate low/high word cutoffs.  This free-Heisenberg product is not the
`n>=2` large-`c` Virasoro vacuum seed used inside the CCY recursion.

The radius-dependent compact lattice factor is

```text
R Theta_R^(2) = R sum_(m,n in Z^2)
 exp[-pi R^2 (m+conj(Omega)n)^T Y^(-1)(m+Omega n)].      (6)
```

The same theta function is now evaluated automatically in either this direct
form or the exactly Poisson-resummed form

```text
Theta_R^(2) = sqrt(det Y)/R^2 sum_(k,n in Z^2)
 exp[-pi k^T Y k/R^2 - pi R^2 n^T Y n]
 exp[2 pi i k^T X n],  Omega=X+iY.                      (6a)
```

Both forms use the fundamental-domain period matrix.  The implementation
estimates the two truncation costs and selects the cheaper representation.
This removes the large direct momentum cutoff at a nonseparating cusp without
changing frames.  Every production row records the selected representation
and both integer cutoffs; explicit `lattice_nmax` requests retain the old
direct-box behavior for cutoff audits.

## 4. Monte Carlo measure

The sampler draws iid points from

```text
p(Omega)d^6Omega
  = d^6Omega / [Vol(F_2) det(Y)^3].
```

Therefore the implemented estimator is

```text
(Fhat_2/g_s^2) = Vol(F_2)/(2N)
  sum_i det(Y_i)^3 K_2^c1(Omega_i;R).                   (7)
```

The factor `det(Y)^3` in (7) is only importance reweighting.  It is not part
of the ghost correlator.  The factor `1/2` is the generic genus-two stack
weight for the coarse `PSp(4,Z)` domain and is applied exactly once.  Failed
nodes are saved and invalidate a complete-sample estimate; they are not
silently discarded.

## 5. Normalization ledger

The following pieces are implemented and independently checked:

1. Liouville two-point metric `pi delta(P-P')` and completeness `dP/pi`.
2. Scalar target zero mode `dX/(2 pi)`, compact volume `R`, and loop Gaussian
   `det(Y)^(-1/2)` with coefficient one.
3. Raw theta-product convention, `N_Phi=2 pi i`, and the exact `2^24`
   nonchiral conversion to unit separating Mumford residue.
4. Same-frame Weyl cancellation in `Z_L/(Z_X)^25`.
5. The generic moduli-stack weight `1/2`.

The string-note plumbing-unitarity derivation instead fixes the pure
gauge-fixing differential-form phase

```text
N_(h,n) = i^(3h-3+n),       K_S2 = 8 pi/alpha',
N_(2,0) = -i.                                                   (8)
```

Equations (4.105)--(4.106) of the string notes additionally supply a minus sign
and `g_s^2 alpha'/(8 pi)`.  In the displayed ordering,

```text
d^3Omega wedge d^3 bar(Omega) = -8 i d^3X d^3Y,
(-i)(-1)(-8i) g_s^2 alpha'/(8 pi) = g_s^2 alpha'/pi.    (9)
```

Equations (8)--(9) and both sewing recurrences are implemented and checked in
`genus2_integrand_normalization.py`. These establish local identities, but the
previous audit incorrectly treated their combination with the BRY topology
coefficient as an independent absolute sewing check. Consequently:

```text
local CFT sewing normalization:                 certified
formal Polyakov N_(h,n) recurrence:             certified
N_(2,0) -> positive real period measure:        certified
local BRY/Xi critical/scalar/ghost bridge:       certified, Lambda_local=1
c=1 sphere-topology correction:                  certified, 2/alpha'
c=1 correction applied to production v3:        yes, exactly once
archived convention migration:                  explicit and tested
full genus-two integration-kernel normalization: c1 v3 convention active
```

The intrinsic Liouville bridge is now settled: Xi and BRY use the same
normalized primary, two-point metric, `dP/pi`, DOZZ coefficient, and sewing
coordinate, so `Z_L^Xi=Z_L^BRY` in the plumbing construction. The remaining
free-scalar measure is also now explicit:
`Z_X^Xi=Z_X,p/(4 pi^2 alpha')` and
`Z_XR^Xi=(2 pi R_phys) Z_X^Xi Theta_R`. Combining this with the 26-scalar
critical seed gives the net factor `1/(2 pi sqrt(alpha'))`. Normalized
genus-one-anchored separating sewing then fixes the correlated
critical/scalar/ghost bridge to `Lambda_local=1`. Because the torus has zero
Euler characteristic, this genus-one anchor does not test the sphere
topology constant. The independent sphere audit supplies
`Lambda_top=2/alpha'`; at `alpha'=1` it doubles the genus-two kernel while
leaving the genus-one vacuum unchanged.

The BRY and Xi couplings must be distinguished. The positive BRY relation is

```text
C_Sigma2 C_S2 = C_T2^2,
C_S2=2 pi/(g_s_BRY)^2, C_T2=1,
g_s_BRY=2 g_s_Xi,
C_Sigma2=(g_s_BRY)^2/(2 pi)=1/(8 pi^3 mu^2).
```

The old `1/(32 pi^3 mu^2)` value resulted from inserting `g_s_Xi` directly
into BRY's formula. The separate convention audit finds a bare genus-two
measure factor `8` and coupling weight `1/4`, but these must not be reapplied
to the kernel because it already uses Xi's positive real measure and Xi's
coupling. BRY's extrapolated coefficient and Xi's `g_s_Xi^2/pi` coefficient
still bundle different full-amplitude conventions. With
`mu=1/(4 pi g_s_Xi)`, the matrix-model comparison still defines the algebraic
unit-Mumford stack target `16 pi^3 f_2(R)` and sampled string-note target
`16 pi^2 f_2(R)`, but agreement with those targets is the normalization test,
not an already-certified consequence.

## 6. Current numerical pilot

The original 32-point `R=1` data are in
`results/genus2_c1_moduli_mc/pilot_R1_N32/`.  Their order-`4/4` local integral
was

```text
J_2^local(1)|_4/4 = 2.4098783961e-4 +/- 3.8564457467e-5.
```

Its `1.52%` aggregate order-`2/3` to order-`4/4` change concealed node-level
movements of approximately `+13.2%` and `-12.3%`.  The atlas chose 31 theta
charts and one glasses chart; five selected points have `max |q_e|>0.16`, and
the largest is about `0.326`.

The cutoff refinement is implemented by `refine_genus2_c1_mc_nodes.py` and
assembled by `assemble_genus2_c1_refined_pilot.py`.  The sample has now been
extended to 64 independent period matrices.  Every node is evaluated at least
at block/quadrature order `6/6`; 20 nodes are at `8/8`, one is at `10/10`,
and one slowly converging theta node is at `12/12`.  The current result,
stored in `results/genus2_c1_moduli_mc/pilot_R1_N64_refined/`, is

```text
J_2^local(1) = 2.5522176944e-4 +/- 3.1122056872e-5  (MC error only).
```

The uniform order-`6/6` estimate differs from order `4/4` by `+0.0755%`.
Adaptive promotion then changes it by `-0.0665%`.  All 64 latest diagonal
steps are below `0.4861%`; the largest isolated block and quadrature steps are
`0.4434%` and `0.4364%`.  The sum of absolute observed last-step changes is
`0.1306%` of the final estimate, while their RMS aggregate is `0.0232%`.
These are convergence diagnostics, not rigorous residual-error bounds.  The
maximum saved scalar low/high movement is only `0.0120%`.
The final assembly independently raises the scalar primitive-word cutoff from
6 through 8 to 10.  Its aggregate word-10 correction is `+0.000019%`; the
largest node-level word-8 to word-10 density movement is `0.000350%`.

The CFT cutoff uncertainty is therefore now much smaller than the roughly
`12.2%` Monte Carlo standard error.  The quoted integral remains locally
normalized because the separate worldsheet normalization issue in section 5
is unchanged.  The two independently refined 32-point halves give
`2.41575e-4 +/- 3.87640e-5` and `2.68869e-4 +/- 4.92154e-5`; they differ by
only `0.436` combined standard deviations.

These saved integral values predate the direct holomorphic-form bulk period
map.  They are retained as convergence history, not as the final production
estimate.  New Monte Carlo rows use the hybrid atlas inversion and post-atlas
certificate: collocation in the bulk, adaptive multiprecision Schottky words
only when all three plumbing parameters are small, rescaled multiprecision
holomorphic forms in mixed cusps, and a mandatory cross-method comparison in
the overlap.  An
explicit Schottky validity-envelope CSV supplies an optional stronger local
bound.  The sample must be regenerated under that policy.

### 6.1 Direct plumbing sample and channel comparison

A new direct sample starts from plumbing coordinates instead of modular-domain
points.  `sample_genus2_plumbing_moduli.py` maps 64 theta and 64 glasses
Sobol points with `0.01 <= |q_e| <= 0.20` to marked period matrices by solving
for normalized holomorphic one-forms.  All 128 maps succeed.  The maximum
raised-basis movement is `1.622e-8`, the maximum seam residual is `8.571e-8`,
and the maximum period-matrix symmetry error is `1.304e-9`.  A finite
difference check of the complex log-q Jacobian obeys the Cauchy--Riemann
relations to `1.592e-9` relatively.

A separate 32-point overlap set starts near the glasses point `q_e=0.15` and
inverts the theta map with one fixed symplectic marking.  All 32 inverse solves
pass a raised-basis direct-period validation; the maximum target residual is
`4.109e-8` and the largest sewing modulus in either frame is `0.179586`.

The actual channel comparison is not the raw Liouville ratio.  At each matched
period matrix it evaluates the full noncompact scalar in the same frame and
tests

```text
Q_L^theta / Q_L^glasses
  = [Z_L^theta/(Z_X^theta)^25]
    / [Z_L^glasses/(Z_X^glasses)^25].                   (9)
```

For three stratified overlap points the ratio converges as follows:

| Point | max both-frame `|q|` | order `4/6` | order `6/8` | order `8/10` |
|---|---:|---:|---:|---:|
| `o0006` | 0.15347 | 1.01503517 | 1.00312640 | 1.00090309 |
| `o0025` | 0.16063 | 1.01479855 | 1.00368674 | 1.00159741 |
| `o0027` | 0.17959 | 1.01210398 | 1.00288007 | 1.00103018 |

At the worst selected point, `o0025`, block orders `8,10,12,14` at fixed
quadrature order 10 give respectively

```text
1.0015974117, 1.0012794930, 1.0012165812, 1.0012035560.
```

The high-order momentum rule, scalar primitive product, and direct period map
are all stable far below the remaining `1.2e-3` difference.  Independent
checks also validate the one-edge recursions, glasses separating limit, CCY
large-c seed, Liouville structure constants and `dP/pi` measure.  Consequently
the data show reproducible near-agreement at several period matrices, but do
not yet establish exact channel equality.  The approximately `0.12%` floor is
an unresolved genus-two consistency condition, possibly in the combined local
coordinate/frame convention.  No fitted correction is applied.

The complete reviewer bundle is
`results/genus2_plumbing_moduli_samples/direct_bulk_N128_overlap_N32/`.
Although the sample records the local log-q Jacobian, its union of plumbing
charts is not yet an integration proposal: chart multiplicities and a
partition of unity remain to be fixed.

### 6.2 Broad locality-contrast stress test

The first overlap sample keeps both charts near `qmax=0.15--0.18`.  A second
deterministic sample deliberately searches for common period matrices where
one plumbing chart is efficient and the other is not.  Of 32 broad proposals,
17 pass the direct period-map and raised-basis inverse tests.  Eight are used
for CFT comparisons, split evenly between theta-efficient and
glasses-efficient points.  The largest chart contrasts are

```text
(qmax_glasses,qmax_theta) = (0.25242,0.05303),
(qmax_glasses,qmax_theta) = (0.11566,0.45941).
```

All eight theta/glasses ratios are within `0.93%` of unity at block/quadrature
order `6/8` and within `0.60%` at `8/8`.  After independent promotions through
block order 12 and quadrature order 12 where needed, the best available
absolute differences range from `0.193%` to `0.834%`, with median `0.407%`.
This order ladder is a convergence diagnostic and does not define the sample.
The geometry sample is generated independently of all CFT truncation orders.

The fixed production benchmark currently contains three points, all evaluated
at block/quadrature order `12/12` in both channels:

```text
o0024: (qmax_g,qmax_t)=(0.25242,0.05303), ratio=0.9969062074
o0031: (qmax_g,qmax_t)=(0.11566,0.45941), ratio=1.0020631344
o0014: (qmax_g,qmax_t)=(0.11931,0.45693), ratio=1.0083382913
```

No point evaluated at a different order is assigned a production value.

Two high-order sequences show why the raw best ratio is not itself an error
estimate:

```text
o0024, theta efficient, (0.25242,0.05303):
  8/10  0.99912055
 10/10  0.99736242
 12/10  0.99690735

o0014, glasses efficient, (0.11931,0.45693):
  8/12  1.00908957
 10/12  1.00849808
 12/12  1.00833829
```

The final block movements are `4.55e-4` and `1.60e-4`, smaller than the
remaining `3.09e-3` and `8.34e-3` channel differences.  Raising the free-scalar
primitive-word cutoff from 10 to 12 changes the logarithm of the final ratio by
at most `3.49e-7` after the central-charge power 25 is included.  Period-map
errors are below `7e-8`.  Raising the large-c Virasoro seed from word/mode
cutoff `(6,24)` to `(8,32)` changes its nonchiral frame ratio by at most
`8.63e-13`.  The discrepancy is therefore not explained by the period map,
scalar frame factor, or recursion seed.

The refined block steps can also be assigned to individual channels.  The
smaller-`q` channel changes by at most `4.17e-6` in `log Z_L`, while the
larger-`q` channel changes by as much as `1.76e-3` and accounts for essentially
the entire movement of the locality ratio.  This is direct numerical support
for selecting the smaller-`q` chart in the production atlas.

The current conservative conclusion is an empirical approximately one-percent
locality envelope in this difficult contrast region, not uniform per-mille
accuracy.  For modular integration, choose the chart with smaller sewing
parameters and promote block and quadrature orders independently at every
node; `qmax` alone is not an accuracy certificate.

The complete bundle and table are in
`results/genus2_plumbing_moduli_samples/contrast_overlap_N32/`.

### 6.3 Pointwise CFT stress audit

A separate 12-node audit now probes pointwise CFT accuracy without changing
the overall normalization or performing a moduli integral.  The set contains
all eight hard nodes from the 64-point period-map preflight and the four
highest-weight controls.  Block and momentum-quadrature orders are raised
independently; a small diagonal movement is not accepted if either component
still moves by more than `5e-4`.

All 12 nodes pass this empirical criterion.  The final cutoff pairs are not a
function of `q_max` alone:

| Class | Observed requirement |
|---|---|
| Four hardest initial nodes | two pass at `(12,12)`, one at `(12,14)`, one at `(14,12)` |
| Other four hard-chart nodes | three pass at `(8,8)`, one at `(10,10)` |
| Four high-weight controls near `q_max=0.15` | two pass at `(8,8)`, two at `(10,10)` |

At the two axis-promoted nodes, the final observed movements are `2.13e-4`
for theta quadrature and `4.25e-4` for the glasses block.  Across the stress
set, the mixed block/quadrature term is tiny, at most a few times `1e-5` in
the coarse steps and below `1e-8` in the final high-order steps.  The maximum
saved scalar cutoff movement is `1.32e-4`.

Two further checks isolate approximations outside the recursion order:

1. Raising Upsilon precision from 28 to 40 digits changes full theta and
   glasses bulk integrals by zero at displayed double precision.
2. Raising the large-c vacuum seed from word/mode `(6,20)` to `(8,30)` changes
   the two formerly problematic seeds by `1.1e-11` and zero.  The seed now
   uses the stabilized projective Schottky multiplier backend and is cached,
   because it is independent of Liouville momentum.

The primary-Gaussian momentum result at the extreme glasses point was also
compared with the independent split finite-interval rule.  The split sequence
at `8+4`, `10+5`, and `12+6` nodes per edge differs from the primary result by
`+1.38e-2`, `-1.07e-3`, and `-2.05e-4`, respectively.  Thus the independent
high-order value agrees inside the target tolerance, while also showing why a
low-order finite-interval rule is unsafe for a tiny sewing parameter.

The relevant drivers and saved outputs are:

- `prepare_genus2_cft_accuracy_stress.py`
- `stress_genus2_cft_convergence.py`
- `refine_genus2_cft_axis.py`
- `compare_genus2_liouville_quadratures.py`
- `audit_genus2_cft_dps.py`
- `results/genus2_c1_moduli_mc/cft_accuracy_stress_M64/`

The production implication is that a fixed `q<=0.16` envelope is useful for
chart selection but is not an accuracy certificate.  Each sampled node must
carry a measured CFT cutoff certificate, with block and quadrature promotion
available independently.

### 6.4 Fixed-order direct-period RQMC radius test

The current modular-integration staging run separates the three error sources
that were mixed in the older pilot. Geometry is sampled by eight independent
Owen-scrambled Sobol nets. The completed `M=64` data predate the adaptive hybrid
policy: 37 nodes were routed to a fixed-cutoff Schottky computation merely
because `min |q_e|<1e-12`.  They are useful convergence history but are not a
current period-map certificate and must be regenerated with the word-tail and
overlap checks. CFT
production is uniform at block/quadrature order `8/8`, vacuum-seed cutoff
`(8,32)`, and scalar primitive-word length 12.  No recursion order is selected
from a node's sampling data.

All 98 in-domain nodes of the first `M=16` design are present.  The raw local
self-dual estimate is

```text
J_2^local(1) = 7.5601469e-4 +/- 4.1687642e-4.
```

One long-tube node dominates this small design.  This is a sampling-variance
problem, not evidence for an IR subtraction.  The proposal covers the complete
infinite cusp, every certified node retains its original weight, and the
physical nonseparating-shell observable is

```text
[det(Im Omega)^3 I_2]/T^2 per d log T,  T=-log|q_degenerate|.
```

It decreases at the extreme node.  The code's temporary removal of
`prod_e |q_e|^2` is only an exact log-domain rescaling of the Liouville
integral; that factor is restored before the matter--ghost density is formed.

The compact-boson lattice is then reweighted on the same period matrices, so
all Liouville, ghost, scalar, and Monte Carlo fluctuations remain paired across
radii.  Against

```text
f_2(R) = (7 R^2 + 10 + 7/R^2)/(5760 R),
```

the normalization-independent endpoint comparison is

```text
R=2: worldsheet shape = 0.8621727,
     matrix-model shape = 0.8281250,
     worldsheet/matrix-model = 1.041114 +/- 0.081082.
```

The full 17-radius curve is within `4.36%` of the matrix-model shape and is
statistically compatible with it.  Exact nodewise and integrated T-duality
hold to `4.5e-16`.  The nested `M=32` run reuses all `M=16` nodes and evaluates
only the new Sobol cells; it is the relevant next convergence test because it
halves the old tail-cell weight without changing or subtracting the integrand.

The completed `M=32` assembly contains 193 nodes and gives

```text
J_2^local(1) = 5.3787290e-4 +/- 2.0519607e-4,
R=2 worldsheet/matrix-model shape = 1.077521 +/- 0.0700.
```

The largest node fraction falls from `56.8%` to `39.9%` and the contribution
effective sample size doubles from `3.05` to `6.10`.

The completed same-seed `M=64` extension contains 383 nodes and, after the
exact string-note multiplier `1/pi`, gives

```text
F_2(1)/g_s^2 = 1.6876429e-4 +/- 3.5949293e-5,
matrix-model target = pi^2/15 = 0.6579736267,
estimate/target = 2.56491e-4,
R=2 worldsheet shape = 0.9118022,
R=2 worldsheet/matrix-model shape = 1.101044 +/- 0.03250.
```

The corresponding unit-Mumford stack integral is exactly `pi` times the first
line, namely the previously reported `5.3018864e-4 +/- 1.1293803e-4`.

The largest node fraction is `20.25%` and the contribution effective sample
size is `14.74`. Thus the tail resolution improves monotonically, but the
radius curve is not yet in numerical agreement with the matrix-model shape.
The split by period algorithm is diagnostic: 37 deep-cusp nodes carry
`55.21%` of the self-dual estimate and give endpoint ratio `0.9528`, while the
346 regular collocation nodes drive the upward shift. The mismatch therefore
does not originate from using the converged cusp series in place of an
ill-conditioned boundary collocation solve.

The largest selected sewing modulus is `0.3790`. Promoting that extreme-cusp
node from block/quadrature order `8/8` to `10/10` changes its integrand by
`-0.1801%`; promoting the most influential regular node changes it by only
`-0.00159%`. These pointwise checks do not constitute a uniform CFT theorem,
but they rule out a ten-percent local recursion cutoff at the nodes most able
to move the estimate. The next decisive check is the nested `M=128` or an
explicit tail-stratified integration. Two reused period certificates retain
their older `5e-6` tolerance and all others use `1e-6`; this is recorded rather
than hidden, but is far below the present integration uncertainty.

### 6.5 Tail-stratified all-node CFT promotion

The completed `t3`-stratified design has 576 proposals, 438 accepted
fundamental-domain nodes, and eight independent Owen-scrambled replicates. Its
geometry, period certificates, proposal weights, domain indicators, and one
generic stack factor are held fixed throughout the CFT-order audit.

All 438 nodes were first evaluated at matched block/momentum order `10/10`,
starting from the uniform `8/8` baseline. The eight nodes with the largest
absolute possible effect on the complete integral were then promoted to
`12/12`. The resulting sequence is

```text
8/8 baseline:       2.53387342801e-5 +/- 2.24375e-6,
uniform 10/10:      2.53407132422e-5 +/- 2.24812e-6,
adaptive 12/12:     2.53474483208e-5 +/- 2.25199e-6.
```

The signed changes are `+0.00781%` and `+0.02658%`. The corresponding sums of
absolute weighted node movements are `0.1343%` and `0.0363%`; thus the decrease
with order is visible before cancellations. The final documented residual
envelope is `0.1623%`, well below the `8.9%` scramble error. No CFT evaluation
failed, and an independent column audit confirms that only Liouville-dependent
values and cutoff metadata changed.

This establishes stability of the assembled integrand at the present sampling
accuracy, not uniform pointwise recursion convergence. The node with
`q_max=0.39596` changes by `-9.75%` and then `+5.79%`, although its conservative
effect on the full integral is only `0.0761%`. The axis audit identifies this
movement as block-order dominated; other important nodes are
quadrature-dominated. A cluster run should promote the remaining oscillatory
nodes to `14/14` or higher and evaluate the larger nested tail-stratified
design.

The CFT-order stability also sharpens the normalization conclusion. The
former near-`2^12` difference from the matrix-model absolute value was quoted
before the Xi scalar correction.  In Xi's convention the corresponding
multiplier is larger by `2 pi`, namely `2 pi * 2^12`; this numerical
coincidence is not the derivation of a convention factor. The analytic sphere
audit independently derives the factor `2/alpha'`; it is recorded without
using this numerical comparison and has not yet been propagated through the
saved integrations.

The resumable driver and result ledger are

- `promote_genus2_cft_rqmc.py`
- `results/genus2_c1_moduli_mc/rqmc_t3_stratified_R8_K8_M8/cft_adaptive_promotion/report.md`
- `results/genus2_c1_moduli_mc/rqmc_t3_stratified_R8_K8_M8/cft_adaptive_promotion/convergence_manifest.json`

## 7. Checks and recommended review order

Run the focused checks:

```bash
python3 plumbing/ccy_genus2_block_checks.py
python3 plumbing/ccy_genus2_glasses_block_checks.py
python3 plumbing/liouville_momentum_quadrature_checks.py
python3 plumbing/free_boson_pair_of_pants_checks.py
python3 plumbing/free_boson_plumbing_checks.py
python3 plumbing/genus2_bc_ghost_checks.py
python3 plumbing/genus2_c1_string_integrand_checks.py
python3 plumbing/genus2_integrand_normalization_checks.py
python3 plumbing/genus2_integrand_factorization_audit.py
python3 plumbing/genus2_siegel_fundamental_domain_checks.py
python3 plumbing/genus2_plumbing_atlas_checks.py
python3 plumbing/audit_q_to_omega_accuracy_checks.py
python3 plumbing/audit_omega_q_omega_roundtrip_checks.py
python3 plumbing/sample_genus2_plumbing_moduli_checks.py
python3 plumbing/summarize_plumbing_frame_contrast_checks.py
python3 plumbing/monte_carlo_integrate_genus2_c1_checks.py
python3 plumbing/refine_genus2_c1_mc_nodes_checks.py
python3 plumbing/assemble_genus2_c1_refined_pilot_checks.py
python3 plumbing/prepare_genus2_cft_accuracy_stress_checks.py
python3 plumbing/stress_genus2_cft_convergence_checks.py
python3 plumbing/refine_genus2_cft_axis_checks.py
python3 plumbing/promote_genus2_cft_rqmc_checks.py
python3 plumbing/assemble_genus2_c1_rqmc_checks.py
```

For code review, read in this order:

1. `genus2_current_computation_summary.md`
2. `genus2_integrand_normalization.py`
3. `genus2_c1_string_integrand.py`
4. `monte_carlo_integrate_genus2_c1.py`
5. `refine_genus2_c1_mc_nodes.py`
6. `assemble_genus2_c1_refined_pilot.py`
7. `genus2_siegel_fundamental_domain.py`
8. `genus2_plumbing_atlas.py` and `plumbing_algorithms.py`
9. `audit_q_to_omega_accuracy.py` and `audit_omega_q_omega_roundtrip.py`
10. `liouville_genus2_ccy.py` and `liouville_genus2_glasses.py`
11. `ccy_genus2_block.py` and `ccy_genus2_glasses_block.py`
12. `free_boson_plumbing.py`
13. `genus2_bc_ghost.py`
14. `promote_genus2_cft_rqmc.py`

The gauge-fixing convention quoted in (8) comes from
`/Users/yutaizhang/Desktop/string notes-3.pdf`, especially equations
(4.12), (4.58)-(4.71), and (4.97).  The genus-two period-matrix measure and
critical bosonic formula are discussed around equations (4.103)-(4.109).
