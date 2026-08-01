# Genus-Two Free-Energy Procedure

Status: 2026-07-21. This note records the production prescription currently
used for the genus-two vacuum amplitude of the compact `c=1` string. It is
intended as the shortest reviewer-facing entry point. The longer derivation
and numerical history are in `genus2_current_computation_summary.md` and
`liouville_genus2_plumbing_recipe.tex`.

> **Normalization update (2026-07-21).** Archived production integrations
> used the critical-string sphere coefficient `alpha'/pi`. The
> c=1 sphere audit fixes an additional topology factor `2/alpha'`, so the
> final c=1 coefficient is `2/pi` (a factor two at `alpha'=1`). In general the
> vacuum correction is `(2/alpha')^(g-1)`, hence it is exactly one at genus
> one. A constant shift in the `g_s`--`mu` dictionary may alter only the
> scheme-dependent additive part of the genus-one `log(mu)` term, not its
> universal coefficient. The v3 production kernel now uses `2/pi`; archived
> rows are migrated by an explicit source-convention identifier.

## 1. Conventions and target

We use

```text
alpha' = 1,                 X ~ X + 2 pi R,
c_L = 25, b = 1,           mu = 1/(4 pi g_s).
```

The matrix-model genus-two coefficient is

```text
f_2(R) = (7 R^2 + 10 + 7/R^2)/(5760 R).
```

In the real period-matrix and string-note convention implemented below, the
quantity compared with it is

```text
F_2(R)/g_s^2 = 16 pi^2 f_2(R).                           (1)
```

At the self-dual radius this gives

```text
F_2(1)/g_s^2 = pi^2/15 = 0.6579736267392905.             (2)
```

## 2. Pointwise CFT integrand

Write `Omega = X + iY`. At a fixed period matrix the code evaluates

```text
I_2^Mum(Omega;R)
 = 2^24 (2 pi) R Theta_R^(2)(Omega)
   / [det(Y)^13 |chi10_product(Omega)|^2]
   * Z_L^a(Omega)/[Z_X,p^a(Omega)]^25.                  (3)
```

Here `chi10_product` is the raw product of the ten even theta constants
squared. The exact factor `2^24` converts this convention to the nonchiral
Mumford form with unit separating residue.

The compact scalar lattice is

```text
R Theta_R^(2)
 = R sum_(m,n in Z^2)
   exp[-pi R^2 (m+bar(Omega)n)^T Y^(-1)(m+Omega n)].     (4)
```

The label `a` denotes the selected theta or glasses plumbing frame. Both
`Z_L^a` and `Z_X,p^a` are evaluated in that same frame. If the frame change to
the canonical metric is `W_a`, then

```text
Z_X,p^a = W_a Z_X,p^B,          Z_L^a = W_a^25 Z_L^B,
```

so the quotient in (3) is frame independent. The production code does not
insert a separately guessed Weyl factor.

The noncompact scalar plumbing partition function produced by the sewing
code includes both pieces

```text
Z_X,p^a = det(Im Omega_a)^(-1/2)
          * product_[primitive gamma] product_[n>=1]
            |1-k_gamma^n|^(-2).                         (5)
```

Here the subscript `p` means that the loop zero modes are integrated using
the dimensionless variables `p=sqrt(alpha') k` with measure `d^2p`. Xi's
free-field convention instead uses the physical measure `d^2k/(2 pi)^2`.
Therefore the exact dictionary is

```text
Z_X,Xi^a = Z_X,p^a/(4 pi^2 alpha'),
Z_XR,Xi^a(R) = (2 pi R_phys) Z_X,Xi^a Theta_R^(2),
R_phys = sqrt(alpha') R.                                (5a)
```

The critical 26-boson Gaussian must use the same Xi convention. Substituting
(5a) in the complete critical-times-compact product gives the correlated
cancellation

```text
(4 pi^2 alpha')^(-26) (4 pi^2 alpha')^(25)
  (2 pi sqrt(alpha')) = 1/(2 pi sqrt(alpha')).          (5b)
```

Thus the corrected pointwise density is the historically stored density
divided by `2 pi` when `alpha'=1`; it is not multiplied by an isolated
25-scalar power. The scalar zero-mode momentum integral was present, but its
measure convention is now matched explicitly to Xi's. The product in (5) is
the full Heisenberg oscillator product; it is distinct from the
`n>=2` large-central-charge Virasoro vacuum seed used inside the CCY
recursion.

For Liouville theory, the theta-frame expression is

```text
Z_L^theta
 = integral_(P_i>=0) dP1 dP2 dP3/pi^3
   C(P1,P2,P3)^2
   |q1^h1 q2^h2 q3^h3 F_theta|^2,                      (6)
```

and the glasses frame uses the two pair-of-pants coefficients

```text
C(P_L,P_B,P_L) C(P_R,P_B,P_R).                         (7)
```

Here `h(P)=1+P^2`. The physical sewing factors are literally `q^h`; no
threshold or Casimir shift is applied. The descendant blocks use the CCY
central-charge recursion. The baseline fixes block and primary-Gaussian
momentum orders to `8/8` at every node. Its truncation audit keeps the same
sample and evaluates all nodes at `10/10`, then promotes only the nodes with
the largest contribution-weighted residual to `12/12`.

## 3. Absolute integration normalization

The string-note differential-form convention gives

```text
N_(h,n) = i^(3h-3+n),       N_(2,0) = -i.
```

Combining its additional sign and `g_s^2 alpha'/(8 pi)` with

```text
d^3Omega wedge d^3bar(Omega) = -8i d^3X d^3Y
```

gives the critical-string positive-real coefficient

```text
(-i)(-1)(-8i) g_s^2 alpha'/(8 pi)
  = g_s^2 alpha'/pi.                                    (8)
```

For `alpha'=1`, the historical pre-correction kernel was

```text
K_2^note(Omega;R) = I_2^Mum(Omega;R)/pi.                (9)
```

The c=1 timelike sphere constant is

```text
K_tilde_S2 = 2/sqrt(alpha'),
Khat_S2^c1 = 2*pi*sqrt(alpha') K_tilde_S2 = 4*pi.
```

With `C_T2=1`, sewing gives `C_g=(K_S2)^(1-g)`, so

```text
C_g^c1/C_g^crit = (2/alpha')^(g-1),
K_2^c1(Omega;R) = (2/alpha') K_2^note(Omega;R)
                 = 2 I_2^Mum(Omega;R)/pi.              (9a)
```

The first equality also shows explicitly why the genus-one vacuum is
unaffected. Equation (9a) is the v3 executable production integrand.

The generic genus-two stack weight is kept separate and applied exactly
once:

```text
F_2(R)/g_s^2
 = (1/2) integral_(F_2^coarse) d^3X d^3Y
   K_2^c1(Omega;R).                                    (10)
```

No overall factor is fitted to the matrix-model answer. The factor in (9a)
comes only from the sphere normalization and topology sewing.

## 4. Geometry and plumbing atlas

The integration nodes are period matrices in a coarse Gottschling
fundamental domain. The proposal density is the invariant measure

```text
p(Omega) d^6Omega
 = d^6Omega/[Vol(F_2) det(Y)^3],
Vol(F_2) = pi^3/270.                                   (11)
```

For each accepted `Omega`, the atlas enumerates relevant symplectic markings,
inverts both theta and glasses plumbing charts where possible, and chooses a
certified chart with small sewing parameters. Raw sewing parameters are never
identified between channels.

In the bulk, the map between plumbing coordinates and `Omega` is obtained
from normalized holomorphic one-forms by collocation. Schottky data provide
initial guesses. In the deep cusp, where boundary collocation becomes poorly
conditioned, the converged Schottky/Poincare representation of the same
holomorphic forms is used. Each selected chart records a fixed-`q` period
residual and a raised-basis stability check.

Matched theta/glasses locality tests first match `Omega` (including its
symplectic marking) and then compare

```text
[Z_L^theta/(Z_X,p^theta)^25]
/
[Z_L^glasses/(Z_X,p^glasses)^25].                      (12)
```

These tests support the atlas selection rule at the sub-percent to percent
level through difficult points with one chart as large as `q_max~0.46`; they
do not justify comparing raw Liouville blocks or raw plumbing coordinates.

## 5. RQMC estimator

The completed baseline uses eight independent Owen-scrambled Sobol
replicates, with 64 proposals per replicate before the exact
fundamental-domain indicator. There are 383 accepted nodes. No failed or
inconvenient node is dropped, and there is no cusp cutoff or infrared
subtraction.

From (10)-(11), the estimator is

```text
Fhat_2(R)/g_s^2
 = Vol(F_2)/(2N)
   sum_i 1_F(Omega_i) det(Y_i)^3 K_2^c1(Omega_i;R).     (13)
```

The `det(Y)^3` factor is importance reweighting, not part of the ghost
correlator. Uncertainty is estimated from the eight independent scramble
means. Radius reweighting is paired node by node: only the compact lattice in
(4) changes, so all Liouville, ghost, scalar, and sampling fluctuations remain
common across radii.

The baseline CFT policy is

```text
Liouville block / momentum order:       8 / 8
large-c vacuum seed word / mode cutoff: 8 / 32
scalar primitive-word cutoff:           12
```

The higher-order audit changes only the Liouville CFT value and its derived
integrand columns. It verifies every saved evaluation against the exact six
period-matrix coordinates before assembly.

### Tail-stratified replacement

The baseline badly under-resolves one proposal coordinate. With
`t3=-log(1-u3)/2`, divide `u3` into the exact cells

```text
L_l  = [1-2^-l, 1-2^-(l+1)),  l=0,...,7,
L8+  = [1-2^-8, 1).
```

The last cell maps to the complete semi-infinite region `t3>=4 log(2)`, so
this is a stratification rather than a cusp cutoff. For stratum probability
`P_s`, base importance weight `w`, and `n_s` Sobol points, each independent
replicate is

```text
Ihat_r = (1/2) sum_s (P_s/n_s)
                    sum_j 1_F(Omega_sj) w_sj G(Omega_sj). (13a)
```

The emitted effective weight `M P_s w/n_s`, with `M=sum_s n_s`, lets the
existing strict assembler evaluate (13a) without a second estimator path.
Independent scrambles are retained separately for the error estimate.

The nested `n_s=8` pilot has 72 proposals per replicate and 438 in-domain
CFT nodes; 49 of those nodes lie in `L8+`. Its measure-only volume check is
`0.11278365 +/- 0.00298258`, consistent with the exact `0.11483806`. The
bitwise-nested `n_s=32` extension has 288 proposals per replicate, 1723
in-domain nodes, and 191 nodes in `L8+`; its volume check is
`0.11456383 +/- 0.00162844`. Every pilot coordinate is unchanged in the
extension, so completed CFT data can be reused.

This allocation directly addresses the baseline pathology: only three
baseline nodes reached `L8+`, yet they supplied `37.51%` of the self-dual
estimate. The baseline contribution effective sample size was only `14.74`,
and its largest node supplied `20.25%`. The analogous `t1` level-eight tail
supplied only `0.64%`; at level six and above the `t3` and `t1` contributions
were `41.38%` and `5.42%`, respectively. The observed concentration therefore
supports stratifying `t3` first instead of immediately paying for a
two-coordinate tail grid.

The `n_s=8` design is now a completed fixed-order physical pilot.  All 438
in-domain nodes were evaluated at Liouville block/momentum order `8/8`; no
failed node was dropped or assigned zero.  A fresh preflight on the hardest
node in every nonempty tier and scramble passes `24/24` cases, with maximum
final period residual `1.97e-7` and maximum certified `q_max=0.3960`.  The
separate audit of the single deepest realized `t3` node in every scramble
passes `8/8`, with maximum residual `1.08e-7` and maximum `q_max=0.4106`.
Their union contains 29 distinct stress nodes, all of which pass.  In the
complete sample, including nodes outside that stress selection, the maximum
certified `q_max` is `0.52013` and the maximum final period residual is
`9.59e-7`.

The earlier level-15/16 failures with reported leading `q_max=0.989` and
`0.973` were not genuinely large-`q` surfaces. In the canonical theta chart
one sewing parameter has `log|q|=-3606.7` or `-1447.3`, below binary64's
exponentiated range, while the true chart maxima are only `0.00996` and
`0.17225`. The atlas now carries `log(q)`, evaluates the finite-word map in a
controlled `q -> 0` limit, and restores the exact tropical period. The same
logarithmic coordinate is used for the Liouville Gaussian width and primary
propagator; `log|chi10|` is assembled from scaled theta-constant sums. Both
nodes complete an end-to-end block-order-zero, quadrature-order-two CFT smoke
test. This is a representation repair, not an infrared subtraction.

All 438 nodes have now been promoted from `8/8` to `10/10` on the unchanged
design. The eight nodes with the largest possible effect on the complete
integral were subsequently evaluated at `12/12`. The assembled sequence is

```text
8/8:                 0.000025338734280,
uniform 10/10:       0.000025340713242,   change +0.00781%,
adaptive 12/12:      0.000025347448321,   change +0.02658% from 10/10.
```

The sums of absolute weighted node movements are `0.1343%` and `0.0363%` for
the two steps, so the small signed shifts are not the whole stability claim.
The final contribution-weighted residual envelope is `0.1623%`, compared with
an `8.9%` scramble standard error. This rules out CFT truncation as the source
of an order-4096 normalization discrepancy at the accuracy of this pilot.

Pointwise convergence is weaker. The node with `q_max=0.39596` moves by
`-9.75%` from `8/8` to `10/10` and `+5.79%` from `10/10` to `12/12`; it is not
declared locally converged. Its conservative effect on the full estimate is
only `0.0761%`. A cluster run should promote this and the other oscillatory
nodes to `14/14` or higher while extending the nested tail-stratified sample.

`allocate_genus2_tail_strata.py` estimates the conditional variance in each
cell, restores known out-of-domain zeros, and recommends the dyadic powers
`[5,4,6,4,3,6,3,4,6]`.  The associated counts
`[32,16,64,16,8,64,8,16,64]` use 288 proposals per replicate and predict a
standard-error ratio `0.3925` relative to the pilot.  The nested design has
1,722 in-domain nodes and reuses all 438 completed values.  This is an
allocation heuristic; the final uncertainty must still come from complete
independent replicate estimates.

## 6. Current result and honest conclusion

The complete tail-stratified fixed-`8/8` pilot gives

```text
F_2(1)/g_s^2 = 0.0000253387343 +/- 0.0000022437318,
target       = 0.6579736267,
estimate/target = 0.0000385102582.                      (14)
```

The error is the standard error of eight independent complete scrambled
replicates.  The contribution effective sample size is `53.43`, the largest
node supplies `8.69%`, and the relative scramble error is `8.85%`.  These are
substantial improvements over the unstratified baseline, but they still make
this a pilot rather than a precision integral.

After the CFT-order audit, the best current fixed-design estimate is

```text
F_2(1)/g_s^2 = 0.000025347448321 +/- 0.000002252010582,
4096 * estimate / target = 0.157792.                    (14a)
```

The higher-order shift is `+0.03439%` relative to `8/8`. It is much too small
to explain the absolute discrepancy discussed below.

A striking diagnostic is

```text
2 pi 4096 * estimate / target = 0.99110.                (15)
```

After the exact free-field measure conversion, the same historical numerical
clue is an overall factor `2 pi 2^12`, not `2^12`. Neither factor is inserted
or fit. The genus-one-anchored separating audit described below fixes the
correlated convention bridge to one, so this clue is now recorded only as a
numerical disagreement with the matrix-model target.

The subsequent normalization review found one definite bookkeeping error but
did not derive `2^12`. BRY's coupling obeys
`mu^-1=2*pi*g_s_BRY`, whereas the string notes use
`mu^-1=4*pi*g_s_Xi`; hence `g_s_BRY=2*g_s_Xi`. The old topology ledger was
therefore low by four. The corrected statement is
`mu^-1=2*pi*g_s_BRY=4*pi*g_s_Xi`, with the matrix-model expansion variable
`g=mu^-1`; the factor of two converts the two string couplings and does not
rescale the coefficients of the `g` expansion.

More precisely, the intrinsic Liouville factor is one because Xi and BRY use
the same normalized Liouville CFT data. If
`I_crit^Xi=A_crit I_crit^code`, `Z_X^Xi=A_X Z_X^pl`, and
`Z_XR^Xi=A_XR Z_XR^code`, the implemented density differs from Xi's by the
single correlated local constant

```text
Lambda_local = A_crit*A_XR/A_X^26.
```

The scalar part of this dictionary is fixed by (5a)--(5b), which give the net
factor `1/(2 pi sqrt(alpha'))`. The normalized genus-one critical density,
once-punctured-torus separating residue, scalar state metric, Liouville
inverse metric, Polyakov recurrence, and single global genus-two stack weight
then give

```text
Lambda_local = A_crit*A_XR/A_X^26 = 1.
```

Because this anchor is at genus one, it cannot test the Euler-characteristic
sphere normalization. The separate sphere audit gives
`Lambda_top=2/alpha'`, also without using the matrix-model genus-two answer.
The much larger multiplier near `2 pi * 2^12` that would force numerical
agreement is not an allowed convention adjustment.

Two numerically suggestive explanations are ruled out. The identity
`2^12=64^2` is already present as the leading raw-theta-product coefficient
and is accounted for by the `2^24` nonchiral Mumford conversion. A sharper
three-edge possibility would be `q_CFT=4*q_geom`, since the nonchiral
threshold factor would be `|4^3|^2=2^12`. It is ruled out by the literal CCY
gluing `u*v=q`, which is also the transition used by the period solver, and by
the momentum-dependent factor it would generate away from threshold. The
sphere four-point elliptic replacement `q -> 16q` is inapplicable and would
give `2^24`, not `2^12`, in the nonchiral three-edge threshold partition. A
direct full-CFT maximal-degeneration sewing remains a useful independent
cross-check. The genus-one-anchored separating calculation fixes
`Lambda_local=1`, while the sphere calculation fixes
`Lambda_top=2/alpha'`. The dedicated `bry_xi_convention_map.md` audit shows
that this is not a Liouville normalization question.

The normalization-free radius shape is already much more discriminating.  On
a 17-point reciprocal grid,

```text
R=sqrt(2): worldsheet/matrix-model shape = 1.01529 +/- 0.01239,
R=2:       worldsheet/matrix-model shape = 1.04632 +/- 0.03792. (16)
```

The maximum discrepancy for `1/2 <= R <= 2` is `4.63%`, about `1.2` standard
errors, while nodewise and integrated T-duality hold below `3.4e-16`.  The
present result therefore supports the predicted radius dependence and
isolates the remaining absolute discrepancy as moduli independent at current
precision.  The next controlled step is the nested variance-targeted design,
not a fitted Weyl or normalization factor.

## 7. Code and review order

The production path is:

```text
prepare_genus2_rqmc_production.py
  -> genus2_moduli_rqmc.py / genus2_moduli_tail_stratified_rqmc.py
  -> genus2_siegel_fundamental_domain.py
  -> preflight_rqmc_period_map.py
  -> genus2_plumbing_atlas.py + plumbing_algorithms.py
  -> monte_carlo_integrate_genus2_c1.py
  -> genus2_c1_string_integrand.py
  -> ccy_genus2_block.py / ccy_genus2_glasses_block.py
  -> free_boson_plumbing.py + genus2_bc_ghost.py
  -> assemble_genus2_c1_rqmc.py
  -> reweight_genus2_c1_rqmc_radius.py
```

Tail-design diagnostics and adaptation are in
`analyze_genus2_rqmc_tail_sampling.py` and
`allocate_genus2_tail_strata.py`.

Review the normalization first in
`genus2_integrand_normalization.py`, then the pointwise density in
`genus2_c1_string_integrand.py`, the geometry in
`genus2_plumbing_atlas.py` and `plumbing_algorithms.py`, and finally the
sampling and assembly code.

The current compact output is

```text
results/genus2_c1_moduli_mc/rqmc_t3_stratified_R8_K8_M8/
  README.md
  assembled_b8q8/summary.json
  cft_adaptive_promotion/report.md
  cft_adaptive_promotion/convergence_manifest.json
  cft_adaptive_promotion/assemblies/adaptive_round_001/summary.json
  radius_sweep_b8q8/radius_sweep.csv
  radius_sweep_b8q8/summary.json
  radius_sweep_b8q8/radius_dependence.png
  allocation_b8q8_M288/recommended_tail_allocation.json

results/genus2_c1_moduli_mc/rqmc_t3_stratified_R8_K8_Neyman288/
  domain_nodes.csv
  production_nodes.csv
  new_production_nodes.csv
  summary.json
```

Focused checks are the adjacent `*_checks.py` files for the normalization,
integrand, Siegel-domain sampler, RQMC design, plumbing atlas, period-map
preflight, assembler, and radius reweighter. The exact factorization ledger is
also emitted by `genus2_integrand_factorization_audit.py`.
