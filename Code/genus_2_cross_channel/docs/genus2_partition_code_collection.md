# Genus-Two Partition Function Code Collection

> **Status note (2026-07-11).**  This historical code inventory remains useful
> for the block implementations, but its sections 10--12 predate the separation
> between the local CFT sewing coefficient and the Polyakov gauge-fixing factor
> `N_(h,n)`.  In particular, their claims of an "absolute" genus-two
> normalization are superseded.  The current reviewer-facing account is
> [`genus2_current_computation_summary.md`](genus2_current_computation_summary.md).

This note collects the code currently used to produce, check, and diagnose a
genus-two Liouville partition function from plumbing data.

The main principle is:

```text
plumbing q's -> period matrix Omega -> Virasoro block -> Liouville momentum integral
```

Raw plumbing coordinates are chart coordinates.  Equal raw `q` values in two
different plumbing frames do not, by themselves, describe the same Riemann
surface.  Comparisons between frames must be made at the same period matrix
`Omega`, or after inverting the transformed `Omega` into the target plumbing
chart.

## 1. Current Main Production Path

The current finite-`c` Liouville genus-two computation is the CCY
two-holed-disc, or theta-graph, plumbing path:

```text
plumbing/liouville_genus2_ccy.py
    uses plumbing/ccy_genus2_block.py
    uses plumbing/liouville_torus.py
    optionally uses the large-c Schottky vacuum seed from plumbing/genus2_vacuum_blocks.py
```

The object evaluated by `liouville_genus2_ccy_partition` is the raw CCY
plumbing-frame integral

```text
Z_CCY,raw^(N) =
int dP1 dP2 dP3 / pi^3
    C(P1,P2,P3)^2
    | prod_i q_i^h_i
      F_CCY^(N)(h1,h2,h3;c;q1,q2,q3) |^2.
```

Here `F_CCY^(N)` is the genus-two Virasoro block truncated to order `N` in the
CCY plumbing variables.  Faithfully following CCY, the block contains the
descendant powers `q^level`, while `prod_i q_i^h_i` is the separated primary
sewing prefactor.  This is not the physical modular-invariant genus-two
partition function until the appropriate conformal-frame/anomaly normalization
has been derived.

Minimal smoke run:

```bash
python3 plumbing/liouville_genus2_ccy.py \
  --b 1 \
  --q1 0.003 \
  --q2 0.0025 \
  --q3 0.0012 \
  --block-order 5 \
  --p-max 2.0 \
  --quadrature-order 8 \
  --dps 24 \
  --vacuum-word-len 2 \
  --vacuum-oscillator-level-max 6 \
  --no-store-samples
```

The convergence plotting driver for this path is:

```text
plumbing/plot_liouville_genus2_ccy_convergence.py
```

It writes CSV/SVG output under:

```text
plumbing/results/liouville_genus2_ccy/
```

## 2. Riemann-Surface Layer: Plumbing q To Omega

These files define the Riemann surface associated to plumbing coordinates.

```text
plumbing/plumbing_algorithms.py
```

Important functions:

```text
generators_for_glasses(q1, q2, q_bridge)
generators_for_sunrise(q0, q1, q2)
generators_for_theta(q_zero, q_one, q_infty)

schottky_glasses_period_matrix(...)
schottky_sunrise_period_matrix(...)
schottky_theta_period_matrix(...)
schottky_theta_period_matrix_cross_ratio(...)

solve_glasses_inverse_from_omega(...)
solve_theta_inverse_from_omega(...)
solve_glasses_collocation(...)
solve_theta_collocation(...)
theta_leading_period_matrix(...)
```

The practical chart-identification diagnostic is:

```text
plumbing/identify_genus2_q_coordinates.py
```

It keeps the two relevant leading maps explicit.

Separating pair-of-tori chart:

```text
Omega_11 ~ log(q1)/(2 pi i)
Omega_22 ~ log(q2)/(2 pi i)
Omega_12 ~ q_bridge/(-2 pi i)
```

CCY theta-graph chart:

```text
Omega_11 ~ log(Q1 Q3)/(2 pi i)
Omega_22 ~ log(Q2 Q3)/(2 pi i)
Omega_12 ~ log(Q3)/(2 pi i)
```

This is the file to use before comparing two plumbing calculations.  It tells
us whether two sets of `q` values are actually representing the same `Omega`.

## 3. Virasoro Block Layer

### Genus-Two CCY Block

```text
plumbing/ccy_genus2_block.py
```

This is the main genus-two Virasoro block code.  It implements the
Cho-Collier-Yin no-puncture block in the two-holed-disc frame:

```text
F =
sum q1^|A| q2^|C| q3^|E|
    G_h1^{AB} G_h2^{CD} G_h3^{EF}
    rho(L_-A h1, L_-C h2, L_-E h3)
    rho(L_-B h1, L_-D h2, L_-F h3).
```

The public evaluator uses the central-charge recursion.  The regular part is
a large-`c` seed: a global `SL(2)` block times an optional Schottky
primitive-class vacuum product.  The pole part is kept as a partial fraction in
`c`, which is important when equal-weight Kac collisions produce higher-order
poles.  The universal residue factor
`-dc_rs/dh * A_rs` is evaluated in the simplified `x=b^2` form, with the
`x^2-1` zeros cancelled against the matching factors in the `A_rs` product
before numerical evaluation.  This removes the `0 * infinity` problem at
resonant points such as the `c = 25` order-seven collision.

Checks:

```bash
python3 plumbing/ccy_genus2_block_checks.py
```

### Large-c Vacuum Seed

```text
plumbing/genus2_vacuum_blocks.py
```

This contains the Schottky primitive-class product

```text
Z_vac = product_{primitive gamma} product_{n >= 2} (1 - q_gamma^n)^(-1/2).
```

In our Liouville computation this is not the full finite-`c` partition
function.  It is the `c = infinity` vacuum-block seed used inside the CCY
central-charge recursion, plus a useful independent check of Schottky plumbing
and vacuum-channel limits.

### Torus One-Point Blocks

```text
plumbing/virasoro_blocks.py
```

This implements the Zamolodchikov/Poghossian recursion for torus one-point
Virasoro blocks.  It is used by the separating pair-of-tori approximations and
by the genus-one modular tests.

Checks:

```bash
python3 plumbing/virasoro_blocks_checks.py
```

### Separating Descendant Blocks

```text
plumbing/torus_descendant_blocks.py
plumbing/zhu_torus_descendants.py
```

These files support the separating-channel bridge-descendant experiment.  They
construct Virasoro descendant bases, Gram matrices, and Zhu-recursion
operators for torus one-point descendants.

Checks:

```bash
python3 plumbing/torus_descendant_blocks_checks.py
python3 plumbing/zhu_torus_descendants_checks.py
```

## 4. Liouville CFT Data

```text
plumbing/liouville_torus.py
```

This file supplies the Liouville data used by both genus-one and genus-two
wrappers:

```text
Upsilon_b
DOZZ / Xi-Yin structure constants
h_P = Q^2/4 + P^2
dP/pi Liouville completeness measure
torus one-point Liouville momentum integral G(P_ext, q)
```

Checks:

```bash
python3 plumbing/liouville_torus_checks.py
```

## 5. Partition Function Assemblers

### Main CCY Liouville Wrapper

```text
plumbing/liouville_genus2_ccy.py
```

Use this when we want the current finite-`c` raw genus-two Liouville plumbing
integral in the CCY frame.  It performs the three-dimensional
Liouville momentum quadrature and calls `ccy_genus2_block.py` at each
quadrature point.

### Separating Pair-Of-Tori Primary Approximation

```text
plumbing/liouville_genus2.py
```

This computes only the primary bridge term in the separating pair-of-tori
channel:

```text
Z_2^(0) =
int dP3/pi |q_bridge|^{2(h3-c/24)} G(P3,q1) G(P3,q2).
```

It includes full torus one-point descendant sums around each handle, but not
the bridge descendants.  It is therefore a controlled separating-channel
approximation, not the full genus-two partition function.

### Separating Zhu Bridge-Descendant Experiment

```text
plumbing/liouville_genus2_separating_zhu.py
```

This is the experimental next step in the separating channel.  It uses Zhu
recursion to add bridge descendants on top of the pair-of-tori picture.  It is
useful for low-level comparisons against the CCY recursion, but it is not yet
the primary production path.

## 6. Modular And Frame Checks

```text
plumbing/liouville_genus2_modular_check.py
```

This applies `Sp(4,Z)` transformations in period-matrix variables,

```text
Omega' = (A Omega + B) (C Omega + D)^(-1),
```

and compares the Liouville matter result against the expected chiral-section
covariance

```text
Z_L(Omega', Omegabar') / Z_L(Omega, Omegabar)
  ?= |det(C Omega + D)|^(-c).
```

The check is faithful only when `Omega'` can be inverted into an independent
target plumbing chart.  Otherwise the script reports a chart failure or a
bookkeeping-only result.

Run the generator suite:

```bash
python3 plumbing/liouville_genus2_modular_check.py \
  --expected-law chiral-section \
  --suite sp4-generators
```

Frame comparison diagnostic:

```text
plumbing/compare_liouville_genus2_frames.py
```

This compares CCY and separating-Zhu approximations, but it explicitly warns
that same raw `q` values are not the same Riemann surface.  Use it only after
matching the period matrix.

## 7. Sweeps, Plots, And Result Files

```text
plumbing/sweep_liouville_genus2_ccy.py
plumbing/plot_liouville_genus2_ccy_convergence.py
plumbing/bolza_ccy_recursion.py
plumbing/sweep_ccy_block_theta_omega.py
plumbing/plot_liouville_genus2_q3_scan.py
```

`bolza_ccy_recursion.py` is a special high-symmetry-point driver.  It starts
from the Bolza period matrix

```text
Omega_B = 1/2 [[1 + i sqrt(2), 1], [1, 1 + i sqrt(2)]],
```

applies a fixed `Sp(4,Z)` move into a theta-chart marking, shifts the period
matrix to the principal logarithm branch, and evaluates the CCY Liouville
integral using the resulting leading theta q-values.  The script labels this
as a leading-coordinate evaluation because the all-order theta inverse at the
Bolza point is not yet reliable.

The most relevant current result bundle is:

```text
plumbing/results/liouville_genus2_ccy/
```

Example files already produced there include:

```text
ccy_b1_q003_0025_0012_quad9_11_block.csv
ccy_b1_q003_0025_0012_quad9_11_partition.csv
ccy_b1_q003_0025_0012_quad9_11.svg
```

## 8. What To Hand To A Reviewer

For a review of the actual genus-two Liouville computation, start with these
files in this order:

```text
1. plumbing/identify_genus2_q_coordinates.py
2. plumbing/plumbing_algorithms.py
3. plumbing/ccy_genus2_block.py
4. plumbing/liouville_torus.py
5. plumbing/liouville_genus2_ccy.py
6. plumbing/plot_liouville_genus2_ccy_convergence.py
7. plumbing/liouville_genus2_modular_check.py
```

The earlier separating-channel files are still important, but should be
described as approximations or cross-checks:

```text
plumbing/liouville_genus2.py
plumbing/liouville_genus2_separating_zhu.py
plumbing/virasoro_blocks.py
plumbing/torus_descendant_blocks.py
plumbing/zhu_torus_descendants.py
```

## 9. Current Caveats

1. The CCY and separating plumbing coordinates are different charts.  Matching
   requires matching `Omega`, not raw `q`.
2. The Schottky vacuum product is a large-`c` seed/check, not the full
   finite-`c` Liouville partition function.
3. All practical outputs are truncated in block order and in Liouville
   momentum quadrature.
4. The `c = 25` case is delicate because pole collisions can occur in the
   central-charge recursion.  The CCY code keeps partial fractions in `c` to
   track higher-order poles, and simplifies the universal residue prefactor
   before evaluation so zero-pole cancellations do not become numerical
   singularities.
5. A faithful genus-two modular check needs enough independent plumbing charts
   and inverse maps to evaluate the transformed period matrix in the correct
   frame.

## 10. Physical Integrand And Modular Monte Carlo

The current local string density is assembled by:

```text
plumbing/genus2_c1_string_integrand.py
plumbing/genus2_integrand_normalization.py
plumbing/genus2_bc_ghost.py
plumbing/free_boson_plumbing.py
```

`genus2_c1_string_integrand.py` requires Liouville and noncompact-scalar
partitions carrying the same explicit conformal-frame label.  At a plumbing
node it evaluates

```text
I2 = |N_Phi|^2 R Theta_R /
     (det(Im Omega)^13 |chi10|^2) * Z_L^pl / (Z_X^pl)^25
```

and exposes the separate `2^24` conversion from the raw even-theta product to
the unit separating-residue convention.  The global genus-two stack factor is
not included in the local density.

The full-domain driver is:

```text
plumbing/monte_carlo_integrate_genus2_c1.py
```

It consumes iid points drawn from

```text
p(Omega) d^6 Omega = d^6 Omega /
    (Vol(F2) det(Im Omega)^3)
```

and therefore uses

```text
F2/g_s^2 = (Vol(F2)/2) mean[det(Im Omega)^3 K2_note],
K2_note = I2_factorized/pi  (alpha'=1).
```

The `det(Im Omega)^3` is only the importance weight converting the invariant
proposal back to the physical period-coordinate form `d^3X d^3Y I2`.  It is
not a physical ghost factor.  The generic stack weight `1/2` is applied once.

Independent completed batches are combined without reevaluating the CFT by:

```text
plumbing/assemble_genus2_c1_mc_pilot.py
```

The current 32-point output is stored in:

```text
plumbing/results/genus2_c1_moduli_mc/pilot_R1_N32/
```

It historically reported the unit-Mumford value
`J2 = 2.4098784e-4 +/- 3.8564e-5` with an aggregate low/high CFT change of
`1.52%`.  Production assembly now applies the analytically fixed string-note
multiplier `1/pi`.  The corresponding self-dual target is
`F2/g_s^2=pi^2/15=0.6579736267`.  No multiplier is fitted to this mismatch.

The deterministic tail diagnostics are:

```text
plumbing/scan_genus2_c1_separating_cusp.py
plumbing/scan_genus2_c1_nonseparating_cusp.py
```

Both finite-range scans are compatible with integrable boundary behavior.
They do not replace an asymptotic proof or cusp-stratified Monte Carlo.

## 11. Coupling-Dictionary Correction Found During The Pilot

The BRY primary-source formulas give

```text
g = 2*pi*g_s,
C_S2 = 2*pi/g_s^2,
C_T2 = 1.
```

The symbol `g_s` in those equations is BRY's coupling. Xi's string notes use
a different coupling with `mu=(4*pi*g_s_Xi)^-1`. Hence

```text
g_s_BRY = 2*g_s_Xi,
g = mu^-1,
C_Sigma2^BRY = g_s_BRY^2/(2*pi) = 1/(8*pi^3*mu^2).
```

The previous ledger silently used `g_s_Xi` in the BRY formula and was low by
four. The string notes separately give `g_s_Xi^2/pi` multiplying their ordered
positive real period form because
`d^3Omega wedge d^3bar(Omega)=-8i d^3X d^3Y` and `N_(2,0)=-i`.  The code keeps
this `1/pi` kernel multiplier and the generic stack weight `1/2` separate. An
explicit conversion between the BRY amplitude measure and this string-note
form is still required; the two displayed coefficients must not simply be
equated.

For a review of the modular pilot, add these files to the list in section 8:

```text
8.  plumbing/genus2_siegel_fundamental_domain.py
9.  plumbing/genus2_plumbing_atlas.py
10. plumbing/free_boson_plumbing.py
11. plumbing/genus2_bc_ghost.py
12. plumbing/genus2_c1_string_integrand.py
13. plumbing/genus2_integrand_normalization.py
14. plumbing/monte_carlo_integrate_genus2_c1.py
15. plumbing/assemble_genus2_c1_mc_pilot.py
```

## 12. Absolute Factorization Audit

The reviewer-facing audit is:

```text
plumbing/genus2_integrand_factorization_audit.py
plumbing/genus2_integrand_factorization_review.md
plumbing/audit_bry_xi_convention_map.py
plumbing/audit_bry_xi_convention_map_checks.py
plumbing/bry_xi_convention_map.md
plumbing/results/genus2_c1_moduli_mc/factorization_audit.json
plumbing/results/genus2_c1_moduli_mc/bry_xi_convention_map.json
```

Its verdict is `local_cft_normalization_certified=true` and
`full_worldsheet_normalization_certified=true`. The BRY/Yin state metric, raw
`chi10` separating residue, compact zero mode, topology algebra, and the
genus-one-anchored once-punctured-torus matter-plus-ghost sewing identities
pass, fixing the correlated bridge to one without matrix-model input.

The formerly apparent D'Hoker--Phong discrepancy is reconciled exactly:

```text
D'Hoker--Phong: dmu_B = pi^-12 d^3Omega / Psi10_product
repository after unit-residue conversion: 2^24 * |2*pi*i|^2
nonchiral prefactor ratio: (2*pi)^26 = (Z_X^code/Z_X^DHP)^26
```

The code scalar uses `dX/(2*pi)`, whereas the cited determinant is quoted per
ordinary target volume, so `Z_X^code=2*pi Z_X^DHP`.  The same `(2*pi)^26`
conversion follows from the full genus-one critical measure.
Delta-normalized momentum states with completeness `dp` also give the
genus-two loop Gaussian `det(Im Omega)^(-1/2)` with coefficient one.

The observed near-`2^12` discrepancy is not explained by adding another
`64^2`: that number is already the `2^12` leading coefficient of the raw
theta product and is included through the `2^24` nonchiral conversion. A
hypothetical factor four on each sewing coordinate would give
`|4^3|^2=2^12` at the Liouville threshold, but CCY and the period solver use
the same literal gluing equation `u*v=q`; away from threshold it would also
produce a momentum-dependent change. The sphere four-point elliptic factor
`16` is inapplicable and would give `2^24` in the nonchiral three-edge
threshold partition. The intrinsic BRY/Xi Liouville conversion is exactly
one. The correlated critical-boson/scalar replacement in the string-note
path-integral measure is fixed by normalized genus-one-anchored sewing. If
`I_crit^Xi=A_crit I_crit^code`, `Z_X^Xi=A_X Z_X^pl`, and
`Z_XR^Xi=A_XR Z_XR^code`, the combined multiplier is
`Lambda_full=A_crit*A_XR/A_X^26=1`. This value follows from the critical
residue, scalar and Liouville state metrics, and the once-punctured-torus
sewing identity; it is not determined from radius-ratio tests.
