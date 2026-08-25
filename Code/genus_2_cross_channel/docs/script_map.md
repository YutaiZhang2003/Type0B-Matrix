# Plumbing Python Script Map

This file is a navigation map for the top-level Python scripts in `plumbing/`.
It summarizes what each script is for and which scripts are usually run as
checks, drivers, or reusable libraries.  Generated artifacts, result bundles,
TeX/PDF notes, and companion `.md` explanations are not listed individually.

## Core Plumbing And Inverse Maps

### `plumbing_algorithms.py`

Reusable numerical backbone for plumbing surfaces.  It contains Schottky
generators, small-`q` holomorphic one-form computations, all-`q` boundary
collocation, genus-one ribbon/plumbing helpers, lookup-table inversion, and
genus-two plumbing-to-ribbon utilities.

Use it as a library.  The companion explanation is `plumbing_algorithms.md`,
and reproducible checks live in `plumbing_checks.py`.

### `plumbing_checks.py`

Main reproducible check and data-generation driver for `plumbing_algorithms.py`.
It checks exact torus/Schottky identities, glasses and sunrise genus-two period
matrices, genus-one lookup/refinement behavior, and can write generated lookup
tables/plots under `plumbing/generated/`.

Use this when validating the plumbing algorithms or regenerating the standard
plumbing datasets.

### `plumbing_m21_inverse.py`

Experimental inverse map for the marked genus-two moduli space
`M_{2,1}`.  The target coordinate is `(Omega_11, Omega_12, Omega_22, u)`,
where `u` is an Abel coordinate of the marked point; the search target is a
fixed-perimeter one-face ribbon graph with eight independent edge parameters.

Use this for marked-point genus-two inverse-by-forward-map experiments.  The
companion note is `plumbing_m21_inverse.md`.

### `plumbing_m21_checks.py`

Checks for `plumbing_m21_inverse.py`.  It validates marked plumbing targets,
large-`L` inverse searches, and lookup-table based inverse selection.

Use this before trusting `M_{2,1}` inverse output.

## Genus-Two Period-Table Preparation

### `genus2_period_table_grid.py`

Defines the nested logarithmic/circular Sobol grid and writes a deterministic
theta/glasses cluster manifest without evaluating a period matrix.  Its
configuration is `config/genus2_period_table_cluster.json`.

### `genus2_period_table_cluster.py`

Guarded Slurm-array worker, routing preflight, checkpointed per-shard JSONL
output, fail-closed assembly, and portable feature-index construction.  The
worker runs only with an explicit `--execute` flag.  Production remains gated
on the high-precision holomorphic-form backend for mixed cusps.

### `genus2_period_table.py`

Mixed-backend table loader and KD-tree indices.  It supports a regularized
local `q -> Omega` interpolation and topology-adapted `Omega -> q` inverse
seeds.

### `genus2_period_table_coverage.py`

Post-assembly finite-sample Omega-space coverage audit.  It searches theta and
glasses markings, reports normalized table holes, and prepares refinement and
round-trip target manifests without running new period-map evaluations.

### `genus2_period_table_checks.py`

Preparation-only regression checks for grid size, routing, phase-periodic
indices, leading maps, local interpolation, and the cluster execution guard.
The complete design rationale and launch gates are in
`genus2_period_table_cluster_plan.md`.

## Virasoro And Liouville Building Blocks

### `virasoro_blocks.py`

Universal torus one-point Virasoro block implementation.  It implements the
Zamolodchikov/Poghossian recursion for the elliptic block and Dedekind eta
factor helpers.  It does not include CFT-specific structure constants.

Use it as the chiral block layer for torus one-point CFT calculations.  The
companion note is `virasoro_blocks.md`.

### `virasoro_blocks_checks.py`

Regression checks for `virasoro_blocks.py`.  It tests identity insertion,
partition-number coefficients, the level-one Ward identity, `b -> 1/b`
invariance, and small-`q` order stability.

Use this after touching the Virasoro recursion.

### `genus2_vacuum_blocks.py`

Genus-two vacuum Virasoro block tools in the plumbing/Schottky frame of
arXiv:1703.09805.  It computes the large-`c` primitive Schottky conjugacy-class
product for both the glasses/sunglasses and sunrise channels, using the
Schottky generators from `plumbing_algorithms.py`; it also contains a direct
finite-`c` theta-frame descendant-sum evaluator with a per-tube level cutoff
and low-order HMPZ Schottky-sewing comparison formulae.

Use this for the current genus-two vacuum block seed and for checks of
separating degeneration, Schottky marking invariance, low-level finite-`c`
coefficients, HMPZ vacuum-channel coefficients, and channel-level block
evaluation.  The companion note is `genus2_vacuum_blocks.md`.

### `liouville_torus.py`

Generic Liouville torus one-point machinery built on `virasoro_blocks.py`.
It implements the real-`b` Upsilon function, DOZZ/Xi-Yin structure constants,
and the one-dimensional Liouville momentum integral for diagonal torus
one-point functions.

Use this for generic non-resonant Liouville torus one-point evaluations.  Exact
`b=1` generic blocks are guarded because the current recursion is resonant
there.  The companion note is `liouville_torus.md`.

### `liouville_torus_checks.py`

Checks for Liouville Upsilon/DOZZ data and the torus one-point integral.  It
tests Upsilon reflection and shift identities, lambda/alpha DOZZ convention
conversion, the Xi-Yin `b=1` normalization, the `b=1` resonance guard, and a
small torus integral quadrature-stability sample.

Use this after touching Liouville CFT data or the generic torus one-point
integral.

### `liouville_modular_check.py`

Numerical S-modular covariance scan for Liouville torus one-point functions.
It can check either the full scalar one-point function or the HJS stripped
normalization, and writes CSV/SVG summaries.

Use this to diagnose modular behavior of the generic Liouville one-point
implementation.

### `liouville_partition.py`

Volume-normalized genus-one `c=25` Liouville partition density.  For the
genus-one free-energy check it evaluates the continuum character trace
analytically as a Gaussian momentum integral, with a numerical quadrature
version available as a control.

Use this as the Liouville partition factor in compact-boson times Liouville
genus-one checks.

## Genus-One Compact Boson Times Liouville Integrals

### `stratified_integrate_rho_ribbon.py`

Integer-shell, cusp-stratified Monte Carlo integration of the genus-one ribbon
density

```text
rho = (1/3) * (1/tau2) * ((2*pi)^18/3)
      * bghost_density * Z_compact(R,tau) * Z_Liouville(q).
```

It provides the shared `evaluate_rho` routine used by newer UV-stratified
drivers, plus older shell-based sampling helpers.

Use it as the common evaluator for ribbon-density integrations.

### `uv_integrate_rho_ribbon.py`

UV-stratified Monte Carlo driver for the genus-one ribbon density.  It samples
continuous UV variables `(u,t,v)`, rounds to integer theta-graph lengths, and
writes samples, stratum summaries, and an SVG contribution plot.

Use this for independent equal-allocation UV Monte Carlo runs.

### `adaptive_uv_integrate_rho_ribbon.py`

Adaptive UV-stratified Monte Carlo driver.  It reads a pilot stratum CSV,
allocates samples proportional to `full_area * std_rho` with a per-stratum
floor, then evaluates the same ribbon density as `uv_integrate_rho_ribbon.py`.

Use this for variance-weighted follow-up runs after an equal-allocation pilot.

### `integrate_tau_compact_liouville.py`

Direct cutoff integral over the standard genus-one fundamental domain in
`tau`.  It integrates the analytically simplified compact-boson times
Liouville density with the physical torus reflection weight `1/2`, and
appends the large-cusp zero-mode tail estimate.

Use this as the tau-coordinate normalization/control check for the ribbon
Monte Carlo result.

### `sweep_tau_compact_liouville_radius.py`

Direct tau-coordinate sweep over compact boson radius `R`.  It reuses
`integrate_tau_compact_liouville.py`, appends the same large-cusp zero-mode
tail estimate, and writes a summary/plot comparing the physical direct values
with `(R + 1/R)/24`.  The doubled output
column is retained only as an unquotiented factor-two diagnostic; it is not
the genus-one free energy.

Use this as the tau-coordinate control for the compact-boson radius dependence.

### `genus1_tau_normalization_checks.py`

Reviewer-facing absolute-normalization check.  It compares the unsimplified
matter--ghost product with the Poisson-resummed density, integrates seven
radii directly over the fundamental domain, checks T-duality and cusp
convergence, and verifies that omitting the torus `1/2` doubles the MQM result.

Use this before importing genus-one normalization conventions into a
higher-genus calculation.

### `mapped_cell_analytic_pullback.py`

Mapped-cell analytic pullback check in ribbon coordinates.  It triangulates
the fixed-perimeter ribbon-length simplex, maps each cell to the raw period
`tau`, and weights by the image area in the `tau` plane instead of using
pointwise finite-difference Jacobians.

Use this as an independent Jacobian/pullback control for genus-one ribbon
coordinates.

### `sweep_uv_rho_ribbon_radius_from_samples.py`

Post-processing script that reweights saved UV Monte Carlo samples over compact
boson radius `R`.  It recomputes the compact partition ratio
`Z_compact(R,tau) / Z_compact(1,tau)` and compares the estimate with
`(R + 1/R)/12`.

Use this when the `R=1` samples are already available and only the compact
boson radius dependence is being scanned.

### `plot_uv_sampling_convergence_independent.py`

Plotting/post-processing script for independent UV Monte Carlo convergence
runs.  It reads multiple stratum-summary CSV files, totals their estimates and
errors, writes a combined summary CSV, and produces a PNG convergence plot.

Use this to summarize sample-count convergence across separate runs.

## Genus-Two Integrand And Plumbing Atlas

### genus2_c1_string_integrand.py

Assembles the compact boson, Liouville matter input, and canonical bc ghost
density into the Weyl-frame-independent genus-two integrand.  It keeps the
raw theta-product and locally normalized unit-Mumford-residue cusp-form
conventions explicit.

Use this as the local integrand layer for the eventual modular-domain
integration.

### genus2_integrand_normalization.py

Stores the Polyakov `N_(h,n)` recurrence, separating residue normalization,
raw theta-product conversion, the distinct BRY topology constants, and the
mu = 1/(4*pi*g_s) dictionary.  The BRY
relations `g=2*pi*g_s` and `C_S2=2*pi/g_s^2` are kept explicit so they cannot
be combined incorrectly with `g=mu^-1`. It also records the c=1 timelike
sphere constant, the reduced metric `Khat_S2=4*pi`, and the genus-g vacuum
correction `(2/alpha')^(g-1)`. The latter is unity at genus one and
`2/alpha'` at genus two.

Use this for the exact cusp-form conversion, topology dictionary, and
provisional MQM target of the genus-two integral.  The local executable sewing proof is
in `genus2_integrand_factorization_audit.py`.

### genus2_integrand_factorization_audit.py

Runs the reviewer-facing normalization ledger.  It checks the BRY resonance
identity, raw `chi10` residue, compact and noncompact scalar sewing, the full
genus-one critical measure, the D'Hoker--Phong convention conversion, and the
topology algebra.  It certifies the local CFT normalization and separately
checks the conversion of `N_(2,0)=-i` to the positive real period measure. It
reports the c=1 sphere-topology factor separately from the local bridge and
keeps full production certification false until the factor is applied.

### audit_c1_sphere_topology_normalization.py

Performs the analytic critical-to-c=1 sphere-topology audit without a
matrix-model genus-two input or a numerical fit. It checks
`K_S2^crit=8*pi/alpha'`, `Khat_S2^c1=4*pi`, the genus-two correction
`2/alpha'`, the final positive-real kernel coefficient `2/pi`, and the
independent BRY coupling cross-check. Its companion `_checks.py` verifies the
identities at several values of `alpha'`.

### genus2_integrand_factorization_review.md

Derives the local sewing constant, including why the `(2*pi)^26`
D'Hoker--Phong ratio is exactly the 26-scalar target-volume convention
conversion already present at genus one.  Its status correction points to the
current computation summary for the remaining gauge-fixing issue.

### genus2_plumbing_atlas.py

Searches finite-depth Sp(4,Z) images of the theta and glasses pants graphs.
It uses leading plumbing coordinates and a finite-word Schottky solve only to
shortlist markings and initialize the inverse.  In the bulk it solves the
normalized holomorphic one-forms, records the direct integral branch and
modular matrix, and rejects inaccurate, basis-unstable, seam-inaccurate, or
asymmetric inverse solves.  Schottky is the final map only in a deep cusp.
This is a homology-marking atlas; Torelli-distinct pants decompositions and
multiple inverse roots still need a pants-move continuation layer.

Use this at every modular-integration node to choose a candidate plumbing
chart.  Its q <= 0.16 label is the current real order-12 calibration envelope,
not a universal convergence theorem.

### genus2_hybrid_period_map.py

Implements the shared full-chart numerical split for `q -> Omega`: adaptive
normalized-form collocation in the bulk, a rescaled multiprecision
holomorphic-form solve when one tube is extremely long but the others are
finite, Schottky words only when every plumbing parameter is small, and a
two-method overlap where the period matrices must agree modulo integral
B-period shifts. It also checks standard sewing-disk geometry and provides
both inverse refiners used by the atlas.

### genus2_hybrid_period_map_checks.py

Checks the cusp, overlap, transition, bulk, invalid-chart, and public `auto`
dispatch paths.  In the overlap it requires both methods to converge and agree
within the configured numerical bar.

### genus2_plumbing_atlas_checks.py

Checks the leading coordinate maps, finite-depth handle-swap invariance, the
efficient Bolza theta image, and the known theta--glasses overlap point.

Use this before changing the atlas search, period inversion, or chart scoring.

### sample_genus2_plumbing_moduli.py

Generates a six-real-dimensional Sobol cloud directly in complex theta and
glasses sewing parameters, maps every point to a marked period matrix with
normalized holomorphic one-forms, and records basis, seam, symmetry,
holomorphy, Jacobian, and finite-depth Gottschling diagnostics.  It separately
constructs a fixed-marking theta--glasses overlap sample by matching period
matrices and validating both direct maps at a raised basis.

Use this to test the bulk `q -> Omega` map and to prepare period-matched points
for channel comparisons.  The resulting chart cloud is not itself an
integration proposal because chart multiplicities and a partition of unity
have not been supplied.

### compare_sampled_plumbing_frames.py

Evaluates the Liouville momentum integral and full noncompact scalar partition
in both frames on selected rows of the direct overlap sample.  Its consistency
observable is the frame-independent ratio
`[Z_L/(Z_X)^25]_theta / [Z_L/(Z_X)^25]_glasses`; the raw Liouville ratio is
reported only as a local-coordinate diagnostic.  Block and momentum orders can
be promoted independently, and explicit overlap IDs can be selected.

Use this for the pointwise two-channel test at the same marked period matrix.
It applies no fitted frame factor, propagator shift, or normalization constant.

### sample_genus2_plumbing_moduli_checks.py

Checks the saved direct-map certificates and the convergence pattern of the
three-point theta--glasses comparison.  It also guards the current scientific
status: the ratios move toward unity, but the refined worst point retains an
unresolved approximately `1.2e-3` residual.

Use this as the focused regression check for the direct plumbing sample and
same-modulus frame comparison.

### summarize_plumbing_frame_contrast.py

Assembles the broad fixed-period theta--glasses stress test from its separate
block/quadrature batches.  It records the efficient topology, chart contrast,
best available ratio, latest independent block and quadrature movements,
period-map certificates, and scalar word-cutoff correction for every selected
point.  Plotting is optional so the numerical summary does not require
Matplotlib.

Use this to quantify locality accuracy when one chart has small sewing
parameters and the other reaches `qmax` between roughly `0.20` and `0.46`.

### summarize_plumbing_frame_contrast_checks.py

Checks the 17/32 broad inverse success count, the symmetric eight-point CFT
selection, period-map tolerances, order-ladder ceilings, scalar word
convergence, and the two refined unresolved channel differences.

Use this as the regression check for the empirical broad-chart accuracy
envelope.

### plot_genus2_plumbing_atlas.py

Scans the symmetric imaginary period-matrix slice
Omega = i*y*[[1,rho],[rho,1]].  It writes a leading-score grid plus sparse
finite-q certification data and plots the preferred topology and uncovered
samples.

Use this as the first coverage diagnostic; it does not claim coverage of the
full six-real-dimensional genus-two modular domain.

### refine_genus2_plumbing_atlas.py

Reads a saved atlas scan and reruns selected unresolved statuses at deeper
symplectic search and Schottky word cutoffs.  The default hard-band result
shows that the two initially unresolved slice samples have valid theta period
charts but max |q| near 0.25.

Use this before interpreting a low-cutoff red sample as a period-map failure.

### genus2_siegel_fundamental_domain.py

Implements the complete finite Gottschling-domain test and an exact
accept/reject sampler for the invariant measure d^3X d^3Y/det(Y)^3. It also
estimates the domain volume and checks it against pi^3/270.

Use this to generate full six-dimensional modular-integration nodes; do not
replace it by a rectangular Siegel-set sample.

### genus2_siegel_fundamental_domain_checks.py

Checks all 15 determinant shifts, scalar/vector domain membership, the
accept/reject envelope, and recovery of the exact genus-two Siegel volume.

### scan_genus2_moduli_plumbing_coverage.py

Runs the plumbing atlas on invariant-volume samples from the full genus-two
fundamental domain. It keeps the 768-point leading census, the 96-point iid
finite-q audit, and a 24-point adversarial hard-tail audit separate.

### refine_genus2_full_moduli_coverage.py

Rechecks every unresolved full-domain sample at symplectic depth four and
Schottky word length six. The iid sample reaches 96/96 certified period
charts; 69/96 are inside the current q <= 0.16 reference envelope.

### assemble_genus2_full_moduli_coverage.py

Combines the base and refined scans into the reviewer-facing CSV, JSON, and
three-panel figure without rerunning the nonlinear inversions.

### genus2_full_moduli_coverage_checks.py

Audits all saved domain memberships, volume normalization, iid and hard-tail
selection rules, combined status counts, and the deep-cusp regression point.

### monte_carlo_integrate_genus2_c1.py

Evaluates the unit-Mumford-residue c=1 matter--ghost density on iid
invariant-domain points.  It selects a checked plumbing chart, evaluates
low/high Liouville and scalar truncations in the same frame, and uses
`(Vol(F2)/2) mean[det(Y)^3 I2]`.  The determinant power is proposal
reweighting only. In nonseparating cusps it keeps the true `log(q)` in the
Liouville propagator and quadrature scale, uses the surviving-handle scalar
oscillator limit, and evaluates `log|chi10|` without product underflow.

It also accepts direct-importance scrambled-Sobol manifests.  Those rows carry
their own `w/(2M)` integration weights, and the driver can evaluate one entire
scramble with `--rqmc-replicate`.  `--resume` verifies the source hash and all
numerical settings, preserves successful node IDs with atomic checkpoints,
and retries failed or interrupted nodes.

It additionally accepts `scrambled_sobol_physical_mixture` rows. Those carry
the exact `J_Y/(2 N p_mix)` physical-measure coefficient and set the local
kernel determinant power to zero. A portable period-table index can be loaded
to seed the Omega-to-q atlas search; every seed is still recomputed and
certified by the live hybrid period map.

### genus2_moduli_rqmc.py

Generates independently scrambled, nested Sobol designs in all six variables
of the Minkowski proposal.  It retains the exact Gottschling indicator and
importance weight, gives every in-domain node a stable nested-design ID, and
uses the exact Siegel volume as a control integral.

### genus2_moduli_physical_mixture_rqmc.py

Generates the preferred four-component physical-period-measure Sobol design.
Bulk, one-handle, common-scale, and double-cusp components are combined with
the deterministic balance heuristic. It evaluates the exact invariant-volume
control without CFT work and reports component, shell, cumulative-tail, ESS,
and largest-node diagnostics after the local kernel has been evaluated.

### genus2_moduli_tail_stratified_rqmc.py

Partitions the unbounded `t3` proposal coordinate into exact dyadic cells and
one semi-infinite remainder.  Each cell receives an independently scrambled
Sobol net and its exact probability mass, so the full cusp is retained without
a cutoff.  Per-stratum powers may differ while stable node IDs preserve nested
CFT reuse.

### allocate_genus2_tail_strata.py

Uses a completed fixed-order pilot to estimate `P_s sigma_s` in every `t3`
cell, including known out-of-domain zeros, and recommends power-of-two sample
counts under a fixed proposal budget.  This is a scheduling heuristic only;
reported errors still come from complete independent scrambles.

### analyze_genus2_rqmc_tail_sampling.py

Reports which proposal-coordinate tails dominate the integral, contribution
effective sample sizes, largest-node fractions, and reciprocal-radius
diagnostics.  It is used to distinguish a sampling failure from a CFT or
normalization failure.

### prepare_genus2_rqmc_production.py

Adds leading plumbing difficulty diagnostics without changing any integration
weight or dropping hard nodes.  For a nested extension it writes a manifest
containing only new CFT nodes while preserving the complete current design.

### assemble_genus2_c1_rqmc.py

Combines CFT outputs by stable node ID and current design weight.  It refuses a
headline estimate if any expected node is missing or failed or fewer than two
independent scrambles are complete.  A successful retry replaces an earlier
failed row with the same stable ID, while conflicting successful duplicates
remain an error.

### reweight_genus2_c1_rqmc_radius.py

Reuses one complete fixed-order RQMC CFT sample at every compactification
radius.  It replaces only the exact compact-boson lattice factor, enforces the
holomorphic-form period certificate, computes paired ratio-of-means errors
across independent scrambles, checks nodewise T-duality, and compares with
`f_2(R)/f_2(1)` without fitting an overall normalization.

### reweight_genus2_c1_rqmc_radius_checks.py

Checks complete-scramble and fixed-order validation, certified deep-cusp
handling, paired jackknife identities, T-duality, and the compact-radius
reweighting formula.

### benchmark_genus2_moduli_rqmc.py

Benchmarks iid and scrambled-Sobol proposals at equal cost on the exact
`pi^3/270` Siegel-volume control integral.

### audit_genus2_iid_pilot_coverage.py

Compares the old 64-node iid pilot with a large independent invariant-domain
sample in five cusp marginals and reports diagnostic quartile
post-stratifications of the saved radius sweep.

### audit_q_to_omega_accuracy.py

Reconstructs the marked target period matrix for every saved plumbing node,
checks bulk points at increasing Laurent basis and cusp points at increasing
Schottky word order, and selectively re-solves the direct inverse map.  Its
live `validate_or_refine_period_map` entry point applies the full shared hybrid
policy, including the overlap comparison, and is mandatory in the production
CFT driver.  Sewing coordinates below binary64 range are carried as `log(q)`
and validated by the multiprecision certificate.

### preflight_rqmc_period_map.py

Selects the largest leading-q node in every nonempty plumbing tier of every
RQMC scramble, performs the full finite-q atlas inverse, and applies the
production holomorphic-form period certificate before any Liouville block is
evaluated. It can instead select the deepest tail node in each scramble or an
explicit list of stable node IDs, and checkpoints each timeout-isolated node.

### assemble_genus2_c1_mc_pilot.py

Combines completed independent CFT batches, rejects failed or duplicate rows,
and reports Monte Carlo error, bootstrap diagnostics, batch consistency, and
low/high truncation changes separately.  It does not fit the normalization to
the matrix-model target.

### refine_genus2_c1_mc_nodes.py

Reuses the certified charts saved by the pilot and evaluates selected nodes
on a componentwise block/quadrature ladder such as `4/4 -> 6/4 -> 6/6` or
`6/6 -> 8/6 -> 8/8`.  This separates recursion and momentum-quadrature
movement and checkpoints every completed node.

### assemble_genus2_c1_refined_pilot.py

Requires every pilot node at order `6/6`, applies available `8/8` overrides,
and independently raises the scalar primitive-word cutoff through lengths 8
and 10.  It reports the Monte Carlo error, signed ladder shifts, and observed
absolute/RMS last-step diagnostics without treating them as rigorous bounds.

### reweight_genus2_c1_radius.py

Reuses the same refined period matrices and CFT values at every compactification
radius.  It recomputes only the exact compact-boson lattice factor, preserves
nodewise correlations, checks T-duality, and compares the normalization-free
radius shape with the matrix-model coefficient using a paired jackknife.

### reweight_genus2_c1_radius_checks.py

Checks the self-dual coefficient, exact recovery of the saved `R=1` pilot,
recomputed winding sums, reciprocal-radius T-duality, and paired-jackknife
identities.

### scan_genus2_c1_separating_cusp.py

Follows fixed-period separating rays and tests the contribution per
logarithmic transverse radius against the codimension-two integrability
threshold.

### scan_genus2_c1_nonseparating_cusp.py

Sends one theta sewing parameter directly to zero at fixed values of the other
two.  It records `det(Y)^3 I2/T^2`, the physical contribution per logarithmic
tube-length interval, without an inverse-period solve.

### audit_bry_xi_convention_map.py

Separates the intrinsic BRY/Xi Liouville identity from the full string
amplitude convention map. It records `g_s^BRY=2*g_s^Xi`, the differential-form
factor per complex modulus, the genus-one two-point cancellation, and why none
of these factors should be reapplied to the present Xi-normalized kernel.

### audit_bry_xi_convention_map_checks.py

Checks the exact coupling and real-measure powers, the unit Liouville factor,
and the reviewer-facing verdict that the BRY/Xi map does not derive `2^12`.

## Result Locations

The current compact-boson times Liouville genus-one result bundle is stored in
`plumbing/results/genus1_compact_liouville/`.  Its `README.md` explains which
CSV/PNG/SVG files are intended as the small reviewable output set, and which
raw Monte Carlo files are intentionally excluded from Git.

The first genus-two plumbing-atlas scan is stored in
plumbing/results/genus2_plumbing_atlas/.  Its JSON records the scan settings
and every sparse finite-q certification result; the CSV contains the full
leading-score grid.

The full six-dimensional coverage test is stored in
plumbing/results/genus2_full_moduli_coverage/. The combined JSON distinguishes
invariant-volume chart coverage from local-integrand weight.

The direct plumbing-coordinate cloud and fixed-marking theta--glasses overlap
test are stored in
`plumbing/results/genus2_plumbing_moduli_samples/direct_bulk_N128_overlap_N32/`.
Its `README.md` records the direct-period accuracy, three-point convergence
table, numerical component audits, and the remaining `0.12%` channel
consistency floor.

The deliberately asymmetric channel test is stored in
`plumbing/results/genus2_plumbing_moduli_samples/contrast_overlap_N32/`.
It contains 17 certified broad overlap pairs, eight partition-function order
ladders, free-scalar word convergence through length 12, and the current
approximately one-percent empirical locality envelope.

The locally normalized modular pilots and cusp scans are stored in
`plumbing/results/genus2_c1_moduli_mc/`. The current reviewer-facing result is
the complete fixed-order nested run in `rqmc_holomorphic_R8_M64_b8q8/`. The
local normalization ledger and the correlated critical-boson/scalar bridge
are recorded in `factorization_audit.json`; the source-by-source convention map
is in `bry_xi_convention_map.json`. Its compact-radius shape test is
independent of the unresolved radius-independent overall normalization.

The coverage-controlled replacement designs are stored in
`rqmc_design_R8_M64/` and `rqmc_design_R8_M128/`.  The latter is nested in the
former and requires only 374 new CFT evaluations.  Proposal-method convergence
is recorded in `rqmc_volume_benchmark/`, while the old pilot marginal audit is
in `pilot_R1_N64_coverage_audit/`.

The first fixed-order direct-period staging result is stored in
`rqmc_holomorphic_R8_M16_b8q8/`.  It contains eight complete `M=16` scrambles,
98 CFT nodes evaluated uniformly at block/quadrature order `8/8`, and the
normalization-independent compact-radius sweep.  Its nested `M=32` extension
is in `rqmc_holomorphic_R8_M32_b8q8/`, and the completed 383-node `M=64`
extension is in `rqmc_holomorphic_R8_M64_b8q8/`.
