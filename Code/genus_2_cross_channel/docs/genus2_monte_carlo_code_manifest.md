# Complete code manifest for the genus-two free-energy computation

This is the reviewer-facing source manifest for the current genus-two
Monte Carlo calculation.  It is deliberately organized by whether a file can
change the numerical answer.  Historical pilots, plots, and abandoned sampling
designs are not silently mixed into the production path.

The current target is

```text
mathcal F_2^str(R)/g_s^2
  = (1/2) integral_{F_2^coarse} d^3X d^3Y K_2^{c=1}(Omega;R),

K_2^{c=1} = (2/pi) I_2^Xi                         (alpha'=1).
```

The factor `1/2` is the generic genus-two stack weight.  The preferred sampler
uses the physical period-coordinate measure directly, so its estimator is

```text
(mathcal F_2^str/g_s^2)_s
  = (1/2) mean_{all 4M proposals in scramble s}
      [1_{F_2}(Omega) (J_Y/p_mix) K_2^{c=1}(Omega;R)].
```

There is no extra `det(Im Omega)^3` in this estimator.  That factor appears
only in the older invariant-measure importance sampler.  Independent complete
Owen scrambles, rather than individual nodes, determine the statistical error.

## 1. End-to-end production entry points

These are the files one runs, in order, once the period-table index exists.

| Stage | Source | What it does | Affects the value? |
|---|---|---|---|
| sample | `genus2_moduli_physical_mixture_rqmc.py` | Builds the four-component scrambled-Sobol design, exact mixture weights, Gottschling-domain indicator, and invariant-volume control | yes: nodes and weights |
| schedule | `prepare_genus2_rqmc_production.py` | Attaches plumbing difficulty estimates and constructs the production manifest | no mathematical reweighting; yes if it misidentifies a node |
| coverage helper | `scan_genus2_moduli_plumbing_coverage.py` | Supplies the leading plumbing scan used by the scheduler | chart scheduling only |
| preflight | `preflight_rqmc_period_map.py` | Queries and then independently certifies period-map seeds before the expensive CFT evaluation | acceptance/certification |
| pointwise CFT | `monte_carlo_integrate_genus2_c1.py` | Evaluates one or more nodes: chart, period certificate, Liouville, scalar, compact lattice, local kernel, and design-weighted contribution | yes |
| strict assembly | `assemble_genus2_c1_rqmc.py` | Requires every in-domain node, checks node coordinates and conventions, forms complete scramble estimates, and computes diagnostics | yes |
| radius reweight | `reweight_genus2_c1_rqmc_radius.py` | Recomputes the compact genus-two theta sum on the fixed nodes and assembles each radius | yes for the radius curve |
| radius utilities | `reweight_genus2_c1_radius.py` | Radius grids, scalar reweighting helpers, and optional plots | yes when called by the preceding file |
| direct-grid merge | `merge_genus2_direct_radius_sweeps.py` | Merges directly evaluated low- and high-radius sweeps, retaining one checked `R=1` row | output bookkeeping |
| release export | `export_genus2_free_energy_release.py` | Applies the declared saved-kernel convention migration, distinguishes coarse integral/connected functional/thermal free energy, and writes compact output | yes: final convention conversion |

`extend_genus2_radius_sweep_t_duality.py` is an optional exact T-duality
extension.  It is not used for the final 39-point release, whose radii were
evaluated directly.  `merge_genus2_direct_radius_sweeps.py` is the file used
there instead.

## 2. Exact pointwise-evaluator import closure

Starting from `monte_carlo_integrate_genus2_c1.py`, the complete local-Python
import closure contains the following **31 files**.  This list was obtained
from the source imports, not copied from an older procedure note.

### 2.1 Integrand and normalization

| Source | Responsibility |
|---|---|
| `monte_carlo_integrate_genus2_c1.py` | Production node evaluator and per-node result schema |
| `genus2_c1_string_integrand.py` | Forms `I_2^Xi`, the compact theta sum, the raw/factorization-normalized densities, and the final `K_2^{c=1}` kernel |
| `genus2_integrand_normalization.py` | All explicit constants: `Psi_10` convention, Xi scalar measure, target zero mode, gauge/form coefficient, c=1 sphere-topology factor, stack weight, and saved-convention migration |
| `free_boson_plumbing.py` | Full noncompact scalar plumbing partition, loop Gaussian, Schottky oscillator products, eta functions, and raw even-theta product `Psi_10` |
| `conformal_frame_labels.py` | Prevents mixing theta, glasses, and unit-area Bergman-frame matter partitions |

### 2.2 Liouville and Virasoro blocks

| Source | Responsibility |
|---|---|
| `liouville_genus2_ccy.py` | Three-momentum Liouville integral in the theta/CCY frame |
| `liouville_genus2_glasses.py` | Three-momentum Liouville integral in the glasses frame |
| `ccy_genus2_block.py` | Theta-graph genus-two Virasoro block and central-charge recursion |
| `ccy_genus2_glasses_block.py` | Glasses-frame genus-two Virasoro block |
| `ccy_plumbing_conventions.py` | Literal `q^h` primary and `q^N` descendant sewing conventions |
| `liouville_momentum_quadrature.py` | The `dP/pi` half-line completeness quadrature and tail handling |
| `liouville_torus.py` | `Upsilon_b`, Xi/DOZZ structure constants, conformal weights, and torus one-point data |
| `virasoro_blocks.py` | Torus one-point recursion used transitively by the Liouville layer |
| `genus2_vacuum_blocks.py` | Schottky primitive-class vacuum seed and truncation control |
| `liouville_genus2.py` | Shared genus-two parsing and separating-frame helpers used transitively |
| `liouville_genus2_modular_check.py` | Symplectic-transform definitions imported by the shared Liouville helpers |

### 2.3 Period map, atlas, and certification

| Source | Responsibility |
|---|---|
| `plumbing_algorithms.py` | Schottky generators/products, theta and glasses forward maps, collocation solvers, inverse maps, and residuals |
| `genus2_plumbing_atlas.py` | Searches theta/glasses charts, tracks the `Sp(4,Z)` marking, and chooses the certified frame |
| `audit_q_to_omega_accuracy.py` | Recomputes/refines `q -> Omega` and returns the production period certificate |
| `genus2_period_table.py` | Loads and queries the schema-v3 period-map index |
| `genus2_holomorphic_period_table.py` | Period-table record contract and Schottky validity envelope |
| `genus2_hybrid_period_map.py` | Bulk collocation versus cusp Schottky routing and cross-check policy |
| `genus2_multiprecision_collocation.py` | High-precision rescaled holomorphic-form solver for difficult/mixed cusp nodes |
| `genus2_calibrated_schottky.py` | Guarded finite-word Schottky evaluator |
| `genus2_period_table_grid.py` | Period-table grid/configuration definitions imported by the table layer |
| `genus2_period_table_selector.py` | Atlas-aware selection of table plumbing points |
| `bolza_torus_plumbing_reach.py` | `Sp(4,Z)` period-matrix transform used by the production marking logic |
| `bolza_ccy_recursion.py` | Bolza/CCY recursion helper imported by the atlas |

### 2.4 Moduli-domain and estimator support

| Source | Responsibility |
|---|---|
| `genus2_moduli_physical_mixture_rqmc.py` | Preferred physical-measure proposal and its estimator/diagnostics |
| `genus2_moduli_rqmc.py` | Shared legacy invariant-measure estimator interface still supported by the evaluator |
| `genus2_siegel_fundamental_domain.py` | Exact Gottschling-domain membership and `Vol(F_2)=pi^3/270` control |

No call in the current period-coordinate production path reaches the
`compact_partition.py` or `riemann_surface_tools.py` imports that occur inside
the optional ribbon-graph functions at the bottom of `plumbing_algorithms.py`.
Those two external helpers are therefore not production dependencies of this
Monte Carlo result.

## 3. Period-table construction code

The node evaluator needs a data artifact, not merely Python source:

```text
plumbing/results/genus2_period_table/fundamental_local_merge_v1/assembled/
    period_query_index.npz
    table_fundamental.csv.gz
```

The index is required; the CSV is used for validation/provenance and is passed
in the standard run.  To reproduce these artifacts from scratch, the complete
code path is:

| Source | Responsibility |
|---|---|
| `config/genus2_period_table_cluster.json` | Fixed sampling domain, tolerances, precision tiers, and shard layout |
| `genus2_period_table_grid.py` | Deterministic table manifest |
| `genus2_period_table_selector.py` | Atlas-aware point selection |
| `genus2_period_table_cluster.py` | Sharded forward-map evaluation, checkpointing, certification, and assembly |
| `genus2_hybrid_period_map.py` | Backend routing |
| `genus2_multiprecision_collocation.py` | Dynamically loaded high-precision backend |
| `genus2_table_fundamental_cluster.py` | Reduction of table rows to the fundamental domain with the exact marking |
| `genus2_table_fundamental_reduction.py` | Fundamental-domain reduction algorithm |
| `build_genus2_fundamental_period_index.py` | Builds the schema-v3 nearest-neighbor/query index |
| `validate_genus2_period_index.py` | Rejects stale schema, hash mismatch, malformed markings, and table/index disagreement |

These files reuse the period-map modules in section 2.3 rather than defining a
second forward map.

## 4. Cluster orchestration actually used by the pipeline

The shell and scheduler files do not define mathematical factors, but they do
define the exact array bounds, numerical truncations, dependencies, staging,
and completeness checks.

### 4.1 Moduli/CFT production

```text
cluster/stage_submit_genus2_physical_mixture.sh
cluster/genus2_physical_mixture_cft_array.slurm
cluster/genus2_physical_mixture_assemble.slurm
cluster/pull_genus2_physical_mixture_results.sh
cluster/genus2_low_radius_direct_reweight.slurm
```

The staging script copies the Python sources, validates the period index, runs
the sampler and compact-integrand checks, submits one task per in-domain node,
and makes assembly depend on success of the full array.  The pull script
requires `RUN_COMPLETE.json`; failed nodes are not silently omitted.

### 4.2 Period-table production

```text
cluster/stage_submit_genus2_period_table.sh
cluster/genus2_period_table_array.slurm
cluster/genus2_period_table_assemble.slurm
cluster/genus2_period_table_validate.slurm
cluster/pull_genus2_period_table_results.sh

cluster/stage_submit_genus2_table_fundamental.sh
cluster/genus2_table_fundamental_array.slurm
cluster/genus2_table_fundamental_assemble.slurm
cluster/genus2_table_fundamental_validate.slurm
cluster/pull_genus2_table_fundamental_results.sh
```

The `*_retry_*`, `*_period_recovery_*`, and `*_cft_recovery_*` scripts in
`plumbing/cluster/` are conditional recovery tools.  They are not part of a
successful clean run, but they preserve node identities and use the same
certification/assembly path if a scheduler or difficult node must be retried:

```text
cluster/stage_submit_genus2_period_table_retry.sh
cluster/genus2_period_table_retry_assemble.slurm
cluster/stage_submit_genus2_table_fundamental_retry.sh
cluster/genus2_table_fundamental_retry_array.slurm
cluster/genus2_table_fundamental_retry_assemble.slurm
cluster/stage_submit_genus2_period_recovery.sh
cluster/genus2_period_recovery_array.slurm
cluster/genus2_period_recovery_assemble.slurm
cluster/pull_genus2_period_recovery_results.sh
cluster/stage_submit_genus2_period_branch_recovery.sh
cluster/genus2_period_branch_recovery_array.slurm
cluster/genus2_period_branch_recovery_assemble.slurm
cluster/stage_submit_genus2_cft_recovery.sh
cluster/genus2_cft_recovery_array.slurm
cluster/genus2_cft_recovery_finalize.slurm
cluster/pull_genus2_cft_recovery_results.sh
```

The Python helpers for those conditional paths are
`genus2_period_branch_recovery.py`, `assemble_genus2_period_recovery.py`, and
the ordinary production evaluator/assembler already listed above.

## 5. Tests that should be run before accepting a result

### 5.1 Sampling, domain, assembly, and release

```text
genus2_moduli_physical_mixture_rqmc_checks.py
genus2_siegel_fundamental_domain_checks.py
monte_carlo_integrate_genus2_c1_checks.py
assemble_genus2_c1_rqmc_checks.py
reweight_genus2_c1_radius_checks.py
reweight_genus2_c1_rqmc_radius_checks.py
merge_genus2_direct_radius_sweeps_checks.py
export_genus2_free_energy_release_checks.py
```

### 5.2 Integrand, normalization, scalar, and CFT

```text
genus2_c1_string_integrand_checks.py
genus2_integrand_normalization_checks.py
free_boson_plumbing_checks.py
ccy_plumbing_conventions_checks.py
ccy_genus2_block_checks.py
ccy_genus2_glasses_block_checks.py
liouville_momentum_quadrature_checks.py
liouville_torus_checks.py
virasoro_blocks_checks.py
```

### 5.3 Period map and table

```text
preflight_rqmc_period_map_checks.py
audit_q_to_omega_accuracy_checks.py
genus2_plumbing_atlas_checks.py
genus2_period_table_checks.py
genus2_holomorphic_period_table_checks.py
genus2_hybrid_period_map_checks.py
genus2_multiprecision_collocation_checks.py
genus2_period_table_cluster_checks.py
genus2_table_fundamental_cluster_checks.py
genus2_table_fundamental_reduction_checks.py
build_genus2_fundamental_period_index_checks.py
```

### 5.4 Analytic normalization and factorization audits

These are not imported in the numerical hot path.  They independently test
the conventions used there and are essential to reviewing the derivation:

```text
audit_bry_xi_convention_map.py
audit_bry_xi_convention_map_checks.py
audit_genus2_factor_ledger.py
audit_genus2_factor_ledger_checks.py
audit_genus2_from_genus1_sewing.py
audit_c1_full_separating_factorization.py
audit_c1_sphere_topology_normalization.py
audit_c1_sphere_topology_normalization_checks.py
audit_genus0_one_to_two_amplitude.py
audit_genus0_one_to_two_amplitude_checks.py
audit_free_boson_long_tube_normalization.py
audit_free_boson_long_tube_normalization_checks.py
audit_genus2_hyperbolic_volume_sampling.py
audit_genus2_marked_plumbing_factorization.py
audit_genus2_genus1_pointwise_matching.py
```

The analytic derivation these audits implement is
`xi_c1_normalization_from_scratch.tex` (and its compiled PDF).  In particular,
the intrinsic Liouville momentum `P` is never shifted to manufacture a
plumbing `q` factor; the Liouville and scalar partitions are evaluated in the
same plumbing conformal frame, and the common Weyl anomaly cancels in
`Z_L/(Z_X)^25`.

## 6. Required runtime and data inputs

The current local environment used to review the code is:

```text
Python      3.11.15
numpy       2.4.4
scipy       1.17.1
mpmath      1.3.0
matplotlib  3.10.8   (plots only)
```

`numpy`, `scipy`, and `mpmath` are numerical dependencies of the production
path.  `matplotlib` is optional for diagnostics and plots.  There is currently
no repository lock file, so a fully reproducible rerun should record these
versions in the submission metadata.

The production input set is:

```text
period_query_index.npz                schema-v3 inverse-seed index
table_fundamental.csv.gz              validation/provenance table
production_nodes.csv                  exact nodes and physical-mixture weights
per-node JSON/CSV evaluations         pointwise CFT outputs
```

The source-of-truth output is not a mean over the successful rows.  It is a
mean over **all proposals** in each complete scramble, with out-of-domain
proposals contributing known zero and every in-domain proposal required to
have a certified CFT value.

## 7. Current production instance and normalization boundary

The current design directory is

```text
plumbing/results/genus2_c1_moduli_mc/physical_mixture_R8_C4_M256/
```

It contains 8 independent scrambles, 4 proposal components, 256 proposals per
component and scramble, 8192 total proposals, and 6846 in-domain CFT nodes.
Its exact invariant-volume control is `pi^3/270`; the saved design estimate is
`0.11494121354456449 +/- 0.0003635770486117367`, compared with
`0.11483806177888821` exactly.

The compact release is

```text
output/data/genus2_c1_free_energy_direct_39/
```

The release exporter records explicitly that no fitted worldsheet-to-matrix-
model conversion has been applied.  The production code currently applies the
derived local Xi scalar conversion and c=1 sphere-topology factor.  The broader
BRY/Xi absolute genus-two amplitude dictionary remains marked uncertified in
`genus2_integrand_normalization.py`; the code must not turn that open analytic
comparison into a numerical fit.

## 8. Recommended review order

1. `xi_c1_normalization_from_scratch.tex` and
   `genus2_integrand_normalization.py`.
2. `genus2_c1_string_integrand.py` and its checks.
3. `free_boson_plumbing.py`, the two Liouville wrappers, and their block and
   quadrature dependencies.
4. `plumbing_algorithms.py`, `genus2_plumbing_atlas.py`, and the period
   certificate/table modules.
5. `genus2_moduli_physical_mixture_rqmc.py` and the exact volume control.
6. `monte_carlo_integrate_genus2_c1.py` and
   `assemble_genus2_c1_rqmc.py`.
7. The radius reweight/merge/export files.
8. The independent normalization and factorization audits in section 5.4.

That order follows one numerical contribution from the action and CFT
normalization through the local kernel, period chart, proposal density, stack
quotient, and final reported free energy without using the matrix-model value
as an input.
