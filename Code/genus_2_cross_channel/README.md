# Genus-two computational snapshot

This directory is a frozen copy of the parts of StringMC that are relevant to
the planned noncritical Type 0B genus-two calculation.  It is deliberately a
copy: the original StringMC implementation remains in place, while subsequent
Type 0B changes should be made here.

The modules retain their original flat layout because the production scripts
support direct, same-directory imports.  Run scripts and checks from this
directory unless a command explicitly says otherwise.

## Included mathematical layers

### Genus-two Virasoro blocks

- `ccy_genus2_block.py` and `liouville_genus2_ccy.py`: theta-graph/CCY
  (plumbing) channel recursion and the associated three-momentum Liouville
  integral.
- `ccy_genus2_glasses_block.py` and `liouville_genus2_glasses.py`: glasses
  channel recursion and Liouville integral.
- `ccy_plumbing_conventions.py`: the literal primary `q^h` and descendant
  `q^N` sewing convention.
- `liouville_torus.py`, `virasoro_blocks.py`, and
  `liouville_momentum_quadrature.py`: shared structure constants, lower-genus
  recursion, and momentum integration.

### Plumbing coordinates to the period matrix

- `plumbing_algorithms.py`: forward theta/glasses plumbing maps, Schottky
  products, collocation solvers, inverse maps, and residuals.
- `genus2_plumbing_atlas.py`: chart search, marking, and frame selection.
- `genus2_hybrid_period_map.py`, `genus2_multiprecision_collocation.py`, and
  `genus2_calibrated_schottky.py`: bulk/cusp routing and difficult-node
  refinement.
- `genus2_period_table*.py`, `genus2_table_fundamental_*.py`, and
  `build_genus2_fundamental_period_index.py`: construction and validation of
  the indexed period-map table.

The production period data are under `data/period_map/`.  In particular, the
pointwise evaluator uses `period_query_index.npz`; the compressed
`table_fundamental.csv.gz` is retained for validation and provenance.

### Genus-two Monte Carlo calculation

The active path is

1. `genus2_moduli_physical_mixture_rqmc.py` for the four-component scrambled
   Sobol proposal and fundamental-domain weights;
2. `prepare_genus2_rqmc_production.py` and
   `preflight_rqmc_period_map.py` for scheduling and period-map certification;
3. `monte_carlo_integrate_genus2_c1.py` for the pointwise CFT and string
   integrand;
4. `assemble_genus2_c1_rqmc.py` for complete-scramble assembly;
5. `reweight_genus2_c1_rqmc_radius.py` and
   `merge_genus2_direct_radius_sweeps.py` for the radius curve; and
6. `export_genus2_free_energy_release.py` for the final convention conversion
   and compact release.

The corresponding production run is copied to `data/monte_carlo/`.  The small
39-radius release tables are separately exposed in `data/free_energy/`.

## Conventions and provenance

Read `docs/genus2_monte_carlo_code_manifest.md` first.  It identifies the
exact pointwise import closure and distinguishes active production code from
historical experiments.  The convention and normalization trail is recorded
in `docs/bry_xi_convention_map.md`,
`docs/xi_c1_normalization_from_scratch.tex`, and
`docs/genus2_free_energy_procedure.md`.

This snapshot is a bosonic/c=1 numerical foundation, not yet a Type 0B
genus-two implementation.  In particular, the supermoduli measure, spin
structure sum, PCO prescription, and vertical-integration terms must be added
before interpreting a modified output as the Type 0B free energy.

## Environment and smoke checks

The recorded production environment used Python 3.11.15 with the versions in
`requirements.txt`.  A small first check set is:

    python3 ccy_genus2_block_checks.py
    python3 ccy_genus2_glasses_block_checks.py
    python3 genus2_period_table_checks.py
    python3 genus2_moduli_physical_mixture_rqmc_checks.py
    python3 monte_carlo_integrate_genus2_c1_checks.py

See `SNAPSHOT_MANIFEST.md` for the exact source-to-copy mapping and exclusions.
