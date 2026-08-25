# StringMC snapshot manifest

Snapshot date: 2026-07-22

Source project: `/Users/yutaizhang/Desktop/Project/StringMC`

Destination within this project: `Code/genus_2_cross_channel/`

## Selection rule

The copy follows `docs/genus2_monte_carlo_code_manifest.md`.  It contains the
31-file pointwise-evaluator import closure, all current end-to-end Python entry
points, period-table construction code, focused checks, and the analytic
normalization/factorization audits most relevant to replacing the c=1 matter
system by noncritical Type 0B data.

Nine additional local modules are included because they are imported by the
retained checks and audits, even though they are not in the 31-file production
closure.  The resulting flat Python tree has a complete local import closure.

The channel-specific core is:

- theta/CCY plumbing channel: `ccy_genus2_block.py`,
  `liouville_genus2_ccy.py`;
- glasses channel: `ccy_genus2_glasses_block.py`,
  `liouville_genus2_glasses.py`;
- shared sewing convention: `ccy_plumbing_conventions.py`;
- plumbing-to-period map: `plumbing_algorithms.py`, the
  `genus2_*period*.py` family, the atlas, collocation, Schottky, reduction, and
  index-building modules; and
- Monte Carlo: physical-mixture RQMC sampling, pointwise integration, strict
  assembly, radius reweighting/merging, and release export.

## Data mapping

| Original StringMC path | Copied path | Purpose |
|---|---|---|
| `output/data/genus2_c1_free_energy_direct_39/` | `data/free_energy/` | Compact 39-radius free-energy CSV release and provenance |
| `plumbing/results/genus2_c1_moduli_mc/physical_mixture_R8_C4_M256/` | `data/monte_carlo/` | Production nodes, assembled samples, recovery records, summaries, and radius CSVs |
| `plumbing/results/genus2_period_table/fundamental_local_merge_v1/assembled/` | `data/period_map/` | Validated period table, schema-v3 query index, reductions, hashes, and summaries |

The copied period-map directory retains its original `SHA256SUMS` and
`period_query_index.sha256` files.

## Intentionally excluded

- historical pilot samplers and superseded numerical experiments;
- large raw period-table shard archives, because the assembled table and query
  index are the production artifacts;
- the full scheduler-specific shell/Slurm orchestration and recovery suite
  (only the two period-table guard scripts read by a retained check are
  included under `cluster/`); and
- unrelated StringMC calculations.

These exclusions keep this a modifiable research snapshot rather than a
second copy of the full StringMC repository.  The source files and source data
remain unchanged in StringMC.
