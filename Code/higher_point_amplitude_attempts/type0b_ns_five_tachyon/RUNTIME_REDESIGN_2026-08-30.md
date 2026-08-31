# Five-point c-recursion runtime redesign

The redesign is implemented and tested. At completion of the initial design
review no replacement job had been submitted; see the submission update below.
No increase in cluster memory has been requested.
The worldsheet amplitude remains unfrozen; no matrix-model comparison was used.

## Unchanged numerical problem

The runtime uses the same fixed-weight c recurrence, channel selection,
physical `+i epsilon` prescription, degree-zero subtraction forest, and collar
radii `(0.01,0.005,0.0025)`. The production quadrature remains `(6,7)` per
unit momentum panel. The configured cutoff remains **twice-level** `(8,8)`,
total 16: physical descendant levels up to `(4,4)`. No convergence upgrade is
claimed by the runtime changes.

## Changed execution design

- Compile final c-series coefficients independently of moduli, then release
  the recursive scratch dictionary. A fixed-parity block at this cutoff
  needs at most 25 final coefficients.
- Evaluate the final polynomial by Horner contraction at the original
  mpmath precision, preserving the explicitly supplied complex logarithms.
- Bound the resident block cache at 2,048 objects and each auxiliary cache at
  4,096 entries. Use a source/parameter-addressed, losslessly serialized
  per-shard SQLite store for final coefficients. A single-writer lock protects
  each store; no cross-shard shared WAL is used.
- Atomically checkpoint completed bulk samples, complete subtracted face
  samples, and corner orbits. Validate source/config/seed identity on restart.
  A failure can lose an in-progress sample, but no longer loses all finished
  samples in the shard. Collar certificates can be recomputed on restart.
- Stream progress and peak RSS/cache counters to the job logs instead of
  retaining stdout in an in-memory buffer.

## Measured local benchmarks

Both comparisons ran in separate fresh processes with allocation tracing.
Their deliberately small **16-block** resident cache stress-tests eviction;
the production cap is 2,048, so the measured RSS is not a production-memory
prediction.

| Workload / metric | Original | Redesigned |
| --- | ---: | ---: |
| 96 production momentum pairs: retained Python memory | 46.57 MiB | 0.387 MiB |
| Same workload: process peak RSS | 215.81 MiB | 95.17 MiB |
| Same workload: retained recursive states | 42,336 | 0 |
| Same workload: cold first pass | 115.41 s | 137.52 s |
| Same workload: second pass | 0.68 s | 0.27 s |
| One full integrand point, reduced `(2,3)` quadrature: peak RSS | 205.70 MiB | 97.77 MiB |
| Same integrand point: cold evaluation | 140.83 s | 138.44 s |
| Same integrand point: second evaluation | 1.011 s | 0.412 s |

The table benchmark retained about 0.36–0.39 MiB as the distinct-block count
grew from 16 to 96. All 96 tables were reloaded on the second pass with no
coefficient recompilation. Both benchmark comparisons produced identical
binary64-reported values. These timings include tracing and local disk I/O;
the table workload's slower cold pass is an explicit persistence tradeoff.

Raw records are in `Data Set/type0b_fivepoint_bounded_memory_tables_20260830.json`
and `Data Set/type0b_fivepoint_bounded_memory_integrand_20260830.json`.

## Validation and remaining gate

All 88 regression tests pass. Tests compare old/new polynomials below absolute `1e-40` at 45-digit working
precision, including parity sectors, lower/total cutoffs, selected leading
terms, and conjugate branch lifts. They check exact high-precision disk round
trips, cache bounds, concurrent-writer rejection, failure cleanup, and an
interrupted integral followed by sample-exact restart. A separate command-line
smoke calculation successfully reused all eight completed samples with zero
CFT recomputation on restart.

The cluster scripts and config now include the runtime tests, persistent
tables, checkpoints, and progress diagnostics. The existing 12-hour wall-time
cap and memory allocation are unchanged. Before a full production resubmission,
run a short pilot with production momentum orders, all boundary components,
and the production cache cap; use its RSS, disk-hit rate and sample timings
to select the memory request and estimate runtime. No such full-production
pilot has yet been certified.

## User-requested production rerun

On 2026-08-30, following the request to reuse saved data and rerun on Cannon,
all 88 tests passed again in the cluster runtime. Array `43044857` (four shards)
and dependent reducer `43044858` were submitted. All shards were confirmed
running with the unchanged 12-hour and 12 GB limits. The new run root is
`/n/home09/yutaizhang/Type0B-Matrix-runs/type0b-ns-fivepoint-ea-quarter-c-bounded-20260830-v3`.

The preceding two c runs left no completed shard, sample checkpoint, or c
coefficient table. The older h run retained 1,736 fitted tables, but these
were not validated as c-recursion inputs and were not imported. All previous
artifacts remain untouched. Physics and Sobol seeds are unchanged; numerical
production artifacts reused: zero. The persistent databases and checkpoints
created by this rerun can be resumed using this same staged source/config.
See `Data Set/type0b_ns_five_tachyon_bounded_memory_reuse_audit_20260830.json`
and `Data Set/type0b_ns_five_tachyon_c_recursion_bounded_memory_submission_v3.json`.
