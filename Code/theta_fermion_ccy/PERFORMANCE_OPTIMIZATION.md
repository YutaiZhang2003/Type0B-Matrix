# Performance work, September 11, 2026

The user requested optimization before repeating the physical level-ten
benchmark. The earlier level-ten process was confirmed absent, its exec
handle was missing, and no completed output file existed. Its last known
outer-stage time was 855.083 seconds; it supplies no full-block runtime.

## Is the large-central-charge seed the bottleneck?

The universal Gaussian vacuum factor is restored after the reduced CCY
sum, so it was not responsible for the long CCY stage. The separate,
weight-dependent global SL(2) seed does contribute to that stage.

`profile_ccy.py` profiled one actual ordinary Virasoro series, at 100 decimal
digits with remaining weighted descendant cutoff 16 and labels
`(-1/2,-1/4,1/4,-1/4)`, using the first Virasoro copy of the production
benchmark. It constructed 1,485 coefficients. No coefficient comparison
was performed in either profiling run.

| Profiled component | Original implementation |
|---|---:|
| Complete profiled series | 87.851091250 s |
| Global SL(2) seed, including its callees | 18.802871554 s |
| Pole construction, including fusion products | 13.276184884 s |
| Fusion products, already included in pole construction | 7.930558873 s |

Thus the seed accounted for about 21% in this example. The larger cost was
recursion arithmetic and repeated transition handling. These are profiler
measurements for this specific branch, not full physical-block timings.

The first CCY optimization reduced the same profiled calculation to
31.865102917 seconds (about 2.76 times faster). Its global seed cost was then
16.273395716 seconds, motivating the subsequent real-arithmetic and
normalized-vertex reuse. The final implementation includes those additional
changes; it is validated through the complete physical pipeline below,
rather than through additional component comparisons.

The records are `results/profile_ccy_baseline_cut16.json` and
`results/profile_ccy_optimized_cut16.json`, with raw `.prof` files alongside
them. The original profiled source is preserved in
`profiling_baseline/punctured_ccy.py`; its SHA-256 matches the baseline
profile. That directory is an archival profiling input, not production code.

## Exact changes

- Cache Kac pole geometry by its null-edge weight, fusion factors by their
  adjacent weights, and transition multipliers independently of remaining
  descendant levels. Pair opposite fusion factors algebraically.
- Reuse the global seed's three-point core and normalized vertex factors;
  retain real arithmetic until complex promotion is necessary. Return the
  seed directly when no null-state residue is admissible.
- Reuse the scalar Virasoro-block product for the reverse split orientation
  by transposing its split exponents. Retain both sets of branching factors.
- Reuse outer Ward matrices across eta values, retain only action-closed
  requested primary triples, and accumulate sparse multiprecision residuals.
- Obtain the second Ramond action parity by the exact Theta intertwiner.
- Reuse descendant suffix states and embedded-generator images of oscillator
  basis states within each action solve. Normalize sparse span entries once
  before refinement, and release the temporary caches afterward.
- Group series terms by degree; propagate solved quotient coefficients
  forward using the same triangular recurrence and unchanged sector guard.

No plumbing cutoff or arithmetic precision is lowered. Unequal split-edge
powers are retained, and no projection is used to force physical invariance.
The detailed algebra and independent static reviews are in
`CCY_SPEED_REVIEW.md`, `OUTER_SPEED_REVIEW.md`, and
`ASSEMBLY_SPEED_REVIEW.md`, `ACTION_SPEED_REVIEW.md`, and
`ACTION_SPEED_STATIC_AUDIT.md`.

## Validation and complete timing

The first optimized full level-five run is saved as `ccy_fast_level5.json`.
Its two authorized checks pass in `validation_ccy_fast_level5.json`:
maximum scaled PBW difference `3.83649e-52` and fixed-product split variation
`1.66563e-58`. This run exposed action construction as the remaining cost:
83.11896 seconds, versus 16.29905 seconds for the Virasoro-block stage.
It predates the final descendant-suffix and action-span optimization.

The preceding optimized N5 result is `results/ccy_cached_actions_level5.json`.
`results/validation_ccy_cached_actions_level5.json` records both prescribed
checks as passed: maximum scaled PBW difference
`3.836489189392701e-52`, and fixed-product split variation
`1.665633005810069e-58`. Its full process time is **89.890251917 s**, with
**88.264166334 s** on the internal computation timer. The outer stage took
56.984985208 s, including 55.415058834 s of action preparation. Every one of
the 17 recorded source hashes matches the implementation imported by the
subsequent `ccy_cached_actions_level10` run. This is now a prior implementation:
the later constant-reuse edit below changes the inherited `compute_target.py`.
The archived old source matches that manifest. Earlier N5 variants are also
retained with their own manifests.

That full N10 run was launched after these passes and subsequently completed
in 1189.8001477500002 s internally and 1191.6387245000005 s from launch
through exit, using the preceding implementation.
Its designated files are `results/ccy_cached_actions_level10.json` and
`results/ccy_cached_actions_level10_walltime.json`.
The `run_timed_pipeline.py` wrapper records elapsed time from child-process
launch through exit, including imports and all result writes. The pipeline
also retains its more narrowly scoped internal computation timer. Neither
the profiled branch nor the standalone fermion benchmark substitutes for
the complete physical-block measurement.

## Subsequent user request: reuse constant zero and tolerance values

The user requested the small sparse-accumulation optimization directly and
asked for a fresh complete N5 timing. The change reuses an exact immutable
complex zero and caches the original tolerance expression using the target
decimal precision, active binary precision and rounding as its key. It changes
neither the arithmetic/pruning rule nor the operation order in `add_term`.
`CONSTANT_REUSE_REVIEW.md` records an independent static audit. The old
`compute_target.py` is retained as
`profiling_baseline/compute_target_pre_constants.py`.

The new result is `results/ccy_reused_constants_level5.json`; its child-process
timer is `results/ccy_reused_constants_level5_walltime.json`. The measured
complete process took **12.693745042000955 s** and the internal computation
took **12.246520542001235 s** at the unchanged benchmark parameters, 70-digit
outer precision, 100-digit middle/CCY precision, direct auxiliary backend,
and full four-variable cutoff. It retains 581 coefficient vectors and records
seed power two, 296 branching cases, 300 actual ordinary series, 146 reused
transposed products and 24 middle recurrence records.

| New recorded stage | Seconds |
| --- | ---: |
| Outer branching | 5.71879374999844 |
| Action preparation, included in outer branching | 5.292387083001813 |
| Middle recurrence | 1.0367864490181091 |
| Virasoro block products | 4.577204588022141 |
| Direct auxiliary | 0.11444854100045632 |
| Division and restoration of two vacuum factors | 0.3801997089976794 |

A nearby fresh-process rerun of the archived pre-constant source took
**19.722815040997375 s** in full, or **19.32923983300134 s** internally.
Compared with the new **12.693745042000955 s** process, this is about a
**1.55-fold** speedup. Outer branching improved from **11.730626958997163 s**
to **5.71879374999844 s**, about **2.05-fold**. Its unchanged CCY stage took
4.546881951991963 s before and 4.577204588022141 s after. The earlier
89.8903 s run is retained as history, not used to attribute the improvement
to constant reuse. Both nearby retimings ran concurrently with the prior
N10 process; these are single-run measurements, not an isolated scaling
study. The current N5 process remains slower than the separately measured direct
physical PBW process (1.4991 s), by about a factor of 8.5 in full-process time.
The PBW comparison uses a different arithmetic backend and precision; see
`LEVEL5_TIMING_COMPARISON.md` for the exact timer boundaries and qualifications.

The parent repeated only the two prescribed physical-output checks.
`results/validation_ccy_reused_constants_level5.json` reports both passed:
maximum scaled PBW difference `3.836489189392701e-52` and maximum
fixed-product variation `1.665633005810069e-58`, across 4,648 parity slots
and 24 split evaluations. A separate read-only metadata/hash inspection
confirmed that all 17 production and copied-validation hashes match the
current files. It performed no conformal-block computation or comparison.
The new inherited-source hash is
`3368f697c713860f9c2a8619a29c8a5e1113da54240337dc1ebc65a846390c42`;
the archived old hash is
`46ebb1a8bd10b356ee8697637f61ffb62b680258858e95aabbb69ad68b6568ca`.

The completed `ccy_cached_actions_level10` process imported the old source
and captured the old manifest before the constant edit; it never reloaded
the new function definitions. Its timing therefore belongs to the preceding
implementation. A fresh `ccy_reused_constants_level10` run is now active,
with its result and `_walltime.json` companion pending. No completed
updated-source level-ten measurement is claimed here.

The nearby baseline is saved in `results/ccy_pre_constants_retimed_level5.json`
and its `_walltime.json` companion. Its driver
`profiling_baseline/run_pre_constants_level5.py` preloads only the archived
`compute_target` module before importing the unchanged production pipeline.
It changes the provenance function to record that actual archive and its own
driver path; it does not change the mathematics or compare coefficients.
Independent static inspection verified the import binding, checked that the
archive's changed `HERE` affects only unused standalone output defaults, and
matched all 18 recorded hashes. The current-source N5 result still matches
all 17 current-source hashes. See `LEVEL5_TIMING_COMPARISON.md` for the full
baseline provenance and timer scopes.
