# Completion audit: current Ramond fermion-insertion pipeline

Audit snapshot: September 11, 2026. **The overall goal remains incomplete.**
The current Schottky-vacuum implementation has completed N5 production and
passed both required validations. Its full N5 process takes 12.6089 seconds.
The active constant-reuse N10 run still uses the preceding Gaussian backend;
a current Schottky N10 run has not yet been launched.

This audit uses source inspection, saved metadata and SHA-256 manifests.
It imports no production module and computes or compares no block coefficients.

## Scope and user amendments

The required pipeline is: derive and recursively compute the middle
\(\mathbb B(n',1/2,n;\alpha,1-\alpha)\) from the physical \(L_1\) Ward
identity; combine recursive outer/middle coefficients with genuine CCY
central-charge recursion for the enlarged \(\Theta v_{1/2}\) block; compute
the auxiliary; and recover the physical \(\eta\eta'=-1\) superconformal
block by convolution division. The user explicitly replaced the original
Ising-recursion auxiliary requirement with **direct fermion sewing**, while
retaining CCY for the enlarged block and local ingredients usable at arbitrary
genus. The implemented assembler is the specified subdivided theta graph.

The only prescribed numerical validations are the final physical coefficients
against independent PBW through total level five and final evaluations at
several split positions with fixed product. Profiling and timing reruns add
no coefficient-comparison suite. All four-variable terms must be retained;
sector projection or removal of unequal split powers must not force success.
After validation, a complete physical N10 timing remains required. Separate
detailed Machine Notes and adversarial mathematical/presentation reviews are
also required. The existing draft and Human Notes remain outside this work.

The zero/tolerance reuse and N5 retiming request was completed. The user
then required CCY Eq. (5.5) for the universal vacuum factor; that Schottky
construction and its current N5 validation are complete. The user separately
authorized a timing crossover for individual Virasoro CCY versus Virasoro
PBW blocks, which is distinct from the full physical-SCA benchmark.

## Requirement-by-requirement status

| Requirement | Status and evidence |
| --- | --- |
| Tube geometry and spin convention | Established in `Machine Notes/theta_fermion_ramond_recovery.tex`: `q2=qL*qR` plus the separately specified transported spin frame. |
| Human external-state normalization | The notes use the irreducible vacuum relation, `v_(1/2)(Q/2)=Q*psi`, and explicit `Q!=0`/generic embedding assumptions. |
| Full double-Virasoro decomposition | The notes derive the four-cut contraction, all four norms, normalized middle coefficient, fusion support and native-to-human contour phase. |
| Middle primary recurrence | Implemented in `middle_branching.py`, using both physical-L1 orientations, ground anchors and reusable action coefficients. Reviewed in `FINAL_BRANCHING_REVIEW.md`. |
| Recursive outer coefficients | `outer_branching.py` retains action-closed support, low anchors, full rank and all-row residual guards. Reflection and Theta parity transport are derived and reviewed. |
| Genuine CCY blocks | `punctured_ccy.py` retains the global seed, all admissible null residues, fixed external weight and unresolved-pole errors; it does not substitute finite-c Gram contractions. |
| Full direct auxiliary | `direct_fermion.py` includes zero/nonzero modes and both split orientations. Complete standalone level-ten output is saved in `results/direct_fermion_level10.json`. |
| Arbitrary-genus flavor | Local Fock spaces, forms, pairings and insertion matrices are graph-independent ingredients. Notes explicitly retain graph-dependent cocycles, insertion choices and sector identities; only the stated theta assembler is claimed. |
| Detailed convolution and inverse | Notes derive the physical identity factor, parity cocycle, sector identity and triangular division. `series_algebra.py` retains all exponents and raises on a failed sector residual. |
| Universal normalization | Current production records `universal_seed_power_after_recovery=2`: divide the twice-reduced numerator by the full auxiliary, then restore two universal factors. |
| Schottky universal factor | `schottky_vacuum.py` implements CCY Eq. (5.5) as an exact primitive graph product. `SCHOTTKY_VACUUM_REVIEW.md` independently audits the sewing maps, inverse-class multiplicity and degree cutoff. |
| Individual Virasoro timing crossover | In progress under the user's separate authorization; not a comparison of full physical-SCA pipelines. |
| Speed improvements | Implemented and statically reviewed. The latest nearby archived baseline is 19.7228 s versus 12.6937 s with constant reuse, about 1.55 times faster. |
| Final-current-source PBW check through N5 | **Passed**, `validation_ccy_schottky_level5.json`: 4,648 slots; maximum scaled error `3.836489189392701e-52`. |
| Final-current-source split check | **Passed** in the same report: 24 evaluations covering three split positions and all eight spin signs; maximum scaled variation `1.665633005810069e-58`. |
| Complete current-source N10 block and total runtime | **Pending for Schottky source.** The preceding `ccy_cached_actions_level10` run completed; the active `ccy_reused_constants_level10` still uses its imported Gaussian backend. No Schottky N10 run has launched. |
| Detailed separate LaTeX notes | Present with current N5 implementation, validations, timing and nearby baseline. Parent records a clean 17-page compilation. Final N10 entry remains pending. |
| Adversarial mathematical review | Core reviews and independent optimization reviews are complete; see the review list below. |
| Adversarial presentation review | Core and optimization/N5 debates and revisions are recorded in `FINAL_PRESENTATION_REVIEW.md` and `FINAL_PRESENTATION_DIRECT.md`. Current timing prose is updated; final N10 prose remains pending. |

## Current numerical evidence and timer scopes

Authoritative current files:

- `results/ccy_schottky_level5.json`.
- `results/validation_ccy_schottky_level5.json`.
- `results/ccy_schottky_level5_walltime.json`.

The production's literal status remains `computed; validation pending` because
validation is saved separately. The corresponding validation says `passed`,
and the wrapper records successful exit. Parameters are `b=7/5`,
`P=(11/23,13/29,17/31)`, `p=f=0`, `(eta,eta')=(1,-1)`, with 70-digit outer
Ward arithmetic, 100-digit middle/CCY arithmetic, and exact rational auxiliary
sewing before normalization. The independent reference uses 384-bit FLINT
and is `pbw_level5_p0_f0_eta_plus_minus.json`.

The current physical output contains 581 four-variable vectors. Validation
compares all 4,648 parity slots, including unequal split powers expected to
vanish. Its split evaluations use `u=0.5,1,2` at
`q1=0.013`, `q2=0.007`, `q3=0.009`, for all eight spin signs. Both errors are
below the requested `1e-8` tolerance; they are measured differences, not
interval bounds or a claim of all-parameter numerical coverage.

| Current N5 measurement | Seconds |
| --- | ---: |
| Full child-process elapsed time | 12.608884041997953 |
| Internal computation | 12.240286874999583 |
| Outer branching | 5.731198166999093 |
| Action preparation, included in outer branching | 5.308157999999821 |
| Middle recurrence | 1.0206773309728305 |
| Virasoro block products | 4.571471287981694 |
| Direct auxiliary | 0.13050362499780022 |
| Division and two restored vacuum factors | 0.370597249999264 |
| Schottky factor alone, included in the preceding row | 0.0013366249986574985 |

The run records 296 branching cases, 300 actual ordinary series, 146 reused
transposed products and 24 middle recurrence records. Peak memory is
114.765625 MiB on Python 3.14.3, Darwin arm64. The internal timer excludes imports,
argument parsing and final physical encoding/save; the child-process clock
includes all imports and result writes through exit. They are not
interchangeable clocks.

The nearby archived-source baseline is
`ccy_pre_constants_retimed_level5.json` with its `_walltime.json` companion:
19.32923983300134 s internally, 19.722815040997375 s in full, and
11.730626958997163 s for outer branching. It changes only the imported
`compute_target` source to its archive and records that actual provenance.
It is a timing-only run, not another coefficient comparison. Both nearby runs
were concurrent with the old N10 process. The older 89.8903 s N5 measurement
is retained as history and is not used to attribute the constant-reuse gain.

Direct physical PBW was separately retimed at 1.4991441250022035 s for the
full process. Its backend, precision, internal clock and three-variable
representation differ; the current implementation comparison favors PBW by
about 8.4 times at N5 without establishing asymptotic or precision-matched
scaling. `LEVEL5_TIMING_COMPARISON.md` records these qualifications.

The standalone direct auxiliary N10 benchmark remains 0.186281209000299 s
for construction and 0.2902146670003276 s through its initial save. It stores
4,793 nonzero vectors and 9,586 rational parity entries. It is not the complete
physical-block timing.

The standalone Schottky vacuum benchmark is `schottky_vacuum_level10.json`: 
0.04323433299941826 s, six inverse-paired primitive classes, and 70 exact
coefficients. It excludes imports and result writes. Its recorded source
digest matches `schottky_vacuum.py`. This is a vacuum-only timing, not a
physical N10 block or another coefficient comparison.

## Source provenance and independent reviews

All 18 current production hashes and all 18 copied validation hashes match
their current files. The new `compute_target.py` digest is
`3368f697c713860f9c2a8619a29c8a5e1113da54240337dc1ebc65a846390c42`.
The archived pre-constant digest is
`46ebb1a8bd10b356ee8697637f61ffb62b680258858e95aabbb69ad68b6568ca`, matching
the preceding `ccy_cached_actions_level5` manifest. Later changes to
`pipeline.py` and `punctured_ccy.py`, and the new Schottky source, also
distinguish current production from these historical manifests.

At the archived-baseline audit, all 18 entries matched their named files.
The subsequent Schottky update changes the live pipeline/CCY sources, so
that manifest is now historical. Its driver
loads the archived source as `sys.modules['compute_target']` before importing
production; ordinary imports consequently bind to that module. Its only
pipeline override substitutes the actual archive and driver paths in the
manifest. The archive's different `HERE` affects only unused standalone CLI
output defaults, not the production calculation. No saved coefficients are
inputs to either timing. This was verified by source/metadata inspection,
without running or importing production.

The preceding N10 process imported the old module before the disk edit and
never reloaded it. It completed in 1189.8001477500002 s internally and
1191.6387245000005 s from launch through exit. Its captured manifest belongs
to that preceding implementation; the inherited source must be resolved
against its archived old hash. The `ccy_reused_constants_level10` run is
now active but imported the old Gaussian factor before the Schottky change.
Its eventual result is also historical for the vacuum backend. No full
Schottky N10 measurement is yet available.

The source manifests name the specified local/inherited files; they are not
claimed to hash every Python/environment dependency. Both hashes in the
standalone direct auxiliary benchmark still match its unchanged sources.

Mathematical and implementation audits:

- `FINAL_MATH_REVIEW.md`, `FINAL_BRANCHING_REVIEW.md`, and
  `REVIEW_DIRECT_FERMION_SIGNS.md` establish insertion, normalization,
  recurrence, convolution and sector identities.
- `CCY_SPEED_REVIEW.md` and the independent audit in
  `ASSEMBLY_SPEED_REVIEW.md` cover Kac/fusion/transition reuse, global seeds,
  exact charge interning, real/complex promotion, graph transposition and
  degree-aware triangular operations.
- `OUTER_SPEED_REVIEW.md`, `ACTION_SPEED_REVIEW.md`, and independent
  `ACTION_SPEED_STATIC_AUDIT.md` cover Ward support/factorization, action
  parity transport, ordered suffixes, generator images, precision and
  all-column/all-row residual guards.
- `SCHOTTKY_VACUUM_REVIEW.md` independently derives and reviews CCY Eq. (5.5),
  the theta sewing maps, primitive/inverse classes, exact multipliers and
  complete degree cutoff. No Gaussian-component comparison was performed.
- `CONSTANT_REUSE_REVIEW.md` independently checks exact immutable zero,
  precision/rounding-aware tolerance caching, unchanged summation/pruning,
  and old/new run provenance.
- `FINAL_PRESENTATION_REVIEW.md` records the later debate with this auditor:
  retain detailed derivations but omit low-level cache counters from the
  notes, explicitly scope parity reuse and precision, normalize columns,
  and distinguish measured stages and clocks. The applied prose was reread.

## Remaining completion evidence

1. Complete and inspect the required N10 physical calculation and its actual
   total/runtime artifacts for the Schottky implementation. The completed
   old-source run and active Gaussian-backend run are historical; no
   Schottky N10 run has started in this snapshot.
2. Document the final N10 parameter/precision/support, seed power, stage
   timings, memory, environment and source provenance. Do not substitute a
   partial-stage, standalone-auxiliary or N5 timing.
3. Update the notes, README and this audit from terminal evidence; review
   the final N10 presentation and compile after the last edits.

4. Complete the separately authorized individual-Virasoro timing crossover
   and report its actual timing/precision scope.

No overall completion is asserted while this evidence is pending.
