# Branching with shared descendants and sparse generator actions

The production provider constructs the physical `L_{-1}` and `L_1` action
coefficients and solves the same outer and middle Ward equations. All reuse
is within the current computation, at fixed module parameters and precision.
No saved numerical coefficients are loaded to accelerate a fresh run. The
conformal-block and CCY routines are unchanged by these optimizations.

## Current construction

1. **Share descendant suffixes between commuting Virasoro copies.** For
   ordered partitions `A,B`, peel the larger leading mode, construct the
   remaining descendant, and apply the peeled mode last. The copies commute,
   while the relative order within each copy is preserved. Cache each
   intermediate by its primary object and remaining ordered partitions.
   Retain a reference to the primary so its object identity cannot be reused.

2. **Reuse oscillator actions with different spectators.** Store auxiliary
   stress-tensor images by `(mode, auxiliary state)`. For the mixed operator,
   store auxiliary transitions and graded signs by `(mode, auxiliary state,
   physical level)`. Physical supercurrent images have their existing
   separate cache. Compute the fixed coefficients in the two embedded
   Virasoro generators once per module.

3. **Give oscillator states integer identifiers.** Descendant accumulation
   and generator caches use these identifiers rather than repeatedly hashing
   long tuples of oscillator modes. Decode completed columns to the original
   oscillator basis before fitting. This changes storage, not the basis of
   the equations supplied to the fitter.

4. **Remove known phases during descendant construction for real parameters.**
   The phase of a physical oscillator state is `i` raised to its number of
   bosons plus physical fermions, including the Ramond ground bit. In this
   rephased basis the embedded generators are real for real `b,P`: their
   pair terms and momentum terms have exactly the corresponding phase
   differences. Strip the phase from eligible primaries, apply real
   generators, and restore it on completed columns. Eligibility and every
   matrix entry are checked exactly; no small imaginary part is discarded.
   Complex momenta or ineligible primaries retain complex arithmetic.

5. **Reuse sparse generator matrices at machine precision.** For each copy
   and mode, construct a generator column only when its input oscillator
   state is first encountered. Append newly required columns to a cached
   SciPy CSC matrix. Apply it to sparse descendant vectors, visiting their
   occupied input columns. This replaces millions of Python dictionary
   updates with sparse matrix products; it does not construct all states at
   a level. The implementation remains Python using the existing NumPy/SciPy
   dependencies. At arbitrary precision, accumulation instead uses scalar
   `mpf` or `mpc` arithmetic with the integer identifiers; there is no
   conversion of multiprecision descendants to machine precision.

6. **Share minus and plus construction where both are required.** For
   positive `n>1/4`, the neighboring-primary parts of `L_{-1}v_n` and `L_1v_n`
   are descendants of `v_{n-1}`, at degrees `4n-1` and `4n-3`. On the second
   edge, compute the needed plus action before closing the minus cache.
   Release completed minus columns before fitting the plus action. The
   third edge does not construct an unused plus action. Precomputation is
   disabled when the outer and middle precisions differ. Negative labels
   use the existing reflection `(n,P) -> (-n,-P)`.

7. **Transport the second Ramond parity.** Since Theta commutes with physical
   `L_{+/-1}` and both Virasoro algebras, the coefficient taking `n` to `m`
   at parity one is the parity-zero coefficient multiplied by

   \[
   2^{\{(-1)^{2|m|-1/2}-(-1)^{2|n|-1/2}\}/2}.
   \]

   This includes boundary crossings and follows from the raw-primary
   normalization `Theta v_n^0 = -2^{(-1)^(2|n|-1/2)/2} v_n^1`. Both actions
   use this transport, with no second span solve.

Construction caches are cleared when the action or requested minus/plus
pair finishes, including on failure. Only fitted action coefficients persist
for later Ward equations. Module and action providers check precision,
including on cache hits.

## Numerical behavior

The span columns and level cutoffs are unchanged. Native sparse matrix
products change summation order and apply the existing numerical-zero
threshold after a complete generator action, rather than after each scalar
addition. Consequently bitwise equality and identical numerical row support
are not expected. For `L_{-1}v_{11/4}`, the constructed union has 23,842 rows
instead of 23,814, with the same 483 columns. There is no additional row,
level, or column cutoff.

The fitter still requires full column rank and checks the residual on all
rows of the constructed span against the independently constructed target.
The native residual threshold remains `1e-8`. Multiprecision refinement and
its selected-row and full-row tolerances are unchanged. The complex span
fitter itself is unchanged: the real-basis optimization applies only to
descendant construction.

## Fresh measurements

Parameters are `b=7/5`, `P=(11/23,13/29,17/31)`, `p=f=0`, and
`(eta,eta')=(1,-1)`. Individual Ramond-action samples use `P=13/29`.
Times include construction, fitting, and existing diagnostics; they exclude
imports, comparison with saved results, and final serialization. Each timing
starts in a fresh process and reuses only data computed within that process.
The previous column is the immediately preceding implementation, which
already shared suffixes, auxiliary actions, and minus/plus construction.

| Work | Previous | Current | Speedup |
| --- | ---: | ---: | ---: |
| Machine `L_{-1}v_{9/4}` | 1.75 s | 1.10 s | 1.59x |
| Machine `L_{-1}v_{11/4}` | 44.43 s | **19.58 s** | 2.27x |
| Machine paired `L_{-1}v_{11/4}`, `L_1v_{11/4}` | 45.56 s | **21.49 s** | 2.12x |
| Machine outer + middle branching, total level 10 | 7.64 s | **5.26 s** | 1.45x |
| 40-digit outer + middle branching, total level 5 | 4.42 s | **3.71 s** | 1.19x |

The level-ten outer and middle stages take 5.2499 and 0.00560 seconds.
Preparing the needed middle actions occurs during the outer stage, so their
combined time is the meaningful comparison. In the paired high-label sample,
the additional plus construction takes 1.79 seconds. Peak resident memory
for that process falls from 2,646.8 MiB to 1,979.4 MiB (about 25%).

The original small-action profile found 4.17 million calls to scalar sparse
accumulation. For the current high-label paired action, 884 sparse generator
applications reuse 20 matrices containing 5,423,422 entries. All 668 requested
span columns are retained. Compact identifiers alone reduced the small
native action from 1.75 to 1.08 seconds; sparse products provide their larger
benefit on the high-label action. Individual timings are single samples,
not statistical estimates of variability.

## Targeted validation

Saved numerical results are loaded only after each timed computation.
Scaled differences below mean `abs(new-old)/max(1,abs(old))`.

| Comparison with the previous implementation | Coefficients | Maximum scaled difference |
| --- | ---: | ---: |
| Machine high-label minus action | 483 | `1.89e-12` |
| Machine high-label plus action | 185 | `1.05e-13` |
| Machine level-ten outer branching | 1,936 | `5.53e-9` |
| Machine level-ten middle branching | 36 | `9.64e-15` |
| 40-digit level-five outer branching | 880 | exactly identical saved values |
| 40-digit level-five middle branching | 28 | exactly identical saved values |

The high-label minus and plus full-row relative residuals are `1.51e-11`
and `2.76e-12`, with full ranks 483 and 185. Comparisons measure agreement
between numerical calculations, not certified errors against exact answers
or the accuracy of a recovered superconformal block. Raw outer coefficients
can have large normalizations: the largest absolute outer difference is
139.6, while the maximum scaled difference is the `5.53e-9` above.

Five small additional action comparisons cover NS, the other Ramond parity,
the Ramond boundary label, and complex-momentum fallback at machine and
40-digit precision. They agree within `1e-14` at machine precision and
exactly at 40 digits; cache cleanup is checked as well. No PBW block or full
physical-block comparison was run.

## Level-fifteen branching estimate

The actual label sets add `n=+/-11/4` on each Ramond edge and `n=+/-5/2`
on the NS edge between levels ten and fifteen. There are two new second-edge
paired constructions and two third-edge minus constructions. Using the
measured current actions and retaining the saved 0.94-second NS sample gives

\[
5.26+2(21.49)+2(19.58)+2(0.94)\simeq89.3\ \mathrm{s}.
\]

This updates the same sampling estimate from 189.5 seconds to **about
1.5 minutes at machine precision**, before additional high-level outer Ward
overhead. Sign and momentum dependence can also change the actual runtime.
It is not a complete level-fifteen measurement or a full-pipeline estimate.
No level-fifteen run was performed. These native sparse-product gains do
not establish a level-fifteen 40-digit runtime.

## Implementation and saved evidence

The new storage, phase, prefactor, and sparse-product changes are in
`action_optimization.py`. Existing sharing is connected through
`middle_branching.py`, `outer_branching.py`, and the precision-matched
precomputation option in `pipeline.py`.

Reproducible drivers are `benchmarks/branching_reuse.py`,
`benchmarks/branching_pipeline.py`, and
`benchmarks/validate_branching_storage.py`. The action driver supports
`--tuple-states`, `--complex-descendants`, and `--scalar-native` to isolate
storage and arithmetic choices, and `--with-plus` to time a paired action.

Current evidence is in `results/branching_sparse_*.json`,
`results/branching_indexed_level5_40dps.json`,
`results/comparison_branching_sparse_plus_n11over4_machine.json`, and
`results/validation_branching_storage.json`. The current source hashes,
timings, estimate, and evidence list are recorded together in
`results/branching_sparse_summary.json`. Earlier measurements remain saved.
