# Faster machine-precision level-ten CCY pipeline

The optimized complete pipeline took **61.90 s**, versus **92.57 s** for the saved original run: a **1.496-fold speedup**, or **33.13% less time**. Both are fresh-process machine-precision computations from empty coefficient caches through total physical level ten. Imports, branching construction and reuse, the direct fermion factor, Schottky vacuum factor, convolution, result writes and process exit are included.

Parameters: b=7/5, P=(11/23,13/29,17/31), p=f=0, (eta,eta')=(+,-).

| Measurement | Original | Optimized |
| --- | ---: | ---: |
| CCY evaluation | 77.430810 s | 46.573809 s |
| Complete process | 92.569641 s | 61.898062 s |
| Internal pipeline timer | 92.061964 s | 61.448477 s |
| Peak resident memory | 451.125 MiB | 425.500 MiB |

These are single-run measurements. The separately saved direct physical PBW machine-precision run took 33.550881 s, so it remains faster at this cutoff.

## What changed

The four-edge level-lowering lists are cached globally because they depend only on integer levels. Instead of rebuilding them at every recursion state, this run constructed 3,130 lists and reused them 22,125,946 times. A table of 3,146 exact integer vectors supplies compact IDs for shifts and levels, reducing numerical-cache key costs without changing which values are equated. The native transition cache retains 131,072 entries, reducing transition misses from 6,175,992 to 5,358,392. Numerical caches remain local to each Virasoro block; the existing exchange symmetry still reused 400 full products. All 804 ordinary Virasoro series and the same n-dependent cutoffs were retained.

For the benchmark weights, the 402 base tuples in each Virasoro copy are distinct modulo integer shifts, so a cache across different n branches would not identify identical full shifted blocks. The shared objects introduced here are the integer recursion rules and vector IDs. No changes were made to recurrence factors, accumulation order, precision, coefficient selection, or the Schottky computation.

## Exact preservation of the saved output

The optimized physical block contains 5,786 vectors and 46,288 parity slots. Every saved real and imaginary coefficient string is exactly identical to the original machine-precision result. The reduced numerator and full auxiliary coefficients are also exactly identical. The comparison took 0.040726 s and did not recompute either block.

The accuracy is therefore unchanged: the existing machine-precision discrepancies against direct PBW remain. No new PBW computation or other CFT cross-check was performed. Syntax compilation and a targeted comparison of the modified recursion against its archived implementation also passed.

## Optimized full-process segments

| Segment | Seconds |
| --- | ---: |
| outer branching | 10.338901 |
| middle branching | 1.021460 |
| ccy recursion | 46.573809 |
| virasoro products | 1.775751 |
| direct fermion | 0.293540 |
| convolution division | 0.699796 |
| schottky vacuum | 0.042366 |
| vacuum restoration | 0.054592 |
| other setup assembly imports saving exit | 1.097847 |

The outer calculation built and stored 1,936 branching coefficients; the middle calculation stored 36 and reused the prepared outer actions. No previously saved branching coefficients were loaded.

## Files and reproduction

- [Optimized physical block](../Code/theta_fermion_ccy/results/ccy_recursion_reuse_machine_level10.json).
- [Full-process timing](../Code/theta_fermion_ccy/results/ccy_recursion_reuse_machine_level10_walltime.json).
- [Exact coefficient comparison](../Code/theta_fermion_ccy/results/comparison_recursion_reuse_machine_level10.json).
- [Timing summary and source hashes](../Code/theta_fermion_ccy/results/ccy_recursion_reuse_machine_level10_summary.json).
- [Implementation and cache-dependence explanation](../Code/theta_fermion_ccy/RECURSION_REUSE.md).

From the repository root:

```sh
/private/tmp/theta_fermion_ccy_env/bin/python Code/theta_fermion_ccy/run_timed_pipeline.py --timing-json Code/theta_fermion_ccy/results/ccy_recursion_reuse_machine_level10_walltime.json --level 10 --auxiliary direct --machine --sector-policy record --json Code/theta_fermion_ccy/results/ccy_recursion_reuse_machine_level10.json
```

The previous production source is retained under Code/theta_fermion_ccy/profiling_baseline/pre_recursion_steps. The manuscript was not edited.
