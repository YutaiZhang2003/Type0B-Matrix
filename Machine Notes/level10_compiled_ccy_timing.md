# Compiled CCY recursion: further level-ten speedup

This is a historical experiment. The user requested a Python implementation; the compiled integration has been removed from production and archived under Code/theta_fermion_ccy/profiling_baseline/compiled_experiment. The current algorithm is documented in [the Python pole-expansion report](level10_rational_python_timing.md).

The full machine-precision level-ten pipeline completed in **43.11 s**, including a fresh compilation of the optional extension. The previous optimized Python loop took **61.90 s**. This is another **1.436-fold speedup**, or **30.35% less time**. Relative to the original 92.569641 s machine pipeline, the complete speedup is **2.147**.

Both full runs start with empty coefficient caches and construct their branching coefficients within the process. The new run explicitly sets CCY_REBUILD_NATIVE=1, and its 0.715886 s compilation is included in both the full-process and CCY-stage times. No previously saved branching or block coefficients are loaded.

Parameters: b=7/5, P=(11/23,13/29,17/31), p=f=0, (eta,eta')=(+,-).

| Measurement | Previous Python loop | Compiled loop |
| --- | ---: | ---: |
| CCY stage | 46.573809 s | 27.571987 s |
| Complete process | 61.898062 s | 43.110992 s |
| Peak resident memory | 425.500 MiB | 398.156 MiB |

These are single-run measurements. The separately measured direct physical PBW machine run remains faster at 33.550881 s.

## Implementation

ccy_native.cpp implements the same depth-first recurrence and memoization using C++ hash maps and decoded level-step lists. It calls the existing Python methods for global coefficients and pole transitions. It also uses CPython's PyNumber_Multiply and PyNumber_Add for every accumulated residue contribution, preserving the existing numeric types, operation order and precision. It neither introduces a different mathematical recurrence nor lowers any n-dependent cutoff.

Each series evaluation retains its coefficient, global and transition caches until all requested levels are complete. The caches are then released together. Python cache_info reports their counters and sizes at completion. Unchanged residue, fusion and three-point formula caches still belong to the Python engine and are cleared at the usual branch boundary. The shared integer-vector and level-rule tables remain reusable across branching labels.

The compiled evaluator reuses lower coefficients exactly as before: 22,129,076 coefficient misses and 36,836,320 hits across the 804 series. It avoids the Python recursive dispatch, argument/tuple handling and associated cache lookup overhead. All 804 ordinary series used the compiled backend; the 400 symmetry-related products were reused as before.

The native backend is selected by default for dps=0. Multiprecision continues to use the Python implementation. Passing native_recursion=False to PuncturedCCY selects the Python loop explicitly. The loader builds a source/ABI-keyed extension in __pycache__ using the installed C++ compiler; no package installation is needed. If the build/import is unavailable, it records the reason and uses the Python loop. Mathematical errors raised during evaluation propagate normally; they do not trigger fallback or omission of a term.

## Exact preservation

All 46,288 physical coefficient slots have exactly the same saved real/imaginary strings as the preceding machine-precision pipeline. The reduced numerator and full auxiliary series are also exactly identical. The comparison took 0.041086 s. The existing absolute and scaled discrepancies against PBW therefore remain unchanged.

The targeted maximum-cutoff branch (0,-1/4,1/4,-1/4), first Virasoro copy, also matched exactly. In its paired test the Python loop took 1.270915250 s and the compiled loop 0.696336500 s, excluding the separately recorded build. No new PBW computation or additional CFT comparison was performed.

## Full-process segments

| Segment | Seconds |
| --- | ---: |
| outer branching | 10.343405 |
| middle branching | 1.080052 |
| ccy including native build | 27.571987 |
| virasoro products | 1.782625 |
| direct fermion | 0.306849 |
| convolution division | 0.696717 |
| schottky vacuum | 0.043176 |
| vacuum restoration | 0.054384 |
| other setup assembly imports saving exit | 1.231797 |

The CCY row includes the 0.715886 s native build, so that build is not added again when summing the table. The outer and middle stages construct and reuse 1,936 and 36 branching coefficients, respectively. The Schottky factor remains exact rational arithmetic.

## Saved files

- [Computed physical block](../Code/theta_fermion_ccy/results/ccy_compiled_machine_level10.json).
- [Complete-process timing](../Code/theta_fermion_ccy/results/ccy_compiled_machine_level10_walltime.json).
- [Exact coefficient comparison](../Code/theta_fermion_ccy/results/comparison_compiled_ccy_machine_level10.json).
- [Segments, cache counters, build provenance and source hashes](../Code/theta_fermion_ccy/results/ccy_compiled_machine_level10_summary.json).
- [Compiled recurrence](../Code/theta_fermion_ccy/profiling_baseline/compiled_experiment/ccy_native.cpp).
- [Optional extension loader](../Code/theta_fermion_ccy/profiling_baseline/compiled_experiment/native_recursion.py).

Historical command used with the archived compiled implementation (current production uses Python):

```sh
CCY_REBUILD_NATIVE=1 /private/tmp/theta_fermion_ccy_env/bin/python Code/theta_fermion_ccy/run_timed_pipeline.py --timing-json Code/theta_fermion_ccy/results/ccy_compiled_machine_level10_walltime.json --level 10 --machine --auxiliary direct --sector-policy record --json Code/theta_fermion_ccy/results/ccy_compiled_machine_level10.json
```

The previous Python sources are archived under profiling_baseline/pre_native_recursion. The paper draft was not edited.
