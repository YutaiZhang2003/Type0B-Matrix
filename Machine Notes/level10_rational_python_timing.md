# Python CCY with cached dependence on the central charge

The compiled experiment was removed from production at the user's request. The current implementation is Python and reorganizes the CCY recurrence itself by caching its dependence on the evaluation central charge.

For fixed shifted internal weights and remaining levels, the recurrence has the form

\[
F_{\boldsymbol\ell}(c)=G_{\boldsymbol\ell}
+\sum_{e,r,s}\frac{R_{rs}^{(e)}}{c-c_{rs}(h_e)}
F_{\boldsymbol\ell-rs\mathbf e_e}
\bigl(c_{rs}(h_e);h_e\to h_e+rs\bigr).
\]

Every lower block on the right is independent of the evaluation c on the left. The previous implementation cached numerical results separately for (c, shifted weights, levels), and revisited the same lower-block cache entries while evaluating each new c. The new implementation builds the global seed and the complete ordered list of pole data/lower-block values once for (shifted weights, levels). Reusing that list at another c requires only evaluating its rational terms; it makes no further lower-block calls.

The implementation keeps the residue and lower-block value separate. It still evaluates (residue/(c-pole))*lower in the original order before accumulation, preserving the existing floating-point arithmetic exactly. It uses sparse factor tables indexed first by evaluation c and then by pole-template ID. Each template holds the pole, residue and shifted internal weights. Nonterminal global seeds are stored inside the rational-block entries; only terminal global seeds need a separate cache. There is no separate cache of numerical block values for every c.

This is a reorganization of the same recurrence, not an approximation. It preserves all four-edge coefficients, the n-dependent truncation, the selected Schottky factor, and the singular-denominator guards. The pole-list construction is acyclic because every lower block has smaller remaining level. For generic inputs the same coefficients result. No automatic prescription for confluent singular limits is added; because lower pole lists are now built before evaluation at a parent c, the order in which errors are encountered can change, but errors are not discarded.

## Full fresh-process level-ten measurement

Parameters: b=7/5, P=(11/23,13/29,17/31), p=f=0, (eta,eta')=(+,-). Both runs use native binary64 numerical arithmetic. Each constructs the branching coefficients from scratch and reuses them within the run. The auxiliary and Schottky factors remain exact rational series before their numerical normalizations.

| Measurement | Previous Python | New Python |
| --- | ---: | ---: |
| CCY stage | 46.573809 s | 33.996989 s |
| Full process | 61.898062 s | 49.128484 s |
| Peak resident memory | 425.500 MiB | 399.719 MiB |

The new complete run is 1.260 times faster, or 20.63% shorter, than the previous Python implementation. Both complete-process timers include imports, computation, result writes and exit. These are single-run measurements. The separate direct PBW machine benchmark remains faster at 33.550881 s.

## What work is reused

Across the 804 ordinary Virasoro series, the new run built 5,640,620 nonterminal pole expansions and reused them 16,488,456 times. There were still 22,129,076 numerical evaluations at particular central charges. Thus the reduction is in constructing and traversing the recursive lower-block lists, not in the number of required evaluations at c.

It also constructed 1,008,188 distinct terminal global coefficients and 1,046,040 pole templates. The existing split-edge symmetry reused 400 complete Virasoro products. Branching still stored and reused 1,936 outer and 36 middle coefficients. No previously saved block or branching data were loaded.

## Verification

All 5,786 physical coefficient vectors, containing 46,288 parity slots, have exactly identical saved real/imaginary strings to the previous Python result. The reduced numerator and full auxiliary series are also exactly identical. The existing numerical discrepancies against PBW are therefore unchanged. Only comparisons needed to verify the modified recurrence were performed; no new PBW run or other CFT test was added.

The maximum-cutoff test branch (0,-1/4,1/4,-1/4), first Virasoro copy, changed from 556,736 cached numerical recursion states to 108,130 cached nonterminal pole expansions. It matched exactly and took 0.729946 s versus 1.180231 s in the paired test. The full-process timings above are the primary performance measurements.

## Segments

| Segment | Seconds |
| --- | ---: |
| outer branching | 10.199578 |
| middle branching | 1.019354 |
| ccy recursion | 33.996989 |
| virasoro products | 1.759461 |
| direct fermion | 0.297879 |
| convolution division | 0.669782 |
| schottky vacuum | 0.041150 |
| vacuum restoration | 0.051476 |
| other setup assembly imports saving exit | 1.092815 |

## Files

- [Current Python implementation](../Code/theta_fermion_ccy/punctured_ccy.py).
- [Computed physical block](../Code/theta_fermion_ccy/results/ccy_rational_python_machine_level10.json).
- [Full-process timer](../Code/theta_fermion_ccy/results/ccy_rational_python_machine_level10_walltime.json).
- [Exact coefficient comparison](../Code/theta_fermion_ccy/results/comparison_rational_python_machine_level10.json).
- [Summary, counts and source hashes](../Code/theta_fermion_ccy/results/ccy_rational_python_machine_level10_summary.json).

Reproduce from the repository root:

```sh
/private/tmp/theta_fermion_ccy_env/bin/python Code/theta_fermion_ccy/run_timed_pipeline.py --timing-json Code/theta_fermion_ccy/results/ccy_rational_python_machine_level10_walltime.json --level 10 --machine --auxiliary direct --sector-policy record --json Code/theta_fermion_ccy/results/ccy_rational_python_machine_level10.json
```

No compiled recursion module is imported or built. Its former sources and binaries are kept only in profiling_baseline/compiled_experiment as historical records. The previous Python sources are archived in profiling_baseline/pre_rational_blocks. The manuscript was not edited.
