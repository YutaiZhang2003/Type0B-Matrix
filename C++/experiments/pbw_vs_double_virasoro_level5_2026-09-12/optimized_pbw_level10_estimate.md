# Optimized PBW estimate: independent-edge level ten

This estimate uses the newly measured 40-digit / 136-bit independent-edge level-five stage times and exact basis/work counts. No new physical Gram matrix, Ward value, or PBW block was evaluated. The target is q1,q2,q3 <= 10 independently.

| Work | Level 5 | Level 10 | Growth |
|---|---:|---:|---:|
| Vertex entries, positive sector | 353,934 | 659,037,706 | 1862.0x |
| Vertex entries, negative sector | 707,868 | 1,318,075,412 | 1862.0x |
| Dense contraction multiplications | 15,344,984 | 254,809,434,024 | 16605.4x |
| Gram triangle entries | 2,167 | 226,919 | 104.7x |
| Cubic Gram proxy | 76,475 | 78,765,711 | 1030.0x |

| Sector | Constant unit costs | Longer-Ward sensitivity scenario |
|---|---:|---:|
| ordinary | 39.2 h | 78.6 h |
| inserted | 75.6 h | 156.1 h |

The first scenario keeps the measured cost per requested vertex entry and contraction product fixed. It predicts 34.7/70.9 hours of vertex work and 4.5/4.7 hours of contractions for the positive/negative sectors. The remaining stages contribute little.

The second scenario multiplies only the vertex cost by 2.137: the entry-weighted mean total level grows from 11.71 to 25.02. This illustrates sensitivity to longer Ward recursions; it is not an actual count of their internal operations. The two scenarios are not upper/lower bounds or a confidence interval.

## Memory condition

The unchanged code retains at least 659,037,706 and 1,318,075,412 distinct top-level Ward keys in the two sectors. At the measured CPython six-argument tuple size, these keys alone occupy approximately 63.3 and 126.5 GB (decimal), excluding all values, dictionary storage, internal Ward states, and mode-action caches. The persistent tensor-permutation tables add approximately 35.5 GB. These are partial memory estimates, not total peak RSS.

Consequently the time estimates require sufficient RAM without paging. A bounded or more compact cache design would be needed on machines below that requirement; changing cache retention can also change the runtime.

The earlier double-Virasoro level-ten results are 6m57s for the ordinary production pipeline and 27m35s for the inserted production pipeline, with approximately 19 minutes projected for the two latest inserted CCY optimizations. Only the first two are measured full level-ten runs.

Reproduce using the benchmark Python environment:

```sh
/private/tmp/theta_fermion_ccy_env/bin/python C++/experiments/pbw_vs_double_virasoro_level5_2026-09-12/estimate_level10.py
```

Raw counts, phase estimates, assumptions, and source hashes are in `results/optimized_pbw_level10_estimate.json`.
