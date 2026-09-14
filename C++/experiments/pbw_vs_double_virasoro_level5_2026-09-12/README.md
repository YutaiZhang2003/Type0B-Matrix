# Optimized direct PBW versus double Virasoro: independent-edge level five

All three physical plumbing powers are independently cut off at level 5. Each result contains 396 monomials and 3,168 parity slots, including monomials of total level as high as 15. This is not a total-level-five benchmark.

Parameters: b=7/5, P=(11/23,13/29,17/31), p=f=0. Ordinary uses (+,+); inserted uses (+,-). Both algorithms use 40 decimal digits / 136 working bits. Each run starts in a separate fresh process, with no loaded numerical caches, and stores/reuses intermediates within that run. Runs are sequential; compilation is excluded. Process wall time includes imports/startup, final serialization, cleanup, and exit. These are single-run measurements.

| Sector | Optimized PBW wall | Double Virasoro wall | PBW / DV |
|---|---:|---:|---:|
| eta eta-prime = +1 | 68.854 s | 3.614 s | 19.05x |
| eta eta-prime = -1 | 139.200 s | 7.556 s | 18.42x |

## Direct PBW implementation

This is an optimized **Python/FLINT physical SCA PBW** implementation, not a C++ Ward engine. It computes the physical block directly in both sign sectors. The CLI word "inserted" identifies the negative sign sector; PBW itself does not introduce the auxiliary insertion.

- Cache descendant words, ground phases, word levels, and parities.
- Share mode-action module caches between the eta = +1 and -1 three-point forms.
- Construct one triangle of each symmetric bilinear Gram matrix and retain its inverse.
- Reuse the whole vertex tensor when the two signs agree.
- Contract through native FLINT matrices and raw FLINT scalar arrays, with cached tensor permutations and midpoint extraction between stages. No complex conjugation is used.
- Write the final output once, instead of rewriting accumulated JSON after every monomial.

The physical Ward equations are unchanged and remain memoized. Scalar arithmetic inherits the existing midpoint wrapper and its 1e-80 small-term threshold. Native contraction intermediates are projected to midpoints without that threshold. This is numerical arithmetic, not certified interval evaluation.

| PBW stage | Positive sector | Negative sector |
|---|---:|---:|
| Load/setup | 0.3213 s | 0.4525 s |
| Basis metadata | 0.0007 s | 0.0007 s |
| Gram entries | 0.0754 s | 0.0731 s |
| Gram inverses | 0.0096 s | 0.0090 s |
| Vertex tensors / Ward recursion | 67.0551 s | 137.0297 s |
| Tensor contractions | 0.9758 s | 1.0121 s |
| Internal total | 68.4533 s | 138.6029 s |

## Double-Virasoro implementation

The full C++ pipeline includes stored recursive branching, CCY recursion, Schottky vacuum computation, direct free-fermion factors, and convolution recovery. This isolated build also includes the two recently measured CCY optimizations: grouped inserted assembly and dense vertex caches, plus reusable MPC scratch during propagation. Ordinary three-edge global assembly retains the production formula. Production headers and binary are unchanged.

| Double-Virasoro stage | Positive sector | Negative sector |
|---|---:|---:|
| Branching, excluding inserted middle | 0.1065 s | 0.0954 s |
| Inserted middle | 0.0000 s | 0.0012 s |
| CCY | 2.5784 s | 6.5942 s |
| Virasoro products | 0.3037 s | 0.5684 s |
| Assembly | 0.0281 s | 0.0472 s |
| Direct fermion | 0.0311 s | 0.0319 s |
| Convolution division | 0.0863 s | 0.0839 s |
| Schottky | 0.0044 s | 0.0043 s |
| Vacuum restoration | 0.0057 s | 0.0058 s |
| Internal total | 3.1665 s | 7.5393 s |

## Directed validation

Before the level-five runs, both PBW implementations were compared over the entire independent-edge level-two box in each sign sector. All 360 components per sector agree exactly in the serialized results. No baseline PBW level-five run was performed, so this experiment does not measure a whole-block speedup relative to the previous PBW implementation.

The requested level-five comparison checks every physical component against the independently computed double-Virasoro result. The scaled difference is abs(a-b)/max(1,abs(a),abs(b)).

| Sector | Maximum scaled difference | Maximum absolute difference |
|---|---:|---:|
| ordinary | 8.598e-21 | 3.951e-16 |
| inserted | 1.395e-20 | 1.724e-16 |

All four computations completed with finite coefficients. The positive sector passes the chosen 1e-20 scaled criterion; the negative sector narrowly misses it at 1.395e-20. The timing record retains this failure instead of relaxing the threshold after seeing the result.

A read-only comparison with the previously saved production level-ten box, restricted to these same powers, finds a 1.498e-20 maximum difference from the new negative-sector PBW result. The latest CCY result differs from that saved production result by at most 2.596e-21. Thus a discrepancy of this size was already present relative to the saved production output; it cannot be attributed solely to the two latest CCY changes. This comparison does not isolate the error in either algorithm. No higher-precision run was made.

Agreement is a cross-check, not a certified bound on the exact coefficients. These timings compare the specified implementations at this cutoff and parameter point; they do not determine the speed of a future compiled Ward engine or an algorithmic lower bound for PBW.

## Reproduction and artifacts

```sh
python3 C++/experiments/pbw_vs_double_virasoro_level5_2026-09-12/build_ccy.py
python3 C++/experiments/pbw_vs_double_virasoro_level5_2026-09-12/measure.py --validate
python3 C++/experiments/pbw_vs_double_virasoro_level5_2026-09-12/measure.py
python3 C++/experiments/pbw_vs_double_virasoro_level5_2026-09-12/render_report.py
```

The PBW source is `Code/theta_fermion_ccy/optimized_pbw.py`. `results/timings.json` contains parent and stage clocks and comparisons; the four adjacent result JSONs contain complete physical coefficient arrays. `validation/` records the small directed PBW check. `results/saved_production_comparison.json` records the read-only comparison with the earlier production output. `build.json` records C++ compiler flags and source/executable hashes.
