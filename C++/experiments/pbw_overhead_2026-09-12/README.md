# Direct PBW optimization assessment

The existing physical SCA PBW reference can be optimized. The measurements below isolate avoidable overhead at **40 decimal digits / 136 bits**. No physical block, Gram matrix, mode action, or Ward value was evaluated, and no production code was modified.

| Candidate | Measured component improvement | Scope |
|---|---:|---|
| Cache each basis state's descendant word and the three ground-state phase powers | **6.99x** | Entry preparation, including the final phase multiplication, excluding the Ward evaluation |
| Keep matrix operands in FLINT instead of converting Python object arrays for every multiplication | **1.65–1.99x** | Synthetic matrix multiplication kernels, excluding changes of tensor layout and final extraction |

Three repetitions were made, alternating order. The metadata sample contains 10,000 triples drawn from edge-level-ten basis metadata. Matrix shapes were `(32,32,64)`, `(161,161,64)`, `(232,232,64)`, and `(232,232,512)`. The last is the largest tested shape and gives 1.65x. The experiment took 4.53 seconds internally.

For the independent-edge level-ten box in the negative sector, the current top-level vertex loop requests 1,318,075,412 entries. It rebuilds three words per entry: **3,954,226,236 word constructions**, although only **3,369 words** are needed when retaining both Ramond ground labels separately on each edge. It also reevaluates the same phase expression **1,318,075,412 times**, although its exponent is only 0, 1, or 2. These counts exclude internal Ward work.

Applying the sampled metadata saving uniformly to every entry would save about 3.5 hours, but this is a component-only projection. Lower-level words are shorter; the sample does not reproduce the full joint distribution of states or large-cache behavior. It should not be presented as a measured whole-block saving.

There are further concrete opportunities in the source:

- Share the mode-action caches between the two `GeneralizedNRRWard` objects for eta = +1 and -1. Their module parameters are identical; eta appears in the ground normalization. The current method cache includes the module object, so separately constructed modules do not share cached actions.
- Use integer state identifiers, stored levels/parities, and precomputed nonzero Ward transitions instead of repeatedly hashing nested words, summing their levels, and visiting empty action sums. The existing Ward recursion is already memoized; the proposed gain is in its representation and shared work, not adding memoization for the first time.
- For equal eta signs, reuse the constructed vertex tensor itself. The current code reuses cached form values but still builds two arrays and repeats word/phase preparation.
- Symmetry can reduce the distinct Gram entries from 450,469 to 226,919 by constructing one triangle of each bilinear Gram matrix. This nearly halves entry construction, not inversion or sewing. Gram matrices and inverses are already reused across level triples.

Those additional optimizations were not timed. The dense contraction workload remains **254,809,434,024 complex multiplications** under the current contraction algorithm. The matrix backend already uses compiled FLINT, so a C++ port alone does not make all this work disappear.

There is **no measured whole-block speedup** here. Component factors cannot be multiplied, and the earlier whole-block fits do not provide measured stage fractions. Consequently the saved multi-day estimates for the Python reference are not limits on an optimized PBW implementation, and these microbenchmarks do not establish a revised end-to-end time.

## Numerical and timing scope

The native FLINT matrix timing is an implementation target rather than a drop-in replacement: a complete implementation must handle tensor permutations, retain the chosen midpoint/precision policy, and extract results. Inputs are synthetic; no PBW coefficient validation was performed. Basis metadata was enumerated using the existing code, and all counts are integer counts.

Sources inspected:

- `Code/ramond_zero_mode_recovery/check_level10_modular.py`: `physical_gram`, `contract`, and `physical_coefficient`.
- `Code/ramond_zero_mode_recovery/modular_backend.py`: `PBWModule.word` and cached actions/inner products.
- `Code/double_virasoro/nsrr/ramond_pbw_generalized_ward.py`: `SuperVirasoroModule` and `GeneralizedNRRWard`.
- `Code/ramond_zero_mode_recovery/complex_arithmetic.py`: Python/FLINT conversions and `mm`.

Reproduce the small overhead measurement from the repository root:

```sh
/private/tmp/theta_fermion_ccy_env/bin/python C++/experiments/pbw_overhead_2026-09-12/measure.py
```

Raw measurements and scope limitations are in `measurements.json`.
