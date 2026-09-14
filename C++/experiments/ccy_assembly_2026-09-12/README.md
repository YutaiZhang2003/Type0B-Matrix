# CCY assembly optimization measurements, 12 September 2026

These are isolated prototypes. The production headers, binary, and machine notes were not changed. The measurements concern the inserted Ramond pipeline with independent cutoffs (q_1,q_2,q_3\leq 10), using 40 decimal digits (136 MPC bits).

## Variants

- **Baseline:** the current production CCY implementation with phase timers added.
- **1 — grouped assembly:** loop over recursion shifts and the two middle descendant levels first; reuse the product of the incoming amplitude and middle vertex over the outer descendant levels. Enumerate only compatible target/shift pairs, using a dense array of pointers to output coefficients. Each coefficient retains the original order of summation over shifts; the grouping of multiplications changes.
- **2 — dense caches and scratch reuse:** replace complete-vertex hash lookups by dense arrays with the same shift and descendant labels. Reuse an MPC multiplication temporary during forward propagation. Core vertex, norm, fusion, pole, and denominator formulas are unchanged.
- **Both:** combine these changes. Their savings overlap and should not be added.

The grouped prototype supports full four-edge rectangular domains only. These prototypes have not been validated for the ordinary three-edge pipeline or other cutoff domains.

## Measurements

Each variant was run in a fresh sequential process, for both Virasoro copies at three branch profiles. Two rounds used reversed ordering to reduce order effects. Compilation and result serialization are excluded from the CCY timer. Cache preparation is included; engine construction and destruction are outside that timer. All 48 measurements took 100.96 seconds including process and output overhead.

The profile labels below are four times the physical branching labels. Parameters are (b=7/5) and ((P_1,P_2,P_3)=(11/23,13/29,17/31)), matching the full benchmark.

| Profile | Labels | Remaining Virasoro caps |
|---|---|---|
| Largest | `(0,-1,1,1)` | `(10,10,10,10)` |
| Medium | `(4,3,5,-5)` | `(8,9,7,7)` |
| Small | `(6,5,7,7)` | `(5,7,4,4)` |

Median CCY seconds across two rounds:

| Profile / copy | Baseline | 1 | 2 | Both |
|---|---:|---:|---:|---:|
| Largest / 0 | 7.022 | 5.447 | 5.131 | 4.111 |
| Largest / 1 | 6.486 | 5.195 | 4.588 | 3.779 |
| Medium / 0 | 1.013 | 0.861 | 0.798 | 0.678 |
| Medium / 1 | 1.134 | 0.975 | 0.909 | 0.778 |
| Small / 0 | 0.060 | 0.054 | 0.052 | 0.048 |
| Small / 1 | 0.061 | 0.057 | 0.054 | 0.050 |

## Full-pipeline estimates

The completed production benchmark took **1655.360 seconds (27m35s)**, including **1460.243 seconds in CCY**. The full pipeline was not rerun for this experiment.

For each of the actual 1,620 Virasoro factors, assign the nearest sampled profile by logarithmic seed-term count, then scale its measured time by the exact seed count and use the corresponding Virasoro copy. This enumeration totals **1,740,820,224 seed terms**, agreeing with the saved full benchmark. Normalize the resulting baseline estimate to its measured CCY time and keep the other 195.117 seconds unchanged.

| Variant | Estimated CCY time reduction | Estimated full runtime | Estimated saving |
|---|---:|---:|---:|
| 1 | 17.4% | 23m21s | 4m14s |
| 2 | 23.7% | 21m49s | 5m47s |
| Both | 36.1% | 18m48s | 8m48s |

These are workload-weighted projections, not measurements of complete runs. Three profiles do not capture all variation in aspect ratios, weights, propagation work, or cache behavior. Construction, cleanup, and all non-CCY costs are held fixed. Rounded expectations of **23, 22, and 19 minutes**, respectively, are more appropriate than treating the displayed seconds as predictive precision.

## Directed coefficient comparison

Every coefficient from each timed variant was compared with its matching baseline at the same precision and parameters. Seed-term and recursion-transition counts agree. The dense-cache variant agrees exactly in the serialized coefficients. Regrouping multiplication produces a maximum scaled difference of **1.401e-35**, where the scale is `max(1, abs(baseline), abs(variant))`. All coefficients are finite. This checks preservation of the sampled CCY computations; it is not an independent accuracy bound or a full-pipeline validation. No PBW calculation was run.

## Reproduction and saved data

From the repository root:

```sh
python3 C++/experiments/ccy_assembly_2026-09-12/build_prototype.py
python3 C++/experiments/ccy_assembly_2026-09-12/measure.py
```

- `build.json`: compiler commands and source hashes.
- `measurements.json`: all timing records, comparisons, workload weights, and projections.
- `results/`: coefficients and counters from each timed process.
- `quick_*.json`: preliminary level-four directed comparisons, excluded from the estimates.
- Production timing reference: `C++/results/per_edge_level10_2026-09-11/optimized_full_pipeline_40dps/timings.json`.
