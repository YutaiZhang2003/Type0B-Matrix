# Recursive branching and level-10 timings

The directly computed initial table is exactly the boundary box in Human Notes/SCblock.tex: NS n=0,+/-1/2, and both Ramond labels n=+/-1/4,+/-3/4. Higher coefficients are computed in increasing order of floor(abs(n1))+floor(abs(n2)-1/4)+floor(abs(n3)-1/4), and every result is stored for reuse. The bulk update is scalar; explicit boundary actions couple at most four unanchored coefficients at a time. There is no global outer Ward matrix.

The second conformal Ward identity is used when the first local relation is insufficient. Additional mode actions are constructed lazily and cached. Ward rows, embedded weights and single-leg descendant factors are reused, and the requested vertex signs share each local elimination. Dependencies and used Ward identities are checked after each update.

## Fresh level-10 branching benchmark

Each case ran sequentially in a separate process with empty numerical caches. Parameters: b=7/5, (P1,P2,P3)=(11/23,13/29,17/31), p=f=0. Ordinary uses (+,+); inserted uses (+,-), matching production. Times are seconds; compilation is excluded.

| Mode | Precision | Actions | Outer | Middle | Total | Wall |
| --- | --- | --- | --- | --- | --- | --- |
| Ordinary | Machine | 1.1045 | 0.0123 | 0.0000 | 1.1170 | 1.5163 |
| Inserted | Machine | 1.0917 | 0.0203 | 0.0002 | 1.1123 | 1.1291 |
| Ordinary | 40 digits | 8.6929 | 0.0816 | 0.0000 | 8.7747 | 8.7966 |
| Inserted | 40 digits | 9.2128 | 0.1314 | 0.0122 | 9.3567 | 9.3778 |

Total includes initial mode actions, direct boundary values, outer recursion, and the middle coefficients for the inserted mode. Wall also includes startup, final coefficient serialization, cleanup and exit. These branching-only runs do not compute CCY or full physical blocks. Initial action construction takes over 98% of these totals.

Ordinary stores 792 outer coefficients: 96 direct boundary and 696 recursive values. Inserted stores 1,936 outer coefficients: 192 direct boundary and 1,744 recursive values, and evaluates 36 middle coefficients. No first-prefactor zero or second-identity fallback occurs at this parameter point.

## Fresh full level-10 pipelines

| Mode | Precision | Branching | CCY | Other | Total | Wall |
| --- | --- | --- | --- | --- | --- | --- |
| Ordinary | Machine | 1.0477 | 0.0914 | 0.0617 | 1.2008 | 1.5959 |
| Inserted | Machine | 1.0670 | 0.7288 | 0.0810 | 1.8768 | 1.8931 |
| Ordinary | 40 digits | 8.6019 | 2.0786 | 0.4152 | 11.0958 | 11.1174 |
| Inserted | 40 digits | 8.8693 | 21.6939 | 0.4958 | 31.0589 | 31.0811 |

These are complete production runs: stored branching recursion, double-Virasoro CCY with Schottky, direct free-fermion factor, convolution and vacuum restoration. Branching includes the middle coefficients. Other contains products, assembly, auxiliary factor, division, Schottky and restoration. All use sector-policy record; completion alone is not an accuracy certificate. No new physical PBW block was run.

### Before and after recursive branching: level 10, 40 digits

| Mode | Old outer | New outer | Old wall | New wall |
| --- | --- | --- | --- | --- |
| Ordinary | 1.291 | 0.079 | 38.35 | 11.12 |
| Inserted | 2.336 | 0.127 | 110.41 | 31.08 |

All times are seconds. Outer measures the outer-coefficient stage alone; wall measures the complete process. The former global solver is retained here only as a timing baseline. These are single runs on the same machine at the same parameters, measured in different sessions. Unchanged mode-action and CCY stages also ran faster; neither ratio isolates the effect of the recursion change. The baseline source is C++/results/current_algorithm_2026-09-11/timings.json.

## Full-pipeline accuracy

| Mode | Precision | Scaled difference | Check |
| --- | --- | --- | --- |
| Ordinary | Machine | 6.973e-07 | failed |
| Ordinary | 40 digits | 2.865e-21 | passed |
| Inserted | Machine | 1.202e-03 | failed |
| Inserted | 40 digits | 6.605e-21 | passed |

All 4,048 physical components through level 10 are compared against existing 384-bit physical-result files. The ordinary reference is the saved restricted recovery result; the inserted reference is the saved physical PBW result. No PBW was recomputed. The scaled difference is abs(a-b)/max(1,abs(a),abs(b)). The criteria are 2e-8 at machine precision and 1e-20 at 40 digits.

Both 40-digit runs pass. The machine runs do not meet the requested accuracy at level 10: about 6.97e-7 for ordinary and 1.20e-3 for inserted. A diagnostic run of the former global-branching pipeline has essentially the same large discrepancy (6.91e-7 and 1.20e-3). This is an existing machine-precision limitation, not evidence that the new 40-digit recursion is wrong. Machine runtimes are reported as performance measurements, not accuracy-qualified level-10 results. See saved_high_precision_validation.json and machine_precision_diagnosis.json.

## Directed validation

- All 768 level-5 ordinary-support coefficients agree with the previous global solver: maximum scaled differences 1.67e-25 at the default point and 3.76e-25 at equal Ramond momenta, using 40 digits.
- All 880 level-5 inserted-support coefficients agree within 1.59e-25 at 40 digits.
- The equal-momentum case explicitly exercises vanishing first prefactors. Machine and 40-digit coefficients agree within 3.29e-9 (maximum scaled difference).
- Each low-level run additionally checks 608 second Ward identities, covering both vertex signs and Ramond parities. These checks use stored coefficients, without CCY or PBW blocks.
- The previously failing level-20 ordinary-support branching problem completes at 40 digits for both vertex signs: 4,432 stored coefficients, comprising 192 boundary and 4,240 recursive values. Initial actions took 150.24 s; the outer recursion took 0.697669 s. No vanishing prefactor or second-identity fallback occurred.

Level-20 action reconstruction residuals are at most 8.51e-26. The maximum residual of the first Ward equations used for recursion is 1.71e-40; those equations determine the coefficients, so this checks arithmetic consistency rather than independently certifying output accuracy. The full level-20 physical pipelines remain stopped.

## Reproduction

    make -C C++ all bin/outer_ward_driver
    python3 C++/tools/time_branching.py --level 10 --dps 0 40 --output /tmp/branching_level10
    python3 C++/tools/time_current_pipelines.py --levels 10 --dps 40 --output /tmp/full_level10

The directories level10_timings and full_pipeline_*dps contain commands, source hashes, stage timings and coefficient arrays. The diagnostic driver also saves a partial coefficient table if its outer recurrence fails; the accompanying log identifies the failed labels.
