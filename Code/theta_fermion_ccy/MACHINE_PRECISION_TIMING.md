# Native machine-precision level-five timing

Measured on 2026-09-11 for the same physical superconformal block as the
preceding precision benchmarks: b=7/5, P=(11/23,13/29,17/31), p=f=0,
(eta,eta')=(+,-), through total physical q-level 5.

Each measurement starts a fresh Python process with empty coefficient caches.
All branching preparation is included; computed intermediates are reused
within that process. These are single-run elapsed times, not averages.

| Numerical arithmetic | Complete process, seconds | Internal timer, seconds |
| --- | ---: | ---: |
| Outer 70 digits; middle/CCY 100 digits | 12.608884041997953 | 12.240286874999583 |
| Outer, middle and CCY 40 digits | 11.307500124999933 | 10.901327416999266 |
| Native binary64 throughout numerical stages | 1.603974334000668 | 1.2467321669973899 |

The full timer includes imports, all computation, checkpoint/result writes,
and process exit. The internal timer excludes imports and final result
serialization. The native run is approximately 7.05 times faster than the
40-digit run by the full-process measurement.

Native stages: outer branching 0.6113571249989036 s, middle branching
0.08534612994844792 s, CCY/products 0.35989051303477027 s, direct auxiliary
0.11016333299994585 s, and division/restored vacuum factors
0.017196000000694767 s. The Schottky product alone took
0.0013844590030203108 s, included in the last figure. Stage timers exclude
some assembly and serialization overhead and do not sum to the total.

The `--machine` option selects Python float/complex and NumPy/SciPy
float64/complex128 arithmetic, including the fixed-precision `mpmath.fp`
context for the CCY scalar operations. It does not run multiprecision
arithmetic at 15 decimal digits. The normalized direct-fermion series and
Schottky vacuum product retain their exact rational algorithms; their
coefficients and the overall fermion normalization are converted to binary64
for numerical assembly.

The optimized outer action preparation, reflection/parity reuse, closed Ward
support, cached factorizations, recursive middle coefficients and CCY product
reuse are retained. Native action spans use normalized NumPy least squares;
outer Ward systems use the same pivoted square restriction with native LU.
No multiprecision iterative refinement is performed in native mode. Native
span and full Ward residuals must pass 1e-8; the sector-division guard remains
1e-8. The inherited native sparse arithmetic cutoff is 1e-13. The native CCY
pole-separation guard is 64 binary64 epsilons, scaled by the central charges;
unresolved poles raise an error.

The run completed with maximum recorded action-span relative residual
8.52126757318685e-13 and outer Ward relative residual
9.058688035133073e-14. It computed 300 ordinary CCY series and saved 581
four-variable coefficient vectors for the physical block. Solver residuals
do not establish final coefficient accuracy. No PBW, split-parameter or
precision comparison was run during the timing request. The subsequent
precision comparison requested by the user is documented below.

Reproduce from the repository root:

```sh
/private/tmp/theta_fermion_ccy_env/bin/python Code/theta_fermion_ccy/run_timed_pipeline.py \
  --timing-json Code/theta_fermion_ccy/results/ccy_schottky_machine_level5_walltime.json \
  --level 5 --auxiliary direct --machine \
  --json Code/theta_fermion_ccy/results/ccy_schottky_machine_level5.json
```

Full coefficients and source hashes are in
`results/ccy_schottky_machine_level5.json`; complete elapsed time is in
`results/ccy_schottky_machine_level5_walltime.json`. The five source files
preceding native support are archived with hashes in
`profiling_baseline/pre_machine_backend/`. The manuscript was not edited.

## Requested comparison with the saved 40-digit run

All 4,648 parity slots in 581 physical coefficient vectors were compared,
including unequal split-edge exponents. Parameters, cutoffs and normalization
metadata matched; neither block was recomputed. Differences were evaluated
at 70 decimal digits from the saved coefficient strings, so the comparison
itself does not subtract using machine precision.

Maximum absolute error and maximum scaled error are both
7.4235875570866353e-9, where the scaled error is
`abs(machine-reference)/max(1,abs(reference))`. All slots pass the existing
1e-8 criterion, with little margin. This does not establish that machine
precision is adequate at other parameters or at level ten.

The worst slot has exponents `(0,7,1,2)`, parity index 6, at total level five.
The machine value is
`-6.712442151960701e-9 + 3.170610757301019e-9 i`, whereas the 40-digit value
is approximately `-1.7142008274080541e-23`. Its unequal split powers identify
a coefficient that should cancel in the recovered physical block. No
coefficient was projected out or discarded before comparison.

The comparison took 0.13473079200048232 seconds internally, including reading
the saved files and assembling the report. This was not part of the block
runtime. Results, worst slots, per-level maxima and input-file hashes are
saved in `results/accuracy_machine_vs_40dps_level5.json`.

```sh
/private/tmp/theta_fermion_ccy_env/bin/python Code/theta_fermion_ccy/compare_saved_precision.py \
  --candidate Code/theta_fermion_ccy/results/ccy_schottky_machine_level5.json \
  --reference Code/theta_fermion_ccy/results/ccy_schottky_40dps_level5.json \
  --json Code/theta_fermion_ccy/results/accuracy_machine_vs_40dps_level5.json
```
