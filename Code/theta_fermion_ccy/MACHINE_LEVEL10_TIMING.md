# Fresh machine-precision level-ten attempt

The subsequent run completed in **92.56964104100189 seconds**, following the
user's instruction to record sector residuals without stopping. See
[the completed run](../../Machine%20Notes/level10_machine_timing.md). The
remainder of this file documents the earlier stopped attempt.

On 2026-09-11 the full production pipeline was started in a fresh process
with no saved coefficients loaded, using native binary64 for branching,
CCY recursion and numerical assembly. Parameters were b=7/5,
P=(11/23,13/29,17/31), p=f=0, (eta,eta')=(+,-), total physical q-level 10.
The direct fermion auxiliary and Schottky vacuum product use exact rational
series before conversion to binary64.

**The run stopped at the unchanged 1e-8 sector-division guard. No accepted
level-ten physical block was produced.** Elapsed time from process launch
through failure and saving was **96.01631708299828 seconds**; the internal
timer was 95.49225591700088 seconds.

| Segment | Seconds |
| --- | ---: |
| Outer branching, including action preparation | 10.5588706669987 |
| Middle branching, reusing outer edge-two actions | 1.0271720829987316 |
| Ordinary Virasoro CCY recursion | 80.75978157710415 |
| Virasoro series products and transpose reuse | 1.8026418361041578 |
| Other CCY setup | 0.10796369381932891 |
| Direct fermion auxiliary | 0.30465137499777484 |
| Convolution division until guard failure | 0.3496536249986093 |
| Remaining assembly, imports, checkpoints, serialization and exit | 1.105582225976832 |

Action preparation accounts for 10.127247583001008 seconds of the outer
branching time; it is not an additional segment. These clocks are a single
run, not averages. Full-scale output accuracy was not compared with PBW,
another precision, or split-parameter evaluations.

The run stored and reused 1,936 outer and 36 middle branching coefficients.
The middle recurrence now obtains already computed edge-two L-minus action
coefficients directly from the outer stage whenever their precision matches;
its L-plus action provider also retains computed coefficients. Outer Ward
factorizations continue to serve both spin signs. The run computed 804
ordinary CCY series, reused 400 transposed products, and saved 5,786 enlarged
numerator vectors and 4,793 direct auxiliary vectors. The branching
checkpoint was written before beginning CCY assembly.

The guard first failed at exponents `(0,6,1,2)` with residual
2.0868810380187958e-8. A bounded follow-up diagnostic used compensated
double-precision summation (`math.fsum`) for division of the saved numerator
by the saved auxiliary. It failed at the same exponents with residual
2.0868810343057392e-8 (1.178807208998478 seconds internally). This does not
resolve the input inconsistency; neither the guard nor the coefficients
were altered or projected to force acceptance.

The largest recorded outer Ward condition estimate was approximately
3.91e7, with maximum full Ward relative residual 1.31e-11 and action-span
relative residual 5.82e-12. Small equation residuals do not guarantee an
accurate recovered block for this conditioning and cancellation pattern.

Because division precedes vacuum restoration in the production pipeline,
the Schottky factor was not reached in that process. To measure the
requested segment, a separate fresh process computed the exact level-ten
Schottky product in **0.04344200000195997 seconds**, excluding imports and
saving. This is recorded separately and is not included in the 96.0163-second
failed-run total. Multiplication of an accepted quotient by the two vacuum
factors remains unperformed, since there was no accepted quotient.

Artifacts:

- `results/ccy_schottky_machine_level10_summary.json`: complete timing summary.
- `results/ccy_schottky_machine_level10.json`: enlarged numerator, full
  auxiliary, diagnostics, error and source hashes.
- `results/ccy_schottky_machine_level10_branching.json`: branching coefficients
  and parameters, computed during this run.
- `results/ccy_schottky_machine_level10_walltime.json`: full process clock.
- `results/ccy_schottky_machine_level10_vacuum.json`: separately timed exact
  Schottky coefficients and provenance.

The pipeline and outer-branching sources preceding segment instrumentation
and shared middle actions are archived with hashes under
`profiling_baseline/pre_level10_segments/`. No manuscript edits were made.

Reproduce the attempted run from the repository root:

```sh
/private/tmp/theta_fermion_ccy_env/bin/python Code/theta_fermion_ccy/run_timed_pipeline.py \
  --timing-json Code/theta_fermion_ccy/results/ccy_schottky_machine_level10_walltime.json \
  --level 10 --auxiliary direct --machine \
  --json Code/theta_fermion_ccy/results/ccy_schottky_machine_level10.json
```
