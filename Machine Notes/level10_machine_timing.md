# Completed level-ten machine-precision timing

The full physical superconformal-block calculation completed from scratch in
**92.56964104100189 seconds** on 2026-09-11. This is a fresh-process wall
time, including imports, all preparation, checkpoint/result writes, and
process exit. The internal timer was 92.06196366700169 seconds.

The parameters were b=7/5, P=(11/23,13/29,17/31), p=f=0 and
(eta,eta')=(+,-), through total physical q-level 10. Branching, ordinary
CCY recursion and assembly used native binary64. The normalized direct
fermion factor and Schottky vacuum product used exact rational arithmetic
before conversion for numerical assembly. No previously saved coefficients
were loaded.

| Segment | Seconds |
| --- | ---: |
| Outer branching, including action coefficients | 10.095476957998471 |
| Middle branching | 1.0357965830007743 |
| Ordinary Virasoro CCY recursion | 77.4308096249697 |
| Virasoro series products | 1.7683383860930917 |
| Direct fermion factor | 0.25903358399955323 |
| Convolution division | 0.7088261249991774 |
| Schottky vacuum product | 0.0411950829984562 |
| Restoration of the two vacuum factors | 0.05376541700024973 |
| Remaining setup, assembly, imports, saving and exit | 1.1763992799424159 |
| **Complete process** | **92.56964104100189** |

Action preparation accounts for 9.667809042002773 seconds within the outer
branching segment. These are single-run timings, not averages; the small
difference from the preceding stopped attempt does not establish a speedup
caused by changing the residual policy.

The calculation stored and reused 1,936 outer and 36 middle branching
coefficients. The middle recurrence reused the outer stage's edge-two
action coefficients at the same precision. Outer Ward factorizations served
both signs. There were 804 computed ordinary CCY series and 400 transposed
product reuses. The completed physical output contains 5,786 coefficient
vectors, or 46,288 parity slots, with all four plumbing exponents retained.

The Schottky product uses six primitive classes modulo inversion at this
cutoff. The calculation divides the enlarged series with its two universal
vacuum factors removed by the full direct fermion factor, then restores
both vacuum factors to obtain the final physical series.

At the user's instruction, this run used `--sector-policy record`:
convolution continues through numerical sector residuals, keeping every
computed coefficient unchanged. No projection or deletion enforces sector
membership. The auxiliary constant and nonfinite-number guards remain
active. The default policy for other runs still raises on excess residuals.

The first residual exceeding 1e-8 was 2.0868810380187958e-8 at exponents
`(0,6,1,2)`. Continuing through level ten found a **maximum scaled sector
residual of 0.040660192019849634**, at `(4,5,3,8)`, with 2,466 coefficient
vectors exceeding the reporting threshold. This diagnostic is
`max(abs(value-minus_projection(value)))/max(1,max(abs(value)))` during
recovery, before vacuum restoration; the maximum absolute residual is the
same here. It is not a comparison with an independent or higher-precision
physical block. No additional numerical cross-checks were run for this
timing request.

Saved artifacts:

- [Full timing summary](../Code/theta_fermion_ccy/results/ccy_schottky_machine_complete_level10_summary.json).
- [Physical block, enlarged numerator, auxiliary factor, diagnostics and source hashes](../Code/theta_fermion_ccy/results/ccy_schottky_machine_complete_level10.json).
- [Branching coefficients computed during this run](../Code/theta_fermion_ccy/results/ccy_schottky_machine_complete_level10_branching.json).
- [Complete process wall time](../Code/theta_fermion_ccy/results/ccy_schottky_machine_complete_level10_walltime.json).

Reproduce from the repository root:

```sh
/private/tmp/theta_fermion_ccy_env/bin/python Code/theta_fermion_ccy/run_timed_pipeline.py \
  --timing-json Code/theta_fermion_ccy/results/ccy_schottky_machine_complete_level10_walltime.json \
  --level 10 --auxiliary direct --machine --sector-policy record \
  --json Code/theta_fermion_ccy/results/ccy_schottky_machine_complete_level10.json
```

The source preceding the explicit residual-recording option is archived
with hashes under `Code/theta_fermion_ccy/profiling_baseline/pre_record_sector_residuals/`.
The paper draft was not edited.

The subsequently requested direct physical PBW comparison through level six
is documented in [the comparison note](level10_machine_pbw_comparison_level6.md).
It compares the saved level-ten result truncated to level six, retaining all
unequal split powers, against a fresh PBW reference.
