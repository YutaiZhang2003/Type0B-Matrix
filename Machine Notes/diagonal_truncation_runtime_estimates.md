# Runtime estimates with the diagonal-target cutoff

Historical estimates for the original backward recurrence on the smaller domain. The subsequently implemented [forward recurrence](../Code/theta_fermion_ccy/FORWARD_CCY.md) further reduces CCY work; the revised complete level-ten pipeline measured 16.91 seconds.

The working estimates for a fresh machine-precision physical-block pipeline are **about 23 seconds at total level 10** (planning range 20–25 seconds) and **about 12 minutes at total level 15** (planning range 10–15 minutes). These estimates combine actual CCY timings on the proposed index sets with branching-action measurements. Neither complete modified physical-block pipeline has been run.

The parameters remain b=7/5, P=(11/23,13/29,17/31), p=f=0, eta=1, eta'=-1. The intended pipeline retains Schottky, the direct auxiliary factor, and branching reuse within the run. No saved branching/block values were used to accelerate the measured CCY or branching samples. Production files were not modified.

| Component | Level 10 | Level 15 |
| --- | ---: | ---: |
| CCY recursion | **8.58 s measured for all 652 series** | **391.61 s estimated from 212 measured series** |
| Outer and middle branching | 11.22 s, previously measured | approximately 304.59 s, from measured new actions |
| Full physical pipeline | approximately **23 s** | approximately **12 min** |

## CCY measurement and weighting

The existing Python CCY engine accepts arbitrary downward-closed index sets. The timing driver supplies a+d<=min(L,R), b<=L-a-d, c<=R-a-d, with the branch-dependent left and right budgets derived in [the truncation note](diagonal_target_truncation.md). It computes each series, records the normal cache counters, and clears the numerical caches exactly as in production.

At level 10, all 652 required series were evaluated. Their aggregate recursion time was **8.576511644 s**, compared with **33.996988873 s** for the former four-variable truncation. The measured pole-addition count was **17,977,704**, exactly the direct structural count. The driver internally took 8.894107334 s including its indexing, reporting and file writes; these are not full physical-pipeline times.

At level 15, the driver evaluated both Virasoro copies for one actual branching tuple in each of the **106 distinct budget pairs**: **212 series** out of the 1,248 required. It then multiplied each measured time by the actual number of branching tuples with the same budget pair and summed. The samples took **92.760562373 s** of recursion time and give **391.605485306 s**, or **6.53 minutes**, for all CCY series. The weighted measured addition count is **1,608,644,832**, matching the structural count. Peak resident memory during this sampling process was about **1,005 MiB**, excluding a full pipeline's branching caches.

As a calibration of this sampling rule, applying it to the fully measured level-10 data predicts **8.680202656 s**, compared with the measured **8.576511644 s**, a **1.21%** difference. That observation does not establish a 1.21% error bound at level 15; weight-dependent arithmetic and larger caches can change the spread.

Computed ordinary Virasoro coefficients were saved in pickle streams. No recovered physical coefficients or PBW comparisons were computed.

## Branching measurements

The new level-15 labels require additional action coefficients. A separate process timed the production machine-precision solvers, with their ordinary guards and action-local reuse. The R samples used P=13/29 and parity zero; the NS sample used P=11/23.

| Action | Measured time | Span dimensions |
| --- | ---: | ---: |
| R L_minus1 at n=9/4 | 2.3193 s | 2,880 rows x 187 columns |
| R L_minus1 at n=11/4 | **64.3053 s** | 23,819 rows x 483 columns |
| R L_plus1 at n=11/4 | **8.5689 s** | 10,406 rows x 185 columns |
| NS L_plus1 at n=5/2 | **0.9368 s** | 1,666 rows x 110 columns |

All samples completed with the existing numerical guards. These are measurements of action-coefficient construction, not new physical-block accuracy checks. Their coefficients and diagnostics were stored.

The full outer stage needs four new R L_minus1 solves: two R momenta and the two reflected signs. The other parity is obtained by the existing Theta transport. The middle stage needs four new R L_plus1 solves for two signs and two parities, and the NS stage needs the two signs of the new n=5/2 action. Approximating these by their representative samples gives

\[
11.2189+4(64.3053)+4(8.5689)+2(0.9368)
\simeq304.59\ \mathrm{s}.
\]

The 11.2189 s is the previously measured level-10 outer-plus-middle time. Additional Ward-system assembly/solve overhead at level 15 is included in the allowance below. Sign, momentum and parity dependence of the sampled action timings is a remaining uncertainty.

## Remaining stages and full-time estimate

For a budget pair (L,R), direct assembly of only the desired equal-power product coefficients requires

\[
\sum_{t=0}^{\min(L,R)}(L-t+1)(R-t+1)\binom{t+4}{4}
\]

scalar product pairs. Summing over the actual branching multiplicities gives **871,740 pairs at level 10** and **13,589,684 at level 15**. These include unequal-power Virasoro inputs whose product contributes to an equal-power target. They are direct counts, not measured product-stage times.

Replacing only the old level-10 CCY-stage time, while keeping every other old stage cost unchanged, gives

\[
49.1285-33.9970+8.5765=23.7080\ \mathrm{s}.
\]

Computing only the needed products and diagonal auxiliary/convolution coefficients should reduce the remaining work, motivating the **20–25 second** planning range.

At level 15, the CCY and branching estimates sum to **696.19 seconds**, or **11.60 minutes**. An allowance of 10–25 seconds for the products, direct auxiliary, restricted division, Schottky/restoration, additional Ward-system work and I/O leads to the rounded **12-minute** estimate. The broader **10–15 minute** planning range accounts for sampling and memory uncertainty. It is not a confidence interval or a completed timing measurement.

## Saved artifacts and reproduction

- [Machine-readable estimate](../Code/theta_fermion_ccy/results/timing_estimate_diagonal_pipeline_level10_level15.json)
- [All level-10 CCY timings and counters](../Code/theta_fermion_ccy/results/timing_diagonal_ccy_level10.json)
- [Level-15 sampled CCY timings and counters](../Code/theta_fermion_ccy/results/timing_diagonal_ccy_level15_sample.json)
- [High-label action timings and diagnostics](../Code/theta_fermion_ccy/results/timing_level15_branching_samples.json)
- [Level-10 ordinary Virasoro series](../Code/theta_fermion_ccy/results/diagonal_virasoro_series_level10.pkl)
- [Level-15 sampled ordinary Virasoro series](../Code/theta_fermion_ccy/results/diagonal_virasoro_series_level15_sample.pkl)
- [Sampled action coefficients](../Code/theta_fermion_ccy/results/level15_branching_action_samples.pkl)

Run these sequentially to avoid timing interference:

```sh
/private/tmp/theta_fermion_ccy_env/bin/python Code/theta_fermion_ccy/time_diagonal_ccy.py --level 10 --scope all --json Code/theta_fermion_ccy/results/timing_diagonal_ccy_level10.json --series Code/theta_fermion_ccy/results/diagonal_virasoro_series_level10.pkl
/private/tmp/theta_fermion_ccy_env/bin/python Code/theta_fermion_ccy/time_diagonal_ccy.py --level 15 --scope sample --json Code/theta_fermion_ccy/results/timing_diagonal_ccy_level15_sample.json --series Code/theta_fermion_ccy/results/diagonal_virasoro_series_level15_sample.pkl
/private/tmp/theta_fermion_ccy_env/bin/python Code/theta_fermion_ccy/time_high_branching_actions.py --json Code/theta_fermion_ccy/results/timing_level15_branching_samples.json --coefficients Code/theta_fermion_ccy/results/level15_branching_action_samples.pkl
```
