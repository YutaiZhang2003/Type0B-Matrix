# Where machine-precision CCY spends time

The saved full level-ten pipeline spent 77.430810 s evaluating the CCY series, out of 92.569641 s complete-process time. The Schottky vacuum factor took 0.041195 s.

A fresh cProfile measurement isolated one actual branch with labels (0,-1/4,1/4,-1/4), Virasoro copy 1, and remaining weighted cutoff 20. All numerical arithmetic was native binary64. It computed 3,146 output coefficients in 3.328482 s with profiling enabled. This is a profile of one maximum-cutoff branch, not a new full-pipeline benchmark. Percentages include profiling overhead and should not be assigned directly to the full run.

| Work | Profiled seconds | Share |
| --- | ---: | ---: |
| Recursive coefficient loop, exclusive time | 2.482736 | 74.59% |
| Pole transitions including residue evaluation | 0.395093 | 11.87% |
| Global-block coefficients including three-point factors | 0.174238 | 5.23% |
| Other calls and bookkeeping | 0.276415 | 8.30% |

The coefficient loop alone accounts for the largest exclusive cost. It iterates over eligible null-vector residues for each shifted weight/level state, retrieves cached transitions and lower coefficients, and accumulates their products. This branch required 556,736 coefficient-cache misses. Across the saved complete level-ten run, the corresponding count was 22,129,076, plus 36,836,320 cache hits. These are states of the c-recursion, not a sum over a PBW basis.

The transition row includes residue evaluation. Actual residue evaluation (_pole_uncached including fusion polynomials and pole geometry) contributed 0.046026 s within that row. The global row uses the closed SL(2) three-point formulas for shifted weights; the Schottky vacuum factor is accounted for separately in the full-pipeline timing above.

No coefficient comparison, accuracy check, or full-pipeline rerun was performed for this diagnosis.

- [Native CCY profile](../Code/theta_fermion_ccy/results/profile_ccy_machine_cut20.json).
- [Raw cProfile data](../Code/theta_fermion_ccy/results/profile_ccy_machine_cut20.prof).
- [Saved full-pipeline segment timings](../Code/theta_fermion_ccy/results/ccy_schottky_machine_complete_level10_summary.json).
