# Direct physical PBW versus CCY at total level ten

A fresh direct physical PBW calculation through total q-level ten completed on 2026-09-11. Its caches started empty and were reused within the run. It computes physical SCA Gram matrices and three-point forms directly, without an enlarged block, branching coefficients or a fermion division.

The parameters and sector match the saved CCY benchmark: b=7/5, P=(11/23,13/29,17/31), p=f=0, (eta,eta')=(+,-).

| Implementation | Arithmetic | Complete process | Internal timer |
| --- | --- | ---: | ---: |
| Direct physical PBW | 384-bit FLINT complex midpoints | 112.828279666 s | 110.189838417 s |
| CCY pipeline | Native binary64, exact rational auxiliary/vacuum factors | 92.569641041 s | 92.061963667 s |

CCY used 17.96% less complete-process time, a factor of 1.2188 in these implementation timings. The precisions and arithmetic backends differ: this comparison does not measure native-double PBW performance or establish a precision-matched algorithmic crossover.

Both complete-process clocks include imports, computation, result writes and process exit. The PBW internal clock includes earlier coefficient checkpoint writes but excludes imports and final cleanup. The CCY clock includes branching preparation and its intermediate files. These are single-run measurements, not averages.

The PBW result has 506 physical monomials and 4,048 parity slots in three plumbing variables. CCY retains 5,786 four-variable coefficient vectors through recovery, including unequal split powers. Both runs target the same physical block, but their intermediate work and output representations differ.

## PBW growth within this run

These are cumulative internal times at the end of each integer total level within the one fresh level-ten run, not separate repeated benchmarks. Intervening half levels are included.

| Through total level | Cumulative seconds |
| --- | ---: |
| 0 | 0.004566 |
| 1 | 0.008140 |
| 2 | 0.021811 |
| 3 | 0.069229 |
| 4 | 0.222830 |
| 5 | 0.684293 |
| 6 | 1.970342 |
| 7 | 5.520373 |
| 8 | 15.224872 |
| 9 | 41.166206 |
| 10 | 110.189838 |

## Previously measured absolute errors

The appropriate error for a coefficient whose exact value vanishes is absolute error; the existing scaling abs(delta)/max(1,abs(PBW)) already uses absolute error for reference magnitudes below one. Vanishing alone is not a reason to impose a tighter absolute criterion.

In the previously requested comparison through level six, the largest vanishing split-term error was 2.71353688e-06, while the largest error in a nonzero physical coefficient was 1.82702532e-07, a factor of 14.852. These are numerical results from that existing comparison. No new level-ten coefficient comparison was performed for the timing request.

Artifacts:

- [Timing comparison and source hashes](../Code/theta_fermion_ccy/results/timing_pbw_vs_ccy_level10.json).
- [Saved direct physical PBW coefficients through level ten](../Code/theta_fermion_ccy/results/pbw_level10.json).
- [PBW full-process timing](../Code/theta_fermion_ccy/results/pbw_level10_walltime.json).
- [CCY full-process timing](../Code/theta_fermion_ccy/results/ccy_schottky_machine_complete_level10_walltime.json).

Reproduce the PBW calculation from the repository root:

```sh
/private/tmp/theta_fermion_ccy_env/bin/python Code/theta_fermion_ccy/pbw_reference.py --level 10 --json Code/theta_fermion_ccy/results/pbw_level10.json
```

The complete-process measurement wraps this command with a parent process timer and writes stdout to results/pbw_level10.log. Only the wrapper's authorization cutoff was extended from six to ten; the physical PBW computation and precision were unchanged. The paper draft was not edited.
