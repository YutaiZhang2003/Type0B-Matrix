# Machine-precision direct PBW timing through level ten

The fresh direct physical SCA PBW run completed using native Python complex scalars and NumPy complex128 Gram matrices (53-bit significands). The existing physical Ward identities and bilinear Gram contractions are reused; no numerical terms are clipped. No enlarged block or branching calculation enters this direct computation.

Parameters: b=7/5, P=(11/23,13/29,17/31), p=f=0, (eta,eta')=(+,-).

| Calculation | Complete process | Internal timer |
| --- | ---: | ---: |
| Direct physical PBW, binary64 | 33.550880583 s | 31.402591083 s |
| Full CCY pipeline, binary64 | 92.569641041 s | 92.061963667 s |

PBW was 2.759 times faster in these runs. Both complete-process clocks include imports, computation, result writes and exit. Both runs start without cached coefficients and reuse intermediates within the run. CCY includes branching, the Schottky vacuum factor, direct fermion factor and convolution recovery; its auxiliary and vacuum factors use exact rational arithmetic. These are single-run implementation timings.

The PBW output contains 506 physical monomials and 4,048 parity slots in three plumbing variables. CCY retains 5,786 four-variable coefficient vectors through recovery. No new coefficient comparison or accuracy validation was performed for this timing request.

## Cumulative PBW time

Times below come from the single fresh level-ten run, including intervening half levels and earlier checkpoint writes. They are not separate runs at each cutoff.

| Through total level | Internal seconds |
| --- | ---: |
| 0 | 0.000404 |
| 1 | 0.002753 |
| 2 | 0.009564 |
| 3 | 0.031860 |
| 4 | 0.101005 |
| 5 | 0.289410 |
| 6 | 0.803534 |
| 7 | 2.097862 |
| 8 | 5.284844 |
| 9 | 12.982122 |
| 10 | 31.402591 |

## Saved files

- [Physical PBW coefficients](../Code/theta_fermion_ccy/results/pbw_machine_level10.json)
- [PBW complete-process timing](../Code/theta_fermion_ccy/results/pbw_machine_level10_walltime.json)
- [Timing comparison and source hashes](../Code/theta_fermion_ccy/results/timing_machine_pbw_vs_ccy_level10.json)
- [Saved CCY complete-process timing](../Code/theta_fermion_ccy/results/ccy_schottky_machine_complete_level10_walltime.json)

Reproduce the computation from the repository root:

```sh
/private/tmp/theta_fermion_ccy_env/bin/python Code/theta_fermion_ccy/pbw_reference.py --level 10 --machine --json Code/theta_fermion_ccy/results/pbw_machine_level10.json
```

The complete-process measurement wraps this command in a parent timer and captures stdout in pbw_machine_level10.log. The manuscript was not edited.
