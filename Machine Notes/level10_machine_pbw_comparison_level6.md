# Physical PBW comparison through level six

The saved complete level-ten machine-precision CCY result was truncated to total physical q-level six and compared with a fresh independent physical SCA PBW calculation through level six. The CCY block was not recomputed. Parameters are b=7/5, P=(11/23,13/29,17/31), p=f=0, (eta,eta')=(+,-).

The reference uses 384-bit FLINT complex midpoint arithmetic with the inherited 1e-80 scalar clipping cutoff. It directly contracts physical SCA Gram matrices and three-point forms; it does not divide an enlarged block by an auxiliary factor. The comparison uses 70-digit arithmetic to subtract saved coefficient strings.

Compared 8,288 parity slots in 1,036 four-variable coefficient vectors. No coefficients were projected out. PBW coefficients at twice-levels (a,b,c) were embedded at (a,b/2,b/2,c); unequal split powers were compared to zero.

Errors are scaled as abs(machine-PBW)/max(1,abs(PBW)). Each table entry is the maximum over all coefficients **through** the indicated total level, including intervening half levels.

| Through total level | Maximum scaled error |
| --- | ---: |
| 0 | 1.00000000e-15 |
| 1 | 2.33626435e-14 |
| 2 | 4.58526694e-11 |
| 3 | 1.63751507e-10 |
| 4 | 7.95964025e-09 |
| 5 | 1.38859706e-07 |
| 6 | 2.71353688e-06 |

Maximum absolute error: 0.000002713536875051450103320106931378045373210928255743526601512509832046636. It equals the maximum scaled error here because the worst coefficient has zero PBW value.

For equal split-edge powers alone (1,120 slots), the maximum scaled error is 1.35521710e-07, and the maximum absolute error is 1.82702532e-07. For unequal split powers (7,168 slots), the largest residual is 2.71353688e-06.

The worst overall slot has exponents (4,3,1,4), parity index 0, at total level six. Its machine value is approximately 2.71353687505145e-6 - 2.3679650339164776e-14 i, while PBW gives zero. The worst equal-split slot is (4,2,2,4), parity index 0: machine 1.3481419934132068 - 1.6970174970636053e-12 i versus PBW 1.34814217611573917584.

These are errors of the low-level coefficients obtained from the level-ten run; the larger branching system means they need not equal the errors of a separate computation truncated at level five.

The fresh PBW reference took 2.766599167 seconds including imports and result writes (1.974445959 seconds on its internal timer). The coefficient comparison took 0.422460708 seconds internally. No split-parameter evaluations or other CFT cross-checks were performed.

Artifacts:

- [Full coefficient comparison, worst slots and per-level maxima](../Code/theta_fermion_ccy/results/comparison_machine_level10_vs_pbw_through_level6.json).
- [Fresh physical PBW reference](../Code/theta_fermion_ccy/results/pbw_level6.json).
- [PBW process timing](../Code/theta_fermion_ccy/results/pbw_level6_walltime.json).

Reproduce from the repository root:

```sh
/private/tmp/theta_fermion_ccy_env/bin/python Code/theta_fermion_ccy/pbw_reference.py --level 6 --json Code/theta_fermion_ccy/results/pbw_level6.json
/private/tmp/theta_fermion_ccy_env/bin/python Code/theta_fermion_ccy/compare_saved_pbw.py --production Code/theta_fermion_ccy/results/ccy_schottky_machine_complete_level10.json --pbw Code/theta_fermion_ccy/results/pbw_level6.json --level 6 --json Code/theta_fermion_ccy/results/comparison_machine_level10_vs_pbw_through_level6.json
```
