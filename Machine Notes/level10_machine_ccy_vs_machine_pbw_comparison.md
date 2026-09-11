# Machine-precision CCY versus machine-precision PBW through level ten

Compared the two saved complete level-ten results, both computed with native binary64 numerical arithmetic. Parameters: b=7/5, P=(11/23,13/29,17/31), p=f=0, (eta,eta')=(+,-). The CCY pipeline uses exact rational auxiliary and Schottky factors; the direct physical PBW backend uses native complex scalars and complex128 Gram matrices without clipping small coefficients.

The comparison covers all 46,288 parity slots in 5,786 four-variable coefficient vectors: 4,048 equal-split slots corresponding to the 506 physical PBW monomials, and 42,240 unequal-split slots whose physical coefficients are zero. No terms are removed or projected out.

For PBW twice-level labels (a,b,c), the corresponding CCY exponent key is (a,b/2,b/2,c). A CCY key (a,l,r,d) denotes q1^(a/2) q2_left^l q2_right^r q3^(d/2), with q2=q2_left*q2_right. Equal split powers therefore reproduce the physical three-variable monomials; unequal powers test residual dependence on the split.

## Coefficient differences

Every row gives cumulative maxima through the stated total physical level, including intervening half levels. The scaled difference is abs(CCY-PBW)/max(1,abs(PBW)); it is an absolute difference below unit PBW magnitude and a relative difference above it. These are differences between the two numerical calculations, not errors certified against an exact reference.

| Through level | Physical maximum absolute difference | Physical maximum scaled difference | Maximum unequal-split residual |
| --- | ---: | ---: | ---: |
| 0 | 1.00000000e-15 | 1.00000000e-15 | 0.00000000e+00 |
| 1 | 5.34550439e-15 | 5.34550439e-15 | 2.33626435e-14 |
| 2 | 1.06330000e-13 | 1.06330000e-13 | 4.58526694e-11 |
| 3 | 9.19444018e-12 | 9.19444018e-12 | 1.63751507e-10 |
| 4 | 3.98500800e-10 | 3.98500800e-10 | 7.95964025e-09 |
| 5 | 6.88658184e-09 | 6.88658184e-09 | 1.38859706e-07 |
| 6 | 1.82702533e-07 | 1.35521710e-07 | 2.71353688e-06 |
| 7 | 2.48965664e-06 | 1.32409874e-06 | 3.42196015e-05 |
| 8 | 4.22786099e-05 | 4.21506963e-06 | 5.88856607e-04 |
| 9 | 4.18947792e-04 | 5.29427722e-05 | 5.97147807e-03 |
| 10 | 3.31413271e-02 | 1.01312322e-03 | 1.24444631e-01 |

The largest physical absolute and scaled differences both occur at exponent key (4,6,6,4), parity index 0, corresponding to the q1^2 q2^6 q3^2 physical monomial:

- CCY: -32.74518100811575 + 0.00000011253586175885934 i.
- PBW: -32.71203968104746 + 0.0 i.
- Absolute difference: 0.03314132707.
- Relative difference for this coefficient: 0.101312%.

The largest unequal-split residual occurs at key (4,8,4,4), parity index 0. CCY gives 0.12444463053007004 + 1.4564061485410184e-6 i, while the physical block requires zero. Its magnitude is 0.1244446305, or 3.7550 times the largest physical absolute difference. Both are assessed on an absolute scale when making this comparison.

Only the requested comparison of the saved machine-precision results was performed. Error statistics use 70-digit arithmetic to subtract the stored decimal strings; this does not increase either block's computation precision. Comparison runtime: 1.239455 s. No block was recomputed and no comparison to the saved high-precision PBW result was performed.

## Saved artifacts

- [Full comparison with per-level maxima and worst coefficients](../Code/theta_fermion_ccy/results/comparison_machine_ccy_vs_machine_pbw_level10.json).
- [Machine-precision physical PBW coefficients](../Code/theta_fermion_ccy/results/pbw_machine_level10.json).
- [Machine-precision CCY physical block](../Code/theta_fermion_ccy/results/ccy_schottky_machine_complete_level10.json).
- [Timing comparison: PBW 33.55 s, CCY 92.57 s](level10_machine_pbw_vs_ccy_timing.md).

Reproduce the comparison from the repository root:

```sh
/private/tmp/theta_fermion_ccy_env/bin/python Code/theta_fermion_ccy/compare_saved_pbw.py --production Code/theta_fermion_ccy/results/ccy_schottky_machine_complete_level10.json --pbw Code/theta_fermion_ccy/results/pbw_machine_level10.json --level 10 --json Code/theta_fermion_ccy/results/comparison_machine_ccy_vs_machine_pbw_level10.json
```
