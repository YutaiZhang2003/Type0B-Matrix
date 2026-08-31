# Completed one-hour preliminary run

Cannon job **43256507** completed successfully (`COMPLETED`, exit `0:0`) in
**17 minutes 46 seconds**, including four workers, reduction and comparison.
The configuration is `Code/config/type0b_ns_five_tachyon_one_hour_preliminary.json`.
All four outputs were retrieved and the combined result was verified locally.

For outgoing energies `omega_a=0.25+0.02i` and incoming `Omega=1+0.08i`,
the coefficient of `delta(E) mu_F^-3` in the literal all-tachyon amplitude is

```
Worldsheet: (-22.0549 +/- 9.2108) + i (24.2877 +/- 9.3535)
Matrix:     -0.01581584458368 - 0.017243732128 i
```

Errors above are component standard errors from eight independent RQMC
replicates. They do not include block, momentum, collar or other systematic
errors. The coarse estimate is far from the matrix value and does not resolve
the predicted amplitude. This is not evidence for a physical disagreement.
No epsilon-to-zero limit was used or required for this comparison.

The unnormalized integral components were

| Contribution | Real | Imaginary |
| --- | ---: | ---: |
| Bulk mean | 138.513529 | 145.031167 |
| Face mean | -142.568240 | -117.960260 |
| Corner term | 243.764676 | 190.602252 |
| Total | 239.709965 | 217.673160 |

`summary.json` preserves the independent worldsheet reduction;
`comparison.json` contains the separate matrix comparison. `shards/` holds
the original four worker outputs. The submission record is the original
historical record; `completion_record.json` records the final status and
checksums without altering any of those source artifacts.
