# Independent level-10 plumbing cutoffs

Keep q1^(a/2) q2^l q3^m for 0 <= a <= 20 and 0 <= l,m <= 10. There are 2,541 monomials and 20,328 parity components, including the corner q1^10 q2^10 q3^10. This is not a total-level-10 truncation.

Both sectors use 40 decimal digits (136 bits), b=7/5, (P1,P2,P3)=(11/23,13/29,17/31), p=f=0. Ordinary uses (+,+); inserted uses (+,-). Cases run sequentially as fresh processes with empty numerical caches. Compilation and validation are excluded from the measured wall times. Intermediate branching coefficients are stored and reused within each run.

Before zero residues or zero branch prefactors are removed, the independent domains permit at most 42,861,104 global-seed summands for ordinary and 1,740,820,224 for inserted. For a single edge with remaining budget L, the number of pairs (target, accumulated shift) is L+1+L(L-1)/2, since the shift is zero or at least two. Multiply these counts across edges and sum over branches, counting each transposed inserted product only once. These are work bounds, not elapsed-time estimates; actual counters are recorded below.

## Measurements

| Sector | Status | Internal total (s) | Full wall (s) |
| --- | --- | ---: | ---: |
| ordinary | completed | 416.537 | 416.619 |
| inserted | completed | 1655.279 | 1655.360 |

The optimized measurements use complete inserted-vertex caches, reused MPC temporary storage, and cached inverse denominators with common residues factored outside the incoming sums. Precision and retained coefficient domains are unchanged. See ../ccy_vertex_reuse_2026-09-11/README.md for directed validation.

The original ordinary run completed in 508.004 s. The original inserted run was interrupted after 1,608.892 s for this restart; that duration is not a completed block timing. Their records remain in full_pipeline_40dps. The invalid_overlap_attempt directory contains an aborted overlapping attempt and is excluded from every measurement and comparison.

| Stage | ordinary | inserted |
| --- | ---: | ---: |
| Initial mode actions | 9.497 | 9.594 |
| Outer branching recurrence | 0.441 | 0.450 |
| Middle recurrence | 0.000 | 0.013 |
| Reduced CCY | 296.593 | 1460.243 |
| Virasoro products | 26.759 | 88.559 |
| Branch assembly | 0.356 | 0.577 |
| Direct fermion | 5.220 | 5.221 |
| Sector division | 3.115 | 3.130 |
| Schottky vacuum and square | 72.379 | 72.492 |
| Vacuum restoration | 0.375 | 0.380 |
| Other pipeline overhead | 1.801 | 14.620 |
| Startup, serialization and exit | 0.081 | 0.081 |

All stage times are seconds. Internal total is timed independently; other pipeline overhead is its difference from the individually timed stages, including work and cache cleanup outside those timers. Wall additionally includes startup, final serialization, cleanup and exit.

## Output and diagnostics

- ordinary: complete 2,541-monomial box; 899 branch tuples; 1,798 Virasoro blocks; 42,624,112 global-seed summands; 51,242,176 CCY transitions. Maximum recorded recovery-sector residual: 2.226e-15.
- inserted: complete 2,541-monomial box; 1,591 branch tuples; 1,620 Virasoro blocks; 1,740,820,224 global-seed summands; 839,918,556 CCY transitions. Maximum recorded recovery-sector residual: 8.837e-16.

Runs use sector-policy record. Residuals test consistency, not every coefficient's accuracy. No new physical PBW block or independent full-box reference was computed.

Inspection of the saved files confirms all 20,328 physical components in each sector are finite and the complete requested monomial domains are present. Source and executable hashes match those recorded at launch.

Comparing all ordinary physical components with the already saved pre-optimization 40-digit result gives maximum scaled difference 2.107e-20 and maximum absolute difference 8.907e-09. The scale is max(1, abs(reference), abs(current)); coefficients reach 4.166e+12. This is a same-precision implementation comparison, not independent accuracy certification. The inserted full box has no completed pre-optimization reference; its directed small-cutoff and individual-factor comparisons are documented in ../ccy_vertex_reuse_2026-09-11/README.md. This inspection only reads saved results and evaluates no blocks.

## Count-based estimates

At 267 completed inserted tuples (840.1 s elapsed), replaying the weighted branch domains estimated 4.49 hours for the full inserted 40-digit run with the former CCY implementation. This historical forecast assumes similar seconds per seed term; it does not describe the optimized run reported above.

The physical PBW box has 450,469 distinct Gram entries, the same as at total level 10. It instead requests 1,318,075,412 three-point tensor entries and 254,809,434,024 dense-contraction complex multiplications. Its largest single tensor has 8,665,664 entries. These counts do not evaluate any PBW block.

### PBW at matched 40-digit precision (136 bits)

| Saved model, recalibrated to 136 bits | Positive sector (hours) | Negative sector (hours) |
| --- | ---: | ---: |
| lean | 28.8–63.4 | 57.6–63.4 |
| full | 97.3–196.0 | 119.3–196.0 |

These are estimates for the existing Python PBW implementation. Synthetic operations using its actual FLINT scalar and matrix wrappers were timed at 136 and 384 bits; the measured precision ratios rescale the previously saved PBW feature fits. No physical PBW block, Gram matrix, or Ward identity was computed. The short arithmetic calibration took 1.30 s while the inserted run continued.

Measured 136/384-bit ratios were about 0.91–0.92 for wrapped scalar operations and 0.51–0.73 for matrix products. The estimates allow unchanged Python/cache overhead, expose the spread between the lean and full feature models, and assume sufficient memory. They do not certify full-block performance: larger matrix shapes and the persistent Ward cache can change the costs. Both sides use 136 bits, but their implementations differ (C++ MPC versus Python FLINT). Raw timings and assumptions are in pbw_40digit_arithmetic_calibration.json.

## Implemented truncation

Subtract primary levels separately from each edge limit. The ordinary CCY domains are three-dimensional boxes. The inserted domains are four-dimensional boxes; retain unequal middle levels inside each factor and assemble only equal final middle levels after primary shifts. Convolution and Schottky products truncate componentwise. Schottky walks are bounded by both total length and edge visits. Direct fermion occupation levels are independently bounded.

## Cutoff validation

At independent cutoff 2, both full physical pipelines were compared with all 45 corresponding monomials in the saved total-level-10 40-digit results. Maximum scaled differences are 1.55e-36 (ordinary) and 6.00e-36 (inserted). Exact arithmetic checks additionally compare the Schottky boxes at cutoffs 2 and 3 and both fermion boxes at cutoff 2 with projections of their total-level expansions. A shifted-diagonal product is compared with the unrestricted product at a small asymmetric middle cutoff. All pass.

## Reproduction

    make -C C++
    python3 C++/tools/time_current_pipelines.py --levels 10 --truncation per-edge --dps 40 --output /tmp/ramond_per_edge10

The optimized_full_pipeline_40dps directory contains commands, source/executable hashes, logs, stage timings, and all output coefficients.
