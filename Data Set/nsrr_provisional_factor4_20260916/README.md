# Provisional factor four: genus-two convergence test

**Assumption:** `M_provisional = 4 M_local`. This changes the coefficient `B_L B_R/8` to `B_L B_R/2` in the new Human-Note bilinear NSRR matrix. It is applied once to the entire matrix, including interference. BRY constants, descendant blocks, spin phases, measure and channel-specific primary powers are unchanged.

The local BPZ kernel remains the reference. No derivation of the extra global factor is claimed. The old interference matrix is not used.

Comparison: $R=(Z_{NSRR}/Z_{NSNSNS})(Z_{free,t}/Z_{free,s})^\kappa$, with $b=1.4$, $\kappa=9.940408163265307$; agreement means $R=1$.

## Quadrature at fixed source L5 / target R12

Both source and target are complete grids. The source is held at N7 after its two successive changes fall below 1e-4. The target is continued to N10. These are the saved complex matrices, newly scaled by the explicit assumption.

| Source N | Target N | R(s=+1) | R(s=-1) | R(spin sum) |
|---:|---:|---:|---:|---:|
| 3 | 3 | 1.0117029819 | 1.0113455212 | 1.0115168986 |
| 4 | 4 | 1.0092264740 | 1.0085379022 | 1.0088681032 |
| 5 | 5 | 1.0031307822 | 1.0022499261 | 1.0026723365 |
| 6 | 6 | 1.0010232037 | 1.0000928466 | 1.0005389870 |
| 7 | 7 | 1.0004326815 | 0.9994857321 | 0.9999398248 |
| 7 | 8 | 1.0002727510 | 0.9993187135 | 0.9997762035 |
| 7 | 9 | 1.0002256703 | 0.9992691195 | 0.9997278140 |
| 7 | 10 | 1.0002103786 | 0.9992529741 | 0.9997120778 |

## Saved block-order convergence

All eight complex channel/parity coefficient tables are retained through total level 8 on both the adapted N7 grid (343 nodes) and panel N12 grid (1728 nodes). They are independent of q and were evaluated on the same five saved surfaces with the new bilinear pairing. The table fixes the central surface t=0.60 and the all-NS N10/R12 target, using panel N12 for the source. L is total descendant level; R is the target recursion twice-level parameter.

| Source L | R(s=+1) | R(s=-1) | R(spin sum) | Largest absolute nodal order change / Z |
|---:|---:|---:|---:|---:|
| 3 | 0.9999375694 | 0.9991169498 | 0.9995104611 | — |
| 4 | 1.0002406278 | 0.9992798038 | 0.9997405473 | 4.177e-04 |
| 5 | 1.0002014989 | 0.9992453421 | 0.9997038475 | 5.896e-05 |
| 6 | 1.0002080669 | 0.9992517683 | 0.9997103417 | 8.945e-06 |
| 7 | 1.0002068305 | 0.9992506908 | 0.9997091880 | 1.503e-06 |
| 8 | 1.0002070665 | 0.9992508688 | 0.9997093938 | 2.668e-07 |

The absolute nodal diagnostic sums absolute changes before momentum integration, so cancellations between momentum nodes cannot conceal that measured finite-order difference.

### Target order control

| Target R step (N3) | Largest scalar relative change | Matrix relative change |
|---|---:|---:|
| 8 to 12 | 3.915e-07 | 2.388e-07 |
| 12 to 16 | 1.206e-09 | 7.852e-10 |

### Independent source quadrature control at L5

- `panel_N12_vs_Laguerre_N7_at_L5`: largest scalar change `8.878e-06`.
- `panel_N12_vs_adapted_N7_at_L5`: largest scalar change `5.705e-06`.

### Saved-data checks

Three existing panel-grid nodes (head, bulk and tail) were recomputed through L8 with the current native engine, evaluating all four equal-eta channels directly, including f=1 without parity reconstruction. The largest scaled coefficient discrepancy is `3.308e-34`; the projected complex blocks agree at saved floating-point precision. The supplied BRY constants agree with the independent 40-digit evaluator to `4.108e-14` relative. This is a selected-node validation, not a fresh recomputation of all 2071 bank nodes.

9 tests pass, covering independent complex anti data, all tube signs, the normalization at the resummed API boundary, and existing local BPZ/primary bookkeeping.

Across all five saved surfaces on the complete panel N12 source grid, the largest absolute nodal L7-to-L8 change divided by Z is `9.247e-07`. Only the central surface has the refined target quadrature used below.

## Result and phase

| Comparison | Ratio at source L8 / target R12 | Residual (%) |
|---|---:|---:|
| s=+1 | 1.0002070665 | +0.020707 |
| s=-1 | 0.9992508688 | -0.074913 |
| resolved_00 | 0.9990667801 | -0.093322 |
| resolved_11 | 1.0003698399 | +0.036984 |
| spin_sum | 0.9997093938 | -0.029061 |

The off-diagonal integrated-spin-matrix phase difference is `-0.0004796613033` rad; the trace-normalized matrix distance is `0.000571542`. Multiplication by positive four leaves phases unchanged. These matrix phases are distinct from an overall chiral Pfaffian phase. The tested real-slice scalar partition functions are positive.

**The factor removes the gross normalization discrepancy, but residual differences remain.** Block and quadrature convergence must not be identified with a proof of the normalization or the interacting spin transport.

## Scope and reproduction

- Factor four is a user-requested assumption, not a derivation of the global sewing normalization.
- Interacting spin transport still uses the minimal extension of the checked free-spin dictionary.
- Target R12-to-R16 control is at N3; it is not an R16 recomputation of the N10 integral.
- Other four surfaces have higher source block data but only coarse N3 target quadrature.
- Observed changes are finite convergence diagnostics, not rigorous remainder bounds.

`result.json`, `quadrature.csv`, `block_orders.csv`, `source_banks.json` and `provenance.json` retain numerical results and input hashes. `target_R16_N3/` contains the complete fresh target control. The historical factor-one studies remain intact.

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  python3 Code/genus_2/check_nsrr_factor4_convergence.py --stage all --workers 4
```

For the pointwise resummed worker select `--sewing-convention human-bilinear --normalization provisional-times-four --physical-lifts-slots 1 -1 1` (or -1 -1 1 for the other tested sign). That legacy frozen worker retains its supplied target; use the spin-matrix comparison in this report for crossing.
