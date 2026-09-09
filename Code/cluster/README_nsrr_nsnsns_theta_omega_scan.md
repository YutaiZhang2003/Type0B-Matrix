# Moderate NSRR / NSNSNS period-matrix scan

This is a separate replacement experiment, not a continuation or relabeling of
the incomplete order-eight run. Its old shards and frozen code are preserved.

The source period family is `Omega(t)=[[i,t+0.5i],[t+0.5i,i]]` at
`t=0.52,0.56,0.60,0.64,0.68`. The second chart uses the stored symplectic
transformation, including the integer basis change needed by its theta chart.
For each point both inverse plumbing maps are solved with holomorphic-one-form
collocation, checked at basis 24, and independently checked by the Schottky
cross-ratio period series at word length 8. The final collocation residual is
checked in the actual marked basis, without silently rounding integer periods.
The scan avoids the negative-real plumbing branch crossing at t=0.50.

## Numerical design

- Generic b=1.4, with the same three-point convention ledger as the previous run.
- NSRR: Ward-recursive branching coefficients; ordinary genus-two Virasoro
  c-recursion for each of the two factors; physical level 6, with level 4
  obtained by truncating the same coefficient series before evaluation.
- NSNSNS: collision-aware N=1 c-recursion at physical level 8. Its API counts
  twice-levels, so `recursion_order=16` (not 8).
- Maximum accepted branching Ward residual: 1e-5 instead of 1e-7. This is a
  solver acceptance threshold, **not** an error bound for the partition function.
- Momentum quadrature orders 4 and 6, each a full Cartesian product. For each
  channel a common per-edge Gaussian envelope covers all scan points; the
  actual primary factors remain point-dependent. The same branching and
  double-Virasoro coefficient data can therefore serve every geometry.
- 280 momentum nodes per channel, 560 immutable shards total. Each source
  shard includes both levels at all five points. No missing node is dropped.
- One fresh subprocess per node bounds momentum-dependent caches. Four
  70-node batches per channel, each with a five-hour wall-time limit.
- The physical free scalar+Majorana factor is recomputed in each local
  plumbing chart, using modes 36/44 as a stability check. The source physical
  spin change uses theta[01|10]/theta[00|10] at that source period matrix.
- Q=Z_SL/(Z_free)^[1+2(b+1/b)^2]. The cosmological prefactor is omitted on both
  sides, as explicitly recorded; no fitted normalization is applied.

## Prepare and submit

Generate the geometry config locally:

```sh
OPENBLAS_NUM_THREADS=1 python3 Code/genus_2/nsrr_nsnsns_theta_omega_scan.py \
  build-config --output Code/config/nsrr_nsnsns_theta_omega_scan_20260830.json
```

Submit with a new, unused remote run root:

```sh
bash Code/cluster/stage_submit_nsrr_nsnsns_theta_omega_scan.sh \
  cannon REMOTE_RUN_ROOT REMOTE_PYTHON
```

The wrapper stages a separate code snapshot and the existing StringMC/SymPy
runtime dependencies, runs remote regressions, and schedules an `afterok`
reducer. Check queue and the submission record before retrying any interrupted
submission: partial submission must not create duplicate jobs.

## Outputs

The reducer validates shard identity, configuration, implementation fingerprint,
quadrature node, finite sector values, and Ward residual. Only a complete valid
set produces `summary.json` and `summary.svg` in the run root. The SVG needs no
Matplotlib on the cluster. Its upper panel shows both Q curves for N=4/6; its
lower panel shows the raw modular ratio for NSRR levels 4/6 and N=4/6, against
the NSNSNS level-8 result. Unity is a reference line, not imposed on the data.
The summary records relative quadrature changes and source level changes.
These are convergence diagnostics, not certified error bars. Any discrepancy
must be reported as observed, not absorbed into a normalization constant.
