# Type-0B torus modular cluster run

This workflow parallelizes the genus-one spin-resolved one-point modular
checks over spectral momentum nodes. Each node computes its BRY structure
constants and exact-\(c=27/2\) finite-part recursions once, then reuses those
coefficients for every configured \(\tau\) and recursion cutoff. Shards never
append to a shared file, and reduction always sums contributions in
quadrature-node order with compensated real/imaginary summation.

## Accuracy plan

The default configuration is
`config/type0b_torus_modular_cluster.json`. It contains 352 atomic momentum
jobs split evenly over 16 shards:

- recursion cutoffs at twice-levels \(6,8,10,12\);
- the principal benchmark \(\tau=0.2+0.9i\);
- the nonprincipal spin-lift benchmark \(\tau=0.45+0.65i\);
- explicit order-12 modular-residual gates in both lift sectors;
- 64-to-96 point spectral-quadrature convergence;
- \(P_{\max}=4.5\) to \(5.5\) tail convergence;
- 32-to-40 finite-part contour samples;
- 50-decimal structure-constant evaluation.

The companion configuration
`config/type0b_torus_ns_tilde_r_cluster.json` has the same convergence
studies for the modular orbit
\(\widetilde{\mathrm{NS}}\leftrightarrow\mathrm R\).  Its direct frame uses
the reversed NS half-level lift, while the transformed frame uses the
ordinary HJS \(+\) Ramond block with the BRY spectral coefficient
\(2C_{\rm even}\).

Inspect the frozen plan without doing numerical work:

```bash
python3 super_liouville_torus_modular_cluster.py plan
```

## Local parallel validation

```bash
python3 super_liouville_torus_modular_cluster.py local \
  --output-dir /path/to/run/shards \
  --workers 8 \
  --execute

python3 super_liouville_torus_modular_cluster.py reduce \
  --input-dir /path/to/run/shards \
  --output /path/to/run/summary.json
```

Completed shards are idempotent: rerunning the local command skips files
whose configuration digest, implementation fingerprint, artifact schema,
and node ledger are complete.  The implementation fingerprint covers the
numerical Python source closure and the Python, NumPy, and mpmath versions.
Changing any of them invalidates cached shards.  Use `--force` only when an
intentional recomputation is required.

## Slurm launch

The checkout and run directory must both be visible from the compute nodes.
The Python environment needs NumPy and mpmath.

```bash
TYPE0B_PARTITION=yin TYPE0B_ARRAY_CAP=16 \
  cluster/submit_type0b_torus_modular.sh \
  /shared/path/type0b-torus-run \
  /shared/path/python
```

The submission script snapshots the JSON configuration, creates the shard
and log directories, submits the array, then submits the reducer with an
`afterok` dependency. The reduction job exits nonzero if any configured
accuracy target fails. Its `submission.json` records both Slurm job IDs.

To use a modified configuration, pass it as the third argument. Change the
number of array tasks through `default_shard_count` in that configuration;
use `TYPE0B_ARRAY_CAP` only to limit concurrent tasks.

## Validated default anchors

An eight-process simulation of the complete default plan gave:

| diagnostic | observed |
|---|---:|
| principal-frame order-12 modular residual | \(1.5543\times10^{-15}\) |
| nonprincipal-lift order-12 modular residual | \(1.1328\times10^{-10}\) |
| 64-to-96 node maximum relative change | \(1.17\times10^{-15}\) |
| \(P_{\max}:4.5\to5.5\) maximum relative change | \(1.46\times10^{-15}\) |
| production two-radius diagnostic | \(7.75\times10^{-14}\) |

The nonprincipal residual is recursion-truncation dominated; its correct
spin lifts are \((+1,-1)\). The two-radius number is a stability diagnostic,
not a rigorous absolute-error bound.

## Cannon production record

The default plan was run on Cannon on 2026-07-23:

- momentum array: Slurm job `34584457`;
- deterministic reducer: Slurm job `34584458`;
- partition: `serial_requeue`;
- all 16 array elements and the reducer completed with exit code `0:0`;
- remote run root:
  `/n/home09/yutaizhang/Type0B-Matrix-runs/torus-modular-q12-20260723-v1`;
- summary SHA-256:
  `29a63a44f0ed53fbc593694ca0b97e8014be368b7ed60efc24f68728c5f83512`.

For the production study at twice-level 12, Cannon gives

\[
\begin{aligned}
G(0.2+0.9i)&=0.05462332460780035,\\
G(-1/(0.2+0.9i))&=0.049916537550511134,
\end{aligned}
\]

with modular residual \(1.44329\times10^{-15}\). At
\(\tau=0.45+0.65i\), with lifts \((+1,-1)\), the corresponding values are
\(0.06179554626182613\) and \(0.04761928293035410\), with residual
\(1.13284\times10^{-10}\).

The retrieved legacy record is under
`../Data Set/results/type0b_torus_modular_cluster/cannon_q12_20260723_v1/`.
It predates artifact-schema-2 implementation fingerprinting, so its
configuration digest is verified but its source and environment are not
cryptographically bound.  New reductions record the complete fingerprint
in every shard and in `summary.json`.

## Cannon q-scan record

A second Cannon run scanned 19 modular parameters in two families:

- \(\operatorname{Re}\tau=0.20\), with
  \(0.45\leq\operatorname{Im}\tau\leq1.40\);
- \(\operatorname{Re}\tau=0.45\), including points on both sides of the
  transformed-lift boundary
  \(\operatorname{Im}\tau=\sqrt{0.6975}=0.83516\ldots\).

The array job `34586419` and reducer `34586420` completed all 160 momentum
jobs with zero exit status. The 64-to-96 node comparison changed the
correlators by at most \(3.19\times10^{-15}\).

The very small residual at the original benchmark is not generic. At
\(\tau=0.2+0.45i\), where
\(\max(|q|,|\widetilde q|)=0.05916\), the order-12 residual is
\(1.98\times10^{-7}\). It decreases through
\[
1.98\times10^{-7},\quad
3.00\times10^{-9},\quad
4.66\times10^{-11},\quad
7.37\times10^{-13},\quad
1.24\times10^{-14}
\]
as \(\operatorname{Im}\tau\) runs from \(0.45\) to \(0.85\), before reaching
the floating-point floor near the modularly balanced region.  With the
explicit exclusion rule \(\delta_{\rm mod}>10^{-15}\), a log-log fit retains
seven radial points and gives
\[
 \delta_{\rm mod}=16.0691\,q_{\max}^{6.461873097\ldots}.
\]
This is consistent with the expected first omitted NS half-level
\(q^{6.5}\).  The fit is regenerated from `summary.json` by
`plot_torus_modularity_q_scan.py` and recorded in
`torus_modularity_q_scan.fit.json`.

The scan is smooth through the transformed-lift change
\((+1,-1)\to(+1,+1)\); no branch jump is visible. The retrieved data and plot
are under
`../Data Set/results/type0b_torus_modular_cluster/cannon_qscan_20260723_v1/`.

The historical q-scan summary tested quadrature convergence and finite-part
stability but did not contain a physics-level modularity gate.  The current
q-scan configuration adds representative order-12 gates at
\(\tau=0.20+0.85i\) in the principal lift and
\(\tau=0.45+0.65i\) in the nonprincipal lift.  A fresh cluster run is
required before calling a q-scan record fully certified under the new
artifact schema.

## Cannon NS-tilde/R production record

The spin-orbit configuration was run on Cannon on 2026-07-23:

- momentum array: Slurm job `34613287`;
- deterministic reducer: Slurm job `34613300`;
- all 16 array elements and the reducer completed with exit code `0:0`;
- configuration digest:
  `b604e5f67d6c465128225f882ba3b609a285334846fef8d112f00f2853005152`;
- Cannon implementation fingerprint:
  `ded96544c4944a2bd183521659b29d83ce2647bb18610263dd2bb34849c89fe1`;
- summary SHA-256:
  `aef563fb0d6a28f898b914a1dbe3eb9f379588160bb713506f5288ed885ff0c3`.

The production order-12 residual is \(8.88\times10^{-15}\) at
\(\tau=0.2+0.9i\) and \(1.01\times10^{-10}\) at the nonprincipal point
\(\tau=0.45+0.65i\).  The 64-to-96 node and \(P_{\max}=4.5\to5.5\)
comparisons are \(4.62\times10^{-15}\) and \(9.84\times10^{-16}\),
respectively.  All six configured gates pass.  The frozen configuration,
submission record, summary, and reducer logs are stored under
`../Data Set/results/type0b_torus_modular_cluster/cannon_ns_tilde_r_20260723_v1/`.
