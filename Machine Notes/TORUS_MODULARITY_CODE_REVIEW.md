# Torus one-point modularity: code review index

This index collects the complete numerical path for the genus-one NS
one-point modularity check.  The production computation uses the exact
\(c=27/2\) finite part and BRY's \(dP/\pi\) spectral measure.  The current
highest cutoff is twice-level \(12\), i.e. through \(q^6\).

## Suggested review order

1. `superconformal_torus_blocks.py`
   - NS/R plumbing parameters and spin lifts.
   - Generic-\(b\) HJS toric recursion.
   - Exact-\(c=27/2\) finite-part wrappers
     `SelfDualNSTorusOnePointBlock` and
     `SelfDualRamondTorusOnePointBlock`.
2. `super_liouville_torus_one_point.py`
   - BRY-normalized \(dP/\pi\) spectral quadrature.
   - Assembly of the NS torus one-point function.
   - Reconstruction of the NS square-root lift from \(\tau\).
   - Direct versus \(S\)-transformed modular check.
3. `super_liouville_torus_modular_cluster.py`
   - Momentum-node sharding.
   - Reuse of each node's structure constant and block coefficients at all
     \(\tau\) values and cutoffs.
   - Idempotent JSONL shards, deterministic compensated reduction, accuracy
     gates, and command-line interface.
   - Artifact-schema-2 fingerprint of the numerical source closure and the
     Python, NumPy, and mpmath versions; cached reuse and reduction require
     an exact match.
4. `plot_torus_modularity_q_scan.py`
   - Verifies that the configuration snapshot and reduced summary have the
     same canonical SHA-256 digest before reading scan coordinates.
   - Plots only the requested largest recursion cutoff by default.
   - Recomputes the radial power-law fit above an explicit residual floor
     and writes the fitted exponent, normalization, and included-point
     ledger to a JSON sidecar.

## Mathematical dependencies

- `super_liouville_structure_constants.py`: BRY \(b=1\) NS structure
  constants and \(\Upsilon_1\) implementation.
- `superconformal_blocks.py`: central-charge and NS-weight conventions.
- `mixed_ramond_sphere_blocks.py`: NS degenerate weights, residues, and
  fusion polynomials reused by the toric recursion.
- `ramond_sphere_blocks.py`: Ramond momentum and weight conventions.
- `self_dual_superconformal_blocks.py`: rational-\(c\) finite-part
  extraction and diagnostics.

## Independent low-level checks

- `superconformal_torus_descendants.py`: brute-force descendant Gram/Ward
  construction.
- `test_superconformal_torus_blocks.py`: low-order recursion versus
  brute-force coefficients, NS/R plumbing, Ramond ground fiber, and
  exact-\(c\) finite-part checks.
- `test_superconformal_torus_two_point.py`: the one-point modular \(S\)
  benchmarks are at the end of this file; it also checks the
  nonprincipal NS lift.
- `test_super_liouville_torus_modular_cluster.py`: shard partition,
  deterministic reduction versus serial evaluation, and idempotence.

Run the relevant suite with

```bash
python3 -m unittest \
  test_superconformal_torus_blocks.py \
  test_superconformal_torus_two_point.py \
  test_super_liouville_torus_modular_cluster.py
```

## Cluster layer

- `config/type0b_torus_modular_cluster.json`: convergence and benchmark run.
- `config/type0b_torus_modular_q_scan_cluster.json`: 19-point \(\tau\) scan.
- `cluster/submit_type0b_torus_modular.sh`: snapshots the configuration and
  submits the array and dependent reducer.
- `cluster/type0b_torus_modular_array.slurm`: one single-threaded worker per
  array element.
- `cluster/type0b_torus_modular_reduce.slurm`: deterministic reduction and
  accuracy-gate job.
- `cluster/README_type0b_torus_modular.md`: local and Cannon runbook.

Inspect a configuration without computing:

```bash
python3 super_liouville_torus_modular_cluster.py \
  --config config/type0b_torus_modular_q_scan_cluster.json \
  plan
```

Regenerate the highest-cutoff figure from the retrieved Cannon record:

```bash
python3 plot_torus_modularity_q_scan.py
```

This writes a dependency-free SVG next to the retrieved summary.  The
committed PNG is the rasterized copy used by the LaTeX note.

## Cannon records

- `../Data Set/results/type0b_torus_modular_cluster/cannon_q12_20260723_v1/`
  contains the configuration snapshot, submission ledger, reduced summary,
  and reducer logs for jobs `34584457` and `34584458`.
- `../Data Set/results/type0b_torus_modular_cluster/cannon_qscan_20260723_v1/`
  contains the corresponding artifacts and final plot for jobs `34586419`
  and `34586420`.

The `summary.json` files, rather than the plot, are the numerical source of
truth.  These retrieved 2026-07-23 summaries predate implementation
fingerprinting and are therefore legacy records with verified configuration
digests, not fully source/environment-certified artifacts.  New shards and
summaries use artifact schema 2 and bind the source closure and numerical
environment.  The PNG is a deterministic presentation of the production
study at the largest configured recursion cutoff.
