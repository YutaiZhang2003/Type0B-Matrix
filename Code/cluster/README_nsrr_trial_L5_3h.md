# Three-hour level-five NSRR trial package

**Prepared, not submitted.** This is the current factorized-sign trial, not
the retired NSRR assembler or a certified physical fixed-spin partition.
The protected kernels, original trial runner, and previous data are untouched.

## Calculation

- `b=1.4`; the existing five surfaces at `t=0.52,0.56,0.60,0.64,0.68`.
- Total **chiral** descendant cutoff `L=5`; all half-levels from zero to five
  are retained, including a same-grid `L=3,4,5` comparison.
- `N=4`, all 64 momentum nodes. The previous `N=4 -> 5` change was below
  0.083%; this toy spends the available budget on the block cutoff instead.
- Same quadrature nodes, three-point coefficients, plumbing, vertex phases,
  explicit sewing signs, and reference free factors as the audited local run.
- Equal-sign components: branching recursion and two Virasoro c-recursions.
  Mixed-sign components: explicit PBW diagnostic completion, also through
  level five. No missing component is silently discarded.
- Every new node must reproduce the saved `L=3` blocks across all surfaces
  and lifts. The independent reducer reconstructs the contraction without
  calling the trial contraction routine.
- NSNSNS numerator remains the saved `R=16,N=5` value. Both free factors are
  checked in their respective plumbing frames. No fitted normalization.

The physical Ramond projection/lift dictionary remains unresolved. Output
`physical_Z` and `physical_Q` fields remain null.

## Runtime and resource boundary

One Slurm allocation: **1 node, 8 CPUs, 16 GiB, 03:00:00** on `yin`.
This is a wall-time budget **after allocation starts**, not a promise about
queue time. It is not an array with separately queued three-hour elements.

The local unequal-momentum L5 benchmark took 157.62 seconds and peaked at
191 MB resident memory. Nearly all its time was the four explicit PBW
completions. A conservative planning estimate is eight waves of nodes,
multiplied by a factor of three for slower cluster execution, plus 15 minutes
of overhead: approximately 80–100 minutes, comfortably below the requested
three-hour wall limit. This is an estimate; no compute-node timing on Cannon
has yet been measured.

The driver has a 15-minute per-node timeout and a 2h45m total compute deadline.
It reserves the remaining wall time for reduction, plotting, and cleanup.
Each node runs in a fresh process with BLAS/OpenMP limited to one thread.
Completed shards are validated and reused on restart. Missing/failed nodes
produce an explicit incomplete status, never a partial-grid partition value.
A process-held file lock prevents concurrent drivers writing the same output.

The job writes `summary.json`, `comparison.csv`, `comparison.svg`, per-node
shards/logs, and a completion status. `comparison.csv` includes the `L3 -> L4`
and `L4 -> L5` changes at identical `N=4`.

## Frozen local bundle

`Data Set/nsrr_trial_L5_N4_cluster_bundle_20260830/`

The bundle contains code, the protected-kernel manifest, a self-contained
configuration with cached reference data, the required StringMC plumbing
modules, and vendored SymPy. It contains no runtime references to local
`/Users/...` paths. All 1,866 files are SHA-256 checked. A relocated preflight
and ten regression tests pass from inside the bundle. The existing Cannon
runtime was checked read-only: Python 3.12.11, NumPy 2.0.2, SciPy 1.13.1,
mpmath 1.3.0; SymPy is supplied by the bundle.

Only a read-only queue/runtime check has been made on Cannon. No remote
directory has been staged and no Slurm job has been submitted for this trial.

## Staging and optional submission

Use a new dedicated remote run root. For example, with the existing runtime:

```sh
bash Code/cluster/stage_nsrr_trial_L5_3h.sh \
  cannon \
  /n/holylabs/yin_lab/Everyone/yutaizhang/StringMC/nsrr_trial_L5_N4_20260830_v1 \
  /n/holylabs/yin_lab/Everyone/yutaizhang/StringMC/genus2_period_table_20260718_176ab14d/.venv/bin/python
```

The default is **stage and preflight only**. Appending `--submit` stages,
preflights, and submits in one invocation. Existing remote roots are rejected
to prevent overwriting a snapshot. If staging was already performed, inspect
that root and submit its staged Slurm script explicitly; do not blindly rerun
the new-root staging command. Submission creates a remote guard and records
the job ID before returning; inspect it and the queue before retrying an
uncertain submission. No `scancel` or changes to unrelated jobs are performed.

## Local fallback requested by the user

The same portable driver supports `N=3` locally, with all 27 nodes and two
workers. The focus is `t=0.60`, but evaluating the other four surfaces after
the momentum-dependent blocks are computed costs little extra, so they are
retained. This compares `L=3,4,5` on the **same N=3 grid**; it must not be
confused with a pure cutoff comparison against the previous `N=5` table.

```sh
env PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  PYTHONPATH=Code:Code/genus_2:Code/c_Recursion:Code/full_ramond_block_runtime:Code/double_virasoro/nsrr \
  python3 Code/genus_2/nsrr_trial_cluster.py run \
  --config Code/config/nsrr_trial_L5_N3_local_20260830.json \
  --output-dir 'Data Set/nsrr_trial_L5_N3_local_20260830' --workers 2
```

This local run was launched separately from the unsubmitted cluster package.
Its final numerical status is recorded in its output directory and machine
note, not inferred from the cluster preparation status.
