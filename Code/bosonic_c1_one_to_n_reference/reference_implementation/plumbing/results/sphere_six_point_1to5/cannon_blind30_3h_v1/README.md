# Blind 30-point Cannon design for sphere 1->5

> **Run status:** array `41514317` failed because floating-point channel
> reconstruction produced an exactly zero plumbing coordinate.  It and its
> dependent assembly/freeze jobs were cancelled on 2026-08-24.  No shard or
> amplitude result from this run is valid.  The source-chart recovery and
> guarded replacement submission are documented in `../cannon_blind30_3h_v2/`.

This directory is the locally prepared design for
`sphere_six_point_1to5_cannon_blind30_3h_v1`. No matrix-model formula is
present in the worker configuration or driver.

## Numerical design

- 30 shifted uniform points from `t=0.1805` through `t=0.3255`, all strictly
  below the first residue wall at `t=1/3`.
- The half-step shift avoids the removable implementation singularity at
  exactly `t=0.2` without using the matrix-model prediction.
- 14 independently scrambled production replicates with `2^15` Sobol points
  each: 458,752 production samples per value of `t`.
- Six paired-systematics replicates with `2^11` common Sobol points each:
  12,288 common samples per value of `t`.
- Production block order 6 and momentum edge orders `(10,11,12)`.
- Paired block-order, momentum-order, and momentum-cutoff diagnostics.
- Every point must pass an absolute `Q5` stability-envelope gate of `5e-4`.
  A failed point blocks the freeze and therefore blocks all comparison work.

The manifest contains 450 tasks: 420 production shards and 30 systematics
shards. The task cap exposes all 450 simultaneously.

## Runtime policy

The initial three-hour preflight was attempted on Cannon.  The production
benchmark projected 1.72 h before its safety factor, while the high-order
systematics benchmark exhausted its 20-minute benchmark cap during kernel
construction.  No production task was submitted by that failed preflight.
The campaign was therefore relaxed, without changing any numerical setting,
to a 10 h worker-array limit, 30 min assembly, and 20 min freeze limit.

Production workers may requeue after preemption under the relaxed policy.

Slurm queue latency and final elapsed time are recorded but are no longer a
freeze condition.  The numerical `5e-4` gate remains mandatory.

## Commands

The design has already been generated locally. To stage, benchmark, and—only
if the benchmark passes—submit it:

```bash
plumbing/cluster/stage_submit_sphere_six_point_1to5_blind30_3h.sh
```

After the blind freeze exists, pull and perform the separate comparison with:

```bash
plumbing/cluster/pull_compare_sphere_six_point_1to5_blind30_3h.sh
```
