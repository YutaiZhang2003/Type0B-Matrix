#!/bin/bash
#SBATCH --job-name=s6-1to5-blind
#SBATCH --partition=serial_requeue
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=10:00:00
#SBATCH --requeue
#SBATCH --output=logs/s6-1to5-blind-%A-%a.out
#SBATCH --error=logs/s6-1to5-blind-%A-%a.err

set -euo pipefail

: "${STRINGMC_ROOT:?set STRINGMC_ROOT}"
: "${STRINGMC_PYTHON:?set STRINGMC_PYTHON}"
: "${SPHERE6_BLIND_CONFIG:?set SPHERE6_BLIND_CONFIG}"
: "${SPHERE6_BLIND_MANIFEST:?set SPHERE6_BLIND_MANIFEST}"
: "${SPHERE6_BLIND_SHARDS:?set SPHERE6_BLIND_SHARDS}"

cd "${STRINGMC_ROOT}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONPATH="${STRINGMC_ROOT}/plumbing:${PYTHONPATH:-}"

"${STRINGMC_PYTHON}" plumbing/sphere_six_point_cannon_blind.py worker \
  --config "${SPHERE6_BLIND_CONFIG}" \
  --manifest "${SPHERE6_BLIND_MANIFEST}" \
  --task-id "${SLURM_ARRAY_TASK_ID}" \
  --shards-dir "${SPHERE6_BLIND_SHARDS}"
