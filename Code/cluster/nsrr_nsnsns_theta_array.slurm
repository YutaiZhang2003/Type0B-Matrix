#!/bin/bash
#SBATCH --job-name=nsrr-nnn-r8
#SBATCH --partition=yin
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=05:00:00
#SBATCH --requeue
#SBATCH --output=logs/nsrr-nnn-r8-%A_%a.out
#SBATCH --error=logs/nsrr-nnn-r8-%A_%a.err

set -euo pipefail

: "${NSRR_G2_ROOT:?set NSRR_G2_ROOT}"
: "${NSRR_G2_PYTHON:?set NSRR_G2_PYTHON}"
: "${NSRR_G2_CONFIG:?set NSRR_G2_CONFIG}"
: "${NSRR_G2_SHARDS:?set NSRR_G2_SHARDS}"
: "${NSRR_G2_STRINGMC_ROOT:?set NSRR_G2_STRINGMC_ROOT}"
: "${NSRR_G2_VENDOR:?set NSRR_G2_VENDOR}"
: "${NSRR_G2_CHANNEL:?set NSRR_G2_CHANNEL}"
: "${NSRR_G2_TASKS_PER_ARRAY:?set NSRR_G2_TASKS_PER_ARRAY}"

cd "${NSRR_G2_ROOT}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export TYPE0B_STRINGMC_ROOT="${NSRR_G2_STRINGMC_ROOT}"
export PYTHONPATH="${NSRR_G2_VENDOR}:${NSRR_G2_ROOT}/Code:${NSRR_G2_ROOT}/Code/genus_2:${NSRR_G2_ROOT}/Code/c_Recursion:${NSRR_G2_ROOT}/Code/full_ramond_block_runtime:${NSRR_G2_ROOT}/Code/genus_2_cross_channel:${NSRR_G2_ROOT}/Code/ramond_branching_recursion:${NSRR_G2_ROOT}/Code/double_virasoro/nsrr:${NSRR_G2_STRINGMC_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

"${NSRR_G2_PYTHON}" "${NSRR_G2_DRIVER:-Code/genus_2/nsrr_nsnsns_theta_cannon.py}" \
  --config "${NSRR_G2_CONFIG}" \
  channel-worker \
  --output-dir "${NSRR_G2_SHARDS}" \
  --channel "${NSRR_G2_CHANNEL}" \
  --chunk-index "${SLURM_ARRAY_TASK_ID}" \
  --tasks-per-chunk "${NSRR_G2_TASKS_PER_ARRAY}"
