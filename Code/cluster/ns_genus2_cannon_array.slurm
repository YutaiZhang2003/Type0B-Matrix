#!/bin/bash
#SBATCH --job-name=ns-g2-r8
#SBATCH --partition=yin
#SBATCH --cpus-per-task=1
#SBATCH --mem=3G
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --output=logs/ns-g2-r8-%A_%a.out
#SBATCH --error=logs/ns-g2-r8-%A_%a.err

set -euo pipefail

: "${NS_G2_ROOT:?set NS_G2_ROOT}"
: "${NS_G2_PYTHON:?set NS_G2_PYTHON}"
: "${NS_G2_CONFIG:?set NS_G2_CONFIG}"
: "${NS_G2_SHARDS:?set NS_G2_SHARDS}"

cd "${NS_G2_ROOT}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="${NS_G2_ROOT}/Code:${NS_G2_ROOT}/Code/python${PYTHONPATH:+:${PYTHONPATH}}"

"${NS_G2_PYTHON}" Code/ns_genus2_cannon.py \
  --config "${NS_G2_CONFIG}" \
  worker \
  --output-dir "${NS_G2_SHARDS}" \
  --task-index "${SLURM_ARRAY_TASK_ID}"
