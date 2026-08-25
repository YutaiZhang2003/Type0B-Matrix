#!/bin/bash
#SBATCH --job-name=g2-bos-stable
#SBATCH --partition=yin
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --output=logs/g2-bos-stable-%A_%a.out
#SBATCH --error=logs/g2-bos-stable-%A_%a.err

set -euo pipefail

: "${BOSONIC_STABLE_ROOT:?set BOSONIC_STABLE_ROOT}"
: "${BOSONIC_STABLE_PYTHON:?set BOSONIC_STABLE_PYTHON}"

POINTS=(o0043 o0243 o0127 o0019 o0015)
RECURSION=(8 10 12 12)
MOMENTUM=(10 10 10 12)
point_index=$((SLURM_ARRAY_TASK_ID / 4))
order_index=$((SLURM_ARRAY_TASK_ID % 4))
point_id=${POINTS[$point_index]}
recursion=${RECURSION[$order_index]}
momentum=${MOMENTUM[$order_index]}
shard="${BOSONIC_STABLE_ROOT}/shards/${point_id}_R${recursion}_N${momentum}"

mkdir -p "${shard}"
cd "${BOSONIC_STABLE_ROOT}/code"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/type0b-matplotlib"

"${BOSONIC_STABLE_PYTHON}" run_bosonic_genus2_stable_q_check.py \
  --design "${BOSONIC_STABLE_ROOT}/design/genus2_bosonic_stable_q_fivepoint.json" \
  --point-id "${point_id}" \
  --orders "${recursion}:${momentum}" \
  --dps 40 \
  --vacuum-word-length 8 \
  --vacuum-oscillator-level 32 \
  --scalar-word-length 12 \
  --scalar-max-mode 120 \
  --scalar-tolerance 1e-15 \
  --skip-plot \
  --out-dir "${shard}"
