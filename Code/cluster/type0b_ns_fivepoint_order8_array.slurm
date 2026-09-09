#!/bin/bash
#SBATCH --job-name=t0b-5pt-o8
#SBATCH --partition=yin
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=12:00:00
#SBATCH --no-requeue
#SBATCH --output=logs/t0b-5pt-o8-%A_%a.out
#SBATCH --error=logs/t0b-5pt-o8-%A_%a.err

set -euo pipefail

: "${TYPE0B_5PT_ROOT:?set TYPE0B_5PT_ROOT}"
: "${TYPE0B_5PT_PYTHON:?set TYPE0B_5PT_PYTHON}"
: "${TYPE0B_5PT_CONFIG:?set TYPE0B_5PT_CONFIG}"
: "${TYPE0B_5PT_SHARDS:?set TYPE0B_5PT_SHARDS}"
: "${TYPE0B_5PT_COEFFICIENT_CACHE:?set TYPE0B_5PT_COEFFICIENT_CACHE}"
: "${TYPE0B_5PT_TASK_COUNT:?set TYPE0B_5PT_TASK_COUNT}"

if (( SLURM_ARRAY_TASK_ID < 0 || SLURM_ARRAY_TASK_ID >= TYPE0B_5PT_TASK_COUNT )); then
  echo "array index outside planned task range" >&2
  exit 2
fi

cd "${TYPE0B_5PT_ROOT}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export TYPE0B_5PT_PROGRESS=1
export PYTHONPATH="${TYPE0B_5PT_ROOT}/Code/higher_point_amplitude_attempts/type0b_ns_five_tachyon:${TYPE0B_5PT_ROOT}/Code/c_Recursion:${TYPE0B_5PT_ROOT}/Code/bosonic_c1_one_to_n_reference/reference_implementation/plumbing${PYTHONPATH:+:${PYTHONPATH}}"

"${TYPE0B_5PT_PYTHON}" \
  Code/higher_point_amplitude_attempts/type0b_ns_five_tachyon/run_type0b_ns_five_tachyon_cluster.py \
  --config "${TYPE0B_5PT_CONFIG}" \
  worker \
  --output-dir "${TYPE0B_5PT_SHARDS}" \
  --task-index "${SLURM_ARRAY_TASK_ID}"
