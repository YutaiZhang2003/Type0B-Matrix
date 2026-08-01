#!/bin/bash
#SBATCH --job-name=type0b-torus-modular
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=02:00:00
#SBATCH --requeue
#SBATCH --output=logs/type0b-torus-%A_%a.out
#SBATCH --error=logs/type0b-torus-%A_%a.err

set -euo pipefail

: "${TYPE0B_ROOT:?set TYPE0B_ROOT to the repository checkout}"
: "${TYPE0B_PYTHON:?set TYPE0B_PYTHON to the cluster Python}"
: "${TYPE0B_MODULAR_CONFIG:?set TYPE0B_MODULAR_CONFIG to the immutable config snapshot}"
: "${TYPE0B_MODULAR_SHARDS:?set TYPE0B_MODULAR_SHARDS to the shard directory}"
: "${TYPE0B_MODULAR_SHARD_COUNT:?set TYPE0B_MODULAR_SHARD_COUNT}"

cd "${TYPE0B_ROOT}"

# Each array element is a single-process task.  Prevent BLAS from creating
# hidden worker threads and competing with neighboring array elements.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

"${TYPE0B_PYTHON}" super_liouville_torus_modular_cluster.py \
  --config "${TYPE0B_MODULAR_CONFIG}" \
  worker \
  --output-dir "${TYPE0B_MODULAR_SHARDS}" \
  --shard-id "${SLURM_ARRAY_TASK_ID}" \
  --shard-count "${TYPE0B_MODULAR_SHARD_COUNT}" \
  --execute
