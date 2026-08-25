#!/bin/bash
#SBATCH --job-name=type0b-torus-reduce
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:20:00
#SBATCH --output=logs/type0b-reduce-%j.out
#SBATCH --error=logs/type0b-reduce-%j.err

set -euo pipefail

: "${TYPE0B_ROOT:?set TYPE0B_ROOT to the repository checkout}"
: "${TYPE0B_PYTHON:?set TYPE0B_PYTHON to the cluster Python}"
: "${TYPE0B_MODULAR_CONFIG:?set TYPE0B_MODULAR_CONFIG to the immutable config snapshot}"
: "${TYPE0B_MODULAR_SHARDS:?set TYPE0B_MODULAR_SHARDS to the shard directory}"
: "${TYPE0B_MODULAR_SUMMARY:?set TYPE0B_MODULAR_SUMMARY to the reduction output}"
: "${TYPE0B_MODULAR_SHARD_COUNT:?set TYPE0B_MODULAR_SHARD_COUNT}"

cd "${TYPE0B_ROOT}"
export PYTHONPATH="${TYPE0B_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

"${TYPE0B_PYTHON}" h_recursion/super_liouville_torus_modular_cluster.py \
  --config "${TYPE0B_MODULAR_CONFIG}" \
  reduce \
  --input-dir "${TYPE0B_MODULAR_SHARDS}" \
  --output "${TYPE0B_MODULAR_SUMMARY}" \
  --shard-count "${TYPE0B_MODULAR_SHARD_COUNT}"
