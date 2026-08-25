#!/bin/bash
#SBATCH --job-name=ns-g2-r8-theta-reduce
#SBATCH --partition=serial_requeue
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=02:00:00
#SBATCH --output=logs/ns-g2-r8-theta-reduce-%j.out
#SBATCH --error=logs/ns-g2-r8-theta-reduce-%j.err

set -euo pipefail

: "${NS_G2_ROOT:?set NS_G2_ROOT}"
: "${NS_G2_PYTHON:?set NS_G2_PYTHON}"
: "${NS_G2_CONFIG:?set NS_G2_CONFIG}"
: "${NS_G2_SHARDS:?set NS_G2_SHARDS}"
: "${NS_G2_SOURCE_SUMMARY:?set NS_G2_SOURCE_SUMMARY}"
: "${NS_G2_SUMMARY:?set NS_G2_SUMMARY}"

cd "${NS_G2_ROOT}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONPATH="${NS_G2_ROOT}/Code:${NS_G2_ROOT}/Code/genus_2_cross_channel${PYTHONPATH:+:${PYTHONPATH}}"

"${NS_G2_PYTHON}" Code/c_Recursion/ns_genus2_cannon.py \
  --config "${NS_G2_CONFIG}" \
  recombine-theta \
  --theta-shard-dir "${NS_G2_SHARDS}" \
  --source-summary "${NS_G2_SOURCE_SUMMARY}" \
  --output "${NS_G2_SUMMARY}"
