#!/bin/bash
#SBATCH --job-name=ns-g2-channel
#SBATCH --partition=yin
#SBATCH --cpus-per-task=1
#SBATCH --mem=3G
#SBATCH --time=04:00:00
#SBATCH --requeue

set -euo pipefail

: "${NS_G2_ROOT:?set NS_G2_ROOT}"
: "${NS_G2_PYTHON:?set NS_G2_PYTHON}"
: "${NS_G2_CONFIG:?set NS_G2_CONFIG}"
: "${NS_G2_SHARDS:?set NS_G2_SHARDS}"
: "${NS_G2_CHUNK_MANIFEST:?set NS_G2_CHUNK_MANIFEST}"

cd "${NS_G2_ROOT}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="${NS_G2_ROOT}/Code:${NS_G2_ROOT}/Code/genus_2_cross_channel${PYTHONPATH:+:${PYTHONPATH}}"

line_number=$((SLURM_ARRAY_TASK_ID + 1))
line=$(sed -n "${line_number}p" "${NS_G2_CHUNK_MANIFEST}")
if [[ -z ${line} ]]; then
  echo "missing channel chunk ${SLURM_ARRAY_TASK_ID}" >&2
  exit 2
fi
read -r first last <<< "${line}"
if [[ ! ${first} =~ ^[0-9]+$ || ! ${last} =~ ^[0-9]+$ || ${first} -gt ${last} ]]; then
  echo "invalid channel chunk: ${line}" >&2
  exit 2
fi

for ((task_index = first; task_index <= last; task_index++)); do
  "${NS_G2_PYTHON}" Code/c_Recursion/ns_genus2_cannon.py \
    --config "${NS_G2_CONFIG}" \
    worker \
    --output-dir "${NS_G2_SHARDS}" \
    --task-index "${task_index}"
done
