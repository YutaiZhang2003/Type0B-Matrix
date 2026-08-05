#!/bin/bash
#SBATCH --job-name=ns-g2-chunk
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
: "${NS_G2_TASK_START:?set NS_G2_TASK_START}"
: "${NS_G2_TASK_END:?set NS_G2_TASK_END}"
: "${NS_G2_TASKS_PER_CHUNK:?set NS_G2_TASKS_PER_CHUNK}"

cd "${NS_G2_ROOT}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="${NS_G2_ROOT}/Code:${NS_G2_ROOT}/Code/python${PYTHONPATH:+:${PYTHONPATH}}"

first=$((NS_G2_TASK_START + SLURM_ARRAY_TASK_ID * NS_G2_TASKS_PER_CHUNK))
last=$((first + NS_G2_TASKS_PER_CHUNK - 1))
if ((last > NS_G2_TASK_END)); then
  last=${NS_G2_TASK_END}
fi

for ((task_index = first; task_index <= last; task_index++)); do
  "${NS_G2_PYTHON}" Code/ns_genus2_cannon.py \
    --config "${NS_G2_CONFIG}" \
    worker \
    --output-dir "${NS_G2_SHARDS}" \
    --task-index "${task_index}"
done
