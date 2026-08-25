#!/bin/bash
#SBATCH --job-name=g1-3pt-hscan
#SBATCH --partition=yin
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --output=logs/g1-3pt-hscan-%A_%a.out
#SBATCH --error=logs/g1-3pt-hscan-%A_%a.err

set -euo pipefail

: "${STRINGMC_ROOT:?set STRINGMC_ROOT}"
: "${STRINGMC_PYTHON:?set STRINGMC_PYTHON}"
: "${G1_THREE_TASKS:?set G1_THREE_TASKS}"
: "${G1_THREE_SHARDS:?set G1_THREE_SHARDS}"
: "${G1_THREE_CACHE:?set G1_THREE_CACHE}"
: "${G1_THREE_RUN_ROOT:?set G1_THREE_RUN_ROOT}"

cd "${STRINGMC_ROOT}"
export PYTHONUNBUFFERED=1
export PYTHONPYCACHEPREFIX="${G1_THREE_RUN_ROOT}/pycache"
export PYTHONPATH="${STRINGMC_ROOT}/plumbing:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

IFS=',' read -r TASK_ID T_VALUE TAG < <(
  "${STRINGMC_PYTHON}" -c \
  'import csv,os,sys; rows=list(csv.DictReader(open(sys.argv[1]))); row=rows[int(os.environ["SLURM_ARRAY_TASK_ID"])]; print(",".join((row["task_id"],row["t"],row["tag"])))' \
  "${G1_THREE_TASKS}"
)
if [[ "${TASK_ID}" != "${SLURM_ARRAY_TASK_ID}" ]]; then
  echo "task manifest mismatch: ${TASK_ID} != ${SLURM_ARRAY_TASK_ID}" >&2
  exit 2
fi

"${STRINGMC_PYTHON}" plumbing/smoke_genus1_three_point_channel_atlas.py \
  --t "${T_VALUE}" \
  --momentum-order 8 \
  --necklace-order 8 \
  --necklace-low-order 3 \
  --necklace-backend regulated-h-recursion \
  --c-regulator 0.025 \
  --coefficient-workers "${SLURM_CPUS_PER_TASK}" \
  --necklace-qhat-threshold 0.30 \
  --necklace-second-qhat-threshold 0.07 \
  --ope-order 6 \
  --ope-loop-order 4 \
  --ope-low-loop-order 2 \
  --sobol-power 8 \
  --replicates 4 \
  --seed 17051301 \
  --alpha 0.3 \
  --patch-epsilon 0.15 \
  --triple-patch-epsilon 0.10 \
  --tail-proposal-exponent 1.5 \
  --dps 24 \
  --bank-cache "${G1_THREE_CACHE}/${TAG}_ope_banks.npz" \
  --necklace-bank-cache "${G1_THREE_CACHE}/${TAG}_necklace_banks.npz" \
  --prepare-necklace-first \
  --output "${G1_THREE_SHARDS}/${TAG}.json"
