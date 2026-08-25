#!/bin/bash
#SBATCH --job-name=g1-3pt-hassemble
#SBATCH --partition=serial_requeue
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=logs/g1-3pt-hassemble-%j.out
#SBATCH --error=logs/g1-3pt-hassemble-%j.err

set -euo pipefail

: "${STRINGMC_ROOT:?set STRINGMC_ROOT}"
: "${STRINGMC_PYTHON:?set STRINGMC_PYTHON}"
: "${G1_THREE_TASKS:?set G1_THREE_TASKS}"
: "${G1_THREE_SHARDS:?set G1_THREE_SHARDS}"
: "${G1_THREE_REUSED_T075:?set G1_THREE_REUSED_T075}"
: "${G1_THREE_LEGACY_SCAN:?set G1_THREE_LEGACY_SCAN}"
: "${G1_THREE_ASSEMBLED:?set G1_THREE_ASSEMBLED}"
: "${G1_THREE_RUN_ROOT:?set G1_THREE_RUN_ROOT}"

cd "${STRINGMC_ROOT}"
export PYTHONUNBUFFERED=1
export PYTHONPYCACHEPREFIX="${G1_THREE_RUN_ROOT}/pycache"
export PYTHONPATH="${STRINGMC_ROOT}/plumbing:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

"${STRINGMC_PYTHON}" plumbing/genus1_three_point_hdominant_cannon.py assemble \
  --task-manifest "${G1_THREE_TASKS}" \
  --shards-dir "${G1_THREE_SHARDS}" \
  --reused-t075 "${G1_THREE_REUSED_T075}" \
  --legacy-scan "${G1_THREE_LEGACY_SCAN}" \
  --output-dir "${G1_THREE_ASSEMBLED}"
