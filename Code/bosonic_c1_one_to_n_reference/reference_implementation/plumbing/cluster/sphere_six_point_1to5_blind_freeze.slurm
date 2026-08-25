#!/bin/bash
#SBATCH --job-name=s6-1to5-freeze
#SBATCH --partition=serial_requeue
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:20:00
#SBATCH --output=logs/s6-1to5-freeze-%j.out
#SBATCH --error=logs/s6-1to5-freeze-%j.err

set -euo pipefail

: "${STRINGMC_ROOT:?set STRINGMC_ROOT}"
: "${STRINGMC_PYTHON:?set STRINGMC_PYTHON}"
: "${SPHERE6_BLIND_CONFIG:?set SPHERE6_BLIND_CONFIG}"
: "${SPHERE6_BLIND_MANIFEST:?set SPHERE6_BLIND_MANIFEST}"
: "${SPHERE6_BLIND_ASSEMBLED:?set SPHERE6_BLIND_ASSEMBLED}"

cd "${STRINGMC_ROOT}"
export PYTHONUNBUFFERED=1
export PYTHONPATH="${STRINGMC_ROOT}/plumbing:${PYTHONPATH:-}"

"${STRINGMC_PYTHON}" plumbing/sphere_six_point_cannon_blind.py freeze \
  --config "${SPHERE6_BLIND_CONFIG}" \
  --manifest "${SPHERE6_BLIND_MANIFEST}" \
  --output-dir "${SPHERE6_BLIND_ASSEMBLED}"
