#!/bin/bash
#SBATCH --job-name=g2-bos-reduce
#SBATCH --partition=yin
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --output=logs/g2-bos-reduce-%j.out
#SBATCH --error=logs/g2-bos-reduce-%j.err

set -euo pipefail

: "${BOSONIC_STABLE_ROOT:?set BOSONIC_STABLE_ROOT}"
: "${BOSONIC_STABLE_PYTHON:?set BOSONIC_STABLE_PYTHON}"

cd "${BOSONIC_STABLE_ROOT}/code"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/type0b-matplotlib"
"${BOSONIC_STABLE_PYTHON}" reduce_bosonic_genus2_stable_q_check.py \
  --design "${BOSONIC_STABLE_ROOT}/design/genus2_bosonic_stable_q_fivepoint.json" \
  --runner "${BOSONIC_STABLE_ROOT}/code/run_bosonic_genus2_stable_q_check.py" \
  --shards-dir "${BOSONIC_STABLE_ROOT}/shards" \
  --out-dir "${BOSONIC_STABLE_ROOT}/final"
