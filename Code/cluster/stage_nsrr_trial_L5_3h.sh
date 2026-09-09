#!/bin/bash
# Default is stage-only. Submission requires the explicit fourth argument --submit.
set -euo pipefail
if [[ $# -lt 3 || $# -gt 4 || ( $# == 4 && $4 != --submit ) ]]; then
  echo "usage: $0 SSH_HOST NEW_REMOTE_RUN_ROOT REMOTE_PYTHON [--submit]" >&2
  exit 2
fi
NSRR_STAGE_HOST=$1
NSRR_STAGE_REMOTE=$2
NSRR_STAGE_PYTHON=$3
NSRR_STAGE_SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
NSRR_STAGE_LOCAL_ROOT=$(cd "${NSRR_STAGE_SCRIPT_DIR}/../.." && pwd)
NSRR_STAGE_BUNDLE="${NSRR_STAGE_LOCAL_ROOT}/Data Set/nsrr_trial_L5_N4_cluster_bundle_20260830"
# Constrain all remotely interpolated strings; never stage to a broad root.
if [[ ! ${NSRR_STAGE_HOST} =~ ^[A-Za-z0-9_.@-]+$ ||
      ! ${NSRR_STAGE_REMOTE} =~ ^/[A-Za-z0-9_./-]+/nsrr_trial_L5_[A-Za-z0-9_-]+$ ||
      ! ${NSRR_STAGE_PYTHON} =~ ^/[A-Za-z0-9_./-]+$ ||
      ${NSRR_STAGE_REMOTE} == *..* || ${NSRR_STAGE_PYTHON} == *..* ]]; then
  echo "unsafe host/path, or remote run root does not have a dedicated nsrr_trial_L5_ name" >&2
  exit 2
fi
python3 "${NSRR_STAGE_SCRIPT_DIR}/prepare_nsrr_trial_L5_bundle.py" verify --bundle-root "${NSRR_STAGE_BUNDLE}"
ssh "${NSRR_STAGE_HOST}" "test ! -e '${NSRR_STAGE_REMOTE}' && mkdir -p '${NSRR_STAGE_REMOTE}/bundle' '${NSRR_STAGE_REMOTE}/logs'"
rsync -az "${NSRR_STAGE_BUNDLE}/" "${NSRR_STAGE_HOST}:${NSRR_STAGE_REMOTE}/bundle/"
NSRR_STAGE_ROOT="${NSRR_STAGE_REMOTE}/bundle"
NSRR_STAGE_ENV="PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 TYPE0B_STRINGMC_ROOT='${NSRR_STAGE_ROOT}/StringMC' PYTHONPATH='${NSRR_STAGE_ROOT}/vendor:${NSRR_STAGE_ROOT}/Code:${NSRR_STAGE_ROOT}/Code/genus_2:${NSRR_STAGE_ROOT}/Code/c_Recursion:${NSRR_STAGE_ROOT}/Code/full_ramond_block_runtime:${NSRR_STAGE_ROOT}/Code/genus_2_cross_channel:${NSRR_STAGE_ROOT}/Code/ramond_branching_recursion:${NSRR_STAGE_ROOT}/Code/double_virasoro/nsrr:${NSRR_STAGE_ROOT}/StringMC'"
ssh "${NSRR_STAGE_HOST}" "cd '${NSRR_STAGE_ROOT}' && ${NSRR_STAGE_ENV} '${NSRR_STAGE_PYTHON}' Code/cluster/prepare_nsrr_trial_L5_bundle.py verify --bundle-root '${NSRR_STAGE_ROOT}' && ${NSRR_STAGE_ENV} '${NSRR_STAGE_PYTHON}' Code/genus_2/nsrr_trial_cluster.py preflight --config Code/config/nsrr_trial_L5_N4_cluster_20260830.json"
if [[ $# == 3 ]]; then
  echo "Staged and verified; no job submitted."
  exit 0
fi
# Atomic remote guard prevents a blind retry after an uncertain sbatch response.
ssh "${NSRR_STAGE_HOST}" "set -e; cd '${NSRR_STAGE_REMOTE}'; mkdir submission_guard; sbatch --parsable --chdir='${NSRR_STAGE_REMOTE}' --export='ALL,NSRR_TRIAL_ROOT=${NSRR_STAGE_ROOT},NSRR_TRIAL_PYTHON=${NSRR_STAGE_PYTHON},NSRR_TRIAL_OUTPUT=${NSRR_STAGE_REMOTE}/output' '${NSRR_STAGE_ROOT}/Code/cluster/nsrr_trial_L5_3h.slurm' > submission_guard/job_id.txt; test -s submission_guard/job_id.txt; cat submission_guard/job_id.txt"
