#!/bin/bash
# Stage and submit the order-eight NSRR/NSNSNS matched-theta Cannon run.

set -euo pipefail

if [[ $# -lt 3 || $# -gt 5 ]]; then
  echo "usage: $0 SSH_HOST REMOTE_RUN_ROOT REMOTE_PYTHON [LOCAL_SUBMISSION_JSON] [LOCAL_CONFIG]" >&2
  exit 2
fi

SSH_HOST=$1
REMOTE_RUN_ROOT=$2
REMOTE_PYTHON=$3
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
LOCAL_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
LOCAL_SUBMISSION_JSON=${4:-"${LOCAL_ROOT}/Data Set/nsrr_nsnsns_theta_order8_cannon_submission.json"}
LOCAL_CONFIG=${5:-"${LOCAL_ROOT}/Code/config/nsrr_nsnsns_theta_order8_cannon_20260829.json"}
if [[ ${LOCAL_CONFIG} != /* ]]; then
  LOCAL_CONFIG="${LOCAL_ROOT}/${LOCAL_CONFIG}"
fi
if [[ ! -f ${LOCAL_CONFIG} ]]; then
  echo "missing local config: ${LOCAL_CONFIG}" >&2
  exit 2
fi

PARTITION=${NSRR_G2_PARTITION:-yin}
SOURCE_CAP=${NSRR_G2_SOURCE_ARRAY_CAP:-5}
TARGET_CAP=${NSRR_G2_TARGET_ARRAY_CAP:-4}
SOURCE_TASKS_PER_ARRAY=${NSRR_G2_SOURCE_TASKS_PER_ARRAY:-768}
TARGET_TASKS_PER_ARRAY=${NSRR_G2_TARGET_TASKS_PER_ARRAY:-1024}
SOURCE_MEMORY=${NSRR_G2_SOURCE_MEMORY:-8G}
TARGET_MEMORY=${NSRR_G2_TARGET_MEMORY:-4G}
SOURCE_TIME=${NSRR_G2_SOURCE_TIME:-05:00:00}
TARGET_TIME=${NSRR_G2_TARGET_TIME:-05:00:00}
JOB_TAG=${NSRR_G2_JOB_TAG:-nsrr-nnn-r8}
DRIVER=${NSRR_G2_DRIVER:-Code/genus_2/nsrr_nsnsns_theta_cannon.py}
DRIVER_TEST=${NSRR_G2_DRIVER_TEST:-Code/genus_2/test_nsrr_nsnsns_theta_cannon.py}
LOCAL_STRINGMC_ROOT=${NSRR_G2_LOCAL_STRINGMC_ROOT:-"${LOCAL_ROOT}/../Project/StringMC"}
LOCAL_SYMPY_ROOT=${NSRR_G2_LOCAL_SYMPY_ROOT:-$(python3 -c 'from pathlib import Path; import sympy; print(Path(sympy.__file__).resolve().parent)')}
if [[ ! -f ${LOCAL_SYMPY_ROOT}/__init__.py ]]; then
  echo "missing local pure-Python SymPy package: ${LOCAL_SYMPY_ROOT}" >&2
  exit 2
fi
for dependency in \
  ccy_genus2_block.py \
  genus2_vacuum_blocks.py \
  virasoro_plumbing_graph.py \
  plumbing_algorithms.py \
  virasoro_blocks.py \
  virasoro_descendant_algebra.py; do
  if [[ ! -f ${LOCAL_STRINGMC_ROOT}/plumbing/${dependency} ]]; then
    echo "missing local StringMC dependency: ${LOCAL_STRINGMC_ROOT}/plumbing/${dependency}" >&2
    exit 2
  fi
done

REMOTE_ROOT="${REMOTE_RUN_ROOT}/code"
REMOTE_STRINGMC_ROOT="${REMOTE_ROOT}/StringMC"
REMOTE_VENDOR="${REMOTE_ROOT}/vendor"
REMOTE_CONFIG="${REMOTE_ROOT}/Code/config/$(basename "${LOCAL_CONFIG}")"
REMOTE_SHARDS="${REMOTE_RUN_ROOT}/shards"
REMOTE_SUMMARY="${REMOTE_RUN_ROOT}/summary.json"

ssh "${SSH_HOST}" "mkdir -p '${REMOTE_ROOT}/Code/config' '${REMOTE_ROOT}/Code/cluster' '${REMOTE_STRINGMC_ROOT}/plumbing' '${REMOTE_VENDOR}/sympy' '${REMOTE_SHARDS}' '${REMOTE_RUN_ROOT}/logs'"

(
  cd "${LOCAL_ROOT}"
  rsync -azR \
    --exclude='__pycache__' \
    --include='*/' \
    --include='*.py' \
    --exclude='*' \
    ./Code/c_Recursion/ \
    ./Code/genus_2/ \
    ./Code/full_ramond_block_runtime/ \
    ./Code/ramond_branching_recursion/ \
    ./Code/double_virasoro/nsrr/ \
    ./Code/genus_2_cross_channel/ \
    "${SSH_HOST}:${REMOTE_ROOT}/"
  rsync -azR \
    ./Code/cluster/nsrr_nsnsns_theta_array.slurm \
    ./Code/cluster/nsrr_nsnsns_theta_reduce.slurm \
    "${SSH_HOST}:${REMOTE_ROOT}/"
)
rsync -az \
  "${LOCAL_STRINGMC_ROOT}/plumbing/ccy_genus2_block.py" \
  "${LOCAL_STRINGMC_ROOT}/plumbing/genus2_vacuum_blocks.py" \
  "${LOCAL_STRINGMC_ROOT}/plumbing/virasoro_plumbing_graph.py" \
  "${LOCAL_STRINGMC_ROOT}/plumbing/plumbing_algorithms.py" \
  "${LOCAL_STRINGMC_ROOT}/plumbing/virasoro_blocks.py" \
  "${LOCAL_STRINGMC_ROOT}/plumbing/virasoro_descendant_algebra.py" \
  "${SSH_HOST}:${REMOTE_STRINGMC_ROOT}/plumbing/"
rsync -az --exclude='__pycache__' \
  "${LOCAL_SYMPY_ROOT}/" \
  "${SSH_HOST}:${REMOTE_VENDOR}/sympy/"
rsync -az "${LOCAL_ROOT}/Code/config/nsrr_nsnsns_theta_order8_cannon_20260829.json" "${SSH_HOST}:${REMOTE_ROOT}/Code/config/"
rsync -az "${LOCAL_CONFIG}" "${SSH_HOST}:${REMOTE_CONFIG}"

REMOTE_ENV="OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 TYPE0B_STRINGMC_ROOT='${REMOTE_STRINGMC_ROOT}' PYTHONPATH='${REMOTE_VENDOR}:${REMOTE_ROOT}/Code:${REMOTE_ROOT}/Code/genus_2:${REMOTE_ROOT}/Code/c_Recursion:${REMOTE_ROOT}/Code/full_ramond_block_runtime:${REMOTE_ROOT}/Code/genus_2_cross_channel:${REMOTE_ROOT}/Code/ramond_branching_recursion:${REMOTE_ROOT}/Code/double_virasoro/nsrr:${REMOTE_STRINGMC_ROOT}'"
if [[ ${NSRR_G2_SKIP_PREFLIGHT:-0} != 1 ]]; then
  ssh "${SSH_HOST}" "set -e; test -f '${REMOTE_STRINGMC_ROOT}/plumbing/ccy_genus2_block.py'; ${REMOTE_ENV} '${REMOTE_PYTHON}' -c 'import numpy, scipy, mpmath, sympy; print(numpy.__version__, scipy.__version__, mpmath.__version__, sympy.__version__)'; cd '${REMOTE_ROOT}'; ${REMOTE_ENV} '${REMOTE_PYTHON}' -m unittest Code/c_Recursion/test_generic_super_liouville_structure_constants.py Code/full_ramond_block_runtime/test_nsrr_double_virasoro_block.py '${DRIVER_TEST}'; ${REMOTE_ENV} '${REMOTE_PYTHON}' '${DRIVER}' --config '${REMOTE_CONFIG}' plan"
else
  echo "remote preflight skipped only because this identical staged snapshot was already verified"
fi

TASK_COUNT=$(ssh "${SSH_HOST}" "cd '${REMOTE_ROOT}'; ${REMOTE_ENV} '${REMOTE_PYTHON}' '${DRIVER}' --config '${REMOTE_CONFIG}' plan --task-count-only")
SOURCE_ARRAY_COUNT=$(ssh "${SSH_HOST}" "cd '${REMOTE_ROOT}'; ${REMOTE_ENV} '${REMOTE_PYTHON}' '${DRIVER}' --config '${REMOTE_CONFIG}' channel-chunk-count --channel source_nsrr --tasks-per-chunk '${SOURCE_TASKS_PER_ARRAY}'")
TARGET_ARRAY_COUNT=$(ssh "${SSH_HOST}" "cd '${REMOTE_ROOT}'; ${REMOTE_ENV} '${REMOTE_PYTHON}' '${DRIVER}' --config '${REMOTE_CONFIG}' channel-chunk-count --channel target_nsnsns --tasks-per-chunk '${TARGET_TASKS_PER_ARRAY}'")
if [[ ${TASK_COUNT} -le 0 || ${SOURCE_CAP} -le 0 || ${TARGET_CAP} -le 0 || ${SOURCE_ARRAY_COUNT} -le 0 || ${TARGET_ARRAY_COUNT} -le 0 ]]; then
  echo "invalid task count, array count, or array cap" >&2
  exit 2
fi

COMMON_EXPORT="ALL,NSRR_G2_ROOT=${REMOTE_ROOT},NSRR_G2_PYTHON=${REMOTE_PYTHON},NSRR_G2_CONFIG=${REMOTE_CONFIG},NSRR_G2_SHARDS=${REMOTE_SHARDS},NSRR_G2_STRINGMC_ROOT=${REMOTE_STRINGMC_ROOT},NSRR_G2_VENDOR=${REMOTE_VENDOR},NSRR_G2_DRIVER=${DRIVER}"
SOURCE_JOB=$(ssh "${SSH_HOST}" "cd '${REMOTE_RUN_ROOT}'; sbatch --parsable --partition='${PARTITION}' --job-name='${JOB_TAG}-src' --mem='${SOURCE_MEMORY}' --time='${SOURCE_TIME}' --output='logs/${JOB_TAG}-src-%A_%a.out' --error='logs/${JOB_TAG}-src-%A_%a.err' --array='0-$((SOURCE_ARRAY_COUNT-1))%${SOURCE_CAP}' --export='${COMMON_EXPORT},NSRR_G2_CHANNEL=source_nsrr,NSRR_G2_TASKS_PER_ARRAY=${SOURCE_TASKS_PER_ARRAY}' '${REMOTE_ROOT}/Code/cluster/nsrr_nsnsns_theta_array.slurm'")
TARGET_JOB=$(ssh "${SSH_HOST}" "cd '${REMOTE_RUN_ROOT}'; sbatch --parsable --partition='${PARTITION}' --job-name='${JOB_TAG}-tgt' --mem='${TARGET_MEMORY}' --time='${TARGET_TIME}' --output='logs/${JOB_TAG}-tgt-%A_%a.out' --error='logs/${JOB_TAG}-tgt-%A_%a.err' --array='0-$((TARGET_ARRAY_COUNT-1))%${TARGET_CAP}' --export='${COMMON_EXPORT},NSRR_G2_CHANNEL=target_nsnsns,NSRR_G2_TASKS_PER_ARRAY=${TARGET_TASKS_PER_ARRAY}' '${REMOTE_ROOT}/Code/cluster/nsrr_nsnsns_theta_array.slurm'")
REDUCE_JOB=$(ssh "${SSH_HOST}" "cd '${REMOTE_RUN_ROOT}'; sbatch --parsable --partition='serial_requeue' --job-name='${JOB_TAG}-reduce' --output='logs/${JOB_TAG}-reduce-%j.out' --error='logs/${JOB_TAG}-reduce-%j.err' --dependency=afterok:${SOURCE_JOB}:${TARGET_JOB} --kill-on-invalid-dep=yes --export='${COMMON_EXPORT},NSRR_G2_SUMMARY=${REMOTE_SUMMARY}' '${REMOTE_ROOT}/Code/cluster/nsrr_nsnsns_theta_reduce.slurm'")

mkdir -p "$(dirname "${LOCAL_SUBMISSION_JSON}")"
printf '{\n  "status": "submitted",\n  "ssh_host": "%s",\n  "remote_run_root": "%s",\n  "remote_python": "%s",\n  "staged_stringmc_root": "%s",\n  "local_config": "%s",\n  "remote_config": "%s",\n  "partition": "%s",\n  "task_count": %d,\n  "source_array_count": %d,\n  "target_array_count": %d,\n  "source_tasks_per_array_element": %d,\n  "target_tasks_per_array_element": %d,\n  "source_array_cap": %d,\n  "target_array_cap": %d,\n  "source_job_id": "%s",\n  "target_job_id": "%s",\n  "reduce_job_id": "%s",\n  "remote_summary": "%s"\n}\n' \
  "${SSH_HOST}" "${REMOTE_RUN_ROOT}" "${REMOTE_PYTHON}" "${REMOTE_STRINGMC_ROOT}" "${LOCAL_CONFIG}" "${REMOTE_CONFIG}" "${PARTITION}" \
  "${TASK_COUNT}" "${SOURCE_ARRAY_COUNT}" "${TARGET_ARRAY_COUNT}" "${SOURCE_TASKS_PER_ARRAY}" "${TARGET_TASKS_PER_ARRAY}" "${SOURCE_CAP}" "${TARGET_CAP}" "${SOURCE_JOB}" "${TARGET_JOB}" "${REDUCE_JOB}" "${REMOTE_SUMMARY}" \
  > "${LOCAL_SUBMISSION_JSON}"

echo "source_job_id=${SOURCE_JOB}"
echo "target_job_id=${TARGET_JOB}"
echo "reduce_job_id=${REDUCE_JOB}"
echo "task_count=${TASK_COUNT}"
echo "submission_record=${LOCAL_SUBMISSION_JSON}"
echo "remote_summary=${REMOTE_SUMMARY}"
