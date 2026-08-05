#!/bin/bash
# Stage and submit a genus-two NS locality convergence run on Cannon.

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
LOCAL_SUBMISSION_JSON=${4:-"${LOCAL_ROOT}/Data Set/ns_genus2_cannon_fivepoint_order8_submission.json"}
LOCAL_CONFIG=${5:-"${LOCAL_ROOT}/Code/config/ns_genus2_cannon_fivepoint_order8.json"}
if [[ ${LOCAL_CONFIG} != /* ]]; then
  LOCAL_CONFIG="${LOCAL_ROOT}/${LOCAL_CONFIG}"
fi
if [[ ! -f ${LOCAL_CONFIG} ]]; then
  echo "missing local config: ${LOCAL_CONFIG}" >&2
  exit 2
fi
PARTITION=${NS_G2_PARTITION:-yin}
ARRAY_CAP=${NS_G2_ARRAY_CAP:-200}
TASKS_PER_ARRAY=${NS_G2_TASKS_PER_ARRAY:-1}
JOB_TAG=${NS_G2_JOB_TAG:-ns-g2}
REMOTE_ROOT="${REMOTE_RUN_ROOT}/code"
REMOTE_CONFIG="${REMOTE_ROOT}/Code/config/$(basename "${LOCAL_CONFIG}")"
REMOTE_SHARDS="${REMOTE_RUN_ROOT}/shards"
REMOTE_SUMMARY="${REMOTE_RUN_ROOT}/summary.json"

ssh "${SSH_HOST}" "mkdir -p '${REMOTE_ROOT}/Code/python' '${REMOTE_ROOT}/Code/config' '${REMOTE_ROOT}/Code/cluster' '${REMOTE_SHARDS}' '${REMOTE_RUN_ROOT}/logs'"

(
  cd "${LOCAL_ROOT}"
  rsync -azR \
    ./Code/ns_genus2_cannon.py \
    ./Code/ns_genus2_partition.py \
    ./Code/test_ns_genus2_partition.py \
    ./Code/compare_ns_torus_c_h_recursion.py \
    ./Code/ns_genus_c_recursion_checks.py \
    ./Code/ns_recursion_recipe.py \
    ./Code/ns_global_osp_block.py \
    ./Code/ns_regular_block.py \
    ./Code/ns_vacuum_schottky.py \
    ./Code/super_liouville_structure_constants.py \
    ./Code/superconformal_blocks.py \
    ./Code/python/ccy_genus2_block.py \
    ./Code/python/free_boson_plumbing.py \
    ./Code/python/genus2_vacuum_blocks.py \
    ./Code/python/plumbing_algorithms.py \
    ./Code/python/virasoro_blocks.py \
    ./Code/cluster/ns_genus2_cannon_array.slurm \
    ./Code/cluster/ns_genus2_cannon_reduce.slurm \
    "${SSH_HOST}:${REMOTE_ROOT}/"
)
rsync -az "${LOCAL_CONFIG}" "${SSH_HOST}:${REMOTE_CONFIG}"

ssh "${SSH_HOST}" "set -e; '${REMOTE_PYTHON}' -c 'import numpy, scipy, mpmath; print(numpy.__version__, scipy.__version__, mpmath.__version__)'; cd '${REMOTE_ROOT}'; PYTHONPATH='${REMOTE_ROOT}/Code:${REMOTE_ROOT}/Code/python' '${REMOTE_PYTHON}' -m unittest Code/test_ns_genus2_partition.py; PYTHONPATH='${REMOTE_ROOT}/Code:${REMOTE_ROOT}/Code/python' '${REMOTE_PYTHON}' Code/ns_genus2_cannon.py --config '${REMOTE_CONFIG}' plan"

TASK_COUNT=$(ssh "${SSH_HOST}" "cd '${REMOTE_ROOT}'; PYTHONPATH='${REMOTE_ROOT}/Code:${REMOTE_ROOT}/Code/python' '${REMOTE_PYTHON}' Code/ns_genus2_cannon.py --config '${REMOTE_CONFIG}' plan --task-count-only")
if [[ ${TASK_COUNT} -le 0 || ${ARRAY_CAP} -le 0 || ${TASKS_PER_ARRAY} -le 0 ]]; then
  echo "invalid task count, array cap, or tasks-per-array" >&2
  exit 2
fi
ARRAY_TASK_COUNT=$(((TASK_COUNT + TASKS_PER_ARRAY - 1) / TASKS_PER_ARRAY))

COMMON_EXPORT="ALL,NS_G2_ROOT=${REMOTE_ROOT},NS_G2_PYTHON=${REMOTE_PYTHON},NS_G2_CONFIG=${REMOTE_CONFIG},NS_G2_SHARDS=${REMOTE_SHARDS},NS_G2_TASK_COUNT=${TASK_COUNT},NS_G2_TASKS_PER_ARRAY=${TASKS_PER_ARRAY}"
ARRAY_JOB=$(ssh "${SSH_HOST}" "cd '${REMOTE_RUN_ROOT}'; sbatch --parsable --partition='${PARTITION}' --job-name='${JOB_TAG}' --output='logs/${JOB_TAG}-%A_%a.out' --error='logs/${JOB_TAG}-%A_%a.err' --array=0-$((ARRAY_TASK_COUNT-1))%${ARRAY_CAP} --export='${COMMON_EXPORT}' '${REMOTE_ROOT}/Code/cluster/ns_genus2_cannon_array.slurm'")
REDUCE_JOB=$(ssh "${SSH_HOST}" "cd '${REMOTE_RUN_ROOT}'; sbatch --parsable --partition='serial_requeue' --job-name='${JOB_TAG}-reduce' --output='logs/${JOB_TAG}-reduce-%j.out' --error='logs/${JOB_TAG}-reduce-%j.err' --dependency=afterok:${ARRAY_JOB} --kill-on-invalid-dep=yes --export='${COMMON_EXPORT},NS_G2_SUMMARY=${REMOTE_SUMMARY}' '${REMOTE_ROOT}/Code/cluster/ns_genus2_cannon_reduce.slurm'")

mkdir -p "$(dirname "${LOCAL_SUBMISSION_JSON}")"
printf '{\n  "status": "submitted",\n  "ssh_host": "%s",\n  "remote_run_root": "%s",\n  "remote_python": "%s",\n  "local_config": "%s",\n  "remote_config": "%s",\n  "partition": "%s",\n  "task_count": %d,\n  "tasks_per_array_element": %d,\n  "array_task_count": %d,\n  "array_cap": %d,\n  "array_job_id": "%s",\n  "reduce_job_id": "%s",\n  "remote_summary": "%s"\n}\n' \
  "${SSH_HOST}" "${REMOTE_RUN_ROOT}" "${REMOTE_PYTHON}" "${LOCAL_CONFIG}" "${REMOTE_CONFIG}" "${PARTITION}" \
  "${TASK_COUNT}" "${TASKS_PER_ARRAY}" "${ARRAY_TASK_COUNT}" "${ARRAY_CAP}" "${ARRAY_JOB}" "${REDUCE_JOB}" "${REMOTE_SUMMARY}" \
  > "${LOCAL_SUBMISSION_JSON}"

echo "array_job_id=${ARRAY_JOB}"
echo "reduce_job_id=${REDUCE_JOB}"
echo "task_count=${TASK_COUNT}"
echo "tasks_per_array_element=${TASKS_PER_ARRAY}"
echo "array_task_count=${ARRAY_TASK_COUNT}"
echo "submission_record=${LOCAL_SUBMISSION_JSON}"
echo "remote_summary=${REMOTE_SUMMARY}"
