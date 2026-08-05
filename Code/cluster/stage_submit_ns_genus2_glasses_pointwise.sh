#!/bin/bash
# Submit selected glasses nodes at specified recursion orders.  No channel
# comparison or reducer is launched: this run exists solely to establish
# internal recursion convergence of the glasses CFT block.

set -euo pipefail

if [[ $# -lt 3 || $# -gt 6 ]]; then
  echo "usage: $0 SSH_HOST REMOTE_RUN_ROOT REMOTE_PYTHON [LOCAL_SUBMISSION_JSON] [LOCAL_CONFIG] [TARGET_ARRAY]" >&2
  exit 2
fi

SSH_HOST=$1
REMOTE_RUN_ROOT=$2
REMOTE_PYTHON=$3
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
LOCAL_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
LOCAL_CONFIG=${5:-"${LOCAL_ROOT}/Code/config/ns_genus2_cannon_o0026_glasses_pointwise_r14_r16_r18.json"}
LOCAL_SUBMISSION_JSON=${4:-"${LOCAL_ROOT}/Data Set/ns_genus2_o0026_glasses_pointwise_r14_r16_r18_submission.json"}
PARTITION=${NS_G2_PARTITION:-yin}
ARRAY_CAP=${NS_G2_ARRAY_CAP:-24}
JOB_TAG=${NS_G2_JOB_TAG:-ns-g2-gl-pt}
REMOTE_ROOT="${REMOTE_RUN_ROOT}/code"
REMOTE_CONFIG="${REMOTE_ROOT}/Code/config/$(basename "${LOCAL_CONFIG}")"
REMOTE_SHARDS="${REMOTE_RUN_ROOT}/shards"

# For each order the generic task plan contains 512 theta tasks followed by
# 512 glasses tasks.  These are the eight documented glasses nodes at each of
# R=14,16,18.
TARGET_ARRAY=${6:-"512,519,575,583,731,960,967,1023,1536,1543,1599,1607,1755,1984,1991,2047,2560,2567,2623,2631,2779,3008,3015,3071"}
IFS=',' read -r -a TARGET_TASKS <<< "${TARGET_ARRAY}"
TARGET_TASK_COUNT=${#TARGET_TASKS[@]}

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
    ./Code/python/genus2_vacuum_blocks.py \
    ./Code/python/plumbing_algorithms.py \
    ./Code/python/virasoro_blocks.py \
    ./Code/cluster/ns_genus2_cannon_array.slurm \
    "${SSH_HOST}:${REMOTE_ROOT}/"
)
rsync -az "${LOCAL_CONFIG}" "${SSH_HOST}:${REMOTE_CONFIG}"

ssh "${SSH_HOST}" "set -e; cd '${REMOTE_ROOT}'; PYTHONPATH='${REMOTE_ROOT}/Code:${REMOTE_ROOT}/Code/python' '${REMOTE_PYTHON}' -m unittest Code/test_ns_genus2_partition.py; PYTHONPATH='${REMOTE_ROOT}/Code:${REMOTE_ROOT}/Code/python' '${REMOTE_PYTHON}' Code/ns_genus2_cannon.py --config '${REMOTE_CONFIG}' plan"

COMMON_EXPORT="ALL,NS_G2_ROOT=${REMOTE_ROOT},NS_G2_PYTHON=${REMOTE_PYTHON},NS_G2_CONFIG=${REMOTE_CONFIG},NS_G2_SHARDS=${REMOTE_SHARDS}"
ARRAY_JOB=$(ssh "${SSH_HOST}" "cd '${REMOTE_RUN_ROOT}'; sbatch --parsable --partition='${PARTITION}' --job-name='${JOB_TAG}' --output='logs/${JOB_TAG}-%A_%a.out' --error='logs/${JOB_TAG}-%A_%a.err' --array='${TARGET_ARRAY}%${ARRAY_CAP}' --export='${COMMON_EXPORT}' '${REMOTE_ROOT}/Code/cluster/ns_genus2_cannon_array.slurm'")

mkdir -p "$(dirname "${LOCAL_SUBMISSION_JSON}")"
printf '{\n  "status": "submitted",\n  "scope": "pointwise glasses recursion convergence only; no channel comparison",\n  "ssh_host": "%s",\n  "remote_run_root": "%s",\n  "remote_python": "%s",\n  "local_config": "%s",\n  "remote_config": "%s",\n  "partition": "%s",\n  "target_array": "%s",\n  "target_task_count": %d,\n  "array_job_id": "%s",\n  "remote_shards": "%s"\n}\n' \
  "${SSH_HOST}" "${REMOTE_RUN_ROOT}" "${REMOTE_PYTHON}" "${LOCAL_CONFIG}" "${REMOTE_CONFIG}" \
  "${PARTITION}" "${TARGET_ARRAY}" "${TARGET_TASK_COUNT}" "${ARRAY_JOB}" "${REMOTE_SHARDS}" \
  > "${LOCAL_SUBMISSION_JSON}"

echo "array_job_id=${ARRAY_JOB}"
echo "target_task_count=${TARGET_TASK_COUNT}"
echo "submission_record=${LOCAL_SUBMISSION_JSON}"
echo "remote_shards=${REMOTE_SHARDS}"
