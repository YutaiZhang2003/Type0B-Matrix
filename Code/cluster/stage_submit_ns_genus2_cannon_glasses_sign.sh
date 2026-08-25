#!/bin/bash
# Stage the corrected glasses self-loop-sign rerun and consistent recombination.

set -euo pipefail

if [[ $# -lt 5 || $# -gt 6 ]]; then
  echo "usage: $0 SSH_HOST REMOTE_RUN_ROOT REMOTE_PYTHON REMOTE_SOURCE_SUMMARY SOURCE_REDUCE_JOB [LOCAL_SUBMISSION_JSON]" >&2
  exit 2
fi

SSH_HOST=$1
REMOTE_RUN_ROOT=$2
REMOTE_PYTHON=$3
REMOTE_SOURCE_SUMMARY=$4
SOURCE_REDUCE_JOB=$5
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
LOCAL_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
LOCAL_SUBMISSION_JSON=${6:-"${LOCAL_ROOT}/Data Set/ns_genus2_cannon_glasses_toricsign_submission.json"}
PARTITION=${NS_G2_PARTITION:-yin}
ARRAY_CAP=${NS_G2_ARRAY_CAP:-200}
REMOTE_ROOT="${REMOTE_RUN_ROOT}/code"
REMOTE_CONFIG="${REMOTE_ROOT}/Code/config/ns_genus2_cannon_fivepoint_order8.json"
REMOTE_SHARDS="${REMOTE_RUN_ROOT}/glasses-shards"
REMOTE_SUMMARY="${REMOTE_RUN_ROOT}/summary.json"

# The task ordering is point-major and channel-minor, with 8^3=512 nodes per
# channel.  Only the five glasses ranges are changed by the toric sign fix.
GLASSES_ARRAY="512-1023,1536-2047,2560-3071,3584-4095,4608-5119%${ARRAY_CAP}"
GLASSES_TASK_COUNT=2560

ssh "${SSH_HOST}" "mkdir -p '${REMOTE_ROOT}/Code/genus_2' '${REMOTE_ROOT}/Code/genus_2_cross_channel' '${REMOTE_ROOT}/Code/config' '${REMOTE_ROOT}/Code/cluster' '${REMOTE_SHARDS}' '${REMOTE_RUN_ROOT}/logs'"

(
  cd "${LOCAL_ROOT}"
  rsync -azR \
    ./Code/sitecustomize.py \
    ./Code/genus_2/__init__.py \
    ./Code/genus_2/theta_partition.py \
    ./Code/c_Recursion/ns_genus2_cannon.py \
    ./Code/c_Recursion/ns_genus2_partition.py \
    ./Code/c_Recursion/test_ns_genus2_partition.py \
    ./Code/c_Recursion/compare_ns_torus_c_h_recursion.py \
    ./Code/c_Recursion/ns_genus_c_recursion_checks.py \
    ./Code/c_Recursion/ns_recursion_recipe.py \
    ./Code/c_Recursion/ns_global_osp_block.py \
    ./Code/c_Recursion/ns_regular_block.py \
    ./Code/c_Recursion/ns_vacuum_schottky.py \
    ./Code/c_Recursion/super_liouville_structure_constants.py \
    ./Code/c_Recursion/superconformal_blocks.py \
    ./Code/genus_2_cross_channel/ccy_genus2_block.py \
    ./Code/genus_2_cross_channel/genus2_vacuum_blocks.py \
    ./Code/genus_2_cross_channel/plumbing_algorithms.py \
    ./Code/genus_2_cross_channel/virasoro_blocks.py \
    ./Code/config/ns_genus2_cannon_fivepoint_order8.json \
    ./Code/cluster/ns_genus2_cannon_array.slurm \
    ./Code/cluster/ns_genus2_cannon_glasses_recombine.slurm \
    "${SSH_HOST}:${REMOTE_ROOT}/"
)

ssh "${SSH_HOST}" "set -e; cd '${REMOTE_ROOT}'; PYTHONPATH='${REMOTE_ROOT}/Code:${REMOTE_ROOT}/Code/genus_2_cross_channel' '${REMOTE_PYTHON}' -m unittest Code/c_Recursion/test_ns_genus2_partition.py; PYTHONPATH='${REMOTE_ROOT}/Code:${REMOTE_ROOT}/Code/genus_2_cross_channel' '${REMOTE_PYTHON}' Code/c_Recursion/ns_genus2_cannon.py --config '${REMOTE_CONFIG}' recombine-glasses --help >/dev/null"

COMMON_EXPORT="ALL,NS_G2_ROOT=${REMOTE_ROOT},NS_G2_PYTHON=${REMOTE_PYTHON},NS_G2_CONFIG=${REMOTE_CONFIG},NS_G2_SHARDS=${REMOTE_SHARDS}"
ARRAY_JOB=$(ssh "${SSH_HOST}" "cd '${REMOTE_RUN_ROOT}'; sbatch --parsable --partition='${PARTITION}' --array='${GLASSES_ARRAY}' --export='${COMMON_EXPORT}' '${REMOTE_ROOT}/Code/cluster/ns_genus2_cannon_array.slurm'")
REDUCE_JOB=$(ssh "${SSH_HOST}" "cd '${REMOTE_RUN_ROOT}'; sbatch --parsable --partition='serial_requeue' --dependency=afterok:${ARRAY_JOB}:${SOURCE_REDUCE_JOB} --kill-on-invalid-dep=yes --export='${COMMON_EXPORT},NS_G2_SOURCE_SUMMARY=${REMOTE_SOURCE_SUMMARY},NS_G2_SUMMARY=${REMOTE_SUMMARY}' '${REMOTE_ROOT}/Code/cluster/ns_genus2_cannon_glasses_recombine.slurm'")

mkdir -p "$(dirname "${LOCAL_SUBMISSION_JSON}")"
printf '{\n  "status": "submitted",\n  "scope": "glasses self-loop toric-sign fix plus consistent recombination",\n  "source_summary": "%s",\n  "source_reduce_job_id": "%s",\n  "ssh_host": "%s",\n  "remote_run_root": "%s",\n  "remote_python": "%s",\n  "partition": "%s",\n  "glasses_task_count": %d,\n  "array_cap": %d,\n  "array_job_id": "%s",\n  "reduce_job_id": "%s",\n  "remote_summary": "%s"\n}\n' \
  "${REMOTE_SOURCE_SUMMARY}" "${SOURCE_REDUCE_JOB}" "${SSH_HOST}" "${REMOTE_RUN_ROOT}" "${REMOTE_PYTHON}" "${PARTITION}" \
  "${GLASSES_TASK_COUNT}" "${ARRAY_CAP}" "${ARRAY_JOB}" "${REDUCE_JOB}" "${REMOTE_SUMMARY}" \
  > "${LOCAL_SUBMISSION_JSON}"

echo "array_job_id=${ARRAY_JOB}"
echo "reduce_job_id=${REDUCE_JOB}"
echo "source_reduce_job_id=${SOURCE_REDUCE_JOB}"
echo "glasses_task_count=${GLASSES_TASK_COUNT}"
echo "submission_record=${LOCAL_SUBMISSION_JSON}"
echo "remote_summary=${REMOTE_SUMMARY}"
