#!/bin/bash
# Stage the transported-spin theta numerator rerun and bosonized-free recombination.

set -euo pipefail

if [[ $# -lt 3 || $# -gt 4 ]]; then
  echo "usage: $0 SSH_HOST REMOTE_RUN_ROOT REMOTE_PYTHON [LOCAL_SUBMISSION_JSON]" >&2
  exit 2
fi

SSH_HOST=$1
REMOTE_RUN_ROOT=$2
REMOTE_PYTHON=$3
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
LOCAL_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
LOCAL_SUBMISSION_JSON=${4:-"${LOCAL_ROOT}/Data Set/ns_genus2_cannon_theta_spin_bosonized_submission.json"}
LOCAL_SOURCE_SUMMARY=${NS_G2_SOURCE_SUMMARY_LOCAL:-"${LOCAL_ROOT}/Data Set/ns_genus2_fivepoint_n8_r24_confluent_moments_summary.json"}
PARTITION=${NS_G2_PARTITION:-yin}
ARRAY_CAP=${NS_G2_ARRAY_CAP:-200}
REMOTE_ROOT="${REMOTE_RUN_ROOT}/code"
REMOTE_CONFIG="${REMOTE_ROOT}/Code/config/ns_genus2_cannon_fivepoint_order8.json"
REMOTE_SHARDS="${REMOTE_RUN_ROOT}/theta-shards"
REMOTE_SOURCE_SUMMARY="${REMOTE_RUN_ROOT}/source-summary.json"
REMOTE_SUMMARY="${REMOTE_RUN_ROOT}/summary.json"

# The original task ordering is point-major and channel-minor, with 8^3=512
# nodes per channel.  Only the five theta ranges are affected by the fix.
THETA_ARRAY="0-511,1024-1535,2048-2559,3072-3583,4096-4607%${ARRAY_CAP}"
THETA_TASK_COUNT=2560

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
    ./Code/genus_2_cross_channel/free_boson_plumbing.py \
    ./Code/genus_2_cross_channel/genus2_vacuum_blocks.py \
    ./Code/genus_2_cross_channel/plumbing_algorithms.py \
    ./Code/genus_2_cross_channel/virasoro_blocks.py \
    ./Code/config/ns_genus2_cannon_fivepoint_order8.json \
    ./Code/cluster/ns_genus2_cannon_array.slurm \
    ./Code/cluster/ns_genus2_cannon_theta_recombine.slurm \
    "${SSH_HOST}:${REMOTE_ROOT}/"
  rsync -az \
    "${LOCAL_SOURCE_SUMMARY}" \
    "${SSH_HOST}:${REMOTE_SOURCE_SUMMARY}"
)

ssh "${SSH_HOST}" "set -e; cd '${REMOTE_ROOT}'; PYTHONPATH='${REMOTE_ROOT}/Code:${REMOTE_ROOT}/Code/genus_2_cross_channel' '${REMOTE_PYTHON}' -m unittest Code/c_Recursion/test_ns_genus2_partition.py"

COMMON_EXPORT="ALL,NS_G2_ROOT=${REMOTE_ROOT},NS_G2_PYTHON=${REMOTE_PYTHON},NS_G2_CONFIG=${REMOTE_CONFIG},NS_G2_SHARDS=${REMOTE_SHARDS}"
ARRAY_JOB=$(ssh "${SSH_HOST}" "cd '${REMOTE_RUN_ROOT}'; sbatch --parsable --partition='${PARTITION}' --array='${THETA_ARRAY}' --export='${COMMON_EXPORT}' '${REMOTE_ROOT}/Code/cluster/ns_genus2_cannon_array.slurm'")
REDUCE_JOB=$(ssh "${SSH_HOST}" "cd '${REMOTE_RUN_ROOT}'; sbatch --parsable --partition='serial_requeue' --dependency=afterok:${ARRAY_JOB} --kill-on-invalid-dep=yes --export='${COMMON_EXPORT},NS_G2_SOURCE_SUMMARY=${REMOTE_SOURCE_SUMMARY},NS_G2_SUMMARY=${REMOTE_SUMMARY}' '${REMOTE_ROOT}/Code/cluster/ns_genus2_cannon_theta_recombine.slurm'")

mkdir -p "$(dirname "${LOCAL_SUBMISSION_JSON}")"
printf '{\n  "status": "submitted",\n  "scope": "theta [00|11] transported-spin numerator plus exact bosonized free-superfield recombination",\n  "source_summary": "%s",\n  "preserved_channel": "glasses",\n  "theta_edge_lifts": [1, 1, -1],\n  "free_denominator": "det(Im Omega)^(-1/2) |theta_delta| |P_bos|^3",\n  "ssh_host": "%s",\n  "remote_run_root": "%s",\n  "remote_python": "%s",\n  "partition": "%s",\n  "theta_task_count": %d,\n  "array_cap": %d,\n  "array_job_id": "%s",\n  "reduce_job_id": "%s",\n  "remote_summary": "%s"\n}\n' \
  "${LOCAL_SOURCE_SUMMARY}" "${SSH_HOST}" "${REMOTE_RUN_ROOT}" "${REMOTE_PYTHON}" "${PARTITION}" \
  "${THETA_TASK_COUNT}" "${ARRAY_CAP}" "${ARRAY_JOB}" "${REDUCE_JOB}" "${REMOTE_SUMMARY}" \
  > "${LOCAL_SUBMISSION_JSON}"

echo "array_job_id=${ARRAY_JOB}"
echo "reduce_job_id=${REDUCE_JOB}"
echo "theta_task_count=${THETA_TASK_COUNT}"
echo "submission_record=${LOCAL_SUBMISSION_JSON}"
echo "remote_summary=${REMOTE_SUMMARY}"
