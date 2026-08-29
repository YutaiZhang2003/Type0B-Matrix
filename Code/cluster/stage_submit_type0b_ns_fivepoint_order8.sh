#!/bin/bash
# Stage and submit the order-eight genus-zero Type-0B five-point array.

set -euo pipefail

if [[ $# -lt 3 || $# -gt 5 ]]; then
  echo "usage: $0 SSH_HOST REMOTE_RUN_ROOT REMOTE_PYTHON [LOCAL_CONFIG] [LOCAL_SUBMISSION_JSON]" >&2
  exit 2
fi

SSH_HOST=$1
REMOTE_RUN_ROOT=$2
REMOTE_PYTHON=$3
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
LOCAL_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
LOCAL_CONFIG=${4:-"${LOCAL_ROOT}/Code/config/type0b_ns_five_tachyon_c_recursion_order8_small_collar_cluster.json"}
LOCAL_SUBMISSION_JSON=${5:-"${LOCAL_ROOT}/Data Set/type0b_ns_five_tachyon_order8_small_collar_submission.json"}
if [[ ${LOCAL_CONFIG} != /* ]]; then
  LOCAL_CONFIG="${LOCAL_ROOT}/${LOCAL_CONFIG}"
fi
test -f "${LOCAL_CONFIG}"

PARTITION=${TYPE0B_5PT_PARTITION:-yin}
ARRAY_CAP=${TYPE0B_5PT_ARRAY_CAP:-4}
JOB_TAG=${TYPE0B_5PT_JOB_TAG:-t0b-5pt-o8}
STAGE_ONLY=${TYPE0B_5PT_STAGE_ONLY:-0}
REMOTE_ROOT="${REMOTE_RUN_ROOT}/code"
REMOTE_CONFIG="${REMOTE_ROOT}/Code/config/$(basename "${LOCAL_CONFIG}")"
REMOTE_SHARDS="${REMOTE_RUN_ROOT}/shards"
REMOTE_COEFFICIENT_CACHE="${REMOTE_RUN_ROOT}/coefficient-cache"
REMOTE_SUMMARY="${REMOTE_RUN_ROOT}/summary.json"

ssh "${SSH_HOST}" "mkdir -p '${REMOTE_ROOT}/Code/higher_point_amplitude_attempts/type0b_ns_five_tachyon' '${REMOTE_ROOT}/Code/c_Recursion' '${REMOTE_ROOT}/Code/bosonic_c1_one_to_n_reference/reference_implementation/plumbing' '${REMOTE_ROOT}/Code/config' '${REMOTE_ROOT}/Code/cluster' '${REMOTE_SHARDS}' '${REMOTE_COEFFICIENT_CACHE}' '${REMOTE_RUN_ROOT}/logs'"

(
  cd "${LOCAL_ROOT}"
  rsync -azR \
    --include='*/' --include='*.py' --exclude='*' \
    ./Code/higher_point_amplitude_attempts/type0b_ns_five_tachyon/ \
    ./Code/c_Recursion/ \
    ./Code/bosonic_c1_one_to_n_reference/reference_implementation/plumbing/ \
    "${SSH_HOST}:${REMOTE_ROOT}/"
  rsync -azR \
    ./Code/cluster/type0b_ns_fivepoint_order8_array.slurm \
    ./Code/cluster/type0b_ns_fivepoint_order8_reduce.slurm \
    "${SSH_HOST}:${REMOTE_ROOT}/"
)
rsync -az "${LOCAL_CONFIG}" "${SSH_HOST}:${REMOTE_CONFIG}"

REMOTE_PYTHONPATH="${REMOTE_ROOT}/Code/higher_point_amplitude_attempts/type0b_ns_five_tachyon:${REMOTE_ROOT}/Code/c_Recursion:${REMOTE_ROOT}/Code/bosonic_c1_one_to_n_reference/reference_implementation/plumbing"
ssh "${SSH_HOST}" "set -e; '${REMOTE_PYTHON}' -c 'import numpy,scipy,mpmath; print(numpy.__version__,scipy.__version__,mpmath.__version__)'; cd '${REMOTE_ROOT}'; PYTHONPATH='${REMOTE_PYTHONPATH}' '${REMOTE_PYTHON}' -m unittest Code/c_Recursion/test_ns_multipoint_h_recursion.py Code/higher_point_amplitude_attempts/type0b_ns_five_tachyon/test_type0b_ns_five_tachyon.py Code/higher_point_amplitude_attempts/type0b_ns_five_tachyon/test_type0b_ns_five_tachyon_domain.py Code/higher_point_amplitude_attempts/type0b_ns_five_tachyon/test_run_type0b_ns_five_tachyon_cluster.py; PYTHONPATH='${REMOTE_PYTHONPATH}' '${REMOTE_PYTHON}' Code/higher_point_amplitude_attempts/type0b_ns_five_tachyon/run_type0b_ns_five_tachyon_cluster.py --config '${REMOTE_CONFIG}' plan"

TASK_COUNT=$(ssh "${SSH_HOST}" "cd '${REMOTE_ROOT}'; PYTHONPATH='${REMOTE_PYTHONPATH}' '${REMOTE_PYTHON}' Code/higher_point_amplitude_attempts/type0b_ns_five_tachyon/run_type0b_ns_five_tachyon_cluster.py --config '${REMOTE_CONFIG}' plan --task-count-only")
if [[ ${TASK_COUNT} -le 0 || ${ARRAY_CAP} -le 0 ]]; then
  echo "invalid task count or array cap" >&2
  exit 2
fi

if [[ ${STAGE_ONLY} == 1 ]]; then
  mkdir -p "$(dirname "${LOCAL_SUBMISSION_JSON}")"
  printf '{\n  "status": "staged_not_submitted",\n  "ssh_host": "%s",\n  "remote_run_root": "%s",\n  "remote_python": "%s",\n  "local_config": "%s",\n  "remote_config": "%s",\n  "partition": "%s",\n  "task_count": %d,\n  "array_cap": %d,\n  "remote_summary": "%s"\n}\n' \
    "${SSH_HOST}" "${REMOTE_RUN_ROOT}" "${REMOTE_PYTHON}" "${LOCAL_CONFIG}" \
    "${REMOTE_CONFIG}" "${PARTITION}" "${TASK_COUNT}" "${ARRAY_CAP}" \
    "${REMOTE_SUMMARY}" > "${LOCAL_SUBMISSION_JSON}"
  rsync -az "${LOCAL_SUBMISSION_JSON}" "${SSH_HOST}:${REMOTE_RUN_ROOT}/staging_record.json"
  echo "status=staged_not_submitted"
  echo "task_count=${TASK_COUNT}"
  echo "staging_record=${LOCAL_SUBMISSION_JSON}"
  echo "remote_summary=${REMOTE_SUMMARY}"
  exit 0
fi

COMMON_EXPORT="ALL,TYPE0B_5PT_ROOT=${REMOTE_ROOT},TYPE0B_5PT_PYTHON=${REMOTE_PYTHON},TYPE0B_5PT_CONFIG=${REMOTE_CONFIG},TYPE0B_5PT_SHARDS=${REMOTE_SHARDS},TYPE0B_5PT_COEFFICIENT_CACHE=${REMOTE_COEFFICIENT_CACHE},TYPE0B_5PT_TASK_COUNT=${TASK_COUNT}"
ARRAY_JOB=$(ssh "${SSH_HOST}" "cd '${REMOTE_RUN_ROOT}'; sbatch --parsable --partition='${PARTITION}' --job-name='${JOB_TAG}' --array=0-$((TASK_COUNT-1))%${ARRAY_CAP} --export='${COMMON_EXPORT}' '${REMOTE_ROOT}/Code/cluster/type0b_ns_fivepoint_order8_array.slurm'")
REDUCE_JOB=$(ssh "${SSH_HOST}" "cd '${REMOTE_RUN_ROOT}'; sbatch --parsable --partition='serial_requeue' --job-name='${JOB_TAG}-reduce' --dependency=afterok:${ARRAY_JOB} --kill-on-invalid-dep=yes --export='${COMMON_EXPORT},TYPE0B_5PT_SUMMARY=${REMOTE_SUMMARY}' '${REMOTE_ROOT}/Code/cluster/type0b_ns_fivepoint_order8_reduce.slurm'")

mkdir -p "$(dirname "${LOCAL_SUBMISSION_JSON}")"
printf '{\n  "status": "submitted",\n  "ssh_host": "%s",\n  "remote_run_root": "%s",\n  "remote_python": "%s",\n  "local_config": "%s",\n  "remote_config": "%s",\n  "partition": "%s",\n  "task_count": %d,\n  "array_cap": %d,\n  "array_job_id": "%s",\n  "reduce_job_id": "%s",\n  "remote_summary": "%s"\n}\n' \
  "${SSH_HOST}" "${REMOTE_RUN_ROOT}" "${REMOTE_PYTHON}" "${LOCAL_CONFIG}" \
  "${REMOTE_CONFIG}" "${PARTITION}" "${TASK_COUNT}" "${ARRAY_CAP}" \
  "${ARRAY_JOB}" "${REDUCE_JOB}" "${REMOTE_SUMMARY}" > "${LOCAL_SUBMISSION_JSON}"

echo "array_job_id=${ARRAY_JOB}"
echo "reduce_job_id=${REDUCE_JOB}"
echo "task_count=${TASK_COUNT}"
echo "submission_record=${LOCAL_SUBMISSION_JSON}"
echo "remote_summary=${REMOTE_SUMMARY}"
