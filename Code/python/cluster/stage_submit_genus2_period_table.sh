#!/bin/bash
# Stage and submit the certified genus-two period-table run on Cannon.

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
LOCAL_SUBMISSION_JSON=${4:-"${LOCAL_ROOT}/plumbing/results/genus2_period_table/design_v1/cluster_submission.json"}
DESIGN_DIR="${LOCAL_ROOT}/plumbing/results/genus2_period_table/design_v1"
ARRAY_CAP=${PERIOD_TABLE_ARRAY_CAP:-192}
TASK_COUNT=960
TASK_MEMORY=${PERIOD_TABLE_TASK_MEMORY:-3G}
CLUSTER_PARTITION=${STRINGMC_CLUSTER_PARTITION:-yin}

if [[ ${ARRAY_CAP} -le 0 || ${ARRAY_CAP} -gt ${TASK_COUNT} ]]; then
  echo "PERIOD_TABLE_ARRAY_CAP must satisfy 1 <= cap <= ${TASK_COUNT}" >&2
  exit 2
fi

test -f "${DESIGN_DIR}/manifest.csv"
test -f "${DESIGN_DIR}/manifest_summary.json"
test -f "${LOCAL_ROOT}/plumbing/config/genus2_period_table_cluster.json"

ssh "${SSH_HOST}" "mkdir -p '${REMOTE_RUN_ROOT}/code/plumbing/config' '${REMOTE_RUN_ROOT}/code/plumbing/cluster' '${REMOTE_RUN_ROOT}/design' '${REMOTE_RUN_ROOT}/shards' '${REMOTE_RUN_ROOT}/logs' '${REMOTE_RUN_ROOT}/assembled' '${REMOTE_RUN_ROOT}/validation'"

rsync -az \
  --include='*.py' \
  --exclude='*' \
  "${LOCAL_ROOT}/plumbing/" \
  "${SSH_HOST}:${REMOTE_RUN_ROOT}/code/plumbing/"
rsync -az \
  "${LOCAL_ROOT}/plumbing/config/genus2_period_table_cluster.json" \
  "${SSH_HOST}:${REMOTE_RUN_ROOT}/code/plumbing/config/"
rsync -az \
  "${LOCAL_ROOT}/plumbing/cluster/genus2_period_table_array.slurm" \
  "${LOCAL_ROOT}/plumbing/cluster/genus2_period_table_assemble.slurm" \
  "${LOCAL_ROOT}/plumbing/cluster/genus2_period_table_validate.slurm" \
  "${SSH_HOST}:${REMOTE_RUN_ROOT}/code/plumbing/cluster/"
rsync -az \
  "${DESIGN_DIR}/manifest.csv" \
  "${DESIGN_DIR}/manifest_summary.json" \
  "${DESIGN_DIR}/config.snapshot.json" \
  "${LOCAL_ROOT}/plumbing/results/genus2_full_moduli_coverage/full_moduli_combined.csv" \
  "${SSH_HOST}:${REMOTE_RUN_ROOT}/design/"

ssh "${SSH_HOST}" "set -e; cd '${REMOTE_RUN_ROOT}/code'; '${REMOTE_PYTHON}' -c 'import numpy, scipy, mpmath; print(numpy.__version__, scipy.__version__, mpmath.__version__)'; '${REMOTE_PYTHON}' plumbing/genus2_period_table_cluster.py --config plumbing/config/genus2_period_table_cluster.json preflight --manifest '${REMOTE_RUN_ROOT}/design/manifest.csv'"

ARRAY_JOB=$(ssh "${SSH_HOST}" "cd '${REMOTE_RUN_ROOT}'; sbatch --parsable --partition='${CLUSTER_PARTITION}' --array=0-$((TASK_COUNT-1))%${ARRAY_CAP} --mem='${TASK_MEMORY}' --export=ALL,STRINGMC_ROOT='${REMOTE_RUN_ROOT}/code',STRINGMC_PYTHON='${REMOTE_PYTHON}',PERIOD_TABLE_MANIFEST='${REMOTE_RUN_ROOT}/design/manifest.csv',PERIOD_TABLE_SHARDS='${REMOTE_RUN_ROOT}/shards' '${REMOTE_RUN_ROOT}/code/plumbing/cluster/genus2_period_table_array.slurm'")
ASSEMBLY_JOB=$(ssh "${SSH_HOST}" "cd '${REMOTE_RUN_ROOT}'; sbatch --parsable --partition='${CLUSTER_PARTITION}' --dependency=afterok:${ARRAY_JOB} --export=ALL,STRINGMC_ROOT='${REMOTE_RUN_ROOT}/code',STRINGMC_PYTHON='${REMOTE_PYTHON}',PERIOD_TABLE_RUN_ROOT='${REMOTE_RUN_ROOT}' '${REMOTE_RUN_ROOT}/code/plumbing/cluster/genus2_period_table_assemble.slurm'")
VALIDATION_JOB=$(ssh "${SSH_HOST}" "cd '${REMOTE_RUN_ROOT}'; sbatch --parsable --partition='${CLUSTER_PARTITION}' --dependency=afterok:${ASSEMBLY_JOB} --export=ALL,STRINGMC_ROOT='${REMOTE_RUN_ROOT}/code',STRINGMC_PYTHON='${REMOTE_PYTHON}',PERIOD_TABLE_RUN_ROOT='${REMOTE_RUN_ROOT}' '${REMOTE_RUN_ROOT}/code/plumbing/cluster/genus2_period_table_validate.slurm'")

mkdir -p "$(dirname "${LOCAL_SUBMISSION_JSON}")"
printf '{\n  "ssh_host": "%s",\n  "remote_run_root": "%s",\n  "remote_python": "%s",\n  "partition": "%s",\n  "task_count": %d,\n  "array_cap": %d,\n  "task_memory": "%s",\n  "array_job_id": "%s",\n  "assembly_job_id": "%s",\n  "validation_job_id": "%s"\n}\n' \
  "${SSH_HOST}" "${REMOTE_RUN_ROOT}" "${REMOTE_PYTHON}" \
  "${CLUSTER_PARTITION}" "${TASK_COUNT}" "${ARRAY_CAP}" "${TASK_MEMORY}" \
  "${ARRAY_JOB}" "${ASSEMBLY_JOB}" "${VALIDATION_JOB}" > "${LOCAL_SUBMISSION_JSON}"

echo "array_job_id=${ARRAY_JOB}"
echo "assembly_job_id=${ASSEMBLY_JOB}"
echo "validation_job_id=${VALIDATION_JOB}"
echo "submission_record=${LOCAL_SUBMISSION_JSON}"
