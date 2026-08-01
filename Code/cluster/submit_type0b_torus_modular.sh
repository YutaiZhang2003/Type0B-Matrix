#!/bin/bash
# Submit the momentum-node array and its deterministic reduction dependency.

set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "usage: $0 RUN_ROOT CLUSTER_PYTHON [CONFIG_JSON]" >&2
  exit 2
fi

RUN_ROOT=$1
TYPE0B_PYTHON=$2
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
TYPE0B_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
SOURCE_CONFIG=${3:-"${TYPE0B_ROOT}/config/type0b_torus_modular_cluster.json"}
CONFIG_SNAPSHOT="${RUN_ROOT}/config.snapshot.json"
SHARD_DIR="${RUN_ROOT}/shards"
SUMMARY="${RUN_ROOT}/summary.json"
SUBMISSION_RECORD="${RUN_ROOT}/submission.json"
PARTITION=${TYPE0B_PARTITION:-}
ARRAY_CAP=${TYPE0B_ARRAY_CAP:-16}

test -f "${SOURCE_CONFIG}"
test -x "${TYPE0B_PYTHON}"
mkdir -p "${RUN_ROOT}" "${SHARD_DIR}" "${RUN_ROOT}/logs"
cp "${SOURCE_CONFIG}" "${CONFIG_SNAPSHOT}"

SHARD_COUNT=$(
  "${TYPE0B_PYTHON}" -c \
    'import json,sys; print(int(json.load(open(sys.argv[1]))["default_shard_count"]))' \
    "${CONFIG_SNAPSHOT}"
)

if [[ ${ARRAY_CAP} -le 0 || ${ARRAY_CAP} -gt ${SHARD_COUNT} ]]; then
  echo "TYPE0B_ARRAY_CAP must satisfy 1 <= cap <= ${SHARD_COUNT}" >&2
  exit 2
fi

"${TYPE0B_PYTHON}" "${TYPE0B_ROOT}/super_liouville_torus_modular_cluster.py" \
  --config "${CONFIG_SNAPSHOT}" \
  plan \
  --shard-count "${SHARD_COUNT}"

SBATCH_PARTITION=()
if [[ -n ${PARTITION} ]]; then
  SBATCH_PARTITION=(--partition="${PARTITION}")
fi

ARRAY_JOB=$(
  cd "${RUN_ROOT}"
  sbatch --parsable \
    "${SBATCH_PARTITION[@]}" \
    --array="0-$((SHARD_COUNT - 1))%${ARRAY_CAP}" \
    --export="ALL,TYPE0B_ROOT=${TYPE0B_ROOT},TYPE0B_PYTHON=${TYPE0B_PYTHON},TYPE0B_MODULAR_CONFIG=${CONFIG_SNAPSHOT},TYPE0B_MODULAR_SHARDS=${SHARD_DIR},TYPE0B_MODULAR_SHARD_COUNT=${SHARD_COUNT}" \
    "${TYPE0B_ROOT}/cluster/type0b_torus_modular_array.slurm"
)

REDUCE_JOB=$(
  cd "${RUN_ROOT}"
  sbatch --parsable \
    "${SBATCH_PARTITION[@]}" \
    --dependency="afterok:${ARRAY_JOB}" \
    --kill-on-invalid-dep=yes \
    --export="ALL,TYPE0B_ROOT=${TYPE0B_ROOT},TYPE0B_PYTHON=${TYPE0B_PYTHON},TYPE0B_MODULAR_CONFIG=${CONFIG_SNAPSHOT},TYPE0B_MODULAR_SHARDS=${SHARD_DIR},TYPE0B_MODULAR_SUMMARY=${SUMMARY},TYPE0B_MODULAR_SHARD_COUNT=${SHARD_COUNT}" \
    "${TYPE0B_ROOT}/cluster/type0b_torus_modular_reduce.slurm"
)

printf '{\n  "run_root": "%s",\n  "repository_root": "%s",\n  "python": "%s",\n  "config_snapshot": "%s",\n  "shard_count": %d,\n  "array_cap": %d,\n  "partition": "%s",\n  "array_job_id": "%s",\n  "reduce_job_id": "%s",\n  "summary": "%s"\n}\n' \
  "${RUN_ROOT}" "${TYPE0B_ROOT}" "${TYPE0B_PYTHON}" \
  "${CONFIG_SNAPSHOT}" "${SHARD_COUNT}" "${ARRAY_CAP}" "${PARTITION}" \
  "${ARRAY_JOB}" "${REDUCE_JOB}" "${SUMMARY}" > "${SUBMISSION_RECORD}"

echo "array_job_id=${ARRAY_JOB}"
echo "reduce_job_id=${REDUCE_JOB}"
echo "submission_record=${SUBMISSION_RECORD}"
echo "summary=${SUMMARY}"
