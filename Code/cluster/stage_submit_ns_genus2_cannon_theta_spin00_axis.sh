#!/bin/bash
# Rerun only theta numerators after correcting the physical Human Note
# plumbing-spin lift to [00|00].

set -euo pipefail

if [[ $# -lt 3 || $# -gt 5 ]]; then
  echo "usage: $0 SSH_HOST REMOTE_RUN_ROOT REMOTE_PYTHON [LOCAL_SUBMISSION_JSON] [LOCAL_SOURCE_SUMMARY]" >&2
  exit 2
fi

SSH_HOST=$1
REMOTE_RUN_ROOT=$2
REMOTE_PYTHON=$3
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
LOCAL_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
LOCAL_SUBMISSION_JSON=${4:-"${LOCAL_ROOT}/Data Set/ns_genus2_cross_sewing_r24_n10_human_note_spin00_submission.json"}
LOCAL_SOURCE_SUMMARY=${5:-"${LOCAL_ROOT}/Data Set/ns_genus2_cross_sewing_r24_n10_glasses_parity_summary.json"}
LOCAL_CONFIG="${LOCAL_ROOT}/Code/config/ns_genus2_cross_sewing_r24_n10_human_note_spin00.json"
PARTITION=${NS_G2_PARTITION:-yin}
ARRAY_CAP=${NS_G2_ARRAY_CAP:-192}
TASKS_PER_CHUNK=${NS_G2_TASKS_PER_CHUNK:-20}
JOB_TAG=${NS_G2_JOB_TAG:-ns-g2-human-note-spin00-theta}
REMOTE_ROOT="${REMOTE_RUN_ROOT}/code"
REMOTE_CONFIG="${REMOTE_ROOT}/Code/config/$(basename "${LOCAL_CONFIG}")"
REMOTE_SHARDS="${REMOTE_RUN_ROOT}/theta-shards"
REMOTE_SOURCE_SUMMARY="${REMOTE_RUN_ROOT}/source-summary.json"
REMOTE_CHUNKS="${REMOTE_RUN_ROOT}/theta-chunks.tsv"
REMOTE_SUMMARY="${REMOTE_RUN_ROOT}/summary.json"

for required in "${LOCAL_CONFIG}" "${LOCAL_SOURCE_SUMMARY}"; do
  if [[ ! -f ${required} ]]; then
    echo "missing input: ${required}" >&2
    exit 2
  fi
done

ssh "${SSH_HOST}" "mkdir -p '${REMOTE_ROOT}/Code/genus_2' '${REMOTE_ROOT}/Code/genus_2_cross_channel' '${REMOTE_ROOT}/Code/config' '${REMOTE_ROOT}/Code/cluster' '${REMOTE_SHARDS}' '${REMOTE_RUN_ROOT}/logs'"

(
  cd "${LOCAL_ROOT}"
  rsync -azR \
    ./Code/sitecustomize.py \
    ./Code/genus_2/__init__.py \
    ./Code/genus_2/glasses_partition.py \
    ./Code/genus_2/theta_partition.py \
    ./Code/c_Recursion/ns_genus2_cannon.py \
    ./Code/c_Recursion/ns_genus2_partition.py \
    ./Code/c_Recursion/test_ns_genus2_partition.py \
    "./Code/PBW_c_recursion_double_virasoro crosscheck/test_free_majorana_pair_of_pants.py" \
    ./Code/c_Recursion/compare_ns_torus_c_h_recursion.py \
    ./Code/c_Recursion/ns_genus_c_recursion_checks.py \
    ./Code/c_Recursion/ns_human_convention.py \
    ./Code/c_Recursion/ns_recursion_recipe.py \
    ./Code/c_Recursion/ns_global_osp_block.py \
    ./Code/c_Recursion/ns_regular_block.py \
    ./Code/c_Recursion/ns_vacuum_schottky.py \
    ./Code/c_Recursion/super_liouville_structure_constants.py \
    ./Code/c_Recursion/superconformal_blocks.py \
    ./Code/genus_2_cross_channel/ccy_genus2_block.py \
    ./Code/genus_2_cross_channel/free_boson_plumbing.py \
    ./Code/genus_2_cross_channel/free_majorana_pair_of_pants.py \
    ./Code/genus_2_cross_channel/genus2_vacuum_blocks.py \
    ./Code/genus_2_cross_channel/plumbing_algorithms.py \
    ./Code/genus_2_cross_channel/virasoro_blocks.py \
    ./Code/cluster/ns_genus2_cannon_channel_array.slurm \
    ./Code/cluster/ns_genus2_cannon_theta_recombine.slurm \
    "${SSH_HOST}:${REMOTE_ROOT}/"
  rsync -az "${LOCAL_CONFIG}" "${SSH_HOST}:${REMOTE_CONFIG}"
  rsync -az "${LOCAL_SOURCE_SUMMARY}" "${SSH_HOST}:${REMOTE_SOURCE_SUMMARY}"
)

ssh "${SSH_HOST}" "set -e; cd '${REMOTE_ROOT}'; export PYTHONPATH='${REMOTE_ROOT}/Code:${REMOTE_ROOT}/Code/genus_2_cross_channel'; '${REMOTE_PYTHON}' -m unittest Code/c_Recursion/test_ns_genus2_partition.py 'Code/PBW_c_recursion_double_virasoro crosscheck/test_free_majorana_pair_of_pants.py'; '${REMOTE_PYTHON}' Code/c_Recursion/ns_genus2_cannon.py --config '${REMOTE_CONFIG}' plan; '${REMOTE_PYTHON}' Code/c_Recursion/ns_genus2_cannon.py --config '${REMOTE_CONFIG}' channel-chunks --channel theta --tasks-per-chunk '${TASKS_PER_CHUNK}' > '${REMOTE_CHUNKS}'"

CHUNK_COUNT=$(ssh "${SSH_HOST}" "wc -l < '${REMOTE_CHUNKS}' | tr -d ' '")
THETA_TASK_COUNT=$(ssh "${SSH_HOST}" "awk '{n += \$2 - \$1 + 1} END {print n + 0}' '${REMOTE_CHUNKS}'")
if [[ ${CHUNK_COUNT} -le 0 || ${THETA_TASK_COUNT} -le 0 ]]; then
  echo "invalid theta chunk manifest" >&2
  exit 2
fi

COMMON_EXPORT="ALL,NS_G2_ROOT=${REMOTE_ROOT},NS_G2_PYTHON=${REMOTE_PYTHON},NS_G2_CONFIG=${REMOTE_CONFIG},NS_G2_SHARDS=${REMOTE_SHARDS},NS_G2_CHUNK_MANIFEST=${REMOTE_CHUNKS}"
ARRAY_JOB=$(ssh "${SSH_HOST}" "cd '${REMOTE_RUN_ROOT}'; sbatch --parsable --partition='${PARTITION}' --job-name='${JOB_TAG}' --output='logs/${JOB_TAG}-%A_%a.out' --error='logs/${JOB_TAG}-%A_%a.err' --array=0-$((CHUNK_COUNT-1))%${ARRAY_CAP} --export='${COMMON_EXPORT}' '${REMOTE_ROOT}/Code/cluster/ns_genus2_cannon_channel_array.slurm'")
REDUCE_JOB=$(ssh "${SSH_HOST}" "cd '${REMOTE_RUN_ROOT}'; sbatch --parsable --partition='serial_requeue' --job-name='${JOB_TAG}-reduce' --output='logs/${JOB_TAG}-reduce-%j.out' --error='logs/${JOB_TAG}-reduce-%j.err' --dependency=afterok:${ARRAY_JOB} --kill-on-invalid-dep=yes --export='${COMMON_EXPORT},NS_G2_SOURCE_SUMMARY=${REMOTE_SOURCE_SUMMARY},NS_G2_SUMMARY=${REMOTE_SUMMARY}' '${REMOTE_ROOT}/Code/cluster/ns_genus2_cannon_theta_recombine.slurm'")

mkdir -p "$(dirname "${LOCAL_SUBMISSION_JSON}")"
printf '{\n  "status": "submitted",\n  "scope": "theta numerator rerun with the Human Note nonchiral sign at physical spin [00|00]; glasses numerator preserved",\n  "source_summary": "%s",\n  "preserved_channel": "glasses",\n  "theta_edge_lifts": [1, -1, 1],\n  "glasses_edge_lifts": [1, 1, 1],\n  "spin_characteristic": {"alpha": [0, 0], "beta": [0, 0]},\n  "free_denominator_role": "physical one-boson plus one-NS-Majorana partition function; auxiliary double-Virasoro fermion excluded",\n  "free_denominator": "det(Im Omega)^(-1/2) |theta_delta| |P_X|^3",\n  "ssh_host": "%s",\n  "remote_run_root": "%s",\n  "remote_python": "%s",\n  "partition": "%s",\n  "theta_task_count": %d,\n  "tasks_per_array_element": %d,\n  "array_task_count": %d,\n  "array_cap": %d,\n  "array_job_id": "%s",\n  "reduce_job_id": "%s",\n  "remote_summary": "%s"\n}\n' \
  "${LOCAL_SOURCE_SUMMARY}" "${SSH_HOST}" "${REMOTE_RUN_ROOT}" "${REMOTE_PYTHON}" "${PARTITION}" \
  "${THETA_TASK_COUNT}" "${TASKS_PER_CHUNK}" "${CHUNK_COUNT}" "${ARRAY_CAP}" "${ARRAY_JOB}" "${REDUCE_JOB}" "${REMOTE_SUMMARY}" \
  > "${LOCAL_SUBMISSION_JSON}"

echo "array_job_id=${ARRAY_JOB}"
echo "reduce_job_id=${REDUCE_JOB}"
echo "theta_task_count=${THETA_TASK_COUNT}"
echo "array_task_count=${CHUNK_COUNT}"
echo "submission_record=${LOCAL_SUBMISSION_JSON}"
echo "remote_summary=${REMOTE_SUMMARY}"
