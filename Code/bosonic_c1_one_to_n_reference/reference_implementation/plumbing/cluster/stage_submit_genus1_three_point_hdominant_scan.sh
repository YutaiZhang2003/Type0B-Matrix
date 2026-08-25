#!/bin/bash
# Stage and submit the blind ten-point h-dominant torus-three-point scan.

set -euo pipefail

if [[ $# -gt 4 ]]; then
  echo "usage: $0 [SSH_HOST] [REMOTE_RUN_ROOT] [REMOTE_PYTHON] [LOCAL_SUBMISSION_JSON]" >&2
  exit 2
fi

SSH_HOST=${1:-cannon}
REMOTE_RUN_ROOT=${2:-/n/holylabs/yin_lab/Everyone/yutaizhang/StringMC/genus1_three_point_hdominant_scan10_20260824_v1}
REMOTE_PYTHON=${3:-/n/holylabs/yin_lab/Everyone/yutaizhang/StringMC/genus2_period_table_20260718_176ab14d/.venv/bin/python}
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
LOCAL_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
LOCAL_PYTHON=${STRINGMC_LOCAL_PYTHON:-"${LOCAL_ROOT}/.venv/bin/python"}
LOCAL_DESIGN="${LOCAL_ROOT}/plumbing/results/genus1_three_point_worldsheet/hdominant_scan10_p8_h8l3_q030_007_n256_r4_v1"
LOCAL_SUBMISSION_JSON=${4:-"${LOCAL_DESIGN}/cluster_submission.json"}
LOCAL_REUSED_T075="${LOCAL_ROOT}/plumbing/results/genus1_three_point_worldsheet/channel_atlas_hdominant_t075_p8_h8l3_q030_007_n256_r4_v1.json"
LOCAL_LEGACY_SCAN="${LOCAL_ROOT}/plumbing/results/genus1_three_point_worldsheet/equal_split_imaginary_t_scan10_p12_n256_v1/worldsheet_scan_manifest.json"

REMOTE_CODE_ROOT="${REMOTE_RUN_ROOT}/code"
REMOTE_TASKS="${REMOTE_RUN_ROOT}/design/cannon_tasks.csv"
REMOTE_REUSED_T075="${REMOTE_RUN_ROOT}/reused/t075.json"
REMOTE_LEGACY_SCAN="${REMOTE_RUN_ROOT}/legacy/worldsheet_scan_manifest.json"
REMOTE_SHARDS="${REMOTE_RUN_ROOT}/shards"
REMOTE_CACHE="${REMOTE_RUN_ROOT}/cache"
REMOTE_ASSEMBLED="${REMOTE_RUN_ROOT}/assembled"

test -x "${LOCAL_PYTHON}"
test -f "${LOCAL_REUSED_T075}"
test -f "${LOCAL_LEGACY_SCAN}"

"${LOCAL_PYTHON}" "${LOCAL_ROOT}/plumbing/genus1_three_point_hdominant_cannon.py" prepare \
  --design-dir "${LOCAL_DESIGN}" \
  --reused-t075 "${LOCAL_REUSED_T075}"

TASK_COUNT=$("${LOCAL_PYTHON}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["task_count"])' "${LOCAL_DESIGN}/design_summary.json")
if [[ "${TASK_COUNT}" -ne 9 ]]; then
  echo "expected nine new t tasks, got ${TASK_COUNT}" >&2
  exit 1
fi
LAST_TASK=$((TASK_COUNT - 1))

CODE_NAMES=(
  bolza_ccy_recursion.py
  bolza_torus_plumbing_reach.py
  ccy_genus2_block.py
  ccy_plumbing_conventions.py
  genus1_three_point_channel_atlas.py
  genus1_three_point_hdominant_cannon.py
  genus1_three_point_worldsheet.py
  genus1_two_point_worldsheet.py
  genus2_calibrated_schottky.py
  genus2_holomorphic_period_table.py
  genus2_hybrid_period_map.py
  genus2_multiprecision_collocation.py
  genus2_period_table.py
  genus2_period_table_grid.py
  genus2_period_table_selector.py
  genus2_siegel_fundamental_domain.py
  genus2_vacuum_blocks.py
  integrate_genus1_three_point_worldsheet.py
  liouville_genus2.py
  liouville_genus2_ccy.py
  liouville_genus2_modular_check.py
  liouville_momentum_quadrature.py
  liouville_torus.py
  plumbing_algorithms.py
  refine_genus1_three_point_worldsheet.py
  run_genus1_three_point_hdominant_scan.py
  smoke_genus1_three_point_channel_atlas.py
  torus_descendant_blocks.py
  torus_three_point_blocks.py
  torus_three_point_ope_blocks.py
  torus_two_point_blocks.py
  virasoro_blocks.py
  virasoro_descendant_algebra.py
  virasoro_plumbing_graph.py
)
CODE_PATHS=()
for name in "${CODE_NAMES[@]}"; do
  path="${LOCAL_ROOT}/plumbing/${name}"
  test -f "${path}"
  CODE_PATHS+=("${path}")
done
SLURM_PATHS=(
  "${LOCAL_ROOT}/plumbing/cluster/genus1_three_point_hdominant_array.slurm"
  "${LOCAL_ROOT}/plumbing/cluster/genus1_three_point_hdominant_assemble.slurm"
)

mkdir -p "${LOCAL_DESIGN}"
"${LOCAL_PYTHON}" -c 'import datetime,hashlib,json,pathlib,sys; root=pathlib.Path(sys.argv[1]).resolve(); paths=[pathlib.Path(value) for value in sys.argv[3:]]; payload={"status":"blind_worker_stage_manifest","created_at_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),"matrix_comparison_staged":False,"files":{str(path.resolve().relative_to(root)):hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}}; pathlib.Path(sys.argv[2]).write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")' \
  "${LOCAL_ROOT}" "${LOCAL_DESIGN}/staged_code_manifest.json" \
  "${CODE_PATHS[@]}" "${SLURM_PATHS[@]}"

ssh_remote() {
  ssh -o ControlMaster=no -o ServerAliveInterval=30 -o ServerAliveCountMax=4 "${SSH_HOST}" "$@"
}

ssh_remote "set -e; if test -e '${REMOTE_RUN_ROOT}/cluster_submission.json'; then echo 'campaign already submitted' >&2; exit 70; fi; mkdir -p '${REMOTE_CODE_ROOT}/plumbing/cluster' '${REMOTE_RUN_ROOT}/design' '${REMOTE_RUN_ROOT}/reused' '${REMOTE_RUN_ROOT}/legacy' '${REMOTE_SHARDS}' '${REMOTE_CACHE}' '${REMOTE_ASSEMBLED}' '${REMOTE_RUN_ROOT}/logs'"
rsync -e "ssh -o ControlMaster=no" -az "${CODE_PATHS[@]}" "${SSH_HOST}:${REMOTE_CODE_ROOT}/plumbing/"
rsync -e "ssh -o ControlMaster=no" -az "${SLURM_PATHS[@]}" "${SSH_HOST}:${REMOTE_CODE_ROOT}/plumbing/cluster/"
rsync -e "ssh -o ControlMaster=no" -az \
  "${LOCAL_DESIGN}/cannon_tasks.csv" \
  "${LOCAL_DESIGN}/design_summary.json" \
  "${LOCAL_DESIGN}/staged_code_manifest.json" \
  "${SSH_HOST}:${REMOTE_RUN_ROOT}/design/"
rsync -e "ssh -o ControlMaster=no" -az "${LOCAL_REUSED_T075}" "${SSH_HOST}:${REMOTE_REUSED_T075}"
rsync -e "ssh -o ControlMaster=no" -az "${LOCAL_LEGACY_SCAN}" "${SSH_HOST}:${REMOTE_LEGACY_SCAN}"

ssh_remote "set -e; cd '${REMOTE_CODE_ROOT}'; test ! -e plumbing/compare_genus1_three_point_matrix_after_freeze.py; PYTHONDONTWRITEBYTECODE=1 '${REMOTE_PYTHON}' -m py_compile plumbing/smoke_genus1_three_point_channel_atlas.py plumbing/genus1_three_point_hdominant_cannon.py; PYTHONDONTWRITEBYTECODE=1 '${REMOTE_PYTHON}' plumbing/genus1_three_point_hdominant_cannon.py prepare --design-dir '${REMOTE_RUN_ROOT}/design' --reused-t075 '${REMOTE_REUSED_T075}'"

PARTITION=${STRINGMC_CLUSTER_PARTITION:-yin}
ARRAY_CAP=${G1_THREE_ARRAY_CAP:-9}
CPUS_PER_TASK=${G1_THREE_CPUS_PER_TASK:-4}
MEMORY=${G1_THREE_MEMORY:-24G}
WORKER_TIME=${G1_THREE_WORKER_TIME:-12:00:00}
COMMON_EXPORTS="ALL,STRINGMC_ROOT=${REMOTE_CODE_ROOT},STRINGMC_PYTHON=${REMOTE_PYTHON},G1_THREE_RUN_ROOT=${REMOTE_RUN_ROOT},G1_THREE_TASKS=${REMOTE_TASKS},G1_THREE_REUSED_T075=${REMOTE_REUSED_T075},G1_THREE_LEGACY_SCAN=${REMOTE_LEGACY_SCAN},G1_THREE_SHARDS=${REMOTE_SHARDS},G1_THREE_CACHE=${REMOTE_CACHE},G1_THREE_ASSEMBLED=${REMOTE_ASSEMBLED}"
LAUNCHED_AT_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)
PREFLIGHT_QUEUE=$(ssh_remote "squeue -h -p '${PARTITION}' -o '%T' | sort | uniq -c; sinfo -h -p '${PARTITION}' -o '%D|%a|%C|%l'")

ARRAY_JOB=$(ssh_remote "cd '${REMOTE_RUN_ROOT}'; sbatch --parsable --partition='${PARTITION}' --array=0-${LAST_TASK}%${ARRAY_CAP} --cpus-per-task='${CPUS_PER_TASK}' --mem='${MEMORY}' --time='${WORKER_TIME}' --requeue --export='${COMMON_EXPORTS}' '${REMOTE_CODE_ROOT}/plumbing/cluster/genus1_three_point_hdominant_array.slurm'")
ASSEMBLY_JOB=$(ssh_remote "cd '${REMOTE_RUN_ROOT}'; sbatch --parsable --partition='serial_requeue' --dependency=afterok:${ARRAY_JOB} --kill-on-invalid-dep=yes --export='${COMMON_EXPORTS}' '${REMOTE_CODE_ROOT}/plumbing/cluster/genus1_three_point_hdominant_assemble.slurm'")

sleep 5
INITIAL_ARRAY_STATE=$(ssh_remote "squeue -h -j '${ARRAY_JOB}' -o '%T|%R' | sort | uniq -c")
"${LOCAL_PYTHON}" -c 'import hashlib,json,pathlib,sys; design=pathlib.Path(sys.argv[1]); stage=pathlib.Path(sys.argv[2]); payload={"status":"submitted_blind_worldsheet_campaign","ssh_host":sys.argv[3],"remote_run_root":sys.argv[4],"remote_python":sys.argv[5],"launched_at_utc":sys.argv[6],"task_count":int(sys.argv[7]),"reused_t":0.75,"array_cap":int(sys.argv[8]),"partition":sys.argv[9],"cpus_per_task":int(sys.argv[10]),"memory_per_task":sys.argv[11],"worker_wall_time":sys.argv[12],"array_job_id":sys.argv[13],"assembly_job_id":sys.argv[14],"initial_array_state":sys.argv[15],"preflight_queue_snapshot":sys.argv[16],"design_sha256":hashlib.sha256(design.read_bytes()).hexdigest(),"staged_code_manifest_sha256":hashlib.sha256(stage.read_bytes()).hexdigest(),"worldsheet_workers_receive_target_formula":False,"comparison_code_staged_with_workers":False,"comparison_submitted":False,"comparison_allowed_only_after_freeze":True}; pathlib.Path(sys.argv[17]).write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")' \
  "${LOCAL_DESIGN}/design_summary.json" "${LOCAL_DESIGN}/staged_code_manifest.json" \
  "${SSH_HOST}" "${REMOTE_RUN_ROOT}" "${REMOTE_PYTHON}" "${LAUNCHED_AT_UTC}" \
  "${TASK_COUNT}" "${ARRAY_CAP}" "${PARTITION}" "${CPUS_PER_TASK}" "${MEMORY}" "${WORKER_TIME}" \
  "${ARRAY_JOB}" "${ASSEMBLY_JOB}" "${INITIAL_ARRAY_STATE}" "${PREFLIGHT_QUEUE}" "${LOCAL_SUBMISSION_JSON}"
rsync -e "ssh -o ControlMaster=no" -az "${LOCAL_SUBMISSION_JSON}" "${SSH_HOST}:${REMOTE_RUN_ROOT}/cluster_submission.json"

echo "array_job_id=${ARRAY_JOB}"
echo "assembly_job_id=${ASSEMBLY_JOB}"
echo "initial_array_state=${INITIAL_ARRAY_STATE}"
echo "submission_record=${LOCAL_SUBMISSION_JSON}"
