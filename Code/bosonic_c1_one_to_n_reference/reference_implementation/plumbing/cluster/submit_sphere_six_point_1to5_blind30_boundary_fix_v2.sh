#!/bin/bash
# Stage the source-chart boundary fix and submit behind one full production shard.

set -euo pipefail

if [[ $# -gt 4 ]]; then
  echo "usage: $0 [SSH_HOST] [REMOTE_RUN_ROOT] [REMOTE_PYTHON] [LOCAL_SUBMISSION_JSON]" >&2
  exit 2
fi

SSH_HOST=${1:-cannon}
REMOTE_RUN_ROOT=${2:-/n/holylabs/yin_lab/Everyone/yutaizhang/StringMC/sphere_six_point_1to5_blind30_3h_20260824_v2}
REMOTE_PYTHON=${3:-/n/holylabs/yin_lab/Everyone/yutaizhang/StringMC/genus2_period_table_20260718_176ab14d/.venv/bin/python}
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
LOCAL_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
LOCAL_PYTHON=${STRINGMC_LOCAL_PYTHON:-"${LOCAL_ROOT}/.venv/bin/python"}
LOCAL_DESIGN="${LOCAL_ROOT}/plumbing/results/sphere_six_point_1to5/cannon_blind30_3h_v2"
LOCAL_SUBMISSION_JSON=${4:-"${LOCAL_DESIGN}/cluster_submission.json"}
CONFIG="${LOCAL_ROOT}/plumbing/config/sphere_six_point_1to5_cannon_blind30_3h_v1.json"
DRIVER="${LOCAL_ROOT}/plumbing/sphere_six_point_cannon_blind.py"

REMOTE_CODE_ROOT="${REMOTE_RUN_ROOT}/code"
REMOTE_CONFIG="${REMOTE_CODE_ROOT}/plumbing/config/$(basename "${CONFIG}")"
REMOTE_MANIFEST="${REMOTE_RUN_ROOT}/design/manifest.csv"
REMOTE_SHARDS="${REMOTE_RUN_ROOT}/shards"
REMOTE_ASSEMBLED="${REMOTE_RUN_ROOT}/assembled"

test -x "${LOCAL_PYTHON}"
test -f "${CONFIG}"
test -f "${DRIVER}"

"${LOCAL_PYTHON}" "${DRIVER}" prepare \
  --config "${CONFIG}" \
  --design-dir "${LOCAL_DESIGN}" >/dev/null
"${LOCAL_PYTHON}" "${LOCAL_ROOT}/plumbing/sphere_six_point_atlas_checks.py"
"${LOCAL_PYTHON}" "${LOCAL_ROOT}/plumbing/sphere_six_point_cannon_blind30_3h_checks.py"

TASK_COUNT=$("${LOCAL_PYTHON}" -c 'import csv,sys; print(sum(1 for _ in csv.DictReader(open(sys.argv[1]))))' "${LOCAL_DESIGN}/manifest.csv")
LAST_TASK=$((TASK_COUNT - 1))
PARTITION=$("${LOCAL_PYTHON}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["cluster"]["partition"])' "${CONFIG}")
WORKER_TIME=$("${LOCAL_PYTHON}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["cluster"]["worker_wall_time"])' "${CONFIG}")
ASSEMBLY_TIME=$("${LOCAL_PYTHON}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["cluster"]["assembly_wall_time"])' "${CONFIG}")
VALIDATION_TIME=$("${LOCAL_PYTHON}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["cluster"]["validation_wall_time"])' "${CONFIG}")
MEMORY=$("${LOCAL_PYTHON}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["cluster"]["memory_per_worker"])' "${CONFIG}")

if [[ "${TASK_COUNT}" -ne 450 || "${LAST_TASK}" -ne 449 ]]; then
  echo "boundary-fix submission requires the locked 450-task design" >&2
  exit 1
fi

CODE_FILES=(
  ccy_genus2_block.py
  ccy_sphere_five_point.py
  ccy_sphere_six_point.py
  ccy_sphere_six_point_star.py
  genus2_vacuum_blocks.py
  liouville_torus.py
  plumbing_algorithms.py
  sphere_five_point_liouville.py
  sphere_five_point_subtraction.py
  sphere_four_point_subtraction.py
  sphere_six_point_atlas.py
  sphere_six_point_atlas_checks.py
  sphere_six_point_cannon_blind.py
  sphere_six_point_cannon_blind30_3h_checks.py
  sphere_six_point_equal_energy.py
  virasoro_blocks.py
  virasoro_descendant_algebra.py
  virasoro_plumbing_graph.py
)
STAGE_PATHS=()
for name in "${CODE_FILES[@]}"; do
  path="${LOCAL_ROOT}/plumbing/${name}"
  test -f "${path}"
  STAGE_PATHS+=("${path}")
done

SLURM_PATHS=(
  "${LOCAL_ROOT}/plumbing/cluster/sphere_six_point_1to5_blind_array.slurm"
  "${LOCAL_ROOT}/plumbing/cluster/sphere_six_point_1to5_blind_assemble.slurm"
  "${LOCAL_ROOT}/plumbing/cluster/sphere_six_point_1to5_blind_freeze.slurm"
)
"${LOCAL_PYTHON}" -c 'import datetime,hashlib,json,pathlib,sys; root=pathlib.Path(sys.argv[1]); files=[pathlib.Path(value) for value in sys.argv[3:]]; payload={"status":"blind_worker_stage_manifest_boundary_fix_v2","created_at_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),"matrix_comparison_staged":False,"files":{str(path.resolve().relative_to(root.resolve())):hashlib.sha256(path.read_bytes()).hexdigest() for path in files}}; pathlib.Path(sys.argv[2]).write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")' \
  "${LOCAL_ROOT}" "${LOCAL_DESIGN}/staged_code_manifest.json" "${STAGE_PATHS[@]}" "${CONFIG}" "${SLURM_PATHS[@]}"

ssh "${SSH_HOST}" "set -e; if test -e '${REMOTE_RUN_ROOT}/cluster_submission.json'; then echo 'remote boundary-fix campaign was already submitted' >&2; exit 70; fi; mkdir -p '${REMOTE_CODE_ROOT}/plumbing/config' '${REMOTE_CODE_ROOT}/plumbing/cluster' '${REMOTE_RUN_ROOT}/design' '${REMOTE_SHARDS}' '${REMOTE_ASSEMBLED}' '${REMOTE_RUN_ROOT}/logs'"
rsync -az "${STAGE_PATHS[@]}" "${SSH_HOST}:${REMOTE_CODE_ROOT}/plumbing/"
rsync -az "${CONFIG}" "${SSH_HOST}:${REMOTE_CODE_ROOT}/plumbing/config/"
rsync -az "${SLURM_PATHS[@]}" "${SSH_HOST}:${REMOTE_CODE_ROOT}/plumbing/cluster/"
rsync -az \
  "${LOCAL_DESIGN}/manifest.csv" \
  "${LOCAL_DESIGN}/config.snapshot.json" \
  "${LOCAL_DESIGN}/design_summary.json" \
  "${LOCAL_DESIGN}/staged_code_manifest.json" \
  "${SSH_HOST}:${REMOTE_RUN_ROOT}/design/"

ssh "${SSH_HOST}" "set -e; cd '${REMOTE_CODE_ROOT}'; test ! -e plumbing/sphere_six_point_matrix_comparison.py; if grep -R -E 'q5_matrix_model|sphere_six_point_matrix_comparison' plumbing --include='*.py'; then echo 'target comparison code leaked into worker staging' >&2; exit 71; fi; PYTHONDONTWRITEBYTECODE=1 '${REMOTE_PYTHON}' plumbing/sphere_six_point_atlas_checks.py; PYTHONDONTWRITEBYTECODE=1 '${REMOTE_PYTHON}' plumbing/sphere_six_point_cannon_blind30_3h_checks.py"

COMMON_EXPORTS="ALL,STRINGMC_ROOT=${REMOTE_CODE_ROOT},STRINGMC_PYTHON=${REMOTE_PYTHON},SPHERE6_BLIND_CONFIG=${REMOTE_CONFIG},SPHERE6_BLIND_MANIFEST=${REMOTE_MANIFEST},SPHERE6_BLIND_SHARDS=${REMOTE_SHARDS},SPHERE6_BLIND_ASSEMBLED=${REMOTE_ASSEMBLED}"
LAUNCHED_AT_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)
PREFLIGHT_QUEUE=$(ssh "${SSH_HOST}" "squeue -h -p '${PARTITION}' -o '%T' | sort | uniq -c; sinfo -h -p '${PARTITION}' -o '%D|%a|%C|%l'")

# Task zero uses the same 2^15 Sobol sequence that exposed the original bug.
# The remaining 449 workers are held until this full production shard succeeds.
VALIDATION_JOB=$(ssh "${SSH_HOST}" "cd '${REMOTE_RUN_ROOT}'; sbatch --parsable --partition='${PARTITION}' --array=0-0%1 --cpus-per-task=1 --mem='${MEMORY}' --time='${WORKER_TIME}' --requeue --export='${COMMON_EXPORTS}' '${REMOTE_CODE_ROOT}/plumbing/cluster/sphere_six_point_1to5_blind_array.slurm'")
ARRAY_JOB=$(ssh "${SSH_HOST}" "cd '${REMOTE_RUN_ROOT}'; sbatch --parsable --partition='${PARTITION}' --array=1-${LAST_TASK}%449 --dependency=afterok:${VALIDATION_JOB} --kill-on-invalid-dep=yes --cpus-per-task=1 --mem='${MEMORY}' --time='${WORKER_TIME}' --requeue --export='${COMMON_EXPORTS}' '${REMOTE_CODE_ROOT}/plumbing/cluster/sphere_six_point_1to5_blind_array.slurm'")
ASSEMBLY_JOB=$(ssh "${SSH_HOST}" "cd '${REMOTE_RUN_ROOT}'; sbatch --parsable --partition='${PARTITION}' --dependency=afterok:${ARRAY_JOB} --kill-on-invalid-dep=yes --time='${ASSEMBLY_TIME}' --export='${COMMON_EXPORTS}' '${REMOTE_CODE_ROOT}/plumbing/cluster/sphere_six_point_1to5_blind_assemble.slurm'")
FREEZE_JOB=$(ssh "${SSH_HOST}" "cd '${REMOTE_RUN_ROOT}'; sbatch --parsable --partition='${PARTITION}' --dependency=afterok:${ASSEMBLY_JOB} --kill-on-invalid-dep=yes --time='${VALIDATION_TIME}' --export='${COMMON_EXPORTS}' '${REMOTE_CODE_ROOT}/plumbing/cluster/sphere_six_point_1to5_blind_freeze.slurm'")

sleep 10
INITIAL_VALIDATION_STATE=$(ssh "${SSH_HOST}" "squeue -h -j '${VALIDATION_JOB}' -o '%T|%R'")
INITIAL_ARRAY_STATE=$(ssh "${SSH_HOST}" "squeue -h -j '${ARRAY_JOB}' -o '%T|%R' | sort | uniq -c")

"${LOCAL_PYTHON}" -c 'import hashlib,json,pathlib,sys; config=pathlib.Path(sys.argv[1]); manifest=pathlib.Path(sys.argv[2]); stage=pathlib.Path(sys.argv[3]); payload={"status":"submitted_blind_worldsheet_campaign_boundary_fix_v2","ssh_host":sys.argv[4],"remote_run_root":sys.argv[5],"remote_python":sys.argv[6],"launched_at_utc":sys.argv[7],"task_count":450,"full_length_validation_task_id":0,"full_length_validation_job_id":sys.argv[8],"remaining_array_job_id":sys.argv[9],"assembly_job_id":sys.argv[10],"freeze_job_id":sys.argv[11],"initial_validation_state":sys.argv[12],"initial_array_state":sys.argv[13],"preflight_queue_snapshot":sys.argv[14],"worker_wall_time":sys.argv[15],"assembly_wall_time":sys.argv[16],"validation_wall_time":sys.argv[17],"failed_array_job_id":"41514317","failure":"floating-point channel reconstruction produced q=0","fix":"evaluate the unchanged Sobol point in its exact originating convergent plumbing chart","normalization_or_numerical_settings_changed":False,"config_sha256":hashlib.sha256(config.read_bytes()).hexdigest(),"design_manifest_sha256":hashlib.sha256(manifest.read_bytes()).hexdigest(),"staged_code_manifest_sha256":hashlib.sha256(stage.read_bytes()).hexdigest(),"worldsheet_workers_receive_target_formula":False,"comparison_code_staged_with_workers":False,"comparison_submitted":False,"comparison_allowed_only_after_freeze":True}; pathlib.Path(sys.argv[18]).write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")' \
  "${CONFIG}" "${LOCAL_DESIGN}/manifest.csv" "${LOCAL_DESIGN}/staged_code_manifest.json" \
  "${SSH_HOST}" "${REMOTE_RUN_ROOT}" "${REMOTE_PYTHON}" "${LAUNCHED_AT_UTC}" \
  "${VALIDATION_JOB}" "${ARRAY_JOB}" "${ASSEMBLY_JOB}" "${FREEZE_JOB}" \
  "${INITIAL_VALIDATION_STATE}" "${INITIAL_ARRAY_STATE}" "${PREFLIGHT_QUEUE}" \
  "${WORKER_TIME}" "${ASSEMBLY_TIME}" "${VALIDATION_TIME}" "${LOCAL_SUBMISSION_JSON}"
rsync -az "${LOCAL_SUBMISSION_JSON}" "${SSH_HOST}:${REMOTE_RUN_ROOT}/cluster_submission.json"

echo "full_length_validation_job_id=${VALIDATION_JOB}"
echo "remaining_array_job_id=${ARRAY_JOB}"
echo "assembly_job_id=${ASSEMBLY_JOB}"
echo "freeze_job_id=${FREEZE_JOB}"
echo "initial_validation_state=${INITIAL_VALIDATION_STATE}"
echo "initial_array_state=${INITIAL_ARRAY_STATE}"
echo "submission_record=${LOCAL_SUBMISSION_JSON}"
