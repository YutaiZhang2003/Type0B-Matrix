#!/bin/bash
#SBATCH --job-name=t0b-5pt-quick
#SBATCH --partition=yin
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#SBATCH --signal=B:USR1@120

set -euo pipefail
: "${TYPE0B_5PT_ROOT:?set staged source root}"
: "${TYPE0B_5PT_PYTHON:?set Python executable}"
: "${TYPE0B_5PT_CONFIG:?set config}"
: "${TYPE0B_5PT_SHARDS:?set shard directory}"
: "${TYPE0B_5PT_SUMMARY:?set summary path}"
: "${TYPE0B_5PT_COMPARISON:?set comparison path}"
cd "${TYPE0B_5PT_ROOT}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1 TYPE0B_5PT_PROGRESS=1
export TYPE0B_5PT_LOCAL_CACHE_ROOT
TYPE0B_5PT_LOCAL_CACHE_ROOT=$("${TYPE0B_5PT_PYTHON}" -c 'import os,tempfile; print(tempfile.mkdtemp(prefix="type0b-cache-",dir=os.environ.get("SLURM_TMPDIR") or os.environ.get("TMPDIR") or "/tmp"))')
export PYTHONPATH="${TYPE0B_5PT_ROOT}/Code/higher_point_amplitude_attempts/type0b_ns_five_tachyon:${TYPE0B_5PT_ROOT}/Code/c_Recursion:${TYPE0B_5PT_ROOT}/Code/bosonic_c1_one_to_n_reference/reference_implementation/plumbing"
driver=Code/higher_point_amplitude_attempts/type0b_ns_five_tachyon/run_type0b_ns_five_tachyon_cluster.py
count=$("${TYPE0B_5PT_PYTHON}" -B "$driver" --config "${TYPE0B_5PT_CONFIG}" plan --task-count-only)
if (( count != 4 || ${SLURM_CPUS_PER_TASK:-0} < count )); then
  echo "This profile requires four allocated CPUs and four workers." >&2
  exit 2
fi
mkdir -p "${TYPE0B_5PT_SHARDS}"
pids=()
# Give Python workers two minutes to preserve committed coefficient tables.
stop_workers() {
  for pid in "${pids[@]}"; do kill -TERM "$pid" 2>/dev/null || true; done
  for pid in "${pids[@]}"; do wait "$pid" 2>/dev/null || true; done
  exit 124
}
trap stop_workers USR1 TERM
for ((index=0; index<count; index++)); do
  "${TYPE0B_5PT_PYTHON}" -B "$driver" --config "${TYPE0B_5PT_CONFIG}" worker \
    --output-dir "${TYPE0B_5PT_SHARDS}" --task-index "$index" \
    >"${TYPE0B_5PT_SHARDS}/worker_${index}.log" 2>&1 &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do
  wait "$pid" || failed=1
done
pids=()
if (( failed )); then
  echo "A worker failed; preserving logs/checkpoints without fabricating a complete result." >&2
  exit 1
fi
"${TYPE0B_5PT_PYTHON}" -B "$driver" --config "${TYPE0B_5PT_CONFIG}" reduce \
  --output-dir "${TYPE0B_5PT_SHARDS}" --summary "${TYPE0B_5PT_SUMMARY}"
"${TYPE0B_5PT_PYTHON}" -B Code/higher_point_amplitude_attempts/complex_frequency_comparison/compare.py \
  --config "${TYPE0B_5PT_CONFIG}" --summary "${TYPE0B_5PT_SUMMARY}" \
  --output "${TYPE0B_5PT_COMPARISON}"
