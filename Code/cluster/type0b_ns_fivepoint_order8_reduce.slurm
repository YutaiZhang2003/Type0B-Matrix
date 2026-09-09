#!/bin/bash
#SBATCH --job-name=t0b-5pt-o8-reduce
#SBATCH --partition=serial_requeue
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:30:00
#SBATCH --output=logs/t0b-5pt-o8-reduce-%j.out
#SBATCH --error=logs/t0b-5pt-o8-reduce-%j.err

set -euo pipefail

: "${TYPE0B_5PT_ROOT:?set TYPE0B_5PT_ROOT}"
: "${TYPE0B_5PT_PYTHON:?set TYPE0B_5PT_PYTHON}"
: "${TYPE0B_5PT_CONFIG:?set TYPE0B_5PT_CONFIG}"
: "${TYPE0B_5PT_SHARDS:?set TYPE0B_5PT_SHARDS}"
: "${TYPE0B_5PT_SUMMARY:?set TYPE0B_5PT_SUMMARY}"

cd "${TYPE0B_5PT_ROOT}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONPATH="${TYPE0B_5PT_ROOT}/Code/higher_point_amplitude_attempts/type0b_ns_five_tachyon:${TYPE0B_5PT_ROOT}/Code/c_Recursion:${TYPE0B_5PT_ROOT}/Code/bosonic_c1_one_to_n_reference/reference_implementation/plumbing${PYTHONPATH:+:${PYTHONPATH}}"

"${TYPE0B_5PT_PYTHON}" \
  Code/higher_point_amplitude_attempts/type0b_ns_five_tachyon/run_type0b_ns_five_tachyon_cluster.py \
  --config "${TYPE0B_5PT_CONFIG}" \
  reduce \
  --output-dir "${TYPE0B_5PT_SHARDS}" \
  --summary "${TYPE0B_5PT_SUMMARY}"
