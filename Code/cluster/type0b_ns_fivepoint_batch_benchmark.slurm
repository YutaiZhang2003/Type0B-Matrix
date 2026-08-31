#!/bin/bash
#SBATCH --job-name=t0b-batch-audit
#SBATCH --partition=yin
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=02:00:00
#SBATCH --output=logs/batch-audit-%j.out
#SBATCH --error=logs/batch-audit-%j.err
set -euo pipefail
: "${TYPE0B_5PT_ROOT:?set staged source root}"
: "${TYPE0B_5PT_PYTHON:?set Python executable}"
cd "${TYPE0B_5PT_ROOT}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="${TYPE0B_5PT_ROOT}/Code/higher_point_amplitude_attempts/type0b_ns_five_tachyon:${TYPE0B_5PT_ROOT}/Code/c_Recursion"
"${TYPE0B_5PT_PYTHON}" -B Code/higher_point_amplitude_attempts/type0b_ns_five_tachyon/benchmark_batched_fivepoint.py \
  --production-momentum --store "${TYPE0B_5PT_ROOT}/../audit/coefficients.sqlite" \
  --output "${TYPE0B_5PT_ROOT}/../audit/batch_benchmark.json"
