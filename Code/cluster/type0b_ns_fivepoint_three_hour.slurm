#!/bin/bash
#SBATCH --job-name=t0b-5pt-3h
#SBATCH --partition=yin
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=03:00:00
#SBATCH --no-requeue
#SBATCH --signal=B:USR1@120
set -euo pipefail
: "${TYPE0B_5PT_ROOT:?set staged source root}"
: "${TYPE0B_5PT_PYTHON:?set Python executable}"
: "${TYPE0B_5PT_RUN_ROOT:?set run output root}"
cd "${TYPE0B_5PT_ROOT}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1 TYPE0B_5PT_PROGRESS=1 TYPE0B_5PT_PROGRESS_EVERY=16
export PYTHONPATH="${TYPE0B_5PT_ROOT}/Code/higher_point_amplitude_attempts/type0b_ns_five_tachyon:${TYPE0B_5PT_ROOT}/Code/c_Recursion:${TYPE0B_5PT_ROOT}/Code/bosonic_c1_one_to_n_reference/reference_implementation/plumbing"
exec "${TYPE0B_5PT_PYTHON}" -B Code/higher_point_amplitude_attempts/type0b_ns_five_tachyon/run_three_hour_campaign.py \
  --manifest Code/config/type0b_ns_five_tachyon_three_hour_campaign.json \
  --run-root "${TYPE0B_5PT_RUN_ROOT}"
