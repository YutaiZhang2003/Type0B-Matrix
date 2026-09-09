#!/bin/bash
#SBATCH --job-name=nsrr-trial-L5
#SBATCH --partition=yin
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=logs/nsrr-trial-L5-%j.out
#SBATCH --error=logs/nsrr-trial-L5-%j.err

set -euo pipefail
: "${NSRR_TRIAL_ROOT:?set the frozen bundle root}"
: "${NSRR_TRIAL_PYTHON:?set the remote Python executable}"
: "${NSRR_TRIAL_OUTPUT:?set a dedicated output directory}"
cd "${NSRR_TRIAL_ROOT}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export TYPE0B_STRINGMC_ROOT="${NSRR_TRIAL_ROOT}/StringMC"
export PYTHONPATH="${NSRR_TRIAL_ROOT}/vendor:${NSRR_TRIAL_ROOT}/Code:${NSRR_TRIAL_ROOT}/Code/genus_2:${NSRR_TRIAL_ROOT}/Code/c_Recursion:${NSRR_TRIAL_ROOT}/Code/full_ramond_block_runtime:${NSRR_TRIAL_ROOT}/Code/genus_2_cross_channel:${NSRR_TRIAL_ROOT}/Code/ramond_branching_recursion:${NSRR_TRIAL_ROOT}/Code/double_virasoro/nsrr:${NSRR_TRIAL_ROOT}/StringMC"
"${NSRR_TRIAL_PYTHON}" Code/cluster/prepare_nsrr_trial_L5_bundle.py verify --bundle-root "${NSRR_TRIAL_ROOT}"
"${NSRR_TRIAL_PYTHON}" Code/genus_2/nsrr_trial_cluster.py run \
  --config Code/config/nsrr_trial_L5_N4_cluster_20260830.json \
  --output-dir "${NSRR_TRIAL_OUTPUT}" --workers 8
"${NSRR_TRIAL_PYTHON}" Code/genus_2/plot_nsrr_cluster_trial.py --run-dir "${NSRR_TRIAL_OUTPUT}"
