#!/bin/bash
#SBATCH --account=yin_lab
#SBATCH --partition=yin
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="${PWD}/Codes"
export SO7E8_PRECISION_DIGITS=60
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
module load python/3.13.12-fasrc01
source activate /n/holylabs/yin_lab/Lab/schristian/conda/envs/ribbon_graph_py313
python -u Codes/run_precision_campaign.py "$1" --task "${SLURM_ARRAY_TASK_ID:-0}"
