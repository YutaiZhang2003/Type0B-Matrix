#!/bin/bash
#SBATCH --job-name=g2-period-table
#SBATCH --partition=yin
#SBATCH --array=0-959%192
#SBATCH --cpus-per-task=1
#SBATCH --mem=3G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --output=logs/g2-period-%A_%a.out
#SBATCH --error=logs/g2-period-%A_%a.err

set -euo pipefail

# Create the logs directory before calling sbatch. Set these four values in the
# submission shell. STRINGMC_PYTHON should be the absolute path to Python in a
# tested Mamba environment. The run root, manifest, and config snapshot must
# remain on shared storage and immutable throughout the array job.
: "${STRINGMC_ROOT:?set STRINGMC_ROOT to the repository checkout}"
: "${STRINGMC_PYTHON:?set STRINGMC_PYTHON to the cluster environment Python}"
: "${PERIOD_TABLE_MANIFEST:?set PERIOD_TABLE_MANIFEST to manifest.csv}"
: "${PERIOD_TABLE_SHARDS:?set PERIOD_TABLE_SHARDS to a new shard directory}"

cd "${STRINGMC_ROOT}"

# Keep NumPy's linear algebra inside the single CPU allocated to this task.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

"${STRINGMC_PYTHON}" plumbing/genus2_period_table_cluster.py \
  --config plumbing/config/genus2_period_table_cluster.json \
  worker \
  --manifest "${PERIOD_TABLE_MANIFEST}" \
  --output-dir "${PERIOD_TABLE_SHARDS}" \
  --shard-id "${SLURM_ARRAY_TASK_ID}" \
  --execute
