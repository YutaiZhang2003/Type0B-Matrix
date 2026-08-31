#!/bin/bash
#SBATCH --job-name=t0b-complex-compare
#SBATCH --partition=serial_requeue
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:05:00

set -euo pipefail
if [ "$#" -ne 5 ]; then
  echo "usage: sbatch script PYTHON COMPARATOR CONFIG SUMMARY OUTPUT" >&2
  exit 2
fi

# Submit afterok the worldsheet reducer. This reads its immutable inputs only.
"$1" -B "$2" --config "$3" --summary "$4" --output "$5"
