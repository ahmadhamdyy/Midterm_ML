#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${SCRATCH:?SCRATCH is required}/midterm_mtl}"
cd "$PROJECT_ROOT"

if sinfo -h -p gpu -t idle -o '%G' | awk '/h100/{found=1} END{exit !found}'; then
  ACCELERATOR=h100
else
  ACCELERATOR=pvc
fi
SCRIPT="slurm/train_${ACCELERATOR}.sbatch"
echo "Submitting to $ACCELERATOR via $SCRIPT"

sbatch --export=ALL,TASK=single,RUN_NAME=single,AUX_WEIGHT=0 "$SCRIPT"
for weight in 0.05 0.10 0.20; do
  run_name="multi_${weight/./p}"
  sbatch --export=ALL,TASK=multi,RUN_NAME="$run_name",AUX_WEIGHT="$weight" "$SCRIPT"
done
