# Portrait multi-task learning

Reproducible ACES implementation of the assignment in
`Mid-Term_Multi-Task-Learning.pdf`.

## Results

- Single-task test year MAE: **3.442 years**
- Multi-task test year MAE: **3.482 years**
- Multi-task gender accuracy / F1: **96.21% / 95.72%**
- Validation-selected gender-loss weight: **0.05**

The official test split was evaluated once after validation-only selection.

## ACES workflow

```bash
rsync -az ./ aces:$SCRATCH/midterm_mtl/
ssh aces
cd "$SCRATCH/midterm_mtl"
bash setup_aces.sh
bash setup_data.sh
bash submit_training.sh
```

`setup_data.sh` builds a single SQLite image container to avoid scratch inode
limits. H100 and PVC job definitions are under `slurm/`. PVC scripts use the
TAMU-supported Intel/IPEX environment. Each training run checkpoints every
epoch and `--resume` restores model, optimizer, scheduler, scaler, epoch, and
history.

After all training jobs complete:

```bash
python3 select_checkpoints.py \
  --runs-root runs \
  --output artifacts/model_selection.json
sbatch slurm/evaluate_h100.sbatch
```

The final-evaluation command writes a completion marker and refuses to touch the
test split again unless `--force` is explicitly supplied.

## Deliverables

- `portrait_mtl_results.ipynb`: executed checkpoint/results notebook
- `Portrait_MTL_Report.pdf`: 2–4 page report
- `portrait_mtl.py`: dataset, models, training, resume, and evaluation
- `artifacts/`: metrics, predictions, audit, model selection, and figures
- `runs/single/best.pt` and `runs/multi_0p05/best.pt`: selected checkpoints
