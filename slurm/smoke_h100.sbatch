#!/usr/bin/env bash
#SBATCH --job-name=portrait-smoke-h100
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=/scratch/user/u.ae271721/midterm_mtl/logs/h100-smoke-%j.out

set -euo pipefail
cd "$SCRATCH/midterm_mtl"
module purge
module load GCC/13.3.0 OpenMPI/5.0.3 PyTorch/2.9.1-CUDA-12.6.0
exec .venv-h100/bin/python portrait_mtl.py smoke --device cuda
