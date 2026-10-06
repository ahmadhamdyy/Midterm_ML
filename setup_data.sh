#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${SCRATCH:?SCRATCH is required}/midterm_mtl}"
DATA_DIR="${DATA_DIR:-$PROJECT_ROOT/data}"
ARCHIVE="$DATA_DIR/faces_aligned_small_mirrored_co_aligned_cropped_cleaned.tar.gz"
FILE_ID="1LzSjp1NrXUoN6CQccf5WMGrm9m34tkwo"

module purge
module load GCC/13.3.0 OpenMPI/5.0.3 PyTorch/2.9.1-CUDA-12.6.0

mkdir -p "$DATA_DIR"
if [[ ! -s "$ARCHIVE" ]]; then
  "$PROJECT_ROOT/.venv-h100/bin/python" -m gdown "$FILE_ID" --output "$ARCHIVE"
fi

if [[ ! -f "$DATA_DIR/.extracted" ]]; then
  "$PROJECT_ROOT/.venv-h100/bin/python" "$PROJECT_ROOT/extract_selected.py" \
    "$ARCHIVE" "$DATA_DIR"
  touch "$DATA_DIR/.extracted"
fi

"$PROJECT_ROOT/.venv-h100/bin/python" "$PROJECT_ROOT/portrait_mtl.py" audit \
  --data-root "$DATA_DIR" \
  --output "$PROJECT_ROOT/artifacts/data_audit.json"
