#!/usr/bin/env python3
"""Select final checkpoints using validation summaries only."""

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.runs_root)
    single = json.loads((root / "single" / "summary.json").read_text())
    multi_candidates = []
    for summary_path in sorted(root.glob("multi_*/summary.json")):
        summary = json.loads(summary_path.read_text())
        summary["run_dir"] = str(summary_path.parent.resolve())
        multi_candidates.append(summary)
    if not multi_candidates:
        raise RuntimeError("no completed multi-task summaries found")
    selected_multi = min(
        multi_candidates,
        key=lambda item: (
            item["best_validation"]["year_mae"],
            -item["best_validation"]["gender_accuracy"],
        ),
    )
    selection = {
        "selection_basis": "minimum validation year MAE; gender accuracy is tie-breaker",
        "single_checkpoint": single["checkpoint"],
        "multi_checkpoint": selected_multi["checkpoint"],
        "selected_auxiliary_weight": selected_multi["auxiliary_weight"],
        "single_validation": single["best_validation"],
        "multi_validation": selected_multi["best_validation"],
        "all_multi_candidates": [
            {
                "auxiliary_weight": item["auxiliary_weight"],
                "best_epoch": item["best_epoch"],
                "best_validation": item["best_validation"],
            }
            for item in multi_candidates
        ],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(selection, indent=2) + "\n")
    print(json.dumps(selection, indent=2))


if __name__ == "__main__":
    main()
