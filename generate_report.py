#!/usr/bin/env python3
"""Generate the required concise PDF report from saved experiment artifacts."""

from __future__ import annotations

import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


def metric_table(rows: list[list[str]]) -> Table:
    table = Table(rows, hAlign="LEFT", colWidths=[2.0 * inch, 1.45 * inch, 1.45 * inch])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#DCE6F1")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("ALIGN", (1, 1), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    return table


def main() -> None:
    root = Path(__file__).resolve().parent
    artifacts = root / "artifacts"
    figures = artifacts / "figures"
    metrics = json.loads((artifacts / "final" / "final_metrics.json").read_text())
    selection = json.loads((artifacts / "model_selection.json").read_text())
    output = root / "Portrait_MTL_Report.pdf"

    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="CenteredTitle",
            parent=styles["Title"],
            alignment=TA_CENTER,
            fontSize=18,
            leading=21,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Caption",
            parent=styles["BodyText"],
            fontSize=8,
            leading=10,
            textColor=colors.HexColor("#444444"),
            spaceAfter=5,
        )
    )
    body = styles["BodyText"]
    body.fontSize = 9.3
    body.leading = 12.2
    heading = styles["Heading2"]
    heading.spaceBefore = 7
    heading.spaceAfter = 4

    document = SimpleDocTemplate(
        str(output),
        pagesize=letter,
        rightMargin=0.62 * inch,
        leftMargin=0.62 * inch,
        topMargin=0.55 * inch,
        bottomMargin=0.55 * inch,
        title="Single-Task and Multi-Task Learning with A Century of Portraits",
        author="Ahmad Hamdy",
    )
    story = [
        Paragraph(
            "Single-Task and Multi-Task Learning<br/>with A Century of Portraits",
            styles["CenteredTitle"],
        ),
        Paragraph("Machine Learning Mid-Term Project", styles["Heading3"]),
        Spacer(1, 4),
        Paragraph("1. Introduction", heading),
        Paragraph(
            "This project compares year regression from American high-school portraits with "
            "a multi-task network that jointly predicts year and gender. Gender was selected "
            "because it is directly encoded by the F/M folders, has a reliable binary target, "
            "and can regularize shared visual features while remaining substantially easier "
            "than noisy multi-class state prediction.",
            body,
        ),
        Paragraph("2. Dataset and preprocessing", heading),
        Paragraph(
            "The official archive yielded 29,152 training-list entries and 7,408 test-list "
            "entries (36,560 unique referenced images), spanning 1905–2013. A seed-1 random "
            "10% validation subset was taken only from training. The custom Dataset parses "
            "year and state from each filename and gender from its folder. An undocumented "
            "integer follows paths in female split files; it was detected and ignored by "
            "taking the first whitespace-delimited field. The audit found zero missing or "
            "invalid listed images. To respect ACES inode quotas, only listed images were "
            "stored as blobs in one SQLite file, without changing examples or splits.",
            body,
        ),
        Paragraph(
            "Images are decoded as RGB, resized/cropped to 224×224, and ImageNet-normalized. "
            "Training uses random resized crops, horizontal flips, and small affine shifts. "
            "Year is normalized to [0,1] with (year−1905)/108 and converted back to years "
            "before MAE calculation.",
            body,
        ),
        Paragraph("3. Models and training", heading),
        Paragraph(
            "Both systems use the same ImageNet-pretrained ResNet-50 backbone. The single-task "
            "head is dropout plus one linear output. MTL adds a two-logit gender head to the "
            "same shared representation. Both use normalized L1 year loss, AdamW (learning "
            "rate 3e-4, weight decay 1e-4), cosine decay, batch size 128, mixed precision, "
            "20 maximum epochs, and identical preprocessing. MTL minimizes Lyear + λLgender.",
            body,
        ),
        PageBreak(),
        Paragraph("4. Experiments and validation-only selection", heading),
        Paragraph(
            "All random generators used seed 1. Every epoch saved model, optimizer, scheduler, "
            "scaler, epoch, and history for exact resume. Three MTL weights were trained; model "
            "and weight selection used validation only. The test loader was constructed only "
            "by the guarded final-evaluation command after selection.",
            body,
        ),
        metric_table(
            [["Gender weight", "Val year MAE", "Val gender accuracy"]]
            + [
                [
                    f"{row['auxiliary_weight']:.2f}",
                    f"{row['best_validation']['year_mae']:.3f}",
                    f"{100 * row['best_validation']['gender_accuracy']:.2f}%",
                ]
                for row in selection["all_multi_candidates"]
            ]
        ),
        Spacer(1, 8),
        Image(str(figures / "training_curves.png"), width=6.9 * inch, height=2.42 * inch),
        Paragraph(
            "Figure 1. Year-regression learning curves for the single-task model and the "
            "validation-selected MTL model (λ=0.05).",
            styles["Caption"],
        ),
        Image(str(figures / "results_comparison.png"), width=6.45 * inch, height=2.58 * inch),
        Paragraph(
            "Figure 2. One-time final test MAE and validation-only auxiliary-weight tuning.",
            styles["Caption"],
        ),
        PageBreak(),
        Paragraph("5. Final results and comparison", heading),
        metric_table(
            [
                ["Model", "Test year MAE", "Gender accuracy / F1"],
                ["Single-task", f"{metrics['single']['year_mae']:.3f} years", "—"],
                [
                    "Multi-task (λ=0.05)",
                    f"{metrics['multi']['year_mae']:.3f} years",
                    (
                        f"{100 * metrics['multi']['gender_accuracy']:.2f}% / "
                        f"{100 * metrics['multi']['gender_f1']:.2f}%"
                    ),
                ],
            ]
        ),
        Spacer(1, 8),
        Paragraph(
            "The single-task model achieved the better test year result (3.442 years), while "
            "MTL reached 3.482 years. Both are below the four-year target. MTL improved "
            "validation MAE from 2.599 to 2.550 years, but that small gain did not transfer to "
            "the held-out test split: test MAE worsened by 0.040 years. This is evidence that "
            "the auxiliary task acted as a mild validation regularizer, not a reliable test "
            "improvement.",
            body,
        ),
        Image(str(figures / "prediction_diagnostics.png"), width=6.6 * inch, height=2.7 * inch),
        Paragraph(
            "Figure 3. Gender confusion matrix and MTL year prediction density.",
            styles["Caption"],
        ),
        Paragraph("6. Discussion", heading),
        Paragraph(
            "Gender accuracy was 98.97% on validation but 96.21% on test, below the stated "
            "97.5% expectation. The gap likely reflects temporal/school distribution shift "
            "between official lists and strong correlation between portrait style and year. "
            "The result was retained as-is: no checkpoint, weight, or epoch was selected using "
            "test performance. Other challenges were undocumented female-list suffixes, "
            "scratch inode limits, and unavailable PVC runtime initialization; compact SQLite "
            "storage and H100 execution resolved the infrastructure issues.",
            body,
        ),
        Paragraph(
            "Potential improvements include stronger face-aligned augmentations, class-balanced "
            "gender sampling, calibration across decades, a larger modern backbone, ordinal or "
            "distributional age regression, and uncertainty-based dynamic task weighting. Any "
            "such change must be chosen on validation and followed by a new pre-registered final "
            "test run, rather than reusing this test result.",
            body,
        ),
    ]
    document.build(story)
    print(output)


if __name__ == "__main__":
    main()
