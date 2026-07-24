"""Orchestrate WiLoR versus RF-DETR evaluation across all reviewed splits."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hand_benchmark.audit_reports import (
    render_complete_model_review,
    render_error_audit,
    write_complete_review_index,
    write_metric_reports,
    write_threshold_reports,
)
from hand_benchmark.benchmark_predictions import LEFT_HAND, RIGHT_HAND
from hand_benchmark.coco_dataset import SPLIT_NAMES, load_coco_split
from hand_benchmark.evaluation import (
    MetricRow,
    ThresholdSelection,
    evaluate_split,
    ground_truth_boxes,
    load_predictions_for_split,
    select_thresholds,
)

WILOR_MODEL_NAME = "wilor-yolo"
RFDETR_MODEL_NAME = "rfdetr-checkpoint-best-total"
MODEL_NAMES = (WILOR_MODEL_NAME, RFDETR_MODEL_NAME)


@dataclass(frozen=True)
class ComparisonResult:
    """Paths and counts produced by a complete two-model diagnostic audit."""

    run_root: Path
    metric_row_count: int
    error_row_count: int
    selected_f2_thresholds: dict[str, float]


def run_model_comparison(
    *,
    dataset_root: Path,
    run_root: Path,
    iou_threshold: float = 0.5,
    render_overlays: bool = True,
) -> ComparisonResult:
    """Select validation thresholds and compare both models on every split."""
    if not 0 <= iou_threshold <= 1:
        raise ValueError("--iou-threshold must be between 0 and 1")

    splits = {name: load_coco_split(dataset_root, name) for name in SPLIT_NAMES}
    selections: dict[str, ThresholdSelection] = {}
    curves: dict[tuple[str, str, str], list[dict[str, float | int]]] = {}
    for model_name in MODEL_NAMES:
        for split_name, split in splits.items():
            predictions_path = _prediction_path(run_root, model_name, split_name)
            loaded_model_name, predictions, _ = load_predictions_for_split(
                split, predictions_path
            )
            if loaded_model_name != model_name:
                raise ValueError(
                    f"Prediction model mismatch: expected {model_name}, "
                    f"found {loaded_model_name}"
                )
            ground_truths = ground_truth_boxes(split)
            for category in ("overall_micro", LEFT_HAND, RIGHT_HAND):
                selected_ground_truths = (
                    ground_truths
                    if category == "overall_micro"
                    else [box for box in ground_truths if box.category == category]
                )
                selected_predictions = (
                    predictions
                    if category == "overall_micro"
                    else [box for box in predictions if box.category == category]
                )
                curve = select_thresholds(
                    model_name=model_name,
                    ground_truths=selected_ground_truths,
                    predictions=selected_predictions,
                    iou_threshold=iou_threshold,
                )
                curves[(model_name, split_name, category)] = curve.rows
                if split_name == "valid" and category == "overall_micro":
                    selections[model_name] = curve

    selected_thresholds = {
        model_name: selection.selected_f2_threshold
        for model_name, selection in selections.items()
    }
    metric_rows: list[MetricRow] = []
    classifications = {}
    for model_name in MODEL_NAMES:
        threshold = selected_thresholds[model_name]
        for split_name, split in splits.items():
            rows, classification, _ = evaluate_split(
                split=split,
                predictions_path=_prediction_path(run_root, model_name, split_name),
                confidence_threshold=threshold,
                iou_threshold=iou_threshold,
            )
            metric_rows.extend(rows)
            classifications[(model_name, split_name)] = classification

    write_threshold_reports(run_root / "thresholds", selections, curves)
    write_metric_reports(
        run_root / "metrics",
        metric_rows,
        baseline_model=WILOR_MODEL_NAME,
        candidate_model=RFDETR_MODEL_NAME,
    )

    all_error_rows: list[dict[str, Any]] = []
    if render_overlays:
        for split_name, split in splits.items():
            all_error_rows.extend(
                render_error_audit(
                    split=split,
                    classifications={
                        model_name: classifications[(model_name, split_name)]
                        for model_name in MODEL_NAMES
                    },
                    output_dir=run_root / "errors",
                    thresholds=selected_thresholds,
                )
            )
            render_complete_model_review(
                split=split,
                model_name=RFDETR_MODEL_NAME,
                classification=classifications[(RFDETR_MODEL_NAME, split_name)],
                output_dir=run_root / "review",
                threshold=selected_thresholds[RFDETR_MODEL_NAME],
            )
        _write_error_manifest(run_root / "errors", all_error_rows)
        write_complete_review_index(run_root / "review", RFDETR_MODEL_NAME, SPLIT_NAMES)

    run_payload = {
        "dataset_root": dataset_root.resolve().as_posix(),
        "dataset_role": "development-audit",
        "iou_threshold": iou_threshold,
        "models": list(MODEL_NAMES),
        "selected_thresholds": {
            model_name: {
                "f1": selections[model_name].selected_f1_threshold,
                "f2": selections[model_name].selected_f2_threshold,
            }
            for model_name in MODEL_NAMES
        },
        "split_image_counts": {
            split_name: len(split.images) for split_name, split in splits.items()
        },
        "annotation_policy": (
            "Reports are diagnostic; annotation corrections happen manually "
            "in Roboflow."
        ),
    }
    (run_root / "comparison.json").write_text(
        json.dumps(run_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return ComparisonResult(
        run_root=run_root,
        metric_row_count=len(metric_rows),
        error_row_count=len(all_error_rows),
        selected_f2_thresholds=selected_thresholds,
    )


def _prediction_path(run_root: Path, model_name: str, split_name: str) -> Path:
    return run_root / "predictions" / model_name / f"{split_name}.jsonl"


def _write_error_manifest(errors_root: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    errors_root.mkdir(parents=True, exist_ok=True)
    (errors_root / "error_manifest.json").write_text(
        json.dumps(rows, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (errors_root / "error_manifest.csv").open(
        "w", encoding="utf-8", newline=""
    ) as output:
        fieldnames = list(dict.fromkeys(key for row in rows for key in row))
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
