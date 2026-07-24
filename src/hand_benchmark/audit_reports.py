"""Comparison tables, SVG charts, colored overlays, and an audit gallery."""

from __future__ import annotations

import csv
import html
import json
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

import cv2

from hand_benchmark.benchmark_predictions import prediction_provenance
from hand_benchmark.coco_dataset import CANONICAL_CATEGORIES, SPLIT_NAMES, CocoSplit
from hand_benchmark.evaluation import (
    MatchClassification,
    MetricRow,
    ThresholdSelection,
    iou_xyxy,
    metric_row_to_dict,
)

GREEN_BGR = (0, 190, 0)
YELLOW_BGR = (0, 215, 255)
RED_BGR = (0, 0, 230)
MAGENTA_BGR = (220, 0, 220)
WHITE_BGR = (255, 255, 255)
HEADER_BGR = (35, 35, 35)
MODEL_COLORS = ("#2563eb", "#dc2626", "#16a34a", "#9333ea")
METRIC_FIELDS = ("precision", "recall", "f1", "f2", "ap50", "ap75", "map_50_95")


def write_threshold_reports(
    output_dir: Path,
    selections: dict[str, ThresholdSelection],
    curves: dict[tuple[str, str, str], list[dict[str, float | int]]],
) -> None:
    """Write selected thresholds, exact sweep tables, and PR/F-score charts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for model_name, selection in selections.items():
        model_dir = output_dir / model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / "selected_thresholds.json").write_text(
            json.dumps(asdict(selection), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        _write_csv(model_dir / "validation_thresholds.csv", selection.rows)

    charts_dir = output_dir.parent / "charts"
    for split_name in SPLIT_NAMES:
        for category in ("overall_micro", *CANONICAL_CATEGORIES):
            series = [
                (
                    model_name,
                    curves[(model_name, split_name, category)],
                    MODEL_COLORS[index % len(MODEL_COLORS)],
                )
                for index, model_name in enumerate(sorted(selections))
            ]
            _write_pr_svg(
                charts_dir / f"precision_recall_{split_name}_{category}.svg",
                title=f"{split_name}: {category} precision vs recall",
                series=series,
            )
    f_series = [
        (
            model_name,
            selections[model_name].rows,
            MODEL_COLORS[index % len(MODEL_COLORS)],
        )
        for index, model_name in enumerate(sorted(selections))
    ]
    _write_fscore_svg(
        charts_dir / "validation_f1_f2_by_threshold.svg",
        "Validation F1/F2 by confidence threshold",
        f_series,
    )


def write_metric_reports(
    output_dir: Path,
    metric_rows: list[MetricRow],
    baseline_model: str,
    candidate_model: str,
) -> None:
    """Write detailed metrics, diagnostic aggregates, and model deltas."""
    output_dir.mkdir(parents=True, exist_ok=True)
    serialized_rows = [metric_row_to_dict(row) for row in metric_rows]
    _write_csv(output_dir / "metrics.csv", serialized_rows)
    (output_dir / "metrics.json").write_text(
        json.dumps(serialized_rows, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "metrics.md").write_text(
        _metrics_markdown(metric_rows), encoding="utf-8"
    )

    deltas = metric_deltas(metric_rows, baseline_model, candidate_model)
    _write_csv(output_dir / "deltas.csv", deltas)
    (output_dir / "deltas.md").write_text(_deltas_markdown(deltas), encoding="utf-8")
    aggregates = diagnostic_aggregates(metric_rows)
    _write_csv(output_dir / "diagnostic_aggregates.csv", aggregates)


def metric_deltas(
    rows: list[MetricRow], baseline_model: str, candidate_model: str
) -> list[dict[str, Any]]:
    """Compute absolute and relative RF-DETR improvement over WiLoR."""
    by_key = {(row.model_name, row.split, row.category): row for row in rows}
    deltas: list[dict[str, Any]] = []
    for split_name in SPLIT_NAMES:
        for category in ("overall_micro", *CANONICAL_CATEGORIES, "macro_classes"):
            baseline = by_key[(baseline_model, split_name, category)]
            candidate = by_key[(candidate_model, split_name, category)]
            row: dict[str, Any] = {
                "split": split_name,
                "category": category,
                "baseline_model": baseline_model,
                "candidate_model": candidate_model,
            }
            for metric in METRIC_FIELDS:
                baseline_value = float(getattr(baseline, metric))
                candidate_value = float(getattr(candidate, metric))
                absolute_delta = candidate_value - baseline_value
                row[f"{metric}_baseline"] = baseline_value
                row[f"{metric}_candidate"] = candidate_value
                row[f"{metric}_absolute_delta"] = round(absolute_delta, 6)
                row[f"{metric}_relative_percent"] = (
                    round(absolute_delta / baseline_value * 100, 3)
                    if baseline_value
                    else None
                )
            deltas.append(row)
    return deltas


def diagnostic_aggregates(rows: list[MetricRow]) -> list[dict[str, Any]]:
    """Report equal-split means and count-pooled F-scores as diagnostic summaries."""
    aggregates: list[dict[str, Any]] = []
    for model_name in sorted({row.model_name for row in rows}):
        for category in ("overall_micro", *CANONICAL_CATEGORIES, "macro_classes"):
            selected = [
                row
                for row in rows
                if row.model_name == model_name and row.category == category
            ]
            equal_split = {
                "aggregation": "equal_split_average",
                "model_name": model_name,
                "category": category,
            }
            for metric in METRIC_FIELDS:
                equal_split[metric] = round(
                    sum(float(getattr(row, metric)) for row in selected)
                    / len(selected),
                    6,
                )
            aggregates.append(equal_split)

            true_positives = sum(row.true_positives for row in selected)
            false_positives = sum(row.false_positives for row in selected)
            false_negatives = sum(row.false_negatives for row in selected)
            precision = (
                true_positives / (true_positives + false_positives)
                if true_positives + false_positives
                else 0.0
            )
            recall = (
                true_positives / (true_positives + false_negatives)
                if true_positives + false_negatives
                else 0.0
            )
            aggregates.append(
                {
                    "aggregation": "pooled_counts",
                    "model_name": model_name,
                    "category": category,
                    "precision": round(precision, 6),
                    "recall": round(recall, 6),
                    "f1": round(_fbeta(precision, recall, 1.0), 6),
                    "f2": round(_fbeta(precision, recall, 2.0), 6),
                    "ap50": None,
                    "ap75": None,
                    "map_50_95": None,
                }
            )
    return aggregates


def render_error_audit(
    *,
    split: CocoSplit,
    classifications: dict[str, MatchClassification],
    output_dir: Path,
    thresholds: dict[str, float],
) -> list[dict[str, Any]]:
    """Render error-only model panels, comparisons, and review manifest rows."""
    split_dir = output_dir / split.name
    _remove_stale_rendered_images(split_dir)
    error_names_by_model = {
        model: _error_file_names(classification)
        for model, classification in classifications.items()
    }
    all_error_names = set().union(*error_names_by_model.values())
    image_by_name = {image.file_name: image for image in split.images}
    rendered_by_model: dict[str, dict[str, Path]] = defaultdict(dict)
    manifest_rows: list[dict[str, Any]] = []

    for model_name, classification in classifications.items():
        grouped = _group_classification(classification)
        for file_name in sorted(all_error_names):
            image = image_by_name[file_name]
            image_bgr = cv2.imread(str(image.path))
            if image_bgr is None:
                raise ValueError(f"Could not read image: {image.path}")
            annotated = image_bgr.copy()
            manifest_rows.extend(
                _draw_errors_and_manifest(
                    annotated,
                    split.name,
                    model_name,
                    file_name,
                    grouped,
                )
            )
            panel = _add_header(
                annotated, f"{model_name} threshold={thresholds[model_name]:.4f}"
            )
            output_path = output_dir / split.name / "models" / model_name / file_name
            _write_image(output_path, panel)
            rendered_by_model[model_name][file_name] = output_path

    comparison_dir = output_dir / split.name / "comparison"
    model_names = sorted(classifications)
    for file_name in sorted(all_error_names):
        panels = [
            cv2.imread(str(rendered_by_model[model_name][file_name]))
            for model_name in model_names
        ]
        if any(panel is None for panel in panels):
            raise ValueError(f"Could not read rendered comparison for {file_name}")
        comparison = cv2.hconcat(panels)
        _write_image(comparison_dir / file_name, comparison)

    priorities = _review_priorities(classifications)
    for row in manifest_rows:
        row["review_priority"] = priorities.get(
            (str(row["file_name"]), str(row["model_name"])), "model_error"
        )
    _write_csv(split_dir / "error_manifest.csv", manifest_rows)
    (split_dir / "error_manifest.json").write_text(
        json.dumps(manifest_rows, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (split_dir / "index.html").write_text(
        _gallery_html(split.name, sorted(all_error_names), manifest_rows),
        encoding="utf-8",
    )
    return manifest_rows


def render_complete_model_review(
    *,
    split: CocoSplit,
    model_name: str,
    classification: MatchClassification,
    output_dir: Path,
    threshold: float,
) -> list[dict[str, Any]]:
    """Render every image and write IoU-aware local review indexes."""
    review_dir = output_dir / model_name / split.name
    images_dir = review_dir / "images"
    _remove_generated_images(images_dir)

    true_positive_matches: dict[str, list[Any]] = defaultdict(list)
    for match in classification.true_positive_matches:
        true_positive_matches[match.prediction.file_name].append(match)
    false_positives = _group_by_file(classification.false_positives)
    false_negatives = _group_by_file(classification.false_negatives)
    class_confusions = _group_by_file(
        classification.class_confusions,
        file_name_getter=lambda item: item.prediction.file_name,
    )

    image_rows: list[dict[str, Any]] = []
    box_rows: list[dict[str, Any]] = []
    for coco_image in sorted(split.images, key=lambda item: item.file_name):
        image_bgr = cv2.imread(str(coco_image.path))
        if image_bgr is None:
            raise ValueError(f"Could not read image: {coco_image.path}")
        file_name = coco_image.file_name
        matches = true_positive_matches[file_name]
        image_false_positives = false_positives[file_name]
        image_false_negatives = false_negatives[file_name]
        image_confusions = class_confusions[file_name]

        for match in matches:
            _draw_thin_box(image_bgr, match.ground_truth.bbox_xyxy, GREEN_BGR)
            _draw_box(
                image_bgr,
                match.prediction.bbox_xyxy,
                (
                    f"TP {match.prediction.category} "
                    f"{match.prediction.confidence:.2f} IoU={match.iou:.3f}"
                ),
                GREEN_BGR,
            )
            box_rows.append(
                _review_box_row(
                    split.name,
                    model_name,
                    file_name,
                    "true_positive",
                    match.prediction.category,
                    match.prediction.confidence,
                    match.iou,
                    match.prediction.bbox_xyxy,
                    match.ground_truth.bbox_xyxy,
                )
            )
        for prediction in image_false_positives:
            _draw_box(
                image_bgr,
                prediction.bbox_xyxy,
                f"FP {prediction.category} {prediction.confidence:.2f}",
                YELLOW_BGR,
            )
            box_rows.append(
                _review_box_row(
                    split.name,
                    model_name,
                    file_name,
                    "false_positive",
                    prediction.category,
                    prediction.confidence,
                    None,
                    prediction.bbox_xyxy,
                    None,
                )
            )
        for ground_truth in image_false_negatives:
            _draw_box(
                image_bgr,
                ground_truth.bbox_xyxy,
                f"FN {ground_truth.category}",
                RED_BGR,
            )
            box_rows.append(
                _review_box_row(
                    split.name,
                    model_name,
                    file_name,
                    "false_negative",
                    ground_truth.category,
                    None,
                    None,
                    None,
                    ground_truth.bbox_xyxy,
                )
            )
        for confusion in image_confusions:
            _draw_box(
                image_bgr,
                confusion.prediction.bbox_xyxy,
                (
                    f"CLASS {confusion.prediction.category} -> "
                    f"{confusion.ground_truth.category} "
                    f"{confusion.prediction.confidence:.2f} IoU={confusion.iou:.3f}"
                ),
                MAGENTA_BGR,
            )
            box_rows.append(
                _review_box_row(
                    split.name,
                    model_name,
                    file_name,
                    "class_confusion",
                    confusion.ground_truth.category,
                    confusion.prediction.confidence,
                    confusion.iou,
                    confusion.prediction.bbox_xyxy,
                    confusion.ground_truth.bbox_xyxy,
                    predicted_category=confusion.prediction.category,
                )
            )

        ious = [match.iou for match in matches]
        recording, timestamp = prediction_provenance(file_name)
        image_rows.append(
            {
                "split": split.name,
                "file_name": file_name,
                "source_recording": recording,
                "timestamp_seconds": timestamp,
                "model_name": model_name,
                "confidence_threshold": round(threshold, 6),
                "true_positive_count": len(matches),
                "false_positive_count": len(image_false_positives),
                "false_negative_count": len(image_false_negatives),
                "class_confusion_count": len(image_confusions),
                "minimum_true_positive_iou": (round(min(ious), 6) if ious else None),
                "mean_true_positive_iou": (
                    round(sum(ious) / len(ious), 6) if ious else None
                ),
                "maximum_false_positive_confidence": (
                    round(
                        max(
                            prediction.confidence
                            for prediction in image_false_positives
                        ),
                        6,
                    )
                    if image_false_positives
                    else None
                ),
            }
        )
        panel = _add_header(
            image_bgr,
            (
                f"{model_name} {split.name} threshold={threshold:.4f} | "
                f"TP={len(matches)} FP={len(image_false_positives)} "
                f"FN={len(image_false_negatives)} "
                f"CLASS={len(image_confusions)}"
            ),
        )
        _write_image(images_dir / file_name, panel)

    _write_csv(review_dir / "image_manifest.csv", image_rows)
    _write_csv(review_dir / "box_manifest.csv", box_rows)
    (review_dir / "image_manifest.json").write_text(
        json.dumps(image_rows, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (review_dir / "box_manifest.json").write_text(
        json.dumps(box_rows, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    page_specs = {
        "index.html": (
            "All images",
            image_rows,
            lambda row: str(row["file_name"]),
        ),
        "false-positives.html": (
            "False positives",
            [row for row in image_rows if row["false_positive_count"]],
            lambda row: (
                -int(row["false_positive_count"]),
                -float(row["maximum_false_positive_confidence"] or 0),
                str(row["file_name"]),
            ),
        ),
        "false-negatives.html": (
            "False negatives",
            [row for row in image_rows if row["false_negative_count"]],
            lambda row: (
                -int(row["false_negative_count"]),
                str(row["file_name"]),
            ),
        ),
        "lowest-iou.html": (
            "Lowest true-positive IoU",
            [row for row in image_rows if row["minimum_true_positive_iou"] is not None],
            lambda row: (
                float(row["minimum_true_positive_iou"]),
                str(row["file_name"]),
            ),
        ),
        "class-confusions.html": (
            "Left/right class confusions",
            [row for row in image_rows if row["class_confusion_count"]],
            lambda row: (
                -int(row["class_confusion_count"]),
                str(row["file_name"]),
            ),
        ),
    }
    for page_name, (title, rows, sort_key) in page_specs.items():
        ordered_rows = sorted(rows, key=sort_key)
        (review_dir / page_name).write_text(
            _complete_review_html(split.name, title, ordered_rows),
            encoding="utf-8",
        )
    return image_rows


def write_complete_review_index(
    output_dir: Path, model_name: str, split_names: tuple[str, ...]
) -> None:
    """Write the root navigation page for complete local model review."""
    links = "".join(
        f'<li><a href="{html.escape(model_name)}/{html.escape(split_name)}/'
        f'index.html">{html.escape(split_name)}</a></li>'
        for split_name in split_names
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "index.html").write_text(
        f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>{html.escape(model_name)} complete review</title></head>
<body><h1>{html.escape(model_name)} complete review</h1>
<p>Every source image is available through its split page.</p>
<ul>{links}</ul></body></html>
""",
        encoding="utf-8",
    )


def _group_by_file(
    values: list[Any],
    file_name_getter: Any = lambda item: item.file_name,
) -> dict[str, list[Any]]:
    grouped: dict[str, list[Any]] = defaultdict(list)
    for value in values:
        grouped[str(file_name_getter(value))].append(value)
    return grouped


def _review_box_row(
    split_name: str,
    model_name: str,
    file_name: str,
    match_type: str,
    category: str,
    confidence: float | None,
    iou: float | None,
    prediction_bbox: list[float] | None,
    ground_truth_bbox: list[float] | None,
    *,
    predicted_category: str | None = None,
) -> dict[str, Any]:
    recording, timestamp = prediction_provenance(file_name)
    return {
        "split": split_name,
        "file_name": file_name,
        "source_recording": recording,
        "timestamp_seconds": timestamp,
        "model_name": model_name,
        "match_type": match_type,
        "category": category,
        "predicted_category": predicted_category or category,
        "confidence": round(confidence, 6) if confidence is not None else None,
        "iou": round(iou, 6) if iou is not None else None,
        "prediction_bbox_xyxy": (
            json.dumps([round(value, 4) for value in prediction_bbox])
            if prediction_bbox is not None
            else None
        ),
        "ground_truth_bbox_xyxy": (
            json.dumps([round(value, 4) for value in ground_truth_bbox])
            if ground_truth_bbox is not None
            else None
        ),
    }


def _complete_review_html(
    split_name: str, title: str, rows: list[dict[str, Any]]
) -> str:
    navigation = (
        '<a href="index.html">all</a> · '
        '<a href="false-positives.html">false positives</a> · '
        '<a href="false-negatives.html">false negatives</a> · '
        '<a href="lowest-iou.html">lowest IoU</a> · '
        '<a href="class-confusions.html">class confusions</a>'
    )
    cards = "".join(
        "<article>"
        f'<img loading="lazy" src="images/{html.escape(str(row["file_name"]))}">'
        f"<h2>{html.escape(str(row['file_name']))}</h2>"
        f"<p>TP {row['true_positive_count']} · FP {row['false_positive_count']} · "
        f"FN {row['false_negative_count']} · class {row['class_confusion_count']} · "
        f"min IoU {row['minimum_true_positive_iou']}</p>"
        "</article>"
        for row in rows
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>{html.escape(split_name)} — {html.escape(title)}</title>
<style>
body{{font-family:system-ui;margin:24px;background:#f1f5f9;color:#0f172a}}
.navigation{{position:sticky;top:0;background:white;padding:12px;z-index:2}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:18px}}
article{{background:white;padding:12px;border-radius:10px;box-shadow:0 1px 4px #94a3b8}}
img{{width:100%;height:auto}} h2{{font-size:13px;overflow-wrap:anywhere}}
</style></head><body>
<div class="navigation"><strong>{html.escape(split_name)} — {html.escape(title)}
({len(rows)})</strong><br>{navigation}<br>
green TP, yellow FP, red FN, magenta wrong class. TP labels include IoU.</div>
<main class="grid">{cards}</main></body></html>
"""


def _remove_stale_rendered_images(split_dir: Path) -> None:
    """Remove only prior generated panels so reruns cannot retain stale errors."""
    for generated_dir in (split_dir / "models", split_dir / "comparison"):
        if not generated_dir.is_dir():
            continue
        _remove_generated_images(generated_dir)


def _remove_generated_images(generated_dir: Path) -> None:
    if not generated_dir.is_dir():
        return
    for path in generated_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            path.unlink()


def _draw_errors_and_manifest(
    image: Any,
    split_name: str,
    model_name: str,
    file_name: str,
    grouped: dict[str, dict[str, list[Any]]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    recording, timestamp = prediction_provenance(file_name)
    for prediction in grouped["true_positive"].get(file_name, []):
        _draw_box(
            image,
            prediction.bbox_xyxy,
            f"TP {prediction.category} {prediction.confidence:.2f}",
            GREEN_BGR,
        )
    for prediction in grouped["false_positive"].get(file_name, []):
        _draw_box(
            image,
            prediction.bbox_xyxy,
            f"FP {prediction.category} {prediction.confidence:.2f}",
            YELLOW_BGR,
        )
        rows.append(
            _manifest_row(
                split_name,
                model_name,
                file_name,
                recording,
                timestamp,
                "false_positive",
                prediction.category,
                prediction.confidence,
                prediction.bbox_xyxy,
            )
        )
    for ground_truth in grouped["false_negative"].get(file_name, []):
        _draw_box(
            image,
            ground_truth.bbox_xyxy,
            f"FN {ground_truth.category}",
            RED_BGR,
        )
        rows.append(
            _manifest_row(
                split_name,
                model_name,
                file_name,
                recording,
                timestamp,
                "false_negative",
                ground_truth.category,
                None,
                ground_truth.bbox_xyxy,
            )
        )
    for confusion in grouped["class_confusion"].get(file_name, []):
        label = (
            f"CLASS {confusion.prediction.category} -> "
            f"{confusion.ground_truth.category} {confusion.prediction.confidence:.2f}"
        )
        _draw_box(image, confusion.prediction.bbox_xyxy, label, MAGENTA_BGR)
        rows.append(
            {
                **_manifest_row(
                    split_name,
                    model_name,
                    file_name,
                    recording,
                    timestamp,
                    "class_confusion",
                    confusion.ground_truth.category,
                    confusion.prediction.confidence,
                    confusion.prediction.bbox_xyxy,
                ),
                "predicted_category": confusion.prediction.category,
                "iou": round(confusion.iou, 6),
            }
        )
    return rows


def _group_classification(
    classification: MatchClassification,
) -> dict[str, dict[str, list[Any]]]:
    grouped: dict[str, dict[str, list[Any]]] = {
        "true_positive": defaultdict(list),
        "false_positive": defaultdict(list),
        "false_negative": defaultdict(list),
        "class_confusion": defaultdict(list),
    }
    for prediction in classification.true_positives:
        grouped["true_positive"][prediction.file_name].append(prediction)
    for prediction in classification.false_positives:
        grouped["false_positive"][prediction.file_name].append(prediction)
    for ground_truth in classification.false_negatives:
        grouped["false_negative"][ground_truth.file_name].append(ground_truth)
    for confusion in classification.class_confusions:
        grouped["class_confusion"][confusion.prediction.file_name].append(confusion)
    return grouped


def _error_file_names(classification: MatchClassification) -> set[str]:
    return {
        *(prediction.file_name for prediction in classification.false_positives),
        *(ground_truth.file_name for ground_truth in classification.false_negatives),
        *(
            confusion.prediction.file_name
            for confusion in classification.class_confusions
        ),
    }


def _review_priorities(
    classifications: dict[str, MatchClassification],
) -> dict[tuple[str, str], str]:
    models = sorted(classifications)
    if len(models) != 2:
        return {}
    first, second = models
    priorities: dict[tuple[str, str], str] = {}
    first_fn = {
        ground_truth.annotation_id
        for ground_truth in classifications[first].false_negatives
    }
    second_fn = {
        ground_truth.annotation_id
        for ground_truth in classifications[second].false_negatives
    }
    shared_fn = first_fn & second_fn
    for model_name in models:
        for ground_truth in classifications[model_name].false_negatives:
            if ground_truth.annotation_id in shared_fn:
                priorities[(ground_truth.file_name, model_name)] = "shared_missed_gt"

    first_fp = classifications[first].false_positives
    second_fp = classifications[second].false_positives
    for first_prediction in first_fp:
        for second_prediction in second_fp:
            if (
                first_prediction.file_name == second_prediction.file_name
                and first_prediction.category == second_prediction.category
                and iou_xyxy(first_prediction.bbox_xyxy, second_prediction.bbox_xyxy)
                >= 0.5
            ):
                priorities[(first_prediction.file_name, first)] = (
                    "shared_high_confidence_fp"
                )
                priorities[(second_prediction.file_name, second)] = (
                    "shared_high_confidence_fp"
                )
    for model_name in models:
        other_model = second if model_name == first else first
        own_files = _error_file_names(classifications[model_name])
        other_files = _error_file_names(classifications[other_model])
        for file_name in own_files - other_files:
            priorities[(file_name, model_name)] = "model_only_error"
        for confusion in classifications[model_name].class_confusions:
            priorities[(confusion.prediction.file_name, model_name)] = (
                "left_right_disagreement"
            )
    return priorities


def _manifest_row(
    split_name: str,
    model_name: str,
    file_name: str,
    recording: str,
    timestamp: float | None,
    error_type: str,
    category: str,
    confidence: float | None,
    bbox_xyxy: list[float],
) -> dict[str, Any]:
    return {
        "split": split_name,
        "file_name": file_name,
        "source_recording": recording,
        "timestamp_seconds": timestamp,
        "model_name": model_name,
        "error_type": error_type,
        "category": category,
        "predicted_category": category if confidence is not None else None,
        "confidence": confidence,
        "bbox_xyxy": json.dumps([round(value, 4) for value in bbox_xyxy]),
        "review_priority": "model_error",
    }


def _draw_box(
    image: Any, box: list[float], label: str, color: tuple[int, int, int]
) -> None:
    x1, y1, x2, y2 = (round(value) for value in box)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)
    cv2.putText(
        image,
        label,
        (x1, max(18, y1 - 7)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2,
        cv2.LINE_AA,
    )


def _draw_thin_box(image: Any, box: list[float], color: tuple[int, int, int]) -> None:
    """Draw a thin ground-truth outline behind a matched prediction."""
    x1, y1, x2, y2 = (round(value) for value in box)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 1)


def _add_header(image: Any, label: str) -> Any:
    panel = cv2.copyMakeBorder(
        image, 32, 0, 0, 0, cv2.BORDER_CONSTANT, value=HEADER_BGR
    )
    cv2.putText(
        panel,
        label,
        (8, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        WHITE_BGR,
        1,
        cv2.LINE_AA,
    )
    return panel


def _write_image(path: Path, image: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise ValueError(f"Could not write audit overlay: {path}")


def _gallery_html(
    split_name: str,
    file_names: list[str],
    manifest_rows: list[dict[str, Any]],
) -> str:
    rows_by_file: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in manifest_rows:
        rows_by_file[str(row["file_name"])].append(row)
    cards = []
    for file_name in file_names:
        priorities = sorted(
            {str(row["review_priority"]) for row in rows_by_file[file_name]}
        )
        errors = sorted(
            {
                f"{row['model_name']}: {row['error_type']} {row['category']}"
                for row in rows_by_file[file_name]
            }
        )
        cards.append(
            "<article>"
            f'<img loading="lazy" src="comparison/{html.escape(file_name)}">'
            f"<h2>{html.escape(file_name)}</h2>"
            f"<p><strong>{html.escape(', '.join(priorities))}</strong></p>"
            f"<p>{html.escape(' | '.join(errors))}</p>"
            "</article>"
        )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{html.escape(split_name)} hand detection audit</title>
<style>
body{{font-family:system-ui;margin:24px;background:#f1f5f9;color:#0f172a}}
.legend{{position:sticky;top:0;background:white;padding:12px;z-index:2}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(520px,1fr));gap:18px}}
article{{background:white;padding:12px;border-radius:10px;box-shadow:0 1px 4px #94a3b8}}
img{{width:100%;height:auto}} h2{{font-size:13px;overflow-wrap:anywhere}}
</style>
</head>
<body>
<div class="legend"><strong>{html.escape(split_name)}</strong> — green TP,
yellow FP, red FN, magenta wrong class. Showing error images only.</div>
<main class="grid">{"".join(cards)}</main>
</body>
</html>
"""


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _metrics_markdown(rows: list[MetricRow]) -> str:
    header = (
        "| Split | Model | Category | Threshold | Precision | Recall | F1 | F2 | "
        "AP50 | AP75 | mAP50-95 | TP | FP | FN | Class errors |\n"
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n"
    )
    body = "".join(
        f"| {row.split} | {row.model_name} | {row.category} | "
        f"{row.confidence_threshold:.4f} | {row.precision:.4f} | "
        f"{row.recall:.4f} | {row.f1:.4f} | {row.f2:.4f} | "
        f"{row.ap50:.4f} | {row.ap75:.4f} | {row.map_50_95:.4f} | "
        f"{row.true_positives} | {row.false_positives} | "
        f"{row.false_negatives} | {row.class_confusions} |\n"
        for row in rows
    )
    return (
        "# Diagnostic v3 Metrics\n\nThese are development/audit results, not a frozen final benchmark.\n\n"
        + header
        + body
    )


def _deltas_markdown(rows: list[dict[str, Any]]) -> str:
    header = (
        "| Split | Category | Δ Precision | Δ Recall | Δ F1 | Δ F2 | "
        "Δ AP50 | Δ mAP50-95 |\n"
        "|---|---|---:|---:|---:|---:|---:|---:|\n"
    )
    body = "".join(
        f"| {row['split']} | {row['category']} | "
        f"{row['precision_absolute_delta']:+.4f} | "
        f"{row['recall_absolute_delta']:+.4f} | "
        f"{row['f1_absolute_delta']:+.4f} | "
        f"{row['f2_absolute_delta']:+.4f} | "
        f"{row['ap50_absolute_delta']:+.4f} | "
        f"{row['map_50_95_absolute_delta']:+.4f} |\n"
        for row in rows
    )
    return "# RF-DETR Improvement over WiLoR\n\n" + header + body


def _write_pr_svg(
    path: Path,
    title: str,
    series: list[tuple[str, list[dict[str, float | int]], str]],
) -> None:
    width, height, margin = 760, 520, 60
    plot_width, plot_height = width - 2 * margin, height - 2 * margin
    polylines = []
    legend = []
    for index, (label, rows, color) in enumerate(series):
        points = " ".join(
            f"{margin + float(row['recall']) * plot_width:.1f},"
            f"{margin + (1 - float(row['precision'])) * plot_height:.1f}"
            for row in rows
        )
        polylines.append(
            f'<polyline points="{points}" fill="none" stroke="{color}" '
            'stroke-width="3"/>'
        )
        legend.append(
            f'<text x="{margin + 10}" y="{margin + 20 + index * 20}" '
            f'fill="{color}" font-size="14">{html.escape(label)}</text>'
        )
    _write_svg(
        path, title, width, height, margin, polylines + legend, "Recall", "Precision"
    )


def _write_fscore_svg(
    path: Path,
    title: str,
    series: list[tuple[str, list[dict[str, float | int]], str]],
) -> None:
    width, height, margin = 760, 520, 60
    plot_width, plot_height = width - 2 * margin, height - 2 * margin
    elements = []
    legend = []
    for index, (label, rows, color) in enumerate(series):
        ordered = sorted(rows, key=lambda row: float(row["confidence_threshold"]))
        for metric, dash in (("f1", ""), ("f2", ' stroke-dasharray="8 5"')):
            points = " ".join(
                f"{margin + float(row['confidence_threshold']) * plot_width:.1f},"
                f"{margin + (1 - float(row[metric])) * plot_height:.1f}"
                for row in ordered
            )
            elements.append(
                f'<polyline points="{points}" fill="none" stroke="{color}" '
                f'stroke-width="3"{dash}/>'
            )
        legend.append(
            f'<text x="{margin + 10}" y="{margin + 20 + index * 20}" '
            f'fill="{color}" font-size="14">{html.escape(label)} (solid F1, dashed F2)</text>'
        )
    _write_svg(
        path, title, width, height, margin, elements + legend, "Threshold", "F-score"
    )


def _write_svg(
    path: Path,
    title: str,
    width: int,
    height: int,
    margin: int,
    elements: list[str],
    x_label: str,
    y_label: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">
<rect width="100%" height="100%" fill="white"/>
<text x="{width / 2}" y="28" text-anchor="middle" font-size="18">{html.escape(title)}</text>
<line x1="{margin}" y1="{height - margin}" x2="{width - margin}" y2="{height - margin}" stroke="#334155"/>
<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height - margin}" stroke="#334155"/>
<text x="{width / 2}" y="{height - 12}" text-anchor="middle">{html.escape(x_label)}</text>
<text x="16" y="{height / 2}" transform="rotate(-90 16 {height / 2})" text-anchor="middle">{html.escape(y_label)}</text>
{"".join(elements)}
</svg>
"""
    path.write_text(svg, encoding="utf-8")


def _fbeta(precision: float, recall: float, beta: float) -> float:
    denominator = beta * beta * precision + recall
    return (1 + beta * beta) * precision * recall / denominator if denominator else 0.0
