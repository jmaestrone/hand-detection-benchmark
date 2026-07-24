"""Class-aware hand-detection matching, thresholds, and COCO metrics."""

from __future__ import annotations

import contextlib
import io
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from hand_benchmark.benchmark_predictions import (
    CANONICAL_CLASS_IDS,
    LEFT_HAND,
    RIGHT_HAND,
    read_prediction_records,
)
from hand_benchmark.coco_dataset import CocoSplit


@dataclass(frozen=True)
class GroundTruthBox:
    """One canonical COCO hand box."""

    file_name: str
    category: str
    bbox_xyxy: list[float]
    annotation_id: int


@dataclass(frozen=True)
class PredictedBox:
    """One normalized model prediction."""

    file_name: str
    category: str
    bbox_xyxy: list[float]
    confidence: float
    model_name: str


@dataclass(frozen=True)
class ClassConfusion:
    """A spatial hand match whose predicted left/right class is wrong."""

    prediction: PredictedBox
    ground_truth: GroundTruthBox
    iou: float


@dataclass(frozen=True)
class TruePositiveMatch:
    """One class-correct prediction paired with its ground truth and IoU."""

    prediction: PredictedBox
    ground_truth: GroundTruthBox
    iou: float


@dataclass(frozen=True)
class MatchClassification:
    """TP, FP, FN, and wrong-class assignments at one operating point."""

    true_positives: list[PredictedBox]
    false_positives: list[PredictedBox]
    false_negatives: list[GroundTruthBox]
    class_confusions: list[ClassConfusion]
    true_positive_matches: list[TruePositiveMatch] = field(default_factory=list)


@dataclass(frozen=True)
class Counts:
    """Detection counts used for precision/recall/F-scores."""

    true_positives: int
    false_positives: int
    false_negatives: int


@dataclass(frozen=True)
class MetricRow:
    """Metrics for one split, model, and class aggregation."""

    split: str
    model_name: str
    category: str
    confidence_threshold: float
    iou_threshold: float
    image_count: int
    ground_truth_count: int
    prediction_count: int
    true_positives: int
    false_positives: int
    false_negatives: int
    class_confusions: int
    precision: float
    recall: float
    f1: float
    f2: float
    ap50: float
    ap75: float
    map_50_95: float


@dataclass(frozen=True)
class ThresholdSelection:
    """Exact validation sweep and selected F1/F2 operating points."""

    model_name: str
    rows: list[dict[str, float | int]]
    selected_f1_threshold: float
    selected_f2_threshold: float
    selected_f1_row: dict[str, float | int]
    selected_f2_row: dict[str, float | int]


def load_predictions_for_split(
    split: CocoSplit, predictions_path: Path
) -> tuple[str, list[PredictedBox], list[dict[str, Any]]]:
    """Load prediction rows and require exact one-row-per-image coverage."""
    records = read_prediction_records(predictions_path)
    expected_names = {image.file_name for image in split.images}
    record_names = [str(record["file_name"]) for record in records]
    if len(record_names) != len(set(record_names)):
        raise ValueError(f"Duplicate prediction filenames in {predictions_path}")
    missing = expected_names - set(record_names)
    extra = set(record_names) - expected_names
    if missing or extra:
        raise ValueError(
            f"Prediction coverage mismatch for {split.name}: "
            f"missing={len(missing)}, extra={len(extra)}"
        )
    model_names = {str(record["model_name"]) for record in records}
    if len(model_names) != 1:
        raise ValueError(
            f"Expected one model name in {predictions_path}; found {model_names}"
        )
    boxes = [
        PredictedBox(
            file_name=str(record["file_name"]),
            category=str(detection["category"]),
            bbox_xyxy=[float(value) for value in detection["bbox_xyxy"]],
            confidence=float(detection["confidence"]),
            model_name=str(record["model_name"]),
        )
        for record in records
        for detection in record["detections"]
    ]
    unknown_categories = {
        box.category for box in boxes if box.category not in CANONICAL_CLASS_IDS
    }
    if unknown_categories:
        raise ValueError(f"Unknown prediction categories: {unknown_categories}")
    return next(iter(model_names)), boxes, records


def ground_truth_boxes(split: CocoSplit) -> list[GroundTruthBox]:
    """Flatten canonical split annotations into evaluation boxes."""
    image_name_by_id = {image.id: image.file_name for image in split.images}
    return [
        GroundTruthBox(
            file_name=image_name_by_id[annotation.image_id],
            category=annotation.category,
            bbox_xyxy=annotation.bbox_xyxy,
            annotation_id=annotation.id,
        )
        for annotation in split.annotations
    ]


def classify_matches(
    ground_truths: list[GroundTruthBox],
    predictions: list[PredictedBox],
    confidence_threshold: float,
    iou_threshold: float = 0.5,
) -> MatchClassification:
    """Classify detections using greedy class-aware matching by confidence."""
    filtered_predictions = sorted(
        (
            prediction
            for prediction in predictions
            if prediction.confidence >= confidence_threshold
        ),
        key=lambda prediction: prediction.confidence,
        reverse=True,
    )
    ground_truth_by_file: dict[str, list[GroundTruthBox]] = {}
    for ground_truth in ground_truths:
        ground_truth_by_file.setdefault(ground_truth.file_name, []).append(ground_truth)

    matched_annotation_ids: set[int] = set()
    true_positives: list[PredictedBox] = []
    true_positive_matches: list[TruePositiveMatch] = []
    unmatched_predictions: list[PredictedBox] = []
    for prediction in filtered_predictions:
        candidate = _best_ground_truth(
            prediction,
            ground_truth_by_file.get(prediction.file_name, []),
            matched_annotation_ids,
            iou_threshold,
            same_category=True,
        )
        if candidate is None:
            unmatched_predictions.append(prediction)
        else:
            matched_annotation_ids.add(candidate.annotation_id)
            true_positives.append(prediction)
            true_positive_matches.append(
                TruePositiveMatch(
                    prediction=prediction,
                    ground_truth=candidate,
                    iou=iou_xyxy(prediction.bbox_xyxy, candidate.bbox_xyxy),
                )
            )

    class_confusions: list[ClassConfusion] = []
    false_positives: list[PredictedBox] = []
    for prediction in unmatched_predictions:
        candidate = _best_ground_truth(
            prediction,
            ground_truth_by_file.get(prediction.file_name, []),
            matched_annotation_ids,
            iou_threshold,
            same_category=False,
        )
        if candidate is None:
            false_positives.append(prediction)
        else:
            overlap = iou_xyxy(prediction.bbox_xyxy, candidate.bbox_xyxy)
            matched_annotation_ids.add(candidate.annotation_id)
            class_confusions.append(ClassConfusion(prediction, candidate, overlap))

    false_negatives = [
        ground_truth
        for ground_truth in ground_truths
        if ground_truth.annotation_id not in matched_annotation_ids
    ]
    return MatchClassification(
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
        class_confusions=class_confusions,
        true_positive_matches=true_positive_matches,
    )


def counts_from_classification(
    classification: MatchClassification, category: str | None = None
) -> Counts:
    """Return metric counts, treating wrong-class matches as one FP and one FN."""
    true_positives = sum(
        category is None or prediction.category == category
        for prediction in classification.true_positives
    )
    false_positives = sum(
        category is None or prediction.category == category
        for prediction in classification.false_positives
    )
    false_negatives = sum(
        category is None or ground_truth.category == category
        for ground_truth in classification.false_negatives
    )
    false_positives += sum(
        category is None or confusion.prediction.category == category
        for confusion in classification.class_confusions
    )
    false_negatives += sum(
        category is None or confusion.ground_truth.category == category
        for confusion in classification.class_confusions
    )
    return Counts(true_positives, false_positives, false_negatives)


def select_thresholds(
    *,
    model_name: str,
    ground_truths: list[GroundTruthBox],
    predictions: list[PredictedBox],
    iou_threshold: float = 0.5,
) -> ThresholdSelection:
    """Sweep exact prediction confidence breakpoints and select F1 and F2."""
    thresholds = sorted(
        {1.0, 0.0, *(prediction.confidence for prediction in predictions)},
        reverse=True,
    )
    rows = []
    for threshold in thresholds:
        classification = classify_matches(
            ground_truths, predictions, threshold, iou_threshold
        )
        counts = counts_from_classification(classification)
        rows.append(_metric_values(threshold, counts))
    selected_f1 = max(
        rows, key=lambda row: (float(row["f1"]), float(row["confidence_threshold"]))
    )
    selected_f2 = max(
        rows, key=lambda row: (float(row["f2"]), float(row["confidence_threshold"]))
    )
    return ThresholdSelection(
        model_name=model_name,
        rows=rows,
        selected_f1_threshold=float(selected_f1["confidence_threshold"]),
        selected_f2_threshold=float(selected_f2["confidence_threshold"]),
        selected_f1_row=selected_f1,
        selected_f2_row=selected_f2,
    )


def evaluate_split(
    *,
    split: CocoSplit,
    predictions_path: Path,
    confidence_threshold: float,
    iou_threshold: float = 0.5,
) -> tuple[list[MetricRow], MatchClassification, list[dict[str, Any]]]:
    """Evaluate one model on one split overall and per canonical class."""
    model_name, predictions, raw_records = load_predictions_for_split(
        split, predictions_path
    )
    ground_truths = ground_truth_boxes(split)
    classification = classify_matches(
        ground_truths, predictions, confidence_threshold, iou_threshold
    )
    coco_metrics = coco_average_precision(split, predictions)
    rows = [
        _evaluation_row(
            split=split,
            model_name=model_name,
            category="overall_micro",
            confidence_threshold=confidence_threshold,
            iou_threshold=iou_threshold,
            ground_truths=ground_truths,
            predictions=predictions,
            classification=classification,
            ap_metrics=coco_metrics["overall"],
        )
    ]
    for category in (LEFT_HAND, RIGHT_HAND):
        rows.append(
            _evaluation_row(
                split=split,
                model_name=model_name,
                category=category,
                confidence_threshold=confidence_threshold,
                iou_threshold=iou_threshold,
                ground_truths=[
                    box for box in ground_truths if box.category == category
                ],
                predictions=[box for box in predictions if box.category == category],
                classification=classification,
                ap_metrics=coco_metrics[category],
            )
        )
    rows.append(_macro_row(rows))
    return rows, classification, raw_records


def coco_average_precision(
    split: CocoSplit, predictions: list[PredictedBox]
) -> dict[str, dict[str, float]]:
    """Compute standard COCO AP overall and for each canonical category."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    category_ids = {
        name: category_id
        for category_id, name in split.category_id_to_name.items()
        if name in CANONICAL_CLASS_IDS
    }
    image_id_by_name = {image.file_name: image.id for image in split.images}
    coco_results = [
        {
            "image_id": image_id_by_name[prediction.file_name],
            "category_id": category_ids[prediction.category],
            "bbox": _xyxy_to_xywh(prediction.bbox_xyxy),
            "score": prediction.confidence,
        }
        for prediction in predictions
    ]
    metrics = {}
    for label, selected_category_ids in (
        ("overall", list(category_ids.values())),
        (LEFT_HAND, [category_ids[LEFT_HAND]]),
        (RIGHT_HAND, [category_ids[RIGHT_HAND]]),
    ):
        metrics[label] = _run_coco_eval(
            COCO(str(split.annotations_path)),
            coco_results,
            [image.id for image in split.images],
            selected_category_ids,
            COCOeval,
        )
    return metrics


def precision(counts: Counts) -> float:
    denominator = counts.true_positives + counts.false_positives
    return counts.true_positives / denominator if denominator else 0.0


def recall(counts: Counts) -> float:
    denominator = counts.true_positives + counts.false_negatives
    return counts.true_positives / denominator if denominator else 0.0


def fbeta(precision_value: float, recall_value: float, beta: float) -> float:
    beta_squared = beta * beta
    denominator = beta_squared * precision_value + recall_value
    if denominator == 0:
        return 0.0
    return (1 + beta_squared) * precision_value * recall_value / denominator


def iou_xyxy(first: list[float], second: list[float]) -> float:
    """Compute intersection over union for two xyxy boxes."""
    x1 = max(first[0], second[0])
    y1 = max(first[1], second[1])
    x2 = min(first[2], second[2])
    y2 = min(first[3], second[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


def metric_row_to_dict(row: MetricRow) -> dict[str, Any]:
    """Serialize a metric row for deterministic reports."""
    return asdict(row)


def _best_ground_truth(
    prediction: PredictedBox,
    candidates: list[GroundTruthBox],
    matched_annotation_ids: set[int],
    iou_threshold: float,
    *,
    same_category: bool,
) -> GroundTruthBox | None:
    best: GroundTruthBox | None = None
    best_iou = 0.0
    for candidate in candidates:
        if candidate.annotation_id in matched_annotation_ids:
            continue
        category_matches = candidate.category == prediction.category
        if category_matches != same_category:
            continue
        overlap = iou_xyxy(prediction.bbox_xyxy, candidate.bbox_xyxy)
        if overlap >= iou_threshold and overlap > best_iou:
            best = candidate
            best_iou = overlap
    return best


def _metric_values(threshold: float, counts: Counts) -> dict[str, float | int]:
    precision_value = precision(counts)
    recall_value = recall(counts)
    return {
        "confidence_threshold": round(threshold, 6),
        "true_positives": counts.true_positives,
        "false_positives": counts.false_positives,
        "false_negatives": counts.false_negatives,
        "precision": round(precision_value, 6),
        "recall": round(recall_value, 6),
        "f1": round(fbeta(precision_value, recall_value, 1.0), 6),
        "f2": round(fbeta(precision_value, recall_value, 2.0), 6),
    }


def _evaluation_row(
    *,
    split: CocoSplit,
    model_name: str,
    category: str,
    confidence_threshold: float,
    iou_threshold: float,
    ground_truths: list[GroundTruthBox],
    predictions: list[PredictedBox],
    classification: MatchClassification,
    ap_metrics: dict[str, float],
) -> MetricRow:
    selected_category = None if category == "overall_micro" else category
    counts = counts_from_classification(classification, selected_category)
    precision_value = precision(counts)
    recall_value = recall(counts)
    return MetricRow(
        split=split.name,
        model_name=model_name,
        category=category,
        confidence_threshold=round(confidence_threshold, 6),
        iou_threshold=iou_threshold,
        image_count=len(split.images),
        ground_truth_count=len(ground_truths),
        prediction_count=sum(
            prediction.confidence >= confidence_threshold for prediction in predictions
        ),
        true_positives=counts.true_positives,
        false_positives=counts.false_positives,
        false_negatives=counts.false_negatives,
        class_confusions=sum(
            selected_category is None
            or confusion.prediction.category == selected_category
            or confusion.ground_truth.category == selected_category
            for confusion in classification.class_confusions
        ),
        precision=round(precision_value, 6),
        recall=round(recall_value, 6),
        f1=round(fbeta(precision_value, recall_value, 1.0), 6),
        f2=round(fbeta(precision_value, recall_value, 2.0), 6),
        ap50=ap_metrics["ap50"],
        ap75=ap_metrics["ap75"],
        map_50_95=ap_metrics["map_50_95"],
    )


def _macro_row(rows: list[MetricRow]) -> MetricRow:
    class_rows = [row for row in rows if row.category in (LEFT_HAND, RIGHT_HAND)]
    reference = class_rows[0]
    mean_fields = (
        "precision",
        "recall",
        "f1",
        "f2",
        "ap50",
        "ap75",
        "map_50_95",
    )
    means = {
        field: round(
            sum(float(getattr(row, field)) for row in class_rows) / len(class_rows),
            6,
        )
        for field in mean_fields
    }
    return MetricRow(
        split=reference.split,
        model_name=reference.model_name,
        category="macro_classes",
        confidence_threshold=reference.confidence_threshold,
        iou_threshold=reference.iou_threshold,
        image_count=reference.image_count,
        ground_truth_count=sum(row.ground_truth_count for row in class_rows),
        prediction_count=sum(row.prediction_count for row in class_rows),
        true_positives=sum(row.true_positives for row in class_rows),
        false_positives=sum(row.false_positives for row in class_rows),
        false_negatives=sum(row.false_negatives for row in class_rows),
        class_confusions=max(row.class_confusions for row in class_rows),
        **means,
    )


def _run_coco_eval(
    coco_ground_truth: Any,
    coco_results: list[dict[str, Any]],
    image_ids: list[int],
    category_ids: list[int],
    evaluator_type: Any,
) -> dict[str, float]:
    if not coco_results:
        return {"map_50_95": 0.0, "ap50": 0.0, "ap75": 0.0}
    with contextlib.redirect_stdout(io.StringIO()):
        coco_detections = coco_ground_truth.loadRes(coco_results)
        evaluator = evaluator_type(coco_ground_truth, coco_detections, "bbox")
        evaluator.params.imgIds = image_ids
        evaluator.params.catIds = category_ids
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    return {
        "map_50_95": round(max(0.0, float(evaluator.stats[0])), 6),
        "ap50": round(max(0.0, float(evaluator.stats[1])), 6),
        "ap75": round(max(0.0, float(evaluator.stats[2])), 6),
    }


def _xyxy_to_xywh(box: list[float]) -> list[float]:
    return [
        box[0],
        box[1],
        max(0.0, box[2] - box[0]),
        max(0.0, box[3] - box[1]),
    ]
