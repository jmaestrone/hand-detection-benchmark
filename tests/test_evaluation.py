"""Tests for class-aware matching and exact confidence threshold selection."""

import json
from pathlib import Path

from hand_benchmark.coco_dataset import load_coco_split
from hand_benchmark.evaluation import (
    GroundTruthBox,
    PredictedBox,
    classify_matches,
    coco_average_precision,
    counts_from_classification,
    iou_xyxy,
    select_thresholds,
)


def _ground_truth(
    annotation_id: int,
    category: str = "left_hand",
    box: list[float] | None = None,
) -> GroundTruthBox:
    return GroundTruthBox("image.jpg", category, box or [0, 0, 10, 10], annotation_id)


def _prediction(
    confidence: float,
    category: str = "left_hand",
    box: list[float] | None = None,
) -> PredictedBox:
    return PredictedBox(
        "image.jpg", category, box or [0, 0, 10, 10], confidence, "model"
    )


def test_duplicate_prediction_is_one_true_positive_and_one_false_positive() -> None:
    classification = classify_matches(
        [_ground_truth(1)],
        [_prediction(0.9), _prediction(0.8)],
        confidence_threshold=0.5,
    )

    counts = counts_from_classification(classification)
    assert (counts.true_positives, counts.false_positives, counts.false_negatives) == (
        1,
        1,
        0,
    )


def test_wrong_class_match_counts_as_false_positive_and_false_negative() -> None:
    classification = classify_matches(
        [_ground_truth(1, "left_hand")],
        [_prediction(0.9, "right_hand")],
        confidence_threshold=0.5,
    )

    counts = counts_from_classification(classification)
    assert (counts.true_positives, counts.false_positives, counts.false_negatives) == (
        0,
        1,
        1,
    )
    assert len(classification.class_confusions) == 1
    assert not classification.false_positives
    assert not classification.false_negatives


def test_empty_predictions_leave_all_ground_truth_as_false_negatives() -> None:
    classification = classify_matches(
        [_ground_truth(1), _ground_truth(2, "right_hand")],
        [],
        confidence_threshold=0.5,
    )

    counts = counts_from_classification(classification)
    assert (counts.true_positives, counts.false_positives, counts.false_negatives) == (
        0,
        0,
        2,
    )


def test_iou_threshold_boundary_is_inclusive() -> None:
    first = [0, 0, 10, 10]
    second = [0, 0, 10, 20]
    assert iou_xyxy(first, second) == 0.5

    classification = classify_matches(
        [_ground_truth(1, box=first)],
        [_prediction(0.9, box=second)],
        confidence_threshold=0.5,
        iou_threshold=0.5,
    )
    assert len(classification.true_positives) == 1
    assert classification.true_positive_matches[0].iou == 0.5


def test_threshold_selection_records_balanced_and_recall_heavy_optima() -> None:
    ground_truths = [
        GroundTruthBox("one.jpg", "left_hand", [0, 0, 10, 10], 1),
        GroundTruthBox("two.jpg", "left_hand", [0, 0, 10, 10], 2),
    ]
    predictions = [
        PredictedBox("one.jpg", "left_hand", [0, 0, 10, 10], 0.9, "model"),
        PredictedBox("two.jpg", "left_hand", [0, 0, 10, 10], 0.4, "model"),
        PredictedBox("three.jpg", "left_hand", [0, 0, 10, 10], 0.4, "model"),
        PredictedBox("four.jpg", "left_hand", [0, 0, 10, 10], 0.4, "model"),
        PredictedBox("five.jpg", "left_hand", [0, 0, 10, 10], 0.4, "model"),
    ]

    selection = select_thresholds(
        model_name="model",
        ground_truths=ground_truths,
        predictions=predictions,
    )

    assert selection.selected_f1_threshold == 0.9
    assert selection.selected_f2_threshold == 0.4


def test_coco_average_precision_is_perfect_for_exact_two_class_boxes(
    tmp_path: Path,
) -> None:
    split_dir = tmp_path / "test"
    split_dir.mkdir()
    file_name = "recording_frame000000_0000000000ms.jpg"
    (split_dir / file_name).write_bytes(b"not-decoded-for-metrics")
    payload = {
        "info": {},
        "licenses": [],
        "images": [{"id": 1, "file_name": file_name, "width": 20, "height": 20}],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [0, 0, 10, 10],
                "area": 100,
                "iscrowd": 0,
            },
            {
                "id": 2,
                "image_id": 1,
                "category_id": 2,
                "bbox": [10, 10, 10, 10],
                "area": 100,
                "iscrowd": 0,
            },
        ],
        "categories": [
            {"id": 0, "name": "unused"},
            {"id": 1, "name": "left_hand"},
            {"id": 2, "name": "right_hand"},
        ],
    }
    (split_dir / "_annotations.coco.json").write_text(json.dumps(payload))
    split = load_coco_split(tmp_path, "test")
    predictions = [
        PredictedBox(file_name, "left_hand", [0, 0, 10, 10], 0.9, "model"),
        PredictedBox(file_name, "right_hand", [10, 10, 20, 20], 0.9, "model"),
    ]

    metrics = coco_average_precision(split, predictions)

    assert metrics["overall"]["map_50_95"] == 1.0
    assert metrics["left_hand"]["ap50"] == 1.0
    assert metrics["right_hand"]["ap75"] == 1.0
