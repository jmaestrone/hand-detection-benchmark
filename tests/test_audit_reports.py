"""Tests for colored error audits and metric-comparison artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from hand_benchmark.audit_reports import (
    metric_deltas,
    render_complete_model_review,
    render_error_audit,
)
from hand_benchmark.coco_dataset import CocoImage, CocoSplit
from hand_benchmark.evaluation import (
    ClassConfusion,
    GroundTruthBox,
    MatchClassification,
    MetricRow,
    PredictedBox,
    TruePositiveMatch,
)


def _metric_row(model: str, f2: float) -> MetricRow:
    return MetricRow(
        split="test",
        model_name=model,
        category="overall_micro",
        confidence_threshold=0.5,
        iou_threshold=0.5,
        image_count=1,
        ground_truth_count=1,
        prediction_count=1,
        true_positives=1,
        false_positives=0,
        false_negatives=0,
        class_confusions=0,
        precision=f2,
        recall=f2,
        f1=f2,
        f2=f2,
        ap50=f2,
        ap75=f2,
        map_50_95=f2,
    )


def test_metric_deltas_include_absolute_and_relative_improvement() -> None:
    rows = []
    for split in ("train", "valid", "test"):
        for category in (
            "overall_micro",
            "left_hand",
            "right_hand",
            "macro_classes",
        ):
            baseline = _metric_row("wilor-yolo", 0.5)
            candidate = _metric_row("rfdetr-checkpoint-best-total", 0.75)
            rows.extend(
                [
                    MetricRow(
                        **{**baseline.__dict__, "split": split, "category": category}
                    ),
                    MetricRow(
                        **{**candidate.__dict__, "split": split, "category": category}
                    ),
                ]
            )

    deltas = metric_deltas(rows, "wilor-yolo", "rfdetr-checkpoint-best-total")

    assert deltas[0]["f2_absolute_delta"] == 0.25
    assert deltas[0]["f2_relative_percent"] == 50.0


def test_error_audit_writes_all_error_types_and_gallery(tmp_path: Path) -> None:
    file_name = "recording_frame000000_0000000000ms.png"
    image_path = tmp_path / file_name
    assert cv2.imwrite(str(image_path), np.zeros((80, 120, 3), dtype=np.uint8))
    split = CocoSplit(
        name="test",
        directory=tmp_path,
        annotations_path=tmp_path / "_annotations.coco.json",
        images=[CocoImage(1, file_name, 120, 80, image_path)],
        annotations=[],
        annotations_by_image_id={1: []},
        category_id_to_name={1: "left_hand", 2: "right_hand"},
        ignored_category_ids=set(),
        recording_ids={"recording"},
    )
    true_positive = PredictedBox(
        file_name, "left_hand", [5, 5, 20, 20], 0.9, "wilor-yolo"
    )
    false_positive = PredictedBox(
        file_name, "right_hand", [25, 5, 40, 20], 0.8, "wilor-yolo"
    )
    false_negative = GroundTruthBox(file_name, "left_hand", [45, 5, 60, 20], 1)
    confused_prediction = PredictedBox(
        file_name, "right_hand", [65, 5, 80, 20], 0.7, "wilor-yolo"
    )
    confused_ground_truth = GroundTruthBox(file_name, "left_hand", [65, 5, 80, 20], 2)
    wilor = MatchClassification(
        [true_positive],
        [false_positive],
        [false_negative],
        [ClassConfusion(confused_prediction, confused_ground_truth, 1.0)],
    )
    rfdetr = MatchClassification([true_positive], [], [], [])
    stale_path = tmp_path / "errors/test/comparison/stale.jpg"
    stale_path.parent.mkdir(parents=True)
    assert cv2.imwrite(str(stale_path), np.zeros((4, 4, 3), dtype=np.uint8))

    rows = render_error_audit(
        split=split,
        classifications={
            "wilor-yolo": wilor,
            "rfdetr-checkpoint-best-total": rfdetr,
        },
        output_dir=tmp_path / "errors",
        thresholds={"wilor-yolo": 0.5, "rfdetr-checkpoint-best-total": 0.4},
    )

    assert {row["error_type"] for row in rows} == {
        "false_positive",
        "false_negative",
        "class_confusion",
    }
    assert (tmp_path / "errors/test/comparison" / file_name).is_file()
    assert not stale_path.exists()
    assert "magenta" in (tmp_path / "errors/test/index.html").read_text()
    json_rows = json.loads((tmp_path / "errors/test/error_manifest.json").read_text())
    assert len(json_rows) == 3


def test_complete_review_renders_all_images_and_ranks_low_iou(
    tmp_path: Path,
) -> None:
    file_name = "recording_frame000000_0000000000ms.png"
    image_path = tmp_path / file_name
    assert cv2.imwrite(str(image_path), np.zeros((80, 120, 3), dtype=np.uint8))
    split = CocoSplit(
        name="valid",
        directory=tmp_path,
        annotations_path=tmp_path / "_annotations.coco.json",
        images=[CocoImage(1, file_name, 120, 80, image_path)],
        annotations=[],
        annotations_by_image_id={1: []},
        category_id_to_name={1: "left_hand", 2: "right_hand"},
        ignored_category_ids=set(),
        recording_ids={"recording"},
    )
    prediction = PredictedBox(file_name, "left_hand", [10, 10, 35, 35], 0.9, "rfdetr")
    ground_truth = GroundTruthBox(file_name, "left_hand", [15, 15, 40, 40], 1)
    classification = MatchClassification(
        true_positives=[prediction],
        false_positives=[],
        false_negatives=[],
        class_confusions=[],
        true_positive_matches=[TruePositiveMatch(prediction, ground_truth, 0.470588)],
    )

    rows = render_complete_model_review(
        split=split,
        model_name="rfdetr",
        classification=classification,
        output_dir=tmp_path / "review",
        threshold=0.25,
    )

    review_dir = tmp_path / "review/rfdetr/valid"
    assert len(rows) == 1
    assert rows[0]["minimum_true_positive_iou"] == 0.470588
    assert (review_dir / "images" / file_name).is_file()
    assert file_name in (review_dir / "index.html").read_text()
    assert file_name in (review_dir / "lowest-iou.html").read_text()
    box_rows = json.loads((review_dir / "box_manifest.json").read_text())
    assert box_rows[0]["iou"] == 0.470588
