"""Tests for canonical model-class normalization and inference metadata."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from hand_benchmark.benchmark_inference import (
    _normalize_arrays,
    _rfdetr_class_slot_mapping,
    parse_split_names,
    predict_rfdetr_frames,
    read_rfdetr_checkpoint_metadata,
)
from hand_benchmark.benchmark_predictions import (
    canonical_category,
    validate_ordered_class_names,
)
from hand_benchmark.coco_dataset import CocoImage, CocoSplit
from hand_benchmark.roboflow_export import export_roboflow_yolo


def test_model_class_aliases_normalize_to_canonical_schema() -> None:
    assert canonical_category("left") == "left_hand"
    assert canonical_category("left_hand") == "left_hand"
    assert canonical_category("hand_left") == "left_hand"
    assert canonical_category("right") == "right_hand"
    assert canonical_category("hand_right") == "right_hand"

    with pytest.raises(ValueError, match="Unknown hand category"):
        canonical_category("hand")
    with pytest.raises(ValueError, match="Expected ordered left/right"):
        validate_ordered_class_names(["right", "left"])


def test_normalized_detections_clip_boxes_and_retain_raw_class(tmp_path: Path) -> None:
    image = CocoImage(1, "recording_frame000000_0000000000ms.jpg", 100, 50, tmp_path)

    detections = _normalize_arrays(
        boxes=[[-1, 2, 120, 40]],
        confidences=[0.75],
        class_ids=[1],
        raw_names=["hand_left", "hand_right"],
        image=image,
    )

    assert detections[0].category == "right_hand"
    assert detections[0].raw_class_name == "hand_right"
    assert detections[0].bbox_xyxy == [0.0, 2.0, 100, 40.0]


def test_rfdetr_normalization_uses_explicit_slot_mapping(
    tmp_path: Path,
) -> None:
    image = CocoImage(1, "recording_frame000000_0000000000ms.jpg", 100, 50, tmp_path)

    detections = _normalize_arrays(
        boxes=[[0, 0, 10, 10], [20, 20, 30, 30]],
        confidences=[0.9, 0.8],
        class_ids=[1, 2],
        raw_names=["hand_left", "hand_right"],
        image=image,
        emitted_class_names=["hand_right", "__background__"],
        class_id_to_name={0: None, 1: "hand_left", 2: "hand_right"},
    )

    assert [detection.category for detection in detections] == [
        "left_hand",
        "right_hand",
    ]
    assert detections[0].raw_class_id == 1


def test_rfdetr_slot_mapping_preserves_unused_roboflow_parent(
    tmp_path: Path,
) -> None:
    split = CocoSplit(
        name="train",
        directory=tmp_path,
        annotations_path=tmp_path / "_annotations.coco.json",
        images=[],
        annotations=[],
        annotations_by_image_id={},
        category_id_to_name={
            0: "project-parent",
            1: "left_hand",
            2: "right_hand",
        },
        ignored_category_ids={0},
        recording_ids=set(),
    )

    mapping = _rfdetr_class_slot_mapping(
        split, ["hand_left", "hand_right"], classifier_slot_count=3
    )

    assert mapping == {0: None, 1: "left_hand", 2: "right_hand"}


def test_rfdetr_slot_mapping_supports_standard_contiguous_classes(
    tmp_path: Path,
) -> None:
    split = CocoSplit(
        name="train",
        directory=tmp_path,
        annotations_path=tmp_path / "_annotations.coco.json",
        images=[],
        annotations=[],
        annotations_by_image_id={},
        category_id_to_name={1: "left_hand", 2: "right_hand"},
        ignored_category_ids=set(),
        recording_ids=set(),
    )

    mapping = _rfdetr_class_slot_mapping(
        split, ["hand_left", "hand_right"], classifier_slot_count=3
    )

    assert mapping == {0: "hand_left", 1: "hand_right", 2: None}


def test_split_parser_supports_all_or_one_split() -> None:
    assert parse_split_names("all") == ("train", "valid", "test")
    assert parse_split_names("valid") == ("valid",)
    with pytest.raises(ValueError, match="--split"):
        parse_split_names("validation")


def test_rfdetr_checkpoint_metadata_requires_left_then_right(
    tmp_path: Path,
) -> None:
    torch = pytest.importorskip("torch")
    checkpoint_path = tmp_path / "checkpoint.pth"
    torch.save(
        {
            "args": {
                "class_names": ["hand_left", "hand_right"],
                "epochs": 8,
                "group_detr": 13,
                "num_select": 5,
            },
            "model": {
                "refpoint_embed.weight": torch.zeros((65, 4)),
                "class_embed.bias": torch.zeros(3),
            },
            "model_name": "RFDETRLarge",
            "rfdetr_version": "1.8.3",
        },
        checkpoint_path,
    )

    metadata = read_rfdetr_checkpoint_metadata(checkpoint_path)

    assert metadata["model_name"] == "RFDETRLarge"
    assert metadata["class_names"] == ["hand_left", "hand_right"]
    assert metadata["num_queries"] == 5
    assert metadata["num_queries_source"] == "refpoint_embed.weight/group_detr"
    assert metadata["classifier_slot_count"] == 3


def test_rfdetr_frame_prelabels_feed_roboflow_export(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    file_name = "recording_frame000030_0000001000ms.jpg"
    (frames_dir / file_name).touch()
    frame_metadata_path = frames_dir / "metadata.jsonl"
    frame_record = {
        "file_name": file_name,
        "output_path": file_name,
        "source_video": "recording.mp4",
        "source_mcap_path": "/mcaps/recording.mcap",
        "source_mcap_stem": "recording",
        "video_topic": "/head_left/video",
        "frame_index": 30,
        "timestamp_seconds": 1.0,
        "width": 100,
        "height": 50,
    }
    frame_metadata_path.write_text(json.dumps(frame_record) + "\n")

    schema_train_dir = tmp_path / "schema" / "train"
    schema_train_dir.mkdir(parents=True)
    (schema_train_dir / "_annotations.coco.json").write_text(
        json.dumps(
            {
                "categories": [
                    {"id": 0, "name": "project-parent"},
                    {"id": 1, "name": "left_hand"},
                    {"id": 2, "name": "right_hand"},
                ],
                "images": [],
                "annotations": [],
            }
        )
    )
    weights_path = tmp_path / "checkpoint.pth"
    weights_path.touch()
    monkeypatch.setattr(
        "hand_benchmark.benchmark_inference.read_rfdetr_checkpoint_metadata",
        lambda _: {
            "checkpoint_sha256": "abc123",
            "model_name": "RFDETRLarge",
            "rfdetr_version": "1.8.3",
            "class_names": ["hand_left", "hand_right"],
            "classifier_slot_count": 3,
            "epochs": 8,
            "group_detr": 13,
            "num_queries": 5,
            "num_queries_source": "test",
            "num_select": 5,
        },
    )
    monkeypatch.setattr(
        "hand_benchmark.benchmark_inference._load_rfdetr_model",
        lambda *_: object(),
    )
    result = SimpleNamespace(
        xyxy=np.asarray([[0, 2, 20, 30], [70, 10, 110, 45]]),
        confidence=np.asarray([0.9, 0.8]),
        class_id=np.asarray([1, 2]),
    )
    monkeypatch.setattr(
        "hand_benchmark.benchmark_inference._predict_rfdetr_batch",
        lambda *_: ([result], 12.0),
    )
    predictions_path = tmp_path / "predictions.jsonl"

    counts = predict_rfdetr_frames(
        frames_dir=frames_dir,
        frame_metadata_path=frame_metadata_path,
        class_schema_dataset_root=tmp_path / "schema",
        weights_path=weights_path,
        output_path=predictions_path,
        confidence=0.25,
    )

    assert counts == (1, 2, 0)
    prediction = json.loads(predictions_path.read_text())
    assert [item["category"] for item in prediction["detections"]] == [
        "left_hand",
        "right_hand",
    ]
    assert prediction["model_config"]["class_slot_mapping"] == {
        "0": None,
        "1": "left_hand",
        "2": "right_hand",
    }
    export = export_roboflow_yolo(
        frames_dir,
        frame_metadata_path,
        predictions_path,
        tmp_path / "roboflow",
    )
    assert export.image_count == 1
    assert export.detection_count == 2
    manifest = json.loads((tmp_path / "roboflow/manifest.jsonl").read_text())
    assert manifest["prediction_model"] == "rfdetr-checkpoint-best-total"
    assert manifest["prediction_threshold"] == 0.25
    assert manifest["checkpoint_sha256"] == "abc123"
