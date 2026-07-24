"""Tests for deterministic Roboflow YOLO pre-label export."""

import json
from pathlib import Path

import pytest

from hand_benchmark.roboflow_export import export_roboflow_yolo, yolo_label_lines


def test_yolo_label_lines_normalize_and_clip_boxes() -> None:
    lines = yolo_label_lines(
        {
            "file_name": "frame.jpg",
            "width": 100,
            "height": 50,
            "detections": [{"category": "left_hand", "bbox_xyxy": [-10, 10, 50, 60]}],
        }
    )

    assert lines == ["0 0.250000 0.600000 0.500000 0.800000"]


def test_yolo_label_lines_remain_in_bounds_after_serialization() -> None:
    lines = yolo_label_lines(
        {
            "file_name": "frame.jpg",
            "width": 1000,
            "height": 1000,
            "detections": [
                {"category": "right_hand", "bbox_xyxy": [868.447, 100, 1000, 900]}
            ],
        }
    )

    _, center_x, _, box_width, _ = lines[0].split()
    assert float(center_x) + float(box_width) / 2 <= 1.0


def test_export_creates_complete_yolo_import_folder(tmp_path: Path) -> None:
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    (frames_dir / "one.jpg").write_bytes(b"one")
    (frames_dir / "two.jpg").write_bytes(b"two")
    frame_metadata_path = tmp_path / "frames.jsonl"
    prediction_path = tmp_path / "predictions.jsonl"
    frame_metadata_path.write_text(
        "\n".join(
            json.dumps(record)
            for record in [
                {
                    "file_name": "one.jpg",
                    "source_mcap_path": "/one.mcap",
                    "source_mcap_stem": "one",
                    "source_video": "one.mp4",
                    "timestamp_seconds": 1.0,
                },
                {
                    "file_name": "two.jpg",
                    "source_mcap_path": "/two.mcap",
                    "source_mcap_stem": "two",
                    "source_video": "two.mp4",
                    "timestamp_seconds": 2.0,
                },
            ]
        )
        + "\n"
    )
    prediction_path.write_text(
        "\n".join(
            json.dumps(record)
            for record in [
                {
                    "file_name": "one.jpg",
                    "width": 100,
                    "height": 50,
                    "detections": [
                        {"category": "right_hand", "bbox_xyxy": [25, 10, 75, 40]}
                    ],
                },
                {"file_name": "two.jpg", "width": 100, "height": 50, "detections": []},
            ]
        )
        + "\n"
    )
    output_dir = tmp_path / "roboflow"

    result = export_roboflow_yolo(
        frames_dir, frame_metadata_path, prediction_path, output_dir
    )

    assert (result.image_count, result.detection_count, result.empty_label_count) == (
        2,
        1,
        1,
    )
    assert (output_dir / "images/train/one.jpg").read_bytes() == b"one"
    assert (
        output_dir / "labels/train/one.txt"
    ).read_text() == "1 0.500000 0.500000 0.500000 0.600000\n"
    assert (output_dir / "labels/train/two.txt").read_text() == ""
    assert "0: left_hand" in (output_dir / "data.yaml").read_text()
    assert len((output_dir / "manifest.jsonl").read_text().splitlines()) == 2

    with pytest.raises(FileExistsError, match="Pass --overwrite"):
        export_roboflow_yolo(
            frames_dir, frame_metadata_path, prediction_path, output_dir
        )
