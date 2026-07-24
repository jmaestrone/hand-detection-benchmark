"""Tests for immutable COCO dataset import and validation."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from hand_benchmark.coco_dataset import (
    import_coco_dataset,
    stable_image_key,
    validate_coco_dataset,
)


def _write_split(
    root: Path,
    split: str,
    recording_id: str,
    *,
    unused_category_annotations: bool = False,
) -> None:
    split_dir = root / split
    split_dir.mkdir(parents=True)
    file_name = f"{recording_id}_frame000000_0000000000ms_jpg.rf.hash-{split}.jpg"
    (split_dir / file_name).write_bytes(b"image")
    annotations = [
        {
            "id": 1,
            "image_id": 1,
            "category_id": 1,
            "bbox": [1, 2, 5, 6],
            "area": 30,
            "iscrowd": 0,
        }
    ]
    if unused_category_annotations:
        annotations.append(
            {
                "id": 2,
                "image_id": 1,
                "category_id": 0,
                "bbox": [1, 1, 2, 2],
                "area": 4,
                "iscrowd": 0,
            }
        )
    payload = {
        "images": [{"id": 1, "file_name": file_name, "width": 20, "height": 20}],
        "annotations": annotations,
        "categories": [
            {"id": 0, "name": "project-placeholder"},
            {"id": 1, "name": "left_hand"},
            {"id": 2, "name": "right_hand"},
        ],
    }
    (split_dir / "_annotations.coco.json").write_text(json.dumps(payload))


def _fixture_dataset(root: Path) -> None:
    for split, recording in (
        ("train", "recording-train"),
        ("valid", "recording-valid"),
        ("test", "recording-test"),
    ):
        _write_split(root, split, recording)


def test_import_validates_and_records_provenance(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    _fixture_dataset(source_dir)
    archive_path = tmp_path / "dataset.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        for path in source_dir.rglob("*"):
            if path.is_file():
                archive.write(path, path.relative_to(source_dir))

    output_dir = tmp_path / "imported"
    result = import_coco_dataset(archive_path, output_dir)
    repeated = import_coco_dataset(archive_path, output_dir)

    assert result.archive_sha256 == repeated.archive_sha256
    assert result.split_summaries["train"].image_count == 1
    assert result.split_summaries["train"].category_counts == {
        "left_hand": 1,
        "right_hand": 0,
    }
    assert json.loads((output_dir / "source.json").read_text())["dataset_role"] == (
        "development-audit"
    )


def test_validation_rejects_recording_leakage(tmp_path: Path) -> None:
    _write_split(tmp_path, "train", "shared-recording")
    _write_split(tmp_path, "valid", "shared-recording")
    _write_split(tmp_path, "test", "test-recording")

    with pytest.raises(ValueError, match="Recording leakage"):
        validate_coco_dataset(tmp_path)


def test_validation_rejects_annotations_in_placeholder_category(
    tmp_path: Path,
) -> None:
    _write_split(
        tmp_path,
        "train",
        "train-recording",
        unused_category_annotations=True,
    )
    _write_split(tmp_path, "valid", "valid-recording")
    _write_split(tmp_path, "test", "test-recording")

    with pytest.raises(ValueError, match="Unsupported categories"):
        validate_coco_dataset(tmp_path)


def test_stable_image_key_removes_roboflow_revision_hash() -> None:
    assert (
        stable_image_key("recording_frame000030_0000001000ms_jpg.rf.abc123.jpg")
        == "recording_frame000030_0000001000ms.jpg"
    )
