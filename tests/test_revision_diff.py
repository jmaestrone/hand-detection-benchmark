"""Tests for immutable COCO annotation revision comparisons."""

from __future__ import annotations

import json
from pathlib import Path

from hand_benchmark.revision_diff import compare_coco_revisions


def _write_revision(root: Path, hash_value: str, left_box: list[int]) -> None:
    for index, split in enumerate(("train", "valid", "test"), start=1):
        split_dir = root / split
        split_dir.mkdir(parents=True)
        file_name = (
            f"recording-{split}_frame000000_0000000000ms_jpg.rf.{hash_value}.jpg"
        )
        (split_dir / file_name).write_bytes(b"image")
        payload = {
            "images": [
                {
                    "id": index,
                    "file_name": file_name,
                    "width": 100,
                    "height": 100,
                }
            ],
            "annotations": [
                {
                    "id": index,
                    "image_id": index,
                    "category_id": 1,
                    "bbox": left_box if split == "train" else [0, 0, 10, 10],
                    "area": left_box[2] * left_box[3] if split == "train" else 100,
                    "iscrowd": 0,
                }
            ],
            "categories": [
                {"id": 0, "name": "unused"},
                {"id": 1, "name": "left_hand"},
                {"id": 2, "name": "right_hand"},
            ],
        }
        (split_dir / "_annotations.coco.json").write_text(json.dumps(payload))


def test_revision_diff_matches_across_roboflow_hashes_and_reports_adjustment(
    tmp_path: Path,
) -> None:
    old_root = tmp_path / "old"
    new_root = tmp_path / "new"
    _write_revision(old_root, "oldhash", [0, 0, 10, 10])
    _write_revision(new_root, "newhash", [1, 0, 10, 10])

    changes = compare_coco_revisions(
        old_dataset_root=old_root,
        new_dataset_root=new_root,
        output_dir=tmp_path / "report",
        unchanged_iou=0.95,
        related_iou=0.5,
    )

    assert [change.change_type for change in changes] == ["bbox_adjusted"]
    assert changes[0].stable_image_key.endswith("0000000000ms.jpg")
