"""Compare immutable COCO revisions after manual Roboflow annotation review."""

from __future__ import annotations

import csv
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from hand_benchmark.coco_dataset import (
    SPLIT_NAMES,
    CocoAnnotation,
    load_coco_split,
    stable_image_key,
)
from hand_benchmark.evaluation import iou_xyxy


@dataclass(frozen=True)
class RevisionChange:
    """One annotation or split change between two COCO revisions."""

    stable_image_key: str
    old_file_name: str | None
    new_file_name: str | None
    old_split: str | None
    new_split: str | None
    change_type: str
    old_category: str | None
    new_category: str | None
    old_bbox_xyxy: list[float] | None
    new_bbox_xyxy: list[float] | None
    iou: float | None


def compare_coco_revisions(
    *,
    old_dataset_root: Path,
    new_dataset_root: Path,
    output_dir: Path,
    unchanged_iou: float = 0.95,
    related_iou: float = 0.5,
) -> list[RevisionChange]:
    """Write added, removed, relabeled, adjusted, and split-move differences."""
    if not 0 <= related_iou <= unchanged_iou <= 1:
        raise ValueError("Require 0 <= related IoU <= unchanged IoU <= 1")
    old_images = _revision_images(old_dataset_root)
    new_images = _revision_images(new_dataset_root)
    changes: list[RevisionChange] = []
    for image_key in sorted(set(old_images) | set(new_images)):
        old_record = old_images.get(image_key)
        new_record = new_images.get(image_key)
        if old_record is None:
            changes.extend(_image_only_changes(image_key, new_record, "image_added"))
            continue
        if new_record is None:
            changes.extend(_image_only_changes(image_key, old_record, "image_removed"))
            continue
        old_split, old_file_name, old_annotations = old_record
        new_split, new_file_name, new_annotations = new_record
        if old_split != new_split:
            changes.append(
                RevisionChange(
                    image_key,
                    old_file_name,
                    new_file_name,
                    old_split,
                    new_split,
                    "split_moved",
                    None,
                    None,
                    None,
                    None,
                    None,
                )
            )
        changes.extend(
            _annotation_changes(
                image_key,
                old_split,
                new_split,
                old_file_name,
                new_file_name,
                old_annotations,
                new_annotations,
                unchanged_iou,
                related_iou,
            )
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    serialized = [asdict(change) for change in changes]
    (output_dir / "revision_diff.json").write_text(
        json.dumps(serialized, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_csv(output_dir / "revision_diff.csv", serialized)
    summary = {
        "old_dataset_root": old_dataset_root.resolve().as_posix(),
        "new_dataset_root": new_dataset_root.resolve().as_posix(),
        "unchanged_iou": unchanged_iou,
        "related_iou": related_iou,
        "change_counts": dict(Counter(change.change_type for change in changes)),
        "total_changes": len(changes),
    }
    (output_dir / "revision_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return changes


def _revision_images(
    dataset_root: Path,
) -> dict[str, tuple[str, str, list[CocoAnnotation]]]:
    records: dict[str, tuple[str, str, list[CocoAnnotation]]] = {}
    for split_name in SPLIT_NAMES:
        split = load_coco_split(dataset_root, split_name)
        for image in split.images:
            image_key = stable_image_key(image.file_name)
            if image_key in records:
                raise ValueError(f"Duplicate stable image key: {image_key}")
            records[image_key] = (
                split_name,
                image.file_name,
                split.annotations_by_image_id[image.id],
            )
    return records


def _annotation_changes(
    image_key: str,
    old_split: str,
    new_split: str,
    old_file_name: str,
    new_file_name: str,
    old_annotations: list[CocoAnnotation],
    new_annotations: list[CocoAnnotation],
    unchanged_iou: float,
    related_iou: float,
) -> list[RevisionChange]:
    remaining_old = list(old_annotations)
    remaining_new = list(new_annotations)
    changes: list[RevisionChange] = []

    for old_annotation in list(remaining_old):
        best_new, overlap = _best_annotation(
            old_annotation,
            remaining_new,
            require_same_category=True,
        )
        if best_new is not None and overlap >= unchanged_iou:
            remaining_old.remove(old_annotation)
            remaining_new.remove(best_new)

    for old_annotation in list(remaining_old):
        best_new, overlap = _best_annotation(
            old_annotation,
            remaining_new,
            require_same_category=False,
        )
        if best_new is None or overlap < related_iou:
            continue
        change_type = (
            "bbox_adjusted"
            if old_annotation.category == best_new.category
            else "relabeled"
        )
        changes.append(
            _change(
                image_key,
                old_split,
                new_split,
                old_file_name,
                new_file_name,
                change_type,
                old_annotation,
                best_new,
                overlap,
            )
        )
        remaining_old.remove(old_annotation)
        remaining_new.remove(best_new)

    changes.extend(
        _change(
            image_key,
            old_split,
            new_split,
            old_file_name,
            new_file_name,
            "annotation_removed",
            annotation,
            None,
            None,
        )
        for annotation in remaining_old
    )
    changes.extend(
        _change(
            image_key,
            old_split,
            new_split,
            old_file_name,
            new_file_name,
            "annotation_added",
            None,
            annotation,
            None,
        )
        for annotation in remaining_new
    )
    return changes


def _best_annotation(
    annotation: CocoAnnotation,
    candidates: list[CocoAnnotation],
    *,
    require_same_category: bool,
) -> tuple[CocoAnnotation | None, float]:
    best = None
    best_iou = 0.0
    for candidate in candidates:
        if require_same_category and annotation.category != candidate.category:
            continue
        overlap = iou_xyxy(annotation.bbox_xyxy, candidate.bbox_xyxy)
        if overlap > best_iou:
            best = candidate
            best_iou = overlap
    return best, best_iou


def _change(
    image_key: str,
    old_split: str,
    new_split: str,
    old_file_name: str,
    new_file_name: str,
    change_type: str,
    old_annotation: CocoAnnotation | None,
    new_annotation: CocoAnnotation | None,
    overlap: float | None,
) -> RevisionChange:
    return RevisionChange(
        image_key,
        old_file_name,
        new_file_name,
        old_split,
        new_split,
        change_type,
        old_annotation.category if old_annotation else None,
        new_annotation.category if new_annotation else None,
        old_annotation.bbox_xyxy if old_annotation else None,
        new_annotation.bbox_xyxy if new_annotation else None,
        round(overlap, 6) if overlap is not None else None,
    )


def _image_only_changes(
    image_key: str,
    record: tuple[str, str, list[CocoAnnotation]],
    change_type: str,
) -> list[RevisionChange]:
    split_name, file_name, annotations = record
    annotation_values: list[CocoAnnotation | None] = annotations or [None]
    return [
        RevisionChange(
            image_key,
            file_name if change_type == "image_removed" else None,
            file_name if change_type == "image_added" else None,
            split_name if change_type == "image_removed" else None,
            split_name if change_type == "image_added" else None,
            change_type,
            (
                annotation.category
                if annotation is not None and change_type == "image_removed"
                else None
            ),
            (
                annotation.category
                if annotation is not None and change_type == "image_added"
                else None
            ),
            (
                annotation.bbox_xyxy
                if annotation is not None and change_type == "image_removed"
                else None
            ),
            (
                annotation.bbox_xyxy
                if annotation is not None and change_type == "image_added"
                else None
            ),
            None,
        )
        for annotation in annotation_values
    ]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
