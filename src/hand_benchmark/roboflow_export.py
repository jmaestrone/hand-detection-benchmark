"""Export WiLoR pre-labels as a Roboflow-importable YOLO detection dataset."""

from __future__ import annotations

import json
import os
import shutil
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hand_benchmark.wilor import LEFT_HAND_CATEGORY, RIGHT_HAND_CATEGORY

YOLO_CLASS_IDS = {LEFT_HAND_CATEGORY: 0, RIGHT_HAND_CATEGORY: 1}


@dataclass(frozen=True)
class RoboflowExportResult:
    """Counts and paths created by one Roboflow YOLO export."""

    image_count: int
    detection_count: int
    empty_label_count: int
    output_dir: Path


def export_roboflow_yolo(
    frames_dir: Path,
    frame_metadata_path: Path,
    predictions_path: Path,
    output_dir: Path,
    overwrite: bool = False,
) -> RoboflowExportResult:
    """Create a YOLO dataset folder from frame metadata and WiLoR predictions.

    Images are hard-linked where the filesystem permits, so the export remains
    uploadable without duplicating the raw frame pixels. Empty label files are
    written for frames where WiLoR found no hands; they are intentional negative
    examples, not missing labels.
    """
    frame_records = read_jsonl(frame_metadata_path)
    prediction_records = read_jsonl(predictions_path)
    validate_input_records(frame_records, prediction_records, frames_dir)
    ensure_output_is_writable(output_dir, overwrite)

    images_dir = output_dir / "images" / "train"
    labels_dir = output_dir / "labels" / "train"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    predictions_by_filename = {
        str(record["file_name"]): record for record in prediction_records
    }
    manifest_records: list[dict[str, Any]] = []
    detection_count = 0
    empty_label_count = 0
    for frame_record in sorted(
        frame_records, key=lambda record: str(record["file_name"])
    ):
        file_name = str(frame_record["file_name"])
        image_source = frames_dir / file_name
        image_target = images_dir / file_name
        link_or_copy_image(image_source, image_target)

        prediction_record = predictions_by_filename[file_name]
        label_target = labels_dir / f"{Path(file_name).stem}.txt"
        label_lines = yolo_label_lines(prediction_record)
        write_text(
            label_target,
            "\n".join(label_lines) + ("\n" if label_lines else ""),
            overwrite,
        )
        detection_count += len(label_lines)
        if not label_lines:
            empty_label_count += 1
        manifest_records.append(
            {
                "file_name": file_name,
                "image_path": (Path("images") / "train" / file_name).as_posix(),
                "label_path": (Path("labels") / "train" / label_target.name).as_posix(),
                "source_mcap_path": frame_record["source_mcap_path"],
                "source_mcap_stem": frame_record["source_mcap_stem"],
                "source_video": frame_record["source_video"],
                "timestamp_seconds": frame_record["timestamp_seconds"],
                "detection_count": len(label_lines),
            }
        )

    write_text(output_dir / "data.yaml", yolo_data_yaml(), overwrite)
    write_jsonl(output_dir / "manifest.jsonl", manifest_records, overwrite)
    return RoboflowExportResult(
        image_count=len(frame_records),
        detection_count=detection_count,
        empty_label_count=empty_label_count,
        output_dir=output_dir,
    )


def yolo_label_lines(prediction_record: dict[str, Any]) -> list[str]:
    """Convert one source-pixel prediction record to normalized YOLO labels."""
    width = int(prediction_record["width"])
    height = int(prediction_record["height"])
    if width <= 0 or height <= 0:
        raise ValueError(
            f"Prediction has invalid dimensions: {prediction_record['file_name']}"
        )
    lines: list[str] = []
    for detection in prediction_record["detections"]:
        category = str(detection["category"])
        if category not in YOLO_CLASS_IDS:
            raise ValueError(f"Unsupported prediction category {category!r}")
        x1, y1, x2, y2 = (float(value) for value in detection["bbox_xyxy"])
        x1, x2 = sorted((max(0.0, min(x1, width)), max(0.0, min(x2, width))))
        y1, y2 = sorted((max(0.0, min(y1, height)), max(0.0, min(y2, height))))
        box_width = x2 - x1
        box_height = y2 - y1
        if box_width <= 0 or box_height <= 0:
            raise ValueError(
                f"Prediction has an empty bounding box: {prediction_record['file_name']}"
            )
        center_x, box_width = serialized_yolo_axis(
            (x1 + x2) / 2 / width, box_width / width
        )
        center_y, box_height = serialized_yolo_axis(
            (y1 + y2) / 2 / height, box_height / height
        )
        lines.append(
            f"{YOLO_CLASS_IDS[category]} {center_x:.6f} {center_y:.6f} "
            f"{box_width:.6f} {box_height:.6f}"
        )
    return lines


def serialized_yolo_axis(center: float, span: float) -> tuple[float, float]:
    """Round one YOLO axis without allowing serialized edges outside [0, 1]."""
    rounded_center = round(center, 6)
    rounded_span = round(span, 6)
    precision_step = 0.000001
    while (
        rounded_center - rounded_span / 2 < 0 or rounded_center + rounded_span / 2 > 1
    ):
        rounded_span = round(rounded_span - precision_step, 6)
        if rounded_span <= 0:
            raise ValueError("YOLO box became empty after serialization")
    return rounded_center, rounded_span


def validate_input_records(
    frame_records: list[dict[str, Any]],
    prediction_records: list[dict[str, Any]],
    frames_dir: Path,
) -> None:
    """Require one prediction and one image for every frame before exporting."""
    if not frame_records:
        raise ValueError(f"No frame metadata records found in {frames_dir}")
    frame_names = [str(record["file_name"]) for record in frame_records]
    if len(set(frame_names)) != len(frame_names):
        raise ValueError("Frame metadata contains duplicate file names")
    prediction_names = [str(record["file_name"]) for record in prediction_records]
    if len(set(prediction_names)) != len(prediction_names):
        raise ValueError("Prediction metadata contains duplicate file names")
    missing_predictions = sorted(set(frame_names) - set(prediction_names))
    extra_predictions = sorted(set(prediction_names) - set(frame_names))
    if missing_predictions or extra_predictions:
        raise ValueError(
            "Frame and prediction records do not match: "
            f"missing={len(missing_predictions)}, extra={len(extra_predictions)}"
        )
    missing_images = [
        file_name for file_name in frame_names if not (frames_dir / file_name).is_file()
    ]
    if missing_images:
        raise ValueError(f"Frame images are missing: {len(missing_images)}")


def ensure_output_is_writable(output_dir: Path, overwrite: bool) -> None:
    """Protect an existing export until the caller explicitly requests replacement."""
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Export directory already contains files: {output_dir}. Pass --overwrite."
        )


def link_or_copy_image(source_path: Path, target_path: Path) -> None:
    """Hard-link a source image, falling back to a normal copy across filesystems."""
    if target_path.exists():
        if os.path.samefile(source_path, target_path):
            return
        raise FileExistsError(
            f"Export image already exists and differs from source: {target_path}"
        )
    try:
        os.link(source_path, target_path)
    except OSError:
        shutil.copy2(source_path, target_path)


def write_text(path: Path, content: str, overwrite: bool) -> None:
    """Write a generated text artifact without implicitly replacing one."""
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"Export artifact already exists: {path}. Pass --overwrite."
        )
    path.write_text(content, encoding="utf-8")


def yolo_data_yaml() -> str:
    """Return the stable two-class data.yaml required by the annotation schema."""
    return (
        "path: .\ntrain: images/train\nnc: 2\nnames:\n  0: left_hand\n  1: right_hand\n"
    )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read JSONL records and fail clearly when a required input is absent."""
    if not path.is_file():
        raise ValueError(f"Required JSONL file does not exist: {path}")
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def write_jsonl(path: Path, records: Iterable[dict[str, Any]], overwrite: bool) -> None:
    """Write a deterministic JSONL export manifest."""
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"Export artifact already exists: {path}. Pass --overwrite."
        )
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
