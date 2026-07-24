"""WiLoR detector download and per-frame hand pre-label inference helpers."""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import urllib.request
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from hand_benchmark.config import DEFAULT_WILOR_DETECTOR_URL

LEFT_HAND_CATEGORY = "left_hand"
RIGHT_HAND_CATEGORY = "right_hand"
WILOR_CLASS_TO_CATEGORY = {0: LEFT_HAND_CATEGORY, 1: RIGHT_HAND_CATEGORY}


@dataclass(frozen=True)
class DetectorDownloadResult:
    """Provenance for a locally cached WiLoR detector weight file."""

    detector_path: str
    source_url: str
    sha256: str
    byte_count: int


@dataclass(frozen=True)
class HandDetection:
    """One WiLoR detector pre-label in source-image coordinates."""

    category: str
    category_id: int
    confidence: float
    bbox_xyxy: list[float]


@dataclass(frozen=True)
class FramePrediction:
    """Raw WiLoR detector output for one provenance-linked extracted frame."""

    file_name: str
    source_video: str
    source_mcap_path: str
    source_mcap_stem: str
    video_topic: str
    frame_index: int
    timestamp_seconds: float
    width: int
    height: int
    model_name: str
    confidence_threshold: float
    detections: list[HandDetection]


def download_wilor_detector(
    detector_path: Path,
    metadata_path: Path,
    overwrite: bool = False,
    source_url: str = DEFAULT_WILOR_DETECTOR_URL,
) -> DetectorDownloadResult:
    """Download the upstream WiLoR detector and write reproducibility metadata."""
    if detector_path.exists() and not overwrite:
        if metadata_path.exists():
            return DetectorDownloadResult(
                **json.loads(metadata_path.read_text(encoding="utf-8"))
            )
        raise FileExistsError(
            f"Detector exists without metadata; pass --overwrite: {detector_path}"
        )
    detector_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=detector_path.parent, delete=False
    ) as temporary_file:
        temporary_path = Path(temporary_file.name)
    try:
        with (
            urllib.request.urlopen(source_url) as response,
            temporary_path.open("wb") as output,
        ):
            shutil.copyfileobj(response, output)
        result = DetectorDownloadResult(
            detector_path=str(detector_path),
            source_url=source_url,
            sha256=file_sha256(temporary_path),
            byte_count=temporary_path.stat().st_size,
        )
        temporary_path.replace(detector_path)
        metadata_path.write_text(
            json.dumps(asdict(result), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return result
    finally:
        temporary_path.unlink(missing_ok=True)


def predict_wilor_frames(
    frames_dir: Path,
    frame_metadata_path: Path,
    detector_path: Path,
    output_path: Path,
    confidence: float,
    device: str,
    batch_size: int,
    limit: int | None = None,
    preview_dir: Path | None = None,
    max_previews: int = 20,
) -> tuple[int, int, int]:
    """Run WiLoR YOLO pre-labeling over extracted frames and write JSONL rows."""
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("--confidence must be between 0 and 1")
    if batch_size < 1:
        raise ValueError("--batch-size must be at least 1")
    if not detector_path.is_file():
        raise ValueError(f"Missing WiLoR detector weights: {detector_path}")
    records = read_jsonl(frame_metadata_path)
    if not records:
        raise ValueError(f"No frame metadata found at {frame_metadata_path}")
    if limit is not None:
        records = records[:limit]
    frame_paths = [frames_dir / str(record["output_path"]) for record in records]
    missing_frames = [path for path in frame_paths if not path.is_file()]
    if missing_frames:
        raise ValueError(
            f"Frame metadata references missing image: {missing_frames[0]}"
        )

    from ultralytics import YOLO

    model = YOLO(str(detector_path))
    validate_wilor_class_names(model.names)
    resolved_device = resolve_device(device)
    predictions: list[FramePrediction] = []
    preview_count = 0
    for record_batch, path_batch in batched(
        list(zip(records, frame_paths, strict=True)), batch_size
    ):
        results = model.predict(
            source=[str(path) for path in path_batch],
            conf=confidence,
            device=resolved_device,
            verbose=False,
        )
        for record, frame_path, result in zip(
            record_batch, path_batch, results, strict=True
        ):
            detections = detections_from_result(
                result, int(record["width"]), int(record["height"])
            )
            prediction = FramePrediction(
                file_name=str(record["file_name"]),
                source_video=str(record["source_video"]),
                source_mcap_path=str(record["source_mcap_path"]),
                source_mcap_stem=str(record["source_mcap_stem"]),
                video_topic=str(record["video_topic"]),
                frame_index=int(record["frame_index"]),
                timestamp_seconds=float(record["timestamp_seconds"]),
                width=int(record["width"]),
                height=int(record["height"]),
                model_name="wilor-detector",
                confidence_threshold=confidence,
                detections=detections,
            )
            predictions.append(prediction)
            if preview_dir is not None and preview_count < max_previews:
                write_preview(
                    frame_path, prediction, preview_dir / prediction.file_name
                )
                preview_count += 1
    write_jsonl(
        output_path,
        (frame_prediction_to_record(prediction) for prediction in predictions),
    )
    return (
        len(predictions),
        sum(len(item.detections) for item in predictions),
        preview_count,
    )


def validate_wilor_class_names(model_names: Any) -> None:
    """Reject a weight file whose ordered class mapping is not WiLoR left/right."""
    names = (
        model_names if isinstance(model_names, dict) else dict(enumerate(model_names))
    )
    normalized = {
        int(index): str(name).lower().replace("_", " ").strip()
        for index, name in names.items()
    }
    expected = {0: {"left", "left hand"}, 1: {"right", "right hand"}}
    if any(normalized.get(index) not in allowed for index, allowed in expected.items()):
        raise ValueError(f"Unexpected WiLoR class mapping: {normalized}")


def detections_from_result(result: Any, width: int, height: int) -> list[HandDetection]:
    """Convert one Ultralytics result to clipped left/right pre-label rows."""
    if result.boxes is None or len(result.boxes) == 0:
        return []
    boxes = result.boxes.xyxy.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy()
    confidences = result.boxes.conf.cpu().numpy()
    detections: list[HandDetection] = []
    for box, class_id, confidence in zip(boxes, classes, confidences, strict=True):
        category_id = int(class_id)
        if category_id not in WILOR_CLASS_TO_CATEGORY:
            raise ValueError(f"Unexpected WiLoR class id: {category_id}")
        x1, y1, x2, y2 = np.asarray(box, dtype=float)
        clipped = [
            max(0.0, min(x1, width)),
            max(0.0, min(y1, height)),
            max(0.0, min(x2, width)),
            max(0.0, min(y2, height)),
        ]
        if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
            continue
        detections.append(
            HandDetection(
                WILOR_CLASS_TO_CATEGORY[category_id],
                category_id,
                float(confidence),
                clipped,
            )
        )
    return detections


def resolve_device(device: str) -> str:
    """Resolve the benchmark's portable device option for Ultralytics."""
    if device != "auto":
        if device not in {"cpu", "cuda", "mps"}:
            raise ValueError("--device must be auto, cpu, cuda, or mps")
        return device
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def frame_prediction_to_record(prediction: FramePrediction) -> dict[str, Any]:
    """Serialize a prediction while keeping detections as portable JSON objects."""
    return {
        **asdict(prediction),
        "detections": [asdict(item) for item in prediction.detections],
    }


def write_preview(
    frame_path: Path, prediction: FramePrediction, output_path: Path
) -> None:
    """Write an annotated preview image for manual detector sanity checks."""
    import cv2

    image = cv2.imread(str(frame_path))
    if image is None:
        raise ValueError(f"Could not read frame for preview: {frame_path}")
    colors = {LEFT_HAND_CATEGORY: (255, 180, 100), RIGHT_HAND_CATEGORY: (100, 100, 255)}
    for detection in prediction.detections:
        x1, y1, x2, y2 = (round(value) for value in detection.bbox_xyxy)
        color = colors[detection.category]
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            image,
            f"{detection.category} {detection.confidence:.2f}",
            (x1, max(16, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), image):
        raise RuntimeError(f"Could not write preview: {output_path}")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read JSONL records from the benchmark's existing metadata format."""
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    """Write prediction records as newline-delimited JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def file_sha256(path: Path) -> str:
    """Return the SHA-256 digest of a model artifact."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def batched(
    values: list[tuple[dict[str, Any], Path]], batch_size: int
) -> Iterable[tuple[list[dict[str, Any]], list[Path]]]:
    """Yield parallel metadata and image-path batches."""
    for start in range(0, len(values), batch_size):
        batch = values[start : start + batch_size]
        yield [record for record, _ in batch], [path for _, path in batch]
