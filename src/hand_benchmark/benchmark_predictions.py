"""Shared normalized prediction records for hand-detector comparisons."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from hand_benchmark.coco_dataset import recording_id_from_file_name

LEFT_HAND = "left_hand"
RIGHT_HAND = "right_hand"
CANONICAL_CLASS_IDS = {LEFT_HAND: 0, RIGHT_HAND: 1}
CLASS_ALIASES = {
    "left": LEFT_HAND,
    "left hand": LEFT_HAND,
    "hand left": LEFT_HAND,
    "left_hand": LEFT_HAND,
    "hand_left": LEFT_HAND,
    "right": RIGHT_HAND,
    "right hand": RIGHT_HAND,
    "hand right": RIGHT_HAND,
    "right_hand": RIGHT_HAND,
    "hand_right": RIGHT_HAND,
}
TIMESTAMP_PATTERN = re.compile(r"_([0-9]{10})ms(?:_|\.|$)")


@dataclass(frozen=True)
class NormalizedDetection:
    """One canonical hand detection with retained raw model class metadata."""

    category: str
    category_id: int
    confidence: float
    bbox_xyxy: list[float]
    raw_class_id: int
    raw_class_name: str


@dataclass(frozen=True)
class NormalizedPrediction:
    """One model's detections and provenance for a COCO image."""

    file_name: str
    split: str
    width: int
    height: int
    source_recording: str
    timestamp_seconds: float | None
    model_name: str
    model_config: dict[str, Any]
    checkpoint_sha256: str
    device: str
    inference_floor: float
    timing_ms: dict[str, float] = field(default_factory=dict)
    detections: list[NormalizedDetection] = field(default_factory=list)


def canonical_category(raw_name: str) -> str:
    """Normalize a detector's left/right class name to the benchmark schema."""
    normalized = raw_name.lower().strip().replace("-", " ").replace("_", " ")
    normalized = " ".join(normalized.split())
    if normalized not in CLASS_ALIASES:
        raise ValueError(f"Unknown hand category alias: {raw_name!r}")
    return CLASS_ALIASES[normalized]


def validate_ordered_class_names(class_names: list[str] | dict[int, str]) -> list[str]:
    """Require a two-class left-then-right detector mapping."""
    if isinstance(class_names, dict):
        ordered = [str(class_names[index]) for index in sorted(class_names)]
    else:
        ordered = [str(name) for name in class_names]
    canonical = [canonical_category(name) for name in ordered]
    if canonical != [LEFT_HAND, RIGHT_HAND]:
        raise ValueError(
            "Expected ordered left/right hand classes; "
            f"found raw={ordered}, canonical={canonical}"
        )
    return canonical


def prediction_to_record(prediction: NormalizedPrediction) -> dict[str, Any]:
    """Convert a normalized prediction to a JSON-compatible dictionary."""
    return {
        **asdict(prediction),
        "detections": [asdict(detection) for detection in prediction.detections],
    }


def write_predictions(path: Path, predictions: Iterable[NormalizedPrediction]) -> None:
    """Write deterministic normalized prediction JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output:
        for prediction in predictions:
            output.write(
                json.dumps(prediction_to_record(prediction), sort_keys=True) + "\n"
            )


def read_prediction_records(path: Path) -> list[dict[str, Any]]:
    """Read normalized prediction records from JSONL."""
    if not path.is_file():
        raise ValueError(f"Prediction JSONL does not exist: {path}")
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def timestamp_from_file_name(file_name: str) -> float | None:
    """Recover the original sampled timestamp from a Roboflow filename."""
    match = TIMESTAMP_PATTERN.search(file_name)
    return int(match.group(1)) / 1000 if match else None


def prediction_provenance(file_name: str) -> tuple[str, float | None]:
    """Return recording and timestamp provenance encoded in an image filename."""
    return recording_id_from_file_name(file_name), timestamp_from_file_name(file_name)
