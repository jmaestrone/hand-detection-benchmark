"""Run WiLoR and RF-DETR over reviewed COCO splits with normalized outputs."""

from __future__ import annotations

import json
import statistics
from collections.abc import Callable, Iterable
from pathlib import Path
from time import perf_counter
from typing import Any

import cv2
import numpy as np

from hand_benchmark.benchmark_predictions import (
    CANONICAL_CLASS_IDS,
    NormalizedDetection,
    NormalizedPrediction,
    canonical_category,
    prediction_provenance,
    validate_ordered_class_names,
    write_predictions,
)
from hand_benchmark.coco_dataset import (
    SPLIT_NAMES,
    CocoImage,
    CocoSplit,
    load_coco_split,
)
from hand_benchmark.wilor import (
    file_sha256,
    read_jsonl,
    resolve_device,
    validate_wilor_class_names,
)


def predict_wilor_coco(
    *,
    dataset_root: Path,
    detector_path: Path,
    output_root: Path,
    split_names: Iterable[str] = SPLIT_NAMES,
    confidence: float = 0.005,
    device: str = "auto",
    batch_size: int = 8,
    limit: int | None = None,
    preview_dir: Path | None = None,
    max_previews: int = 20,
) -> dict[str, Path]:
    """Run the WiLoR YOLO checkpoint over one or more COCO splits."""
    _validate_prediction_options(
        detector_path, confidence, batch_size, limit, max_previews
    )
    from ultralytics import YOLO

    model = YOLO(str(detector_path))
    validate_wilor_class_names(model.names)
    raw_names = _ordered_names(model.names)
    resolved_device = resolve_device(device)
    checkpoint_sha256 = file_sha256(detector_path)

    def predict_batch(images: list[CocoImage]) -> tuple[list[Any], float]:
        started_at = perf_counter()
        results = model.predict(
            source=[str(image.path) for image in images],
            conf=confidence,
            device=resolved_device,
            verbose=False,
        )
        return list(results), (perf_counter() - started_at) * 1000

    def normalize(result: Any, image: CocoImage) -> list[NormalizedDetection]:
        if result.boxes is None or len(result.boxes) == 0:
            return []
        return _normalize_arrays(
            boxes=result.boxes.xyxy.cpu().numpy(),
            confidences=result.boxes.conf.cpu().numpy(),
            class_ids=result.boxes.cls.cpu().numpy(),
            raw_names=raw_names,
            image=image,
        )

    return _predict_splits(
        dataset_root=dataset_root,
        output_root=output_root,
        model_name="wilor-yolo",
        model_config={
            "backend": "ultralytics",
            "checkpoint": detector_path.name,
            "class_names": raw_names,
        },
        checkpoint_sha256=checkpoint_sha256,
        resolved_device=resolved_device,
        confidence=confidence,
        batch_size=batch_size,
        split_names=split_names,
        limit=limit,
        preview_dir=preview_dir,
        max_previews=max_previews,
        predict_batch=predict_batch,
        normalize=normalize,
    )


def predict_rfdetr_coco(
    *,
    dataset_root: Path,
    weights_path: Path,
    output_root: Path,
    split_names: Iterable[str] = SPLIT_NAMES,
    confidence: float = 0.005,
    device: str = "auto",
    batch_size: int = 4,
    limit: int | None = None,
    preview_dir: Path | None = None,
    max_previews: int = 20,
) -> dict[str, Path]:
    """Run a self-describing RF-DETR checkpoint over one or more COCO splits."""
    _validate_prediction_options(
        weights_path, confidence, batch_size, limit, max_previews
    )
    checkpoint_metadata = read_rfdetr_checkpoint_metadata(weights_path)
    raw_names = checkpoint_metadata["class_names"]
    validate_ordered_class_names(raw_names)
    selected_split_names = tuple(split_names)
    split_mappings = {
        split_name: _rfdetr_class_slot_mapping(
            load_coco_split(dataset_root, split_name),
            raw_names,
            checkpoint_metadata["classifier_slot_count"],
        )
        for split_name in selected_split_names
    }
    class_slot_mapping = next(iter(split_mappings.values()))
    if any(mapping != class_slot_mapping for mapping in split_mappings.values()):
        raise ValueError("RF-DETR class-slot mapping differs between COCO splits")
    resolved_device = resolve_device(device)

    model = _load_rfdetr_model(weights_path, resolved_device, checkpoint_metadata)

    def predict_batch(images: list[CocoImage]) -> tuple[list[Any], float]:
        return _predict_rfdetr_batch(model, images, confidence)

    def normalize(result: Any, image: CocoImage) -> list[NormalizedDetection]:
        boxes = np.asarray(getattr(result, "xyxy", []))
        confidences = np.asarray(getattr(result, "confidence", []))
        class_ids = np.asarray(getattr(result, "class_id", []))
        return _normalize_arrays(
            boxes=boxes,
            confidences=confidences,
            class_ids=class_ids,
            raw_names=raw_names,
            image=image,
            class_id_to_name=class_slot_mapping,
        )

    return _predict_splits(
        dataset_root=dataset_root,
        output_root=output_root,
        model_name="rfdetr-checkpoint-best-total",
        model_config={
            "backend": "rfdetr",
            "checkpoint": weights_path.name,
            "class_slot_mapping": {
                str(class_id): raw_name
                for class_id, raw_name in class_slot_mapping.items()
            },
            **checkpoint_metadata,
        },
        checkpoint_sha256=checkpoint_metadata["checkpoint_sha256"],
        resolved_device=resolved_device,
        confidence=confidence,
        batch_size=batch_size,
        split_names=selected_split_names,
        limit=limit,
        preview_dir=preview_dir,
        max_previews=max_previews,
        predict_batch=predict_batch,
        normalize=normalize,
    )


def predict_rfdetr_frames(
    *,
    frames_dir: Path,
    frame_metadata_path: Path,
    class_schema_dataset_root: Path,
    weights_path: Path,
    output_path: Path,
    confidence: float = 0.25,
    device: str = "auto",
    batch_size: int = 4,
    limit: int | None = None,
    preview_dir: Path | None = None,
    max_previews: int = 20,
) -> tuple[int, int, int]:
    """Pre-label every extracted frame with the reviewed RF-DETR checkpoint."""
    _validate_prediction_options(
        weights_path, confidence, batch_size, limit, max_previews
    )
    if not frame_metadata_path.is_file():
        raise ValueError(f"Frame metadata does not exist: {frame_metadata_path}")
    frame_records = read_jsonl(frame_metadata_path)
    if not frame_records:
        raise ValueError(f"No frame metadata found at {frame_metadata_path}")
    if limit is not None:
        frame_records = frame_records[:limit]
    images = _frame_images_from_metadata(frames_dir, frame_records)

    checkpoint_metadata = read_rfdetr_checkpoint_metadata(weights_path)
    raw_names = checkpoint_metadata["class_names"]
    class_slot_mapping = _rfdetr_class_slot_mapping(
        load_coco_split(class_schema_dataset_root, "train"),
        raw_names,
        checkpoint_metadata["classifier_slot_count"],
    )
    resolved_device = resolve_device(device)
    model = _load_rfdetr_model(weights_path, resolved_device, checkpoint_metadata)
    model_config = {
        "backend": "rfdetr",
        "checkpoint": weights_path.name,
        "class_schema_dataset_root": (class_schema_dataset_root.resolve().as_posix()),
        "class_slot_mapping": {
            str(class_id): raw_name for class_id, raw_name in class_slot_mapping.items()
        },
        **checkpoint_metadata,
    }

    predictions: list[NormalizedPrediction] = []
    inference_times_ms: list[float] = []
    preview_count = 0
    for batch in _batched(images, batch_size):
        results, elapsed_ms = _predict_rfdetr_batch(model, batch, confidence)
        if len(results) != len(batch):
            raise ValueError(
                f"RF-DETR returned {len(results)} results for {len(batch)} frames"
            )
        per_image_ms = elapsed_ms / len(batch)
        for image, result in zip(batch, results, strict=True):
            detections = _normalize_arrays(
                boxes=np.asarray(getattr(result, "xyxy", [])),
                confidences=np.asarray(getattr(result, "confidence", [])),
                class_ids=np.asarray(getattr(result, "class_id", [])),
                raw_names=raw_names,
                image=image,
                class_id_to_name=class_slot_mapping,
            )
            record = frame_records[image.id - 1]
            predictions.append(
                NormalizedPrediction(
                    file_name=image.file_name,
                    split="frames",
                    width=image.width,
                    height=image.height,
                    source_recording=str(record["source_mcap_stem"]),
                    timestamp_seconds=float(record["timestamp_seconds"]),
                    model_name="rfdetr-checkpoint-best-total",
                    model_config=model_config,
                    checkpoint_sha256=checkpoint_metadata["checkpoint_sha256"],
                    device=resolved_device,
                    inference_floor=confidence,
                    timing_ms={"inference": round(per_image_ms, 4)},
                    detections=detections,
                )
            )
            inference_times_ms.append(per_image_ms)
            if preview_dir is not None and preview_count < max_previews:
                _write_preview(
                    image.path,
                    detections,
                    preview_dir / image.file_name,
                )
                preview_count += 1

    write_predictions(output_path, predictions)
    _write_latency(
        output_path.with_suffix(".latency.json"),
        model_name="rfdetr-checkpoint-best-total",
        split_name="frames",
        device=resolved_device,
        image_count=len(predictions),
        detection_count=sum(len(row.detections) for row in predictions),
        inference_times_ms=inference_times_ms,
        model_config=model_config,
        checkpoint_sha256=checkpoint_metadata["checkpoint_sha256"],
    )
    return (
        len(predictions),
        sum(len(prediction.detections) for prediction in predictions),
        preview_count,
    )


def read_rfdetr_checkpoint_metadata(weights_path: Path) -> dict[str, Any]:
    """Safely read reproducibility metadata embedded in an RF-DETR checkpoint."""
    import torch

    checkpoint = torch.load(weights_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise TypeError("RF-DETR checkpoint must contain a metadata dictionary")
    args = checkpoint.get("args")
    if not isinstance(args, dict):
        raise TypeError("RF-DETR checkpoint does not contain dictionary `args`")
    class_names = args.get("class_names")
    if not isinstance(class_names, list):
        raise TypeError("RF-DETR checkpoint does not contain `args.class_names`")
    validate_ordered_class_names([str(name) for name in class_names])
    model_state = checkpoint.get("model")
    if not isinstance(model_state, dict):
        raise TypeError("RF-DETR checkpoint does not contain dictionary `model`")
    classifier_bias = model_state.get("class_embed.bias")
    if classifier_bias is None or not hasattr(classifier_bias, "shape"):
        raise ValueError("RF-DETR checkpoint is missing class_embed.bias")
    classifier_slot_count = int(classifier_bias.shape[0])
    group_detr = int(args.get("group_detr", 1))
    if group_detr <= 0:
        raise ValueError("RF-DETR checkpoint has invalid `args.group_detr`")
    num_queries = args.get("num_queries")
    query_source = "checkpoint_args"
    if num_queries is None:
        refpoint_weight = model_state.get("refpoint_embed.weight")
        if refpoint_weight is None or not hasattr(refpoint_weight, "shape"):
            raise ValueError(
                "Cannot infer RF-DETR num_queries without refpoint_embed.weight"
            )
        query_rows = int(refpoint_weight.shape[0])
        if query_rows <= 0 or query_rows % group_detr:
            raise ValueError("RF-DETR query rows are incompatible with args.group_detr")
        num_queries = query_rows // group_detr
        query_source = "refpoint_embed.weight/group_detr"
    num_queries = int(num_queries)
    if num_queries <= 0:
        raise ValueError("RF-DETR checkpoint has invalid num_queries")
    return {
        "checkpoint_sha256": file_sha256(weights_path),
        "model_name": str(checkpoint.get("model_name", "unknown")),
        "rfdetr_version": str(checkpoint.get("rfdetr_version", "unknown")),
        "class_names": [str(name) for name in class_names],
        "classifier_slot_count": classifier_slot_count,
        "epochs": int(args["epochs"]) if args.get("epochs") is not None else None,
        "group_detr": group_detr,
        "num_queries": num_queries,
        "num_queries_source": query_source,
        "num_select": (
            int(args["num_select"]) if args.get("num_select") is not None else None
        ),
    }


def parse_split_names(raw_split: str) -> tuple[str, ...]:
    """Resolve the CLI's `all` value or one concrete split."""
    if raw_split == "all":
        return SPLIT_NAMES
    if raw_split not in SPLIT_NAMES:
        raise ValueError(f"--split must be all or one of: {', '.join(SPLIT_NAMES)}")
    return (raw_split,)


def _predict_splits(
    *,
    dataset_root: Path,
    output_root: Path,
    model_name: str,
    model_config: dict[str, Any],
    checkpoint_sha256: str,
    resolved_device: str,
    confidence: float,
    batch_size: int,
    split_names: Iterable[str],
    limit: int | None,
    preview_dir: Path | None,
    max_previews: int,
    predict_batch: Callable[[list[CocoImage]], tuple[list[Any], float]],
    normalize: Callable[[Any, CocoImage], list[NormalizedDetection]],
) -> dict[str, Path]:
    """Execute a normalized model adapter and persist each requested split."""
    output_paths: dict[str, Path] = {}
    for split_name in split_names:
        split = load_coco_split(dataset_root, split_name)
        images = split.images[:limit] if limit is not None else split.images
        predictions: list[NormalizedPrediction] = []
        inference_times_ms: list[float] = []
        preview_count = 0
        for batch in _batched(images, batch_size):
            results, elapsed_ms = predict_batch(batch)
            if len(results) != len(batch):
                raise ValueError(
                    f"{model_name} returned {len(results)} results for "
                    f"{len(batch)} images"
                )
            per_image_ms = elapsed_ms / len(batch)
            for image, result in zip(batch, results, strict=True):
                detections = normalize(result, image)
                recording, timestamp = prediction_provenance(image.file_name)
                prediction = NormalizedPrediction(
                    file_name=image.file_name,
                    split=split_name,
                    width=image.width,
                    height=image.height,
                    source_recording=recording,
                    timestamp_seconds=timestamp,
                    model_name=model_name,
                    model_config=model_config,
                    checkpoint_sha256=checkpoint_sha256,
                    device=resolved_device,
                    inference_floor=confidence,
                    timing_ms={"inference": round(per_image_ms, 4)},
                    detections=detections,
                )
                predictions.append(prediction)
                inference_times_ms.append(per_image_ms)
                if preview_dir is not None and preview_count < max_previews:
                    _write_preview(
                        image.path,
                        detections,
                        preview_dir / model_name / split_name / image.file_name,
                    )
                    preview_count += 1
        output_path = output_root / "predictions" / model_name / f"{split_name}.jsonl"
        write_predictions(output_path, predictions)
        _write_latency(
            output_root / "latency" / model_name / f"{split_name}.json",
            model_name=model_name,
            split_name=split_name,
            device=resolved_device,
            image_count=len(predictions),
            detection_count=sum(len(row.detections) for row in predictions),
            inference_times_ms=inference_times_ms,
            model_config=model_config,
            checkpoint_sha256=checkpoint_sha256,
        )
        output_paths[split_name] = output_path
    return output_paths


def _load_rfdetr_model(
    weights_path: Path,
    resolved_device: str,
    checkpoint_metadata: dict[str, Any],
) -> Any:
    """Load RF-DETR with architecture values recovered from its checkpoint."""
    try:
        from rfdetr import RFDETR
    except ImportError as error:
        raise ImportError(
            "RF-DETR inference requires `uv sync --extra rfdetr`"
        ) from error
    return RFDETR.from_checkpoint(
        str(weights_path),
        device=resolved_device,
        num_queries=checkpoint_metadata["num_queries"],
        num_select=checkpoint_metadata["num_select"],
    )


def _predict_rfdetr_batch(
    model: Any,
    images: list[CocoImage],
    confidence: float,
) -> tuple[list[Any], float]:
    """Read and infer one RF-DETR image batch while measuring model latency."""
    images_rgb = []
    for image in images:
        image_bgr = cv2.imread(str(image.path))
        if image_bgr is None:
            raise ValueError(f"Could not read image: {image.path}")
        images_rgb.append(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    started_at = perf_counter()
    results = model.predict(images_rgb, threshold=confidence)
    elapsed_ms = (perf_counter() - started_at) * 1000
    if not isinstance(results, list):
        results = [results]
    return results, elapsed_ms


def _frame_images_from_metadata(
    frames_dir: Path, frame_records: list[dict[str, Any]]
) -> list[CocoImage]:
    """Validate frame metadata and adapt it to the shared image contract."""
    images: list[CocoImage] = []
    file_names: set[str] = set()
    for image_id, record in enumerate(frame_records, start=1):
        file_name = str(record["file_name"])
        if file_name in file_names:
            raise ValueError(f"Duplicate frame metadata filename: {file_name}")
        image_path = frames_dir / str(record.get("output_path", file_name))
        if not image_path.is_file():
            raise ValueError(f"Frame metadata references missing image: {image_path}")
        width = int(record["width"])
        height = int(record["height"])
        if width <= 0 or height <= 0:
            raise ValueError(f"Invalid frame dimensions for {file_name}")
        images.append(
            CocoImage(
                id=image_id,
                file_name=file_name,
                width=width,
                height=height,
                path=image_path,
            )
        )
        file_names.add(file_name)
    return images


def _normalize_arrays(
    *,
    boxes: Any,
    confidences: Any,
    class_ids: Any,
    raw_names: list[str],
    image: CocoImage,
    emitted_class_names: Any | None = None,
    class_id_to_name: dict[int, str | None] | None = None,
) -> list[NormalizedDetection]:
    """Normalize detector arrays and clip boxes to source-image coordinates."""
    detections: list[NormalizedDetection] = []
    values = zip(boxes, confidences, class_ids, strict=True)
    for index, (box, confidence, raw_class_id) in enumerate(values):
        class_id = int(raw_class_id)
        if class_id_to_name is not None:
            if class_id not in class_id_to_name:
                raise ValueError(f"Unexpected detector class id: {class_id}")
            mapped_name = class_id_to_name[class_id]
            if mapped_name is None:
                continue
            raw_name = mapped_name
        elif emitted_class_names is not None:
            raw_name = str(emitted_class_names[index])
            if raw_name == "__background__":
                continue
        else:
            if class_id < 0 or class_id >= len(raw_names):
                raise ValueError(f"Unexpected detector class id: {class_id}")
            raw_name = raw_names[class_id]
        category = canonical_category(raw_name)
        x1, y1, x2, y2 = (float(value) for value in box)
        clipped = [
            max(0.0, min(x1, image.width)),
            max(0.0, min(y1, image.height)),
            max(0.0, min(x2, image.width)),
            max(0.0, min(y2, image.height)),
        ]
        if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
            continue
        detections.append(
            NormalizedDetection(
                category=category,
                category_id=CANONICAL_CLASS_IDS[category],
                confidence=round(float(confidence), 6),
                bbox_xyxy=[round(value, 4) for value in clipped],
                raw_class_id=class_id,
                raw_class_name=raw_name,
            )
        )
    return detections


def _rfdetr_class_slot_mapping(
    split: CocoSplit,
    raw_names: list[str],
    classifier_slot_count: int,
) -> dict[int, str | None]:
    """Reconstruct RF-DETR training slots, including unused COCO categories.

    RF-DETR remaps every declared category in custom COCO datasets, even a
    zero-annotation Roboflow parent category. Checkpoints only persist leaf
    class names, so the generic inference name mapping loses that slot offset.
    """
    if classifier_slot_count <= 0:
        raise ValueError("RF-DETR classifier must expose at least one slot")
    category_ids = sorted(split.category_id_to_name)
    if split.ignored_category_ids and len(category_ids) == classifier_slot_count:
        mapping = {
            slot: (
                split.category_id_to_name[category_id]
                if category_id not in split.ignored_category_ids
                else None
            )
            for slot, category_id in enumerate(category_ids)
        }
        mapped_names = [name for name in mapping.values() if name is not None]
        if mapped_names != ["left_hand", "right_hand"]:
            raise ValueError(
                "COCO category order does not map RF-DETR slots to left/right"
            )
        return mapping
    if classifier_slot_count == len(raw_names) + 1:
        return {
            **{index: raw_name for index, raw_name in enumerate(raw_names)},
            len(raw_names): None,
        }
    raise ValueError(
        "Cannot reconcile RF-DETR classifier slots with COCO categories: "
        f"slots={classifier_slot_count}, categories={category_ids}"
    )


def _ordered_names(raw_names: list[str] | dict[int, str]) -> list[str]:
    if isinstance(raw_names, dict):
        names = [str(raw_names[index]) for index in sorted(raw_names)]
    else:
        names = [str(name) for name in raw_names]
    validate_ordered_class_names(names)
    return names


def _validate_prediction_options(
    weights_path: Path,
    confidence: float,
    batch_size: int,
    limit: int | None,
    max_previews: int,
) -> None:
    if not weights_path.is_file():
        raise ValueError(f"Model checkpoint does not exist: {weights_path}")
    if not 0 <= confidence <= 1:
        raise ValueError("--confidence must be between 0 and 1")
    if batch_size <= 0:
        raise ValueError("--batch-size must be greater than 0")
    if limit is not None and limit <= 0:
        raise ValueError("--limit must be greater than 0")
    if max_previews < 0:
        raise ValueError("--max-previews must be non-negative")


def _batched(values: list[CocoImage], batch_size: int) -> Iterable[list[CocoImage]]:
    for start in range(0, len(values), batch_size):
        yield values[start : start + batch_size]


def _write_preview(
    image_path: Path,
    detections: list[NormalizedDetection],
    output_path: Path,
) -> None:
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not read preview image: {image_path}")
    colors = {"left_hand": (255, 180, 100), "right_hand": (100, 100, 255)}
    for detection in detections:
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
        raise ValueError(f"Could not write preview image: {output_path}")


def _write_latency(
    path: Path,
    *,
    model_name: str,
    split_name: str,
    device: str,
    image_count: int,
    detection_count: int,
    inference_times_ms: list[float],
    model_config: dict[str, Any],
    checkpoint_sha256: str,
) -> None:
    sorted_times = sorted(inference_times_ms)
    payload = {
        "model_name": model_name,
        "split": split_name,
        "device": device,
        "image_count": image_count,
        "detection_count": detection_count,
        "checkpoint_sha256": checkpoint_sha256,
        "model_config": model_config,
        "per_image_inference_ms": {
            "mean": round(statistics.fmean(sorted_times), 4) if sorted_times else 0.0,
            "median": round(statistics.median(sorted_times), 4)
            if sorted_times
            else 0.0,
            "p90": round(_percentile(sorted_times, 90), 4),
            "min": round(sorted_times[0], 4) if sorted_times else 0.0,
            "max": round(sorted_times[-1], 4) if sorted_times else 0.0,
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _percentile(sorted_values: list[float], percentile: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = (len(sorted_values) - 1) * percentile / 100
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = position - lower
    return (
        sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * fraction
    )
