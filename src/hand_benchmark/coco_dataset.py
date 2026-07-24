"""Import and validate immutable two-class COCO hand-detection datasets."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import tempfile
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath

SPLIT_NAMES = ("train", "valid", "test")
CANONICAL_CATEGORIES = ("left_hand", "right_hand")
ANNOTATIONS_FILE_NAME = "_annotations.coco.json"
ROBOFLOW_SUFFIX_PATTERN = re.compile(r"_jpg\.rf\.[^.]+\.jpg$", re.IGNORECASE)


@dataclass(frozen=True)
class CocoImage:
    """One image referenced by a COCO split."""

    id: int
    file_name: str
    width: int
    height: int
    path: Path


@dataclass(frozen=True)
class CocoAnnotation:
    """One validated COCO bounding-box annotation."""

    id: int
    image_id: int
    category_id: int
    category: str
    bbox_xywh: list[float]
    bbox_xyxy: list[float]
    area: float
    iscrowd: int


@dataclass(frozen=True)
class CocoSplit:
    """Loaded records and indexes for one COCO dataset split."""

    name: str
    directory: Path
    annotations_path: Path
    images: list[CocoImage]
    annotations: list[CocoAnnotation]
    annotations_by_image_id: dict[int, list[CocoAnnotation]]
    category_id_to_name: dict[int, str]
    ignored_category_ids: set[int]
    recording_ids: set[str]


@dataclass(frozen=True)
class SplitSummary:
    """Serializable validation counts for one imported split."""

    image_count: int
    annotation_count: int
    negative_image_count: int
    recording_count: int
    category_counts: dict[str, int]


@dataclass(frozen=True)
class DatasetImportResult:
    """Paths, checksum, and validated counts for one imported archive."""

    dataset_dir: Path
    archive_path: Path
    archive_sha256: str
    split_summaries: dict[str, SplitSummary]


def import_coco_dataset(archive_path: Path, output_dir: Path) -> DatasetImportResult:
    """Safely extract a COCO ZIP once and write local source provenance."""
    if not archive_path.is_file():
        raise ValueError(f"Dataset archive does not exist: {archive_path}")
    archive_sha256 = file_sha256(archive_path)
    metadata_path = output_dir / "source.json"
    if output_dir.exists() and any(output_dir.iterdir()):
        if metadata_path.is_file():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata.get("archive_sha256") == archive_sha256:
                summaries = validate_coco_dataset(output_dir)
                return DatasetImportResult(
                    output_dir,
                    archive_path.resolve(),
                    archive_sha256,
                    summaries,
                )
        raise FileExistsError(
            f"Dataset directory already contains a different import: {output_dir}"
        )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="hand-coco-import-", dir=output_dir.parent
    ) as temporary_directory:
        temporary_root = Path(temporary_directory)
        with zipfile.ZipFile(archive_path) as archive:
            _safe_extract_zip(archive, temporary_root)
        summaries = validate_coco_dataset(temporary_root)
        source_payload = {
            "archive_path": archive_path.resolve().as_posix(),
            "archive_sha256": archive_sha256,
            "dataset_role": "development-audit",
            "canonical_categories": list(CANONICAL_CATEGORIES),
            "splits": {name: asdict(summary) for name, summary in summaries.items()},
        }
        (temporary_root / "source.json").write_text(
            json.dumps(source_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if output_dir.exists():
            output_dir.rmdir()
        shutil.move(str(temporary_root), str(output_dir))
    return DatasetImportResult(
        output_dir,
        archive_path.resolve(),
        archive_sha256,
        summaries,
    )


def validate_coco_dataset(dataset_dir: Path) -> dict[str, SplitSummary]:
    """Validate all splits, canonical classes, boxes, files, and recording leakage."""
    splits = {name: load_coco_split(dataset_dir, name) for name in SPLIT_NAMES}
    for left_index, left_name in enumerate(SPLIT_NAMES):
        for right_name in SPLIT_NAMES[left_index + 1 :]:
            overlap = splits[left_name].recording_ids & splits[right_name].recording_ids
            if overlap:
                examples = ", ".join(sorted(overlap)[:3])
                raise ValueError(
                    f"Recording leakage between {left_name} and {right_name}: "
                    f"{len(overlap)} overlap(s), including {examples}"
                )
    return {name: summarize_split(split) for name, split in splits.items()}


def load_coco_split(dataset_root: Path, split_name: str) -> CocoSplit:
    """Load and fully validate one train, valid, or test COCO split."""
    if split_name not in SPLIT_NAMES:
        raise ValueError(f"Split must be one of: {', '.join(SPLIT_NAMES)}")
    split_dir = dataset_root / split_name
    annotations_path = split_dir / ANNOTATIONS_FILE_NAME
    if not annotations_path.is_file():
        raise ValueError(f"Missing COCO annotations: {annotations_path}")
    payload = json.loads(annotations_path.read_text(encoding="utf-8"))
    category_id_to_name = {
        int(category["id"]): str(category["name"])
        for category in payload.get("categories", [])
    }
    category_annotation_counts = {category_id: 0 for category_id in category_id_to_name}
    for annotation in payload.get("annotations", []):
        category_id = int(annotation["category_id"])
        category_annotation_counts[category_id] = (
            category_annotation_counts.get(category_id, 0) + 1
        )
    canonical_ids = {
        category_id
        for category_id, name in category_id_to_name.items()
        if name in CANONICAL_CATEGORIES
    }
    canonical_names = {
        category_id_to_name[category_id] for category_id in canonical_ids
    }
    if canonical_names != set(CANONICAL_CATEGORIES):
        raise ValueError(
            f"Expected categories {CANONICAL_CATEGORIES}; found "
            f"{sorted(category_id_to_name.values())}"
        )
    ignored_category_ids = set(category_id_to_name) - canonical_ids
    populated_ignored = {
        category_id: category_annotation_counts[category_id]
        for category_id in ignored_category_ids
        if category_annotation_counts.get(category_id, 0)
    }
    if populated_ignored:
        raise ValueError(
            f"Unsupported categories contain annotations: {populated_ignored}"
        )

    images: list[CocoImage] = []
    image_ids: set[int] = set()
    file_names: set[str] = set()
    for raw_image in payload.get("images", []):
        image_id = int(raw_image["id"])
        file_name = str(raw_image["file_name"])
        if image_id in image_ids:
            raise ValueError(f"Duplicate COCO image id in {split_name}: {image_id}")
        if file_name in file_names:
            raise ValueError(
                f"Duplicate COCO image filename in {split_name}: {file_name}"
            )
        image_path = split_dir / file_name
        if not image_path.is_file():
            raise ValueError(f"COCO image is missing: {image_path}")
        width = int(raw_image["width"])
        height = int(raw_image["height"])
        if width <= 0 or height <= 0:
            raise ValueError(f"Invalid image dimensions for {file_name}")
        images.append(CocoImage(image_id, file_name, width, height, image_path))
        image_ids.add(image_id)
        file_names.add(file_name)

    image_by_id = {image.id: image for image in images}
    annotations: list[CocoAnnotation] = []
    annotations_by_image_id = {image.id: [] for image in images}
    annotation_ids: set[int] = set()
    for raw_annotation in payload.get("annotations", []):
        category_id = int(raw_annotation["category_id"])
        if category_id in ignored_category_ids:
            continue
        annotation_id = int(raw_annotation["id"])
        image_id = int(raw_annotation["image_id"])
        if annotation_id in annotation_ids:
            raise ValueError(
                f"Duplicate COCO annotation id in {split_name}: {annotation_id}"
            )
        if image_id not in image_by_id:
            raise ValueError(
                f"Annotation {annotation_id} references unknown image {image_id}"
            )
        bbox_xywh = [float(value) for value in raw_annotation["bbox"]]
        if len(bbox_xywh) != 4:
            raise ValueError(f"Annotation {annotation_id} has an invalid bbox")
        x, y, width, height = bbox_xywh
        image = image_by_id[image_id]
        if (
            width <= 0
            or height <= 0
            or x < 0
            or y < 0
            or x + width > image.width + 1e-6
            or y + height > image.height + 1e-6
        ):
            raise ValueError(
                f"Annotation {annotation_id} lies outside image {image.file_name}"
            )
        annotation = CocoAnnotation(
            id=annotation_id,
            image_id=image_id,
            category_id=category_id,
            category=category_id_to_name[category_id],
            bbox_xywh=bbox_xywh,
            bbox_xyxy=[x, y, x + width, y + height],
            area=float(raw_annotation.get("area", width * height)),
            iscrowd=int(raw_annotation.get("iscrowd", 0)),
        )
        annotations.append(annotation)
        annotations_by_image_id[image_id].append(annotation)
        annotation_ids.add(annotation_id)

    recordings = {recording_id_from_file_name(image.file_name) for image in images}
    return CocoSplit(
        name=split_name,
        directory=split_dir,
        annotations_path=annotations_path,
        images=images,
        annotations=annotations,
        annotations_by_image_id=annotations_by_image_id,
        category_id_to_name=category_id_to_name,
        ignored_category_ids=ignored_category_ids,
        recording_ids=recordings,
    )


def summarize_split(split: CocoSplit) -> SplitSummary:
    """Build deterministic counts for a validated split."""
    category_counts = {
        category: sum(
            annotation.category == category for annotation in split.annotations
        )
        for category in CANONICAL_CATEGORIES
    }
    return SplitSummary(
        image_count=len(split.images),
        annotation_count=len(split.annotations),
        negative_image_count=sum(
            not split.annotations_by_image_id[image.id] for image in split.images
        ),
        recording_count=len(split.recording_ids),
        category_counts=category_counts,
    )


def recording_id_from_file_name(file_name: str) -> str:
    """Extract the source recording UUID from a Roboflow image filename."""
    stable_name = stable_image_key(file_name)
    recording_id, separator, _ = stable_name.partition("_frame")
    if not separator or not recording_id:
        raise ValueError(f"Could not derive recording id from {file_name}")
    return recording_id


def stable_image_key(file_name: str) -> str:
    """Strip Roboflow's revision-specific filename hash."""
    return ROBOFLOW_SUFFIX_PATTERN.sub(".jpg", Path(file_name).name)


def file_sha256(path: Path) -> str:
    """Return a streaming SHA-256 digest for a local artifact."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_extract_zip(archive: zipfile.ZipFile, output_dir: Path) -> None:
    """Extract an archive after rejecting absolute and parent-traversal paths."""
    for member in archive.infolist():
        member_path = PurePosixPath(member.filename)
        if member_path.is_absolute() or ".." in member_path.parts:
            raise ValueError(f"Unsafe dataset archive member: {member.filename}")
        target_path = output_dir.joinpath(*member_path.parts)
        if member.is_dir():
            target_path.mkdir(parents=True, exist_ok=True)
            continue
        target_path.parent.mkdir(parents=True, exist_ok=True)
        with archive.open(member) as source, target_path.open("wb") as target:
            shutil.copyfileobj(source, target)
