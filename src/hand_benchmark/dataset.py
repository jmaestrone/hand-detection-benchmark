"""Batch MCAP export and provenance-aware frame extraction workflows."""

from __future__ import annotations

import json
import math
import re
import shutil
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path
from typing import Iterable

from hand_benchmark.config import HEAD_LEFT_VIDEO_TOPIC
from hand_benchmark.mcap_video import VideoExportResult, export_video_topic

SUPPORTED_IMAGE_FORMATS = {"jpg", "jpeg", "png"}


@dataclass(frozen=True)
class VideoProvenance:
    """Persistent provenance for one cached head-left MP4 export."""

    source_mcap_path: str
    source_mcap_stem: str
    video_topic: str
    output_video_path: str
    frame_id: str | None
    frame_count: int
    time_range_sec: list[float]
    average_fps: float
    format: str
    warnings: list[str]


@dataclass(frozen=True)
class FrameMetadata:
    """Provenance and source location for one sampled dataset image."""

    file_name: str
    output_path: str
    source_video: str
    source_mcap_path: str
    source_mcap_stem: str
    video_topic: str
    frame_index: int
    timestamp_seconds: float
    width: int
    height: int


@dataclass(frozen=True)
class VideoInfo:
    """Properties of a cached MP4 needed for deterministic frame sampling."""

    path: Path
    width: int
    height: int
    fps: float
    duration_seconds: float
    frame_count: int | None


def export_head_left_videos(
    input_dir: Path,
    output_dir: Path,
    metadata_path: Path,
    overwrite: bool = False,
    limit: int | None = None,
    start_index: int = 0,
) -> tuple[int, int, list[Path]]:
    """Export head-left MP4s and write one provenance row per source MCAP.

    Existing rows with matching outputs are retained on repeat runs. An output
    with no provenance row is rejected unless ``overwrite`` is requested.
    """
    mcap_paths = list_mcap_paths(input_dir)
    mcap_paths = mcap_paths[start_index:]
    if limit is not None:
        mcap_paths = mcap_paths[:limit]
    existing_records = read_jsonl(metadata_path)
    records_by_source = {str(record["source_mcap_path"]): record for record in existing_records}
    output_dir.mkdir(parents=True, exist_ok=True)
    exported_count = 0
    skipped_paths: list[Path] = []

    for mcap_path in mcap_paths:
        output_path = output_dir / f"{mcap_path.stem}.mp4"
        source_key = str(mcap_path.resolve())
        existing_record = records_by_source.get(source_key)
        if output_path.exists() and not overwrite:
            if existing_record is None:
                raise FileExistsError(
                    f"Existing video has no matching provenance row: {output_path}. Pass --overwrite."
                )
            skipped_paths.append(mcap_path)
            continue
        result = export_video_topic(
            mcap_path=mcap_path,
            output_video_path=output_path,
            video_topic=HEAD_LEFT_VIDEO_TOPIC,
            overwrite=overwrite,
        )
        records_by_source[source_key] = video_provenance(result, output_dir)
        write_jsonl(
            metadata_path,
            sorted(records_by_source.values(), key=lambda record: str(record["source_mcap_path"])),
        )
        exported_count += 1

    write_jsonl(
        metadata_path,
        sorted(records_by_source.values(), key=lambda record: str(record["source_mcap_path"])),
    )
    return len(mcap_paths), exported_count, skipped_paths


def extract_frames(
    video_dir: Path,
    video_metadata_path: Path,
    output_dir: Path,
    metadata_path: Path,
    fps: float = 1.0,
    image_format: str = "jpg",
    quality: int = 95,
    overwrite: bool = False,
    limit: int | None = None,
    start_index: int = 0,
) -> tuple[int, int]:
    """Sample cached MP4s and retain their MCAP provenance in frame metadata."""
    validate_frame_options(fps, image_format, quality)
    provenance_by_video = {
        Path(record["output_video_path"]).name: record for record in read_jsonl(video_metadata_path)
    }
    video_paths = sorted(video_dir.glob("*.mp4"))
    if not video_paths:
        raise ValueError(f"No MP4 videos found in {video_dir}")
    video_paths = video_paths[start_index:]
    if limit is not None:
        video_paths = video_paths[:limit]
    records_by_filename = {
        str(record["file_name"]): record for record in read_jsonl(metadata_path)
    }
    frame_count = 0
    for video_path in video_paths:
        provenance = provenance_by_video.get(video_path.name)
        if provenance is None:
            raise ValueError(f"Missing MCAP provenance for cached video: {video_path}")
        video_info = probe_video(video_path)
        sample_points = list(iter_sample_points(video_info, fps))
        frame_paths = [
            output_dir / build_frame_file_name(video_path.stem, frame_index, timestamp_seconds, image_format)
            for frame_index, timestamp_seconds in sample_points
        ]
        extract_video_frames(video_path, sample_points, frame_paths, image_format, quality, overwrite)
        for (frame_index, timestamp_seconds), frame_path in zip(sample_points, frame_paths, strict=True):
            filename = build_frame_file_name(video_path.stem, frame_index, timestamp_seconds, image_format)
            record = FrameMetadata(
                    file_name=filename,
                    output_path=filename,
                    source_video=video_path.name,
                    source_mcap_path=str(provenance["source_mcap_path"]),
                    source_mcap_stem=str(provenance["source_mcap_stem"]),
                    video_topic=str(provenance["video_topic"]),
                    frame_index=frame_index,
                    timestamp_seconds=round(timestamp_seconds, 6),
                    width=video_info.width,
                    height=video_info.height,
                )
            records_by_filename[filename] = asdict(record)
            frame_count += 1
        write_jsonl(
            metadata_path,
            sorted(records_by_filename.values(), key=lambda record: str(record["file_name"])),
        )
    return len(video_paths), frame_count


def list_mcap_paths(input_dir: Path) -> list[Path]:
    """Return MCAP source files in deterministic filename order."""
    if not input_dir.is_dir():
        raise ValueError(f"Input directory does not exist: {input_dir}")
    paths = sorted(path for path in input_dir.iterdir() if path.is_file() and path.suffix.lower() == ".mcap")
    if not paths:
        raise ValueError(f"No MCAP files found in {input_dir}")
    return paths


def video_provenance(result: VideoExportResult, output_dir: Path) -> dict[str, object]:
    """Convert an MCAP export result into the persistent dataset metadata shape."""
    return asdict(
        VideoProvenance(
            source_mcap_path=str(result.source_mcap_path.resolve()),
            source_mcap_stem=result.source_mcap_path.stem,
            video_topic=result.video_topic,
            output_video_path=result.output_video_path.relative_to(output_dir).as_posix(),
            frame_id=result.frame_id,
            frame_count=result.frame_count,
            time_range_sec=list(result.time_range_sec),
            average_fps=result.average_fps,
            format=result.format,
            warnings=result.warnings,
        )
    )


def probe_video(video_path: Path) -> VideoInfo:
    """Read MP4 dimensions, frame rate, duration, and frame count via ffprobe."""
    process = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height,avg_frame_rate,r_frame_rate,duration,nb_frames", "-of", "json", str(video_path)],
        capture_output=True,
        check=True,
        text=True,
    )
    streams = json.loads(process.stdout).get("streams", [])
    if not streams:
        raise ValueError(f"No video stream found in {video_path}")
    stream = streams[0]
    native_fps = parse_frame_rate(stream.get("avg_frame_rate") or stream.get("r_frame_rate"))
    duration = float(stream.get("duration") or 0)
    frame_count = int(stream["nb_frames"]) if stream.get("nb_frames") else None
    if native_fps <= 0 or (duration <= 0 and frame_count is None):
        raise ValueError(f"Could not determine frame properties for {video_path}")
    return VideoInfo(video_path, int(stream["width"]), int(stream["height"]), native_fps, duration, frame_count)


def iter_sample_points(video_info: VideoInfo, fps: float) -> Iterable[tuple[int, float]]:
    """Yield deterministic time-based sampling positions for a source video."""
    sample_count = max(1, math.ceil(video_info.duration_seconds * fps))
    for sample_number in range(sample_count):
        timestamp = sample_number / fps
        frame_index = math.floor(timestamp * video_info.fps)
        if timestamp >= video_info.duration_seconds or (video_info.frame_count is not None and frame_index >= video_info.frame_count):
            break
        yield frame_index, frame_index / video_info.fps


def build_frame_file_name(video_stem: str, frame_index: int, timestamp_seconds: float, image_format: str) -> str:
    """Build a globally unique, deterministic image filename."""
    safe_stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", video_stem).strip("._")
    return f"{safe_stem}_frame{frame_index:06d}_{round(timestamp_seconds * 1000):010d}ms.{image_format}"


def extract_frame(video_path: Path, output_path: Path, timestamp_seconds: float, image_format: str, quality: int, overwrite: bool) -> None:
    """Extract one image with ffmpeg unless it already exists."""
    if output_path.exists() and not overwrite:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y" if overwrite else "-n", "-i", str(video_path), "-ss", f"{timestamp_seconds:.6f}", "-frames:v", "1"]
    if image_format in {"jpg", "jpeg"}:
        command.extend(["-q:v", str(jpeg_quality_to_qscale(quality))])
    subprocess.run([*command, str(output_path)], check=True)


def extract_video_frames(
    video_path: Path,
    sample_points: list[tuple[int, float]],
    output_paths: list[Path],
    image_format: str,
    quality: int,
    overwrite: bool,
) -> None:
    """Extract deterministic source-frame indices in one ffmpeg decode pass."""
    if len(sample_points) != len(output_paths):
        raise ValueError("Sample points and output paths must have the same length")
    if not overwrite and all(path.exists() for path in output_paths):
        return
    output_paths[0].parent.mkdir(parents=True, exist_ok=True)
    selected_frames = "+".join(f"eq(n\\,{frame_index})" for frame_index, _ in sample_points)
    with tempfile.TemporaryDirectory(prefix="frame-extract-", dir=output_paths[0].parent) as temporary_directory:
        temporary_pattern = Path(temporary_directory) / f"frame-%06d.{image_format}"
        command = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(video_path),
            "-vf", f"select={selected_frames}", "-vsync", "0", "-start_number", "0",
        ]
        if image_format in {"jpg", "jpeg"}:
            command.extend(["-q:v", str(jpeg_quality_to_qscale(quality))])
        subprocess.run([*command, str(temporary_pattern)], check=True)
        temporary_paths = sorted(Path(temporary_directory).glob(f"*.{image_format}"))
        if len(temporary_paths) != len(output_paths):
            raise RuntimeError(
                f"ffmpeg extracted {len(temporary_paths)} frames for {video_path}; expected {len(output_paths)}"
            )
        for temporary_path, output_path in zip(temporary_paths, output_paths, strict=True):
            if overwrite or not output_path.exists():
                shutil.move(str(temporary_path), output_path)


def read_jsonl(path: Path) -> list[dict[str, object]]:
    """Read a JSONL metadata file, returning no rows when it does not exist."""
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_jsonl(path: Path, records: Iterable[dict[str, object]]) -> None:
    """Write JSONL metadata atomically enough for a completed command run."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def validate_frame_options(fps: float, image_format: str, quality: int) -> None:
    """Validate user-controlled frame extraction options."""
    if fps <= 0:
        raise ValueError("--fps must be greater than 0")
    if image_format not in SUPPORTED_IMAGE_FORMATS:
        raise ValueError("--image-format must be jpg, jpeg, or png")
    if not 1 <= quality <= 100:
        raise ValueError("--quality must be between 1 and 100")


def parse_frame_rate(raw_value: str | None) -> float:
    """Parse ffprobe's fractional frame-rate representation."""
    return float(Fraction(raw_value)) if raw_value else 0.0


def jpeg_quality_to_qscale(quality: int) -> int:
    """Convert a user-facing JPEG quality value to ffmpeg's qscale range."""
    return max(2, min(31, round(31 - ((quality - 1) * 29 / 99))))
