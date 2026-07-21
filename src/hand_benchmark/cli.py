"""Command-line interface for the head-left hand dataset bootstrap."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Annotated

import typer

from hand_benchmark.config import (
    DEFAULT_FRAME_FPS,
    DEFAULT_FRAME_METADATA_PATH,
    DEFAULT_FRAMES_DIR,
    DEFAULT_VIDEO_DIR,
    DEFAULT_VIDEO_METADATA_PATH,
)
from hand_benchmark.dataset import extract_frames as extract_dataset_frames
from hand_benchmark.dataset import export_head_left_videos
from hand_benchmark.config import (
    DEFAULT_MODEL_DIR,
    DEFAULT_PREDICTIONS_PATH,
    DEFAULT_WILOR_CONFIDENCE,
    DEFAULT_WILOR_DETECTOR_METADATA_PATH,
    DEFAULT_WILOR_DETECTOR_PATH,
)
from hand_benchmark.wilor import download_wilor_detector, predict_wilor_frames

app = typer.Typer(help="Build a raw head-left hand detection corpus from MCAP recordings.")


@app.command("download-wilor-detector")
def download_wilor_detector_command(
    detector_path: Annotated[Path, typer.Option(help="Ignored path for WiLoR detector.pt.")] = DEFAULT_WILOR_DETECTOR_PATH,
    metadata_path: Annotated[Path, typer.Option(help="Ignored model provenance JSON path.")] = DEFAULT_WILOR_DETECTOR_METADATA_PATH,
    overwrite: Annotated[bool, typer.Option(help="Replace an existing detector weight file.")] = False,
) -> None:
    """Download the upstream WiLoR left/right hand detector with provenance."""
    try:
        result = download_wilor_detector(detector_path, metadata_path, overwrite)
    except (FileExistsError, OSError) as error:
        raise typer.BadParameter(str(error)) from error
    typer.echo(f"WiLoR detector: {result.detector_path}")
    typer.echo(f"SHA-256: {result.sha256}")


@app.command("predict-hands")
def predict_hands_command(
    frames_dir: Annotated[Path, typer.Option(help="Directory containing extracted frame images.")] = DEFAULT_FRAMES_DIR,
    metadata_path: Annotated[Path, typer.Option(help="Frame provenance JSONL path.")] = DEFAULT_FRAME_METADATA_PATH,
    detector_path: Annotated[Path, typer.Option(help="Local WiLoR detector.pt path.")] = DEFAULT_WILOR_DETECTOR_PATH,
    output_path: Annotated[Path, typer.Option(help="Ignored JSONL path for raw pre-labels.")] = DEFAULT_PREDICTIONS_PATH,
    confidence: Annotated[float, typer.Option(min=0.0, max=1.0, help="Detector confidence threshold.")] = DEFAULT_WILOR_CONFIDENCE,
    device: Annotated[str, typer.Option(help="Inference device: auto, cpu, cuda, or mps.")] = "auto",
    batch_size: Annotated[int, typer.Option(min=1, help="Number of frames per YOLO inference batch.")] = 8,
    limit: Annotated[int | None, typer.Option(min=1, help="Only process the first N frames for a smoke run.")] = None,
    preview_dir: Annotated[Path | None, typer.Option(help="Optional annotated-preview directory.")] = None,
    max_previews: Annotated[int, typer.Option(min=0, help="Maximum preview images to write.")] = 20,
) -> None:
    """Run the WiLoR detector on extracted frames and write left/right pre-labels."""
    try:
        image_count, detection_count, preview_count = predict_wilor_frames(
            frames_dir, metadata_path, detector_path, output_path, confidence, device,
            batch_size, limit, preview_dir, max_previews,
        )
    except (ImportError, OSError, RuntimeError, ValueError) as error:
        raise typer.BadParameter(str(error)) from error
    typer.echo(f"Wrote {detection_count} detections for {image_count} frames to {output_path}")
    if preview_count:
        typer.echo(f"Preview images: {preview_count} in {preview_dir}")


@app.command("export-head-left-videos")
def export_head_left_videos_command(
    input_dir: Annotated[Path, typer.Option(help="Directory containing source MCAP files.")],
    output_dir: Annotated[Path, typer.Option(help="Ignored directory for cached head-left MP4s.")] = DEFAULT_VIDEO_DIR,
    metadata_path: Annotated[Path, typer.Option(help="JSONL video provenance output.")] = DEFAULT_VIDEO_METADATA_PATH,
    overwrite: Annotated[bool, typer.Option(help="Regenerate existing MP4s and provenance.")] = False,
    limit: Annotated[int | None, typer.Option(min=1, help="Only export the first N MCAPs for a smoke run.")] = None,
    start_index: Annotated[int, typer.Option(min=0, help="Skip this many sorted MCAPs before applying --limit.")] = 0,
) -> None:
    """Export `/head_left/video` from every source MCAP into the local MP4 cache."""
    try:
        source_count, exported_count, skipped_paths = export_head_left_videos(input_dir, output_dir, metadata_path, overwrite, limit, start_index)
    except (ValueError, FileExistsError, RuntimeError, subprocess.CalledProcessError) as error:
        raise typer.BadParameter(str(error)) from error
    typer.echo(f"Processed {source_count} MCAPs: exported={exported_count}, skipped={len(skipped_paths)}")
    typer.echo(f"Video provenance: {metadata_path}")


@app.command("extract-frames")
def extract_frames_command(
    video_dir: Annotated[Path, typer.Option(help="Directory containing cached head-left MP4s.")] = DEFAULT_VIDEO_DIR,
    video_metadata_path: Annotated[Path, typer.Option(help="Video provenance JSONL path.")] = DEFAULT_VIDEO_METADATA_PATH,
    output_dir: Annotated[Path, typer.Option(help="Ignored directory for sampled image frames.")] = DEFAULT_FRAMES_DIR,
    metadata_path: Annotated[Path, typer.Option(help="JSONL frame provenance output.")] = DEFAULT_FRAME_METADATA_PATH,
    fps: Annotated[float, typer.Option(min=0.001, help="Frames per second to sample.")] = DEFAULT_FRAME_FPS,
    image_format: Annotated[str, typer.Option(help="Output format: jpg, jpeg, or png.")] = "jpg",
    quality: Annotated[int, typer.Option(min=1, max=100, help="JPEG quality.")] = 95,
    overwrite: Annotated[bool, typer.Option(help="Regenerate existing frame images.")] = False,
    limit: Annotated[int | None, typer.Option(min=1, help="Only sample the first N videos for a smoke run.")] = None,
    start_index: Annotated[int, typer.Option(min=0, help="Skip this many sorted videos before applying --limit.")] = 0,
) -> None:
    """Extract provenance-linked frames from cached head-left MP4s."""
    try:
        video_count, frame_count = extract_dataset_frames(video_dir, video_metadata_path, output_dir, metadata_path, fps, image_format, quality, overwrite, limit, start_index)
    except (ValueError, subprocess.CalledProcessError) as error:
        raise typer.BadParameter(str(error)) from error
    typer.echo(f"Extracted {frame_count} frames from {video_count} videos to {output_dir}")
    typer.echo(f"Frame metadata: {metadata_path}")
