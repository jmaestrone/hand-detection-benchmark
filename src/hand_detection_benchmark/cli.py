"""Command-line interface for the head-left hand dataset bootstrap."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Annotated

import typer

from hand_detection_benchmark.config import (
    DEFAULT_FRAME_FPS,
    DEFAULT_FRAME_METADATA_PATH,
    DEFAULT_FRAMES_DIR,
    DEFAULT_VIDEO_DIR,
    DEFAULT_VIDEO_METADATA_PATH,
)
from hand_detection_benchmark.dataset import extract_frames as extract_dataset_frames
from hand_detection_benchmark.dataset import export_head_left_videos

app = typer.Typer(help="Build a raw head-left hand detection corpus from MCAP recordings.")


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
