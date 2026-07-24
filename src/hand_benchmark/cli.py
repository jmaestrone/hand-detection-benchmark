"""Command-line interface for the head-left hand dataset bootstrap."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Annotated

import typer

from hand_benchmark.benchmark_inference import (
    parse_split_names,
    predict_rfdetr_coco,
    predict_wilor_coco,
)
from hand_benchmark.coco_dataset import import_coco_dataset
from hand_benchmark.comparison import run_model_comparison
from hand_benchmark.config import (
    DEFAULT_AUDIT_RUN_DIR,
    DEFAULT_BENCHMARK_INFERENCE_CONFIDENCE,
    DEFAULT_EVALUATION_DATASET_DIR,
    DEFAULT_FRAME_FPS,
    DEFAULT_FRAME_METADATA_PATH,
    DEFAULT_FRAMES_DIR,
    DEFAULT_PREDICTIONS_PATH,
    DEFAULT_RFDETR_WEIGHTS_PATH,
    DEFAULT_ROBOFLOW_EXPORT_DIR,
    DEFAULT_VIDEO_DIR,
    DEFAULT_VIDEO_METADATA_PATH,
    DEFAULT_WILOR_CONFIDENCE,
    DEFAULT_WILOR_DETECTOR_METADATA_PATH,
    DEFAULT_WILOR_DETECTOR_PATH,
)
from hand_benchmark.dataset import export_head_left_videos
from hand_benchmark.dataset import extract_frames as extract_dataset_frames
from hand_benchmark.revision_diff import compare_coco_revisions
from hand_benchmark.roboflow_export import export_roboflow_yolo
from hand_benchmark.wilor import download_wilor_detector, predict_wilor_frames

app = typer.Typer(
    help="Build a raw head-left hand detection corpus from MCAP recordings."
)


@app.command("compare-models")
def compare_models_command(
    dataset_root: Annotated[
        Path, typer.Option(help="Imported COCO dataset root with train/valid/test.")
    ] = DEFAULT_EVALUATION_DATASET_DIR,
    run_root: Annotated[
        Path, typer.Option(help="Audit run containing both models' predictions.")
    ] = DEFAULT_AUDIT_RUN_DIR,
    iou_threshold: Annotated[
        float,
        typer.Option(min=0.0, max=1.0, help="IoU used for TP/FP/FN matching."),
    ] = 0.5,
    render_overlays: Annotated[
        bool,
        typer.Option(help="Render error-only colored model and comparison panels."),
    ] = True,
) -> None:
    """Select validation F2 thresholds and write the complete model audit."""
    try:
        result = run_model_comparison(
            dataset_root=dataset_root,
            run_root=run_root,
            iou_threshold=iou_threshold,
            render_overlays=render_overlays,
        )
    except (OSError, RuntimeError, ValueError) as error:
        raise typer.BadParameter(str(error)) from error
    typer.echo(f"Audit: {result.run_root}")
    typer.echo(f"Metric rows: {result.metric_row_count}")
    typer.echo(f"Error rows: {result.error_row_count}")
    for model_name, threshold in result.selected_f2_thresholds.items():
        typer.echo(f"{model_name}: selected F2 threshold={threshold:.6f}")


@app.command("compare-coco-revisions")
def compare_coco_revisions_command(
    old_dataset_root: Annotated[
        Path, typer.Option(help="Previous immutable COCO dataset root.")
    ],
    new_dataset_root: Annotated[
        Path, typer.Option(help="New corrected immutable COCO dataset root.")
    ],
    output_dir: Annotated[
        Path, typer.Option(help="Ignored output directory for revision reports.")
    ],
    unchanged_iou: Annotated[
        float,
        typer.Option(min=0.0, max=1.0, help="IoU treated as unchanged."),
    ] = 0.95,
    related_iou: Annotated[
        float,
        typer.Option(
            min=0.0, max=1.0, help="Minimum IoU for adjusted/relabelled boxes."
        ),
    ] = 0.5,
) -> None:
    """Compare two manually reviewed COCO revisions without modifying either."""
    try:
        changes = compare_coco_revisions(
            old_dataset_root=old_dataset_root,
            new_dataset_root=new_dataset_root,
            output_dir=output_dir,
            unchanged_iou=unchanged_iou,
            related_iou=related_iou,
        )
    except (OSError, ValueError) as error:
        raise typer.BadParameter(str(error)) from error
    typer.echo(f"Wrote {len(changes)} revision changes to {output_dir}")


@app.command("predict-wilor-coco")
def predict_wilor_coco_command(
    dataset_root: Annotated[
        Path, typer.Option(help="Imported COCO dataset root with train/valid/test.")
    ] = DEFAULT_EVALUATION_DATASET_DIR,
    detector_path: Annotated[
        Path, typer.Option(help="Local WiLoR detector checkpoint.")
    ] = DEFAULT_WILOR_DETECTOR_PATH,
    output_root: Annotated[
        Path, typer.Option(help="Ignored audit run output root.")
    ] = DEFAULT_AUDIT_RUN_DIR,
    split: Annotated[
        str, typer.Option(help="Dataset split: train, valid, test, or all.")
    ] = "all",
    confidence: Annotated[
        float,
        typer.Option(min=0.0, max=1.0, help="Low inference confidence floor."),
    ] = DEFAULT_BENCHMARK_INFERENCE_CONFIDENCE,
    device: Annotated[
        str, typer.Option(help="Inference device: auto, mps, cuda, or cpu.")
    ] = "auto",
    batch_size: Annotated[
        int, typer.Option(min=1, help="Images per inference batch.")
    ] = 8,
    limit: Annotated[
        int | None, typer.Option(min=1, help="Optional per-split smoke limit.")
    ] = None,
    preview_dir: Annotated[
        Path | None, typer.Option(help="Optional preview output root.")
    ] = None,
    max_previews: Annotated[
        int, typer.Option(min=0, help="Maximum previews per split.")
    ] = 20,
) -> None:
    """Run WiLoR on reviewed COCO pixels and write normalized predictions."""
    try:
        outputs = predict_wilor_coco(
            dataset_root=dataset_root,
            detector_path=detector_path,
            output_root=output_root,
            split_names=parse_split_names(split),
            confidence=confidence,
            device=device,
            batch_size=batch_size,
            limit=limit,
            preview_dir=preview_dir,
            max_previews=max_previews,
        )
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as error:
        raise typer.BadParameter(str(error)) from error
    for split_name, output_path in outputs.items():
        typer.echo(f"{split_name}: {output_path}")


@app.command("predict-rfdetr-coco")
def predict_rfdetr_coco_command(
    dataset_root: Annotated[
        Path, typer.Option(help="Imported COCO dataset root with train/valid/test.")
    ] = DEFAULT_EVALUATION_DATASET_DIR,
    weights_path: Annotated[
        Path, typer.Option(help="Self-describing RF-DETR checkpoint.")
    ] = DEFAULT_RFDETR_WEIGHTS_PATH,
    output_root: Annotated[
        Path, typer.Option(help="Ignored audit run output root.")
    ] = DEFAULT_AUDIT_RUN_DIR,
    split: Annotated[
        str, typer.Option(help="Dataset split: train, valid, test, or all.")
    ] = "all",
    confidence: Annotated[
        float,
        typer.Option(min=0.0, max=1.0, help="Low inference confidence floor."),
    ] = DEFAULT_BENCHMARK_INFERENCE_CONFIDENCE,
    device: Annotated[
        str, typer.Option(help="Inference device: auto, mps, cuda, or cpu.")
    ] = "auto",
    batch_size: Annotated[
        int, typer.Option(min=1, help="Images per inference batch.")
    ] = 4,
    limit: Annotated[
        int | None, typer.Option(min=1, help="Optional per-split smoke limit.")
    ] = None,
    preview_dir: Annotated[
        Path | None, typer.Option(help="Optional preview output root.")
    ] = None,
    max_previews: Annotated[
        int, typer.Option(min=0, help="Maximum previews per split.")
    ] = 20,
) -> None:
    """Run RF-DETR on reviewed COCO pixels and write normalized predictions."""
    try:
        outputs = predict_rfdetr_coco(
            dataset_root=dataset_root,
            weights_path=weights_path,
            output_root=output_root,
            split_names=parse_split_names(split),
            confidence=confidence,
            device=device,
            batch_size=batch_size,
            limit=limit,
            preview_dir=preview_dir,
            max_previews=max_previews,
        )
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as error:
        raise typer.BadParameter(str(error)) from error
    for split_name, output_path in outputs.items():
        typer.echo(f"{split_name}: {output_path}")


@app.command("import-coco-dataset")
def import_coco_dataset_command(
    archive_path: Annotated[
        Path,
        typer.Option("--archive", help="Reviewed Roboflow COCO ZIP archive."),
    ],
    output_dir: Annotated[
        Path,
        typer.Option(help="Ignored immutable destination for the extracted dataset."),
    ] = DEFAULT_EVALUATION_DATASET_DIR,
) -> None:
    """Import and validate a recording-disjoint two-class COCO dataset."""
    try:
        result = import_coco_dataset(archive_path, output_dir)
    except (FileExistsError, OSError, ValueError) as error:
        raise typer.BadParameter(str(error)) from error
    typer.echo(f"Dataset: {result.dataset_dir}")
    typer.echo(f"Archive SHA-256: {result.archive_sha256}")
    for split_name, summary in result.split_summaries.items():
        typer.echo(
            f"{split_name}: images={summary.image_count}, "
            f"annotations={summary.annotation_count}, "
            f"recordings={summary.recording_count}, "
            f"negatives={summary.negative_image_count}"
        )


@app.command("export-roboflow-yolo")
def export_roboflow_yolo_command(
    frames_dir: Annotated[
        Path, typer.Option(help="Directory containing extracted frame images.")
    ] = DEFAULT_FRAMES_DIR,
    frame_metadata_path: Annotated[
        Path, typer.Option(help="Frame provenance JSONL path.")
    ] = DEFAULT_FRAME_METADATA_PATH,
    predictions_path: Annotated[
        Path, typer.Option(help="WiLoR prediction JSONL path.")
    ] = DEFAULT_PREDICTIONS_PATH,
    output_dir: Annotated[
        Path, typer.Option(help="Ignored destination for the YOLO import folder.")
    ] = DEFAULT_ROBOFLOW_EXPORT_DIR,
    overwrite: Annotated[
        bool,
        typer.Option(
            help="Replace generated labels and manifest in an existing export."
        ),
    ] = False,
) -> None:
    """Export all frames and WiLoR pre-labels in Roboflow-importable YOLO format."""
    try:
        result = export_roboflow_yolo(
            frames_dir,
            frame_metadata_path,
            predictions_path,
            output_dir,
            overwrite,
        )
    except (FileExistsError, OSError, ValueError) as error:
        raise typer.BadParameter(str(error)) from error
    typer.echo(
        f"Exported {result.image_count} images and {result.detection_count} detections "
        f"to {result.output_dir}"
    )
    typer.echo(f"Empty label files: {result.empty_label_count}")


@app.command("download-wilor-detector")
def download_wilor_detector_command(
    detector_path: Annotated[
        Path, typer.Option(help="Ignored path for WiLoR detector.pt.")
    ] = DEFAULT_WILOR_DETECTOR_PATH,
    metadata_path: Annotated[
        Path, typer.Option(help="Ignored model provenance JSON path.")
    ] = DEFAULT_WILOR_DETECTOR_METADATA_PATH,
    overwrite: Annotated[
        bool, typer.Option(help="Replace an existing detector weight file.")
    ] = False,
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
    frames_dir: Annotated[
        Path, typer.Option(help="Directory containing extracted frame images.")
    ] = DEFAULT_FRAMES_DIR,
    metadata_path: Annotated[
        Path, typer.Option(help="Frame provenance JSONL path.")
    ] = DEFAULT_FRAME_METADATA_PATH,
    detector_path: Annotated[
        Path, typer.Option(help="Local WiLoR detector.pt path.")
    ] = DEFAULT_WILOR_DETECTOR_PATH,
    output_path: Annotated[
        Path, typer.Option(help="Ignored JSONL path for raw pre-labels.")
    ] = DEFAULT_PREDICTIONS_PATH,
    confidence: Annotated[
        float, typer.Option(min=0.0, max=1.0, help="Detector confidence threshold.")
    ] = DEFAULT_WILOR_CONFIDENCE,
    device: Annotated[
        str, typer.Option(help="Inference device: auto, cpu, cuda, or mps.")
    ] = "auto",
    batch_size: Annotated[
        int, typer.Option(min=1, help="Number of frames per YOLO inference batch.")
    ] = 8,
    limit: Annotated[
        int | None,
        typer.Option(min=1, help="Only process the first N frames for a smoke run."),
    ] = None,
    preview_dir: Annotated[
        Path | None, typer.Option(help="Optional annotated-preview directory.")
    ] = None,
    max_previews: Annotated[
        int, typer.Option(min=0, help="Maximum preview images to write.")
    ] = 20,
) -> None:
    """Run the WiLoR detector on extracted frames and write left/right pre-labels."""
    try:
        image_count, detection_count, preview_count = predict_wilor_frames(
            frames_dir,
            metadata_path,
            detector_path,
            output_path,
            confidence,
            device,
            batch_size,
            limit,
            preview_dir,
            max_previews,
        )
    except (ImportError, OSError, RuntimeError, ValueError) as error:
        raise typer.BadParameter(str(error)) from error
    typer.echo(
        f"Wrote {detection_count} detections for {image_count} frames to {output_path}"
    )
    if preview_count:
        typer.echo(f"Preview images: {preview_count} in {preview_dir}")


@app.command("export-head-left-videos")
def export_head_left_videos_command(
    input_dir: Annotated[
        Path, typer.Option(help="Directory containing source MCAP files.")
    ],
    output_dir: Annotated[
        Path, typer.Option(help="Ignored directory for cached head-left MP4s.")
    ] = DEFAULT_VIDEO_DIR,
    metadata_path: Annotated[
        Path, typer.Option(help="JSONL video provenance output.")
    ] = DEFAULT_VIDEO_METADATA_PATH,
    overwrite: Annotated[
        bool, typer.Option(help="Regenerate existing MP4s and provenance.")
    ] = False,
    limit: Annotated[
        int | None,
        typer.Option(min=1, help="Only export the first N MCAPs for a smoke run."),
    ] = None,
    start_index: Annotated[
        int,
        typer.Option(
            min=0, help="Skip this many sorted MCAPs before applying --limit."
        ),
    ] = 0,
) -> None:
    """Export `/head_left/video` from every source MCAP into the local MP4 cache."""
    try:
        source_count, exported_count, skipped_paths = export_head_left_videos(
            input_dir, output_dir, metadata_path, overwrite, limit, start_index
        )
    except (
        ValueError,
        FileExistsError,
        RuntimeError,
        subprocess.CalledProcessError,
    ) as error:
        raise typer.BadParameter(str(error)) from error
    typer.echo(
        f"Processed {source_count} MCAPs: exported={exported_count}, skipped={len(skipped_paths)}"
    )
    typer.echo(f"Video provenance: {metadata_path}")


@app.command("extract-frames")
def extract_frames_command(
    video_dir: Annotated[
        Path, typer.Option(help="Directory containing cached head-left MP4s.")
    ] = DEFAULT_VIDEO_DIR,
    video_metadata_path: Annotated[
        Path, typer.Option(help="Video provenance JSONL path.")
    ] = DEFAULT_VIDEO_METADATA_PATH,
    output_dir: Annotated[
        Path, typer.Option(help="Ignored directory for sampled image frames.")
    ] = DEFAULT_FRAMES_DIR,
    metadata_path: Annotated[
        Path, typer.Option(help="JSONL frame provenance output.")
    ] = DEFAULT_FRAME_METADATA_PATH,
    fps: Annotated[
        float, typer.Option(min=0.001, help="Frames per second to sample.")
    ] = DEFAULT_FRAME_FPS,
    image_format: Annotated[
        str, typer.Option(help="Output format: jpg, jpeg, or png.")
    ] = "jpg",
    quality: Annotated[int, typer.Option(min=1, max=100, help="JPEG quality.")] = 95,
    overwrite: Annotated[
        bool, typer.Option(help="Regenerate existing frame images.")
    ] = False,
    limit: Annotated[
        int | None,
        typer.Option(min=1, help="Only sample the first N videos for a smoke run."),
    ] = None,
    start_index: Annotated[
        int,
        typer.Option(
            min=0, help="Skip this many sorted videos before applying --limit."
        ),
    ] = 0,
) -> None:
    """Extract provenance-linked frames from cached head-left MP4s."""
    try:
        video_count, frame_count = extract_dataset_frames(
            video_dir,
            video_metadata_path,
            output_dir,
            metadata_path,
            fps,
            image_format,
            quality,
            overwrite,
            limit,
            start_index,
        )
    except (ValueError, subprocess.CalledProcessError) as error:
        raise typer.BadParameter(str(error)) from error
    typer.echo(
        f"Extracted {frame_count} frames from {video_count} videos to {output_dir}"
    )
    typer.echo(f"Frame metadata: {metadata_path}")
