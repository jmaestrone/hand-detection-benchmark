# Hand Detection Benchmark

Local tooling for building a reproducible hand-detection corpus from the
`/head_left/video` stream embedded in MCAP recordings. The first workflow
exports cached MP4s and samples source-pixel frames; it does not select a model,
create reviewed labels, or upload data to Roboflow.

## Setup

```bash
uv sync --group dev
```

The MCAP extractor requires `ffmpeg` on `PATH`. MCAP files with LZ4-compressed
chunks additionally require the `lz4` command-line tool on `PATH`.

## Artifact layout

```text
data/videos/                 # ignored head-left MP4 cache and provenance JSONL
data/frames/                 # ignored 1-fps JPEGs and metadata JSONL
data/predictions/            # ignored future pre-label outputs
data/roboflow-export/        # ignored future review exports
data/benchmark/              # ignored future reviewed datasets
data/training/               # ignored future training datasets
models/                      # ignored local model weights
runs/                        # ignored previews and reports
```

Do not commit MCAPs, videos, frames, labels, model weights, or run artifacts.

## Initial corpus workflow

The initial corpus identity is `head-left-raw-1fps`: raw, unrectified pixels
from the `/head_left/video` topic. The source directory remains external to the
repository.

```bash
uv run hand-benchmark export-head-left-videos \
  --input-dir /Users/jpmaestrone/Downloads/fsstudio-hand-training

uv run hand-benchmark extract-frames
```

The first command writes `data/videos/<mcap-stem>.mp4` and
`data/videos/metadata.jsonl`. The second writes JPEGs plus
`data/frames/metadata.jsonl`, retaining the source MCAP path/stem and video
topic for every frame. Both commands are deterministic and skip artifacts that
already have matching metadata; pass `--overwrite` to regenerate them.

Long local runs can be split without changing the dataset identity. For example,
`--start-index 25 --limit 25` processes the second deterministic batch of 25
sources. Video and frame metadata are checkpointed after each completed source;
rerunning `extract-frames` reconciles any image files left by an interrupted
frame extraction.

Use `--limit 1` on either command for a smoke run. Use a distinct output
directory and `--fps 3` for any later denser corpus rather than overwriting this
dataset identity.

## Annotation schema

When this corpus is pre-labeled and reviewed in Roboflow, use exactly these
categories:

- `left_hand`
- `right_hand`

## WiLoR detector pre-labels

The first pre-label candidate is WiLoR's upstream left/right YOLO detector. Its
weights and all generated predictions remain ignored local artifacts:

```bash
uv run hand-benchmark download-wilor-detector
uv run hand-benchmark predict-hands --limit 20 --preview-dir runs/previews/wilor
```

Review the preview images before running the full corpus. `predict-hands` writes
one JSONL row per extracted frame, including frames with no hands, and validates
that the downloaded detector exposes the expected left/right class ordering.

## Roboflow review export

After reviewing the detector previews, create the complete import folder:

```bash
uv run hand-benchmark export-roboflow-yolo
```

This produces an ignored `data/roboflow-export/wilor-detector-yolo/` folder
with `images/train/`, matching `labels/train/` YOLO files, `data.yaml`, and a
provenance `manifest.jsonl`. Images are hard-linked to `data/frames/` when
possible, so this does not duplicate the source pixels. It includes empty label
files for frames with no WiLoR detections.

Roboflow's CLI is appropriate for this 2,000+-image corpus:

```bash
roboflow import -w <workspace-id> -p <project-id> \
  data/roboflow-export/wilor-detector-yolo
```

Create an Object Detection project with exactly `left_hand` and `right_hand`.
Do not upload an earlier partial export alongside this full corpus.
