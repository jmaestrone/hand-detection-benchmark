# Hand Detection Benchmark

Build and audit a reproducible `left_hand` / `right_hand` detection dataset from the `/head_left/video` stream in MCAP recordings.

## Setup

```bash
uv sync --extra rfdetr --group dev
```

Frame extraction requires `ffmpeg`. LZ4-compressed MCAPs also require the `lz4` CLI.

## Core workflows

Build the raw, unrectified 1-fps corpus:

```bash
uv run hand-benchmark export-head-left-videos \
  --input-dir /path/to/mcaps
uv run hand-benchmark extract-frames
```

Create RF-DETR pre-labels for Roboflow:

```bash
uv run hand-benchmark predict-rfdetr-frames \
  --weights-path /path/to/checkpoint_best_total.pth
uv run hand-benchmark export-rfdetr-roboflow-yolo
```

Compare WiLoR and RF-DETR on a reviewed COCO export:

```bash
uv run hand-benchmark import-coco-dataset --archive /path/to/dataset.coco.zip
uv run hand-benchmark predict-wilor-coco --split all
uv run hand-benchmark predict-rfdetr-coco \
  --weights-path /path/to/checkpoint_best_total.pth \
  --split all
uv run hand-benchmark compare-models
```

Open the exhaustive local RF-DETR review:

```bash
open runs/audits/head-left-v3-wilor-vs-rfdetr/review/index.html
```

## Documentation

- [Corpus extraction and model pre-labeling](docs/corpus-and-prelabels.md)
- [Model evaluation and metrics](docs/evaluation.md)
- [Local annotation review](docs/local-review.md)
- [Corrected dataset revisions](docs/dataset-revisions.md)
- [Development and verification](docs/development.md)

Raw media, datasets, weights, predictions, and run artifacts are local ignored files. Commit only source, tests, configuration, and documentation.
