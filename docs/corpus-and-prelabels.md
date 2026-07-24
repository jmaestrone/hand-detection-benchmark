# Corpus extraction and WiLoR pre-labeling

## Dataset identity

The initial corpus is `head-left-raw-1fps`:

- MCAP topic: `/head_left/video`
- Source pixels: raw and unrectified
- Sampling rate: 1 fps
- Categories: `left_hand`, `right_hand`

Different sampling rates or rectified images must use a new output directory and dataset identity rather than overwrite this corpus.

## Artifact layout

```text
data/videos/                 # cached head-left MP4s and provenance
data/frames/                 # sampled images and metadata
data/predictions/            # WiLoR pre-labels
data/roboflow-export/        # Roboflow YOLO import folder
data/evaluation/             # immutable reviewed COCO exports
models/                      # local model weights
runs/                        # previews, predictions, metrics, and reviews
```

These paths are ignored by Git.

## Extract videos and frames

```bash
uv run hand-benchmark export-head-left-videos \
  --input-dir /path/to/mcaps

uv run hand-benchmark extract-frames
```

The video command writes `data/videos/<mcap-stem>.mp4` and `data/videos/metadata.jsonl`. Frame extraction writes deterministic JPEG filenames and `data/frames/metadata.jsonl`, retaining MCAP, topic, video, frame-index, timestamp, and image-size provenance.

Both commands are idempotent. Use `--overwrite` only when artifacts must be regenerated.

For a smoke run:

```bash
uv run hand-benchmark export-head-left-videos \
  --input-dir /path/to/mcaps \
  --limit 1
uv run hand-benchmark extract-frames --limit 1
```

Long jobs can be split deterministically:

```bash
uv run hand-benchmark export-head-left-videos \
  --input-dir /path/to/mcaps \
  --start-index 25 \
  --limit 25
```

The same `--start-index` and `--limit` pattern is available for frame extraction.

## Generate WiLoR pre-labels

Download the upstream left/right YOLO detector:

```bash
uv run hand-benchmark download-wilor-detector
```

Smoke-test predictions and previews:

```bash
uv run hand-benchmark predict-hands \
  --limit 20 \
  --preview-dir runs/previews/wilor
```

After reviewing the previews, run the full prediction command:

```bash
uv run hand-benchmark predict-hands
```

The JSONL output contains one row for every frame, including images with no detections.

## Export to Roboflow

```bash
uv run hand-benchmark export-roboflow-yolo
```

The export contains:

- `images/train/`
- `labels/train/`
- `data.yaml`
- `manifest.jsonl`
- empty label files for images with no predictions

Images are hard-linked when possible to avoid duplicating the source pixels.

Upload the resulting folder to an Object Detection project whose classes are exactly:

```text
left_hand
right_hand
```

Example Roboflow CLI command:

```bash
roboflow import -w <workspace-id> -p <project-id> \
  data/roboflow-export/wilor-detector-yolo
```
