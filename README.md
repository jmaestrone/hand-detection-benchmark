# Hand Detection Benchmark

Local tooling for building a reproducible hand-detection corpus from the
`/head_left/video` stream embedded in MCAP recordings. The first workflow
exports cached MP4s and samples source-pixel frames; it does not select a model,
create predictions, or integrate with Roboflow.

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

## Future annotation schema

When this corpus is pre-labeled and reviewed in Roboflow, use exactly these
categories:

- `left_hand`
- `right_hand`

The present repository does not choose a pre-labeling model or create labels.
