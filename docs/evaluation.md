# Model evaluation and metrics

## Import an immutable COCO dataset

Install the pinned RF-DETR dependency:

```bash
uv sync --extra rfdetr --group dev
```

Import a reviewed Roboflow COCO archive:

```bash
uv run hand-benchmark import-coco-dataset \
  --archive /path/to/head-left-hand-detection-ground.v3i.coco.zip
```

The importer records the archive SHA-256 and validates:

- `train`, `valid`, and `test` image counts
- referenced image files
- image dimensions and bounding boxes
- canonical `left_hand` and `right_hand` categories
- unsupported categories with annotations
- recording-level separation between splits

The unused zero-annotation Roboflow parent category is allowed. An unsupported category containing boxes is rejected. Imported annotations are never rewritten.

## Run both models

Both models run on the exact Roboflow-exported pixels with a default inference floor of `0.005`.

```bash
uv run hand-benchmark predict-wilor-coco --split all

uv run hand-benchmark predict-rfdetr-coco \
  --weights-path /path/to/checkpoint_best_total.pth \
  --split all
```

Useful options include:

```text
--split train|valid|test|all
--limit N
--batch-size N
--device auto|mps|cpu|cuda
--preview-dir PATH
```

Every split output has exactly one normalized JSONL row per source image. Each detection records the canonical and raw class, confidence, `xyxy` box, checkpoint hash, model configuration, and latency provenance.

### RF-DETR category slots

Roboflow COCO exports may retain a zero-annotation parent category. RF-DETR training allocates a classifier slot for every declared category, while the checkpoint stores only leaf class names. The loader reconstructs the training mapping from the validated COCO table and checkpoint head:

```text
slot 0 → unused parent
slot 1 → left_hand
slot 2 → right_hand
```

This prevents slot 1 from being mislabeled as right-hand and slot 2 from being discarded as background.

## Generate the comparison

```bash
uv run hand-benchmark compare-models
```

Matching is one-to-one, class-aware, and uses IoU `0.50` for TP/FP/FN metrics. A spatially matched wrong class counts as one FP plus one FN.

The validation split selects one global threshold per model:

1. Primary: maximum micro F2
2. Secondary: maximum micro F1
3. Ties: higher confidence threshold

The selected F2 threshold is then locked and applied to every split.

Reports include:

- per-class, micro, and macro precision, recall, F1, and F2
- COCO AP50, AP75, and mAP50–95
- exact confidence-breakpoint PR curves
- F1/F2 threshold charts
- RF-DETR absolute and relative deltas over WiLoR
- pooled-count and equal-split diagnostic averages
- same-device latency summaries
- error manifests and HTML galleries

Outputs are written under:

```text
runs/audits/head-left-v3-wilor-vs-rfdetr/
```

Dataset v3 is a development and annotation-audit dataset because all splits may inform corrections. Its test metrics must not be presented as a final unbiased benchmark.
