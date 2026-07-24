# Local annotation review

## Open the exhaustive review

Generate reports first:

```bash
uv run hand-benchmark compare-models
```

Then open:

```bash
open runs/audits/head-left-v3-wilor-vs-rfdetr/review/index.html
```

The RF-DETR review contains every train, validation, and test image at the validation-selected operating threshold.

Each split provides:

- `index.html`: all images
- `false-positives.html`: FP-heavy images, ranked by count and confidence
- `false-negatives.html`: FN-heavy images
- `lowest-iou.html`: matched true positives ordered by increasing IoU
- `class-confusions.html`: left/right disagreements

## Overlay meaning

```text
green    true positive
yellow   false positive
red      false negative
magenta  spatial match with the wrong class
```

TP labels show prediction confidence and exact IoU. The thin green rectangle is the matched ground-truth box; the thicker green rectangle is the prediction.

True positives necessarily have IoU ≥ 0.50. A lower-overlap prediction appears as an FP while the unmatched ground truth appears as an FN, so review both error pages when investigating poor localization.

## Manifests

Each split contains:

- `image_manifest.csv` and `.json`
- `box_manifest.csv` and `.json`

Image rows summarize TP, FP, FN, class-confusion, minimum TP IoU, mean TP IoU, and maximum FP confidence.

Box rows contain:

- exact Roboflow filename
- source recording and timestamp
- category and predicted category
- match type
- confidence
- IoU
- prediction and ground-truth `xyxy` boxes

Raw normalized predictions down to the `0.005` inference floor are available under:

```text
runs/audits/head-left-v3-wilor-vs-rfdetr/predictions/
```

The HTML review uses the selected operating threshold, while the prediction JSONL retains the lower-confidence candidates needed for alternative threshold analysis.

## Sharing the review

The review bundle can be regenerated from ignored run artifacts. Teammates should:

1. Download and fully unzip the bundle.
2. Open `START_HERE.html`.
3. Begin with false negatives and high-confidence false positives.
4. Continue with lowest-IoU matches and class confusions.
5. Use the all-images page for exhaustive review.

The benchmark only identifies review candidates. Annotation changes must be made manually in Roboflow.
