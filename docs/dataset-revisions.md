# Corrected dataset revisions

Never overwrite an imported reviewed dataset. After corrections in Roboflow, export a new immutable version:

```bash
uv run hand-benchmark import-coco-dataset \
  --archive /path/to/head-left-hand-detection-ground.v4i.coco.zip \
  --output-dir data/evaluation/head-left-hand-detection-ground-v4
```

Compare it with the previous version:

```bash
uv run hand-benchmark compare-coco-revisions \
  --old-dataset-root data/evaluation/head-left-hand-detection-ground-v3 \
  --new-dataset-root data/evaluation/head-left-hand-detection-ground-v4 \
  --output-dir runs/audits/head-left-v3-to-v4
```

Images are matched through their stable pre-Roboflow filename. The report identifies:

- images added or removed
- split changes
- annotations added or removed
- left/right relabeling
- materially adjusted boxes

Recommended iteration order:

1. Correct annotations manually in Roboflow.
2. Export and import immutable v4.
3. Generate the v3-to-v4 revision report.
4. Re-run the original WiLoR and RF-DETR checkpoints against v4 ground truth.
5. Separate metric changes caused by annotations from changes caused by a new model.
6. Retrain RF-DETR only after the corrected dataset version is fixed.
7. Compare RF-DETR versions on the same ground truth.
8. Once annotation quality stabilizes, create a new recording-disjoint holdout that is not used for threshold selection or cleanup.
