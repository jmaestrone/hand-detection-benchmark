# Development and verification

Install all development and inference dependencies:

```bash
uv sync --extra rfdetr --group dev
```

Run the complete verification suite:

```bash
uv run ruff check src tests
uv run ruff format --check src tests
uv run pytest
uv run python -m compileall -q src
```

Run CLI help:

```bash
uv run hand-benchmark --help
uv run hand-benchmark <command> --help
```

Generated media, datasets, weights, predictions, overlays, and reports must remain ignored. Do not stage or commit anything under `data/`, `models/`, or `runs/`.

For smoke inference, use `--limit 1` on one or all splits before launching a full run.
