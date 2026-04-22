# Getting Started

This guide is intentionally short. The package is meant to be used through the three high-level pipelines.

## Install

From the project root:

```bash
uv sync
```

If you want automatic organ segmentation:

```bash
uv sync --extra segmentation
```

If you want to run the regression pipeline, you additionally need:

- Elastix (for rigid registration) — install the binary separately.
- Docker with the Impact registration container (for deformable registration).

The optional `registration` extra also installs a SimpleITK-Elastix build, but
the regression CLI and example shell out to a system Elastix binary.

If you want model-based evaluation metrics:

```bash
uv sync --extra evaluation
```

If you want to build the docs:

```bash
uv sync --group docs
```

## Run The Example Workflows

The examples download the dummy patient data on first run.

### Reconstruction Example

```bash
uv run python examples/api_pipeline_example.py
```

Use this when you want reconstruction training data: simulated projections and a known CBCT target generated from a planning CT with the standard simulation workflow.

### Segmentation Example

```bash
uv sync --extra segmentation
uv run python examples/api_segmentation_example.py
```

Use this when you want a simulated CBCT and matching labels for segmentation training.

### Regression Example

```bash
uv run python examples/api_regression_example.py
```

Use this when you want aligned CT/CBCT pairs for regression or synthesis workflows.

## CLI Entry Points

The same workflows are available through the terminal commands:

```bash
simcbct-pipeline-reconstruction --help
simcbct-pipeline-segmentation --help
simcbct-pipeline-regression --help
```

Runnable shell examples live in `examples/cli/` — see
`examples/cli/README.md` for a walk-through against the dummy patient.

## Docs

Serve locally:

```bash
uv run mkdocs serve
```

Build static docs:

```bash
uv run mkdocs build
```
