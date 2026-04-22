# SimCBCTGenerator

SimCBCTGenerator is a framework for generating simulated CBCT data with a simple user-facing structure: load or prepare a patient, run one of the pipelines, and use the produced images for reconstruction training, segmentation, or regression workflows.

## Core Idea

The central simulation path is:

```text
Planning CT -> Motion model -> Projection generation -> CBCT reconstruction
```

Everything else in the package exists to make that path easier to use.

## Which Pipeline To Use

### Reconstruction

Use the reconstruction pipeline when your goal is to generate supervised reconstruction training data from a planning CT. This is the path that produces simulated projections together with a known reconstruction target.

- API: `ProjectionPipeline`
- CLI: `simcbct-pipeline-reconstruction`

### Segmentation

Use the segmentation pipeline when you need a simulated CBCT together with organ labels for segmentation model training.

- API: `SegmentationPipeline`
- CLI: `simcbct-pipeline-segmentation`

### Regression

Use the regression pipeline when you need aligned CT/CBCT image pairs for image-to-image regression tasks.

- API: `RegressionPipeline`
- CLI: `simcbct-pipeline-regression`

## Framework Structure

- `api`: user-facing Python pipeline classes
- `cli`: pipeline logic and terminal entry points for each workflow
- `patient_setup`: patient and loader setup before simulation
- `simulation`: the standard and phantom simulation backends

## Fast Start

Install:

```bash
uv sync
```

Run the main examples:

```bash
uv run python examples/api_pipeline_example.py
uv sync --extra segmentation
uv run python examples/api_segmentation_example.py
uv run python examples/api_regression_example.py
```

Continue with:

- [Getting Started](getting-started.md)
- [Examples](examples.md)
- [Pipelines](api/pipelines.md)
