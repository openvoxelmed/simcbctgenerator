# SimCBCTgenerator

SimCBCTgenerator is a framework for generating simulated CBCT data around one main path: motion model -> projection generation -> reconstruction. It exposes that core through simple user-facing pipelines for reconstruction training, segmentation, and regression workflows.

## What To Use

- `ProjectionPipeline`: generate supervised reconstruction training examples from a planning CT, including simulated projections and the corresponding reconstruction target.
- `SegmentationPipeline`: generate a simulated CBCT together with organ labels for segmentation training.
- `RegressionPipeline`: generate or use a CBCT and align CT/CBCT pairs for regression training.

The package is organized into:

- `simcbctgenerator.api` for direct Python use
- `simcbctgenerator.cli` for terminal workflows
- `simcbctgenerator.patient_setup` for patient/loading setup behind the scenes

## Installation

From the project root:

```bash
uv sync
```

If you want automatic organ segmentation support:

```bash
uv sync --extra segmentation
```

## Run The Main Examples

These examples use the bundled helper that downloads the dummy example data on first run.

Reconstruction:

```bash
uv run python examples/api_pipeline_example.py
```

Segmentation:

```bash
uv sync --extra segmentation
uv run python examples/api_segmentation_example.py
```

Regression:

```bash
uv run python examples/api_regression_example.py
```

## Pipeline Overview

### Reconstruction

Use `ProjectionPipeline` when you want supervised training data for reconstruction models. It is the pipeline for generating projections and the corresponding reconstructed CBCT target from a planning CT under controlled motion and acquisition settings.

```python
from simcbctgenerator import CBCTSystemConfig, ProjectionPipeline, Vendor

system_config = CBCTSystemConfig.for_vendor(Vendor.ELEKTA)
pipeline = ProjectionPipeline(vendor=Vendor.ELEKTA, gpu=True)
cbct = pipeline.run(
    ct_image=ct_image,
    system_config=system_config,
    output_dir="output/reconstruction",
)
```

### Segmentation

Use `SegmentationPipeline` when you want a simulated CBCT and matching organ labels for segmentation model training.

```python
from simcbctgenerator import CBCTSystemConfig, Vendor
from simcbctgenerator.api.segmentation import SegmentationPipeline

system_config = CBCTSystemConfig.for_vendor(Vendor.ELEKTA)
pipeline = SegmentationPipeline(
    method="standard",
    vendor=Vendor.ELEKTA,
    organ_list=["bowel", "bladder", "rectum"],
    priority=[1, 2, 3],
)
results = pipeline.run(
    ct_image=ct_image,
    system_config=system_config,
    output_dir="output/segmentation",
)
```

### Regression

Use `RegressionPipeline` when you want aligned CT/CBCT pairs for image-to-image regression workflows.

```python
from simcbctgenerator import CBCTSystemConfig, Vendor
from simcbctgenerator.api.regression import RegressionPipeline

system_config = CBCTSystemConfig.for_vendor(Vendor.ELEKTA)
pipeline = RegressionPipeline(vendor=Vendor.ELEKTA, gpu=True)
results = pipeline.run(
    ct_image=ct_image,
    simulate_cbct=True,
    system_config=system_config,
    output_dir="output/regression",
)
```

## CLI Entry Points

The same workflows are available from the terminal:

```bash
simcbct-pipeline-reconstruction --help
simcbct-pipeline-segmentation --help
simcbct-pipeline-regression --help
```

## Documentation

Install docs dependencies and build locally:

```bash
uv sync --group docs
uv run mkdocs build
```

Or serve locally:

```bash
uv run mkdocs serve
```
