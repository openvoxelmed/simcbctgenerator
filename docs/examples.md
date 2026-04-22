# Examples

These are the main user-facing examples. They map directly to the three framework pipelines.

## Reconstruction

File: `examples/api_pipeline_example.py`

Run:

```bash
uv run python examples/api_pipeline_example.py
```

Use this when you want supervised reconstruction training data from the standard motion/projection/reconstruction path.

Typical output:

- `examples/output/api_pipeline/cbct_simulated.mha`
- `examples/output/api_pipeline/projections_simulated.mha`
- `examples/output/api_pipeline/motion_config.json`

## Segmentation

File: `examples/api_segmentation_example.py`

Run:

```bash
uv sync --extra segmentation
uv run python examples/api_segmentation_example.py
```

Use this when you want a simulated CBCT plus label images for segmentation model training.

Typical output:

- `examples/output/api_segmentation/cbct_simulated.mha`
- `examples/output/api_segmentation/label_mask.mha`
- `examples/output/api_segmentation/imagesTr/`
- `examples/output/api_segmentation/labelsTr/`

## Regression

File: `examples/api_regression_example.py`

Run:

```bash
uv run python examples/api_regression_example.py
```

Use this when you want aligned CT/CBCT pairs for regression workflows. The example generates a simulated CBCT first and then registers CT to CBCT.

Typical output:

- `examples/output/api_regression/cbct.mha`
- `examples/output/api_regression/ct_registered.mha`
- `examples/output/api_regression/fov_mask.mha`
- `examples/output/api_regression/imagesTr/`
- `examples/output/api_regression/labelsTr/`

## CLI Examples

Shell-script equivalents of the three API examples live under `examples/cli/`.

```bash
uv run python examples/cli/download_data.py
bash examples/cli/run_reconstruction.sh
bash examples/cli/run_segmentation.sh
bash examples/cli/run_regression.sh
```

See `examples/cli/README.md` for details. Each script activates the uv venv
and invokes one of the `simcbct-pipeline-*` commands against the dummy patient.

## Notes

- The reconstruction and regression examples usually require the standard simulation stack to be available.
- The segmentation example additionally requires the segmentation extras.
- The regression example additionally requires Elastix and a running Docker daemon with the Impact registration container — see [Getting Started](getting-started.md).
- The examples are the preferred starting point if you want to call the framework from Python.
