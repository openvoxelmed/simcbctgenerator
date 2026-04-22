# Examples

The examples are organized around the three main framework workflows.

## Reconstruction

Run:

```bash
uv run python examples/api_pipeline_example.py
```

Use this when you want supervised reconstruction training data from the standard motion and projection pipeline.

Output:

- `examples/output/api_pipeline/cbct_simulated.mha`
- `examples/output/api_pipeline/projections_simulated.mha`
- `examples/output/api_pipeline/motion_config.json`

## Segmentation

Run:

```bash
uv sync --extra segmentation
uv run python examples/api_segmentation_example.py
```

Use this when you want a simulated CBCT together with organ labels for segmentation training.

Output:

- `examples/output/api_segmentation/cbct_simulated.mha`
- `examples/output/api_segmentation/label_mask.mha`
- `examples/output/api_segmentation/imagesTr/`
- `examples/output/api_segmentation/labelsTr/`

## Regression

Run:

```bash
uv run python examples/api_regression_example.py
```

Use this when you want aligned CT/CBCT pairs for regression workflows.

Output:

- `examples/output/api_regression/cbct.mha`
- `examples/output/api_regression/ct_registered.mha`
- `examples/output/api_regression/fov_mask.mha`
- `examples/output/api_regression/imagesTr/`
- `examples/output/api_regression/labelsTr/`
