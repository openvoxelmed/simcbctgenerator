# CLI examples

These run the three pipelines from the terminal against the dummy patient.

## 1. Download the data

```bash
uv run python examples/cli/download_data.py
```

This fetches `ct.nii.gz`, `mask.nii.gz`, and `cbct.nii.gz` into `test/test_data/`
and writes `list/dummy_patient.txt`.

## 2. Run a pipeline

```bash
bash examples/cli/run_segmentation.sh
bash examples/cli/run_regression.sh
bash examples/cli/run_reconstruction.sh
```

Each script activates the uv venv and runs one pipeline. Outputs go to
`test_output_cli_<pipeline>/` in the project root.
