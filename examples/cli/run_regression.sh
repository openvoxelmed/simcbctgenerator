#!/usr/bin/env bash
cd "$(dirname "$0")/../.."
source .venv/bin/activate

simcbct-pipeline-regression \
    --config test/test_data/dummy_regression.ini \
    --patient_path list/dummy_patient.txt \
    --output_path test_output_cli_regression
