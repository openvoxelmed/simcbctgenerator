#!/usr/bin/env bash
cd "$(dirname "$0")/../.."
source .venv/bin/activate

simcbct-pipeline-segmentation \
    --config test/test_data/dummy.ini \
    --patient_path list/dummy_patient.txt \
    --output_path test_output_cli_segmentation
