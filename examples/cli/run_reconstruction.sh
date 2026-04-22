#!/usr/bin/env bash
cd "$(dirname "$0")/../.."
source .venv/bin/activate

simcbct-pipeline-reconstruction \
    --patient_dir list/dummy_patient.txt \
    --output_path test_output_cli_reconstruction \
    --ct_filename ct.nii.gz \
    --cbct_filename cbct.nii.gz \
    --geometry_filename geometry_elekta.xml \
    --metadata_filename metadata_elekta.yaml
