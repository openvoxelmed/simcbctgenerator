###############################################################################
# simcbctgenerator
#
# Copyright 2025 Lukas Zimmermann and Michael Rauter
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
###############################################################################

"""
RegressionPipeline API Example
=================================

Shows the high-level ``RegressionPipeline`` API which registers a planning
CT to a CBCT (or simulated CBCT) and produces aligned image pairs suitable
for training image-to-image regression networks (e.g. CBCT-to-CT synthesis).

The pipeline:

1. *(Optionally)* generates a simulated CBCT from the planning CT using
   DRR projection + FDK reconstruction.
2. Registers the planning CT onto the CBCT using rigid + deformable
   registration (Elastix / Docker-based).
3. Creates a field-of-view (FOV) mask and crops all images to the FOV
   bounding box.
4. Saves the aligned pair in nnU-Net folder convention.

Output layout::

    output_dir/
        imagesTr/
            {patient_id}_0000.nii.gz   # CBCT (input channel 0)
            {patient_id}_0001.nii.gz   # registered CT (input channel 1)
        labelsTr/
            {patient_id}.nii.gz        # FOV mask

This example uses the downloadable dummy patient with an Elekta XVI
geometry and metadata file shipped in ``test/test_data/``.

Run from the project root::

    uv run python examples/api_regression_example.py

**Requirements**:

- GPU with CUDA support (for CBCT simulation, if enabled).
- Elastix (for rigid registration).
- Docker with the Impact registration container (for deformable registration).

.. note::

   This example generates a simulated CBCT first and then registers the
   CT to it.  If you already have a clinical CBCT, set ``simulate_cbct=False``
   and provide it directly as ``cbct_image``.

Estimated runtime: 15-30 minutes (depending on GPU and registration method).
"""

import logging
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup – make the package and test helpers importable from a checkout
# ---------------------------------------------------------------------------
_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root / "src"))
sys.path.insert(0, str(_project_root / "test" / "test_scripts"))

import SimpleITK as sitk

from simcbctgenerator.api.regression import RegressionPipeline
from simcbctgenerator.registration import RegistrationConfig
from testConfigPatient import extract_test_data

# Optional: see what the pipeline is doing
logging.basicConfig(level=logging.INFO, format="%(name)s  %(message)s")

# Paths to the test data assets
_TEST_DATA_DIR = _project_root / "test" / "test_data"
_GEOMETRY_ELEKTA_XML = _TEST_DATA_DIR / "geometry_elekta.xml"
_METADATA_ELEKTA_YAML = _TEST_DATA_DIR / "metadata_elekta.yaml"


def main() -> int:
    print("=" * 70)
    print("RegressionPipeline API Example (Elekta)")
    print("=" * 70)

    # 1. Ensure test data is present ---------------------------------------
    print("\n[1/5] Ensuring dummy patient data is available ...")
    extract_test_data()

    ct_path = _TEST_DATA_DIR / "ct.nii.gz"
    cbct_path = _TEST_DATA_DIR / "cbct.nii.gz"
    if not ct_path.exists():
        print(f"  [ERROR] CT file not found at {ct_path}")
        return 1

    # 2. Load images -------------------------------------------------------
    print("\n[2/5] Loading CT image ...")
    ct_image = sitk.ReadImage(str(ct_path))
    print(f"  CT size    : {ct_image.GetSize()}")
    print(f"  CT spacing : {ct_image.GetSpacing()}")

    cbct_image = None
    if cbct_path.exists():
        print("\n        Loading reference CBCT image ...")
        cbct_image = sitk.ReadImage(str(cbct_path))
        print(f"  CBCT size    : {cbct_image.GetSize()}")
        print(f"  CBCT spacing : {cbct_image.GetSpacing()}")

    # 3. Create the regression pipeline ------------------------------------
    print("\n[3/5] Configuring RegressionPipeline ...")

    output_dir = _project_root / "examples" / "output" / "api_regression"

    # Registration configuration — adjust paths for your system
    reg_config = RegistrationConfig(
        threads=8,
        save_visualizations=True,
        visualization_output_dir=output_dir / "visualizations",
    )

    pipeline = RegressionPipeline(
        vendor="elekta",
        gpu=True,
        use_rigid=True,
        use_deformable=True,
        registration_config=reg_config,
    )

    use_real_cbct = cbct_image is not None

    print(f"  Vendor             : {pipeline.vendor}")
    print(f"  Rigid registration : {pipeline.use_rigid}")
    print(f"  Deformable reg.    : {pipeline.use_deformable}")
    print(f"  CBCT source        : {'real (cbct.nii.gz)' if use_real_cbct else 'simulated (DRR + FDK)'}")

    # 4. Run the full pipeline ---------------------------------------------
    print("\n[4/5] Running RegressionPipeline ...")
    if use_real_cbct:
        print("  Step 1: Register CT to real CBCT")
        print("  Step 2: Create FOV mask")
        print("  Step 3: Crop to FOV bounding box")
        print("  Step 4: Return computed images")
        print("  This may take 5-15 minutes ...")
    else:
        print("  Step 1: Generate simulated CBCT (DRR + FDK)")
        print("  Step 2: Register CT to simulated CBCT")
        print("  Step 3: Create FOV mask")
        print("  Step 4: Crop to FOV bounding box")
        print("  Step 5: Return computed images")
        print("  This may take 15-30 minutes ...")

    try:
        # Default: register against the downloaded real CBCT. If the CBCT
        # asset is unavailable, fall back to simulating one from the CT
        # (requires GPU + RTK + cupy).
        results = pipeline.run(
            ct_image=ct_image,
            cbct_image=cbct_image,
            output_dir=output_dir,
            crop_to_fov=True,
            simulate_cbct=not use_real_cbct,
            geometry_xml=_GEOMETRY_ELEKTA_XML if not use_real_cbct else None,
            metadata_yaml=_METADATA_ELEKTA_YAML if not use_real_cbct else None,
        )
    except Exception as exc:
        print(f"\n  [ERROR] Pipeline failed: {exc}")
        print("  Make sure you have:")
        print("    - A CUDA-capable GPU with itk-rtk-cuda124 and cupy-cuda12x")
        print("    - Elastix installed (for rigid registration)")
        print("    - Docker with Impact container (for deformable registration)")
        print("\n  Tip: You can run with use_deformable=False to skip Docker-based")
        print("  registration and only use rigid alignment.")
        return 1

    # 5. Save & report results
    print("\n[5/5] Saving results ...")

    cbct_out = results["cbct_image"]
    ct_reg = results["registered_ct"]
    fov = results["fov_mask"]

    # Save standalone volumes
    sitk.WriteImage(cbct_out, str(output_dir / "cbct.mha"))
    print("  Saved: cbct.mha")

    sitk.WriteImage(ct_reg, str(output_dir / "ct_registered.mha"))
    print("  Saved: ct_registered.mha")

    sitk.WriteImage(fov, str(output_dir / "fov_mask.mha"))
    print("  Saved: fov_mask.mha")

    # Save in nnU-Net format (reuse save_nnunet_format from cli.regression)
    from simcbctgenerator.cli.regression import save_nnunet_format

    patient_id = "dummy_patient"
    save_nnunet_format(
        patient_id=patient_id,
        cbct_image=cbct_out,
        ct_image=ct_reg,
        mask_image=fov,
        images_path=output_dir / "imagesTr",
        labels_path=output_dir / "labelsTr",
    )
    print(f"  Saved nnU-Net: imagesTr/{patient_id}_0000.nii.gz")
    print(f"  Saved nnU-Net: imagesTr/{patient_id}_0001.nii.gz")
    print(f"  Saved nnU-Net: labelsTr/{patient_id}.nii.gz")

    # Report
    print(f"\n  CBCT size              : {cbct_out.GetSize()}")
    print(f"  CBCT spacing           : {cbct_out.GetSpacing()}")
    print(f"  Registered CT size     : {ct_reg.GetSize()}")
    print(f"  Registered CT spacing  : {ct_reg.GetSpacing()}")
    print(f"  FOV mask size          : {fov.GetSize()}")

    if results["transform_path"] is not None:
        print(f"  Deformation transform  : {results['transform_path']}")
    else:
        print("  Deformation transform  : None (rigid only or not applicable)")

    print(f"\n  Output directory       : {output_dir}")
    print(f"  nnU-Net imagesTr/      : {output_dir / 'imagesTr'}")
    print(f"  nnU-Net labelsTr/      : {output_dir / 'labelsTr'}")

    print("\n" + "=" * 70)
    print("[DONE] Regression pipeline example complete!")
    print("  Files saved:")
    print("    cbct.mha                            — CBCT volume")
    print("    ct_registered.mha                   — registered CT")
    print("    fov_mask.mha                        — FOV mask")
    print("    imagesTr/dummy_patient_0000.nii.gz  — CBCT (nnU-Net)")
    print("    imagesTr/dummy_patient_0001.nii.gz  — CT (nnU-Net)")
    print("    labelsTr/dummy_patient.nii.gz       — FOV mask (nnU-Net)")
    print("  View the results in ITK-SNAP or 3D Slicer.")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())