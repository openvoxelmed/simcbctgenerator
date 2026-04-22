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
ProjectionPipeline API Example
================================

Shows the high-level ``ProjectionPipeline`` API which is the recommended
way to generate simulated CBCTs programmatically.  The API takes raw
SimpleITK images together with per-patient metadata files and handles
everything internally (projection generation, reconstruction, optional
motion and contrast-media correction).

This example uses the downloadable dummy patient with an Elekta XVI
geometry and metadata file shipped in ``test/test_data/``.

Run from the project root::

    uv run python examples/api_pipeline_example.py

**Requirements**: GPU with CUDA support (for DRR projection generation).

Estimated runtime: 5-15 minutes (depending on GPU).
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

from simcbctgenerator import ProjectionPipeline, MotionConfig
from testConfigPatient import extract_test_data

# Optional: see what the pipeline is doing
logging.basicConfig(level=logging.INFO, format="%(name)s  %(message)s")

# Paths to the test data assets
_TEST_DATA_DIR = _project_root / "test" / "test_data"
_GEOMETRY_ELEKTA_XML = _TEST_DATA_DIR / "geometry_elekta.xml"
_METADATA_ELEKTA_YAML = _TEST_DATA_DIR / "metadata_elekta.yaml"


def main() -> int:
    print("=" * 70)
    print("ProjectionPipeline API Example (Elekta)")
    print("=" * 70)

    # 1. Ensure test data is present ---------------------------------------
    print("\n[1/4] Ensuring dummy patient data is available ...")
    extract_test_data()

    ct_path = _TEST_DATA_DIR / "ct.nii.gz"
    if not ct_path.exists():
        print(f"  [ERROR] CT file not found at {ct_path}")
        return 1

    # 2. Load images -------------------------------------------------------
    print("\n[2/4] Loading CT image ...")
    ct_image = sitk.ReadImage(str(ct_path))
    print(f"  CT size    : {ct_image.GetSize()}")
    print(f"  CT spacing : {ct_image.GetSpacing()}")

    # 3. Create the pipeline and run it ------------------------------------
    print("\n[3/4] Running ProjectionPipeline (generate projections + reconstruct) ...")
    print("  This may take 5-15 minutes on a GPU ...")

    output_dir = _project_root / "examples" / "output" / "api_pipeline"

    api = ProjectionPipeline(
        vendor="elekta",
        gpu=True,
        correct_contrast_media=False,
    )

    try:
        simulated_cbct = api.run(
            ct_image=ct_image,
            geometry_xml=_GEOMETRY_ELEKTA_XML,
            metadata_yaml=_METADATA_ELEKTA_YAML,
            output_dir=output_dir,
            cleanup_temp=True,
            # Add random pelvis breathing motion:
            random_motion_type=MotionConfig.MotionType.PELVIS,
        )
    except Exception as exc:
        print(f"\n  [ERROR] Pipeline failed: {exc}")
        print("  Make sure you have a CUDA-capable GPU and the required")
        print("  dependencies (itk-rtk-cuda124, cupy-cuda12x).")
        return 1

    # 4. Save & report -----------------------------------------------------
    print("\n[4/4] Results")
    print(f"  Simulated CBCT size    : {simulated_cbct.GetSize()}")
    print(f"  Simulated CBCT spacing : {simulated_cbct.GetSpacing()}")
    print(f"  Output directory       : {output_dir}")

    print("\n" + "=" * 70)
    print("[DONE] API pipeline example complete!")
    print("  The reconstructed CBCT is saved as cbct_simulated.mha")
    print("  The sampled motion config is saved as motion_config.json")
    print("  View the result in ITK-SNAP or 3D Slicer.")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())