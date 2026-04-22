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
SegmentationPipeline API Example
===================================

Shows the high-level ``SegmentationPipeline`` API which combines synthetic
CBCT generation with automatic organ segmentation to produce training data
for segmentation networks (e.g. nnU-Net).

The pipeline:

1. Generates a simulated CBCT from a planning CT using the standard
   DRR projection + FDK reconstruction method (or the fast phantom method).
2. Segments organs in the planning CT using TotalSegmentator.
3. Resamples the organ masks onto the CBCT grid and creates a combined
   multi-label segmentation image.
4. Saves everything in nnU-Net folder convention.

This example uses the downloadable dummy patient with an Elekta XVI
geometry and metadata file shipped in ``test/test_data/``.

Run from the project root::

    uv run python examples/api_segmentation_example.py

**Requirements**:

- GPU with CUDA support (for DRR projection generation).
- TotalSegmentator (install with ``uv sync --extra segmentation``).

Estimated runtime: 10-20 minutes (depending on GPU).
"""

import logging
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Path setup – make the package and test helpers importable from a checkout
# ---------------------------------------------------------------------------
_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root / "src"))
sys.path.insert(0, str(_project_root / "test" / "test_scripts"))

import SimpleITK as sitk

from simcbctgenerator.api.segmentation import SegmentationPipeline
from simcbctgenerator import MotionConfig
from testConfigPatient import extract_test_data

# Optional: see what the pipeline is doing
logging.basicConfig(level=logging.INFO, format="%(name)s  %(message)s")

# Paths to the test data assets
_TEST_DATA_DIR = _project_root / "test" / "test_data"
_GEOMETRY_ELEKTA_XML = _TEST_DATA_DIR / "geometry_elekta.xml"
_METADATA_ELEKTA_YAML = _TEST_DATA_DIR / "metadata_elekta.yaml"


def main() -> int:
    print("=" * 70)
    print("SegmentationPipeline API Example (Elekta)")
    print("=" * 70)

    # 1. Ensure test data is present ---------------------------------------
    print("\n[1/5] Ensuring dummy patient data is available ...")
    extract_test_data()

    ct_path = _TEST_DATA_DIR / "ct.nii.gz"
    if not ct_path.exists():
        print(f"  [ERROR] CT file not found at {ct_path}")
        return 1

    # 2. Load images -------------------------------------------------------
    print("\n[2/5] Loading CT image ...")
    ct_image = sitk.ReadImage(str(ct_path))
    print(f"  CT size    : {ct_image.GetSize()}")
    print(f"  CT spacing : {ct_image.GetSpacing()}")

    # 3. Create the segmentation pipeline ----------------------------------
    print("\n[3/5] Configuring SegmentationPipeline ...")

    output_dir = _project_root / "examples" / "output" / "api_segmentation"

    pipeline = SegmentationPipeline(
        method="standard",
        vendor="elekta",
        gpu=True,
        correct_contrast_media=False,
        # Segmentation settings
        organ_list=["bowel", "bladder", "rectum"],
        priority=[1, 2, 3],
        use_totalsegmentator=True,
        segmentation_device="gpu",
        segmentation_fast_mode=True,
    )

    print(f"  Method          : {pipeline.method}")
    print(f"  Vendor          : {pipeline.vendor}")
    print(f"  Organs          : {pipeline.organ_list}")
    print(f"  Priorities      : {pipeline.priority}")
    print(f"  Auto-segment    : TotalSegmentator (fast={pipeline.segmentation_fast_mode})")

    # 4. Run the full pipeline ---------------------------------------------
    print("\n[4/5] Running SegmentationPipeline ...")
    print("  Step 1: Generate simulated CBCT (DRR + FDK reconstruction)")
    print("  Step 2: Segment organs (TotalSegmentator)")
    print("  Step 3: Resample masks to CBCT grid")
    print("  Step 4: Return computed images")
    print("  This may take 10-20 minutes ...")

    try:
        results = pipeline.run(
            ct_image=ct_image,
            geometry_xml=_GEOMETRY_ELEKTA_XML,
            metadata_yaml=_METADATA_ELEKTA_YAML,
            output_dir=output_dir,
            # Add random pelvis breathing motion:
            random_motion_type=MotionConfig.MotionType.PELVIS,
            cleanup_temp=True,
        )
    except Exception as exc:
        print(f"\n  [ERROR] Pipeline failed: {exc}")
        print("  Make sure you have:")
        print("    - A CUDA-capable GPU with itk-rtk-cuda124 and cupy-cuda12x")
        print("    - TotalSegmentator installed (uv sync --extra segmentation)")
        return 1

    # 5. Save & report results
    print("\n[5/5] Saving results ...")

    sim_cbct = results["simulated_cbct"]
    label_mask = results["label_mask"]
    resampled_ct = results.get("resampled_ct")

    # Save standalone volumes
    sitk.WriteImage(sim_cbct, str(output_dir / "cbct_simulated.mha"))
    print("  Saved: cbct_simulated.mha")

    sitk.WriteImage(label_mask, str(output_dir / "label_mask.mha"))
    print("  Saved: label_mask.mha")

    if resampled_ct is not None:
        sitk.WriteImage(resampled_ct, str(output_dir / "ct_resampled.mha"))
        print("  Saved: ct_resampled.mha")

    # Save in nnU-Net format
    patient_id = "dummy_patient"
    images_dir = output_dir / "imagesTr"
    labels_dir = output_dir / "labelsTr"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    sitk.WriteImage(
        sitk.Cast(sim_cbct, sitk.sitkInt16),
        str(images_dir / f"{patient_id}_0000.nii.gz"),
    )
    sitk.WriteImage(
        sitk.Cast(label_mask, sitk.sitkUInt8),
        str(labels_dir / f"{patient_id}.nii.gz"),
    )
    print(f"  Saved nnU-Net: imagesTr/{patient_id}_0000.nii.gz")
    print(f"  Saved nnU-Net: labelsTr/{patient_id}.nii.gz")

    if resampled_ct is not None:
        ct_labels_dir = output_dir / "labelsTrRegression"
        ct_labels_dir.mkdir(parents=True, exist_ok=True)
        sitk.WriteImage(
            sitk.Cast(resampled_ct, sitk.sitkInt16),
            str(ct_labels_dir / f"{patient_id}.nii.gz"),
        )
        print(f"  Saved nnU-Net: labelsTrRegression/{patient_id}.nii.gz")

    # Report
    print(f"\n  Simulated CBCT size    : {sim_cbct.GetSize()}")
    print(f"  Simulated CBCT spacing : {sim_cbct.GetSpacing()}")

    unique_labels = np.unique(sitk.GetArrayFromImage(label_mask))
    print(f"  Label mask labels      : {unique_labels.tolist()}")
    print("    0 = background")
    for i, organ in enumerate(pipeline.organ_list):
        print(f"    {i + 1} = {organ}")

    if results["organ_masks"]:
        print("  Per-organ masks        :")
        for name, mask in results["organ_masks"].items():
            voxels = int(np.sum(sitk.GetArrayFromImage(mask)))
            print(f"    {name:10s}: {voxels:>8d} voxels")

    print(f"\n  Output directory       : {output_dir}")
    print(f"  nnU-Net imagesTr/      : {images_dir}")
    print(f"  nnU-Net labelsTr/      : {labels_dir}")

    print("\n" + "=" * 70)
    print("[DONE] Segmentation pipeline example complete!")
    print("  Files saved:")
    print("    cbct_simulated.mha           — reconstructed CBCT")
    print("    label_mask.mha               — multi-label organ mask")
    if resampled_ct is not None:
        print("    ct_resampled.mha             — resampled planning CT")
    print(f"    imagesTr/{patient_id}_0000.nii.gz  — CBCT (nnU-Net)")
    print(f"    labelsTr/{patient_id}.nii.gz       — labels (nnU-Net)")
    if resampled_ct is not None:
        print(f"    labelsTrRegression/{patient_id}.nii.gz — CT (nnU-Net)")
    print("  View the results in ITK-SNAP or 3D Slicer.")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())