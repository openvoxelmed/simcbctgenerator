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
Multi-Organ Segmentation Example
==================================

Demonstrates automatic organ segmentation with TotalSegmentator.
Generates masks for bowel, bladder, and rectum and creates a combined
multi-label mask suitable for motion simulation.

**Requires** the segmentation extras::

    uv sync --extra segmentation

Run from the project root::

    uv run python examples/multi_organ_segmentation.py

Estimated runtime: 1-2 minutes (GPU) / 3-5 minutes (CPU).
"""

import sys
from pathlib import Path

# Ensure the package and test helpers are importable from a repo checkout
_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root / "src"))
sys.path.insert(0, str(_project_root / "test" / "test_scripts"))

import SimpleITK as sitk
import numpy as np

from simcbctgenerator.organ_mask_generator import OrganMaskGenerator
from testConfigPatient import get_dummy_patient


def main() -> int:
    print("=" * 70)
    print("Multi-Organ Segmentation Example")
    print("=" * 70)

    # 1. Load CT via the shared test helper (downloads if needed) ----------
    print("\n[1/5] Loading patient CT scan ...")
    try:
        patient = get_dummy_patient(region="PELVIS")
        ct_image = patient.ct_image
        print(f"  Patient ID : {patient.id}")
        print(f"  CT size    : {ct_image.GetSize()}")
        print(f"  CT spacing : {ct_image.GetSpacing()}")
    except Exception as exc:
        print(f"  [ERROR] Could not load patient CT: {exc}")
        return 1

    # 2. Generate organ masks ----------------------------------------------
    print("\n[2/5] Generating organ masks (TotalSegmentator) ...")
    print("  This may take 1-2 minutes ...")
    try:
        generator = OrganMaskGenerator(fast_mode=True, device="gpu")
        organ_list = ["bowel", "bladder", "rectum"]
        masks = generator.generate_multi_organ_masks(ct_image, organ_list)

        print("\n  Segmentation results:")
        for name, mask in masks.items():
            voxels = int(np.sum(sitk.GetArrayFromImage(mask)))
            print(f"    {name.capitalize():10s}: {voxels:>8d} voxels")
    except Exception as exc:
        print(f"  [ERROR] Segmentation failed: {exc}")
        print("  Requires TotalSegmentator — install with: uv sync --extra segmentation")
        return 1

    # 3. Combined multi-label mask -----------------------------------------
    print("\n[3/5] Creating combined multi-label mask ...")
    priorities = [1, 2, 3]  # bowel=1, bladder=2, rectum=3
    combined_mask = generator.create_combined_mask(masks, priorities)

    unique_labels = np.unique(sitk.GetArrayFromImage(combined_mask))
    print(f"  Labels : {unique_labels.tolist()}")
    print("  0=background, 1=bowel, 2=bladder, 3=rectum")

    # 4. Motion surrogate mask ---------------------------------------------
    print("\n[4/5] Generating motion surrogate mask ...")
    motion_mask, organ_names = generator.generate_motion_surrogate_mask(ct_image, "PELVIS")
    print(f"  Surrogate voxels : {int(np.sum(sitk.GetArrayFromImage(motion_mask)))}")

    # 5. Save results ------------------------------------------------------
    print("\n[5/5] Saving results ...")
    output_dir = _project_root / "examples" / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, mask in masks.items():
        p = output_dir / f"mask_{name}.mha"
        sitk.WriteImage(mask, str(p))
        print(f"  Saved: {p.name}")

    sitk.WriteImage(combined_mask, str(output_dir / "mask_combined_multilabel.mha"))
    print("  Saved: mask_combined_multilabel.mha")

    sitk.WriteImage(motion_mask, str(output_dir / "mask_motion_surrogate.mha"))
    print("  Saved: mask_motion_surrogate.mha")

    sitk.WriteImage(ct_image, str(output_dir / "ct_image.mha"))
    print("  Saved: ct_image.mha")

    print("\n" + "=" * 70)
    print("[DONE] Multi-organ segmentation complete!")
    print(f"  Output directory: {output_dir}")
    print("  View results in ITK-SNAP or 3D Slicer.")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())