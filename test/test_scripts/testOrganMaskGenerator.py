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

"""Functional test for organ mask generator - demonstrates multi-organ segmentation."""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "test" / "test_scripts"))

import SimpleITK as sitk
import numpy as np
from simcbctgenerator.organ_mask_generator import OrganMaskGenerator
from testConfigPatient import get_dummy_patient


def main():
    """
    Functional test: Generate organ masks for bowel, bladder, and rectum.
    User can visually inspect the results.
    """
    print("=" * 70)
    print("Organ Mask Generator - Functional Test")
    print("=" * 70)

    # Load test patient
    print("\n[1/4] Loading test patient...")
    patient = get_dummy_patient()
    ct_image = patient.ct_image
    print(f"      CT image size: {ct_image.GetSize()}")
    print(f"      CT spacing: {ct_image.GetSpacing()}")

    # Initialize generator
    print("\n[2/4] Initializing OrganMaskGenerator...")
    generator = OrganMaskGenerator(device="gpu", fast_mode=True)

    # Generate masks for multiple organs
    print("\n[3/4] Generating organ masks (this may take 30-60 seconds)...")
    organ_list = ['bowel', 'bladder', 'rectum']

    try:
        masks = generator.generate_multi_organ_masks(ct_image, organ_list)

        print("\n      Segmentation results:")
        for organ_name, mask in masks.items():
            mask_array = sitk.GetArrayFromImage(mask)
            voxel_count = np.sum(mask_array)
            print(f"      - {organ_name.capitalize():10s}: {voxel_count:8d} voxels")

        # Save masks for inspection
        print("\n[4/4] Saving masks for visual inspection...")
        output_dir = project_root / "test" / "output"
        output_dir.mkdir(exist_ok=True)

        for organ_name, mask in masks.items():
            output_path = output_dir / f"mask_{organ_name}.mhd"
            sitk.WriteImage(mask, str(output_path))
            print(f"      - Saved: {output_path}")

        # Create combined multi-label mask
        print("\n      Creating combined multi-label mask...")
        priorities = [1, 2, 3]  # bowel=1 (highest), bladder=2, rectum=3 (lowest)
        combined = generator.create_combined_mask(masks, priorities)
        combined_path = output_dir / "mask_combined.mhd"
        sitk.WriteImage(combined, str(combined_path))
        print(f"      - Saved: {combined_path}")
        print("      - Labels: 0=background, 1=bowel, 2=bladder, 3=rectum")

        # Test motion surrogate for PELVIS (with alpha-shape smoothing)
        print("\n      Generating motion surrogate mask for PELVIS (bowel with smoothing)...")
        motion_mask_pelvis, organ_names_pelvis = generator.generate_motion_surrogate_mask(ct_image, 'PELVIS')
        motion_path_pelvis = output_dir / "mask_bowel_motion_surrogate_pelvis.mhd"
        sitk.WriteImage(motion_mask_pelvis, str(motion_path_pelvis))
        motion_array = sitk.GetArrayFromImage(motion_mask_pelvis)
        print(f"      - PELVIS motion mask: {np.sum(motion_array):8d} voxels (smoothed)")
        print(f"      - Organs: {organ_names_pelvis}")
        print(f"      - Saved: {motion_path_pelvis}")

        # Test motion surrogate for THORAX (dict of organ masks)
        print("\n      Generating motion surrogate masks for THORAX (heart/aorta/lung/spine)...")
        motion_masks_thorax, organ_names_thorax = generator.generate_motion_surrogate_mask(ct_image, 'THORAX')
        assert isinstance(motion_masks_thorax, dict), "THORAX should return dict of masks"
        print(f"      - THORAX motion masks: {organ_names_thorax}")
        for organ_name, mask in motion_masks_thorax.items():
            organ_array = sitk.GetArrayFromImage(mask)
            motion_path_thorax = output_dir / f"mask_{organ_name}_motion_surrogate_thorax.mhd"
            sitk.WriteImage(mask, str(motion_path_thorax))
            print(f"        * {organ_name}: {np.sum(organ_array):8d} voxels - Saved: {motion_path_thorax}")

        print("\n" + "=" * 70)
        print("[SUCCESS] Functional test completed successfully!")
        print("=" * 70)
        print("\nNext steps:")
        print(f"  1. Inspect masks in: {output_dir}")
        print("  2. Load masks in ITK-SNAP or 3D Slicer alongside CT")
        print("  3. Verify organ segmentations are reasonable")
        print("=" * 70)

    except Exception as e:
        print("\n" + "=" * 70)
        print("[WARNING] Functional test completed with errors")
        print("=" * 70)
        print(f"\nError: {e}")
        print("\nNote: TotalSegmentator requires:")
        print("  - Real patient CT data with proper HU values")
        print("  - GPU with CUDA support (or use device='cpu')")
        print("  - Installation: uv sync --extra segmentation")
        print("=" * 70)


if __name__ == "__main__":
    main()
