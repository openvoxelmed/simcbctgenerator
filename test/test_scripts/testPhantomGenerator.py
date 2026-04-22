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
Test script for RectangularPhantomGenerator
"""

import sys
from pathlib import Path
import numpy as np
import SimpleITK as sitk
import tempfile
import os

sys.path.append(str(Path(__file__).parent))

from simcbctgenerator.phantom_generator import PhantomGenerator, PhantomConfig
from testConfigPatient import get_dummy_patient
import matplotlib.pyplot as plt

if sys.platform == 'linux':
    from PyQt5.QtCore import QLibraryInfo
    os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = QLibraryInfo.location(
        QLibraryInfo.PluginsPath
    )


def get_real_phantom_path() -> Path:
    """
    Get the path to the real phantom data.

    Returns:
        Path to the real phantom file
    """
    phantom_path = Path(__file__).parent.parent / "test_data" / "phantom" / "phantom.mha"

    if not phantom_path.exists():
        raise FileNotFoundError(f"Real phantom data not found at: {phantom_path}")

    print(f"Using real phantom data: {phantom_path}")

    # Load and display info about the real phantom
    phantom_image = sitk.ReadImage(str(phantom_path))
    phantom_array = sitk.GetArrayFromImage(phantom_image)

    print(f"Real phantom shape: {phantom_array.shape}")
    print(f"Real phantom spacing: {phantom_image.GetSpacing()}")
    print(f"Real phantom value range: [{phantom_array.min():.1f}, {phantom_array.max():.1f}]")

    # Check phantom coverage
    zero_count = np.sum(phantom_array == 0)
    total_pixels = phantom_array.size
    print(f"Phantom coverage: {100*(total_pixels-zero_count)/total_pixels:.1f}% non-zero pixels")

    return phantom_path


def test_full_generation_pipeline(output_dir: Path = None):
    """Test the full rectangular phantom generation pipeline."""
    print("\n=== Testing Full Generation Pipeline ===")

    if output_dir is None:
        project_root = Path(__file__).resolve().parent.parent.parent
        output_dir = project_root / "test_output_phantom_generator"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Use real phantom data and real patient data
    phantom_file = get_real_phantom_path()
    patient = get_dummy_patient()

    # Create configuration
    config = PhantomConfig(
        phantom_path=str(phantom_file),
        intensity_factor=1.0,
        water_threshold=300.0,
        enhancement_factor=2.0
    )

    # Create generator
    generator = PhantomGenerator(config)

    # Test initialization
    generator.initialize(patient)
    assert generator.patient == patient
    print("[OK] Generator initialization with real patient works")

    # Test generation
    try:
        synthetic_cbct = generator.generate()
        assert synthetic_cbct is not None
        assert isinstance(synthetic_cbct, sitk.Image)
        assert len(synthetic_cbct.GetSize()) == 3
        print(f"[OK] Synthetic CBCT generated with shape: {synthetic_cbct.GetSize()[::-1]}")
        print(f"  Value range: [{sitk.GetArrayViewFromImage(synthetic_cbct).min():.1f}, {sitk.GetArrayViewFromImage(synthetic_cbct).max():.1f}]")

        # Test save functionality
        generator.save(output_dir, "test_synthetic_cbct")

        saved_file = output_dir / "test_synthetic_cbct_0000.nii.gz"
        assert saved_file.exists()
        print(f"[OK] Synthetic CBCT saved to: {saved_file}")

        # Verify saved file can be loaded
        saved_image = sitk.ReadImage(str(saved_file))
        saved_array = sitk.GetArrayFromImage(saved_image)
        assert saved_array.shape == synthetic_cbct.GetSize()[::-1]
        print("[OK] Saved file verification passed")

    except Exception as e:
        print(f"[FAIL] Generation failed: {e}")
        raise

    # Test reset
    generator.reset()
    assert generator.patient is None
    assert generator.synthetic_cbct is None
    print("[OK] Generator reset works correctly")


def visualize_results_with_real_data(save_path: Path = None):
    """
    Visualize results using real phantom and real patient data.

    Args:
        save_path: Optional path to save the figure
    """
    print("\n=== Generating Visualization with Real Phantom and Real Patient Data ===")

    try:
        with tempfile.TemporaryDirectory():
            # Get real phantom and real patient data
            phantom_file = get_real_phantom_path()
            patient = get_dummy_patient()

            # Generate synthetic CBCT
            config = PhantomConfig(
                phantom_path=str(phantom_file),
                intensity_factor=1.0,
                water_threshold=300.0,
                enhancement_factor=2.0
            )

            generator = PhantomGenerator(config)
            generator.initialize(patient)
            synthetic_cbct_sitk = generator.generate()
            mask_cbct = sitk.GetArrayFromImage(patient.resample_mask(synthetic_cbct_sitk, 'bowel'))

            synthetic_cbct = sitk.GetArrayFromImage(synthetic_cbct_sitk)
            # Load data for visualization
            phantom_image = generator.fov_mask#sitk.ReadImage(str(phantom_file))
            phantom_array = sitk.GetArrayFromImage(phantom_image)

            # Get real CT data from patient
            ct_array = patient.ct_array
            mask_array = patient.mask_array[...,1]

            # Create visualization with mask overlay
            fig, axes = plt.subplots(2, 4, figsize=(20, 10))

            # Get middle slices
            phantom_slice = phantom_array.shape[0] // 2
            ct_slice = ct_array.shape[0] // 2
            cbct_slice = synthetic_cbct.shape[0] // 2

            # Top row - axial slices
            axes[0, 0].imshow(phantom_array[phantom_slice], cmap='gray', vmin=-500, vmax=1000)
            axes[0, 0].set_title('Real Phantom (Axial)')
            axes[0, 0].axis('off')

            axes[0, 1].imshow(ct_array[ct_slice], cmap='gray', vmin=-1000, vmax=1000)
            axes[0, 1].set_title('Real Patient CT (Axial)')
            axes[0, 1].axis('off')

            axes[0, 2].imshow(synthetic_cbct[cbct_slice], cmap='gray', vmin=-1000, vmax=1000)
            axes[0, 2].imshow(mask_cbct[cbct_slice], cmap='Reds', alpha=0.3)
            axes[0, 2].set_title('Synthetic CBCT (Axial)')
            axes[0, 2].axis('off')

            # Show mask overlay
            axes[0, 3].imshow(ct_array[ct_slice], cmap='gray', vmin=-1000, vmax=1000)
            axes[0, 3].imshow(mask_array[ct_slice], cmap='Reds', alpha=0.3)
            axes[0, 3].set_title('Real CT + Mask (Axial)')
            axes[0, 3].axis('off')

            # Bottom row - sagittal slices
            phantom_sag = phantom_array.shape[2] // 2
            ct_sag = ct_array.shape[2] // 2
            cbct_sag = synthetic_cbct.shape[2] // 2

            axes[1, 0].imshow(phantom_array[:, :, phantom_sag], cmap='gray', vmin=-500, vmax=1000)
            axes[1, 0].set_title('Real Phantom (Sagittal)')
            axes[1, 0].axis('off')

            axes[1, 1].imshow(ct_array[:, :, ct_sag], cmap='gray', vmin=-1000, vmax=1000)
            axes[1, 1].set_title('Real Patient CT (Sagittal)')
            axes[1, 1].axis('off')

            axes[1, 2].imshow(synthetic_cbct[:, :, cbct_sag], cmap='gray', vmin=-1000, vmax=1000)
            axes[1, 2].imshow(mask_cbct[:, :, cbct_sag], cmap='Reds', alpha=0.3)
            axes[1, 2].set_title('Synthetic CBCT (Sagittal)')
            axes[1, 2].axis('off')

            # Show mask overlay sagittal
            axes[1, 3].imshow(ct_array[:, :, ct_sag], cmap='gray', vmin=-1000, vmax=1000)
            axes[1, 3].imshow(mask_array[:, :, ct_sag], cmap='Reds', alpha=0.3)
            axes[1, 3].set_title('Real CT + Mask (Sagittal)')
            axes[1, 3].axis('off')

            plt.tight_layout()

            if save_path:
                plt.savefig(save_path / "real_data_test_results.png", dpi=150, bbox_inches='tight')
                print(f"[OK] Visualization saved to: {save_path / 'real_data_test_results.png'}")

            plt.show()

            # Print statistics
            print("\nData Statistics:")
            print(f"Real phantom shape: {phantom_array.shape}, range: [{phantom_array.min():.1f}, {phantom_array.max():.1f}]")
            print(f"Real patient CT shape: {ct_array.shape}, range: [{ct_array.min():.1f}, {ct_array.max():.1f}]")
            print(f"Real patient mask shape: {mask_array.shape}, coverage: {100*np.sum(mask_array > 0)/mask_array.size:.1f}%")
            print(f"Synthetic CBCT shape: {synthetic_cbct.shape}, range: [{synthetic_cbct.min():.1f}, {synthetic_cbct.max():.1f}]")

    except Exception as e:
        print(f"[FAIL] Visualization failed: {e}")
        raise


def main(visualize: bool = False):
    """Run all tests for PhantomGenerator."""
    print("Starting PhantomGenerator tests with REAL phantom data...")

    try:

        # Full pipeline test
        test_full_generation_pipeline()

        print("\nAll PhantomGenerator tests passed!")

        # Optional visualization with real phantom and patient data
        if visualize:
            print("\nGenerating visualization with real phantom and patient data...")
            visualize_results_with_real_data()

    except Exception as e:
        print(f"\n[FAILED] Test failed: {e}")
        raise


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser("Test PhantomGenerator with real phantom data")
    parser.add_argument('--visualize', action='store_true', help='Generate visualization')
    args = parser.parse_args()

    main(visualize=args.visualize)
