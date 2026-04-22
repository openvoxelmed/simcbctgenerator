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
Test script for RTK XML geometry support with per-angle detector offsets.

This test uses the dummy patient data and the varian geometry XML file to verify
that per-angle X and Y detector offsets are correctly applied during projection
generation and reconstruction.

Usage:
    # Run with default temp directory (results are deleted after)
    uv run python test/test_scripts/testRTKGeometry.py

    # Run with custom output directory (results are preserved)
    uv run python test/test_scripts/testRTKGeometry.py --output_dir /path/to/output

    # Skip reconstruction (only generate projections)
    uv run python test/test_scripts/testRTKGeometry.py --skip_reconstruction
"""

import sys
from pathlib import Path
import argparse
import tempfile
import numpy as np

sys.path.append(str(Path(__file__).parent))

from testConfigPatient import (
    patient_config_dummy,
    motion_config_pelvis,
    system_config_varian,
    system_config_elekta,
    physics_config_varian,
    physics_config_elekta,
    reconstruction_volume_config_elekta,
    reconstruction_volume_config_varian,
    extract_test_data
)
from simcbctgenerator.patient import Patient
from simcbctgenerator.generate_4d_ct import FourDCTGenerator
from simcbctgenerator.generate_projections import DRRGenerator
from simcbctgenerator.cbct_reconstruction import SyntheticCBCTReconstruction
from simcbctgenerator.utils.config import (
    GeometryConfig, CBCTSystemConfig
)
import SimpleITK as sitk

# Set random seed for reproducibility
np.random.seed(42)

# Handle Qt platform plugin on Linux
if sys.platform == 'linux':
    import os
    try:
        from PyQt5.QtCore import QLibraryInfo
        os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = QLibraryInfo.location(
            QLibraryInfo.PluginsPath
        )
    except ImportError:
        pass

# Path to the varian geometry XML file
GEOMETRY_XML_ELEKTA_PATH = Path(__file__).parent / "geometry_elekta.xml"
GEOMETRY_XML_VARIAN_PATH = Path(__file__).parent / "geometry_varian.xml"


def create_system_config_with_xml_geometry(vendor: str = 'varian') -> CBCTSystemConfig:
    """Create a CBCTSystemConfig that uses the RTK XML geometry file.

    This overrides the fixed detector offset with per-angle offsets from the XML,
    and also uses the SID/SDD values from the XML file.
    """
    # Get base physics config for the vendor
    if vendor == 'varian':
        print("\nCreating CBCTSystemConfig with Varian XML geometry...")
        physics = physics_config_varian
        geometry = GeometryConfig(
            detector_offset=160.,  # Will be overridden by XML
            source_origin_distance=1000.,  # Will be overridden by XML
            source_detector_distance=1500.,  # Will be overridden by XML
            detector_size_h=297.984,
            detector_size_w=397.312,
            detector_pixels_h=768,
            detector_pixels_w=1024,
            start_angle=0.0,
            end_angle=360.0,
            angle_increments=0.4,
            geometry_xml_path=str(GEOMETRY_XML_VARIAN_PATH)
        )
        reconstruction_volume = reconstruction_volume_config_varian
    else:
        print("\nCreating CBCTSystemConfig with Elekta XML geometry...")
        physics = physics_config_elekta
        geometry = GeometryConfig(
            detector_offset=115.,  # Will be overridden by XML
            source_origin_distance=1000.,  # Will be overridden by XML
            source_detector_distance=1536.,  # Will be overridden by XML
            detector_size_h=409.6,
            detector_size_w=409.6,
            detector_pixels_h=512,
            detector_pixels_w=512,
            start_angle=0.0,
            end_angle=360.0,
            angle_increments=0.545,
            geometry_xml_path=str(GEOMETRY_XML_ELEKTA_PATH)
        )
        reconstruction_volume = reconstruction_volume_config_elekta

    return CBCTSystemConfig(
        physics=physics,
        geometry=geometry,
        reconstruction_volume=reconstruction_volume
    )

def generate_projections(patient: Patient, output_dir: Path, system_config: CBCTSystemConfig):
    """Generate DRR projections using the per-angle geometry."""
    print("\n=== Generating Projections with Per-Angle Offsets ===")

    # Print geometry info
    geo_data = system_config.geometry_data
    print("Geometry source: RTK XML file")
    print(f"Number of projections in XML: {geo_data.num_projections}")
    print(f"Source to isocenter (from XML): {system_config.effective_source_origin_distance}")
    print(f"Source to detector (from XML): {system_config.effective_source_detector_distance}")
    print(f"Angle range (from XML): [{geo_data.angles.min():.2f}, {geo_data.angles.max():.2f}] degrees")
    print(f"Offset X range: [{geo_data.offset_x.min():.2f}, {geo_data.offset_x.max():.2f}]")
    print(f"Offset Y range: [{geo_data.offset_y.min():.4f}, {geo_data.offset_y.max():.4f}]")

    # Initialize 4D CT generator
    generator = FourDCTGenerator(motion_config_pelvis)
    generator.initialize(patient)

    # Create projections directory
    projections_dir = output_dir / "projections"
    projections_dir.mkdir(parents=True, exist_ok=True)

    # Initialize DRR generator with XML geometry
    drr_generator = DRRGenerator(
        projections_dir,
        system_config=system_config
    )

    # Generate projections
    print(f"\nGenerating projections to: {projections_dir}")
    drr_generator.generate_all_projections(generator.patient, generator)

    # Count generated projections
    projection_files = list(projections_dir.glob("*.mhd"))
    print(f"Generated {len(projection_files)} projection files")

    return projections_dir


def reconstruct_cbct(projections_dir: Path, output_dir: Path, system_config: CBCTSystemConfig, fixed:bool=False):
    """Reconstruct CBCT from projections using per-angle geometry."""
    print("\n=== Reconstructing CBCT with Per-Angle Offsets ===")

    reconstruction_dir = output_dir / "reconstruction"
    reconstruction_dir.mkdir(parents=True, exist_ok=True)

    # Create reconstructor
    reconstructor = SyntheticCBCTReconstruction(system_config, gpu=True)

    # Reconstruct
    print(f"Reconstructing from: {projections_dir}")
    recon = reconstructor.reconstruct(projections_dir)

    # Save reconstruction
    output_file = reconstruction_dir / f"recon_rtk_geometry_{'fixed_' if fixed else ''}0000.nii.gz"
    sitk.WriteImage(recon, str(output_file))
    print(f"Reconstruction saved to: {output_file}")

    # Print stats
    recon_array = sitk.GetArrayFromImage(recon)
    print(f"Reconstruction shape: {recon_array.shape}")
    print(f"Reconstruction value range: [{recon_array.min():.1f}, {recon_array.max():.1f}]")

    return reconstruction_dir


def compare_with_fixed_offset(patient: Patient, output_dir: Path, fixed_system_config: CBCTSystemConfig):
    """Optional: Generate projections with fixed offset for comparison."""
    print("\n=== Generating Reference Projections (Fixed Offset) ===")

    print(f"Fixed offset X: {fixed_system_config.geometry.detector_offset}")
    print(f"Has XML geometry: {fixed_system_config.has_xml_geometry}")

    # Initialize 4D CT generator
    generator = FourDCTGenerator(motion_config_pelvis)
    generator.initialize(patient)

    # Create projections directory
    projections_dir = output_dir / "projections_fixed_offset"
    projections_dir.mkdir(parents=True, exist_ok=True)

    # Initialize DRR generator
    drr_generator = DRRGenerator(
        projections_dir,
        system_config=fixed_system_config
    )

    # Generate projections
    print(f"\nGenerating reference projections to: {projections_dir}")
    drr_generator.generate_all_projections(generator.patient, generator)

    return projections_dir


def main(output_dir: Path, skip_reconstruction: bool = False, compare_fixed: bool = False, vendor: str = 'varian'):
    """Main test function."""
    print("=" * 60)
    print("RTK XML Geometry Test - Per-Angle Detector Offsets")
    print("=" * 60)

    # Verify geometry XML exists
    GEOMETRY_XML_PATH = GEOMETRY_XML_VARIAN_PATH if vendor == 'varian' else GEOMETRY_XML_ELEKTA_PATH
    if not GEOMETRY_XML_PATH.exists():
        raise FileNotFoundError(f"Geometry XML file not found: {GEOMETRY_XML_PATH}")
    print(f"\nUsing geometry XML: {GEOMETRY_XML_PATH}")

    # Extract test data if needed
    extract_test_data()

    # Load patient
    data_path = Path(__file__).parent.parent / 'test_data'
    patient = Patient(patient_config_dummy, data_path)

    if not patient.valid:
        raise ValueError("Failed to load patient data")
    print(f"Loaded patient: {patient.id}")

    # Create system config with XML geometry
    system_config = create_system_config_with_xml_geometry(vendor=vendor)

    # Generate projections
    projections_dir = generate_projections(patient, output_dir, system_config)

    # Optionally compare with fixed offset
    if compare_fixed:
        fixed_system_config = system_config_varian if vendor == 'varian' else system_config_elekta
        projections_dir_fixed = compare_with_fixed_offset(patient, output_dir, fixed_system_config)
        reconstruct_cbct(projections_dir_fixed, output_dir, fixed_system_config, fixed=True)

    # Reconstruct
    if not skip_reconstruction:
        reconstruct_cbct(projections_dir, output_dir, system_config)
    else:
        print("\nSkipping reconstruction (--skip_reconstruction flag set)")

    print("\n" + "=" * 60)
    print("Test completed successfully!")
    print(f"Output directory: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test RTK XML geometry support with per-angle detector offsets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run with temp directory (results deleted after)
    uv run python test/test_scripts/testRTKGeometry.py

    # Run with custom output directory (results preserved)
    uv run python test/test_scripts/testRTKGeometry.py --output_dir ./rtk_test_output

    # Only generate projections, skip reconstruction
    uv run python test/test_scripts/testRTKGeometry.py --output_dir ./output --skip_reconstruction

    # Also generate reference projections with fixed offset for comparison
    uv run python test/test_scripts/testRTKGeometry.py --output_dir ./output --compare_fixed
        """
    )

    parser.add_argument(
        '--output_dir', '-o',
        type=str,
        default=None,
        help='Output directory for projections and reconstruction. '
             'If not specified, uses a temp directory that is deleted after the test.'
    )

    parser.add_argument(
        '--vendor', '-v',
        type=str,
        default='varian',
        choices=['varian', 'elekta'],
        help='Vendor geometry to use (varian or elekta). Default is varian.'
    )

    parser.add_argument(
        '--skip_reconstruction',
        action='store_true',
        help='Skip reconstruction step (only generate projections)'
    )

    parser.add_argument(
        '--compare_fixed',
        action='store_true',
        help='Also generate projections with fixed offset for comparison'
    )

    args = parser.parse_args()

    # Determine output directory
    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Using output directory: {output_dir}")
        main(output_dir, args.skip_reconstruction, args.compare_fixed, args.vendor)
    else:
        # Use temp directory
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            print(f"Using temporary directory: {output_dir}")
            print("(Results will be deleted after test completes)")
            print("Use --output_dir to preserve results\n")
            main(output_dir, args.skip_reconstruction, args.compare_fixed)
