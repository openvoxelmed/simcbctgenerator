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
Integration test for the separate pipelines (segmentation, regression, reconstruction).
Tests that pipelines run without crashing using default config files.
"""

import sys
from pathlib import Path
import argparse
import subprocess

from testConfigPatient import extract_test_data

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def create_dummy_patient_list(temp_dir: Path) -> Path:
    """Create a patient list file pointing to test data for dummy testing."""
    patient_list_file = temp_dir / "dummy_patients.txt"

    # Point to the actual test data directory
    test_data_path = Path(__file__).parent.parent / "test_data"

    with open(patient_list_file, 'w') as f:
        f.write(str(test_data_path.absolute()) + "\n")

    return patient_list_file


def test_segmentation_pipeline(dummy: bool, patient_path: str | Path, method: str, temp_dir: Path,
                              motion_model: str = 'PELVIS', use_autoseg: bool = False):
    """Test the segmentation pipeline."""
    print(f"\n=== Testing Segmentation Pipeline (method={method}, motion={motion_model}, autoseg={use_autoseg}) ===")

    output_path = temp_dir / f"output_segmentation_{method}_{motion_model.lower()}"
    drr_path = temp_dir / f"projections_{method}_{motion_model.lower()}"
    if use_autoseg:
        output_path = Path(str(output_path) + "_autoseg")
        drr_path = Path(str(drr_path) + "_autoseg")

    output_path.mkdir(exist_ok=True)
    drr_path.mkdir(exist_ok=True)

    pipeline_script = Path(__file__).parent.parent.parent / "src" / "simcbctgenerator" / "cli" / "segmentation.py"

    # Select config based on method and mode
    if dummy:
        # Use dummy.ini config file with DUMMY modality
        dummy_config = Path(__file__).parent.parent / "test_data" / "dummy.ini"
        cmd = [
            "uv", "run", "python", str(pipeline_script),
            "--init", str(dummy_config),
            "--patient_path", str(patient_path),
            "--output_path", str(output_path),
            "--drr_path", str(drr_path),
            "--method", method,
            "--motion_type", motion_model
        ]
        if use_autoseg:
            cmd.append("--use_totalsegmentator")
    else:
        # Use preset configs for real patient data
        if method == "phantom":
            config = "config-phantom"
        else:
            config = "standard"

        cmd = [
            "uv", "run", "python", str(pipeline_script),
            "--config", config,
            "--patient_path", str(patient_path),
            "--output_path", str(output_path),
            "--drr_path", str(drr_path),
            "--method", method,
            "--store_ct",
            "--store_real_cbct"
        ]

    print(f"Running: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

        if dummy:
            if result.returncode != 0:
                print(f"[FAIL] Segmentation pipeline ({method}) failed (exit code: {result.returncode})")
                print("STDOUT:", result.stdout[-1000:] if len(result.stdout) > 1000 else result.stdout)
                print("STDERR:", result.stderr[-1000:] if len(result.stderr) > 1000 else result.stderr)
                return False
            images_dir = output_path / "imagesTr"
            output_files = list(images_dir.glob("*.nii.gz")) if images_dir.exists() else []
            if not output_files:
                print(f"[FAIL] Segmentation pipeline ({method}) exited 0 but created no output in imagesTr/")
                return False
            print(f"[OK] Segmentation pipeline ({method}) completed, {len(output_files)} image(s) in imagesTr/")
            return True
        else:
            # In real mode, expect success
            if result.returncode == 0:
                print(f"[OK] Segmentation pipeline ({method}) executed successfully")
                print("STDOUT:", result.stdout[-500:] if len(result.stdout) > 500 else result.stdout)

                # Check for output files
                images_dir = output_path / "imagesTr"
                labels_dir = output_path / "labelsTr"

                if images_dir.exists() and labels_dir.exists():
                    image_files = list(images_dir.glob("*.nii.gz"))
                    label_files = list(labels_dir.glob("*.nii.gz"))

                    if image_files or label_files:
                        print(f"[OK] Output files created: {len(image_files)} images, {len(label_files)} labels")
                        return True
                    else:
                        print("[WARNING] No output files generated (possibly no valid patients)")
                        return True  # Still OK if patient list was empty
                else:
                    print("[WARNING] Output directories not created")
                    return True  # Still OK if no patients processed
            else:
                print(f"[FAIL] Segmentation pipeline ({method}) failed with exit code {result.returncode}")
                print("STDOUT:", result.stdout[-500:] if len(result.stdout) > 500 else result.stdout)
                print("STDERR:", result.stderr[-500:] if len(result.stderr) > 500 else result.stderr)
                return False

    except subprocess.TimeoutExpired:
        print(f"[FAIL] Segmentation pipeline ({method}) timed out")
        return False
    except FileNotFoundError:
        print("[FAIL] Pipeline script not found or uv not available")
        return False


def test_regression_pipeline(dummy: bool, patient_path: str | Path, temp_dir: Path,
                            motion_model: str = 'PELVIS', use_autoseg: bool = False):
    """Test the regression pipeline."""
    print(f"\n=== Testing Regression Pipeline (motion={motion_model}, autoseg={use_autoseg}) ===")

    output_path = temp_dir / f"output_regression_{motion_model.lower()}"
    if use_autoseg:
        output_path = Path(str(output_path) + "_autoseg")
    output_path.mkdir(exist_ok=True)

    pipeline_script = Path(__file__).parent.parent.parent / "src" / "simcbctgenerator" / "cli" / "regression.py"

    if dummy:
        # Use dummy.ini config file with DUMMY modality.
        # Note: regression is a static CT↔CBCT registration, so no --motion_type.
        dummy_config = Path(__file__).parent.parent / "test_data" / "dummy_regression.ini"
        cmd = [
            "uv", "run", "python", str(pipeline_script),
            "--init", str(dummy_config),
            "--patient_path", str(patient_path),
            "--output_path", str(output_path),
            "--overwrite",
        ]
        if use_autoseg:
            cmd.append("--use_totalsegmentator")
    else:
        # Use preset config for real patient data
        cmd = [
            "uv", "run", "python", str(pipeline_script),
            "--config", "regression",
            "--patient_path", str(patient_path),
            "--output_path", str(output_path),
            "--overwrite"
        ]

    print(f"Running: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=580)

        if dummy:
            if result.returncode != 0:
                print(f"[FAIL] Regression pipeline failed (exit code: {result.returncode})")
                print("STDOUT:", result.stdout[-1000:] if len(result.stdout) > 1000 else result.stdout)
                print("STDERR:", result.stderr[-1000:] if len(result.stderr) > 1000 else result.stderr)
                return False
            images_dir = output_path / "imagesTr"
            labels_dir = output_path / "labelsTr"
            image_files = list(images_dir.glob("*.nii.gz")) if images_dir.exists() else []
            label_files = list(labels_dir.glob("*.nii.gz")) if labels_dir.exists() else []
            if not image_files or not label_files:
                print(f"[FAIL] Regression pipeline exited 0 but produced no output "
                      f"(imagesTr: {len(image_files)}, labelsTr: {len(label_files)})")
                return False
            print(f"[OK] Regression pipeline completed, "
                  f"{len(image_files)} image(s) in imagesTr/, {len(label_files)} label(s) in labelsTr/")
            return True
        else:
            # In real mode, expect success
            if result.returncode == 0:
                print("[OK] Regression pipeline executed successfully")

                # Check for output files
                images_dir = output_path / "imagesTr"

                if images_dir.exists():
                    image_files = list(images_dir.glob("*.nii.gz"))
                    if image_files:
                        print(f"[OK] Output files created: {len(image_files)} images")
                        return True
                    else:
                        print("[WARNING] No output files generated")
                        return True
                else:
                    print("[WARNING] Output directories not created")
                    return True
            else:
                print(f"[FAIL] Regression pipeline failed with exit code {result.returncode}")
                print("STDOUT:", result.stdout[-500:] if len(result.stdout) > 500 else result.stdout)
                print("STDERR:", result.stderr[-500:] if len(result.stderr) > 500 else result.stderr)
                return False

    except subprocess.TimeoutExpired:
        print("[FAIL] Regression pipeline timed out (>580s)")
        return False
    except FileNotFoundError:
        print("[FAIL] Pipeline script not found or uv not available")
        return False


def create_dummy_reconstruction_patient(temp_dir: Path) -> Path:
    """Create a patient directory with CT, geometry and metadata for reconstruction testing.

    Returns path to a patient list file pointing to the created patient directory.
    """
    test_data = Path(__file__).parent.parent / "test_data"
    patient_dir = temp_dir / "dummy_recon_patient"
    patient_dir.mkdir(exist_ok=True)

    import shutil
    shutil.copy(test_data / "ct.nii.gz",           patient_dir / "ct.nii.gz")
    shutil.copy(test_data / "geometry_elekta.xml",  patient_dir / "geometry_elekta.xml")
    shutil.copy(test_data / "metadata_elekta.yaml", patient_dir / "metadata_elekta.yaml")

    patient_list = temp_dir / "dummy_recon_patients.txt"
    patient_list.write_text(str(patient_dir) + "\n")
    return patient_list


def test_reconstruction_pipeline(dummy: bool, patient_path: str | Path, temp_dir: Path,
                                motion_model: str = 'PELVIS', use_autoseg: bool = False):
    """Test the reconstruction pipeline."""
    print(f"\n=== Testing Reconstruction Pipeline (motion={motion_model}, autoseg={use_autoseg}) ===")

    output_path = temp_dir / f"output_reconstruction_{motion_model.lower()}"
    if use_autoseg:
        output_path = Path(str(output_path) + "_autoseg")
    output_path.mkdir(exist_ok=True)

    pipeline_script = Path(__file__).parent.parent.parent / "src" / "simcbctgenerator" / "cli" / "reconstruction.py"

    if dummy:
        recon_patient_list = create_dummy_reconstruction_patient(temp_dir)
        cmd = [
            "uv", "run", "python", str(pipeline_script),
            "--patient_dir", str(recon_patient_list),
            "--output_path", str(output_path),
            "--ct_filename", "ct.nii.gz",
            "--geometry_filename", "geometry_elekta.xml",
            "--metadata_filename", "metadata_elekta.yaml",
            "--motion_type", motion_model,
        ]
        if use_autoseg:
            cmd.append("--use_totalsegmentator")
    else:
        # Use preset config for real patient data
        cmd = [
            "uv", "run", "python", str(pipeline_script),
            "--patient_dir", str(patient_path),
            "--output_path", str(output_path),
            "--ct_filename", "ct.mha",
            "--geometry_filename", "geometry.xml",
            "--metadata_filename", "metadata.yaml",
        ]

    print(f"Running: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=580)

        if result.returncode == 0:
            output_files = list(output_path.glob("**/*.mha")) + list(output_path.glob("**/*.nii.gz"))
            if output_files:
                print(f"[OK] Reconstruction pipeline completed, {len(output_files)} output file(s)")
                return True
            else:
                print("[FAIL] Reconstruction pipeline exited 0 but created no output files")
                return False
        else:
            print(f"[FAIL] Reconstruction pipeline failed (exit code: {result.returncode})")
            print("STDOUT:", result.stdout[-1000:] if len(result.stdout) > 1000 else result.stdout)
            print("STDERR:", result.stderr[-1000:] if len(result.stderr) > 1000 else result.stderr)
            return False

    except subprocess.TimeoutExpired:
        print("[FAIL] Reconstruction pipeline timed out (>580s)")
        return False
    except FileNotFoundError:
        print("[FAIL] Pipeline script not found or uv not available")
        return False


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Integration test for CBCT pipelines')

    parser.add_argument('--pipeline',
                       choices=['segmentation', 'regression', 'reconstruction', 'all'],
                       default='all',
                       help='Which pipeline to test')
    parser.add_argument('--dummy',
                       action='store_true',
                       help='Run minimal smoke test without real patient data')
    parser.add_argument('--patient_path',
                       type=str,
                       default=None,
                       help='Path to patient list file for testing with real data')
    parser.add_argument('--method',
                       choices=['standard', 'phantom'],
                       default='standard',
                       help='Method for segmentation pipeline testing')
    parser.add_argument('--motion-model',
                       type=str,
                       choices=['PELVIS', 'THORAX'],
                       default='PELVIS',
                       help='Motion model region: PELVIS or THORAX')
    parser.add_argument('--use-autoseg',
                       action='store_true',
                       help='Use TotalSegmentator for automatic organ segmentation')
    parser.add_argument('--output_dir',
                       type=str,
                       default=None,
                       help='Output directory for test results (if not specified, uses temporary directory)')

    return parser.parse_args()


def main():
    """Run integration tests."""
    args = parse_arguments()

    print("Starting Pipeline Integration Tests...")
    print(f"Mode: {'DUMMY (smoke test)' if args.dummy else 'REAL PATIENT DATA'}")
    print(f"Testing: {args.pipeline}")
    print(f"Motion Model: {args.motion_model}")
    print(f"Use TotalSegmentator: {args.use_autoseg}")

    # Validate arguments
    if not args.dummy and not args.patient_path:
        raise ValueError("--patient_path required when not in dummy mode")

    results = {}

    try:
        # Determine output directory: use specified output_dir or create temporary directory
        if args.output_dir:
            data_path = Path(args.output_dir)
        else:
            data_path = PROJECT_ROOT / "test_output_pipeline_integration"
        data_path.mkdir(parents=True, exist_ok=True)
        print(f"Using output directory: {data_path}")

        # Create patient list
        if args.dummy:
            # Ensure dummy CT / mask / CBCT are present (matches examples/cli/download_data.py)
            extract_test_data()
            patient_list = create_dummy_patient_list(data_path)
            print(f"Created dummy patient list: {patient_list}")
        else:
            patient_list = args.patient_path
            print(f"Using patient list: {patient_list}")

        # Run selected pipeline tests
        if args.pipeline in ['segmentation', 'all']:
            results['segmentation'] = test_segmentation_pipeline(
                args.dummy, patient_list, args.method, data_path,
                args.motion_model, args.use_autoseg
            )

        if args.pipeline in ['regression', 'all']:
            results['regression'] = test_regression_pipeline(
                args.dummy, patient_list, data_path,
                args.motion_model, args.use_autoseg
            )

        if args.pipeline in ['reconstruction', 'all']:
            results['reconstruction'] = test_reconstruction_pipeline(
                args.dummy, patient_list, data_path,
                args.motion_model, args.use_autoseg
            )

        # Print summary
        print("\n" + "="*60)
        print("INTEGRATION TEST SUMMARY")
        print("="*60)
        for pipeline_name, passed in results.items():
            status = "✓ PASS" if passed else "✗ FAIL"
            print(f"{pipeline_name.upper():20s}: {status}")
        print("="*60)

        # Exit with appropriate code
        all_passed = all(results.values())
        if all_passed:
            print("\nAll integration tests passed!")
            sys.exit(0)
        else:
            print("\nSome integration tests failed!")
            sys.exit(1)

    except Exception as e:
        print(f"\n[FAILED] Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
