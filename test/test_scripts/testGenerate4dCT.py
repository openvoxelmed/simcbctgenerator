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

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
import cupy as cp
import SimpleITK as sitk
from testConfigPatient import DUMMY_CONFIG, DUMMY_CONFIG2, get_dummy_patient
from simcbctgenerator.generate_4d_ct import FourDCTGenerator
from simcbctgenerator.patient import Patient


def test_4d_ct_generation(region="PELVIS", use_autoseg=False, num_phases=10):
    """Test 4D CT generation for specified region.

    Args:
        region: "PELVIS" or "THORAX"
        use_autoseg: If True, use TotalSegmentator for auto-segmentation
        num_phases: Number of phases to generate
    """
    print(
        f"Configuration: region={region}, use_autoseg={use_autoseg}, phases={num_phases}"
    )

    Dicom_path = Path("")

    # Select config based on region and autoseg flag
    if region == "PELVIS":
        config = DUMMY_CONFIG
    else:  # THORAX
        config = DUMMY_CONFIG2

    # Load patient
    if use_autoseg:
        # Use get_dummy_patient which handles autoseg
        patient = get_dummy_patient(region=region, use_autoseg=use_autoseg)
    else:
        # Use standard Patient loading
        patient = Patient(config.patient_config, Dicom_path)

    # Create generator and initialize
    generator = FourDCTGenerator(config.motion_config)
    generator.initialize(patient)

    # Verify motion surrogate was generated
    if region == "PELVIS":
        assert patient.motion_surrogate is not None, (
            "Motion surrogate should be auto-generated for PELVIS"
        )
        assert isinstance(patient.motion_surrogate, sitk.Image), (
            "PELVIS surrogate should be sitk.Image"
        )
        print("[OK] PELVIS motion surrogate auto-generated")
    elif region == "THORAX":
        assert patient.motion_surrogate is not None, (
            "Motion surrogate should be auto-generated for THORAX"
        )
        assert isinstance(patient.motion_surrogate, dict), (
            "THORAX surrogate should be dict"
        )
        assert set(patient.motion_surrogate.keys()) == {
            "heart",
            "aorta",
            "lung",
            "spine",
        }, "THORAX surrogate should have heart, aorta, lung, spine"
        print(
            f"[OK] THORAX motion surrogate auto-generated with organs: {list(patient.motion_surrogate.keys())}"
        )

    # Generate 4D CT phases
    print(f"Generating {num_phases} 4D CT phases...")
    cts = []
    for idx in range(num_phases):
        ct_phase = cp.asnumpy(generator.generate_dynamic_4d_CT(idx))
        cts.append(ct_phase)
        if (idx + 1) % 5 == 0:
            print(f"  Generated phase {idx + 1}/{num_phases}")

    print(f"[OK] Generated {len(cts)} 4D CT phases")

    # Optionally save phases
    output_dir = Path("/mnt/c/Users/lukas/OneDrive/data_new")
    if output_dir.exists() or len(sys.argv) > 1 and "--save" in sys.argv:
        output_dir.mkdir(exist_ok=True)
        ct_4d_list = []
        for ct in cts:
            img = sitk.GetImageFromArray(ct)
            img.SetSpacing(patient.ct_image.GetSpacing())
            ct_4d_list.append(img)
        ct4d = sitk.JoinSeries(ct_4d_list)

        ct4d.SetSpacing(list(img.GetSpacing())+[1.0])
        output_path = output_dir / "ct_4d.nrrd"
        sitk.WriteImage(ct4d, str(output_path))
        print(f"[OK] Saved phases to {output_dir}")

    return cts


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Test 4D CT Generation")
    parser.add_argument(
        "--region",
        type=str,
        default="THORAX",
        choices=["PELVIS", "THORAX"],
        help="Motion region: PELVIS or THORAX",
    )
    parser.add_argument(
        "--use-autoseg",
        action="store_true",
        help="Use TotalSegmentator for automatic organ segmentation",
    )
    parser.add_argument(
        "--num-phases",
        type=int,
        default=10,
        help="Number of 4D CT phases to generate (default: 10)",
    )
    parser.add_argument(
        "--save", action="store_true", help="Save generated phases to disk"
    )

    args = parser.parse_args()

    try:
        test_4d_ct_generation(
            region=args.region, use_autoseg=args.use_autoseg, num_phases=args.num_phases
        )
        print("\n[SUCCESS] 4D CT generation test completed!")
    except Exception as e:
        print(f"\n[FAILED] 4D CT generation test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
