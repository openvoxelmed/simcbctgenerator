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

"""Physics noise finetuning test.

Generates a simulated CBCT from a patient CT and optionally compares it
side-by-side with a real CBCT, so you can visually inspect noise characteristics
and tune physics parameters (photon_flux, bp_amplitude, bp_std, etc.).

Usage examples
--------------
# Minimal — CT only, Elekta geometry from challenge data:
uv run python test/test_scripts/testPhysicsNoise.py \\
    --patient_dir /mnt/f/graz/data/output_challenge_Varian/stage-1/G000 \\
    --vendor varian

# With motion and contrast-media correction:
uv run python test/test_scripts/testPhysicsNoise.py \\
    --patient_dir /mnt/f/vienna/data/B000 \\
    --vendor elekta \\
    --motion PELVIS \\
    --correct_cm

# Custom output directory:
uv run python test/test_scripts/testPhysicsNoise.py \\
    --patient_dir /path/to/patient \\
    --output_dir /tmp/physics_test
"""

import argparse
import sys
from pathlib import Path

import SimpleITK as sitk

from simcbctgenerator import ProjectionPipeline
from simcbctgenerator.utils.config import MotionConfig


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a simulated CBCT for physics noise finetuning"
    )
    parser.add_argument(
        "--patient_dir", type=Path, required=True,
        help="Patient directory containing ct, geometry, metadata (and optionally cbct)"
    )
    parser.add_argument(
        "--vendor", choices=["elekta", "varian"], default="elekta",
        help="Scanner vendor (default: elekta)"
    )
    parser.add_argument(
        "--ct_filename", type=str, default="ct_def.mha",
        help="CT filename inside patient_dir (default: ct_def.mha)"
    )
    parser.add_argument(
        "--cbct_filename", type=str, default="cbct_clinical.mha",
        help="Real CBCT filename inside patient_dir for comparison (optional)"
    )
    parser.add_argument(
        "--geometry_filename", type=str, default="geometry.xml",
        help="RTK geometry XML filename (default: geometry.xml)"
    )
    parser.add_argument(
        "--metadata_filename", type=str, default="metadata.yaml",
        help="Metadata YAML filename (default: metadata.yaml)"
    )
    parser.add_argument(
        "--output_dir", type=Path, default=None,
        help="Output directory (default: <patient_dir>/physics_noise_test)"
    )
    parser.add_argument(
        "--motion", type=str, choices=["PELVIS", "THORAX"], default=None,
        help="Enable random motion simulation with the given model"
    )
    parser.add_argument(
        "--correct_cm", action="store_true",
        help="Apply contrast-media correction to CT before projection"
    )
    parser.add_argument(
        "--no_gpu", action="store_true",
        help="Disable GPU (use CPU reconstruction)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    patient_dir  = args.patient_dir
    output_dir   = args.output_dir or (patient_dir / "physics_noise_test")
    ct_path      = patient_dir / args.ct_filename
    cbct_path    = patient_dir / args.cbct_filename
    geometry_xml = patient_dir / args.geometry_filename
    metadata_yaml = patient_dir / args.metadata_filename

    for path, name in [(ct_path, "CT"), (geometry_xml, "Geometry"), (metadata_yaml, "Metadata")]:
        if not path.exists():
            print(f"[ERROR] {name} not found: {path}", file=sys.stderr)
            sys.exit(1)

    has_cbct = cbct_path.exists()
    if not has_cbct:
        print(f"No real CBCT found at {cbct_path} — will use CT for isocenter, skipping comparison.")

    print(f"Patient dir : {patient_dir}")
    print(f"Output dir  : {output_dir}")
    print(f"Vendor      : {args.vendor}")
    print(f"Motion      : {args.motion or 'none (static)'}")
    print(f"Correct CM  : {args.correct_cm}")
    print(f"GPU         : {not args.no_gpu}")

    ct_image   = sitk.ReadImage(str(ct_path))
    cbct_image = sitk.ReadImage(str(cbct_path)) if has_cbct else None

    print(f"\nCT size    : {ct_image.GetSize()}")
    print(f"CT spacing : {ct_image.GetSpacing()}")
    if cbct_image is not None:
        print(f"CBCT size  : {cbct_image.GetSize()}")

    api = ProjectionPipeline(
        vendor=args.vendor,
        gpu=not args.no_gpu,
        correct_contrast_media=args.correct_cm,
    )

    random_motion_type = MotionConfig.MotionType[args.motion] if args.motion else None

    simulated_cbct = api.run(
        ct_image=ct_image,
        cbct_image=cbct_image,
        geometry_xml=geometry_xml,
        metadata_yaml=metadata_yaml,
        output_dir=output_dir,
        cleanup_temp=True,
        random_motion_type=random_motion_type,
    )

    sim_path = output_dir / "cbct_simulated.mha"
    print(f"\nSimulated CBCT size    : {simulated_cbct.GetSize()}")
    print(f"Simulated CBCT spacing : {simulated_cbct.GetSpacing()}")
    print(f"Saved to               : {sim_path}")

    if has_cbct:
        from simcbctgenerator.registration.visualization import save_cbct_comparison
        comparison_path = output_dir / "comparison_physics_noise.png"
        save_cbct_comparison(simulated_cbct, cbct_image, comparison_path, patient_dir.name)
        print(f"Comparison image       : {comparison_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
