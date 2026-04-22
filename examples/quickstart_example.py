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
Quickstart Example — Generate a Simulated CBCT (Phantom Method)
================================================================

Fastest way to produce a simulated CBCT from the downloadable dummy
patient.  Uses the *phantom* method which does not require GPU-based
projection generation.

Run from the project root::

    uv run python examples/quickstart_example.py

Estimated runtime: ~2-5 minutes (depending on hardware).
"""

import sys
from pathlib import Path

# Ensure the package and test helpers are importable from a repo checkout
_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root / "src"))
sys.path.insert(0, str(_project_root / "test" / "test_scripts"))

import SimpleITK as sitk

from simcbctgenerator import PhantomPipeline
from testConfigPatient import (
    extract_test_data,
    get_dummy_patient,
    DUMMY_CONFIG,
)


def main() -> int:
    print("=" * 70)
    print("Quickstart: Simulated CBCT Generation (Phantom Method)")
    print("=" * 70)

    # 1. Download / locate test data
    print("\n[1/4] Ensuring dummy patient data is available ...")
    extract_test_data()

    # 2. Load patient via the shared test helper
    print("\n[2/4] Loading patient ...")
    patient = get_dummy_patient(region="PELVIS")

    print(f"  Patient ID : {patient.id}")
    print(f"  CT size    : {patient.ct_image.GetSize()}")
    print(f"  CT spacing : {patient.ct_image.GetSpacing()}")

    # 3. Generate simulated CBCT using the phantom method
    print("\n[3/4] Generating simulated CBCT (phantom method) ...")
    print("  This may take a few minutes ...")

    pipeline = PhantomPipeline(phantom_config=DUMMY_CONFIG.phantom_config)

    try:
        result = pipeline.run_result(ct_image=patient.ct_image, patient_id=patient.id)
        simulated_cbct = result.cbct
    except Exception as exc:
        print(f"  [ERROR] Generation failed: {exc}")
        return 1

    print(f"  Generated CBCT size    : {simulated_cbct.GetSize()}")
    print(f"  Generated CBCT spacing : {simulated_cbct.GetSpacing()}")

    # 4. Save results
    print("\n[4/4] Saving results ...")
    output_dir = _project_root / "examples" / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    cbct_path = output_dir / "simulated_cbct_quickstart.mha"
    sitk.WriteImage(simulated_cbct, str(cbct_path))
    print(f"  Saved: {cbct_path}")

    fov_mask = result.fov_mask
    if fov_mask is not None:
        fov_path = output_dir / "fov_mask.mha"
        sitk.WriteImage(fov_mask, str(fov_path))
        print(f"  Saved: {fov_path}")

    print("\n" + "=" * 70)
    print("[DONE] Quickstart complete!")
    print(f"  Output directory: {output_dir}")
    print("  View results in ITK-SNAP or 3D Slicer.")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
