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

"""Focused tests for the new simulator backends."""

import argparse
import sys
from pathlib import Path

import SimpleITK as sitk

sys.path.append(str(Path(__file__).parent))

from simcbctgenerator import PhantomPipeline, StandardCBCTSimulator
from testConfigPatient import MODES, DUMMY_CONFIG, get_dummy_patient


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _default_output_dir(name: str) -> Path:
    out = PROJECT_ROOT / f"test_output_{name}"
    out.mkdir(parents=True, exist_ok=True)
    return out


def test_phantom_pipeline(output_dir: Path, region: str = "PELVIS", use_autoseg: bool = False) -> bool:
    print("\n=== Testing PhantomPipeline ===")
    patient = get_dummy_patient(region=region, use_autoseg=use_autoseg)
    pipeline = PhantomPipeline(phantom_config=DUMMY_CONFIG.phantom_config)
    result = pipeline.run_result(
        ct_image=patient.ct_image,
        cbct_image=None,
        output_dir=output_dir,
        patient_id=f"phantom_{region.lower()}",
    )

    cbct_path = output_dir / "cbct_simulated.mha"
    fov_path = output_dir / "fov_mask.mha"
    sitk.WriteImage(result.fov_mask, str(fov_path))

    assert result.cbct is not None
    assert cbct_path.exists()
    assert fov_path.exists()
    print(f"[OK] Phantom pipeline output: {cbct_path}")
    return True


def test_standard_simulator(output_dir: Path, motion_model: str = "PELVIS", use_autoseg: bool = False,
                           vendor: str = "elekta") -> bool:
    print("\n=== Testing StandardCBCTSimulator ===")
    mode_key = motion_model.lower()
    if use_autoseg:
        mode_key += "_autoseg"
    elif vendor == "varian":
        mode_key += "_varian"
    mode = MODES[mode_key]
    patient = get_dummy_patient(region=motion_model, use_autoseg=use_autoseg)

    simulator = StandardCBCTSimulator(gpu=True)
    try:
        result = simulator.run(
            patient=patient,
            output_dir=output_dir,
            system_config=mode.system_config,
            motion_config=mode.motion_config,
            cleanup_temp=True,
        )
    except Exception as exc:
        print(f"[WARNING] Standard simulator execution failed: {exc}")
        print("This usually means the local environment does not provide the required GPU/CUDA projector stack.")
        return False

    cbct_path = output_dir / "cbct_simulated.mha"
    assert result.cbct is not None
    assert cbct_path.exists()
    print(f"[OK] Standard simulator output: {cbct_path}")
    return True


def main():
    parser = argparse.ArgumentParser("Test simulation backends")
    parser.add_argument("--backend", choices=["phantom", "standard", "all"], default="all")
    parser.add_argument("--motion-model", choices=["PELVIS", "THORAX"], default="PELVIS")
    parser.add_argument("--vendor", choices=["elekta", "varian"], default="elekta")
    parser.add_argument("--use-autoseg", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    results = {}

    if args.backend in ("phantom", "all"):
        phantom_out = args.output_dir or _default_output_dir("phantom")
        results["phantom"] = test_phantom_pipeline(phantom_out, region=args.motion_model, use_autoseg=args.use_autoseg)

    if args.backend in ("standard", "all"):
        suffix = args.motion_model.lower()
        if args.use_autoseg:
            suffix += "_autoseg"
        elif args.vendor == "varian":
            suffix += "_varian"
        standard_out = args.output_dir or _default_output_dir(f"standard_{suffix}")
        results["standard"] = test_standard_simulator(
            standard_out,
            motion_model=args.motion_model,
            use_autoseg=args.use_autoseg,
            vendor=args.vendor,
        )

    success = all(results.values())
    if success:
        print("\nAll selected simulator tests passed.")
        raise SystemExit(0)

    print("\nOne or more simulator tests failed.")
    raise SystemExit(1)


if __name__ == "__main__":
    main()
