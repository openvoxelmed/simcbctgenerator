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
Test runner for all tests
"""

import sys
import subprocess
import argparse
from pathlib import Path
import time

def run_test(test_script: str, args: list = None, timeout: int = 300):
    """
    Run a test script and capture results.

    Args:
        test_script: Name of the test script to run
        args: Additional arguments for the test script
        timeout: Timeout in seconds

    Returns:
        bool: True if test passed, False otherwise
    """
    print(f"\n{'='*60}")
    print(f"Running: {test_script}")
    print(f"{'='*60}")

    cmd = ["uv", "run", "python", test_script]
    if args:
        cmd.extend(args)

    start_time = time.time()

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        end_time = time.time()
        duration = end_time - start_time

        print(f"Duration: {duration:.2f}s")

        if result.returncode == 0:
            print("[PASSED]")
            print("STDOUT:")
            print(result.stdout)
            return True
        else:
            print("[FAILED]")
            print("STDOUT:")
            print(result.stdout)
            print("STDERR:")
            print(result.stderr)
            return False

    except subprocess.TimeoutExpired:
        end_time = time.time()
        duration = end_time - start_time
        print(f"[TIMEOUT] after {duration:.2f}s")
        return False
    except Exception as e:
        print(f"[ERROR]: {e}")
        return False


def main():
    """Run all tests for the rectangular phantom functionality."""
    parser = argparse.ArgumentParser("Run all rectangular phantom tests")
    parser.add_argument('--quick', action='store_true',
                       help='Run only quick tests')
    parser.add_argument('--test', type=str, help='Run specific test')
    parser.add_argument('--skip-autoseg', action='store_true',
                       help='Skip autosegmentation tests (requires TotalSegmentator)')

    args = parser.parse_args()

    # Change to test scripts directory
    test_dir = Path(__file__).parent

    try:
        sys.path.insert(0, str(test_dir))

        print("Starting Test Suite")
        print(f"Working directory: {test_dir}")

        results = {}

        def integration_test(name: str, motion_model: str, pipeline: str, quick: bool,
                             requires_autoseg: bool = False, use_autoseg: bool = False,
                             timeout: int = 720):
            args = ['--dummy', '--motion-model', motion_model, '--pipeline', pipeline]
            if use_autoseg:
                args.append('--use-autoseg')
            return {
                'name': name,
                'script': './test/test_scripts/testPipelineIntegration.py',
                'args': args,
                'quick': quick,
                'timeout': timeout,
                'requires_autoseg': requires_autoseg,
            }

        # Define test configurations
        tests = [
            {
                'name': 'PhantomGenerator',
                'script': './test/test_scripts/testPhantomGenerator.py',
                'args': [],
                'quick': False,
                'timeout': 120,
                'requires_autoseg': False
            },
            # Phantom backend
            {
                'name': 'SimulationBackend_phantom',
                'script': './test/test_scripts/testSimulationBackends.py',
                'args': ['--backend', 'phantom'],
                'quick': False,
                'timeout': 180,
                'requires_autoseg': False
            },
            # Elekta tests
            {
                'name': 'SimulationBackend_standard_elekta_pelvis',
                'script': './test/test_scripts/testSimulationBackends.py',
                'args': ['--backend', 'standard', '--motion-model', 'PELVIS', '--vendor', 'elekta'],
                'quick': False,
                'timeout': 320,
                'requires_autoseg': False
            },
            {
                'name': 'SimulationBackend_standard_elekta_thorax',
                'script': './test/test_scripts/testSimulationBackends.py',
                'args': ['--backend', 'standard', '--motion-model', 'THORAX', '--vendor', 'elekta'],
                'quick': False,
                'timeout': 320,
                'requires_autoseg': False
            },
            # Varian tests
            {
                'name': 'SimulationBackend_standard_varian_pelvis',
                'script': './test/test_scripts/testSimulationBackends.py',
                'args': ['--backend', 'standard', '--motion-model', 'PELVIS', '--vendor', 'varian'],
                'quick': False,
                'timeout': 320,
                'requires_autoseg': False
            },
            {
                'name': 'SimulationBackend_standard_varian_thorax',
                'script': './test/test_scripts/testSimulationBackends.py',
                'args': ['--backend', 'standard', '--motion-model', 'THORAX', '--vendor', 'varian'],
                'quick': False,
                'timeout': 320,
                'requires_autoseg': False
            },
            # Autoseg tests (Elekta)
            {
                'name': 'SimulationBackend_standard_pelvis_autoseg',
                'script': './test/test_scripts/testSimulationBackends.py',
                'args': ['--backend', 'standard', '--motion-model', 'PELVIS', '--use-autoseg'],
                'quick': False,
                'timeout': 360,
                'requires_autoseg': True
            },
            {
                'name': 'SimulationBackend_standard_thorax_autoseg',
                'script': './test/test_scripts/testSimulationBackends.py',
                'args': ['--backend', 'standard', '--motion-model', 'THORAX', '--use-autoseg'],
                'quick': False,
                'timeout': 360,
                'requires_autoseg': True
            },
            integration_test(
                name='Pipeline_Integration_segmentation_elekta_pelvis',
                motion_model='PELVIS',
                pipeline='segmentation',
                quick=True,
                timeout=360,
            ),
            integration_test(
                name='Pipeline_Integration_regression_elekta_pelvis',
                motion_model='PELVIS',
                pipeline='regression',
                quick=True,
                timeout=360,
            ),
            integration_test(
                name='Pipeline_Integration_reconstruction_elekta_pelvis',
                motion_model='PELVIS',
                pipeline='reconstruction',
                quick=True,
                timeout=420,
            ),
            integration_test(
                name='Pipeline_Integration_segmentation_elekta_thorax',
                motion_model='THORAX',
                pipeline='segmentation',
                quick=True,
                timeout=360,
            ),
            integration_test(
                name='Pipeline_Integration_regression_elekta_thorax',
                motion_model='THORAX',
                pipeline='regression',
                quick=True,
                timeout=360,
            ),
            integration_test(
                name='Pipeline_Integration_reconstruction_elekta_thorax',
                motion_model='THORAX',
                pipeline='reconstruction',
                quick=True,
                timeout=420,
            ),
            integration_test(
                name='Pipeline_Integration_segmentation_pelvis_autoseg',
                motion_model='PELVIS',
                pipeline='segmentation',
                quick=False,
                use_autoseg=True,
                requires_autoseg=True,
                timeout=720,
            ),
            integration_test(
                name='Pipeline_Integration_regression_pelvis_autoseg',
                motion_model='PELVIS',
                pipeline='regression',
                quick=False,
                use_autoseg=True,
                requires_autoseg=True,
                timeout=720,
            ),
            integration_test(
                name='Pipeline_Integration_reconstruction_pelvis_autoseg',
                motion_model='PELVIS',
                pipeline='reconstruction',
                quick=False,
                use_autoseg=True,
                requires_autoseg=True,
                timeout=720,
            ),
            integration_test(
                name='Pipeline_Integration_segmentation_thorax_autoseg',
                motion_model='THORAX',
                pipeline='segmentation',
                quick=False,
                use_autoseg=True,
                requires_autoseg=True,
                timeout=720,
            ),
            integration_test(
                name='Pipeline_Integration_regression_thorax_autoseg',
                motion_model='THORAX',
                pipeline='regression',
                quick=False,
                use_autoseg=True,
                requires_autoseg=True,
                timeout=720,
            ),
            integration_test(
                name='Pipeline_Integration_reconstruction_thorax_autoseg',
                motion_model='THORAX',
                pipeline='reconstruction',
                quick=False,
                use_autoseg=True,
                requires_autoseg=True,
                timeout=720,
            ),
        ]

        # Filter tests based on arguments
        if args.test:
            tests = [t for t in tests if args.test.lower() in t['name'].lower()]
            if not tests:
                print(f"No tests found matching: {args.test}")
                return 1
        elif args.quick:
            tests = [t for t in tests if t['quick']]

        # Filter out autoseg tests if --skip-autoseg is specified
        if args.skip_autoseg:
            tests = [t for t in tests if not t.get('requires_autoseg', False)]
            print("Skipping autosegmentation tests (use without --skip-autoseg to include them)")

        # Run tests
        total_tests = len(tests)
        passed_tests = 0

        for test in tests:
            success = run_test(
                test['script'],
                test['args'],
                test['timeout']
            )

            results[test['name']] = success
            if success:
                passed_tests += 1

        # Print summary
        print(f"\n{'='*60}")
        print("TEST SUMMARY")
        print(f"{'='*60}")

        for test_name, passed in results.items():
            status = "[PASSED]" if passed else "[FAILED]"
            print(f"{test_name:<40} {status}")

        print(f"\nTotal: {total_tests}, Passed: {passed_tests}, Failed: {total_tests - passed_tests}")

        if passed_tests == total_tests:
            print("\nAll tests passed!")
            return 0
        else:
            print(f"\n[WARNING] {total_tests - passed_tests} test(s) failed!")
            return 1

    except KeyboardInterrupt:
        print("\nTests interrupted by user")
        return 1
    except Exception as e:
        print(f"\n[ERROR] Test runner error: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
