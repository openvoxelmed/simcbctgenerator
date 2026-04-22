# Testing Suite

This directory contains tests for patient setup, simulation backends, and end-to-end pipeline integration.

## Quick Setup

Before running tests, download the dummy patient dataset:

```bash
cd test/test_scripts
python -c "from testConfigPatient import extract_test_data; extract_test_data()"
```

This downloads `ct.nii.gz`, `mask.nii.gz`, and `cbct.nii.gz` from Google Drive
to `test/test_data/`. The reference CBCT enables the regression pipeline test
to run end-to-end against real data (previously it was a soft smoke-test
because no CBCT asset was shipped). `testPipelineIntegration.py --dummy` also
triggers this download automatically, matching `examples/cli/download_data.py`.

## Test Structure

### Core Component Tests

1. **`testPhantomGenerator.py`** - Tests the PhantomGenerator class
   - Full generation pipeline with test data
   - Visualization capabilities

2. **`testOrganMaskGenerator.py`** - Tests organ segmentation
   - Multi-organ mask generation (bowel, bladder, rectum)
   - Motion surrogate mask generation
   - Functional test approach (user inspection)

3. **`testSimulationBackends.py`** - Tests the simulation backends
   - `PhantomPipeline` / `PhantomCBCTSimulator`
   - `StandardCBCTSimulator`

### Integration Tests

4. **`testPipelineIntegration.py`** - Comprehensive pipeline integration tests
   - Command-line interface testing
   - Automatic test patient file generation
   - End-to-end workflow verification

### Configuration and Utilities

5. **`testConfigPatient.py`** - configuration definitions
   - Test modes include `dummy`, `xvi` and `synrad`
   - default is `dummy`

6. **`runAllTests.py`** - Test runner for the complete test suite
   - Runs all tests with proper error handling
   - Provides comprehensive test reporting

## Running Tests

### Prerequisites

```bash
# Ensure you have the environment set up
uv sync
```

### Quick Testing (Recommended for Development)

```bash
# Run only the fast, core component tests
uv run python runAllTests.py --quick
```

### Full Test Suite

```bash
# Run all tests including integration tests
uv run python runAllTests.py
```

### Individual Tests

```bash
# Test the phantom generator
uv run python testPhantomGenerator.py

# Test the phantom backend
uv run python testSimulationBackends.py --backend phantom

# Test the standard backend
uv run python testSimulationBackends.py --backend standard --motion-model PELVIS

# Generate visualization with real phantom data
uv run python testPhantomGenerator.py --visualize
```

### Specific Test Categories

```bash
# Run all quick integration tests
uv run python runAllTests.py --quick

# Run a specific test by name
uv run python runAllTests.py --test "PhantomGenerator"
```

## Test Data

The tests use real test data and should be located at `test/test_data/`.

- **Real Phantom Data**: Uses actual phantom measurement from `test/test_data/phantom/phantom.mha`
- **Test CT Data**: located at `test/test_data/ct_image.mhd`
- **Test segmentation Data**: located at `test/test_data/mask.mhd`


## Test Modes

The tests support several test modes defined in `testConfigPatient.py`:

- **`dummy`**: Basic test mode with minimal data
- **`XVI`**: Test mode for real XVI data (requires external data)
- **`SYNRAD`**: Test mode for real SYNRAD data (requires external data)

## Expected Outputs

### Successful Test Run

```
Starting Phantom Test Suite
============================================================
Running: testPhantomGenerator.py
============================================================
Duration: 15.23s
[PASSED]

...

TEST SUMMARY
============================================================
PhantomGenerator              [PASSED]
SimulationBackend_phantom     [PASSED]
SimulationBackend_standard    [PASSED]
...

Total: 7, Passed: 7, Failed: 0
All tests passed!
```
