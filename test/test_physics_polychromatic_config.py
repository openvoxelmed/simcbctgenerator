from pathlib import Path
from unittest.mock import patch

import numpy as np
import SimpleITK as sitk

from simcbctgenerator.generate_projections import DRRGenerator
from simcbctgenerator.simulation.standard import StandardCBCTSimulator
from simcbctgenerator.utils.config import CBCTSystemConfig, Vendor


TEST_DATA_DIR = Path(__file__).parent / "test_data"


def test_varian_vendor_defaults_include_inherent_filtration():
    system_config = CBCTSystemConfig.for_vendor(Vendor.VARIAN)

    assert system_config.physics.inherent_filtration_al_mm == 2.5
    assert system_config.physics.filtration_material == "Ti"
    assert system_config.physics.filtration_mm == 0.4


def test_polychromatic_metadata_config_preserves_thresholds_and_filtration():
    simulator = StandardCBCTSimulator(vendor="varian", polychromatic=True, gpu=False)

    system_config = simulator.build_system_config(
        metadata_yaml=TEST_DATA_DIR / "metadata_varian.yaml",
    )

    assert system_config.physics.polychromatic is True
    assert system_config.physics.T1 == -200.0
    assert system_config.physics.T2 == 400.0
    assert system_config.physics.inherent_filtration_al_mm == 2.5
    assert system_config.physics.filtration_material == "Ti"
    assert system_config.physics.filtration_mm == 0.4


def test_drr_generator_passes_inherent_filtration_to_spectral_setup(tmp_path):
    system_config = CBCTSystemConfig.for_vendor(Vendor.VARIAN).with_physics(
        polychromatic=True,
        photon_flux=1.0,
        mAs=1.2,
    )

    with patch("simcbctgenerator.generate_projections.generate_spectral_data") as mock_generate:
        mock_generate.return_value = object()
        DRRGenerator(output_dir=tmp_path, system_config=system_config)

    _, kwargs = mock_generate.call_args
    assert kwargs["inherent_filtration_al_mm"] == 2.5
    assert kwargs["filtration_material"] == "Ti"
    assert kwargs["filtration_mm"] == 0.4


def test_save_stacked_projections_preserves_open_beam_dynamic_range(tmp_path):
    simulator = StandardCBCTSimulator(gpu=False)
    system_config = CBCTSystemConfig.for_vendor(Vendor.ELEKTA)

    # Open beam (log attenuation = 0) should export to the exact uint16 max,
    # not clip from an oversized exponential approximation.
    projections = sitk.GetImageFromArray(
        np.array([[[0.0, 0.5], [1.0, 2.0]]], dtype=np.float32)
    )

    simulator._save_stacked_projections(projections, tmp_path, system_config)

    exported = sitk.GetArrayFromImage(sitk.ReadImage(str(tmp_path / "projections_simulated.mha")))
    counts = exported.astype(np.float32)
    recovered = -np.log(np.clip(counts, 1.0, None) / np.iinfo(np.uint16).max)

    assert exported.dtype == np.uint16
    assert int(exported[0, 0, 0]) == np.iinfo(np.uint16).max
    assert np.allclose(recovered, sitk.GetArrayFromImage(projections), atol=2e-4)
