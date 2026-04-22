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

"""Physics configuration for per-patient X-ray parameters."""
import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from pydantic import BaseModel
import numpy as np

# Import PhysicsConfig from config.py to ensure single source of truth
from .config import PhysicsConfig, CBCTSystemConfig
import logging

logger = logging.getLogger(__name__)

def photon_flux(systemconfig: CBCTSystemConfig) -> float:
    """Photon flux in photons/mm²/mAs at the detector.

    Uses spekpy to compute the spectral fluence at the detector plane,
    accounting for inverse-square law, inherent filtration, and air path.

    Returns fluence per mAs per mm² so the calling code can compute
    incident photons per pixel as: bp * mAs * flux * pixel_area_mm².

    Notes:
        - mas=1 is used so the result is per-mAs (calling code multiplies by mAs separately)
        - Filtration material and thickness come from PhysicsConfig (filtration_material /
          filtration_mm).  Elekta uses Al (~13.5 mm); Varian OBI uses Ti (~0.4 mm).
          Using the wrong material causes a ~10× flux error because Ti transmits far
          more of a 120 kVp spectrum per mm than Al does.
        - get_flu() returns photons/cm²; we divide by 100 to convert to photons/mm²
    """
    import spekpy as sp
    sdd_cm = systemconfig.effective_source_detector_distance / 10.0  # mm -> cm
    # Use mas=1 to get fluence per mAs (calling code multiplies by mAs separately)
    s = sp.Spek(kvp=systemconfig.physics.kv, mas=1.0, z=sdd_cm, th=14)
    # Inherent tube filtration (Al), applied before the added flat filter
    if systemconfig.physics.inherent_filtration_al_mm > 0:
        s.filter('Al', systemconfig.physics.inherent_filtration_al_mm)
    # Added flat filter: material and thickness are scanner-specific
    s.filter(systemconfig.physics.filtration_material, systemconfig.physics.filtration_mm)
    # Air path attenuation from source to detector
    s.filter('Air', systemconfig.effective_source_detector_distance)
    # get_flu() returns photons/cm²; convert to photons/mm² (1 cm² = 100 mm²)
    return s.get_flu() / 100.0

def load_physics_config(json_path: Path) -> PhysicsConfig:
    """Load physics configuration from a JSON file.

    Args:
        json_path: Path to the JSON file containing physics parameters

    Returns:
        PhysicsConfig object with loaded parameters

    Raises:
        FileNotFoundError: If the JSON file does not exist
        ValidationError: If required fields are missing or invalid
    """
    json_path = Path(json_path)
    if not json_path.exists():
        raise FileNotFoundError(f"Physics config file not found: {json_path}")

    with open(json_path, 'r') as f:
        data = json.load(f)

    return PhysicsConfig(**data)


def save_physics_config(config: PhysicsConfig, json_path: Path) -> None:
    """Save physics configuration to a JSON file.

    Args:
        config: PhysicsConfig object to save
        json_path: Path where to save the JSON file
    """
    json_path = Path(json_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)

    with open(json_path, 'w') as f:
        json.dump(config.model_dump(), f, indent=4)


# Manufacturer-specific default parameters
MANUFACTURER_DEFAULTS = {
    'elekta': {
        'spr': 1.6,
        'saturation_factor': 2.0,
        'bp_amplitude': 1.07,
        'bp_std': 522.0,
        'bp_floor': 0.0,
        'bp_ds_edge1': 0.0, 'bp_ds_slope1': 0.0, 'bp_ds_Afloor': 0.0,
        'bp_ds_edge2': 0.0, 'bp_ds_slope2': 0.0,
        'source_detector_distance': 1536.0,
        'source_origin_distance': 1000.0,
        'detector_offset': -115.0,
        # XVI S20 full rotation ~120 s / ~660 frames
        'time_per_projection': 0.18,
        # photon_gain calibrated so that spekpy × photon_gain matches the
        # hand-calibrated ini value (4.16e5).
        # spekpy at 120 kV / 1536 mm SDD / 13.5 mm Al → ~4.888e5 ph/mm²/mAs
        # → photon_gain = 4.16e5 / 4.888e5 ≈ 0.851
        'photon_gain': 0.851,
        'inherent_filtration_al_mm': 0.0,
        # Elekta XVI uses an aluminium flat filter
        'filtration_material': 'Al',
        'filtration_mm': 13.5,
    },
    'varian': {
        'spr': 1.3,
        'saturation_factor': 1.0,
        'bp_amplitude': 2.0,
        'bp_std': 126.0,
        'bp_floor': 0.25,
        # Double-sigmoid fit to FilterBowtie.xim (replaces Gaussian + hard floor for Varian).
        # Edges in code-x convention: x = linspace(-pw*ps/2, pw*ps/2, pw) + offset.
        # slope converted from px → mm: slope_mm = slope_px / 0.388
        'bp_ds_edge1': -187.9,   # mm  (pixel ~440 from detector left edge)
        'bp_ds_slope1':  0.0833, # 1/mm (0.0323 px^-1 / 0.388 mm/px)
        'bp_ds_Afloor':  0.158,  # floor/peak ratio (0.1604/1.0161 from fit)
        'bp_ds_edge2': -144.9,   # mm  (pixel ~551 from detector left edge)
        'bp_ds_slope2': -0.0253, # 1/mm (-0.0098 px^-1 / 0.388 mm/px)
        'source_detector_distance': 1500.0,
        'source_origin_distance': 1000.0,
        'detector_offset': -160.0,
        # OBI full rotation ~120 s / ~900 frames
        'time_per_projection': 0.13,
        # Varian OBI uses a Ti flat filter (0.4 mm) — NOT aluminium.
        # spekpy at 120 kV / 1500 mm SDD / 0.4 mm Ti → ~1.570e6 ph/mm²/mAs.
        # The remaining gap to the calibrated ini value (4.16e6) is absorbed
        # by photon_gain, which represents the higher CsI scintillator / DQE
        # efficiency of the Varian flat-panel detector vs the Elekta reference.
        # photon_gain = 4.16e6 / 1.570e6 ≈ 2.65
        'photon_gain': 2.65,
        'inherent_filtration_al_mm': 2.5,
        'filtration_material': 'Ti',
        'filtration_mm': 0.4,
    }
}


def load_cbct_physics_from_yaml(yaml_path: Path) -> PhysicsConfig:
    """Load CBCT physics configuration from A001-style metadata.yaml.

    Args:
        yaml_path: Path to metadata.yaml file

    Returns:
        PhysicsConfig with CBCT parameters mapped from YAML

    Uses manufacturer-specific defaults based on the 'Manufacturer' field:
        - Elekta: spr=1.6, bp_amplitude=1.07, bp_std=522.0
        - Varian: spr=1.3, bp_amplitude=1.06, bp_std=939.0

    The mapping from metadata.yaml CBCT section:
        - TubeVoltage -> kv
        - TubeCurrent * PulseLength / 1000 -> mAs (40mA * 40ms = 1.6 mAs)
    """
    yaml_path = Path(yaml_path)
    if not yaml_path.exists():
        raise FileNotFoundError(f"Metadata YAML file not found: {yaml_path}")

    with open(yaml_path, 'r') as f:
        metadata = yaml.safe_load(f)

    cbct = metadata.get('cbct', {})

    # Determine manufacturer and get defaults
    manufacturer = cbct.get('Manufacturer', 'Elekta').lower()
    if 'varian' in manufacturer:
        defaults = MANUFACTURER_DEFAULTS['varian']
    else:
        # Default to Elekta for Elekta, Siemens, or unknown
        defaults = MANUFACTURER_DEFAULTS['elekta']

    # Extract and map parameters
    kv = float(cbct.get('TubeVoltage', 120))
    tube_current = float(cbct.get('TubeCurrent', 40))  # mA
    pulse_length = float(cbct.get('PulseLength', 40))  # ms
    mAs = tube_current * pulse_length / 1000.0  # Convert ms to seconds

    return PhysicsConfig(
        photon_flux=None,  # Will be computed by spekpy at runtime
        spr=defaults['spr'],
        mAs=mAs,
        kv=kv,
        saturation_factor=defaults['saturation_factor'],
        bp_amplitude=defaults['bp_amplitude'],
        bp_std=defaults['bp_std'],
        bp_floor=defaults['bp_floor'],
        bp_ds_edge1=defaults.get('bp_ds_edge1', 0.0),
        bp_ds_slope1=defaults.get('bp_ds_slope1', 0.0),
        bp_ds_Afloor=defaults.get('bp_ds_Afloor', 0.0),
        bp_ds_edge2=defaults.get('bp_ds_edge2', 0.0),
        bp_ds_slope2=defaults.get('bp_ds_slope2', 0.0),
        photon_gain=defaults['photon_gain'],
        inherent_filtration_al_mm=defaults.get('inherent_filtration_al_mm', 0.0),
        filtration_material=defaults['filtration_material'],
        filtration_mm=defaults['filtration_mm'],
        threads=8,
        max_block_index=200
    )


def load_geometry_from_yaml(yaml_path: Path) -> Dict[str, Any]:
    """Extract CBCT geometry parameters from metadata.yaml.

    Args:
        yaml_path: Path to metadata.yaml file

    Returns:
        Dict with geometry parameters:
            - detector_pixels_h, detector_pixels_w
            - detector_size_h, detector_size_w (computed from pixels * spacing)
            - detector_offset_x, detector_offset_y
            - source_origin_distance, source_detector_distance
            - recon_size, recon_origin, recon_spacing
            - num_frames

    Uses manufacturer-specific defaults for source distances and detector offset.
    """
    yaml_path = Path(yaml_path)
    if not yaml_path.exists():
        raise FileNotFoundError(f"Metadata YAML file not found: {yaml_path}")

    with open(yaml_path, 'r') as f:
        metadata = yaml.safe_load(f)

    cbct = metadata.get('cbct', {})

    # Determine manufacturer and get defaults
    manufacturer = cbct.get('Manufacturer', 'Elekta').lower()
    if 'varian' in manufacturer:
        defaults = MANUFACTURER_DEFAULTS['varian']
    else:
        defaults = MANUFACTURER_DEFAULTS['elekta']

    detector_pixels_h = int(cbct.get('ImagerSizeY', 504))
    detector_pixels_w = int(cbct.get('ImagerSizeX', 504))
    pixel_spacing_h = float(cbct.get('ImagerResY', 0.8))
    pixel_spacing_w = float(cbct.get('ImagerResX', 0.8))

    detector_size_h = detector_pixels_h * pixel_spacing_h
    detector_size_w = detector_pixels_w * pixel_spacing_w

    recon_size_x = int(cbct.get('ReconstructionSizeX', 410))
    recon_size_y = int(cbct.get('ReconstructionSizeY', 410))
    recon_size_z = int(cbct.get('ReconstructionSizeZ', 264))

    recon_spacing_x = float(cbct.get('ReconstructionSpacingX', 1.0))
    recon_spacing_y = float(cbct.get('ReconstructionSpacingY', 1.0))
    recon_spacing_z = float(cbct.get('ReconstructionSpacingZ', 1.0))

    # ITK/SimpleITK convention: [x, y, z]
    recon_size = [recon_size_x, recon_size_z, recon_size_y]
    recon_spacing = [recon_spacing_x, recon_spacing_z, recon_spacing_y]

    # Compute origin (centered at isocenter)
    recon_origin = [
        -recon_size[0] * recon_spacing[0] / 2.0,
        -recon_size[1] * recon_spacing[1] / 2.0,
        -recon_size[2] * recon_spacing[2] / 2.0
    ]

    # Use detector offset from YAML if available, otherwise use manufacturer default
    detector_offset_x = float(cbct.get('DetectorOffsetX', defaults['detector_offset']))
    detector_offset_y = float(cbct.get('DetectorOffsetY', 0.0))

    return {
        'detector_pixels_h': detector_pixels_h,
        'detector_pixels_w': detector_pixels_w,
        'detector_size_h': detector_size_h,
        'detector_size_w': detector_size_w,
        'detector_offset_x': detector_offset_x,
        'detector_offset_y': detector_offset_y,
        'source_origin_distance': defaults['source_origin_distance'],
        'source_detector_distance': defaults['source_detector_distance'],
        'recon_size': recon_size,
        'recon_origin': recon_origin,
        'recon_spacing': recon_spacing,
        'num_frames': int(cbct.get('Frames', 656))
    }

# ============================================================================
# Polychromatic spectral data (xraydb + spekpy)
# ============================================================================

# ICRU-44 Cortical Bone elemental composition (weight fractions, sum = 1.0)
CORTICAL_BONE_COMPOSITION: dict[str, float] = {
    "H":  0.034, "C":  0.155, "N":  0.042, "O":  0.435,
    "Na": 0.001, "Mg": 0.002, "P":  0.103, "S":  0.003, "Ca": 0.225,
}
CORTICAL_BONE_DENSITY = 1.92  # g/cm^3



class SpectralData(BaseModel):
    """All arrays the polychromatic kernel needs.

    ``spectrum`` is pre-normalised (sum = 1).
    ``ratio_*`` arrays are ``mu(E) / mu(E0)`` so both equal 1.0 at E0.
    """
    spectrum: list[float]        # [num_bins]  normalised weights
    ratio_water: list[float]     # [num_bins]  mu_water(E) / mu_water(E0)
    ratio_bone: list[float]      # [num_bins]  mu_bone(E)  / mu_bone(E0)
    mu_ref_water: float         # water  mu at E0  [cm^-1]
    mu_ref_bone: float          # bone   mu at E0  [cm^-1]
    energies_kev: list[float]    # [num_bins]  bin centres
    ref_energy_kev: float       # E0

    @property
    def num_bins(self) -> int:
        return len(self.spectrum)


def _water_mu(energies_ev: np.ndarray) -> np.ndarray:
    """Linear attenuation of water [cm^-1] via xraydb."""
    import xraydb
    return np.array([
        xraydb.material_mu("H2O", float(e), density=1.0)
        for e in energies_ev
    ], dtype=np.float64)


def _bone_mu(energies_ev: np.ndarray) -> np.ndarray:
    """Linear attenuation of ICRU-44 cortical bone [cm^-1] via xraydb."""
    import xraydb
    mu = np.zeros(len(energies_ev), dtype=np.float64)
    for i, e in enumerate(energies_ev):
        mu_rho = sum(
            w * xraydb.mu_elam(elem, float(e))
            for elem, w in CORTICAL_BONE_COMPOSITION.items()
        )
        mu[i] = mu_rho * CORTICAL_BONE_DENSITY
    return mu


def generate_spectral_data(
    kvp: float = 120.0,
    mAs: float = 1.6,
    sdd_mm: float = 1536.0,
    inherent_filtration_al_mm: float = 0.0,
    filtration_material: str = 'Al',
    filtration_mm: float = 13.5,
    target_angle: float = 14.0,
    dk: float = 1.0,
    ref_energy_kev: Optional[float] = None,
) -> SpectralData:
    """Generate spectral tables for the polychromatic kernel.

    Uses spekpy for the spectrum (matching the existing pipeline) and
    xraydb (NIST XCOM) for energy-dependent water / bone attenuation.

    Parameters
    ----------
    kvp : float
        Tube voltage [kV].
    sdd_mm : float
        Source-to-detector distance [mm].
    inherent_filtration_al_mm : float
        Inherent tube filtration [mm Al], applied before the flat filter.
        Elekta: 0 (lumped into filtration_mm); Varian OBI: ~2.5 mm.
    filtration_material : str
        Added flat-filter material recognised by spekpy (e.g. 'Al', 'Ti').
        Elekta XVI uses 'Al'; Varian OBI uses 'Ti'.
    filtration_mm : float
        Added flat-filter thickness [mm].
    target_angle : float
        Anode target angle [deg] (use 14 to match existing pipeline).
    dk : float
        Energy bin width [keV].
    ref_energy_kev : float or None
        Reference energy E0 [keV].  *None* → spectrum-weighted mean.

    Returns
    -------
    SpectralData
    """
    import spekpy as sp

    # ---- spectrum (same params as photon_flux) ----
    sdd_cm = sdd_mm / 10.0
    s = sp.Spek(kvp=kvp, mas=mAs, z=sdd_cm, th=target_angle, dk=dk)
    if inherent_filtration_al_mm > 0:
        s.filter("Al", inherent_filtration_al_mm)  # inherent tube filtration
    s.filter(filtration_material, filtration_mm)   # added flat filter
    s.filter("Air", sdd_mm)                        # air path in mm

    energies_kev = s.get_k()
    spectrum_raw = s.get_spk()
    num_bins = len(energies_kev)

    spectrum_sum = spectrum_raw.sum()
    if spectrum_sum <= 0:
        raise ValueError("Spectrum has zero total fluence")
    spectrum = (spectrum_raw / spectrum_sum).astype(np.float32)

    logger.info("Spectrum: %d bins, %.1f-%.1f keV", num_bins,
                energies_kev[0], energies_kev[-1])

    # ---- reference energy ----
    if ref_energy_kev is None:
        ref_energy_kev = float(np.sum(energies_kev * spectrum_raw) / spectrum_sum)
    logger.info("Reference energy E0 = %.1f keV", ref_energy_kev)
    logger.info("Reference energy (SpekPy) E0 = %.1f keV", s.get_eeff())

    # ---- attenuation curves (xraydb) ----
    energies_ev = (energies_kev * 1000.0).astype(np.float64)
    ref_ev = ref_energy_kev * 1000.0

    mu_water_curve = _water_mu(energies_ev)
    mu_bone_curve  = _bone_mu(energies_ev)
    mu_ref_water   = float(_water_mu(np.array([ref_ev]))[0])
    mu_ref_bone    = float(_bone_mu(np.array([ref_ev]))[0])

    logger.info("mu_ref_water(E0) = %.4f cm^-1", mu_ref_water)
    logger.info("mu_ref_bone (E0) = %.4f cm^-1", mu_ref_bone)

    # ---- ratio arrays ----
    ratio_water = (mu_water_curve / mu_ref_water).astype(np.float32)
    ratio_bone  = (mu_bone_curve  / mu_ref_bone).astype(np.float32)

    return SpectralData(
        spectrum=spectrum.tolist(),
        ratio_water=ratio_water.tolist(),
        ratio_bone=ratio_bone.tolist(),
        mu_ref_water=mu_ref_water,
        mu_ref_bone=mu_ref_bone,
        energies_kev=energies_kev.astype(np.float32).tolist(),
        ref_energy_kev=ref_energy_kev,
    )
