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

from configparser import ConfigParser
from enum import Enum, auto
from typing import Any, Tuple, List, Optional, Union

import numpy as np
from pydantic import BaseModel, PrivateAttr, field_validator

from pathlib import Path
from simcbctgenerator.utils.rtk_geometry import RTKGeometryData, parse_rtk_geometry_xml, create_geometry_data


class Errors(Enum):
    NOT_ALL_STRUCTURES = auto()
    NO_PLAN = auto()
    MULTIPLE_PLANS = auto()


class Vendor(str, Enum):
    ELEKTA = "elekta"
    VARIAN = "varian"

    @classmethod
    def from_value(cls, value: "Vendor | str") -> "Vendor":
        if isinstance(value, cls):
            return value
        normalized = value.lower()
        return cls(normalized)

class ImageType(Enum):
    DICOM = auto()
    NIFTI = auto()

class ImageCenter(Enum):
    PLANISOCENTER = auto()
    IMAGECENTER = auto()
    PTVCENTER = auto()

class ImagingModalityBase(BaseModel):
    image_type: ImageType
    image_center: ImageCenter
    image: str|None = None
    segmentation: str|None = None
    cbct: str|None = None
    ct_dir: Path
    seg_dir: Path


class XVI(ImagingModalityBase):
    plan_dir: Path
    cbct_dir: Path

    def get_id(self, pat_path:Path) -> str:
        return pat_path.name.split('_')[-1]

class SYNRAD(ImagingModalityBase):
    pass

    def get_id(self, pat_path:Path) -> str:
        return Path(pat_path).name

class DUMMY(ImagingModalityBase):
    pass
    def get_id(self, pat_path:Path) -> str:
        return 'DUMMY'

class ImagingModality(Enum):
    xvi = XVI(image_type = ImageType.DICOM, image_center = ImageCenter.PLANISOCENTER, ct_dir='CT_PLAN', seg_dir='CT_SET', plan_dir='DICOM_PLAN', cbct_dir='IMAGES')
    synrad = SYNRAD(image_type= ImageType.NIFTI, image_center = ImageCenter.IMAGECENTER, ct_dir='IMAGES', seg_dir='IMAGES', image='ct.nii.gz', segmentation='masks.nii.gz')
    synrad2025 = SYNRAD(image_type= ImageType.NIFTI, image_center = ImageCenter.IMAGECENTER, ct_dir='.', seg_dir='.', image='ct.mha', cbct='cbct.mha')
    dummy = DUMMY(image_type= ImageType.NIFTI, image_center = ImageCenter.IMAGECENTER, ct_dir='test/test_data', seg_dir='test/test_data', image='ct.nii.gz', segmentation='mask.nii.gz', cbct='cbct.nii.gz')
    dummy2 = DUMMY(image_type=ImageType.NIFTI, image_center=ImageCenter.IMAGECENTER, ct_dir='test/test_data/thorax', seg_dir='test/test_data/thorax', image='ct.mha', segmentation='mask.mha')


class PhysicsConfig(BaseModel):
    """X-ray physics and execution parameters for projection generation.

    Replaces DRRConfig. Includes both physics parameters (photon flux, scatter, exposure)
    and execution parameters (CUDA threads, block limits).
    """
    # X-ray physics parameters
    photon_flux: Optional[float] = None  # Photon flux in photons/mm²/mAs. If None, computed via spekpy.
    spr: float  # Scatter-to-Primary Ratio
    mAs: float  # milliampere-seconds (exposure)
    kv: float  # Kilovoltage peak
    saturation_factor: float = 1.0  # Detector saturation correction
    bp_amplitude:float # Beam profile amplitude for response estimation
    bp_std: float # Beam profile standard deviation for response estimation
    bp_floor: float = 0.0  # Minimum beam profile value (floor). Non-zero for Varian half-bowtie edge.

    # Double-sigmoid beam profile parameters (Varian only).
    # When bp_ds_slope1 != 0 the double sigmoid replaces the Gaussian + hard floor.
    # All 0.0 → disabled (Gaussian model used instead).
    bp_ds_edge1: float = 0.0   # left sigmoid centre [mm, code x = linspace + offset convention]
    bp_ds_slope1: float = 0.0  # left sigmoid slope [1/mm]; non-zero enables double sigmoid
    bp_ds_Afloor: float = 0.0  # floor/peak ratio for the double sigmoid [0-1]
    bp_ds_edge2: float = 0.0   # right sigmoid centre [mm, same convention as bp_ds_edge1]
    bp_ds_slope2: float = 0.0  # right sigmoid slope [1/mm]

    # Beam filtration used for spekpy flux computation and polychromatic kernel.
    # Two-layer model: inherent tube filtration (always Al) + added flat filter.
    #
    # Elekta XVI:  inherent_filtration_al_mm=0  (lumped into filtration_mm=13.5 Al)
    # Varian OBI:  inherent_filtration_al_mm=2.5 (tube) + filtration_mm=0.4 Ti (flat)
    #
    # Separating the two matters for the polychromatic kernel: using only 0.4 mm Ti
    # without inherent filtration gives an unrealistically soft beam
    # (HVL ≈ 4.8 mm Al, Eff. E ≈ 41 keV) whereas the full model yields
    # HVL ≈ 6.25 mm Al, Eff. E ≈ 47 keV — matching real Varian OBI measurements.
    inherent_filtration_al_mm: float = 0.0   # inherent tube filtration [mm Al], applied first
    filtration_material: str = 'Al'           # added flat-filter material (spekpy name)
    filtration_mm: float = 13.5              # added flat-filter thickness [mm]
    filter_amplitude: float = 1.45  # Beam hardening filter amplitude
    filter_std: float = 164  # Beam hardening filter standard deviation

    # Polychromatic projection parameters
    polychromatic: bool = False  # Enable spectral beam-hardening simulation
    T1: float|None = 200.0   # Lower HU threshold: water -> bone transition start
    T2: float|None = 1500.0  # Upper HU threshold: water -> bone transition end

    # Detector gain: compensates for per-vendor differences in pixel area,
    # DQE, and scintillator efficiency so that simulated noise levels match
    # the real scanner.  Elekta (0.8 mm pixels) is the reference (gain=1.0).
    # Varian OBI has ~0.388 mm pixels → (0.8/0.388)^2 ≈ 4.25 to equalise.
    photon_gain: float = 1.0

    # Execution parameters
    threads: int = 8  # CUDA threads
    max_block_index: int = 200  # CUDA block limit

    #f1: elekta  amplitude = 1.07, std=522
    #varian: amplitude = 1.06, std=939


class GeometryConfig(BaseModel):
    """Complete CBCT geometry configuration.

    Contains all parameters that can be defined in RTK XML geometry files:
    - C-arm geometry (source distances, detector offset)
    - Detector specifications (size, pixels)
    - Acquisition angles
    Supports XML override for per-angle geometry variations.
    """
    # C-arm geometry
    source_origin_distance: float  # Source to isocenter distance (mm)
    source_detector_distance: float  # Source to detector distance (mm)
    detector_offset: float  # Detector lateral offset (mm)

    # Detector specifications
    detector_size_h: float  # Detector height in mm
    detector_size_w: float  # Detector width in mm
    detector_pixels_h: int  # Number of pixels vertically
    detector_pixels_w: int  # Number of pixels horizontally

    # Acquisition angles
    start_angle: float = 0.0  # Start gantry angle (degrees, 0-360 convention)
    end_angle: float = 360.0  # End gantry angle (degrees, 0-360 convention)
    angle_increments: float = 1.0  # Angle increment between projections

    # Optional RTK XML geometry file (overrides all geometry if provided)
    geometry_xml_path: Optional[str] = None

    # Private cached properties
    _pixel_size: tuple[float, float] = PrivateAttr(default=None)
    _geometry_data: Optional[RTKGeometryData] = PrivateAttr(default=None)

    class Config:
        arbitrary_types_allowed = True

    @property
    def pixel_size(self) -> tuple[float, float]:
        """Compute pixel size from detector dimensions."""
        if self._pixel_size is None:
            self._pixel_size = (
                self.detector_size_h / self.detector_pixels_h,
                self.detector_size_w / self.detector_pixels_w
            )
        return self._pixel_size

    @property
    def angles(self) -> np.ndarray:
        """Compute angle array from start/end/increments."""
        return np.arange(self.start_angle, self.end_angle, self.angle_increments)

    @property
    def num_projections(self) -> int:
        """Number of projections in the scan."""
        return len(self.angles)

    @property
    def has_xml_geometry(self) -> bool:
        """Check if RTK XML geometry is configured."""
        return self.geometry_xml_path is not None

    def get_geometry_data(self) -> RTKGeometryData:
        """Get or create geometry data with per-angle offsets.

        When geometry_xml_path is set, loads all geometry from XML.
        Otherwise, creates geometry data from config values with fixed offset.
        """
        if self._geometry_data is None:
            if self.geometry_xml_path is not None:
                self._geometry_data = parse_rtk_geometry_xml(self.geometry_xml_path)
            else:
                self._geometry_data = create_geometry_data(
                    angles=self.angles,
                    detector_offset=self.detector_offset,
                    source_to_isocenter=self.source_origin_distance,
                    source_to_detector=self.source_detector_distance
                )
        return self._geometry_data

    @property
    def geometry_data(self) -> RTKGeometryData:
        """Cached geometry data property."""
        return self.get_geometry_data()

    def get_effective_source_origin_distance(self) -> float:
        """Get source-to-isocenter distance (from XML if available, else config)."""
        if self.has_xml_geometry:
            return self.geometry_data.source_to_isocenter
        return self.source_origin_distance

    def get_effective_source_detector_distance(self) -> float:
        """Get source-to-detector distance (from XML if available, else config)."""
        if self.has_xml_geometry:
            return self.geometry_data.source_to_detector
        return self.source_detector_distance

    @property
    def effective_angles(self) -> np.ndarray:
        """Get projection angles (from XML if available, else config)."""
        if self.has_xml_geometry:
            return self.geometry_data.angles
        return self.angles

    def get_offset_at_index(self, idx: int) -> tuple[float, float]:
        """Get (offset_x, offset_y) for projection index."""
        return self.geometry_data.get_offsets_at_index(idx)

    def get_offset_at_angle(self, angle: float) -> tuple[float, float]:
        """Get (offset_x, offset_y) for given angle.

        Uses angle-based lookup for XML geometry, falls back to index 0 for config.
        """
        if self.has_xml_geometry:
            return self.geometry_data.get_offsets_at_angle(angle % 360)
        # For config-based geometry, offset is constant
        return self.geometry_data.get_offsets_at_index(0)


class ReconstructionVolumeConfig(BaseModel):
    """Reconstruction volume specifications."""
    recon_size: List[int]  # Voxel dimensions [x, y, z]
    recon_origin: List[float]  # Origin in mm [x, y, z]
    recon_spacing: List[float]  # Voxel spacing in mm [x, y, z]


class CBCTSystemConfig(BaseModel):
    """Complete CBCT system configuration.

    Replaces ReconConfig. Composes physics, geometry, and reconstruction volume configs
    into a single system configuration.
    """
    physics: PhysicsConfig
    geometry: GeometryConfig
    reconstruction_volume: ReconstructionVolumeConfig

    # Private cache for geometry_data accessed through geometry
    _geometry_data: Optional[RTKGeometryData] = PrivateAttr(default=None)

    class Config:
        arbitrary_types_allowed = True

    @classmethod
    def for_vendor(cls, vendor: Vendor | str) -> "CBCTSystemConfig":
        """Create a default system configuration for a supported vendor."""
        from simcbctgenerator.utils.physics_config import MANUFACTURER_DEFAULTS

        vendor = Vendor.from_value(vendor)
        defaults = MANUFACTURER_DEFAULTS[vendor.value]

        if vendor is Vendor.ELEKTA:
            detector_size_h = 410.0
            detector_size_w = 410.0
            detector_pixels_h = 1024
            detector_pixels_w = 1024
        else:
            detector_size_h = 397.312
            detector_size_w = 397.312
            detector_pixels_h = 1024
            detector_pixels_w = 1024

        return cls(
            physics=PhysicsConfig(
                photon_flux=None,
                spr=defaults["spr"],
                mAs=1.6,
                kv=120.0,
                saturation_factor=defaults["saturation_factor"],
                bp_amplitude=defaults["bp_amplitude"],
                bp_std=defaults["bp_std"],
                bp_floor=defaults["bp_floor"],
                bp_ds_edge1=defaults.get("bp_ds_edge1", 0.0),
                bp_ds_slope1=defaults.get("bp_ds_slope1", 0.0),
                bp_ds_Afloor=defaults.get("bp_ds_Afloor", 0.0),
                bp_ds_edge2=defaults.get("bp_ds_edge2", 0.0),
                bp_ds_slope2=defaults.get("bp_ds_slope2", 0.0),
                photon_gain=defaults["photon_gain"],
                inherent_filtration_al_mm=defaults.get("inherent_filtration_al_mm", 0.0),
                filtration_material=defaults["filtration_material"],
                filtration_mm=defaults["filtration_mm"],
            ),
            geometry=GeometryConfig(
                source_origin_distance=defaults["source_origin_distance"],
                source_detector_distance=defaults["source_detector_distance"],
                detector_offset=defaults["detector_offset"],
                detector_size_h=detector_size_h,
                detector_size_w=detector_size_w,
                detector_pixels_h=detector_pixels_h,
                detector_pixels_w=detector_pixels_w,
                start_angle=0.0,
                end_angle=360.0,
                angle_increments=1.0,
            ),
            reconstruction_volume=ReconstructionVolumeConfig(
                recon_size=[512, 512, 512],
                recon_origin=[-255.0, -255.0, -255.0],
                recon_spacing=[1.0, 1.0, 1.0],
            ),
        )

    @classmethod
    def elekta_defaults(cls) -> "CBCTSystemConfig":
        """Create CBCTSystemConfig with Elekta XVI default parameters."""
        return cls.for_vendor(Vendor.ELEKTA)

    @classmethod
    def varian_defaults(cls) -> "CBCTSystemConfig":
        """Create CBCTSystemConfig with Varian TrueBeam default parameters."""
        return cls.for_vendor(Vendor.VARIAN)

    @classmethod
    def from_ini(cls, ini_path: Union[str, Path]) -> "CBCTSystemConfig":
        """Load a system configuration from an INI file."""
        ini_path = Path(ini_path)
        if not ini_path.exists():
            raise FileNotFoundError(f"INI config file not found: {ini_path}")

        parser = ConfigParser()
        parser.read(ini_path)

        def parse_value(value: str) -> Any:
            value = value.strip()
            lowered = value.lower()
            if lowered in {"true", "false", "yes", "no"}:
                return lowered in {"true", "yes"}
            if "," in value:
                return [parse_value(item) for item in value.split(",") if item != ""]
            try:
                return int(value)
            except ValueError:
                pass
            try:
                return float(value)
            except ValueError:
                return value

        def section_dict(name: str) -> dict[str, Any]:
            if not parser.has_section(name):
                raise ValueError(f"Missing [{name}] section in {ini_path}")
            return {key: parse_value(value) for key, value in parser.items(name)}

        physics = PhysicsConfig(**section_dict("PhysicsConfig"))
        geometry = GeometryConfig(**section_dict("GeometryConfig"))
        reconstruction_volume = ReconstructionVolumeConfig(**section_dict("ReconstructionVolumeConfig"))
        return cls(
            physics=physics,
            geometry=geometry,
            reconstruction_volume=reconstruction_volume,
        )

    @classmethod
    def from_preset(cls, preset_name: str) -> "CBCTSystemConfig":
        """Load a system configuration from a packaged INI preset."""
        from simcbctgenerator.utils.arg_parser import resolve_config_path

        return cls.from_ini(resolve_config_path(preset_name))

    @staticmethod
    def list_presets() -> list[str]:
        """List available packaged INI preset names."""
        from simcbctgenerator.utils.arg_parser import discover_config_presets

        return sorted(discover_config_presets())

    @staticmethod
    def preset_path(preset_name: str) -> Path:
        """Resolve a preset name to its packaged INI file path."""
        from simcbctgenerator.utils.arg_parser import resolve_config_path

        return resolve_config_path(preset_name)

    # Configuration update methods
    def with_physics(self, **updates) -> "CBCTSystemConfig":
        """Return a new config with updated physics parameters.

        Args:
            **updates: Physics parameters to update (photon_flux, spr, mAs, kv, etc.)

        Example:
            new_config = config.with_physics(mAs=50.0, kv=125.0)
        """
        updated_physics = self.physics.model_copy(update=updates)
        return self.model_copy(update={'physics': updated_physics})

    def with_geometry(self, **updates) -> "CBCTSystemConfig":
        """Return a new config with updated geometry parameters.

        Args:
            **updates: Geometry parameters to update (source_origin_distance, detector_size_h, etc.)

        Example:
            new_config = config.with_geometry(start_angle=10.0, end_angle=370.0)
        """
        updated_geometry = self.geometry.model_copy(update=updates)
        return self.model_copy(update={'geometry': updated_geometry})

    def with_reconstruction_volume(self, **updates) -> "CBCTSystemConfig":
        """Return a new config with updated reconstruction volume parameters.

        Args:
            **updates: Reconstruction volume parameters to update (recon_size, recon_origin, etc.)

        Example:
            new_config = config.with_reconstruction_volume(recon_size=[256, 256, 256])
        """
        updated_recon = self.reconstruction_volume.model_copy(update=updates)
        return self.model_copy(update={'reconstruction_volume': updated_recon})

    # Backward compatibility properties - delegate to sub-configs
    @property
    def pixel_size(self) -> tuple[float, float]:
        """Detector pixel size."""
        return self.geometry.pixel_size

    @property
    def angles(self) -> np.ndarray:
        """Projection angles array."""
        return self.geometry.angles

    @property
    def has_xml_geometry(self) -> bool:
        """Check if RTK XML geometry is configured."""
        return self.geometry.has_xml_geometry

    @property
    def geometry_data(self) -> RTKGeometryData:
        """Get or create geometry data with per-angle offsets."""
        if self._geometry_data is None:
            self._geometry_data = self.geometry.get_geometry_data()
        return self._geometry_data

    @property
    def effective_source_origin_distance(self) -> float:
        """Get source-to-isocenter distance (from XML if available, else config)."""
        return self.geometry.get_effective_source_origin_distance()

    @property
    def effective_source_detector_distance(self) -> float:
        """Get source-to-detector distance (from XML if available, else config)."""
        return self.geometry.get_effective_source_detector_distance()

    @property
    def effective_angles(self) -> np.ndarray:
        """Get projection angles (from XML if available, else config)."""
        return self.geometry.effective_angles

    def get_offset_at_index(self, idx: int) -> tuple[float, float]:
        """Get (offset_x, offset_y) for projection index."""
        return self.geometry_data.get_offsets_at_index(idx)

    def get_offset_at_angle(self, angle: float) -> tuple[float, float]:
        """Get (offset_x, offset_y) for given angle."""
        return self.geometry.get_offset_at_angle(angle)

    # Direct property access for backward compatibility
    @property
    def detector_offset(self) -> float:
        return self.geometry.detector_offset

    @property
    def source_origin_distance(self) -> float:
        return self.geometry.source_origin_distance

    @property
    def source_detector_distance(self) -> float:
        return self.geometry.source_detector_distance

    @property
    def detector_size_h(self) -> float:
        return self.geometry.detector_size_h

    @property
    def detector_size_w(self) -> float:
        return self.geometry.detector_size_w

    @property
    def detector_pixels_h(self) -> int:
        return self.geometry.detector_pixels_h

    @property
    def detector_pixels_w(self) -> int:
        return self.geometry.detector_pixels_w

    @property
    def recon_size(self) -> List[int]:
        return self.reconstruction_volume.recon_size

    @property
    def recon_origin(self) -> List[float]:
        return self.reconstruction_volume.recon_origin

    @property
    def recon_spacing(self) -> List[float]:
        return self.reconstruction_volume.recon_spacing

    @property
    def geometry_xml_path(self) -> Optional[str]:
        return self.geometry.geometry_xml_path

    @property
    def bp_amplitude(self) -> float:
        return self.physics.bp_amplitude

    @property
    def bp_std(self) -> float:
        return self.physics.bp_std

    @property
    def bp_floor(self) -> float:
        return self.physics.bp_floor

    @property
    def bp_ds_edge1(self) -> float:
        return self.physics.bp_ds_edge1

    @property
    def bp_ds_slope1(self) -> float:
        return self.physics.bp_ds_slope1

    @property
    def bp_ds_Afloor(self) -> float:
        return self.physics.bp_ds_Afloor

    @property
    def bp_ds_edge2(self) -> float:
        return self.physics.bp_ds_edge2

    @property
    def bp_ds_slope2(self) -> float:
        return self.physics.bp_ds_slope2

    @property
    def filter_amplitude(self) -> float:
        return self.physics.filter_amplitude

    @property
    def filter_std(self) -> float:
        return self.physics.filter_std


#TODO: refactor to make ELEKTA format not mandatory
class PatientConfig(BaseModel):
    plan_dir: str #= 'DICOM_PLAN'
    ct_dir: str #= 'CT_SET'
    cbct_dir: str #= 'IMAGES'
    export_structures: List[str] #= {'bowel', 'bladder', 'rectum'}
    priority: List[int]
    cm_mask: str|None = None
    use_totalsegmentator: bool = False  # Auto-generate export_structures using TotalSegmentator
    image_modality: ImagingModality

    @field_validator('image_modality', mode='before')
    @classmethod
    def validate_motion_type(cls, v):
        """Accept string values like 'xvi', 'synrad' or 'dummy' and convert to enum."""
        if isinstance(v, str):
            try:
                return ImagingModality[v.lower()]
            except KeyError:
                raise ValueError(f"Invalid motion type: {v}. Must be {', '.join([m.name for m in ImagingModality])}.")
        return v

def sample_motion_config(
    motion_type: "MotionConfig.MotionType",
    amplitude_range: Tuple[float, float],
    frequency_range: Tuple[float, float],
    time_per_projection: float,
    uncertainty_range: Tuple[float, float] = (0.01, 0.05),
    phase_offset_breathing: float = 0.0,
    amplitude_heart: Optional[float] = None,
    frequency_heartbeat: int = 80,
    phase_offset_heart: float = 0.0,
    rng: Optional[np.random.Generator] = None
) -> "MotionConfig":
    """Sample random motion parameters for a patient.

    Motion surrogate organs are automatically inferred from motion_type:
    - PELVIS → bowel
    - THORAX → heart, aorta, lung, spine
    - ABDOMEN → bowel + heart, aorta, lung, spine (blended thorax/pelvis)

    Args:
        motion_type: Type of motion model (PELVIS or THORAX)
        amplitude_range: (min, max) amplitude of breathing motion
        frequency_range: (min, max) breathing frequency in breaths per minute
        time_per_projection: Time per projection in seconds (vendor-defined)
        uncertainty_range: (min, max) temporal uncertainty in seconds, sampled per patient
        phase_offset_breathing: Phase offset for breathing in radians
        amplitude_heart: Amplitude of heart motion (for THORAX type)
        frequency_heartbeat: Heart rate in beats per minute, default 80
        phase_offset_heart: Phase offset for heart in radians
        rng: Random number generator (uses default if None)

    Returns:
        MotionConfig with randomly sampled parameters
    """
    if rng is None:
        rng = np.random.default_rng()

    amplitude = rng.uniform(*amplitude_range)
    frequency = int(rng.uniform(*frequency_range))
    uncertainty = rng.uniform(*uncertainty_range)

    return MotionConfig(
        motion_type=motion_type,
        amplitude_breathing=amplitude,
        amplitude_heart=amplitude_heart,
        phase_offset_breathing=phase_offset_breathing,
        phase_offset_heart=phase_offset_heart,
        frequency_breathing=frequency,
        frequency_heartbeat=frequency_heartbeat,
        time_per_projection=time_per_projection,
        uncertainty=uncertainty
    )

#TODO: include function for random sampling of motion config
class MotionConfig(BaseModel):
    class MotionType(Enum):
        PELVIS = auto()
        THORAX = auto()
        ABDOMEN = auto()

    """Configuration for motion."""
    motion_type: MotionType
    amplitude_breathing: float  # = 10
    amplitude_heart: float|None = None
    phase_offset_breathing: float = 0.0 # phase offset for breathing state in radians - 2*pi equals 0 phase offset
    phase_offset_heart: float = 0.0  # phase offset for heart state in radians - 2*pi equals 0 phase offset
    # frequncy --> breathing
    contour_name: str | None = None  # DEPRECATED: Surrogate organs now inferred from motion_type
    # phases:int #= 10
    # info for dynamic loading of CT
    frequency_breathing: int  # = 20 # breaths per minute (12-20)
    frequency_heartbeat: int|None=None  # = 80 # heartbeats per minute (60-100)
    time_per_projection: float  # = 0.18 # seconds
    uncertainty: float  # = 0.02 # seconds
    frequency_breathing_uncertainty: float|None = None  # breaths per minute

    _time_per_breathing_cycle: int = PrivateAttr(default=None)
    _time_per_heartbeat_cycle: int = PrivateAttr(default=None)

    @field_validator('motion_type', mode='before')
    @classmethod
    def validate_motion_type(cls, v):
        """Accept string values like 'PELVIS' or 'THORAX' and convert to enum."""
        if isinstance(v, str):
            try:
                return cls.MotionType[v.upper()]
            except KeyError:
                raise ValueError(f"Invalid motion type: {v}. Must be {', '.join([m.name for m in cls.MotionType])}")
        return v

    @field_validator('contour_name', mode='after')
    @classmethod
    def validate_contour_name(cls, v):
        """Warn if contour_name is provided (deprecated field)."""
        if v is not None:
            import warnings
            warnings.warn(
                "The 'contour_name' field is deprecated. "
                "Motion surrogate organs are now automatically inferred from motion_type "
                "(PELVIS → bowel, THORAX → heart/aorta/lung/spine, "
                "ABDOMEN → bowel + heart/aorta/lung/spine). "
                "This field will be removed in v2.0.",
                DeprecationWarning,
                stacklevel=2
            )
        return v

    @property
    def time_per_breathing_cycle(self) -> float:
        if self._time_per_breathing_cycle is None:
            self._time_per_breathing_cycle = 60 / (self.frequency_breathing)
        return self._time_per_breathing_cycle

    @property
    def time_per_breathing_half_cycle(self) -> float:
        return self.time_per_breathing_cycle / 2

    @property
    def time_per_heartbeat_cycle(self) -> float:
        if self._time_per_heartbeat_cycle is None:
            self._time_per_heartbeat_cycle = 60 / (self.frequency_heartbeat)
        return self._time_per_heartbeat_cycle

    @property
    def time_per_heartbeat_half_cycle(self) -> float:
        return self.time_per_heartbeat_cycle / 2

    @property
    def effective_frequency_breathing(self) -> float:
        """Return breathing frequency including uncertainty if set."""
        if self.frequency_breathing_uncertainty is not None:
             return np.random.normal(self.frequency_breathing, self.frequency_breathing_uncertainty)
        return self.frequency_breathing


class PhantomConfig(BaseModel):
    """Configuration for phantom generation."""
    phantom_path: str
    intensity_factor: float = 1.0
    water_threshold: float = 300.0
    enhancement_factor: float = 4.0
    lower_threshold: float = -600.0
    body_threshold: float = -200.0
    gaussian_sigma: float = 2.0
    noise_range: Tuple[float, float] = (-35.0, 35.0)
