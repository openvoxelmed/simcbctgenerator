"""Standard motion + projection + reconstruction simulator."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional, Tuple, Union

import numpy as np
import SimpleITK as sitk

from simcbctgenerator.cbct_reconstruction import SyntheticCBCTReconstruction
from simcbctgenerator.generate_4d_ct import FourDCTGenerator
from simcbctgenerator.generate_projections import DRRGenerator, double_sigmoid_profile, gaussian
from simcbctgenerator.utils.config import (
    CBCTSystemConfig,
    GeometryConfig,
    MotionConfig,
    ReconstructionVolumeConfig,
    Vendor,
    sample_motion_config,
)
from simcbctgenerator.utils.physics_config import (
    MANUFACTURER_DEFAULTS,
    load_cbct_physics_from_yaml,
    load_geometry_from_yaml,
)

from .base import BaseCBCTSimulator, SimulationResult

logger = logging.getLogger(__name__)


class StandardCBCTSimulator(BaseCBCTSimulator):
    """Canonical motion + projection + reconstruction simulator."""

    def __init__(
        self,
        vendor: Vendor | str = Vendor.ELEKTA,
        correct_contrast_media: bool = False,
        polychromatic: bool = False,
        gpu: bool = True,
        threads: int = 8,
        max_block_index: int = 200,
        no_scatter: bool = False,
        no_noise: bool = False,
    ):
        self.vendor = Vendor.from_value(vendor)
        self.correct_contrast_media = correct_contrast_media
        self.polychromatic = polychromatic
        self.gpu = gpu
        self.threads = threads
        self.max_block_index = max_block_index
        self.no_scatter = no_scatter
        self.no_noise = no_noise

    @staticmethod
    def compute_isocenter(image: sitk.Image) -> np.ndarray:
        origin = np.array(image.GetOrigin())
        size = np.array(image.GetSize(), dtype=float)
        spacing = np.array(image.GetSpacing())
        return origin + (size - 1) * spacing / 2.0

    def build_system_config(
        self,
        geometry_xml: Optional[Union[str, Path]] = None,
        metadata_yaml: Optional[Union[str, Path]] = None,
    ) -> CBCTSystemConfig:
        vendor_defaults = MANUFACTURER_DEFAULTS[self.vendor.value]
        system_config = CBCTSystemConfig.for_vendor(self.vendor)

        physics_updates = {
            "threads": self.threads,
            "max_block_index": self.max_block_index,
            "polychromatic": self.polychromatic,
            "no_scatter": self.no_scatter,
            "no_noise": self.no_noise,
        }
        if self.polychromatic:
            physics_updates.update({"T1": -200.0, "T2": 400.0})

        system_config = system_config.with_physics(**physics_updates)

        if metadata_yaml is not None:
            metadata_yaml = Path(metadata_yaml)
            physics = load_cbct_physics_from_yaml(metadata_yaml)
            physics = physics.model_copy(update={
                "spr": vendor_defaults["spr"],
                "saturation_factor": vendor_defaults["saturation_factor"],
                "bp_amplitude": vendor_defaults["bp_amplitude"],
                "bp_std": vendor_defaults["bp_std"],
                "bp_floor": vendor_defaults["bp_floor"],
                "bp_ds_edge1": vendor_defaults.get("bp_ds_edge1", 0.0),
                "bp_ds_slope1": vendor_defaults.get("bp_ds_slope1", 0.0),
                "bp_ds_Afloor": vendor_defaults.get("bp_ds_Afloor", 0.0),
                "bp_ds_edge2": vendor_defaults.get("bp_ds_edge2", 0.0),
                "bp_ds_slope2": vendor_defaults.get("bp_ds_slope2", 0.0),
                "photon_gain": vendor_defaults["photon_gain"],
                "inherent_filtration_al_mm": vendor_defaults.get("inherent_filtration_al_mm", 0.0),
                "filtration_material": vendor_defaults["filtration_material"],
                "filtration_mm": vendor_defaults["filtration_mm"],
                "threads": self.threads,
                "max_block_index": self.max_block_index,
                "polychromatic": self.polychromatic,
                "no_scatter": self.no_scatter,
                "no_noise": self.no_noise,
                "T1": -200.0 if self.polychromatic else physics.T1,
                "T2": 400.0 if self.polychromatic else physics.T2,
            })
            system_config = system_config.model_copy(update={"physics": physics})

            geometry = load_geometry_from_yaml(metadata_yaml)
            system_config = system_config.model_copy(update={
                "geometry": GeometryConfig(
                    detector_offset=geometry["detector_offset_x"],
                    source_origin_distance=vendor_defaults["source_origin_distance"],
                    source_detector_distance=vendor_defaults["source_detector_distance"],
                    detector_size_h=geometry["detector_size_h"],
                    detector_size_w=geometry["detector_size_w"],
                    detector_pixels_h=geometry["detector_pixels_h"],
                    detector_pixels_w=geometry["detector_pixels_w"],
                    start_angle=0.0,
                    end_angle=360.0,
                    angle_increments=1.0,
                    geometry_xml_path=system_config.geometry.geometry_xml_path,
                ),
                "reconstruction_volume": ReconstructionVolumeConfig(
                    recon_size=geometry["recon_size"],
                    recon_origin=geometry["recon_origin"],
                    recon_spacing=geometry["recon_spacing"],
                ),
            })

        if geometry_xml is not None:
            system_config = system_config.with_geometry(geometry_xml_path=str(Path(geometry_xml)))

        return system_config

    def _apply_contrast_media_correction(
        self,
        ct_image: sitk.Image,
        cm_mask_image: sitk.Image,
    ) -> sitk.Image:
        try:
            import cupy as cp
            from cupyx.scipy.ndimage import gaussian_filter as cp_gaussian_filter
        except ImportError as exc:
            raise RuntimeError("cupy is required for contrast-media correction.") from exc

        if cm_mask_image.GetSize() != ct_image.GetSize():
            cm_mask_image = sitk.Resample(
                cm_mask_image, ct_image, sitk.Transform(), sitk.sitkNearestNeighbor, 0, cm_mask_image.GetPixelID()
            )
        ct_gpu = cp.asarray(sitk.GetArrayFromImage(ct_image).astype(np.float32))
        mask_gpu = cp.asarray(sitk.GetArrayFromImage(cm_mask_image).astype(np.uint8))
        cm = mask_gpu.astype(bool) & (ct_gpu > 50)
        blurred = cp_gaussian_filter(cm.astype(cp.float32), (0.0, 1.0, 1.0), mode="constant", cval=0)
        noise = cp.random.normal(loc=1.0, scale=0.02, size=cm.shape)
        corrected = sitk.GetImageFromArray(cp.asnumpy(ct_gpu - blurred * ct_gpu * noise * 0.92))
        corrected.CopyInformation(ct_image)
        return corrected

    def _save_motion_config(self, output_dir: Path, motion_config: MotionConfig) -> None:
        data = {
            "motion_type": motion_config.motion_type.name,
            "amplitude_breathing": motion_config.amplitude_breathing,
            "amplitude_heart": motion_config.amplitude_heart,
            "phase_offset_breathing": motion_config.phase_offset_breathing,
            "phase_offset_heart": motion_config.phase_offset_heart,
            "contour_name": motion_config.contour_name,
            "frequency_breathing": motion_config.frequency_breathing,
            "frequency_heartbeat": motion_config.frequency_heartbeat,
            "time_per_projection": motion_config.time_per_projection,
            "uncertainty": motion_config.uncertainty,
            "time_per_breathing_cycle": motion_config.time_per_breathing_cycle,
        }
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "motion_config.json", "w") as f:
            json.dump(data, f, indent=4)

    def _save_stacked_projections(
        self,
        projections_sitk: sitk.Image,
        output_dir: Path,
        system_config: CBCTSystemConfig,
        include_beam_profile: bool = False,
    ) -> None:
        arr = sitk.GetArrayFromImage(projections_sitk)
        # Export the stacked projections as uint16 detector counts so callers can
        # recover log-attenuation with `-log(counts / 65535)`. Using an
        # approximation like exp(11.1) slightly exceeds uint16 max and clips all
        # open-beam pixels to 65535, which collapses the dynamic range.
        linear = np.iinfo(np.uint16).max * np.exp(-arr)
        if include_beam_profile:
            off = system_config.geometry.detector_offset
            pw = system_config.geometry.detector_pixels_w
            ps = system_config.pixel_size[0]
            x = np.linspace(-pw * ps / 2, pw * ps / 2, pw) + off
            phys = system_config.physics
            if phys.bp_ds_slope1 != 0.0:
                bp_norm = double_sigmoid_profile(
                    x, phys.bp_ds_Afloor, phys.bp_ds_edge1, phys.bp_ds_slope1, phys.bp_ds_edge2, phys.bp_ds_slope2
                ).astype(np.float32)
            else:
                bp = np.maximum(gaussian(x, system_config.bp_amplitude, 0, system_config.bp_std), system_config.bp_floor)
                bp_norm = (bp / bp.max()).astype(np.float32)
            linear = linear * bp_norm

        arr = np.clip(linear, 0, np.iinfo(np.uint16).max).astype(np.uint16)
        out = sitk.GetImageFromArray(arr)
        px = system_config.pixel_size
        out.SetSpacing((px[0], px[1], 1.0))
        out.SetOrigin((-(arr.shape[2] - 1) / 2 * px[0], -(arr.shape[1] - 1) / 2 * px[1], 1.0))
        output_dir.mkdir(parents=True, exist_ok=True)
        sitk.WriteImage(out, str(output_dir / "projections_simulated.mha"))

    def run(
        self,
        patient,
        output_dir: Union[str, Path],
        geometry_xml: Optional[Union[str, Path]] = None,
        metadata_yaml: Optional[Union[str, Path]] = None,
        system_config: Optional[CBCTSystemConfig] = None,
        cbct_image: Optional[sitk.Image] = None,
        cm_mask_image: Optional[sitk.Image] = None,
        save_stacked: bool = True,
        include_beam_profile: bool = False,
        motion_config: Optional[MotionConfig] = None,
        random_motion_type: Optional[MotionConfig.MotionType] = None,
        motion_surrogate: Optional[Any] = None,
        random_motion_amplitude_range: Tuple[float, float] = (5.0, 20.0),
        random_motion_frequency_range: Tuple[float, float] = (12.0, 20.0),
        random_motion_uncertainty_range: Tuple[float, float] = (0.01, 0.05),
        cleanup_temp: bool = True,
    ) -> SimulationResult:
        if motion_config is not None and random_motion_type is not None:
            raise ValueError("Provide either motion_config or random_motion_type, not both.")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        if system_config is None:
            system_config = self.build_system_config(geometry_xml, metadata_yaml)

        # Projector geometry is set up by Patient.from_images / loaders before run() is
        # called.  A second call here would re-apply the shift on an already-shifted CT
        # (compounding the error), so we only fall back to setting it up if the patient
        # was constructed without going through those paths.
        if not hasattr(patient, 'shifted_origin'):
            reference_image = cbct_image if cbct_image is not None else patient.ct_image
            iso_center = self.compute_isocenter(reference_image)
            patient.set_projector_geometry(iso_center=iso_center, reference_cbct=cbct_image)

        if motion_surrogate is not None:
            patient.motion_surrogate = motion_surrogate

        if self.correct_contrast_media:
            if cm_mask_image is None and getattr(patient.config, "cm_mask", None):
                cm_mask_image = patient.mask_dictionary.get(patient.config.cm_mask)
            if cm_mask_image is None:
                logger.info("Segmenting bowel for contrast-media correction via TotalSegmentator")
                from simcbctgenerator.organ_mask_generator import OrganMaskGenerator
                gen = OrganMaskGenerator(fast_mode=True, device="gpu")
                cm_mask_image = gen.generate_multi_organ_masks(patient.ct_image, ["bowel"])["bowel"]
            patient.ct_image = self._apply_contrast_media_correction(patient.ct_image, cm_mask_image)
            patient.ct_array = sitk.GetArrayFromImage(patient.ct_image).astype(np.float32)

        active_motion_config = motion_config
        if random_motion_type is not None:
            active_motion_config = sample_motion_config(
                motion_type=random_motion_type,
                amplitude_range=random_motion_amplitude_range,
                frequency_range=random_motion_frequency_range,
                time_per_projection=MANUFACTURER_DEFAULTS[self.vendor.value]["time_per_projection"],
                uncertainty_range=random_motion_uncertainty_range,
            )
            self._save_motion_config(output_dir, active_motion_config)

        ct_generator = None
        if active_motion_config is not None:
            ct_generator = FourDCTGenerator(active_motion_config)
            ct_generator.initialize(patient)

        proj_dir = output_dir / "drr_temp"
        drr = DRRGenerator(output_dir=proj_dir, system_config=system_config)
        projections = drr.generate_all_projections(
            patient=patient,
            ct_generator=ct_generator,
            return_projections=save_stacked,
            save_individual=True,
        )

        if save_stacked and projections is not None:
            self._save_stacked_projections(projections, output_dir, system_config, include_beam_profile)

        reconstructor = SyntheticCBCTReconstruction(system_config, gpu=self.gpu)
        simulated_cbct = reconstructor.reconstruct(proj_dir)
        sitk.WriteImage(simulated_cbct, str(output_dir / "cbct_simulated.mha"))

        if cleanup_temp and proj_dir.exists():
            for file in proj_dir.iterdir():
                file.unlink()
            proj_dir.rmdir()

        return SimulationResult(
            cbct=simulated_cbct,
            fov_mask=getattr(reconstructor, "fov_img", None),
            system_config=system_config,
            projections=projections,
            sampled_motion_config=active_motion_config if random_motion_type is not None else None,
        )
