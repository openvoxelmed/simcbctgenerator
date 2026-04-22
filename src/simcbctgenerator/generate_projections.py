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

"""Module for generating Digitally Reconstructed Radiographs (DRRs) from CT volumes."""

from pathlib import Path
from simcbctgenerator.projector import CBCT, Projector, Volume, geo
import SimpleITK as sitk
from simcbctgenerator.utils.config import CBCTSystemConfig
from simcbctgenerator.utils.math import expit
from simcbctgenerator.utils.physics_config import photon_flux, generate_spectral_data
import numpy as np
from tqdm import tqdm
import logging
from simcbctgenerator.utils import log_time
from typing import TypeAlias
import cupy as cp

FourDCTGenerator: TypeAlias = "FourDCTGenerator"

logger = logging.getLogger(__name__)


def gaussian(x, amplitude, mean, std_dev):
    return amplitude * np.exp(-((x - mean) ** 2) / (2 * std_dev ** 2))


def double_sigmoid_profile(x, Afloor, edge1, slope1, edge2, slope2):
    """Normalized double-sigmoid beam profile shape (peak = 1).

    Parameters use the code-x convention: x = linspace(-pw*ps/2, pw*ps/2, pw) + offset.

    Args:
        x:      detector x-coordinates [mm]
        Afloor: floor/peak ratio for the left plateau (0-1)
        edge1:  centre of the left sigmoid [mm]
        slope1: slope of the left sigmoid [1/mm] (positive → rises left-to-right)
        edge2:  centre of the right sigmoid [mm]
        slope2: slope of the right sigmoid [1/mm] (negative → continues rising rightward)

    Returns:
        Normalized profile array, shape (len(x),), peak value = 1.
    """
    rise = expit((x - edge1) * slope1)          # 0 → 1 from left
    second = 1.0 - expit((x - edge2) * slope2)  # shape of right fall/rise
    raw = rise * second + Afloor * (1.0 - rise)
    return raw / raw.max()


class DRRGenerator:
    """Generator for Digitally Reconstructed Radiographs (DRRs) from CT volumes."""

    def __init__(self, output_dir: Path, system_config: CBCTSystemConfig):
        self.system_config = system_config
        self.output_dir = output_dir
        self.projector: Projector = None
        self.lps_from_ijk: geo.FrameTransform = None
        self.calib = None
        self.delete_projections()
        self._define_scatter()

        # Polychromatic spectral data (generated once, reused for all projections)
        self.spectral_data = None
        if self.system_config.physics.polychromatic:
            logger.info('Generating spectral data for polychromatic projection...')
            self.spectral_data = generate_spectral_data(
                kvp=self.system_config.physics.kv,
                mAs=self.system_config.physics.mAs,
                sdd_mm=self.system_config.effective_source_detector_distance,
                inherent_filtration_al_mm=self.system_config.physics.inherent_filtration_al_mm,
                filtration_material=self.system_config.physics.filtration_material,
                filtration_mm=self.system_config.physics.filtration_mm,
                target_angle=14.0,
            )

    def delete_projections(self):
        logger.info('delete projections...')
        self.lps_from_ijk = None
        if self.output_dir.exists():
            for file in self.output_dir.iterdir():
                file.unlink()


    def save_projections(self, imgs:np.ndarray, outdir:Path, index:int):
        """Save projection image to disk.

        Args:
            imgs: Projection image array
            outdir: Output directory
            index: Projection index (used as filename)
        """
        if not outdir.exists():
            outdir.mkdir(exist_ok=True, parents=True)
        sitkimg = sitk.GetImageFromArray(imgs)
        sitkimg.SetSpacing((self.system_config.pixel_size[0], self.system_config.pixel_size[1]))
        file_path = Path(f'{outdir}/{index:04d}.mhd')
        sitk.WriteImage(sitkimg, file_path)

    def set_projector(self, iso_center:np.ndarray = np.array([0,0,0])):
        # Get initial offset (first projection or fixed offset)
        initial_offset_x = self.system_config.geometry.detector_offset
        initial_offset_y = 0.0
        if self.system_config.has_xml_geometry:
            # XML offsets are in RTK convention, negate for DRR
            xml_offset_x, xml_offset_y = self.system_config.get_offset_at_index(0)
            initial_offset_x = -xml_offset_x
            initial_offset_y = -xml_offset_y

        self.cbct = CBCT(
        beta=0.0,
        isocenter=iso_center,
        source_to_detector_distance=self.system_config.effective_source_detector_distance,
        source_to_isocenter_vertical_distance=self.system_config.effective_source_origin_distance,
        pixel_size=self.system_config.pixel_size[0],
        sensor_height=self.system_config.geometry.detector_pixels_h,
        sensor_width=self.system_config.geometry.detector_pixels_w,
        detector_offset_x=initial_offset_x,
        detector_offset_y=initial_offset_y
        )

        if self.projector is not None:
            #free memory on gpu
            self.projector.free()
            self.projector = None

        # Convert T1/T2 from HU to density units (same conversion as volume)
        T1_density = 0.0
        T2_density = 0.0
        if self.spectral_data is not None:
            mu_ref = self.spectral_data.mu_ref_water
            T1_density = mu_ref * (self.system_config.physics.T1 + 1000.0) / 10000.0
            T2_density = mu_ref * (self.system_config.physics.T2 + 1000.0) / 10000.0

        self.projector = Projector(
        volume=self.patient,
        device=self.cbct,
        threads=self.system_config.physics.threads,
        spectral_data=self.spectral_data,
        T1=T1_density,
        T2=T2_density
        )
        self.projector.initialize()

    def _drr_alpha(self, angle: float) -> float:
        """Map clinical gantry angle to DRR projector alpha.

        Elekta XVI angles are 180° offset from the DRR coordinate system,
        so a +180 correction is applied.  Varian OBI angles align directly —
        no offset needed.

        Elekta is identified by filtration_material == 'al' (Varian uses 'ti').
        """
        is_elekta = self.system_config.physics.filtration_material.lower() == 'al'
        return float(angle) + 180.0 if is_elekta else float(angle) + 90.0

    def create_projections(self, angle: float):
        """Create projection at specified angle.

        Args:
            angle: Gantry angle in degrees
        """
        alpha = self._drr_alpha(angle)
        # Get per-angle offsets by angle (not index) for proper XML geometry support
        if self.system_config.has_xml_geometry:
            offset_x, offset_y = self.system_config.get_offset_at_angle(angle)
            # XML offsets are in RTK convention, need to negate for DRR
            self.cbct.move_to(
                beta=90,
                alpha=alpha,
                detector_offset_x=-offset_x,
                detector_offset_y=-offset_y
            )
        else:
            self.cbct.move_to(beta=90, alpha=alpha)

        img = self.projector().copy()

        return img

    def convert_patient_volume(self, patient, origin=None):
        spacing = patient.ct_image.GetSpacing()
        volume = patient.ct_array.transpose(2, 1, 0) # x, y, z
        #normalize between 0 and 1
        # (mu*(HU+1000))/1000 = (HU+1000) *(mu/1000) = 0.0178/1000 * (1/0.0178)/(1/0.0178)
        # density = (volume+1000)/53220
        # HU -> density (linear attenuation in mm^-1)
        # Uses spectral mu_ref_water when polychromatic, otherwise the
        # original fixed constant for ~120 kVp effective energy.
        if self.spectral_data is not None:
            mu_ref = self.spectral_data.mu_ref_water          # cm^-1
            density = mu_ref * (volume + 1000.0) / 10000.0    # mm^-1
        else:
            density = (volume + 1000) / 53220                 # original

        density = np.maximum(density, 0.0)   # air / below-air -> 0

        # Use the center of the volume as the "world" coordinates. The origin is the (0, 0, 0) index of the volume in the world frame.
        if self.lps_from_ijk is None:
            if origin is not None:
                translation = origin
            else:
                translation = np.array(patient.ct_image.GetOrigin())#origin

            rotation = np.identity(3)*spacing

            transform = np.block([[rotation, translation.reshape((3,1))],
                                [0,0,0,1]])

            self.lps_from_ijk =geo.FrameTransform(transform)

        # Create the volume object with segmentation
        self.patient = Volume(
            data=np.ascontiguousarray(density),#volume,
            anatomical_from_IJK=self.lps_from_ijk
        )

    def _define_scatter(self):
        # model beam profile
        bp = self._build_beam_profile(self.system_config.geometry.detector_offset,
                                self.system_config.geometry.detector_pixels_h,
                                self.system_config.geometry.detector_pixels_w,
                                self.system_config.pixel_size[0],
                                std=self.system_config.bp_std,
                                amplitude=self.system_config.bp_amplitude,
                                floor=self.system_config.bp_floor,
                                ds_edge1=self.system_config.bp_ds_edge1,
                                ds_slope1=self.system_config.bp_ds_slope1,
                                ds_Afloor=self.system_config.bp_ds_Afloor,
                                ds_edge2=self.system_config.bp_ds_edge2,
                                ds_slope2=self.system_config.bp_ds_slope2)

        # model response correction
        #f1: elekta  amplitude = 1.07, std=522
        #varian: amplitude = 1.06, std=939

        #incident_count requires pixel size, photon flux and mAs
        # Use config photon_flux if specified, otherwise compute via spekpy
        if self.system_config.physics.photon_flux is not None:
            flux = self.system_config.physics.photon_flux
        else:
            logger.info('Computing photon flux via spekpy...')
            flux = photon_flux(self.system_config)
        self.incident_count = bp * self.system_config.physics.mAs * flux * self.system_config.pixel_size[0] * self.system_config.pixel_size[1] * self.system_config.physics.photon_gain
        self.incident_count = np.maximum(self.incident_count, 1e-6)

    def _build_beam_profile(self, offset, pixels_h, pixels_w, pixel_size, std, amplitude, floor=0.0,
                             ds_edge1=0.0, ds_slope1=0.0, ds_Afloor=0.0, ds_edge2=0.0, ds_slope2=0.0):
        length = pixels_w * pixel_size
        x = np.linspace(-length / 2, length / 2, pixels_w) + offset
        if ds_slope1 != 0.0:
            shape = double_sigmoid_profile(x, ds_Afloor, ds_edge1, ds_slope1, ds_edge2, ds_slope2)
            profile = amplitude * shape
        else:
            profile = np.maximum(gaussian(x, amplitude, 0, std), floor)
        return np.ones((pixels_h, pixels_w)) * profile[None, :]

    def add_noise_and_scatter(self, proj):
        primary_transmission = np.exp(-proj)

        # convert from relative intensity to primary count

        primary_count = primary_transmission * self.incident_count

        # Estimate scatter: proportional to primary attenuation
        # More realistic scatter model: scatter is related to the amount of attenuated primary
        attenuated_primary = (primary_transmission < 0.5)
        attenuated_vals = primary_count[attenuated_primary]
        if attenuated_vals.size > 0:
            min_attenuated_primary = np.percentile(attenuated_vals, 5)
        else:
            min_attenuated_primary = self.incident_count

        scatter_count = self.system_config.physics.spr*min_attenuated_primary
        # Total detected photons
        total_count = primary_count
        total_count+= scatter_count #* mask

        # Poisson noise — RTK formulation:
        #   noisy = max(Poisson(total_count), 1);  noisy_proj = log(I0 / noisy)
        raw = np.random.poisson(np.maximum(total_count, 0.0))
        noisy_count = np.maximum(raw, 1).astype(float)  # floor: ≥1 photon (RTK-style)

        # Convert back to transmission (avoid division by zero)
        transmission = np.divide(noisy_count, self.incident_count)

        # Apply calibration if available
        if self.calib is not None:
            transmission *= self.calib

        # Clamp transmission to valid range
        transmission = np.clip(transmission*self.system_config.physics.saturation_factor, 1e-6, 1.0) #*2

        # Convert to log attenuation
        noisy_proj = -np.log(transmission)

        return noisy_proj

    @log_time(logger)
    def generate_all_projections(
        self,
        patient,
        ct_generator: FourDCTGenerator = None,
        return_projections: bool = False,
        save_individual: bool = True
    ):
        """Generate projections for all angles with optional motion.

        Args:
            patient: Patient object or SimpleCTVolume with ct_image, ct_array, and iso_center
            ct_generator: Optional FourDCTGenerator for dynamic motion. If None, static CT is used.
            return_projections: If True, return projections as SimpleITK image. If False, save to disk.
            save_individual: If True, save individual projections to output_dir (for reconstruction)

        Returns:
            SimpleITK image of projections if return_projections=True, otherwise None
        """
        # Setup patient volume and projector
        self.convert_patient_volume(patient)
        self.set_projector(patient.iso_center)

        projs = []

        # Determine description for progress bar
        desc = "Generating projections with motion" if ct_generator else "Generating projections (static)"

        for i, angle in enumerate(pbar := tqdm(self.system_config.effective_angles, desc=desc)):
            # Update volume with deformed CT if motion is enabled
            if ct_generator is not None:
                resampledVolumeGPUArray: cp.ndarray = ct_generator.generate_dynamic_4d_CT(i)
                self.projector.update_volume_textures(volume=resampledVolumeGPUArray)

            pbar.set_description(f'{desc} - angle: {angle:.1f}°')

            # Generate projection
            proj_log = self.create_projections(angle)
            proj_log = self.add_noise_and_scatter(proj_log)


            # Save or collect projections
            if save_individual:
                self.save_projections(proj_log, self.output_dir, index=i)
            if return_projections:
                projs.append(proj_log)

        # Cleanup motion generator if used
        if ct_generator is not None:
            ct_generator.reset()

        # Return as SimpleITK image if requested
        if return_projections:
            img = sitk.GetImageFromArray(np.array(projs))
            img.SetSpacing((1, self.system_config.pixel_size[0], self.system_config.pixel_size[1]))
            img.SetOrigin((1, -(self.system_config.geometry.detector_pixels_w-1)/2 * self.system_config.pixel_size[0],
                            -(self.system_config.geometry.detector_pixels_h-1)/2 * self.system_config.pixel_size[1]))
            return img

        return None
