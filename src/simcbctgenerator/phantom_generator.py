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
Phantom Generator for Synthetic CBCT Generation
"""

import SimpleITK as sitk
import numpy as np
from pathlib import Path
from typing import Tuple
from simcbctgenerator.patient import Patient
from simcbctgenerator.utils.config import PhantomConfig
from simcbctgenerator import utils

logger = utils.setup_logger()


class PhantomGenerator:
    """Generator for synthetic CBCT using phantom method."""

    def __init__(self, config: PhantomConfig):
        self.config = config
        self.patient = None
        self.synthetic_cbct = None
        self._initialize_phantom_measurement()

    def _initialize_phantom_measurement(self):

        # Load phantom and get CT from patient
        phantom = self._load_medical_image(self.config.phantom_path)

        # Normalize phantom measurement
        normalized_phantom, water_mask = self._normalize_phantom_measurement(
            phantom,
            self.config.water_threshold,
            self.config.enhancement_factor,
            self.config.lower_threshold
        )
        # Apply phantom FOV mask
        self.fov_mask = sitk.NotEqual(phantom, 0.0)

        self.phantom = normalized_phantom * self.config.intensity_factor

    def _load_medical_image(self, file_path: str) -> sitk.Image:
        """Load medical image using SimpleITK."""
        logger.debug(f"Loading: {file_path}")
        image = sitk.ReadImage(file_path)

        # use 4mm spacing
        new_size = (410, 410, 66)
        new_spacing = (1., 1., 4.)

        # Calculate the center of the original image
        original_center = image.TransformContinuousIndexToPhysicalPoint([(sz-1)/2.0 for sz in image.GetSize()])

        # Calculate the new origin to center the resampled image at the same location
        new_origin = [center - (sz-1)*sp/2.0 for center, sz, sp in zip(original_center, new_size, new_spacing)]

        # Resample with calculated origin
        image_resampled = sitk.Resample(
            image,
            size=new_size,
            interpolator=sitk.sitkLinear,
            outputSpacing=new_spacing,
            outputOrigin=new_origin,
            outputDirection=image.GetDirection()
        )
        # center voxel as origin
        image_resampled.SetOrigin([-204.5, -204.5, -130])
        return image_resampled

    def _normalize_phantom_measurement(self, phantom: sitk.Image, water_threshold: float,
                                     enhancement_factor: float, lower_threshold: float) -> Tuple[sitk.Image, np.ndarray]:
        """Normalize phantom measurement to water-equivalent values."""
        logger.debug("Normalizing phantom measurement...")

        phantom_array = sitk.GetArrayFromImage(phantom)
        # Create water mask
        water_mask = phantom_array > water_threshold

        # Calculate average pixel value in water region
        water_pixels = phantom_array[water_mask]
        if len(water_pixels) == 0:
            logger.warning(f"No pixels above threshold {water_threshold} found!")
            water_average = phantom_array.mean()
        else:
            water_average = water_pixels.mean()

        # Normalize phantom
        normalized_phantom = phantom_array.copy().astype(np.float32)
        normalized_phantom = (normalized_phantom - water_average) * enhancement_factor
        normalized_phantom[~water_mask] = 0

        # Exclude low values
        low_value_mask = normalized_phantom < lower_threshold
        normalized_phantom[low_value_mask] = 0
        final_mask = water_mask & ~low_value_mask

        normalized_phantom_sitk = sitk.GetImageFromArray(normalized_phantom)
        normalized_phantom_sitk.CopyInformation(phantom)
        normalized_phantom_sitk.SetOrigin(phantom.GetOrigin())
        normalized_phantom_sitk.SetSpacing(phantom.GetSpacing())
        normalized_phantom_sitk.SetDirection(phantom.GetDirection())

        return normalized_phantom_sitk, final_mask

    def initialize(self, patient: Patient):
        """Initialize generator with patient data."""
        self.patient = patient
        if self.patient.config is not None and self.patient.config.cm_mask is not None:
            self.patient.correct_CM()
        logger.info(f"Initialized rectangular phantom generator for patient {patient.id}")

    def generate(self) -> sitk.Image:
        """Generate synthetic CBCT using rectangular phantom method."""
        if self.patient is None:
            raise ValueError("Generator not initialized. Call initialize() first.")

        logger.info(f"Generating synthetic CBCT for patient {self.patient.id}")

        ct_image = self.patient.ct_image
        # shift origin to have origin in the isocenter
        ct_image.SetOrigin(self.patient.shifted_origin)

        # Align phantom and CT centers
        aligned_ct = self._align_ct_to_phantom(ct_image)

        # Create synthetic CBCT
        self.synthetic_cbct = self._create_synthetic_cbct(aligned_ct)

        logger.info(f"Generated synthetic CBCT with shape {self.synthetic_cbct.GetSize()[::-1]}")
        return self.synthetic_cbct

    def save(self, output_path: Path, file_name: str) -> None:
        """Save synthetic CBCT to specified path."""
        if self.synthetic_cbct is None:
            raise ValueError("No synthetic CBCT generated. Call generate() first.")

        output_path.mkdir(parents=True, exist_ok=True)
        output_file = output_path / f"{file_name}_0000.nii.gz"

        sitk.WriteImage(self.synthetic_cbct, str(output_file))

        logger.info(f"Saved synthetic CBCT to {output_file}")

    def reset(self):
        """Reset generator state."""
        self.patient = None
        self.synthetic_cbct = None


    def _align_ct_to_phantom(self, ct: sitk.Image) -> sitk.Image:
        """Align phantom to CT using centered transform."""

        # Resample phantom to CT space
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(self.phantom)
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetDefaultPixelValue(-1024)

        aligned_phantom_sitk = resampler.Execute(ct)
        return aligned_phantom_sitk

    def _create_body_mask_with_soft_edges(self, ct_sitk: sitk.Image) -> Tuple[sitk.Image, sitk.Image]:
        """Create soft body mask from CT with gaussian smoothing."""
        # Create binary mask
        body_mask = sitk.BinaryThreshold(ct_sitk, lowerThreshold=self.config.body_threshold,
                                        upperThreshold=32767, insideValue=1, outsideValue=0)

        # Fill holes and apply morphological closing
        filled_mask = sitk.BinaryFillhole(body_mask, fullyConnected=False, foregroundValue=1.0)
        closing_filter = sitk.BinaryMorphologicalClosingImageFilter()
        closing_filter.SetKernelRadius([2, 2, 2])
        closing_filter.SetKernelType(sitk.sitkBall)
        closed_mask = closing_filter.Execute(filled_mask)

        # Apply gaussian smoothing for soft edges
        float_mask = sitk.Cast(closed_mask, sitk.sitkFloat32)
        gaussian_filter = sitk.DiscreteGaussianImageFilter()
        gaussian_filter.SetVariance(self.config.gaussian_sigma**2)
        soft_mask_sitk = gaussian_filter.Execute(float_mask)

        # Create inverse mask
        ones_image = sitk.Image(soft_mask_sitk.GetSize(), sitk.sitkFloat32)
        ones_image.CopyInformation(soft_mask_sitk)
        ones_image = ones_image + 1.0
        inverse_mask_sitk = ones_image - soft_mask_sitk

        return soft_mask_sitk, inverse_mask_sitk

    def _add_background_noise(self, image_sitk: sitk.Image, inverse_mask_sitk: sitk.Image) -> sitk.Image:
        """Add random noise to background regions."""
        noise_image = sitk.Image(image_sitk.GetSize(), sitk.sitkFloat32)
        noise_image.CopyInformation(image_sitk)

        noise_min, noise_max = self.config.noise_range
        noise_mean = (noise_min + noise_max) / 2.0
        noise_std = (noise_max - noise_min) / 6.0

        scaled_noise = sitk.AdditiveGaussianNoise(noise_image, standardDeviation=noise_std, mean=noise_mean, seed=42)
        background_noise = scaled_noise * inverse_mask_sitk

        float_image = sitk.Cast(image_sitk, sitk.sitkFloat32)
        return float_image + background_noise

    def _create_synthetic_cbct(self, aligned_ct: sitk.Image) -> sitk.Image:
        """Create synthetic CBCT by combining CT and phantom."""
        # Create soft body mask
        soft_mask_sitk, inverse_mask_sitk = self._create_body_mask_with_soft_edges(aligned_ct)

        # Create synthetic CBCT with phantom
        synthetic_cbct_sitk = sitk.Cast(aligned_ct, sitk.sitkFloat32)

        # Add phantom contribution
        synthetic_cbct_sitk = synthetic_cbct_sitk + self.phantom

        # Apply soft masking
        body_contribution = synthetic_cbct_sitk * soft_mask_sitk
        noisy_background = self._add_background_noise(aligned_ct, inverse_mask_sitk)
        background_contribution = noisy_background * inverse_mask_sitk

        # Combine body and background
        final_synthetic_cbct = body_contribution + background_contribution

        # Apply FOV mask
        final_synthetic_cbct_masked = sitk.Mask(final_synthetic_cbct, self.fov_mask, -1024)

        return sitk.Cast(final_synthetic_cbct_masked, sitk.sitkInt16)
