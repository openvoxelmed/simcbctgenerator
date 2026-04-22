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

"""Module for generating organ masks using TotalSegmentator and alpha shapes."""

from __future__ import annotations

import SimpleITK as sitk
import numpy as np
import logging
from typing import Dict, List, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    import nibabel as nib

logger = logging.getLogger(__name__)

MOTION_SURROGATE_THORAX_ORGANS = [
    "lung",
    "heart",
    "aorta",
    "spine"
]

MOTION_SURROGATE_PELVIS_ORGANS = ['bowel']

MOTION_SURROGATE_ABDOMEN_ORGANS = MOTION_SURROGATE_THORAX_ORGANS + MOTION_SURROGATE_PELVIS_ORGANS

class OrganMaskError(Exception):
    """Base exception for organ mask generation errors."""
    pass


class InvalidInputError(OrganMaskError):
    """Exception raised for invalid input."""
    pass


class SegmentationError(OrganMaskError):
    """Exception raised during segmentation."""
    pass


class OrganMaskGenerator:
    """
    Generator for automatic organ segmentation using TotalSegmentator.

    Supports any organ available in TotalSegmentator (117+ organs).
    Provides special handling for motion surrogate masks used in 4D simulation.
    """

    # Mapping of user-friendly names to TotalSegmentator labels
    SUPPORTED_ORGANS = {
        'bowel': ['small_bowel', 'colon', 'duodenum'],
        'bladder': ['urinary_bladder'],
        'heart': ['heart'],
        'lung': ['lung_upper_lobe_left', 'lung_upper_lobe_right',
                 'lung_middle_lobe_right', 'lung_lower_lobe_left', 'lung_lower_lobe_right'],
        'liver': ['liver'],
        'kidney': ['kidney_left', 'kidney_right'],
        'spleen': ['spleen'],
        'stomach': ['stomach'],
        'esophagus': ['esophagus'],
        'pancreas': ['pancreas'],
        'aorta': ['aorta'],
        'spine': ['vertebrae_L5', 'vertebrae_L4', 'vertebrae_L3', 'vertebrae_L2', 'vertebrae_L1',
                  'vertebrae_T12', 'vertebrae_T11', 'vertebrae_T10', 'vertebrae_T9', 'vertebrae_T8',
                  'vertebrae_T7', 'vertebrae_T6', 'vertebrae_T5', 'vertebrae_T4', 'vertebrae_T3',
                  'vertebrae_T2', 'vertebrae_T1'],
        #THORAX OARs
        "lung_l": ["lung_upper_lobe_left", "lung_lower_lobe_left"],
        "lung_r": ["lung_upper_lobe_right", "lung_middle_lobe_right", "lung_lower_lobe_right"],
        "spinal_cord": ["spinal_cord"]
    }

    def __init__(self, fast_mode: bool = False, device: str = "gpu"):
        """
        Initialize OrganMaskGenerator.

        Args:
            fast_mode: Use fast mode (lower quality, faster processing)
            device: Device for TotalSegmentator ("gpu" or "cpu")
        """
        self.fast_mode = fast_mode
        self.device = device
        logger.info(f"OrganMaskGenerator initialized (fast={fast_mode}, device={device})")

    def generate_organ_mask(self, ct_image: sitk.Image, organ_name: str) -> sitk.Image:
        """
        Generate mask for a single organ (for general use, e.g., segmentation export).

        This method generates standard organ masks WITHOUT alpha-shape smoothing.

        Args:
            ct_image: CT volume as SimpleITK Image
            organ_name: Organ name (e.g., 'bowel', 'bladder', 'rectum')

        Returns:
            Binary mask as SimpleITK Image

        Raises:
            InvalidInputError: If inputs are invalid
            SegmentationError: If segmentation fails

        Example:
            >>> generator = OrganMaskGenerator()
            >>> ct = sitk.ReadImage("ct.mha")
            >>> bowel_mask = generator.generate_organ_mask(ct, 'bowel')
        """
        if not isinstance(ct_image, sitk.Image):
            raise InvalidInputError(f"Expected sitk.Image, got {type(ct_image)}")

        if organ_name not in self.SUPPORTED_ORGANS:
            raise InvalidInputError(
                f"Unsupported organ '{organ_name}'. "
                f"Supported organs: {list(self.SUPPORTED_ORGANS.keys())}"
            )

        logger.info(f"Generating {organ_name} mask for image size: {ct_image.GetSize()}")

        # Get TotalSegmentator labels for this organ
        roi_subset = self.SUPPORTED_ORGANS[organ_name]

        # Run segmentation
        nib_image = self._sitk_to_nibabel(ct_image)
        seg_image_nib = self._run_totalsegmentator(nib_image, roi_subset)
        mask_array = self._extract_organ_labels(seg_image_nib, organ_name)
        # NO alpha-shape smoothing for general organ masks
        output_mask = self._create_output_mask(mask_array, ct_image)

        logger.info(f"{organ_name.capitalize()} mask generation completed")
        return output_mask

    def generate_multi_organ_masks(self, ct_image: sitk.Image,
                                    organ_list: List[str]) -> Dict[str, sitk.Image]:
        """
        Generate masks for multiple organs in a single TotalSegmentator call.

        More efficient than calling generate_organ_mask() multiple times.
        These masks do NOT have alpha-shape smoothing applied.

        Args:
            ct_image: CT volume as SimpleITK Image
            organ_list: List of organ names (e.g., ['bowel', 'bladder'])

        Returns:
            Dictionary mapping organ names to their binary masks

        Example:
            >>> generator = OrganMaskGenerator()
            >>> ct = sitk.ReadImage("ct.mha")
            >>> masks = generator.generate_multi_organ_masks(ct, ['bowel', 'bladder'])
            >>> bowel_mask = masks['bowel']
        """
        if not isinstance(ct_image, sitk.Image):
            raise InvalidInputError(f"Expected sitk.Image, got {type(ct_image)}")

        # Validate all organ names
        for organ_name in organ_list:
            if organ_name not in self.SUPPORTED_ORGANS:
                raise InvalidInputError(
                    f"Unsupported organ '{organ_name}'. "
                    f"Supported organs: {list(self.SUPPORTED_ORGANS.keys())}"
                )

        logger.info(f"Generating multi-organ masks: {organ_list}")

        # Check all labels
        for organ_name in organ_list:
            if self.SUPPORTED_ORGANS.get(organ_name, None) is None:
                raise TypeError(f"{organ_name} is not implemented. Possible organs are: {list(self.SUPPORTED_ORGANS.keys())}")


        # Extract each organ separately
        result_masks = {}
        for organ_name in organ_list:
            # Run segmentation once for all organs
            nib_image = self._sitk_to_nibabel(ct_image)
            seg_image_nib = self._run_totalsegmentator(nib_image, self.SUPPORTED_ORGANS[organ_name])
            mask_array = self._extract_organ_labels(seg_image_nib, organ_name)
            output_mask = self._create_output_mask(mask_array, ct_image)
            result_masks[organ_name] = output_mask
            logger.info(f"Processed {organ_name} mask")

        logger.info("Multi-organ mask generation completed")
        return result_masks

    def generate_motion_surrogate_mask(
        self, ct_image: sitk.Image, region: str
    ) -> Tuple[sitk.Image | Dict[str, sitk.Image], List[str]]:
        """
        Generate organ mask(s) for use as motion surrogate in 4D CT simulation.

        This is the equivalent of the old BowelMaskGenerator.generate_bowel_mask().
        For PELVIS: applies alpha-shape smoothing per slice to create smooth motion boundaries.
        For THORAX: generates individual organ masks without alpha-shape smoothing.
        For ABDOMEN: generates both thorax and pelvis masks for blended motion model.

        IMPORTANT: PELVIS uses alpha-shape smoothing, which is specifically
        needed for motion simulation but NOT for general organ segmentation.

        Args:
            ct_image: CT volume as SimpleITK Image
            region: Motion region type ('PELVIS', 'THORAX', or 'ABDOMEN')

        Returns:
            For PELVIS: (sitk.Image, ['bowel'])
            For THORAX: (Dict[str, sitk.Image], ['heart', 'aorta', 'lung', 'spine'])
            For ABDOMEN: (Dict[str, sitk.Image], ['lung', 'heart', 'aorta', 'spine', 'bowel'])

        Example:
            >>> generator = OrganMaskGenerator()
            >>> ct = sitk.ReadImage("ct.mha")
            >>> surrogate, organ_names = generator.generate_motion_surrogate_mask(ct, 'PELVIS')
            >>> surrogate, organ_names = generator.generate_motion_surrogate_mask(ct, 'THORAX')
            >>> surrogate, organ_names = generator.generate_motion_surrogate_mask(ct, 'ABDOMEN')
        """
        if not isinstance(ct_image, sitk.Image):
            raise InvalidInputError(f"Expected sitk.Image, got {type(ct_image)}")

        if not isinstance(region, str) or region.upper() not in ["PELVIS", "THORAX", "ABDOMEN"]:
            raise InvalidInputError(
                f"Invalid region '{region}'. Must be 'PELVIS', 'THORAX', or 'ABDOMEN'"
            )

        logger.info(f"Generating motion surrogate mask for: {region}")

        nib_image = self._sitk_to_nibabel(ct_image)

        # Get TotalSegmentator labels for this organ
        if region.upper() == "PELVIS":
            organ_name = MOTION_SURROGATE_PELVIS_ORGANS[0]

            roi_subset = self.SUPPORTED_ORGANS[organ_name]
            seg_image_nib = self._run_totalsegmentator(nib_image, roi_subset)
            mask_array = self._extract_organ_labels(seg_image_nib, organ_name)

            # Apply alpha-shape smoothing (equivalent to old _apply_convex_hull_per_slice)
            smoothed_array = self._apply_alpha_shape_per_slice(mask_array)
            output_mask = self._create_output_mask(smoothed_array, ct_image)

            logger.info(f"Motion surrogate mask for {region} completed")
            return output_mask, MOTION_SURROGATE_PELVIS_ORGANS

        elif region.upper() == "THORAX":
            result_masks = {}
            for sub in MOTION_SURROGATE_THORAX_ORGANS:
                roi_subset = self.SUPPORTED_ORGANS[sub]
                seg_image_nib = self._run_totalsegmentator(nib_image, roi_subset)
                mask_array = self._extract_organ_labels(seg_image_nib, sub)
                output_mask = self._create_output_mask(mask_array, ct_image)
                result_masks[sub] = output_mask
                logger.info(f"Processed {sub} mask")

            logger.info(f"Motion surrogate mask for {region} completed")
            return result_masks, MOTION_SURROGATE_THORAX_ORGANS

        else:  # region.upper() == "ABDOMEN"
            result_masks = {}

            # Generate thorax organ masks (no alpha-shape smoothing)
            for sub in MOTION_SURROGATE_THORAX_ORGANS:
                roi_subset = self.SUPPORTED_ORGANS[sub]
                seg_image_nib = self._run_totalsegmentator(nib_image, roi_subset)
                mask_array = self._extract_organ_labels(seg_image_nib, sub)
                output_mask = self._create_output_mask(mask_array, ct_image)
                result_masks[sub] = output_mask
                logger.info(f"Processed {sub} mask")

            # Generate pelvis organ mask (with alpha-shape smoothing)
            organ_name = MOTION_SURROGATE_PELVIS_ORGANS[0]
            roi_subset = self.SUPPORTED_ORGANS[organ_name]
            seg_image_nib = self._run_totalsegmentator(nib_image, roi_subset)
            mask_array = self._extract_organ_labels(seg_image_nib, organ_name)
            smoothed_array = self._apply_alpha_shape_per_slice(mask_array)
            result_masks[organ_name] = self._create_output_mask(smoothed_array, ct_image)
            logger.info(f"Processed {organ_name} mask (with alpha-shape smoothing)")

            logger.info(f"Motion surrogate mask for {region} completed")
            return result_masks, MOTION_SURROGATE_ABDOMEN_ORGANS

    def create_combined_mask(self, organ_masks: Dict[str, sitk.Image],
                            priorities: List[int]) -> sitk.Image:
        """
        Create multi-label mask from individual organ masks with priority-based overlap resolution.

        Args:
            organ_masks: Dictionary mapping organ names to their binary masks
            priorities: List of priority values (1=highest) corresponding to organ order

        Returns:
            Multi-label mask where voxel value indicates organ:
            - 0: background
            - 1: first organ (highest priority)
            - 2: second organ
            - etc.

        Example:
            >>> masks = generator.generate_multi_organ_masks(ct, ['bowel', 'bladder', 'rectum'])
            >>> combined = generator.create_combined_mask(masks, priorities=[1, 2, 3])
        """
        if len(organ_masks) != len(priorities):
            raise InvalidInputError(
                f"Number of organ masks ({len(organ_masks)}) must match "
                f"number of priorities ({len(priorities)})"
            )

        # Sort organs by priority (lowest priority value = highest priority)
        organ_priority_pairs = list(zip(organ_masks.keys(), priorities))
        sorted_organs = sorted(organ_priority_pairs, key=lambda x: x[1], reverse=True)

        # Get reference image
        first_organ = list(organ_masks.values())[0]
        combined_array = np.zeros(sitk.GetArrayFromImage(first_organ).shape, dtype=np.uint8)

        # Apply masks in order of priority (low priority value first, so it gets overwritten)
        for label, (organ_name, priority) in enumerate(reversed(sorted_organs), start=1):
            mask = organ_masks[organ_name]
            mask_array = sitk.GetArrayFromImage(mask)
            combined_array[mask_array > 0] = label

        # Create output image
        combined_image = sitk.GetImageFromArray(combined_array)
        combined_image.CopyInformation(first_organ)
        combined_image = sitk.Cast(combined_image, sitk.sitkUInt8)

        logger.info(f"Created combined mask with {len(organ_masks)} organs")
        return combined_image

    # Private helper methods

    def _sitk_to_nibabel(self, sitk_image: sitk.Image) -> nib.Nifti1Image:
        """Convert SimpleITK to nibabel with proper affine transformation."""
        import nibabel as nib
        array = np.transpose(sitk.GetArrayFromImage(sitk_image), (2, 1, 0))
        spacing = np.array(sitk_image.GetSpacing())
        origin = np.array(sitk_image.GetOrigin())
        direction = np.array(sitk_image.GetDirection()).reshape(3, 3)

        affine = np.eye(4)
        affine[:3, :3] = direction @ np.diag(spacing)
        affine[:3, 3] = origin

        # Convert from LPS to RAS coordinate system
        lps_to_ras = np.diag([-1, -1, 1, 1])
        affine = lps_to_ras @ affine

        return nib.Nifti1Image(array, affine)

    def _run_totalsegmentator(self, nib_image: nib.Nifti1Image,
                              roi_subset: List[str]) -> nib.Nifti1Image:
        """Run TotalSegmentator Python API on nibabel image."""
        try:
            from totalsegmentator.python_api import totalsegmentator
        except ImportError as e:
            raise SegmentationError(
                "TotalSegmentator is required for automatic organ segmentation. "
                "Install it with: uv sync --extra segmentation"
            ) from e

        logger.info(f"Running TotalSegmentator for {len(roi_subset)} ROIs (30-60 seconds)")
        seg_image = totalsegmentator(
            nib_image,
            roi_subset=roi_subset,
            fast=self.fast_mode,
            device=self.device,

        )
        return seg_image

    def _extract_organ_labels(self, seg_image_nib: nib.Nifti1Image,
                             organ_name: str) -> np.ndarray:
        """
        Extract specific organ labels into binary mask.

        This is equivalent to the old _extract_bowel_labels() method.
        """
        seg_array = np.transpose(seg_image_nib.get_fdata(), (2, 1, 0))

        # TotalSegmentator returns multi-label image
        # Any non-zero value indicates presence of organ structure
        organ_mask = (seg_array > 0).astype(np.uint8)

        if np.sum(organ_mask) == 0:
            logger.warning(f"No {organ_name} structures found in segmentation")
            # Don't raise error, just return empty mask
        else:
            logger.debug(f"Extracted {np.sum(organ_mask)} {organ_name} voxels")

        return organ_mask

    def _apply_alpha_shape_per_slice(self, mask_array: np.ndarray) -> np.ndarray:
        """
        Apply 2D alpha shape (concave hull) on each axial slice.

        This is equivalent to the old _apply_convex_hull_per_slice() method.
        This smoothing is ONLY used for motion surrogate masks, not general segmentation.

        This smooths the mask boundaries while preserving concave features,
        which is important for smooth motion field computation.
        """
        output = np.zeros_like(mask_array)
        num_slices = mask_array.shape[0]
        processed = 0

        for z in range(num_slices):
            slice_2d = mask_array[z, :, :]
            points = np.column_stack(np.where(slice_2d > 0))

            if len(points) < 3:
                output[z, :, :] = slice_2d
                continue

            try:
                import alphashape
                # Create alpha shape with automatic alpha parameter selection
                alpha_shape = alphashape.alphashape(points, alpha=0.05)

                if alpha_shape is None or alpha_shape.is_empty:
                    output[z, :, :] = slice_2d
                    continue

                # Vectorized point-in-polygon test using meshgrid
                ny, nx = slice_2d.shape
                y_grid, x_grid = np.mgrid[0:ny, 0:nx]
                all_points = np.column_stack([y_grid.ravel(), x_grid.ravel()])

                # Test containment using shapely
                from shapely.vectorized import contains
                inside = contains(alpha_shape, all_points[:, 0], all_points[:, 1])
                output[z, :, :] = inside.reshape((ny, nx)).astype(np.uint8)
                processed += 1

            except Exception as e:
                logger.warning(f"Alpha shape failed for slice {z}: {e}")
                output[z, :, :] = slice_2d

        logger.debug(f"Processed {processed}/{num_slices} slices with alpha shape")
        return output.astype(np.uint8)

    def _create_output_mask(self, mask_array: np.ndarray,
                           reference_image: sitk.Image) -> sitk.Image:
        """Create SimpleITK mask with reference image metadata."""
        mask_img = sitk.GetImageFromArray(mask_array)
        mask_img.SetSpacing(reference_image.GetSpacing())
        mask_img.SetOrigin(reference_image.GetOrigin())
        mask_img.SetDirection(reference_image.GetDirection())
        mask_img = sitk.Cast(mask_img, sitk.sitkUInt8)
        return mask_img
