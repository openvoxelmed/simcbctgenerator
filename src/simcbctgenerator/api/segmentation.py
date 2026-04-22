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

"""High-level API for synthetic CBCT generation with organ segmentation.

Internally delegates to:

* :class:`~simcbctgenerator.api.reconstruction.ProjectionPipeline` for
  standard DRR-based CBCT simulation.
* :class:`~simcbctgenerator.phantom_generator.PhantomGenerator` for the
  fast phantom method.
* :class:`~simcbctgenerator.organ_mask_generator.OrganMaskGenerator` for
  automatic organ segmentation and mask combination (``generate_multi_organ_masks``
  and ``create_combined_mask``).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import SimpleITK as sitk
from pydantic import BaseModel, field_validator

from simcbctgenerator.api.phantom import PhantomPipeline
from simcbctgenerator.api.reconstruction import ProjectionPipeline
from simcbctgenerator.utils.config import CBCTSystemConfig, MotionConfig, Vendor
from simcbctgenerator.utils.config import PhantomConfig

logger = logging.getLogger(__name__)


class SegmentationPipeline(BaseModel):
    """High-level API for synthetic CBCT generation with organ segmentation.

    Combines CBCT simulation (via projection/reconstruction or phantom method)
    with automatic organ segmentation to produce training data for
    segmentation networks (e.g. nnU-Net).

    The pipeline generates:
      - A simulated CBCT volume (input image for the network).
      - Multi-label organ segmentation masks (ground-truth labels).
      - Optionally, the resampled planning CT as a regression target.

    Two generation methods are supported:
      - ``"standard"`` — physically realistic DRR projection + FDK
        reconstruction (requires GPU with CUDA).
      - ``"phantom"`` — fast phantom-based generation (no GPU required).

    Internally the heavy lifting is delegated to existing building blocks:

    * Organ segmentation reuses
      :meth:`~simcbctgenerator.organ_mask_generator.OrganMaskGenerator.generate_multi_organ_masks`.
    * Multi-label mask combination reuses
      :meth:`~simcbctgenerator.organ_mask_generator.OrganMaskGenerator.create_combined_mask`.
    * CBCT simulation reuses
      :class:`~simcbctgenerator.api.reconstruction.ProjectionPipeline` or
      :class:`~simcbctgenerator.phantom_generator.PhantomGenerator`.

    Example::

        pipeline = SegmentationPipeline(
            organ_list=["bowel", "bladder", "rectum"],
        )
        results = pipeline.run(
            ct_image=ct,
            cbct_image=cbct,
            geometry_xml="geometry.xml",
            metadata_yaml="metadata.yaml",
        )
        simulated_cbct = results["simulated_cbct"]
        label_mask = results["label_mask"]
    """

    # --- Generation settings ------------------------------------------------
    method: str = "standard"
    vendor: Vendor = Vendor.ELEKTA
    gpu: bool = True
    threads: int = 8
    max_block_index: int = 200
    polychromatic: bool = False
    correct_contrast_media: bool = False

    # --- Segmentation settings ----------------------------------------------
    organ_list: List[str] = ["bowel", "bladder", "rectum"]
    priority: List[int] = [1, 2, 3]
    use_totalsegmentator: bool = True
    segmentation_device: str = "gpu"
    segmentation_fast_mode: bool = True

    # --- Phantom settings (only used when method="phantom") -----------------
    phantom_config: Optional[PhantomConfig] = None

    class Config:
        arbitrary_types_allowed = True

    @field_validator("method", mode="before")
    @classmethod
    def _validate_method(cls, v: str) -> str:
        v = v.lower()
        if v not in ("standard", "phantom"):
            raise ValueError(
                f"Unknown method {v!r}. Valid options: 'standard', 'phantom'"
            )
        return v

    @field_validator("vendor", mode="before")
    @classmethod
    def _validate_vendor(cls, v: Vendor | str) -> Vendor:
        return Vendor.from_value(v)

    @field_validator("priority", mode="before")
    @classmethod
    def _validate_priority(cls, v, info):
        return v

    # ------------------------------------------------------------------
    # Organ mask helpers — delegate to OrganMaskGenerator
    # ------------------------------------------------------------------

    def _get_organ_mask_generator(self):
        """Lazily import and instantiate an OrganMaskGenerator."""
        from simcbctgenerator.organ_mask_generator import OrganMaskGenerator
        return OrganMaskGenerator(
            fast_mode=self.segmentation_fast_mode,
            device=self.segmentation_device,
        )

    def generate_organ_masks(
        self,
        ct_image: sitk.Image,
        organ_list: Optional[List[str]] = None,
        mask_image: Optional[sitk.Image] = None,
    ) -> Dict[str, sitk.Image]:
        """Generate organ masks from a CT image.

        If *mask_image* is provided it is assumed to be a pre-computed
        multi-label mask and is split into per-organ binary masks using the
        label values in ``self.priority``.  Otherwise, delegates to
        :meth:`~simcbctgenerator.organ_mask_generator.OrganMaskGenerator.generate_multi_organ_masks`.

        Parameters
        ----------
        ct_image:
            Planning CT as a SimpleITK image.
        organ_list:
            Override for ``self.organ_list``.
        mask_image:
            Optional pre-existing multi-label mask (skips auto-segmentation).

        Returns
        -------
        dict
            Mapping of organ names to binary ``sitk.Image`` masks.
        """
        organs = organ_list or self.organ_list

        if mask_image is not None:
            logger.info("Splitting pre-computed multi-label mask into per-organ masks")
            masks: Dict[str, sitk.Image] = {}
            mask_arr = sitk.GetArrayFromImage(mask_image)
            for name, label_val in zip(organs, self.priority):
                binary = (mask_arr == label_val).astype(np.uint8)
                img = sitk.GetImageFromArray(binary)
                img.CopyInformation(mask_image)
                masks[name] = img
            return masks

        # Delegate to OrganMaskGenerator.generate_multi_organ_masks
        logger.info(f"Running TotalSegmentator for organs: {organs}")
        gen = self._get_organ_mask_generator()
        return gen.generate_multi_organ_masks(ct_image, organs)

    def create_combined_label_mask(
        self,
        organ_masks: Dict[str, sitk.Image],
        reference_image: Optional[sitk.Image] = None,
        fov_mask: Optional[sitk.Image] = None,
    ) -> sitk.Image:
        """Merge per-organ binary masks into a single multi-label image.

        Resamples each mask onto *reference_image* (if given), optionally
        clips to *fov_mask*, and then delegates to
        :meth:`~simcbctgenerator.organ_mask_generator.OrganMaskGenerator.create_combined_mask`
        for priority-based label combination.

        Parameters
        ----------
        organ_masks:
            Per-organ binary masks (output of ``generate_organ_masks``).
        reference_image:
            If given, each mask is resampled onto this grid first.
        fov_mask:
            Optional field-of-view mask; labels outside this region are zeroed.

        Returns
        -------
        sitk.Image
            Multi-label segmentation image (uint8).
        """
        # Resample masks to reference grid and apply FOV if needed
        resampler: Optional[sitk.ResampleImageFilter] = None
        if reference_image is not None:
            resampler = sitk.ResampleImageFilter()
            resampler.SetReferenceImage(reference_image)
            resampler.SetInterpolator(sitk.sitkNearestNeighbor)
            resampler.SetDefaultPixelValue(0)

        resampled_masks: Dict[str, sitk.Image] = {}
        for organ_name, mask in organ_masks.items():
            current = mask

            if resampler is not None:
                current = resampler.Execute(current)

            if fov_mask is not None:
                current = sitk.Multiply(
                    sitk.Cast(current, sitk.sitkUInt8),
                    sitk.Cast(fov_mask, sitk.sitkUInt8),
                )

            resampled_masks[organ_name] = current

        # Delegate to OrganMaskGenerator.create_combined_mask
        gen = self._get_organ_mask_generator()
        return gen.create_combined_mask(resampled_masks, self.priority)

    # ------------------------------------------------------------------
    # Main entry points
    # ------------------------------------------------------------------

    def generate_cbct(
        self,
        ct_image: sitk.Image,
        output_dir: Union[str, Path],
        system_config: Optional[CBCTSystemConfig] = None,
        geometry_xml: Optional[Union[str, Path]] = None,
        metadata_yaml: Optional[Union[str, Path]] = None,
        cbct_image: Optional[sitk.Image] = None,
        motion_config: Optional[MotionConfig] = None,
        random_motion_type: Optional[MotionConfig.MotionType] = None,
        motion_surrogate: Optional[Any] = None,
        random_motion_amplitude_range: Tuple[float, float] = (5.0, 20.0),
        random_motion_frequency_range: Tuple[float, float] = (12.0, 20.0),
        random_motion_uncertainty_range: Tuple[float, float] = (0.01, 0.05),
        cleanup_temp: bool = True,
    ) -> sitk.Image:
        """Generate a simulated CBCT using the standard projection method.

        Delegates to :meth:`ProjectionPipeline.run`.

        Returns
        -------
        sitk.Image
            The reconstructed simulated CBCT volume.
        """
        proj_pipeline = ProjectionPipeline(
            vendor=self.vendor,
            correct_contrast_media=self.correct_contrast_media,
            polychromatic=self.polychromatic,
            gpu=self.gpu,
            threads=self.threads,
            max_block_index=self.max_block_index,
        )
        return proj_pipeline.run(
            ct_image=ct_image,
            cbct_image=cbct_image,
            system_config=system_config,
            geometry_xml=geometry_xml,
            metadata_yaml=metadata_yaml,
            output_dir=output_dir,
            motion_config=motion_config,
            random_motion_type=random_motion_type,
            motion_surrogate=motion_surrogate,
            random_motion_amplitude_range=random_motion_amplitude_range,
            random_motion_frequency_range=random_motion_frequency_range,
            random_motion_uncertainty_range=random_motion_uncertainty_range,
            cleanup_temp=cleanup_temp,
        )

    def generate_cbct_phantom(
        self,
        ct_image: sitk.Image,
        phantom_config: Optional[PhantomConfig] = None,
        cbct_image: Optional[sitk.Image] = None,
    ) -> Tuple[sitk.Image, Optional[sitk.Image]]:
        """Generate a simulated CBCT using the fast phantom method.

        Delegates to :class:`~simcbctgenerator.phantom_generator.PhantomGenerator`.

        Parameters
        ----------
        ct_image:
            Planning CT volume.
        cbct_image:
            Reference CBCT (used for resampling the CT to CBCT space).
        phantom_config:
            Override for ``self.phantom_config``.

        Returns
        -------
        tuple
            ``(simulated_cbct, fov_mask)`` — the simulated CBCT and an
            optional field-of-view mask.
        """
        cfg = phantom_config or self.phantom_config
        if cfg is None:
            raise ValueError(
                "phantom_config must be provided either at construction or "
                "as argument to generate_cbct_phantom()"
            )
        pipeline = PhantomPipeline(phantom_config=cfg)
        result = pipeline.run_result(ct_image=ct_image, cbct_image=cbct_image, patient_id="api_phantom")
        return result.cbct, result.fov_mask

    def run(
        self,
        ct_image: sitk.Image,
        system_config: Optional[CBCTSystemConfig] = None,
        geometry_xml: Optional[Union[str, Path]] = None,
        metadata_yaml: Optional[Union[str, Path]] = None,
        output_dir: Union[str, Path] = "output_segmentation",
        cbct_image: Optional[sitk.Image] = None,
        mask_image: Optional[sitk.Image] = None,
        motion_config: Optional[MotionConfig] = None,
        random_motion_type: Optional[MotionConfig.MotionType] = None,
        motion_surrogate: Optional[Any] = None,
        random_motion_amplitude_range: Tuple[float, float] = (5.0, 20.0),
        random_motion_frequency_range: Tuple[float, float] = (12.0, 20.0),
        random_motion_uncertainty_range: Tuple[float, float] = (0.01, 0.05),
        phantom_config: Optional[PhantomConfig] = None,
        cleanup_temp: bool = True,
    ) -> Dict[str, Any]:
        """Run the full segmentation pipeline: CBCT generation + organ masks.

        Steps:

        1. Generate a simulated CBCT (standard or phantom method).
        2. Segment organs in the planning CT (auto via
           :meth:`OrganMaskGenerator.generate_multi_organ_masks` or from
           pre-existing mask).
        3. Resample masks onto the CBCT grid and combine via
           :meth:`OrganMaskGenerator.create_combined_mask`.
        4. Resample the planning CT to the simulated CBCT grid.

        Parameters
        ----------
        ct_image:
            Planning CT (native resolution, not resampled).
        system_config:
            Optional prebuilt system configuration for the standard method.
        cbct_image:
            Reference CBCT for isocenter derivation (standard) or CT
            resampling (phantom). If omitted, the CT centre is used as
            the isocenter and the CT is kept at native resolution.
        geometry_xml:
            Optional path to RTK geometry XML override for ``method="standard"``.
        metadata_yaml:
            Optional path to per-patient metadata YAML override for ``method="standard"``.
        output_dir:
            Working directory for intermediate projection files
            (used by the standard method).
        mask_image:
            Optional pre-computed multi-label mask (skips auto-segmentation).
        motion_config / random_motion_type / motion_surrogate:
            Motion parameters (forwarded to ``generate_cbct``).
        phantom_config:
            Override phantom config for the phantom method.
        cleanup_temp:
            Remove temporary projection files after reconstruction.

        Returns
        -------
        dict
            ``"simulated_cbct"``  — the generated CBCT volume.
            ``"label_mask"``      — multi-label segmentation on the CBCT grid.
            ``"organ_masks"``     — per-organ binary masks (CT space).
            ``"fov_mask"``        — field-of-view mask (if available).
            ``"resampled_ct"``    — planning CT resampled to the CBCT grid.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        fov_mask: Optional[sitk.Image] = None

        if self.method == "standard":
            logger.info("Generating simulated CBCT (standard projection method)")
            simulated_cbct = self.generate_cbct(
                ct_image=ct_image,
                cbct_image=cbct_image,
                system_config=system_config,
                geometry_xml=geometry_xml,
                metadata_yaml=metadata_yaml,
                output_dir=output_dir,
                motion_config=motion_config,
                random_motion_type=random_motion_type,
                motion_surrogate=motion_surrogate,
                random_motion_amplitude_range=random_motion_amplitude_range,
                random_motion_frequency_range=random_motion_frequency_range,
                random_motion_uncertainty_range=random_motion_uncertainty_range,
                cleanup_temp=cleanup_temp,
            )
        else:
            logger.info("Generating simulated CBCT (phantom method)")
            simulated_cbct, fov_mask = self.generate_cbct_phantom(
                ct_image=ct_image,
                cbct_image=cbct_image,
                phantom_config=phantom_config,
            )

        logger.info("Generating organ segmentation masks")
        organ_masks = self.generate_organ_masks(
            ct_image=ct_image, mask_image=mask_image,
        )

        logger.info("Creating combined multi-label mask on CBCT grid")
        label_mask = self.create_combined_label_mask(
            organ_masks=organ_masks,
            reference_image=simulated_cbct,
            fov_mask=fov_mask,
        )

        logger.info("Resampling planning CT to simulated CBCT grid")
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(simulated_cbct)
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetDefaultPixelValue(-1000)
        resampled_ct = resampler.Execute(ct_image)

        if fov_mask is not None:
            resampled_ct = sitk.Mask(
                sitk.Cast(resampled_ct, sitk.sitkInt16),
                sitk.Cast(fov_mask, sitk.sitkUInt8),
                -1024,
            )

        return {
            "simulated_cbct": simulated_cbct,
            "label_mask": label_mask,
            "organ_masks": organ_masks,
            "fov_mask": fov_mask,
            "resampled_ct": resampled_ct,
        }
