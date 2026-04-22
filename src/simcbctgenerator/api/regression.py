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

"""High-level API for CT-CBCT registration and regression dataset generation."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import SimpleITK as sitk
import numpy as np
from pydantic import BaseModel, field_validator

from simcbctgenerator.registration import (
    RegistrationConfig,
    RegistrationEngine,
    assert_docker_prerequisites,
)
from simcbctgenerator.api.reconstruction import ProjectionPipeline
from simcbctgenerator.utils.config import CBCTSystemConfig, Vendor
from simcbctgenerator.utils.fov import create_circular_mask_per_slice

logger = logging.getLogger(__name__)


def _create_fov_mask(cbct_image: sitk.Image) -> sitk.Image:
    """Create a circular field-of-view mask from a CBCT image.

    Derives an initial binary mask from non-background voxels, applies
    morphological cleaning, extracts the boundary contour, and then
    delegates to :func:`~simcbctgenerator.cli.regression.create_circular_mask_per_slice`
    for proper per-slice circular FOV fitting (the same logic used by the
    CLI regression pipeline).

    Parameters
    ----------
    cbct_image:
        CBCT volume whose FOV defines the mask.

    Returns
    -------
    sitk.Image
        Binary uint8 mask with 1 inside the FOV.
    """
    cbct_array = sitk.GetArrayFromImage(cbct_image)
    mask_array = (cbct_array != -1024).astype(np.uint8)

    mask_image = sitk.GetImageFromArray(mask_array)
    mask_image.CopyInformation(cbct_image)

    # Morphological closing + erosion/dilation to get a clean boundary
    mask_closed = sitk.BinaryMorphologicalClosing(mask_image, (12, 12, 0))
    erod_mask = sitk.ErodeObjectMorphology(mask_closed, (1, 1, 0))
    dil_mask = sitk.DilateObjectMorphology(erod_mask, (10, 10, 1))
    boundary = sitk.BinaryContour(dil_mask)

    # Reuse the pipeline's per-slice circular mask fitting
    return create_circular_mask_per_slice(boundary)


def _crop_to_mask(
    images: Dict[str, sitk.Image],
    mask: sitk.Image,
) -> Dict[str, sitk.Image]:
    """Crop co-registered images to the bounding box of *mask*.

    Uses :class:`sitk.LabelShapeStatisticsImageFilter` to find the
    foreground bounding box — the same approach used in
    :func:`~simcbctgenerator.cli.regression.process_patient_registration`.

    Parameters
    ----------
    images:
        Named images to crop (must share the same grid as *mask*).
    mask:
        Binary mask whose bounding box defines the crop region.

    Returns
    -------
    dict
        Cropped images with the same keys.
    """
    label_stats = sitk.LabelShapeStatisticsImageFilter()
    label_stats.Execute(mask)

    if 1 not in label_stats.GetLabels():
        logger.warning("No foreground label in mask — returning images uncropped")
        return images

    bbox = label_stats.GetBoundingBox(1)
    size = [bbox[3], bbox[4], bbox[5]]
    index = [bbox[0], bbox[1], bbox[2]]

    cropped: Dict[str, sitk.Image] = {}
    for name, img in images.items():
        cropped[name] = sitk.RegionOfInterest(img, size=size, index=index)

    first_key = next(iter(cropped))
    logger.info(
        f"Cropped images from {mask.GetSize()} to {cropped[first_key].GetSize()}"
    )
    return cropped


class RegressionPipeline(BaseModel):
    """High-level API for generating aligned CT-CBCT pairs for regression.

    The pipeline registers a planning CT to a CBCT (or simulated CBCT)
    using rigid + deformable registration, creates a field-of-view mask,
    and returns the aligned pair as in-memory images, leaving any I/O to
    the caller.

    Internally the heavy lifting is delegated to existing building blocks:

    * FOV mask creation reuses
      :func:`~simcbctgenerator.cli.regression.create_circular_mask_per_slice`
      (per-slice circular fitting with ``find_fov_center``).
    * Registration reuses
      :class:`~simcbctgenerator.registration.RegistrationEngine`.
    * Optional CBCT simulation reuses
      :class:`~simcbctgenerator.api.reconstruction.ProjectionPipeline`.

    Two workflows are supported:

    1. **With real clinical CBCT** — provide ``cbct_image`` directly.
    2. **With simulated CBCT** — call ``generate_simulated_cbct`` first
       (or let ``run()`` handle it when ``simulate_cbct=True``).

    Example::

        pipeline = RegressionPipeline()
        results = pipeline.run(
            ct_image=ct,
            cbct_image=cbct,
            output_dir="workdir",
        )
        registered_ct = results["registered_ct"]
    """

    # --- Registration settings ----------------------------------------------
    use_rigid: bool = True
    use_deformable: bool = True
    registration_config: Optional[RegistrationConfig] = None

    # --- CBCT simulation settings (optional) --------------------------------
    vendor: Vendor = Vendor.ELEKTA
    gpu: bool = True
    threads: int = 8
    max_block_index: int = 200
    polychromatic: bool = False
    correct_contrast_media: bool = False

    class Config:
        arbitrary_types_allowed = True

    @field_validator("vendor", mode="before")
    @classmethod
    def _validate_vendor(cls, v: Vendor | str) -> Vendor:
        return Vendor.from_value(v)

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register_ct_to_cbct(
        self,
        ct_image: sitk.Image,
        cbct_image: sitk.Image,
        output_dir: Union[str, Path],
        fixed_mask: Optional[sitk.Image] = None,
        moving_mask: Optional[sitk.Image] = None,
    ) -> Tuple[sitk.Image, Optional[Path]]:
        """Register a planning CT onto a CBCT volume.

        Delegates to
        :meth:`~simcbctgenerator.registration.RegistrationEngine.register_ct_to_cbct`.

        Parameters
        ----------
        ct_image:
            Planning CT to align.
        cbct_image:
            CBCT used as the fixed reference.
        output_dir:
            Working directory for intermediate registration files.
        fixed_mask:
            Optional mask for the CBCT (fixed image).
        moving_mask:
            Optional mask for the CT (moving image).

        Returns
        -------
        tuple
            ``(registered_ct, transform_path)`` — the aligned CT and the
            path to the deformation transform file (or ``None``).
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Fail fast if the deformable stage would later invoke Docker without the
        # required image available — users otherwise see a swallowed subprocess error.
        if self.use_deformable:
            assert_docker_prerequisites()

        reg_config = self.registration_config or RegistrationConfig()
        engine = RegistrationEngine(reg_config)

        try:
            registered_ct, transform_file = engine.register_ct_to_cbct(
                ct_image=ct_image,
                cbct_image=cbct_image,
                output_dir=output_dir,
                use_rigid=self.use_rigid,
                use_deformable=self.use_deformable,
                fixed_mask=fixed_mask,
                moving_mask=moving_mask,
            )
        finally:
            engine.cleanup_persistent_data_dir()

        logger.info("CT-to-CBCT registration complete")
        return registered_ct, transform_file

    # ------------------------------------------------------------------
    # Optional CBCT simulation
    # ------------------------------------------------------------------

    def generate_simulated_cbct(
        self,
        ct_image: sitk.Image,
        output_dir: Union[str, Path],
        system_config: Optional[CBCTSystemConfig] = None,
        geometry_xml: Optional[Union[str, Path]] = None,
        metadata_yaml: Optional[Union[str, Path]] = None,
        cbct_image: Optional[sitk.Image] = None,
        cleanup_temp: bool = True,
    ) -> sitk.Image:
        """Generate a simulated CBCT from the planning CT.

        Delegates to :meth:`ProjectionPipeline.run`.

        Parameters
        ----------
        ct_image:
            Planning CT volume.
        cbct_image:
            Reference CBCT for isocenter derivation. If omitted, the
            isocenter defaults to the physical centre of ct_image.
        geometry_xml:
            Optional path to RTK geometry XML override.
        metadata_yaml:
            Optional path to per-patient metadata YAML override.
        output_dir:
            Directory for projection and reconstruction output.
        cleanup_temp:
            Remove temporary DRR projection files after reconstruction.

        Returns
        -------
        sitk.Image
            The reconstructed simulated CBCT.
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
            cleanup_temp=cleanup_temp,
        )

    # ------------------------------------------------------------------
    # FOV mask & cropping
    # ------------------------------------------------------------------

    @staticmethod
    def create_fov_mask(cbct_image: sitk.Image) -> sitk.Image:
        """Create a circular field-of-view mask from a CBCT image.

        Delegates to :func:`_create_fov_mask` which internally calls
        :func:`~simcbctgenerator.cli.regression.create_circular_mask_per_slice`.
        """
        return _create_fov_mask(cbct_image)

    @staticmethod
    def crop_to_mask(
        images: Dict[str, sitk.Image],
        mask: sitk.Image,
    ) -> Dict[str, sitk.Image]:
        """Crop co-registered images to the bounding box of *mask*.

        Delegates to :func:`_crop_to_mask`.
        """
        return _crop_to_mask(images, mask)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(
        self,
        ct_image: sitk.Image,
        output_dir: Union[str, Path] = "output_regression",
        cbct_image: Optional[sitk.Image] = None,
        crop_to_fov: bool = True,
        simulate_cbct: bool = False,
        system_config: Optional[CBCTSystemConfig] = None,
        geometry_xml: Optional[Union[str, Path]] = None,
        metadata_yaml: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Any]:
        """Run the full regression pipeline.

        Steps:

        1. *(Optional)* Generate a simulated CBCT from the planning CT.
        2. Register the CT onto the (real or simulated) CBCT.
        3. Create a FOV mask from the CBCT (reuses
           :func:`~simcbctgenerator.cli.regression.create_circular_mask_per_slice`).
        4. Crop images to the FOV bounding box.

        Parameters
        ----------
        ct_image:
            Planning CT (original resolution).
        output_dir:
            Working directory for registration and optional CBCT simulation.
        cbct_image:
            Real clinical CBCT. Required when ``simulate_cbct=False`` (used as
            the registration target). When ``simulate_cbct=True``, omitting it
            causes the isocenter to be derived from the CT centre instead.
        crop_to_fov:
            Crop all outputs to the FOV bounding box.
        simulate_cbct:
            If ``True``, generate a simulated CBCT first.
        system_config:
            Optional prebuilt system configuration used when ``simulate_cbct=True``.
        geometry_xml:
            Optional RTK geometry XML override (only with ``simulate_cbct=True``).
        metadata_yaml:
            Optional metadata YAML override (only with ``simulate_cbct=True``).

        Returns
        -------
        dict
            ``"cbct_image"``     — the (possibly simulated) CBCT.
            ``"registered_ct"``  — CT aligned to the CBCT grid.
            ``"fov_mask"``       — field-of-view mask.
            ``"transform_path"`` — path to the deformation file (or ``None``).
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Step 1: optionally simulate CBCT
        if simulate_cbct:
            logger.info("Generating simulated CBCT for regression target")
            target_cbct = self.generate_simulated_cbct(
                ct_image=ct_image,
                cbct_image=cbct_image,
                system_config=system_config,
                geometry_xml=geometry_xml,
                metadata_yaml=metadata_yaml,
                output_dir=output_dir / "cbct_simulation",
            )
        else:
            if cbct_image is None:
                raise ValueError("cbct_image is required when simulate_cbct=False")
            target_cbct = cbct_image

        # Step 2: register CT to CBCT
        logger.info("Registering CT to CBCT")
        reg_output = output_dir / "registration"
        registered_ct, transform_path = self.register_ct_to_cbct(
            ct_image=ct_image,
            cbct_image=target_cbct,
            output_dir=reg_output,
        )

        # Step 3: FOV mask (reuses create_circular_mask_per_slice)
        logger.info("Creating FOV mask")
        fov_mask = self.create_fov_mask(target_cbct)

        # Step 4: crop
        if crop_to_fov:
            logger.info("Cropping to FOV bounding box")
            cropped = self.crop_to_mask(
                {"cbct": target_cbct, "ct": registered_ct, "mask": fov_mask},
                fov_mask,
            )
            target_cbct = cropped["cbct"]
            registered_ct = cropped["ct"]
            fov_mask = cropped["mask"]

        return {
            "cbct_image": target_cbct,
            "registered_ct": registered_ct,
            "fov_mask": fov_mask,
            "transform_path": transform_path,
        }
