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

"""High-level API for standard CBCT projection generation and reconstruction."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional, Tuple, Union

import SimpleITK as sitk
from pydantic import BaseModel, field_validator

from simcbctgenerator.patient import Patient
from simcbctgenerator.simulation import StandardCBCTSimulator
from simcbctgenerator.utils.config import (
    CBCTSystemConfig,
    MotionConfig,
    PatientConfig,
    Vendor,
)
from simcbctgenerator.utils.arg_parser import discover_config_presets, resolve_config_path
from simcbctgenerator.utils.physics_config import MANUFACTURER_DEFAULTS

logger = logging.getLogger(__name__)


def save_motion_config(output_dir: Path, motion_config: MotionConfig) -> None:
    simulator = StandardCBCTSimulator()
    simulator._save_motion_config(Path(output_dir), motion_config)


class ProjectionPipeline(BaseModel):
    """High-level API for CBCT projection generation and reconstruction.

    High-level knobs: vendor type, contrast-media correction, polychromatic
    mode, and hardware settings. By default the pipeline builds a vendor
    system configuration. Users can optionally provide a prebuilt
    ``CBCTSystemConfig`` or override defaults via metadata YAML and RTK XML.
    """

    vendor: Vendor = Vendor.ELEKTA
    correct_contrast_media: bool = False
    polychromatic: bool = False
    gpu: bool = True
    threads: int = 8
    max_block_index: int = 200
    no_scatter: bool = False
    no_noise: bool = False

    @field_validator("vendor", mode="before")
    @classmethod
    def _validate_vendor(cls, v: Vendor | str) -> Vendor:
        if isinstance(v, Vendor):
            return v
        v = v.lower()
        if v not in MANUFACTURER_DEFAULTS:
            valid = ", ".join(f'"{k}"' for k in MANUFACTURER_DEFAULTS)
            raise ValueError(f"Unknown vendor {v!r}. Valid options: {valid}")
        return Vendor(v)

    @staticmethod
    def available_config_presets() -> list[str]:
        return sorted(discover_config_presets())

    @staticmethod
    def config_preset_path(name: str) -> Path:
        return resolve_config_path(name)

    def _create_simulator(self) -> StandardCBCTSimulator:
        return StandardCBCTSimulator(
            vendor=self.vendor,
            correct_contrast_media=self.correct_contrast_media,
            polychromatic=self.polychromatic,
            gpu=self.gpu,
            threads=self.threads,
            max_block_index=self.max_block_index,
            no_scatter=self.no_scatter,
            no_noise=self.no_noise,
        )

    @staticmethod
    def _patient_from_image(
        ct_image: Optional[sitk.Image],
        cbct_image: Optional[sitk.Image] = None,
    ) -> Patient:
        """Build a bare projection patient from a CT image (single origin shift)."""
        if ct_image is None:
            raise ValueError("Provide either a `patient` or a `ct_image`.")
        return Patient.from_images(
            ct_image=ct_image,
            config=PatientConfig(
                plan_dir=".",
                ct_dir=".",
                cbct_dir=".",
                export_structures=[],
                priority=[],
                image_modality="synrad",
                use_totalsegmentator=False,
            ),
            reference_cbct=cbct_image,
            patient_id="projection_pipeline",
        )

    def generate_projections(
        self,
        ct_image: Optional[sitk.Image] = None,
        output_dir: Union[str, Path] = "output_projection",
        system_config: Optional[CBCTSystemConfig] = None,
        geometry_xml: Optional[Union[str, Path]] = None,
        metadata_yaml: Optional[Union[str, Path]] = None,
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
        patient: Optional[Patient] = None,
    ) -> tuple[Optional[sitk.Image], Any]:
        """Generate projections and return `(stacked_projections, system_config)`.

        Provide either a pre-built ``patient`` or a ``ct_image``.
        """
        patient = patient or self._patient_from_image(ct_image, cbct_image)
        simulator = self._create_simulator()
        result = simulator.run(
            patient=patient,
            system_config=system_config,
            geometry_xml=geometry_xml,
            metadata_yaml=metadata_yaml,
            output_dir=output_dir,
            cbct_image=cbct_image,
            cm_mask_image=cm_mask_image,
            save_stacked=save_stacked,
            include_beam_profile=include_beam_profile,
            motion_config=motion_config,
            random_motion_type=random_motion_type,
            motion_surrogate=motion_surrogate,
            random_motion_amplitude_range=random_motion_amplitude_range,
            random_motion_frequency_range=random_motion_frequency_range,
            random_motion_uncertainty_range=random_motion_uncertainty_range,
            cleanup_temp=False,
        )
        return result.projections, result.system_config

    def reconstruct(
        self,
        proj_dir: Union[str, Path],
        system_config: CBCTSystemConfig,
        output_dir: Optional[Union[str, Path]] = None,
    ) -> sitk.Image:
        """Reconstruct CBCT from individual projection files via FDK."""
        from simcbctgenerator.cbct_reconstruction import SyntheticCBCTReconstruction

        simulated_cbct = SyntheticCBCTReconstruction(system_config, gpu=self.gpu).reconstruct(Path(proj_dir))

        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            sitk.WriteImage(simulated_cbct, str(output_dir / "cbct_simulated.mha"))
            logger.info(f"Saved simulated CBCT to {output_dir / 'cbct_simulated.mha'}")

        return simulated_cbct

    def run(
        self,
        ct_image: Optional[sitk.Image] = None,
        output_dir: Union[str, Path] = "output_projection",
        system_config: Optional[CBCTSystemConfig] = None,
        geometry_xml: Optional[Union[str, Path]] = None,
        metadata_yaml: Optional[Union[str, Path]] = None,
        cbct_image: Optional[sitk.Image] = None,
        cm_mask_image: Optional[sitk.Image] = None,
        cleanup_temp: bool = True,
        motion_config: Optional[MotionConfig] = None,
        random_motion_type: Optional[MotionConfig.MotionType] = None,
        motion_surrogate: Optional[Any] = None,
        random_motion_amplitude_range: Tuple[float, float] = (5.0, 20.0),
        random_motion_frequency_range: Tuple[float, float] = (12.0, 20.0),
        random_motion_uncertainty_range: Tuple[float, float] = (0.01, 0.05),
        patient: Optional[Patient] = None,
        return_result: bool = False,
    ):
        """Generate projections and reconstruct in one call.

        Provide either a pre-built ``patient`` (its existing projector geometry
        is reused as-is) or a ``ct_image`` (a patient is built from it). Passing
        a patient avoids re-shifting an already-positioned CT, keeping the
        reconstruction in the patient's coordinate frame.

        Returns the reconstructed CBCT, or the full ``SimulationResult`` (which
        also carries the FOV mask) when ``return_result=True``.

        See generate_projections for motion parameter documentation.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        patient = patient or self._patient_from_image(ct_image, cbct_image)
        result = self._create_simulator().run(
            patient=patient,
            system_config=system_config,
            geometry_xml=geometry_xml,
            metadata_yaml=metadata_yaml,
            output_dir=output_dir,
            cbct_image=cbct_image,
            cm_mask_image=cm_mask_image,
            motion_config=motion_config,
            random_motion_type=random_motion_type,
            motion_surrogate=motion_surrogate,
            random_motion_amplitude_range=random_motion_amplitude_range,
            random_motion_frequency_range=random_motion_frequency_range,
            random_motion_uncertainty_range=random_motion_uncertainty_range,
            cleanup_temp=cleanup_temp,
        )
        return result if return_result else result.cbct
