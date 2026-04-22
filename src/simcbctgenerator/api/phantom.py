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

"""High-level API for phantom-based CBCT generation."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import SimpleITK as sitk
from pydantic import BaseModel

from simcbctgenerator.patient import Patient
from simcbctgenerator.simulation import PhantomCBCTSimulator, SimulationResult
from simcbctgenerator.utils.config import ImagingModality, PatientConfig, PhantomConfig


class PhantomPipeline(BaseModel):
    phantom_config: PhantomConfig

    class Config:
        arbitrary_types_allowed = True

    def run_result(
        self,
        ct_image: sitk.Image,
        cbct_image: Optional[sitk.Image] = None,
        output_dir: Optional[Union[str, Path]] = None,
        patient_id: str = "api_phantom",
    ) -> SimulationResult:
        patient = Patient.from_images(
            ct_image=ct_image,
            config=PatientConfig(
                plan_dir=".",
                ct_dir=".",
                cbct_dir=".",
                export_structures=[],
                priority=[],
                image_modality=ImagingModality.dummy,
                use_totalsegmentator=False,
            ),
            reference_cbct=cbct_image,
            patient_id=patient_id,
        )
        result = PhantomCBCTSimulator(self.phantom_config).run(patient)
        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            sitk.WriteImage(result.cbct, str(output_dir / "cbct_simulated.mha"))
        return result

    def run(
        self,
        ct_image: sitk.Image,
        cbct_image: Optional[sitk.Image] = None,
        output_dir: Optional[Union[str, Path]] = None,
        patient_id: str = "api_phantom",
    ) -> sitk.Image:
        return self.run_result(
            ct_image=ct_image,
            cbct_image=cbct_image,
            output_dir=output_dir,
            patient_id=patient_id,
        ).cbct
