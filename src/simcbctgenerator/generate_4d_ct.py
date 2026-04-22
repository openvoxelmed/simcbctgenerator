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

"""Module to generate 4D CT images with simulated respiratory motion."""



# Import Windows CUDA_PATH for dll (required for some Python versions,
# https://docs.python.org/3/whatsnew/3.8.html#bpo-36085-whatsnew)
import sys
import os

if sys.platform == 'win32':
    os.add_dll_directory(os.path.join(os.environ['CUDA_PATH'], 'bin'))

import SimpleITK as sitk
import numpy as np
from simcbctgenerator.utils.config import MotionConfig
from simcbctgenerator.motion.motion_deformation_model_pelvis import MotionDeformationModelPelvis
from simcbctgenerator.motion.motion_deformation_model_thorax import MotionDeformationModelThorax
from simcbctgenerator.motion.motion_deformation_model_abdomen import MotionDeformationModelAbdomen
from simcbctgenerator.motion.apply_motion_field import ApplyMotionField
import logging
from simcbctgenerator.utils import log_time

logger = logging.getLogger(__name__)

class FourDCTGenerator:
    """Generator for 4D CT with respiratory motion simulation."""

    def __init__(self, motion_config:MotionConfig, createDebugOutput=False):
        self.motion_config = motion_config

        self.rng = np.random.default_rng()

        self._init_resampler()

        self.createDebugOutput = createDebugOutput

        self.instanceApplyMotionFieldWithCupy = None

    def initialize(self, patient):
        self.patient = patient
        self.create_motion_deformation()

    def _init_resampler(self):
        self.resampler = sitk.ResampleImageFilter()
        self.resampler.SetInterpolator(sitk.sitkLinear)

    def resample(self, deformation_transform:sitk.DisplacementFieldTransform):

        self.resampler.SetTransform(deformation_transform)
        self.resampler.SetReferenceImage(self.patient.ct_image)

        return self.resampler.Execute(self.patient.ct_image)

    def _validate_and_generate_surrogate(self):
        """Ensure motion_surrogate matches the expected format for the selected motion type.

        PELVIS expects a single sitk.Image.
        THORAX expects a dict with keys: heart, aorta, lung, spine.
        ABDOMEN expects a dict with keys: heart, aorta, lung, spine, bowel.

        If the existing surrogate is None or has the wrong format / missing
        keys, auto-generate it (or the missing parts) via TotalSegmentator.
        """
        from simcbctgenerator.organ_mask_generator import OrganMaskGenerator

        motion_type = self.motion_config.motion_type
        surrogate = self.patient.motion_surrogate

        if motion_type == MotionConfig.MotionType.PELVIS:
            # Needs a single sitk.Image
            if isinstance(surrogate, sitk.Image):
                return  # already valid
            if isinstance(surrogate, dict) and 'bowel' in surrogate:
                self.patient.motion_surrogate = surrogate['bowel']
                return
            # Otherwise auto-generate
            logger.info("Auto-generating PELVIS motion surrogate using TotalSegmentator")
            generator = OrganMaskGenerator(fast_mode=True, device="gpu")
            self.patient.motion_surrogate, _ = generator.generate_motion_surrogate_mask(
                self.patient.ct_image, "PELVIS"
            )

        elif motion_type == MotionConfig.MotionType.THORAX:
            required = {'heart', 'aorta', 'lung', 'spine'}
            if isinstance(surrogate, dict) and required.issubset(surrogate.keys()):
                return  # already valid
            logger.info("Auto-generating THORAX motion surrogate using TotalSegmentator")
            generator = OrganMaskGenerator(fast_mode=True, device="gpu")
            self.patient.motion_surrogate, _ = generator.generate_motion_surrogate_mask(
                self.patient.ct_image, "THORAX"
            )

        elif motion_type == MotionConfig.MotionType.ABDOMEN:
            required = {'heart', 'aorta', 'lung', 'spine', 'bowel'}
            if isinstance(surrogate, dict) and required.issubset(surrogate.keys()):
                return  # already valid

            # We may already have a partial set – keep what we have and
            # generate only the missing organs.
            existing = {}
            if isinstance(surrogate, dict):
                existing = dict(surrogate)
            elif isinstance(surrogate, sitk.Image):
                # Likely a single bowel mask from the PELVIS path
                existing['bowel'] = surrogate

            missing = required - set(existing.keys())
            if missing:
                logger.info(
                    f"Motion surrogate for ABDOMEN is incomplete (missing {missing}). "
                    "Auto-generating missing organs using TotalSegmentator."
                )
                generator = OrganMaskGenerator(fast_mode=True, device="gpu")

                thorax_keys = {'heart', 'aorta', 'lung', 'spine'}
                pelvis_keys = {'bowel'}
                missing_thorax = missing & thorax_keys
                missing_pelvis = missing & pelvis_keys

                if missing_thorax:
                    thorax_masks, _ = generator.generate_motion_surrogate_mask(
                        self.patient.ct_image, "THORAX"
                    )
                    existing.update(thorax_masks)

                if missing_pelvis:
                    bowel_mask, _ = generator.generate_motion_surrogate_mask(
                        self.patient.ct_image, "PELVIS"
                    )
                    existing['bowel'] = bowel_mask

            self.patient.motion_surrogate = existing

    @log_time(logger)
    def create_motion_deformation(self):
        self._validate_and_generate_surrogate()

        if self.motion_config.motion_type == MotionConfig.MotionType.PELVIS:
            volume_sitk = self.patient.ct_image

            # motion_surrogate is sitk.Image for PELVIS
            mask_surrogate_sitk = sitk.Cast(self.patient.motion_surrogate, sitk.sitkUInt8)

            paramsMotionDeformationMotionField = MotionDeformationModelPelvis.ParamsMotionDeformation()
            paramsMotionDeformationMotionField.createDebugOutput = self.createDebugOutput
            self.motionDeformationModel = MotionDeformationModelPelvis(paramsMotionDeformationMotionField, self.motion_config)
            self.motionDeformationModel.computeMotionDeformation(volume_sitk, mask_surrogate_sitk, self.motion_config.amplitude_breathing)
            self.debugOutput = self.motionDeformationModel.debugOutput

        elif self.motion_config.motion_type == MotionConfig.MotionType.THORAX:
            volume_sitk = self.patient.ct_image

            # motion_surrogate is Dict[str, sitk.Image] for THORAX

            paramsMotionDeformationMotionField = MotionDeformationModelThorax.ParamsMotionDeformation()
            paramsMotionDeformationMotionField.createDebugOutput = self.createDebugOutput
            self.motionDeformationModel = MotionDeformationModelThorax(paramsMotionDeformationMotionField, self.motion_config)
            self.motionDeformationModel.computeMotionDeformation(volume_sitk, self.patient.motion_surrogate, self.motion_config.amplitude_breathing, self.motion_config.amplitude_heart)
            self.debugOutput = self.motionDeformationModel.debugOutput

        elif self.motion_config.motion_type == MotionConfig.MotionType.ABDOMEN:
            volume_sitk = self.patient.ct_image

            # motion_surrogate is Dict[str, sitk.Image] for ABDOMEN (thorax + pelvis organs)
            amplitude_heart = self.motion_config.amplitude_heart if self.motion_config.amplitude_heart is not None else 3.0

            paramsMotionDeformationMotionField = MotionDeformationModelAbdomen.ParamsMotionDeformation()
            paramsMotionDeformationMotionField.createDebugOutput = self.createDebugOutput
            self.motionDeformationModel = MotionDeformationModelAbdomen(paramsMotionDeformationMotionField, self.motion_config)
            self.motionDeformationModel.computeMotionDeformation(volume_sitk, self.patient.motion_surrogate, self.motion_config.amplitude_breathing, amplitude_heart)
            self.debugOutput = self.motionDeformationModel.debugOutput

        else:
            raise NotImplementedError('motion config type not implemented')

    def generate_dynamic_4d_CT(self, i: int) -> sitk.Image:
        """generates a SimpleITK Image of the 4d CT at a specific index (projection).
           This uses the breathing motion defined in the configs.

        Args:
            i (int): index of the projection

        Returns:
            sitk.Image: deformed CT image
        """

        t = i*self.motion_config.time_per_projection + self.rng.normal(0, self.motion_config.uncertainty,1)
        ct = self.get_4d_CT(t)
        return ct

    def get_4d_CT(self, t : float):
        self.motion_field_cp = self.motionDeformationModel.getMotionField_cp(t) #, self.motion_config.time_per_breathing_half_cycle)

        if self.instanceApplyMotionFieldWithCupy is None:
           self.instanceApplyMotionFieldWithCupy = ApplyMotionField(self.patient.ct_array) #, self.displacement_field)

        self.instanceApplyMotionFieldWithCupy.setMotionField_cp(self.motion_field_cp)

        self.instanceApplyMotionFieldWithCupy.resampleVolume(True)

        self.instanceApplyMotionFieldWithCupy.updateResampledVolumeOnHost()

        return self.resampledVolumeGPU

    def reset(self):
        self.instanceApplyMotionFieldWithCupy.free()
        self.instanceApplyMotionFieldWithCupy = None

    #add easy access
    @property
    def resampledVolumeGPU(self):
        return self.instanceApplyMotionFieldWithCupy.outputVolumeOnGPU_gpuarray
