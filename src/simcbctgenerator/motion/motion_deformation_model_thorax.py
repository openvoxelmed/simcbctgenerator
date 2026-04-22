###############################################################################
# syncbctgenerator
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

"""Motion deformation model for thorax region."""

import numpy as np
import SimpleITK as sitk
import cupy as cp
from simcbctgenerator.utils.config import MotionConfig
from simcbctgenerator.motion.binary_operations_with_cupy import BinaryOperationsWithCupy
from simcbctgenerator.motion.custom_image_processing import DistanceTransformGPU, IntegralVolumeGPU, MotionFieldPropagation
from pathlib import Path

CUDA_KERNEL_PATH = Path(__file__).parent/'cuda_kernels'

class MotionDeformationModelThorax:
    class ParamsMotionDeformation:
        minRequiredGradMagHeart = -200.0  # TODO: discuss how to come up with value here (hardcoded for now, empirically determined)
        minRequiredGradMagLung = -800.0  # TODO: discuss how to come up with value here (hardcoded for now, empirically determined)
        heartMotionBlendDistance = 5.0
        lungMotionBlendDistance = 10.0
        lungMotionGeneralTrendDirection = [0.0, -0.5, 1.0] # component values between 0 and 1
        factorIncreasingExtentOfMotionInfluence = 2.0 # needed to prevent artifacts (multiplied with amplitude to increase motion influence area slightly)
        foreground_mask_opening_kernel_radius = 3
        foreground_mask_closing_kernel_radius = 21
        noDeformationBlendDistance = 10.0  # original value 10.0 (higher values give larger transition)
        createDebugOutput: bool = False

    def __init__(self, params: ParamsMotionDeformation, motion_config: MotionConfig):
        self.params = params
        self.motion_config = motion_config
        self.displacement_field_heart = None
        self.displacement_field_heart_cp = None
        self.displacement_field_lung = None
        self.displacement_field_lung_cp = None
        self.debugOutput = None

    def getMotionField_cp(self, t: float, lung_state_min_: float = -1.0, lung_state_max_: float = 1.0, heart_state_min_: float = -1.0, heart_state_max_: float = 1.0):
        state_lung = np.sin(np.pi * (t / self.motion_config.time_per_breathing_half_cycle) + self.motion_config.phase_offset_breathing)
        state_lung_01 = (state_lung + 1) / 2
        state_lung_scale = state_lung_01 * (lung_state_max_ - lung_state_min_) + lung_state_min_
        displacement_lung = self.displacement_field_lung_cp * state_lung_scale[0]

        state_heart = np.sin(np.pi * (t / self.motion_config.time_per_heartbeat_half_cycle) + self.motion_config.phase_offset_heart)
        state_heart_01 = (state_heart + 1) / 2
        state_heart_scale = state_heart_01 * (heart_state_max_ - heart_state_min_) + heart_state_min_
        displacement_heart = self.displacement_field_heart_cp * state_heart_scale[0]

        displacement = (displacement_lung + displacement_heart) * (1- self.no_deformation_cp[:,:,:,cp.newaxis])
        return displacement

    def computeMotionDeformation(self, volume_sitk : sitk.Image, mask_dict : dict, max_amplitude_breathing : float, max_amplitude_heart : float):
        def unpadBordersOfCupyImage(img_padded: cp.array, borderSize: np.array):
            return img_padded[borderSize[0]: img_padded.shape[0] - borderSize[0],
                              borderSize[1]: img_padded.shape[1] - borderSize[1],
                              borderSize[2]: img_padded.shape[2] - borderSize[2]]

        class DebugOutput:
            pass

        debugOutput = DebugOutput()

        propagatedGradient_heart = None
        propagatedGradient_lung = None

        params = self.params
        createDebugOutput = params.createDebugOutput

        if 'heart' not in mask_dict.keys():
            raise KeyError('mask_dict has missing key "heart"')
        if 'aorta' not in mask_dict.keys():
            raise KeyError('mask_dict has missing key "aorta"')
        if 'lung' not in mask_dict.keys():
            raise KeyError('mask_dict has missing key "lung"')
        if 'spine' not in mask_dict.keys():
            raise KeyError('mask_dict has missing key "spine"')

        spacing = volume_sitk.GetSpacing()[::-1]

        mask_seg_heart_cp = cp.array(sitk.GetArrayFromImage(mask_dict['heart']))
        mask_seg_aorta_cp = cp.array(sitk.GetArrayFromImage(mask_dict['aorta']))
        mask_seg_lung_cp = cp.array(sitk.GetArrayFromImage(mask_dict['lung']))
        mask_seg_spine_cp = cp.array(sitk.GetArrayFromImage(mask_dict['spine']))

        bone_mask_cp = mask_seg_spine_cp

        mask_foreground_heart_cp = mask_seg_heart_cp
        mask_foreground_heart_cp = BinaryOperationsWithCupy.binary_opening_sphere(mask_foreground_heart_cp, spacing, params.foreground_mask_opening_kernel_radius)
        mask_foreground_heart_cp = BinaryOperationsWithCupy.binary_closing_sphere(mask_foreground_heart_cp, spacing, params.foreground_mask_closing_kernel_radius)

        distance_foreground_heart_cp = DistanceTransformGPU.signed_distance_transform_gpu(mask_foreground_heart_cp.astype(np.float32), sampling=spacing)

        padded_mask_cp = cp.pad(mask_foreground_heart_cp.astype(np.float32), ((1,1,1),(1,1,1)), mode='edge') #mode='constant', constant_values=-1000)
        gradX_cp = cp.gradient(padded_mask_cp, axis=2)
        gradY_cp = cp.gradient(padded_mask_cp, axis=1)
        gradZ_cp = cp.gradient(padded_mask_cp, axis=0)
        gradX_cp = unpadBordersOfCupyImage(gradX_cp, (1, 1, 1))
        gradY_cp = unpadBordersOfCupyImage(gradY_cp, (1, 1, 1))
        gradZ_cp = unpadBordersOfCupyImage(gradZ_cp, (1, 1, 1))
        del padded_mask_cp

        gradX_cp = cp.ascontiguousarray(gradX_cp) / spacing[2]
        gradY_cp = cp.ascontiguousarray(gradY_cp) / spacing[1]
        gradZ_cp = cp.ascontiguousarray(gradZ_cp) / spacing[0]

        # compute gradient magnitude
        gradMag_cp = cp.sqrt(gradX_cp**2 + gradY_cp**2 + gradZ_cp**2)

        # compute border of foreground mask (body surface voxels)
        border_mask_heart_cp = cp.bitwise_xor(mask_foreground_heart_cp, BinaryOperationsWithCupy.binary_erosion(mask_foreground_heart_cp, (1,1,1)))

        # determine mask of border voxels with reasonable gradient vector magnitude
        mask_border_voxels_with_valid_gradient_heart_cp = cp.logical_and(gradMag_cp > params.minRequiredGradMagHeart, border_mask_heart_cp)

        distance_next_border_voxel_with_gradient_heart_cp = DistanceTransformGPU.signed_distance_transform_gpu(
            mask_border_voxels_with_valid_gradient_heart_cp.astype(np.float32), sampling=(1., 1., 1.))

        gradX_heart_cp = gradX_cp.copy()
        gradY_heart_cp = gradY_cp.copy()
        gradZ_heart_cp = gradZ_cp.copy()

        # obtain gradient vectors component images of strong gradient vectors in foreground mask border positions
        inverted_mask_cp = cp.logical_not(mask_border_voxels_with_valid_gradient_heart_cp)
        cp.putmask(gradX_heart_cp, inverted_mask_cp, cp.zeros(gradX_heart_cp.shape).astype(bool))
        cp.putmask(gradY_heart_cp, inverted_mask_cp, cp.zeros(gradY_heart_cp.shape).astype(bool))
        cp.putmask(gradZ_heart_cp, inverted_mask_cp, cp.zeros(gradZ_heart_cp.shape).astype(bool))

        del inverted_mask_cp

        # normalize remaining gradient vectors
        border_gradient_mag_heart_cp = cp.sqrt(gradX_heart_cp ** 2 + gradY_heart_cp ** 2 + gradZ_heart_cp ** 2) + 0.00001
        gradX_heart_cp /= border_gradient_mag_heart_cp
        gradY_heart_cp /= border_gradient_mag_heart_cp
        gradZ_heart_cp /= border_gradient_mag_heart_cp

        if createDebugOutput:
            xcomp = cp.asnumpy(gradX_heart_cp)
            ycomp = cp.asnumpy(gradY_heart_cp)
            zcomp = cp.asnumpy(gradZ_heart_cp)
            border_gradient_heart_np = np.stack((xcomp, ycomp, zcomp), axis=3)

        if not createDebugOutput:
            del border_gradient_mag_heart_cp

        distance_for_cuboid_extent_heart_cp = cp.abs(distance_next_border_voxel_with_gradient_heart_cp)

        intVolGradX_cp = IntegralVolumeGPU.computeIntegralVolume(gradX_heart_cp)
        intVolGradY_cp = IntegralVolumeGPU.computeIntegralVolume(gradY_heart_cp)
        intVolGradZ_cp = IntegralVolumeGPU.computeIntegralVolume(gradZ_heart_cp)
        intVolValidGradOcc_cp = IntegralVolumeGPU.computeIntegralVolume(mask_border_voxels_with_valid_gradient_heart_cp.astype(cp.float32))

        spacing2 = (1.,1.,1.)
        MotionFieldPropagation.propagateMotionField(intVolGradX_cp, intVolGradY_cp, intVolGradZ_cp, intVolValidGradOcc_cp, distance_for_cuboid_extent_heart_cp, gradX_heart_cp, gradY_heart_cp, gradZ_heart_cp, spacing2)

        mask = cp.where(    distance_next_border_voxel_with_gradient_heart_cp > -max_amplitude_heart,
                             1.0,
                             cp.clip(1 + (distance_next_border_voxel_with_gradient_heart_cp+max_amplitude_heart) / params.heartMotionBlendDistance, 0, 1))

        propagatedGradient_heart_cp = cp.stack((gradX_heart_cp, gradY_heart_cp, gradZ_heart_cp),axis=3)

        propagatedGradient_heart_cp *= mask[:, :, :, cp.newaxis]

        propagatedGradient_heart = cp.asnumpy(propagatedGradient_heart_cp)



        mask_foreground_lung_cp = mask_seg_lung_cp
        mask_foreground_lung_cp = BinaryOperationsWithCupy.binary_opening_sphere(mask_foreground_lung_cp, spacing, params.foreground_mask_opening_kernel_radius)
        mask_foreground_lung_cp = BinaryOperationsWithCupy.binary_closing_sphere(mask_foreground_lung_cp, spacing, params.foreground_mask_closing_kernel_radius)

        padded_mask_cp = cp.pad(mask_foreground_lung_cp.astype(np.float32), ((1,1,1),(1,1,1)), mode='edge') #mode='constant', constant_values=-1000)
        gradX_cp = cp.gradient(padded_mask_cp, axis=2)
        gradY_cp = cp.gradient(padded_mask_cp, axis=1)
        gradZ_cp = cp.gradient(padded_mask_cp, axis=0)
        gradX_cp = unpadBordersOfCupyImage(gradX_cp, (1, 1, 1))
        gradY_cp = unpadBordersOfCupyImage(gradY_cp, (1, 1, 1))
        gradZ_cp = unpadBordersOfCupyImage(gradZ_cp, (1, 1, 1))
        del padded_mask_cp

        gradX_cp = cp.ascontiguousarray(gradX_cp) / spacing[2]
        gradY_cp = cp.ascontiguousarray(gradY_cp) / spacing[1]
        gradZ_cp = cp.ascontiguousarray(gradZ_cp) / spacing[0]

        # compute gradient magnitude
        gradMag_cp = cp.sqrt(gradX_cp**2 + gradY_cp**2 + gradZ_cp**2)

        # compute border of foreground mask
        border_mask_lung_cp = cp.bitwise_xor(mask_foreground_lung_cp, BinaryOperationsWithCupy.binary_erosion(mask_foreground_lung_cp, (1,1,1)))

        gradX_lung_cp = gradX_cp.copy()
        gradY_lung_cp = gradY_cp.copy()
        gradZ_lung_cp = gradZ_cp.copy()

        # determine mask of border voxels with reasonable gradient vector magnitude
        mask_border_voxels_with_valid_gradient_lung_cp = cp.logical_and(gradMag_cp > params.minRequiredGradMagLung, border_mask_lung_cp)

        distance_next_border_voxel_with_gradient_lung_cp = DistanceTransformGPU.signed_distance_transform_gpu(
            mask_border_voxels_with_valid_gradient_lung_cp.astype(np.float32), sampling=(1., 1., 1.))

        # obtain gradient vectors component images of strong gradient vectors in foreground mask border positions
        inverted_mask_cp = cp.logical_not(mask_border_voxels_with_valid_gradient_lung_cp)
        cp.putmask(gradX_lung_cp, inverted_mask_cp, cp.zeros(gradX_lung_cp.shape).astype(bool))
        cp.putmask(gradY_lung_cp, inverted_mask_cp, cp.zeros(gradY_lung_cp.shape).astype(bool))
        cp.putmask(gradZ_lung_cp, inverted_mask_cp, cp.zeros(gradZ_lung_cp.shape).astype(bool))

        del inverted_mask_cp

        # normalize remaining gradient vectors
        border_gradient_mag_lung_cp = cp.sqrt(gradX_lung_cp ** 2 + gradY_lung_cp ** 2 + gradZ_lung_cp ** 2) + 0.00001
        gradX_lung_cp /= border_gradient_mag_lung_cp
        gradY_lung_cp /= border_gradient_mag_lung_cp
        gradZ_lung_cp /= border_gradient_mag_lung_cp

        if createDebugOutput:
            xcomp = cp.asnumpy(gradX_lung_cp)
            ycomp = cp.asnumpy(gradY_lung_cp)
            zcomp = cp.asnumpy(gradZ_lung_cp)
            border_gradient_lung_np = np.stack((xcomp, ycomp, zcomp), axis=3)

        if not createDebugOutput:
            del border_gradient_mag_lung_cp

        distance_for_cuboid_extent_lung_cp = cp.abs(distance_next_border_voxel_with_gradient_lung_cp)

        intVolGradX_cp = IntegralVolumeGPU.computeIntegralVolume(gradX_lung_cp)
        intVolGradY_cp = IntegralVolumeGPU.computeIntegralVolume(gradY_lung_cp)
        intVolGradZ_cp = IntegralVolumeGPU.computeIntegralVolume(gradZ_lung_cp)
        intVolValidGradOcc_cp = IntegralVolumeGPU.computeIntegralVolume(mask_border_voxels_with_valid_gradient_lung_cp.astype(cp.float32))

        spacing2 = (1.,1.,1.)
        MotionFieldPropagation.propagateMotionField(intVolGradX_cp, intVolGradY_cp, intVolGradZ_cp, intVolValidGradOcc_cp, distance_for_cuboid_extent_lung_cp, gradX_lung_cp, gradY_lung_cp, gradZ_lung_cp, spacing2)

        mask = cp.where(    distance_next_border_voxel_with_gradient_lung_cp > -max_amplitude_breathing,
                             1.0,
                             cp.clip(1 + (distance_next_border_voxel_with_gradient_lung_cp+max_amplitude_breathing) / params.lungMotionBlendDistance, 0, 1))

        gradX_lung_cp += params.lungMotionGeneralTrendDirection[0]
        gradY_lung_cp += params.lungMotionGeneralTrendDirection[1]
        gradZ_lung_cp += params.lungMotionGeneralTrendDirection[2]

        propagatedGradient_lung_cp = cp.stack((gradX_lung_cp, gradY_lung_cp, gradZ_lung_cp),axis=3)

        propagatedGradient_lung_cp *= mask[:,:,:,cp.newaxis]

        propagatedGradient_lung = cp.asnumpy(propagatedGradient_lung_cp)

        if propagatedGradient_heart_cp is not None:
            displacement_field_heart_cp = propagatedGradient_heart_cp
            displacement_field_heart = cp.asnumpy(displacement_field_heart_cp)
        else:
            displacement_field_heart = None

        if propagatedGradient_lung_cp is not None:
            displacement_field_lung_cp = propagatedGradient_lung_cp
            displacement_field_lung = cp.asnumpy(displacement_field_lung_cp)
        else:
            displacement_field_lung = None

        mask_no_deformation_cp = bone_mask_cp
        distance_no_deformation_cp = DistanceTransformGPU.signed_distance_transform_gpu(mask_no_deformation_cp.astype(np.float32), spacing)

        no_deformation_cp = cp.where(distance_no_deformation_cp > 0,
                                  1.0,
                                  cp.clip(1 + distance_no_deformation_cp / params.noDeformationBlendDistance, 0, 1))

        if createDebugOutput:
            debugOutput.mask_seg_heart = cp.asnumpy(mask_seg_heart_cp)
            debugOutput.mask_seg_aorta = cp.asnumpy(mask_seg_aorta_cp)
            debugOutput.mask_seg_lung = cp.asnumpy(mask_seg_lung_cp)
            debugOutput.mask_seg_spine = cp.asnumpy(mask_seg_spine_cp)
            debugOutput.distance_foreground_heart = cp.asnumpy(distance_foreground_heart_cp)
            debugOutput.bone_mask = cp.asnumpy(bone_mask_cp).astype(np.uint8)
            debugOutput.no_deformation = cp.asnumpy(no_deformation_cp)
            debugOutput.mask_foreground_heart = cp.asnumpy(mask_foreground_heart_cp).astype(np.uint8)
            debugOutput.border_mask_heart = cp.asnumpy(border_mask_heart_cp).astype(np.uint8)
            debugOutput.mask_foreground_lung = cp.asnumpy(mask_foreground_lung_cp).astype(np.uint8)
            debugOutput.border_mask_lung = cp.asnumpy(border_mask_lung_cp).astype(np.uint8)
            debugOutput.mask_border_voxels_with_valid_gradient_heart = cp.asnumpy(mask_border_voxels_with_valid_gradient_heart_cp).astype(np.uint8)
            debugOutput.distance_next_border_voxel_with_gradient_heart = cp.asnumpy(distance_next_border_voxel_with_gradient_heart_cp)
            debugOutput.mask_border_voxels_with_valid_gradient_lung = cp.asnumpy(mask_border_voxels_with_valid_gradient_lung_cp).astype(np.uint8)
            debugOutput.distance_next_border_voxel_with_gradient_lung = cp.asnumpy(distance_next_border_voxel_with_gradient_lung_cp)

            if propagatedGradient_heart is not None:
                debugOutput.border_gradient_heart = border_gradient_heart_np
                debugOutput.propagated_gradient_heart = propagatedGradient_heart
            else:
                debugOutput.border_gradient_heart = None
                debugOutput.propagated_gradient_heart = None

            if propagatedGradient_lung is not None:
                debugOutput.border_gradient_lung = border_gradient_lung_np
                debugOutput.propagated_gradient_lung = propagatedGradient_lung
            else:
                debugOutput.propagated_gradient_lung = None
                debugOutput.propagatedGradient_lung = None

        self.debugOutput = debugOutput
        self.displacement_field_heart = displacement_field_heart
        self.displacement_field_heart_cp = displacement_field_heart_cp
        self.displacement_field_lung = displacement_field_lung
        self.displacement_field_lung_cp = displacement_field_lung_cp
        self.no_deformation_cp = no_deformation_cp
