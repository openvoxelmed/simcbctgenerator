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

"""Motion deformation model for pelvis region."""

import numpy as np
import SimpleITK as sitk
import cupy as cp
from simcbctgenerator.utils.config import MotionConfig
from simcbctgenerator.motion.binary_operations_with_cupy import BinaryOperationsWithCupy
from simcbctgenerator.motion.custom_image_processing import DistanceTransformGPU, IntegralVolumeGPU, MotionFieldPropagation
from pathlib import Path

CUDA_KERNEL_PATH = Path(__file__).parent.parent/'cuda_kernels'

class MotionDeformationModelPelvis:
    class ParamsMotionDeformation:
        minRequiredGradMag = 200.0  # TODO: discuss how to come up with value here (hardcoded for now, empirically determined)
        factorIncreasingExtentOfMotionInfluence = 8.0 # needed to prevent artifacts (multiplied with amplitude to increase motion influence area slightly)
        foreground_mask_min_hu_threshold = -300
        foreground_mask_opening_kernel_radius = 3
        foreground_mask_closing_kernel_radius = 11
        bone_threshold = 200.0
        minBoneDistanceFromBorder = 3.0
        extraPixelsForExtendedBorder = 30.0
        doRefinementMaskSurrogateExtended = False
        extraPixelToleranceBoneMask = -6.0
        noDeformationBlendDistance = 30.0 # original value 10.0 (higher values give larger transition)
        distanceThresholdForVoxelsDistantToSurrogate = -40.0
        maxDistExtendedSurrogate = 10.0
        mainMotionDirection = [0.,1.,0.] # main motion direction (x,y,z)
        attenuationWithDivergingVectorDirection = 1.0 # will be used as exponent in reduction
        vectorBoostFactor = 1.1 # scale dot product of vectors to prevent shortening of (well aligned) vectors
        createDebugOutput: bool = False

    def __init__(self, params: ParamsMotionDeformation, motion_config : MotionConfig):
        self.params = params
        self.motion_config = motion_config
        self.displacement_field = None
        self.displacement_field_cp = None
        self.debugOutput = None

    def getMotionField_cp(self, t: float, min_: float = -1.0, max_: float = 1.0):
        state = np.sin(np.pi * (t / self.motion_config.time_per_breathing_half_cycle) + self.motion_config.phase_offset_breathing)
        state_01 = (state + 1) / 2
        state_scale = state_01 * (max_ - min_) + min_

        return self.displacement_field_cp * state_scale[0]

    def computeMotionDeformation(self, volume_sitk : sitk.Image, mask_surrogate_sitk : sitk.Image, max_amplitude : float):
        """
        Simulate breathing motion within a mask with smooth falloff
        """

        def unpadBordersOfCupyImage(img_padded: cp.array, borderSize: np.array):
            return img_padded[borderSize[0]: img_padded.shape[0] - borderSize[0],
                              borderSize[1]: img_padded.shape[1] - borderSize[1],
                              borderSize[2]: img_padded.shape[2] - borderSize[2]]

        class DebugOutput:
            pass

        debugOutput = DebugOutput()

        propagatedGradient = None

        params = self.params
        createDebugOutput = params.createDebugOutput

        spacing = volume_sitk.GetSpacing()[::-1]
        # Convert images to numpy for easier manipulation
        volume = sitk.GetArrayFromImage(volume_sitk)
        mask_surrogate_np = sitk.GetArrayFromImage(mask_surrogate_sitk)

        volume_cp = cp.array(volume).astype(np.float32)
        mask_surrogate_cp = cp.array(mask_surrogate_np)

        # Create distance map from mask boundaries
        distance_surrogate_cp = DistanceTransformGPU.signed_distance_transform_gpu(mask_surrogate_cp, spacing)

        # compute foreground mask (TODO: reevaluate lowerThreshold, and kernel sizes for morphological operations - empirically determined for now)
        mask_foreground_cp = cp.logical_and(volume_cp >= params.foreground_mask_min_hu_threshold, volume_cp <= 10000)

        mask_foreground_cp = BinaryOperationsWithCupy.binary_opening_sphere(mask_foreground_cp, spacing, params.foreground_mask_opening_kernel_radius)
        mask_foreground_cp = cp.bitwise_or(mask_surrogate_cp, mask_foreground_cp)
        mask_foreground_cp = BinaryOperationsWithCupy.binary_closing_sphere(mask_foreground_cp, spacing, params.foreground_mask_closing_kernel_radius)

        # compute signed distance map of foreground mask
        distance_foreground_cp = DistanceTransformGPU.signed_distance_transform_gpu(mask_foreground_cp.astype(np.float32), sampling=spacing)

        # compute border of foreground mask (body surface voxels)
        border_mask_cp = cp.bitwise_xor(mask_foreground_cp, BinaryOperationsWithCupy.binary_erosion(mask_foreground_cp, (1,1,1)))

        # compute signed distance map of border mask
        distance_border_cp = DistanceTransformGPU.signed_distance_transform_gpu(border_mask_cp.astype(np.float32), sampling=spacing)

        # extract voxel intensities for foreground voxels (throw away intensities outside body)
        foreground_values_cp = volume_cp.copy()
        foreground_values_cp[mask_foreground_cp == 0] = -1000

        padded_foreground_values_cp = cp.pad(foreground_values_cp, ((1,1,1),(1,1,1)), mode='edge') #mode='constant', constant_values=-1000)
        gradX_cp = cp.gradient(padded_foreground_values_cp, axis=2)
        gradY_cp = cp.gradient(padded_foreground_values_cp, axis=1)
        gradZ_cp = cp.gradient(padded_foreground_values_cp, axis=0)
        gradX_cp = unpadBordersOfCupyImage(gradX_cp, (1, 1, 1))
        gradY_cp = unpadBordersOfCupyImage(gradY_cp, (1, 1, 1))
        gradZ_cp = unpadBordersOfCupyImage(gradZ_cp, (1, 1, 1))
        del padded_foreground_values_cp

        gradX_cp = cp.ascontiguousarray(gradX_cp) / spacing[2]
        gradY_cp = cp.ascontiguousarray(gradY_cp) / spacing[1]
        gradZ_cp = cp.ascontiguousarray(gradZ_cp) / spacing[0]

        if not createDebugOutput:
            del foreground_values_cp

        # compute gradient magnitude
        gradMag_cp = cp.sqrt(gradX_cp**2 + gradY_cp**2 + gradZ_cp**2)

        # determine mask of border voxels with reasonable gradient vector magnitude
        mask_border_voxels_with_valid_gradient_cp = cp.logical_and(gradMag_cp > params.minRequiredGradMag, border_mask_cp)

        if not createDebugOutput:
            del border_mask_cp

        if not createDebugOutput:
            del gradMag_cp

        distance_next_border_voxel_with_gradient_cp = DistanceTransformGPU.signed_distance_transform_gpu(mask_border_voxels_with_valid_gradient_cp.astype(np.float32), sampling=(1.,1.,1.))

        # obtain gradient vectors component images of strong gradient vectors in foreground mask border positions
        inverted_mask_cp = cp.logical_not(mask_border_voxels_with_valid_gradient_cp)
        cp.putmask(gradX_cp, inverted_mask_cp, cp.zeros(gradX_cp.shape).astype(bool))
        cp.putmask(gradY_cp, inverted_mask_cp, cp.zeros(gradY_cp.shape).astype(bool))
        cp.putmask(gradZ_cp, inverted_mask_cp, cp.zeros(gradZ_cp.shape).astype(bool))

        del inverted_mask_cp

        # normalize remaining gradient vectors
        border_gradient_mag_cp = cp.sqrt(gradX_cp ** 2 + gradY_cp ** 2 + gradZ_cp ** 2) + 0.00001
        gradX_cp /= border_gradient_mag_cp
        gradY_cp /= border_gradient_mag_cp
        gradZ_cp /= border_gradient_mag_cp

        if createDebugOutput:
            xcomp = cp.asnumpy(gradX_cp)
            ycomp = cp.asnumpy(gradY_cp)
            zcomp = cp.asnumpy(gradZ_cp)
            border_gradient_np = np.stack((xcomp, ycomp, zcomp), axis=3)

        if not createDebugOutput:
            del border_gradient_mag_cp

        distance_for_cuboid_extent_cp = cp.abs(distance_next_border_voxel_with_gradient_cp)

        intVolGradX_cp = IntegralVolumeGPU.computeIntegralVolume(gradX_cp)
        intVolGradY_cp = IntegralVolumeGPU.computeIntegralVolume(gradY_cp)
        intVolGradZ_cp = IntegralVolumeGPU.computeIntegralVolume(gradZ_cp)
        intVolValidGradOcc_cp = IntegralVolumeGPU.computeIntegralVolume(mask_border_voxels_with_valid_gradient_cp.astype(cp.float32))

        spacing2 = (1.,1.,1.)
        MotionFieldPropagation.propagateMotionField(intVolGradX_cp, intVolGradY_cp, intVolGradZ_cp, intVolValidGradOcc_cp, distance_for_cuboid_extent_cp, gradX_cp, gradY_cp, gradZ_cp, spacing2)

        propagatedGradient_cp = cp.stack((gradX_cp, gradY_cp, gradZ_cp),axis=3)

        gradXnorm_cp = gradX_cp * cp.linalg.norm(propagatedGradient_cp, axis=3)
        gradYnorm_cp = gradY_cp * cp.linalg.norm(propagatedGradient_cp, axis=3)
        gradZnorm_cp = gradZ_cp * cp.linalg.norm(propagatedGradient_cp, axis=3)
        propagatedGradientNormalized = cp.stack((gradXnorm_cp, gradYnorm_cp, gradZnorm_cp),axis=3)
        dotProduct = cp.dot(propagatedGradientNormalized, cp.array(params.mainMotionDirection))
        dotProduct[dotProduct <= 0] = 0
        dotProduct = cp.clip(params.vectorBoostFactor * dotProduct, 0, 1)
        propagatedGradient_cp[:, :, :, 0] *= dotProduct ** params.attenuationWithDivergingVectorDirection
        propagatedGradient_cp[:, :, :, 1] *= dotProduct ** params.attenuationWithDivergingVectorDirection
        propagatedGradient_cp[:, :, :, 2] *= dotProduct ** params.attenuationWithDivergingVectorDirection

        propagatedGradient = cp.asnumpy(propagatedGradient_cp)

        del gradX_cp
        del gradY_cp
        del gradZ_cp

        bone_mask_cp = cp.logical_and(volume_cp >= params.bone_threshold, volume_cp <= 10000)

        # get rid of bone parts in bone mask at a small region inside border and everywhere outside
        cp.putmask(bone_mask_cp, distance_foreground_cp < params.minBoneDistanceFromBorder, cp.zeros(bone_mask_cp.shape).astype(bool))

        distance_bone_cp = DistanceTransformGPU.signed_distance_transform_gpu(bone_mask_cp.astype(np.float32), spacing)

        bone_mask_refined_cp = distance_bone_cp >= params.extraPixelToleranceBoneMask

        if not createDebugOutput:
            del bone_mask_cp

        # compute voxels proximate to surrogate
        mask_distant_to_surrogate_cp = cp.logical_and(mask_foreground_cp, distance_surrogate_cp < params.distanceThresholdForVoxelsDistantToSurrogate)


        # compute distance to voxels from class "distant to surrogate"
        distance_distant_to_surrogate_cp = DistanceTransformGPU.signed_distance_transform_gpu(mask_distant_to_surrogate_cp, spacing)
        if not createDebugOutput:
            del mask_distant_to_surrogate_cp

        if not createDebugOutput:
            del mask_foreground_cp

        # extend the surrogate mask - derive the extension size from max_amplitude
        mask_surrogate_extended_cp = distance_surrogate_cp >= -params.factorIncreasingExtentOfMotionInfluence * max_amplitude

        # create mask with slightly extended border to inside and outside of patient
        mask_border_extended_cp = cp.abs(distance_border_cp) < params.extraPixelsForExtendedBorder # whole outside + small region near to border inside

        if not createDebugOutput:
            del distance_border_cp

        # create extended surrogate mask by taking intersection between extended mask and extended border + surrogate mask
        # i.g. this result mask will be smaller than previous mask_surrogate_extended
        if params.doRefinementMaskSurrogateExtended:
            mask_surrogate_extended_cp = cp.logical_or(cp.logical_and(mask_surrogate_extended_cp, mask_border_extended_cp), mask_surrogate_cp)

        # finally compute distance map on extended surrogate mask
        distance_surrogate_extended_cp = DistanceTransformGPU.signed_distance_transform_gpu(mask_surrogate_extended_cp.astype(np.float32), spacing)

        if not createDebugOutput:
            del mask_surrogate_extended_cp

        mask_no_deformation_cp = bone_mask_refined_cp.copy()

        if not createDebugOutput:
            del bone_mask_refined_cp

        mask_no_deformation_cp = cp.logical_or(mask_no_deformation_cp, distance_distant_to_surrogate_cp >= params.maxDistExtendedSurrogate)

        distance_no_deformation_cp = DistanceTransformGPU.signed_distance_transform_gpu(mask_no_deformation_cp.astype(np.float32), spacing)

        if not createDebugOutput:
            del mask_no_deformation_cp

        weights_cp = cp.where(
            distance_surrogate_extended_cp >= 0,
            1.0,
            cp.clip((1 + distance_surrogate_extended_cp / max_amplitude), 0, 1)
        )

        deformation_cp = max_amplitude * weights_cp

        no_deformation_cp = cp.where(distance_no_deformation_cp > 0,
                                  1.0,
                                  cp.clip(1 + distance_no_deformation_cp / params.noDeformationBlendDistance, 0, 1))

        if propagatedGradient_cp is not None:
            displacement_field_cp = propagatedGradient_cp * (deformation_cp * (1 - no_deformation_cp))[..., cp.newaxis]
            displacement_field = cp.asnumpy(displacement_field_cp)
        else:
            displacement_field = None
            displacement_field_cp = None

        if createDebugOutput:
            debugOutput.mask_surrogate = sitk.GetArrayFromImage(mask_surrogate_sitk)
            debugOutput.distance_surrogate = cp.asnumpy(distance_surrogate_cp)
            debugOutput.bone_mask = cp.asnumpy(bone_mask_cp).astype(np.uint8)
            debugOutput.distance_bone = cp.asnumpy(distance_bone_cp)
            debugOutput.bone_mask_refined = cp.asnumpy(bone_mask_refined_cp).astype(np.uint8)
            debugOutput.mask_no_deformation = cp.asnumpy(mask_no_deformation_cp).astype(np.uint8)
            debugOutput.distance_no_deformation = cp.asnumpy(distance_no_deformation_cp)
            debugOutput.no_deformation = cp.asnumpy(no_deformation_cp)
            debugOutput.weights = cp.asnumpy(weights_cp)
            debugOutput.deformation = cp.asnumpy(deformation_cp)
            debugOutput.deformation_amount = (deformation_cp * (1 - no_deformation_cp))[..., cp.newaxis]
            debugOutput.mask_foreground = cp.asnumpy(mask_foreground_cp).astype(np.uint8)
            debugOutput.border_mask = cp.asnumpy(border_mask_cp).astype(np.uint8)
            debugOutput.distance_foreground = cp.asnumpy(distance_foreground_cp)
            debugOutput.foreground_values = cp.asnumpy(foreground_values_cp)
            debugOutput.mask_border_voxels_with_valid_gradient = cp.asnumpy(mask_border_voxels_with_valid_gradient_cp).astype(np.uint8)
            debugOutput.mask_distant_to_surrogate = cp.asnumpy(mask_distant_to_surrogate_cp).astype(np.uint8)
            debugOutput.mask_border_extended = cp.asnumpy(mask_border_extended_cp).astype(np.uint8)
            if propagatedGradient is not None:
                debugOutput.border_gradient = border_gradient_np
                debugOutput.propagated_gradient_without_attenuation = cp.asnumpy(propagatedGradientNormalized)
                debugOutput.propagated_gradient = propagatedGradient
            else:
                debugOutput.border_gradient = None
                debugOutput.propagated_gradient = None
            debugOutput.mask_surrogate_extended = cp.asnumpy(mask_surrogate_extended_cp).astype(np.uint8)
            debugOutput.distance_surrogate_extended = cp.asnumpy(distance_surrogate_extended_cp)

        self.debugOutput = debugOutput
        self.displacement_field = displacement_field
        self.displacement_field_cp = displacement_field_cp
