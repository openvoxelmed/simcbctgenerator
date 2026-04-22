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

from simcbctgenerator.generate_4d_ct import FourDCTGenerator, MotionConfig
from simcbctgenerator.patient import Patient, PatientConfig
from pathlib import Path
import SimpleITK as sitk
import numpy as np
from PyQt5.QtCore import QLibraryInfo

if __name__ == "__main__":
    import sys
    if sys.platform != 'win32':
        import os
        os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = QLibraryInfo.location(
            QLibraryInfo.PluginsPath
        )
    import matplotlib.pyplot as plt
    Dicom_path = Path(r'')

    motion_config = MotionConfig(
        motion_type = MotionConfig.MotionType.PELVIS,
        amplitude_breathing= 10,
        amplitude_heart= 0,
        # frequncy --> breathing
        contour_name= 'bowel',
        contour_names= [],
        frequency_breathing= 20, # breaths per minute (12-20)
        frequency_heartbeat= 0,
        time_per_projection= 0.18, # seconds
        uncertainty= 0.02, # seconds
        )
    patient_config = PatientConfig(
        plan_dir= 'DICOM_PLAN',
        ct_dir= 'CT_SET',
        cbct_dir= 'CT_SET',
        export_structures= ['bowel'],
        priority=[1],
        cm_mask='bowel',
        image_modality='dummy')
    patient = Patient(patient_config, path=Path('dummy_123'))
    patient.correct_CM()

    generator = FourDCTGenerator(motion_config, True)
    generator.initialize(patient)
    cts = generator.generate_dynamic_4d_CT(0)

    mask_surrogate = generator.debugOutput.mask_surrogate
    distance_surrogate = generator.debugOutput.distance_surrogate
    bone_mask = generator.debugOutput.bone_mask
    distance_bone = generator.debugOutput.distance_bone
    bone_mask_refined = generator.debugOutput.bone_mask_refined
    mask_no_deformation = generator.debugOutput.mask_no_deformation
    distance_no_deformation = generator.debugOutput.distance_no_deformation
    weights = generator.debugOutput.weights
    deformation = generator.debugOutput.deformation
    deformation_amount = generator.debugOutput.deformation_amount
    no_deformation = generator.debugOutput.no_deformation
    mask_foreground = generator.debugOutput.mask_foreground
    border_mask = generator.debugOutput.border_mask
    distance_foreground = generator.debugOutput.distance_foreground
    foreground_values = generator.debugOutput.foreground_values
    border_gradient = generator.debugOutput.border_gradient
    propagated_gradient = generator.debugOutput.propagated_gradient
    propagated_gradient_without_attenuation = generator.debugOutput.propagated_gradient_without_attenuation
    mask_distant_to_surrogate = generator.debugOutput.mask_distant_to_surrogate
    mask_surrogate_extended = generator.debugOutput.mask_surrogate_extended
    distance_surrogate_extended = generator.debugOutput.distance_surrogate_extended
    mask_border_voxels_with_valid_gradient = generator.debugOutput.mask_border_voxels_with_valid_gradient
    mask_border_extended = generator.debugOutput.mask_border_extended

    idxSlice = 59#int(generator.debugOutput.distance_surrogate.shape[0] // 2) #+ 27 # idxSlice = 74

    img = patient.ct_array[idxSlice,:,:]

    if generator.motionDeformationModel.displacement_field is not None:
        displacement = sitk.GetImageFromArray(generator.motionDeformationModel.displacement_field * 1)
        displacement.SetSpacing(patient.ct_image.GetSpacing())
        motion_field_slice = sitk.GetArrayFromImage(displacement)[idxSlice, :, :, :]

        gradX = motion_field_slice[:,:,0]
        gradY = motion_field_slice[:,:,1]
        mag = np.sqrt(gradX**2 + gradY**2)

    mask_surrogate = mask_surrogate[idxSlice,:,:]
    mask_surrogate_extended = mask_surrogate_extended[idxSlice,:,:]
    bone_mask = bone_mask[idxSlice,:,:]
    distance_bone = distance_bone[idxSlice,:,:]
    bone_mask_refined = bone_mask_refined[idxSlice,:,:]
    distance_no_deformation = distance_no_deformation[idxSlice,:,:]
    mask_foreground = mask_foreground[idxSlice,:,:]
    border_mask = border_mask[idxSlice,:,:]
    distance_foreground = distance_foreground[idxSlice,:,:]
    foreground_values = foreground_values[idxSlice,:,:]
    mask_border_voxels_with_valid_gradient = mask_border_voxels_with_valid_gradient[idxSlice,:,:]
    mask_distant_to_surrogate = mask_distant_to_surrogate[idxSlice,:,:]
    mask_border_extended = mask_border_extended[idxSlice,:,:]
    mask_no_deformation = mask_no_deformation[idxSlice,:,:]
    no_deformation = no_deformation[idxSlice,:,:]
    weights = weights[idxSlice,:,:]
    deformation = deformation[idxSlice,:,:]
    deformation_amount = deformation_amount[idxSlice,:,:]

    def drawDistanceMap(dist):
        plt.imshow(dist, cmap='PuOr', vmin=-dist.max(), vmax=dist.max())

    # plt.figure(figsize=(18,5))
    # plt.subplot(2,3,1)
    # plt.title('slice image data')
    # plt.imshow(img, cmap='gray')
    # plt.subplot(2,3,2)
    # plt.title('slice foreground')
    # plt.imshow(mask_foreground, cmap='gray')
    # plt.subplot(2,3,3)
    # plt.title('slice foreground distance')
    # drawDistanceMap(distance_foreground)
    # plt.subplot(2,3,4)
    # plt.title('bone mask')
    # plt.imshow(bone_mask, cmap='gray')
    # plt.subplot(2,3,5)
    # plt.title('bowel surrogate mask')
    # plt.imshow(mask_surrogate, cmap='gray')
    # plt.subplot(2,3,6)
    # plt.title('bowel surrogate mask distance')
    # drawDistanceMap(distance_surrogate[idxSlice, :, :])

    # plt.figure(figsize=(18,5))
    # plt.subplot(2,2,1)
    # plt.title('refined bone mask')
    # plt.imshow(bone_mask_refined, cmap='gray')
    # plt.subplot(2,2,2)
    # plt.title('bone mask distance')
    # drawDistanceMap(distance_bone)
    # # plt.subplot(2,2,3)
    # # plt.title('bone mask max distance')
    # # drawDistanceMap(max_distance_bone)
    # plt.subplot(2,2,3)
    # plt.title('no deformation mask')
    # plt.imshow(no_deformation[idxSlice,:,:], cmap='gray')

    # if generator.displacement_field is not None:
    #     flowVectorScale = 1.0
    #     motionVectorFieldX = propagated_gradient[idxSlice,:,:,0] * flowVectorScale
    #     motionVectorFieldY = propagated_gradient[idxSlice,:,:,1] * flowVectorScale
    #
    #     def visualizeVectorField(compX: np.ndarray, compY: np.ndarray, step = 1, scaleVectors = 1.0):
    #         [X_all, Y_all] = np.meshgrid(np.arange(0, compX.shape[1]),
    #                                      np.arange(0, compX.shape[0]))
    #         plt.quiver(X_all[::step, ::step], Y_all[::step, ::step], compX[::step, ::step], compY[::step, ::step], \
    #                    angles='xy', scale_units='xy', scale=1.0/scaleVectors, color='y', alpha=0.5, width=0.002, minshaft=0.01, minlength=0)
    #
    #     plt.figure(figsize=(8,8))
    #     plt.imshow(foreground_values, cmap='gray')
    #     visualizeVectorField(motionVectorFieldX, motionVectorFieldY, 5, 10.0)
    #
    #     plt.figure(figsize=(8,8))
    #     plt.imshow(foreground_values, cmap='gray')
    #     visualizeVectorField(gradX, gradY, 5)

    xrange = range(0,propagated_gradient.shape[2])
    yrange = range(0,propagated_gradient.shape[1])
    stepSize = 4
    indices = np.ix_(yrange, xrange)

    if generator.motionDeformationModel.displacement_field is not None:
        flowVectorScale = 1.0
        motionVectorFieldX = propagated_gradient[idxSlice,:,:,0] * flowVectorScale
        motionVectorFieldY = propagated_gradient[idxSlice,:,:,1] * flowVectorScale

        border_gradient_x = border_gradient[idxSlice, :, :, 0] * flowVectorScale
        border_gradient_y = border_gradient[idxSlice, :, :, 1] * flowVectorScale

        propagated_gradient_without_attenuation_x = propagated_gradient_without_attenuation[idxSlice, :, :, 0] * flowVectorScale
        propagated_gradient_without_attenuation_y = propagated_gradient_without_attenuation[idxSlice, :, :, 1] * flowVectorScale

        def visualizeVectorField(compX: np.ndarray, compY: np.ndarray, step = 1, scaleVectors = 1.0):
            [X_all, Y_all] = np.meshgrid(np.arange(0, compX.shape[1]),
                                        np.arange(0, compX.shape[0]))
            magnitude = np.sqrt(compX[::step, ::step]**2 + compY[::step, ::step]**2)
            plt.quiver(X_all[::step, ::step], Y_all[::step, ::step], compX[::step, ::step], compY[::step, ::step], magnitude,\
                    angles='xy', scale_units='xy', scale=1.0/scaleVectors, cmap='coolwarm', alpha=0.5, width=0.004, minshaft=0.01, minlength=0) #, headwidth=1.0)
            plt.clim(0, 1.0)

        plt.figure(figsize=(8,8))
        plt.imshow(foreground_values[indices], cmap='gray')
        visualizeVectorField(border_gradient_x[indices], border_gradient_y[indices], 1, 10.0)
        plt.gca().set_axis_off()
        plt.savefig("./border_grads.pdf", format="pdf", bbox_inches="tight")

        plt.figure(figsize=(8,8))
        plt.imshow(foreground_values[indices], cmap='gray')
        visualizeVectorField(propagated_gradient_without_attenuation_x[indices], propagated_gradient_without_attenuation_y[indices], stepSize, 10.0)
        plt.gca().set_axis_off()
        plt.savefig("./propagated_grad.pdf", format="pdf", bbox_inches="tight")

        plt.figure(figsize=(8,8))
        plt.imshow(foreground_values[indices], cmap='gray')
        visualizeVectorField(motionVectorFieldX[indices], motionVectorFieldY[indices], stepSize, 10.0)
        plt.gca().set_axis_off()
        plt.savefig("./attenuated_propagated_grad.pdf", format="pdf", bbox_inches="tight")

        plt.figure(figsize=(8,8))
        plt.imshow(foreground_values[indices], cmap='gray')
        visualizeVectorField(gradX[indices], gradY[indices], stepSize)
        plt.gca().set_axis_off()
        plt.savefig("./final_motion_field.pdf", format="pdf", bbox_inches="tight")

    plt.figure(figsize=(8,8))
    plt.imshow(img, cmap='gray')
    plt.figure(figsize=(8,8))
    plt.imshow(mask_surrogate, cmap='gray')
    plt.figure(figsize=(8,8))
    plt.imshow(mask_foreground, cmap='gray')
    plt.show()
