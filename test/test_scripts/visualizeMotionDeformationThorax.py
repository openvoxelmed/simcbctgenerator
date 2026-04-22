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


if __name__ == '__main__':
    import os
    os.environ['DISPLAY'] = ':10'
    os.environ['MPLBACKEND'] = 'Qt5Agg'  # Must be set before any matplotlib import

    from simcbctgenerator.utils.config import MotionConfig
    from simcbctgenerator.generate_4d_ct import FourDCTGenerator
    from simcbctgenerator.patient import Patient, PatientConfig
    from pathlib import Path
    import SimpleITK as sitk
    import numpy as np
    from PyQt5.QtCore import QLibraryInfo

    import os

    os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = QLibraryInfo.location(
        QLibraryInfo.PluginsPath
    )
    print(os.environ['DISPLAY'])
    import sys
    if sys.platform != 'win32': 
        import matplotlib
        matplotlib.use('Qt5Agg')
    import matplotlib.pyplot as plt
    Dicom_path = Path(r'')


    motion_config = MotionConfig(
        motion_type = MotionConfig.MotionType.THORAX,
        amplitude_breathing= 20,
        amplitude_heart= 3,
        # contour_name= '',
        frequency_breathing= 20, # breaths per minute (12-20)
        frequency_heartbeat= 80,
        time_per_projection= 0.18, # seconds
        uncertainty= 0.02, # seconds
    )
    patient_config = PatientConfig(
        plan_dir= 'DICOM_PLAN',
        ct_dir= 'CT_SET',
        cbct_dir= 'CT_SET',
        export_structures= ['heart', 'aorta', 'lung', 'spine'],
        priority=[1,2,3,4],
        cm_mask=None,
        use_totalsegmentator=False,
        image_modality='dummy2')
    patient = Patient(patient_config, path=Path('dummy_123'))

    generator = FourDCTGenerator(motion_config, True)
    generator.initialize(patient)
    cts = generator.generate_dynamic_4d_CT(0)

    # Reinitialize matplotlib after multiprocessing (spawn corrupts Qt state)
    import matplotlib
    matplotlib.use('Qt5Agg', force=True)
    import importlib
    importlib.reload(plt)

    mask_seg_heart = generator.debugOutput.mask_seg_heart
    mask_seg_aorta = generator.debugOutput.mask_seg_aorta
    mask_seg_lung = generator.debugOutput.mask_seg_lung
    mask_seg_spine = generator.debugOutput.mask_seg_spine
    distance_foreground_heart = generator.debugOutput.distance_foreground_heart
    bone_mask = generator.debugOutput.bone_mask
    no_deformation = generator.debugOutput.no_deformation
    mask_foreground_heart = generator.debugOutput.mask_foreground_heart
    border_mask_heart = generator.debugOutput.border_mask_heart
    mask_foreground_lung = generator.debugOutput.mask_foreground_lung
    border_mask_lung = generator.debugOutput.border_mask_lung
    border_gradient_heart = generator.debugOutput.border_gradient_heart
    propagated_gradient_heart = generator.debugOutput.propagated_gradient_heart
    border_gradient_lung = generator.debugOutput.border_gradient_lung
    propagated_gradient_lung = generator.debugOutput.propagated_gradient_lung
    mask_border_voxels_with_valid_gradient_heart = generator.debugOutput.mask_border_voxels_with_valid_gradient_heart
    distance_next_border_voxel_with_gradient_heart = generator.debugOutput.distance_next_border_voxel_with_gradient_heart
    mask_border_voxels_with_valid_gradient_lung = generator.debugOutput.mask_border_voxels_with_valid_gradient_lung
    distance_next_border_voxel_with_gradient_lung = generator.debugOutput.distance_next_border_voxel_with_gradient_lung
    idxSlicez = int(generator.debugOutput.mask_seg_heart.shape[0] // 2)
    idxSlicey = int(generator.debugOutput.mask_seg_heart.shape[1] // 2)
    idxSlicex = int(generator.debugOutput.mask_seg_heart.shape[2] // 2)

    imgz = patient.ct_array[idxSlicez,:,:]
    imgy = patient.ct_array[:,idxSlicey,:]
    imgx = patient.ct_array[:,:,idxSlicex]

    if generator.motionDeformationModel.displacement_field_heart is not None:
        displacement_heart = sitk.GetImageFromArray(generator.motionDeformationModel.displacement_field_heart * 1)
        displacement_heart.SetSpacing(patient.ct_image.GetSpacing())
        motion_field_slice = sitk.GetArrayFromImage(displacement_heart)[idxSlicez, :, :, :]

        gradX_heart = motion_field_slice[:,:,0]
        gradY_heart = motion_field_slice[:,:,1]
        mag_heart = np.sqrt(gradX_heart**2 + gradY_heart**2)

    if generator.motionDeformationModel.displacement_field_lung is not None:
        displacement_lung = sitk.GetImageFromArray(generator.motionDeformationModel.displacement_field_lung * 1)
        displacement_lung.SetSpacing(patient.ct_image.GetSpacing())
        motion_field_slice = sitk.GetArrayFromImage(displacement_lung)[idxSlicez, :, :, :]

        gradX_lung = motion_field_slice[:,:,0]
        gradY_lung = motion_field_slice[:,:,1]
        mag_lung = np.sqrt(gradX_lung**2 + gradY_lung**2)

    mask_seg_heart = mask_seg_heart[idxSlicez,:,:]
    mask_seg_aorta = mask_seg_aorta[idxSlicez,:,:]
    mask_seg_lung = mask_seg_lung[idxSlicez,:,:]
    mask_seg_spine = mask_seg_spine[idxSlicez,:,:]
    distance_foreground_heart = distance_foreground_heart[idxSlicez,:,:]
    bone_mask = bone_mask[idxSlicez,:,:]
    mask_foreground_heart = mask_foreground_heart[idxSlicez,:,:]
    border_mask_heart = border_mask_heart[idxSlicez,:,:]
    mask_foreground_lung = mask_foreground_lung[idxSlicez,:,:]
    border_mask_lung = border_mask_lung[idxSlicez,:,:]
    mask_border_voxels_with_valid_gradient_heart = mask_border_voxels_with_valid_gradient_heart[idxSlicez,:,:]
    distance_next_border_voxel_with_gradient_heart = distance_next_border_voxel_with_gradient_heart[idxSlicez,:,:]
    mask_border_voxels_with_valid_gradient_lung = mask_border_voxels_with_valid_gradient_lung[idxSlicez,:,:]
    distance_next_border_voxel_with_gradient_lung = distance_next_border_voxel_with_gradient_lung[idxSlicez,:,:]
    no_deformation = no_deformation[idxSlicez,:,:]

    def drawDistanceMap(dist):
        plt.imshow(dist, cmap='PuOr', vmin=-dist.max(), vmax=dist.max())

    xrange = range(0,propagated_gradient_heart.shape[2])
    yrange = range(0,propagated_gradient_heart.shape[1])
    zrange = range(0,propagated_gradient_heart.shape[0])

    stepSize = 6
    indices_xy = np.ix_(yrange, xrange)
    indices_yz = np.ix_(zrange, yrange)
    indices_xz = np.ix_(zrange, xrange)

    if generator.motionDeformationModel.displacement_field_heart is not None:
        flowVectorScale = 1.0
        motionVectorFieldX_direction_z = propagated_gradient_heart[idxSlicez,:,:,0] * flowVectorScale
        motionVectorFieldY_direction_z = propagated_gradient_heart[idxSlicez,:,:,1] * flowVectorScale

        motionVectorFieldZ_direction_y = propagated_gradient_heart[:, idxSlicey, :, 2] * flowVectorScale
        motionVectorFieldX_direction_y = propagated_gradient_heart[:, idxSlicey, :, 0] * flowVectorScale

        motionVectorFieldZ_direction_x = propagated_gradient_heart[:, :, idxSlicex, 2] * flowVectorScale
        motionVectorFieldY_direction_x = propagated_gradient_heart[:, :, idxSlicex, 1] * flowVectorScale

        border_gradient_x_direction_z = border_gradient_heart[idxSlicez, :, :, 0] * flowVectorScale
        border_gradient_y_direction_z = border_gradient_heart[idxSlicez, :, :, 1] * flowVectorScale

        border_gradient_x_direction_y = border_gradient_heart[:, idxSlicey, :, 0] * flowVectorScale
        border_gradient_y_direction_y = border_gradient_heart[:, idxSlicey, :, 2] * flowVectorScale

        border_gradient_x_direction_x = border_gradient_heart[:, :, idxSlicex, 1] * flowVectorScale
        border_gradient_y_direction_x = border_gradient_heart[:, :, idxSlicex, 2] * flowVectorScale
        def visualizeVectorField(compX: np.ndarray, compY: np.ndarray, step = 1, scaleVectors = 1.0, ax = None):
            [X_all, Y_all] = np.meshgrid(np.arange(0, compX.shape[1]),
                                         np.arange(0, compX.shape[0]))
            ax.quiver(X_all[::step, ::step], Y_all[::step, ::step], compX[::step, ::step], compY[::step, ::step], \
                       angles='xy', scale_units='xy', scale=1.0/scaleVectors, color='y', alpha=0.9, width=0.004, minshaft=0.01, minlength=0)

        fig, ax = plt.subplots(1,3, figsize=(21,7))
        ax[0].imshow(imgz[indices_xy], cmap='gray')
        visualizeVectorField(border_gradient_x_direction_z[indices_xy], border_gradient_y_direction_z[indices_xy], 1, 10.0, ax=ax[0])
        ax[1].imshow(imgy[indices_xz], cmap='gray')
        visualizeVectorField(border_gradient_x_direction_y[indices_xz], border_gradient_y_direction_y[indices_xz], 1, 10.0, ax=ax[1])
        ax[2].imshow(imgx[indices_yz], cmap='gray')
        visualizeVectorField(border_gradient_x_direction_x[indices_yz], border_gradient_y_direction_x[indices_yz], 1, 10.0, ax=ax[2])
        plt.gca().set_axis_off()
        plt.show()

        fig, ax = plt.subplots(1,3,figsize=(21,7))
        ax[0].imshow(imgz[indices_xy], cmap='gray')
        visualizeVectorField(motionVectorFieldX_direction_z[indices_xy], motionVectorFieldY_direction_z[indices_xy], stepSize, 10.0, ax=ax[0])
        ax[0].set_axis_off()
        ax[1].imshow(imgy[indices_xz], cmap='gray')
        visualizeVectorField(motionVectorFieldX_direction_y[indices_xz], motionVectorFieldZ_direction_y[indices_xz], stepSize, 10.0, ax=ax[1])
        ax[1].set_axis_off()
        ax[2].imshow(imgx[indices_yz], cmap='gray')
        visualizeVectorField(motionVectorFieldY_direction_x[indices_yz], motionVectorFieldZ_direction_x[indices_yz], stepSize, 10.0, ax=ax[2])
        ax[2].set_axis_off()
        plt.show()

    if generator.motionDeformationModel.displacement_field_lung is not None:
        flowVectorScale = 1.0
        motionVectorFieldX_direction_z = propagated_gradient_lung[idxSlicez,:,:,0] * flowVectorScale
        motionVectorFieldY_direction_z = propagated_gradient_lung[idxSlicez,:,:,1] * flowVectorScale
        motionVectorFieldZ_direction_y = propagated_gradient_lung[:, idxSlicey, :, 2] * flowVectorScale
        motionVectorFieldX_direction_y = propagated_gradient_lung[:, idxSlicey, :, 0] * flowVectorScale
        motionVectorFieldZ_direction_x = propagated_gradient_lung[:, :, idxSlicex, 2] * flowVectorScale
        motionVectorFieldY_direction_x = propagated_gradient_lung[:, :, idxSlicex, 1] * flowVectorScale

        border_gradient_x_direction_z = border_gradient_lung[idxSlicez, :, :, 0] * flowVectorScale
        border_gradient_y_direction_z = border_gradient_lung[idxSlicez, :, :, 1] * flowVectorScale
        border_gradient_z_direction_z = border_gradient_lung[idxSlicez, :, :, 2] * flowVectorScale
        border_gradient_x_direction_y = border_gradient_lung[:, idxSlicey, :, 0] * flowVectorScale
        border_gradient_y_direction_y = border_gradient_lung[:, idxSlicey, :, 1] * flowVectorScale
        border_gradient_z_direction_y = border_gradient_lung[:, idxSlicey, :, 2] * flowVectorScale
        def visualizeVectorField(compX: np.ndarray, compY: np.ndarray, step = 1, scaleVectors = 1.0, ax=None):
            [X_all, Y_all] = np.meshgrid(np.arange(0, compX.shape[1]),
                                         np.arange(0, compX.shape[0]))
            ax.quiver(X_all[::step, ::step], Y_all[::step, ::step], compX[::step, ::step], compY[::step, ::step], \
                       angles='xy', scale_units='xy', scale=1.0/scaleVectors, color='y', alpha=0.9, width=0.004, minshaft=0.01, minlength=0)

        fig, ax = plt.subplots(1,3, figsize=(21,7))
        ax[0].imshow(imgz[indices_xy], cmap='gray')
        visualizeVectorField(border_gradient_x_direction_z[indices_xy], border_gradient_y_direction_z[indices_xy], 1, 10.0, ax=ax[0])
        ax[0].set_axis_off()
        ax[1].imshow(imgy[indices_xz], cmap='gray')
        visualizeVectorField(border_gradient_x_direction_y[indices_xz], border_gradient_y_direction_y[indices_xz], 1, 10.0, ax=ax[1])
        ax[1].set_axis_off()
        ax[2].imshow(imgx[indices_yz], cmap='gray')
        visualizeVectorField(border_gradient_x_direction_x[indices_yz], border_gradient_y_direction_x[indices_yz], 1, 10.0, ax=ax[2])
        ax[2].set_axis_off()

        fig, ax = plt.subplots(1,3,figsize=(21,7))
        ax[0].imshow(imgz[indices_xy], cmap='gray')
        visualizeVectorField(motionVectorFieldX_direction_z[indices_xy], motionVectorFieldY_direction_z[indices_xy], stepSize, 10.0, ax=ax[0])
        ax[0].set_axis_off()
        ax[1].imshow(imgy[indices_xz], cmap='gray')
        visualizeVectorField(motionVectorFieldX_direction_y[indices_xz], motionVectorFieldZ_direction_y[indices_xz], stepSize, 10.0, ax=ax[1])
        ax[1].set_axis_off()
        ax[2].imshow(imgx[indices_yz], cmap='gray')
        visualizeVectorField(motionVectorFieldY_direction_x[indices_yz], motionVectorFieldZ_direction_x[indices_yz], stepSize, 10.0, ax=ax[2])
        ax[2].set_axis_off()
        plt.show()







    plt.show()
