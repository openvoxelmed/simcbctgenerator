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

"""Custom image processing utilities."""

import numpy as np
import cupy as cp
from simcbctgenerator.dependencies.cucim_operation._distance_transform import distance_transform_edt
from simcbctgenerator.motion.binary_operations_with_cupy import BinaryOperationsWithCupy
from pathlib import Path

CUDA_KERNEL_PATH = Path(__file__).parent.parent/'cuda_kernels'

class IntegralVolume:
    @staticmethod
    def computeIntegralVolume(vol):
        paddedVol = np.zeros(np.array(vol.shape) + 1, dtype=np.float32)  # pad by 1 voxel in each axis direction
        paddedVol[1:paddedVol.shape[0], 1:paddedVol.shape[1],
        1:paddedVol.shape[2]] = vol.astype(np.float32)  # first row, col 0 values, rest volume
        np.cumsum(paddedVol, axis=0, out=paddedVol)
        np.cumsum(paddedVol, axis=1, out=paddedVol)
        np.cumsum(paddedVol, axis=2, out=paddedVol)
        return paddedVol

    @staticmethod
    def sampleFromIntegralVolume(intVol, x, y, z, shape):
        values = intVol[x.flatten(), y.flatten(), z.flatten()]
        values = np.reshape(values, shape)
        return values

    @staticmethod
    def clip(array, size):
        np.minimum(array, size - 2, out=array)
        np.maximum(0, array, out=array)
        return array.astype(np.int16)

    @staticmethod
    def computeVoxelSum(intVol, x1, x2, y1, y2, z1, z2):
        # enforce bound limits

        x1 = IntegralVolume.clip(x1, intVol.shape[2])#np.maximum(0, np.minimum(x1, intVol.shape[2] - 2)).astype(
            #int)  # -2... -1 for indexing, -1 because intVol upper bounds of indices are increased by 1
        x2 = IntegralVolume.clip(x2, intVol.shape[2])#np.maximum(0, np.minimum(x2, intVol.shape[2] - 2)).astype(int)
        y1 = IntegralVolume.clip(y1, intVol.shape[1])#np.maximum(0, np.minimum(y1, intVol.shape[1] - 2)).astype(int)
        y2 = IntegralVolume.clip(y2, intVol.shape[1])#np.maximum(0, np.minimum(y2, intVol.shape[1] - 2)).astype(int)
        z1 = IntegralVolume.clip(z1, intVol.shape[0])#np.maximum(0, np.minimum(z1, intVol.shape[0] - 2)).astype(int)
        z2 = IntegralVolume.clip(z2, intVol.shape[0])#np.maximum(0, np.minimum(z2, intVol.shape[0] - 2)).astype(int)

        dic = {'x1': x1,
               'x2': x2,
               'y1': y1,
               'y2': y2,
               'z1': z1,
               'z2': z2,}

        shape = x1.shape

        for n, v, operator in zip([[1, 1, 1],          [1, 1, 0],          [1, 0, 1],          [0, 1, 1],          [1, 0, 0],          [0, 1, 0],          [0, 0, 1],          [0, 0, 0]],
                               [['z2', 'y2', 'x2'], ['z2', 'y2', 'x1'], ['z2', 'y1', 'x2'], ['z1', 'y2', 'x2'], ['z2', 'y1', 'x1'], ['z1', 'y2', 'x1'], ['z1', 'y1', 'x2'], ['z1', 'y1', 'x1']],
                               [None, "sub", "sub", "sub", 'add', 'add', 'add', 'sub']):
            if operator is None:
                image = IntegralVolume.sampleFromIntegralVolume(intVol, dic[v[0]]+n[0], dic[v[1]]+n[1],dic[v[2]]+n[2], shape)
            elif operator == 'sub':
                image -= IntegralVolume.sampleFromIntegralVolume(intVol, dic[v[0]]+n[0], dic[v[1]]+n[1],dic[v[2]]+n[2], shape)
            elif operator == 'add':
                image += IntegralVolume.sampleFromIntegralVolume(intVol, dic[v[0]]+n[0], dic[v[1]]+n[1],dic[v[2]]+n[2], shape)
            else:
                raise NotImplementedError

        return image

class IntegralVolumeGPU:
    @staticmethod
    def computeIntegralVolume(vol: cp.array):
        paddedVol = cp.zeros(tuple(np.array(vol.shape) + 1), dtype=cp.float32)  # pad by 1 voxel in each axis direction
        paddedVol[1:paddedVol.shape[0], 1:paddedVol.shape[1], 1:paddedVol.shape[2]] = vol.astype(
            cp.float32)  # first row, col 0 values, rest volume
        cp.cumsum(paddedVol, axis=0, dtype=cp.float32, out=paddedVol)
        cp.cumsum(paddedVol, axis=1, dtype=cp.float32, out=paddedVol)
        cp.cumsum(paddedVol, axis=2, dtype=cp.float32, out=paddedVol)
        return paddedVol

class DistanceTransformGPU:
    @staticmethod
    def signed_distance_transform_gpu(_mask : cp.array, sampling=[1.,1.,1.]):
        inverted_mask = cp.logical_not(_mask)
        distance_outside = distance_transform_edt(inverted_mask, return_distances=True, return_indices=False, sampling=sampling)
        combined_distance = -distance_outside
        mask = _mask.astype(bool)
        eroded_mask = BinaryOperationsWithCupy.binary_erosion(mask, (1,1,1))
        distance_inside = distance_transform_edt(eroded_mask, return_distances=True, return_indices=False, sampling=sampling)
        combined_distance += distance_inside
        return combined_distance

    @staticmethod
    def signed_distance_transform_to_mask_border_gpu(_mask : cp.array, sampling=[1.,1.,1.]):
        inverted_mask = cp.logical_not(_mask)
        distance_outside = distance_transform_edt(inverted_mask, return_distances=True, return_indices=False, sampling=sampling)
        combined_distance = -distance_outside
        mask = _mask.astype(bool)
        distance_inside = distance_transform_edt(mask, return_distances=True, return_indices=False, sampling=sampling)
        combined_distance += distance_inside
        return combined_distance

class MotionFieldPropagation:
    @staticmethod
    def propagateMotionField(intVolCompX : cp.array,
                             intVolCompY : cp.array,
                             intVolCompZ : cp.array,
                             intVolValidGradOcc : cp.array,
                             distanceMap : cp.array,
                             outCompX : cp.array,
                             outCompY : cp.array,
                             outCompZ : cp.array,
                             spacing : tuple):
        kernelSrcPath_motionFieldPropagation = CUDA_KERNEL_PATH/"motionFieldPropagation.cu"
        with open(kernelSrcPath_motionFieldPropagation, "r") as kernel_file:
            src_motionFieldPropagation = kernel_file.read()
        motionFieldPropagation_kernel = cp.RawKernel(src_motionFieldPropagation, 'motionFieldPropagation')

        shape = distanceMap.shape
        blockSize = (8, 8, 8)
        gridSize = ((shape[0] + blockSize[0] - 1) // blockSize[0],
                    (shape[1] + blockSize[1] - 1) // blockSize[1],
                    (shape[2] + blockSize[2] - 1) // blockSize[2])

        motionFieldPropagation_kernel(gridSize, blockSize,
                               (intVolCompX,
                                intVolCompY,
                                intVolCompZ,
                                intVolValidGradOcc,
                                distanceMap,
                                outCompX,
                                outCompY,
                                outCompZ,
                                np.int32(shape[0]),
                                np.int32(shape[1]),
                                np.int32(shape[2]),
                                np.float32(spacing[0]),
                                np.float32(spacing[1]),
                                np.float32(spacing[2])))
