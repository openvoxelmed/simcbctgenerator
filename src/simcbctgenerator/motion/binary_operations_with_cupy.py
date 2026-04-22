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

"""Module for binary morphological operations using CuPy and CUDA."""

import numpy as np
import cupy as cp
# from cucim.core.operations.morphology import distance_transform_edt
from simcbctgenerator.dependencies.cucim_operation._distance_transform import distance_transform_edt
from pathlib import Path

CUDA_KERNEL_PATH = Path(__file__).parent.parent/'cuda_kernels'

class BinaryOperationsWithCupy():
    @staticmethod
    def binary_erosion(binary_mask : cp.array, kernelRadius : tuple):
        if len(kernelRadius) != 3:
            raise Exception("kernelRadius has wrong number of dimensions")

        kernelSrcPath_binaryErosion = CUDA_KERNEL_PATH/"binaryErosion.cu"
        with open(kernelSrcPath_binaryErosion, "r") as kernel_file:
            src_binaryErosion = kernel_file.read()
        binary_erosion_kernel = cp.RawKernel(src_binaryErosion, 'binaryErosion')

        shape = binary_mask.shape
        blockSize = (8, 8, 8)
        shared_memory_size = np.prod(blockSize)
        gridSize = ((shape[0] + blockSize[0]-3)//(blockSize[0]-2),
                    (shape[1] + blockSize[1]-3)//(blockSize[1]-2),
                    (shape[2] + blockSize[2]-3)//(blockSize[2]-2))
        idxMaskProcessed = 1
        empty_mask = cp.zeros(tuple(shape)).astype(bool)
        processMasks = [empty_mask, binary_mask.copy()]
        #processMasks = [cp.ascontiguousarray(cp.zeros_like(binary_mask)), cp.ascontiguousarray(binary_mask.copy())]

        growingStepsPerDim = np.array(kernelRadius)
        numIterations = growingStepsPerDim.max()
        for i in range(numIterations):
            kernel_np = np.ones((3, 3, 3)).astype(cp.uint8)

            # # uncomment block to create 3d cross structuring element (3x3x3)
            # kernel_np[0, 2, 0] = 0
            # kernel_np[0, 0, 2] = 0
            # kernel_np[2, 2, 0] = 0
            # kernel_np[2, 0, 2] = 0
            # kernel_np[0, 2, 2] = 0
            # kernel_np[2, 2, 2] = 0

            kernel_np[min(1, growingStepsPerDim[0]) : 1, :, :] = 0
            kernel_np[2 : 2 + (1 - min(1, growingStepsPerDim[0])), :, :] = 0
            kernel_np[:, min(1, growingStepsPerDim[1]) : 1, :] = 0
            kernel_np[:, 2 : 2 + (1 - min(1, growingStepsPerDim[1])), :] = 0
            kernel_np[:, :, min(1, growingStepsPerDim[2]): 1] = 0
            kernel_np[:, :, 2 : 2 + (1 - min(1, growingStepsPerDim[2]))] = 0

            kernel = cp.array(kernel_np)

            idxMaskProcessed = 1 - idxMaskProcessed
            binary_erosion_kernel(gridSize, blockSize, (processMasks[1-idxMaskProcessed], processMasks[idxMaskProcessed], kernel,
                                                        np.int32(shape[0]),
                                                        np.int32(shape[1]),
                                                        np.int32(shape[2])), shared_mem=int(shared_memory_size))

            growingStepsPerDim -= 1
            growingStepsPerDim = np.maximum(0, growingStepsPerDim)

        return processMasks[idxMaskProcessed]

    @staticmethod
    def binary_dilation(binary_mask : cp.array, kernelRadius : tuple):
        if len(kernelRadius) != 3:
            raise Exception("kernelRadius has wrong number of dimensions")


        kernelSrcPath_binaryDilation = CUDA_KERNEL_PATH/"binaryDilation.cu"
        with open(kernelSrcPath_binaryDilation, "r") as kernel_file:
            src_binaryDilation = kernel_file.read()
        binary_dilation_kernel = cp.RawKernel(src_binaryDilation, 'binaryDilation')

        shape = binary_mask.shape
        blockSize = (8, 8, 8)
        shared_memory_size = np.prod(blockSize)
        gridSize = ((shape[0] + blockSize[0]-3)//(blockSize[0]-2),
                    (shape[1] + blockSize[1]-3)//(blockSize[1]-2),
                    (shape[2] + blockSize[2]-3)//(blockSize[2]-2))
        idxMaskProcessed = 1
        empty_mask = cp.zeros(tuple(shape)).astype(bool)
        processMasks = [empty_mask, binary_mask.copy()]
        #processMasks = [cp.ascontiguousarray(cp.zeros_like(binary_mask)), cp.ascontiguousarray(binary_mask.copy())]

        growingStepsPerDim = np.array(kernelRadius)
        numIterations = growingStepsPerDim.max()
        for i in range(numIterations):
            kernel_np = np.ones((3, 3, 3)).astype(cp.uint8)

            # # uncomment block to create 3d cross structuring element (3x3x3)
            # kernel_np[0, 2, 0] = 0
            # kernel_np[0, 0, 2] = 0
            # kernel_np[2, 2, 0] = 0
            # kernel_np[2, 0, 2] = 0
            # kernel_np[0, 2, 2] = 0
            # kernel_np[2, 2, 2] = 0

            kernel_np[min(1, growingStepsPerDim[0]) : 1, :, :] = 0
            kernel_np[2 : 2 + (1 - min(1, growingStepsPerDim[0])), :, :] = 0
            kernel_np[:, min(1, growingStepsPerDim[1]) : 1, :] = 0
            kernel_np[:, 2 : 2 + (1 - min(1, growingStepsPerDim[1])), :] = 0
            kernel_np[:, :, min(1, growingStepsPerDim[2]): 1] = 0
            kernel_np[:, :, 2 : 2 + (1 - min(1, growingStepsPerDim[2]))] = 0

            kernel = cp.array(kernel_np)

            idxMaskProcessed = 1 - idxMaskProcessed
            binary_dilation_kernel(gridSize, blockSize, (processMasks[1-idxMaskProcessed], processMasks[idxMaskProcessed], kernel,
                                                        np.int32(shape[0]),
                                                        np.int32(shape[1]),
                                                        np.int32(shape[2])), shared_mem=int(shared_memory_size))

            growingStepsPerDim -= 1
            growingStepsPerDim = np.maximum(0, growingStepsPerDim)

        return processMasks[idxMaskProcessed]

    @staticmethod
    def binary_opening(binary_mask: cp.array, kernelRadius: tuple):
        M = BinaryOperationsWithCupy.binary_erosion(binary_mask, kernelRadius)
        return BinaryOperationsWithCupy.binary_dilation(M, kernelRadius)

    @staticmethod
    def binary_closing(binary_mask: cp.array, kernelRadius: tuple, safeBorders : bool = True):
        if safeBorders is True:
            padSize = np.array(kernelRadius)
            padded_binary_mask = cp.zeros((binary_mask.shape) + 2 * padSize).astype(bool)
            padded_binary_mask[padSize[0]:padSize[0] + binary_mask.shape[0],
                               padSize[1]:padSize[1] + binary_mask.shape[1],
                               padSize[2]:padSize[2] + binary_mask.shape[2]] = binary_mask
            D_padded = BinaryOperationsWithCupy.binary_dilation(padded_binary_mask, kernelRadius)
            E_padded = BinaryOperationsWithCupy.binary_erosion(D_padded, kernelRadius)
            resultMask = E_padded[padSize[0]:padSize[0] + binary_mask.shape[0],
                                  padSize[1]:padSize[1] + binary_mask.shape[1],
                                  padSize[2]:padSize[2] + binary_mask.shape[2]]
            return resultMask
        else:
            D = BinaryOperationsWithCupy.binary_dilation(binary_mask, kernelRadius)
            return BinaryOperationsWithCupy.binary_erosion(D, kernelRadius)


    @staticmethod
    def binary_erosion_approx_sphere(binary_mask : cp.array, kernelRadius : tuple):
        if len(kernelRadius) != 3:
            raise Exception("kernelRadius has wrong number of dimensions")

        kernelSrcPath_binaryErosion = CUDA_KERNEL_PATH/"binaryErosion.cu"
        with open(kernelSrcPath_binaryErosion, "r") as kernel_file:
            src_binaryErosion = kernel_file.read()
        binary_erosion_kernel = cp.RawKernel(src_binaryErosion, 'binaryErosion')

        shape = binary_mask.shape
        blockSize = (8, 8, 8)
        shared_memory_size = np.prod(blockSize)
        gridSize = ((shape[0] + blockSize[0]-3)//(blockSize[0]-2),
                    (shape[1] + blockSize[1]-3)//(blockSize[1]-2),
                    (shape[2] + blockSize[2]-3)//(blockSize[2]-2))
        idxMaskProcessed = 1
        empty_mask = cp.zeros(tuple(shape)).astype(bool)
        processMasks = [empty_mask, binary_mask.copy()]
        #processMasks = [cp.ascontiguousarray(cp.zeros_like(binary_mask)), cp.ascontiguousarray(binary_mask.copy())]

        growingStepsPerDim = np.array(kernelRadius)
        numIterations = growingStepsPerDim.max()

        kernels_np = np.ones((2, 3, 3, 3)).astype(cp.uint8)
        kernels_np[0, min(1, growingStepsPerDim[0]): 1, :, :] = 0
        kernels_np[0, 2: 2 + (1 - min(1, growingStepsPerDim[0])), :, :] = 0
        kernels_np[0, :, min(1, growingStepsPerDim[1]): 1, :] = 0
        kernels_np[0, :, 2: 2 + (1 - min(1, growingStepsPerDim[1])), :] = 0
        kernels_np[0, :, :, min(1, growingStepsPerDim[2]): 1] = 0
        kernels_np[0, :, :, 2: 2 + (1 - min(1, growingStepsPerDim[2]))] = 0

        kernels_np[1, 0, 2, 0] = 0
        kernels_np[1, 0, 0, 2] = 0
        kernels_np[1, 2, 2, 0] = 0
        kernels_np[1, 2, 0, 2] = 0
        kernels_np[1, 0, 2, 2] = 0
        kernels_np[1, 2, 2, 2] = 0

        kernels = cp.array(kernels_np)

        for i in range(numIterations):
            idxMaskProcessed = 1 - idxMaskProcessed
            binary_erosion_kernel(gridSize, blockSize, (processMasks[1-idxMaskProcessed], processMasks[idxMaskProcessed], kernels[i % 2, :, :, :],
                                                        np.int32(shape[0]),
                                                        np.int32(shape[1]),
                                                        np.int32(shape[2])), shared_mem=int(shared_memory_size))

            growingStepsPerDim -= 1
            growingStepsPerDim = np.maximum(0, growingStepsPerDim)

        return processMasks[idxMaskProcessed]

    @staticmethod
    def binary_dilation_approx_sphere(binary_mask : cp.array, kernelRadius : tuple):
        if len(kernelRadius) != 3:
            raise Exception("kernelRadius has wrong number of dimensions")

        kernelSrcPath_binaryDilation = CUDA_KERNEL_PATH/"binaryDilation.cu"
        with open(kernelSrcPath_binaryDilation, "r") as kernel_file:
            src_binaryDilation = kernel_file.read()
        binary_dilation_kernel = cp.RawKernel(src_binaryDilation, 'binaryDilation')

        shape = binary_mask.shape
        blockSize = (8, 8, 8)
        shared_memory_size = np.prod(blockSize)
        gridSize = ((shape[0] + blockSize[0]-3)//(blockSize[0]-2),
                    (shape[1] + blockSize[1]-3)//(blockSize[1]-2),
                    (shape[2] + blockSize[2]-3)//(blockSize[2]-2))
        idxMaskProcessed = 1
        empty_mask = cp.zeros(tuple(shape)).astype(bool)
        processMasks = [empty_mask, binary_mask.copy()]
        #processMasks = [cp.ascontiguousarray(cp.zeros_like(binary_mask)), cp.ascontiguousarray(binary_mask.copy())]

        growingStepsPerDim = np.array(kernelRadius)
        numIterations = growingStepsPerDim.max()

        kernels_np = np.ones((2, 3, 3, 3)).astype(cp.uint8)
        kernels_np[0, min(1, growingStepsPerDim[0]): 1, :, :] = 0
        kernels_np[0, 2: 2 + (1 - min(1, growingStepsPerDim[0])), :, :] = 0
        kernels_np[0, :, min(1, growingStepsPerDim[1]): 1, :] = 0
        kernels_np[0, :, 2: 2 + (1 - min(1, growingStepsPerDim[1])), :] = 0
        kernels_np[0, :, :, min(1, growingStepsPerDim[2]): 1] = 0
        kernels_np[0, :, :, 2: 2 + (1 - min(1, growingStepsPerDim[2]))] = 0

        kernels_np[1, 0, 2, 0] = 0
        kernels_np[1, 0, 0, 2] = 0
        kernels_np[1, 2, 2, 0] = 0
        kernels_np[1, 2, 0, 2] = 0
        kernels_np[1, 0, 2, 2] = 0
        kernels_np[1, 2, 2, 2] = 0

        kernels = cp.array(kernels_np)

        for i in range(numIterations):
            idxMaskProcessed = 1 - idxMaskProcessed
            binary_dilation_kernel(gridSize, blockSize, (processMasks[1-idxMaskProcessed], processMasks[idxMaskProcessed], kernels[i % 2, :, :, :],
                                                        np.int32(shape[0]),
                                                        np.int32(shape[1]),
                                                        np.int32(shape[2])), shared_mem=int(shared_memory_size))

            growingStepsPerDim -= 1
            growingStepsPerDim = np.maximum(0, growingStepsPerDim)

        return processMasks[idxMaskProcessed]

    @staticmethod
    def binary_opening_approx_sphere(binary_mask: cp.array, kernelRadius: tuple):
        M = BinaryOperationsWithCupy.binary_erosion_approx_sphere(binary_mask, kernelRadius)
        return BinaryOperationsWithCupy.binary_dilation_approx_sphere(M, kernelRadius)

    @staticmethod
    def binary_closing_approx_sphere(binary_mask: cp.array, kernelRadius: tuple, safeBorders : bool = True):
        if safeBorders is True:
            padSize = np.array(kernelRadius)
            padded_binary_mask = cp.zeros((binary_mask.shape) + 2 * padSize).astype(bool)
            padded_binary_mask[padSize[0]:padSize[0] + binary_mask.shape[0],
                               padSize[1]:padSize[1] + binary_mask.shape[1],
                               padSize[2]:padSize[2] + binary_mask.shape[2]] = binary_mask
            D_padded = BinaryOperationsWithCupy.binary_dilation_approx_sphere(padded_binary_mask, kernelRadius)
            E_padded = BinaryOperationsWithCupy.binary_erosion_approx_sphere(D_padded, kernelRadius)
            resultMask = E_padded[padSize[0]:padSize[0] + binary_mask.shape[0],
                                  padSize[1]:padSize[1] + binary_mask.shape[1],
                                  padSize[2]:padSize[2] + binary_mask.shape[2]]
            return resultMask
        else:
            D = BinaryOperationsWithCupy.binary_dilation_approx_sphere(binary_mask, kernelRadius)
            return BinaryOperationsWithCupy.binary_erosion_approx_sphere(D, kernelRadius)

    @staticmethod
    def computeIntegralVolume(vol : cp.array):
        paddedVol = cp.zeros(tuple(np.array(vol.shape) + 1))  # pad by 1 voxel in each axis direction
        paddedVol = paddedVol.astype(cp.float32)
        paddedVol[1:paddedVol.shape[0], 1:paddedVol.shape[1], 1:paddedVol.shape[2]] = vol.astype(cp.float32)  # first row, col 0 values, rest volume
        cp.cumsum(paddedVol, axis=0, dtype=cp.float32, out=paddedVol)
        cp.cumsum(paddedVol, axis=1, dtype=cp.float32, out=paddedVol)
        cp.cumsum(paddedVol, axis=2, dtype=cp.float32, out=paddedVol)
        return paddedVol
        # paddedVol = np.zeros(np.array(vol.shape) + 1, dtype=np.float32)  # pad by 1 voxel in each axis direction
        # paddedVol[1:paddedVol.shape[0], 1:paddedVol.shape[1], 1:paddedVol.shape[2]] = cp.asnumpy(vol).astype(np.float32)  # first row, col 0 values, rest volume
        # np.cumsum(paddedVol, axis=0, out=paddedVol)
        # np.cumsum(paddedVol, axis=1, out=paddedVol)
        # np.cumsum(paddedVol, axis=2, out=paddedVol)
        # return cp.array(paddedVol)
    @staticmethod
    def binary_cuboid_erosion(binary_mask: cp.array, kernelRadius: tuple):
        kernelSrcPath_binaryCuboidErosion = CUDA_KERNEL_PATH/"binaryCuboidErosion.cu"
        with open(kernelSrcPath_binaryCuboidErosion, "r") as kernel_file:
            src_binaryCuboidErosion = kernel_file.read()
        binary_cuboid_erosion_kernel = cp.RawKernel(src_binaryCuboidErosion, 'binaryCuboidErosion')

        intVolMask = BinaryOperationsWithCupy.computeIntegralVolume(binary_mask)

        shape = binary_mask.shape
        blockSize = (8, 8, 8)
        gridSize = ((shape[0] + blockSize[0] - 1) // blockSize[0],
                    (shape[1] + blockSize[1] - 1) // blockSize[1],
                    (shape[2] + blockSize[2] - 1) // blockSize[2])

        processedMask = cp.zeros(tuple(shape)).astype(bool)

        binary_cuboid_erosion_kernel(gridSize, blockSize,
                               (intVolMask, processedMask,
                                np.int32(kernelRadius[0]),
                                np.int32(kernelRadius[1]),
                                np.int32(kernelRadius[2]),
                                np.int32(shape[0]),
                                np.int32(shape[1]),
                                np.int32(shape[2])))

        return processedMask

    @staticmethod
    def binary_cuboid_dilation(binary_mask: cp.array, kernelRadius: tuple):
        kernelSrcPath_binaryCuboidDilation = CUDA_KERNEL_PATH/"binaryCuboidDilation.cu"
        with open(kernelSrcPath_binaryCuboidDilation, "r") as kernel_file:
            src_binaryCuboidDilation = kernel_file.read()
        binary_cuboid_dilation_kernel = cp.RawKernel(src_binaryCuboidDilation, 'binaryCuboidDilation')

        intVolMask = BinaryOperationsWithCupy.computeIntegralVolume(binary_mask)

        shape = binary_mask.shape
        blockSize = (8, 8, 8)
        gridSize = ((shape[0] + blockSize[0] - 1) // blockSize[0],
                    (shape[1] + blockSize[1] - 1) // blockSize[1],
                    (shape[2] + blockSize[2] - 1) // blockSize[2])

        processedMask = cp.zeros(tuple(shape)).astype(bool)

        binary_cuboid_dilation_kernel(gridSize, blockSize,
                               (intVolMask, processedMask,
                                np.int32(kernelRadius[0]),
                                np.int32(kernelRadius[1]),
                                np.int32(kernelRadius[2]),
                                np.int32(shape[0]),
                                np.int32(shape[1]),
                                np.int32(shape[2])))

        return processedMask

    @staticmethod
    def binary_cuboid_opening(binary_mask: cp.array, kernelRadius: tuple):
        M = BinaryOperationsWithCupy.binary_cuboid_erosion(binary_mask, kernelRadius)
        return BinaryOperationsWithCupy.binary_cuboid_dilation(M, kernelRadius)

    @staticmethod
    def binary_cuboid_closing(binary_mask: cp.array, kernelRadius: tuple, safeBorders : bool = True):
        if safeBorders is True:
            padSize = np.array(kernelRadius)
            padded_binary_mask = cp.zeros((np.array(binary_mask.shape)) + 2 * padSize).astype(bool)
            padded_binary_mask[padSize[0]:padSize[0]+binary_mask.shape[0],
                               padSize[1]:padSize[1] + binary_mask.shape[1],
                               padSize[2]:padSize[2] + binary_mask.shape[2]] = binary_mask
            D_padded = BinaryOperationsWithCupy.binary_cuboid_dilation(padded_binary_mask, kernelRadius)
            E_padded = BinaryOperationsWithCupy.binary_cuboid_erosion(D_padded, kernelRadius)
            resultMask = E_padded[padSize[0]:padSize[0] + binary_mask.shape[0],
                                  padSize[1]:padSize[1] + binary_mask.shape[1],
                                  padSize[2]:padSize[2] + binary_mask.shape[2]]
            return resultMask
        else:
            D = BinaryOperationsWithCupy.binary_cuboid_dilation(binary_mask, kernelRadius)
            return BinaryOperationsWithCupy.binary_cuboid_erosion(D, kernelRadius)

    @staticmethod
    def signed_distance_transform_to_mask_border_gpu(mask: cp.array, sampling=[1., 1., 1.]):
        inverted_mask = cp.logical_not(mask)
        distance_outside = distance_transform_edt(inverted_mask, return_distances=True, return_indices=False,
                                                  sampling=sampling)
        combined_distance = -distance_outside
        distance_inside = distance_transform_edt(mask.astype(bool), return_distances=True, return_indices=False,
                                                 sampling=sampling)
        combined_distance += distance_inside
        return combined_distance

    @staticmethod
    def binary_erosion_sphere(mask: cp.array, kernelRadius: tuple):
        binary_mask = mask.copy()
        distance_tmp = BinaryOperationsWithCupy.signed_distance_transform_to_mask_border_gpu(
            binary_mask.astype(np.float32), sampling = (1., 1., 1.))
        radius = kernelRadius[0]
        binary_mask = -(distance_tmp - 0.5) <= -radius
        return binary_mask

    @staticmethod
    def binary_dilation_sphere(mask: cp.array, kernelRadius: tuple):
        binary_mask = mask.copy()
        distance_tmp = BinaryOperationsWithCupy.signed_distance_transform_to_mask_border_gpu(
            binary_mask.astype(np.float32), sampling = (1., 1., 1.))
        radius = kernelRadius[0]
        binary_mask = -(distance_tmp + 0.5) <= radius
        return binary_mask

    @staticmethod
    def binary_opening_sphere(mask: cp.array, sampling: tuple, radius):
        binary_mask = mask.copy()
        distance_tmp = BinaryOperationsWithCupy.signed_distance_transform_to_mask_border_gpu(
            binary_mask.astype(np.float32), sampling = sampling) #sampling = (1., 1., 1.))
        #radius = kernelRadius[0]
        binary_mask = -(distance_tmp - 0.5) <= -radius
        distance_tmp = BinaryOperationsWithCupy.signed_distance_transform_to_mask_border_gpu(
            binary_mask.astype(np.float32), sampling = sampling) # sampling = (1., 1., 1.))
        binary_mask = -(distance_tmp + 0.5) <= radius
        return binary_mask
    @staticmethod
    def binary_closing_sphere(mask: cp.array, sampling: tuple, radius):
        kernelRadius = np.ceil(radius / np.array(sampling)).astype(np.int32)
        binary_mask = mask.copy()
        padSize = np.array(kernelRadius)
        padded_binary_mask = cp.zeros((np.array(binary_mask.shape)) + 2 * padSize).astype(bool)
        padded_binary_mask[padSize[0]:padSize[0] + binary_mask.shape[0],
        padSize[1]:padSize[1] + binary_mask.shape[1],
        padSize[2]:padSize[2] + binary_mask.shape[2]] = binary_mask

        #radius = kernelRadius #kernelRadius[0]
        #minKernelRadius = np.array(kernelRadius).min()
        #sampling = tuple(np.array(kernelRadius) / minKernelRadius) #(1., 1., 1.)
        distance_tmp = BinaryOperationsWithCupy.signed_distance_transform_to_mask_border_gpu(padded_binary_mask.astype(np.float32), sampling = sampling)

        padded_binary_mask = -(distance_tmp+0.5) <= radius
        distance_tmp = BinaryOperationsWithCupy.signed_distance_transform_to_mask_border_gpu(padded_binary_mask.astype(np.float32), sampling = sampling)
        padded_binary_mask = -(distance_tmp-0.5) <= -radius

        binary_mask = padded_binary_mask[padSize[0]:padSize[0] + binary_mask.shape[0],
                      padSize[1]:padSize[1] + binary_mask.shape[1],
                      padSize[2]:padSize[2] + binary_mask.shape[2]]

        return binary_mask
