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

"""Module to apply motion fields to volumes using CUDA."""

import numpy as np
import cupy as cp
from cupy.cuda import runtime
from pathlib import Path

CUDA_KERNEL_PATH = Path(__file__).parent.parent/'cuda_kernels'

class ApplyMotionField():
    def __init__(self, volume : np.ndarray): #, motionField : np.ndarray):
        self.source_path = str(Path(__file__).resolve().parent) + "/cuda_kernels"
        self.include_dirs = ['-I'+self.source_path]
        self.cudaBlockSize = [8, 8, 8]
        self.volume = volume
        self.motionFieldScaled = None
        self.resampledVolume = None

        self.outputVolumeOnGPU_gpuarray = cp.zeros(self.volume.shape, dtype=np.float32)

        kernelSrcPath_resampleVolume = CUDA_KERNEL_PATH/"resampleVolume.cu"
        with open(kernelSrcPath_resampleVolume, "r") as kernel_file:
            src_resampleVolume = kernel_file.read()
        self.cudaFunc_resampleVolume = cp.RawKernel(code=src_resampleVolume, name="resampleVolume", options=tuple(self.include_dirs), backend='nvcc')

        self._set_texture(np.ascontiguousarray(self.volume.astype(np.float32)))


    def _set_texture(self, array:np.ndarray) -> cp.cuda.texture.TextureObject:
        """Get a texture object from a numpy array.

        Args:
            array (np.ndarray): The array to convert to a texture object.

        Returns:
            cupy.cuda.texture.TextureObject: The texture object.
        """
        # Create 3D CUDA array for segmentation
        tex_desc = cp.cuda.texture.TextureDescriptor(addressModes=(runtime.cudaAddressModeClamp,
                                                                runtime.cudaAddressModeClamp,
                                                                runtime.cudaAddressModeClamp),
                                            filterMode=runtime.cudaFilterModeLinear,
                                            readMode=runtime.cudaReadModeElementType,
                                            borderColors=None,
                                            normalizedCoords=False)

        channelformat_desc = cp.cuda.texture.ChannelFormatDescriptor(x=32,
                                                                    y=0,
                                                                    z=0,
                                                                    w=0,
                                                                    f=runtime.cudaChannelFormatKindFloat)


        arr=cp.asarray(array.copy(), order='C')
        depth, height, width = arr.shape

        self.vol_gpu = cp.cuda.texture.CUDAarray(desc=channelformat_desc,
                                            width=width,
                                            height=height,
                                            depth=depth,
                                            flags=0)

        self.vol_gpu.copy_from(arr)

        resource_desc = cp.cuda.texture.ResourceDescriptor(restype=runtime.cudaResourceTypeArray, cuArr=self.vol_gpu)

                        # Create texture object
        self.cudaFunc_resampleVolume_tex_volume = cp.cuda.texture.TextureObject(ResDesc=resource_desc, TexDesc=tex_desc)

    def free(self):
        del self.vol_gpu
        del self.cudaFunc_resampleVolume_tex_volume

    def setMotionField_cp(self, motionField_cp : cp.array):
        self.motionFieldOnGPU_array = motionField_cp #motionField_cp.copy()

    def setMotionField(self, motionField : np.array):
        self.motionFieldOnGPU_array = cp.asarray(np.ascontiguousarray(motionField))

    def resampleVolume(self, normalizeValues = False):
        shape = self.motionFieldOnGPU_array.shape
        blockSize = self.cudaBlockSize
        gridSize = ((shape[0] + blockSize[0] - 1) // blockSize[0],
                    (shape[1] + blockSize[1] - 1) // blockSize[1],
                    (shape[2] + blockSize[2] - 1) // blockSize[2])
        args = tuple(
            [self.outputVolumeOnGPU_gpuarray, self.motionFieldOnGPU_array, np.int32(shape[0]), np.int32(shape[1]), np.int32(shape[2]), np.int32(normalizeValues), self.cudaFunc_resampleVolume_tex_volume])
        self.cudaFunc_resampleVolume(args=args, block=tuple(blockSize), grid=tuple(gridSize))


    def getResampledVolumeAsArray(self):
        return cp.asnumpy(self.outputVolumeOnGPU_gpuarray, order="C")

    def updateMotionFieldOnHost(self):
        self.motionFieldScaled = cp.asnumpy(self.motionFieldOnGPU_array, order="C")

    def updateResampledVolumeOnHost(self):
        self.resampledVolume = cp.asnumpy(self.outputVolumeOnGPU_gpuarray)
