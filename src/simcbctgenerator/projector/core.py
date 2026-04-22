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

# This code was derived from the original codebase of the DeepDRR project.

"""Simplified projector core for basic ray casting.

This module contains a minimal Projector implementation that performs
simple ray casting without scatter simulation, complex attenuation,
or spectral processing.
"""

from __future__ import annotations
from typing import List, Optional, Union, Tuple
import numpy as np
import cupy as cp
import logging
from pathlib import Path

from .geometry import CameraProjection, CameraIntrinsicTransform
from .volume import Volume
from .device import Device


log = logging.getLogger(__name__)


def _get_texture(array: np.ndarray) -> Tuple[cp.cuda.TextureObject, cp.cuda.texture.CUDAarray]:
    """Get a texture object from a numpy array."""
    from cupy.cuda import runtime

    # Create 3D CUDA array for volume data
    tex_desc = cp.cuda.texture.TextureDescriptor(
        addressModes=(
            runtime.cudaAddressModeClamp,
            runtime.cudaAddressModeClamp,
            runtime.cudaAddressModeClamp
        ),
        filterMode=runtime.cudaFilterModePoint,  # Use point sampling
        readMode=runtime.cudaReadModeElementType,
        borderColors=None,
        normalizedCoords=False
    )

    channelformat_desc = cp.cuda.texture.ChannelFormatDescriptor(
        x=32, y=0, z=0, w=0,
        f=runtime.cudaChannelFormatKindFloat
    )

    # Move axes for CUDA texture layout
    arr = cp.asarray(np.moveaxis(array, [0, 1, 2], [2, 1, 0]).copy(), order='C')
    depth, height, width = arr.shape

    cuda_array = cp.cuda.texture.CUDAarray(
        desc=channelformat_desc,
        width=width,
        height=height,
        depth=depth,
        flags=0
    )

    cuda_array.copy_from(arr)

    resource_desc = cp.cuda.texture.ResourceDescriptor(
        restype=runtime.cudaResourceTypeArray,
        cuArr=cuda_array,
    )

    # Create texture object
    texture = cp.cuda.texture.TextureObject(ResDesc=resource_desc, TexDesc=tex_desc)
    return texture, cuda_array


def _compile_cuda_kernel(filename: str) -> cp.RawModule:
    source_path = Path(__file__).resolve().parent / "cuda" / filename
    with open(source_path, 'r') as f:
        source = f.read()
    log.debug(f"Compiling CUDA kernel from {source_path}")
    return cp.RawModule(code=source, options=(), backend='nvcc')


def _get_simple_kernel_module() -> cp.RawModule:
    return _compile_cuda_kernel("simple_project_kernel.cu")


def _get_polychromatic_kernel_module() -> cp.RawModule:
    return _compile_cuda_kernel("polychromatic_project_kernel.cu")

class Projector:
    """Simplified projector for basic ray casting."""

    def __init__(
        self,
        volume: Union[Volume, List[Volume]],
        camera_intrinsics: Optional[CameraIntrinsicTransform] = None,
        device: Optional[Device] = None,
        step: float = 0.1,
        source_to_detector_distance: float = -1,
        threads: int = 16, #TODO: make it default 16 delete all dependencies
        spectral_data=None,
        T1: float = 0.0,
        T2: float = 0.0,
    ):
        """Initialize the projector.

        Args:
            volume: Volume or list of volumes to project
            camera_intrinsics: Camera intrinsic parameters
            device: Imaging device (optional)
            step: Ray casting step size in voxels
            source_to_detector_distance: Distance from source to detector
        """
        # Handle volume list
        if isinstance(volume, Volume):
            self.volumes = [volume]
        else:
            self.volumes = list(volume)

        if len(self.volumes) > 5:
            raise ValueError("Only up to 5 volumes are supported in simplified projector")

        self.device = device
        self._camera_intrinsics = camera_intrinsics
        self.step = float(step)
        self._source_to_detector_distance = source_to_detector_distance

        # Compile CUDA kernel
        self.mod = _get_simple_kernel_module()
        self.project_kernel = self.mod.get_function("simple_project_kernel")
        self.threads = threads
        # Polychromatic kernel (only when spectral data provided)
        self.spectral_data = spectral_data
        self.T1 = float(T1)
        self.T2 = float(T2)
        if spectral_data is not None:
            self.poly_mod = _get_polychromatic_kernel_module()
            self.poly_kernel = self.poly_mod.get_function("polychromatic_project_kernel")
        else:
            self.poly_mod = None
            self.poly_kernel = None
        # Initialize state
        self.initialized = False
        self.output_shape = None

    @property
    def source_to_detector_distance(self) -> float:
        if self.device is not None:
            return self.device.source_to_detector_distance
        else:
            return self._source_to_detector_distance

    @property
    def camera_intrinsics(self) -> CameraIntrinsicTransform:
        if self.device is not None:
            return self.device.camera_intrinsics
        elif self._camera_intrinsics is not None:
            return self._camera_intrinsics
        else:
            raise RuntimeError("No camera intrinsics available")

    @property
    def volume(self) -> Volume:
        """Get single volume (for backward compatibility)."""
        if len(self.volumes) != 1:
            raise AttributeError("Projector contains multiple volumes")
        return self.volumes[0]

    def initialize(self):
        """Initialize GPU memory and textures."""
        if self.initialized:
            raise RuntimeError("Projector already initialized")

        log.debug("Initializing projector")

        # Create volume textures
        self.volume_textures = []
        self.volume_arrays = []

        for vol in self.volumes:
            texture, array = _get_texture(np.array(vol))
            self.volume_textures.append(texture)
            self.volume_arrays.append(array)

        # Allocate volume metadata on GPU
        num_vols = len(self.volumes)
        self.minPointX_gpu = cp.zeros(num_vols, dtype=cp.float32)
        self.minPointY_gpu = cp.zeros(num_vols, dtype=cp.float32)
        self.minPointZ_gpu = cp.zeros(num_vols, dtype=cp.float32)
        self.maxPointX_gpu = cp.zeros(num_vols, dtype=cp.float32)
        self.maxPointY_gpu = cp.zeros(num_vols, dtype=cp.float32)
        self.maxPointZ_gpu = cp.zeros(num_vols, dtype=cp.float32)
        self.voxelSizeX_gpu = cp.zeros(num_vols, dtype=cp.float32)
        self.voxelSizeY_gpu = cp.zeros(num_vols, dtype=cp.float32)
        self.voxelSizeZ_gpu = cp.zeros(num_vols, dtype=cp.float32)

        for i, vol in enumerate(self.volumes):
            self.minPointX_gpu[i] = -0.5
            self.minPointY_gpu[i] = -0.5
            self.minPointZ_gpu[i] = -0.5
            self.maxPointX_gpu[i] = vol.shape[0] - 0.5
            self.maxPointY_gpu[i] = vol.shape[1] - 0.5
            self.maxPointZ_gpu[i] = vol.shape[2] - 0.5
            self.voxelSizeX_gpu[i] = vol.spacing[0]
            self.voxelSizeY_gpu[i] = vol.spacing[1]
            self.voxelSizeZ_gpu[i] = vol.spacing[2]

        # Allocate source coordinates
        self.sourceX_gpu = cp.zeros(num_vols, dtype=cp.float32)
        self.sourceY_gpu = cp.zeros(num_vols, dtype=cp.float32)
        self.sourceZ_gpu = cp.zeros(num_vols, dtype=cp.float32)

        # Allocate transform matrices
        self.world_from_index_gpu = cp.zeros(9, dtype=cp.float32)
        self.ijk_from_world_gpu = cp.zeros(12, dtype=cp.float32)

        # Initialize output arrays
        self.initialize_output_arrays(self.camera_intrinsics.sensor_size)

        # Upload spectral tables (small: ~1.5 kB total for 120 bins)
        if self.spectral_data is not None:
            sd = self.spectral_data
            self.d_spectrum    = cp.asarray(sd.spectrum,    dtype=cp.float32)
            self.d_ratio_water = cp.asarray(sd.ratio_water, dtype=cp.float32)
            self.d_ratio_bone  = cp.asarray(sd.ratio_bone,  dtype=cp.float32)
            log.info(
                "Polychromatic: %d bins, E0=%.1f keV, "
                "mu_water=%.4f cm^-1, T1=%.6f, T2=%.6f (density units)",
                sd.num_bins, sd.ref_energy_kev,
                sd.mu_ref_water, self.T1, self.T2,
            )

        self.initialized = True

    def initialize_output_arrays(self, sensor_size: Tuple[int, int]):
        """Initialize output arrays for given sensor size."""
        if self.initialized and self.output_shape == sensor_size:
            return

        if self.initialized:
            del self.intensity_gpu

        self.output_shape = sensor_size
        output_size = sensor_size[0] * sensor_size[1]

        self.intensity_gpu = cp.zeros(output_size, dtype=cp.float32)
        log.debug(f"Allocated intensity array: {self.output_shape}")

    def update_volume_textures(self, volume:cp.ndarray):
        volume_ptr = self.volume_arrays[0]
        volume_ptr.copy_from(volume)

    def project(self, *camera_projections: CameraProjection) -> np.ndarray:
        """Perform ray projection.

        Args:
            camera_projections: Camera projection objects

        Returns:
            Projected intensity images
        """
        if not self.initialized:
            raise RuntimeError("Projector not initialized")

        if not camera_projections and self.device is None:
            raise ValueError("Must provide camera projection or device")
        elif not camera_projections and self.device is not None:
            camera_projections = [self.device.get_camera_projection()]

        use_poly = self.spectral_data is not None
        log.debug("Starting projection (polychromatic=%s)", use_poly)

        results = []
        for i, proj in enumerate(camera_projections):
            log.debug(f"Projecting view {i+1}/{len(camera_projections)}")

            # Initialize output arrays for this projection
            self.initialize_output_arrays(proj.intrinsic.sensor_size)

            # Get source position in world coordinates
            sx, sy, sz = proj.get_center_in_world()

            # Set up transforms - exactly like original DeepDRR
            # world_from_index should be 3x3 matrix flattened to 9 elements
            world_from_index = cp.asarray(proj.world_from_index[:-1, :].flatten()).astype(cp.float32)
            self.world_from_index_gpu = world_from_index

            # Set source positions in IJK coordinates for each volume
            for vol_id, vol in enumerate(self.volumes):
                source_ijk = np.array(vol.ijk_from_world.data @ np.array([sx, sy, sz, 1]))[:3]
                self.sourceX_gpu[vol_id] = source_ijk[0]
                self.sourceY_gpu[vol_id] = source_ijk[1]
                self.sourceZ_gpu[vol_id] = source_ijk[2]

                # Set IJK from world transform
                ijk_from_world = vol.ijk_from_world.data.flatten()[:12]
                self.ijk_from_world_gpu[:12] = cp.asarray(ijk_from_world).astype(cp.float32)

            # Calculate max ray length
            if self.source_to_detector_distance > 0:
                max_ray_length = np.sqrt(
                    self.source_to_detector_distance**2 +
                    self.device.detector_height**2 +
                    self.device.detector_width**2
                ) if self.device else self.source_to_detector_distance
            else:
                max_ray_length = -1.0

            # Kernel arguments (simplified for single volume)
            args = [
                np.int32(proj.sensor_width),
                np.int32(proj.sensor_height),
                np.float32(self.step),
                self.minPointX_gpu,
                self.minPointY_gpu,
                self.minPointZ_gpu,
                self.maxPointX_gpu,
                self.maxPointY_gpu,
                self.maxPointZ_gpu,
                self.voxelSizeX_gpu,
                self.voxelSizeY_gpu,
                self.voxelSizeZ_gpu,
                np.float32(sx),
                np.float32(sy),
                np.float32(sz),
                self.sourceX_gpu,
                self.sourceY_gpu,
                self.sourceZ_gpu,
                np.float32(max_ray_length),
                self.world_from_index_gpu,
                self.ijk_from_world_gpu,
                self.volume_textures[0],  # Use first volume texture
                self.intensity_gpu,
            ]

            if use_poly:
                sd = self.spectral_data
                args = args + [
                    self.d_spectrum,
                    self.d_ratio_water,
                    self.d_ratio_bone,
                    np.int32(sd.num_bins),
                    np.float32(self.T1),
                    np.float32(self.T2),
                ]
                kernel = self.poly_kernel
            else:
                kernel = self.project_kernel

            # Launch kernel
            blocks_w = int(np.ceil(proj.sensor_width / self.threads))
            blocks_h = int(np.ceil(proj.sensor_height / self.threads))
            block = (self.threads, self.threads, 1)
            grid = (blocks_w, blocks_h, 1)

            log.debug(f"Launching kernel: {blocks_w}x{blocks_h} blocks")
            kernel(block=block, grid=grid, args=tuple(args))

            # Copy result back
            intensity = cp.asnumpy(self.intensity_gpu)
            intensity = np.swapaxes(
                intensity.reshape(proj.sensor_width, proj.sensor_height), 0, 1
            ).copy()

            results.append(intensity)

        if len(results) == 1:
            return results[0]
        else:
            return np.stack(results)

    def free(self):
        """Free GPU memory."""
        if self.initialized:
            # Free textures
            for texture, array in zip(self.volume_textures, self.volume_arrays):
                del texture
                del array

            # Free GPU arrays
            del self.minPointX_gpu, self.minPointY_gpu, self.minPointZ_gpu
            del self.maxPointX_gpu, self.maxPointY_gpu, self.maxPointZ_gpu
            del self.voxelSizeX_gpu, self.voxelSizeY_gpu, self.voxelSizeZ_gpu
            del self.sourceX_gpu, self.sourceY_gpu, self.sourceZ_gpu
            del self.world_from_index_gpu, self.ijk_from_world_gpu
            del self.intensity_gpu
            if self.spectral_data is not None:
                           del self.d_spectrum, self.d_ratio_water, self.d_ratio_bone
        self.initialized = False

    def __enter__(self):
        self.initialize()
        return self

    def __exit__(self, _exc_type, _exc_val, _exc_tb):
        self.free()

    def __call__(self, *args, **kwargs):
        return self.project(*args, **kwargs)
