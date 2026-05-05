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

"""Minimal geometry module for projector functionality.

This module contains only the essential geometric transforms and objects
needed for basic ray projection, without the full DeepDRR geometry library.
"""

from __future__ import annotations
from typing import Union, Tuple, Optional, List
import numpy as np


class HomogeneousObject:
    """Base class for homogeneous coordinate objects."""

    dtype = np.float32

    def __init__(self, data: np.ndarray):
        data = data.data if isinstance(data, HomogeneousObject) else np.array(data)
        self.data = data.astype(self.dtype)

    def __array__(self, *args, **kwargs):
        return np.array(self.data, *args, **kwargs)

    def __getitem__(self, key):
        return self.data.__getitem__(key)

    def __setitem__(self, key, value):
        return self.data.__setitem__(key, value)

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.data.shape


class Point3D(HomogeneousObject):
    """3D point in homogeneous coordinates."""

    def __init__(self, x: Union[float, np.ndarray, List], y: Optional[float] = None, z: Optional[float] = None):
        if isinstance(x, (list, np.ndarray)) and len(x) >= 3:
            data = np.array([x[0], x[1], x[2], 1.0])
        elif y is not None and z is not None:
            data = np.array([x, y, z, 1.0])
        else:
            raise ValueError("Point3D requires either [x,y,z] array or x,y,z coordinates")
        super().__init__(data)

    @classmethod
    def from_any(cls, data):
        if isinstance(data, cls):
            return data
        return cls(data)

    @property
    def dim(self) -> int:
        return 3


class Vector3D(HomogeneousObject):
    """3D vector in homogeneous coordinates."""

    def __init__(self, x: Union[float, np.ndarray, List], y: Optional[float] = None, z: Optional[float] = None):
        if isinstance(x, (list, np.ndarray)) and len(x) >= 3:
            data = np.array([x[0], x[1], x[2], 0.0])
        elif y is not None and z is not None:
            data = np.array([x, y, z, 0.0])
        else:
            raise ValueError("Vector3D requires either [x,y,z] array or x,y,z coordinates")
        super().__init__(data)

    @property
    def dim(self) -> int:
        return 3


class FrameTransform(HomogeneousObject):
    """4x4 homogeneous transformation matrix."""

    def __init__(self, data: np.ndarray):
        if isinstance(data, (list, tuple)):
            data = np.array(data)
        if data.shape != (4, 4):
            raise ValueError("FrameTransform requires 4x4 matrix")
        super().__init__(data)

    @classmethod
    def identity(cls):
        return cls(np.eye(4))

    @property
    def dim(self) -> int:
        return 3

    @property
    def inv(self):
        """Get the inverse transform."""
        return FrameTransform(np.linalg.inv(self.data))

    def __matmul__(self, other):
        """Matrix multiplication with @ operator."""
        if isinstance(other, (Point3D, Vector3D)):
            return self.data @ other.data
        elif isinstance(other, FrameTransform):
            return FrameTransform(self.data @ other.data)
        else:
            return self.data @ other

    def toarray(self):
        """Get the transform as a numpy array."""
        return self.data.copy()


class CameraIntrinsicTransform(HomogeneousObject):
    """Camera intrinsic parameters."""

    def __init__(self, sensor_height: int, sensor_width: int, pixel_size: float|tuple[float, float],
                 source_to_detector_distance: float, fx: Optional[float] = None, fy: Optional[float] = None,
                 detector_offset_x: float = 0.0, detector_offset_y: float = 0.0):
        self.sensor_height = sensor_height
        self.sensor_width = sensor_width
        if isinstance(pixel_size, (int, float)):
            pixel_size = (pixel_size, pixel_size)
        self.pixel_size = pixel_size
        self.source_to_detector_distance = source_to_detector_distance
        self.detector_offset_x = detector_offset_x
        self.detector_offset_y = detector_offset_y

        # Calculate focal lengths if not provided
        if fx is None:
            fx = source_to_detector_distance / pixel_size[0]
        if fy is None:
            fy = source_to_detector_distance / pixel_size[1]


        self.fx = fx
        self.fy = fy

        # Principal point at center, adjusted for detector offsets
        self.cx = sensor_width / 2.0
        self.cy = sensor_height / 2.0
        if detector_offset_x != 0.0:
           self.cx += detector_offset_x / pixel_size[0]
        if detector_offset_y != 0.0:
           self.cy += detector_offset_y / pixel_size[1]

        # Create intrinsic matrix
        data = np.array([
            [fx, 0, self.cx],
            [0, fy, self.cy],
            [0, 0, 1]
        ])
        super().__init__(data)

    @property
    def sensor_size(self) -> Tuple[int, int]:
        return (self.sensor_height, self.sensor_width)

    @property
    def dim(self) -> int:
        return 3


class CameraProjection:
    """Camera projection combining intrinsic and extrinsic parameters."""

    def __init__(self, intrinsic: CameraIntrinsicTransform, extrinsic: FrameTransform):
        self.intrinsic = intrinsic
        self.extrinsic = extrinsic  # This is camera3d_from_world

        # Store transforms following DeepDRR convention
        self.index_from_camera2d = intrinsic
        self.camera3d_from_world = extrinsic

        # Calculate world_from_camera3d (inverse of extrinsic)
        self.world_from_camera3d = self.camera3d_from_world.inv

        # Calculate the projection matrix exactly like original DeepDRR
        # index_from_world = index_from_camera3d @ camera3d_from_world
        # where index_from_camera3d = index_from_camera2d @ camera2d_from_camera3d

        # Step 1: Create camera2d_from_camera3d (projects 3D camera coords to 2D)
        proj = np.concatenate([np.eye(3), np.zeros((3, 1))], axis=1)  # [[1,0,0,0], [0,1,0,0], [0,0,1,0]]

        # Step 2: index_from_camera3d = index_from_camera2d @ camera2d_from_camera3d
        index_from_camera3d = intrinsic.data @ proj  # (3x3) @ (3x4) = (3x4)

        # Step 3: index_from_world = index_from_camera3d @ camera3d_from_world
        index_from_world_matrix = index_from_camera3d @ self.camera3d_from_world.data  # (3x4) @ (4x4) = (3x4)

        # For DeepDRR compatibility, we need to store as FrameTransform-like object
        # but the original returns 3x4 and 4x3 matrices, not 4x4
        self._index_from_world_3x4 = index_from_world_matrix

        # Create 4x4 version for matrix inversion
        # We need to create a proper 4x4 matrix from the 3x4 projection matrix
        # The 4th row should be [0, 0, 0, 1] to make it a valid homogeneous transform
        index_from_world_4x4 = np.vstack([index_from_world_matrix, [0, 0, 0, 1]])

        # Compute world_from_index as 4x4 inverse, then extract the part we need
        world_from_index_4x4 = np.linalg.inv(index_from_world_4x4)

        # For the CUDA kernel, we need the 4x3 matrix (first 4 rows, first 3 cols)
        # But we actually need the 3x3 part that maps (u,v,1) to ray direction (rx,ry,rz)
        # This is the top-left 3x3 of world_from_index_4x4
        self._world_from_index_4x3 = world_from_index_4x4[:4, :3]

        # Create FrameTransform versions
        self._index_from_world = FrameTransform(index_from_world_4x4)
        self._world_from_index = FrameTransform(world_from_index_4x4)

        # Camera properties
        self.sensor_height = intrinsic.sensor_height
        self.sensor_width = intrinsic.sensor_width

    def get_center_in_world(self):
        """Get the camera center in world coordinates."""
        # Camera center is at origin in camera coordinates [0, 0, 0, 1]
        camera_center = np.array([0, 0, 0, 1])
        # Transform to world: world_from_camera3d @ camera_center
        world_center = self.world_from_camera3d.data @ camera_center
        return world_center[:3]

    @property
    def center_in_world(self):
        """Get camera center as Point3D."""
        return self.get_center_in_world()

    @property
    def index_from_camera3d(self) -> Transform:
        proj = np.concatenate([np.eye(3), np.zeros((3, 1))], axis=1)
        camera2d_from_camera3d = Transform(proj, _inv=proj.T)
        return self.index_from_camera2d @ camera2d_from_camera3d

    @property
    def world_from_index(self):
        """Get world_from_index transform (following DeepDRR convention)."""
        # Return object that behaves like the original - should have .data and support array indexing
        class WorldFromIndexWrapper:
            def __init__(self, matrix_4x3):
                self.data = matrix_4x3
                self._matrix = matrix_4x3

            def __array__(self):
                return self._matrix

            def __getitem__(self, key):
                return self._matrix[key]

        return WorldFromIndexWrapper(self._world_from_index_4x3)

    @property
    def index_from_world(self):
        """Get index_from_world transform (following DeepDRR convention)."""
        # Return object that behaves like the original - should have .data and support array indexing
        class IndexFromWorldWrapper:
            def __init__(self, matrix_3x4):
                self.data = matrix_3x4
                self._matrix = matrix_3x4

            def __array__(self):
                return self._matrix

            def __getitem__(self, key):
                return self._matrix[key]

        return IndexFromWorldWrapper(self._index_from_world_3x4)

    def get_ray_transform(self, volume):
        """Get ray transform for volume projection."""
        # This should return the transform from detector pixels to world ray directions
        # Following DeepDRR's approach
        return self.world_from_camera3d


# Convenience functions
def point(x: float, y: float, z: float) -> Point3D:
    """Create a 3D point."""
    return Point3D(x, y, z)


def vector(x: float, y: float, z: float) -> Vector3D:
    """Create a 3D vector."""
    return Vector3D(x, y, z)


def frame_transform(data: np.ndarray) -> FrameTransform:
    """Create a frame transform."""
    return FrameTransform(data)


# Common transforms
def RAS_from_LPS():
    """Transform from LPS to RAS coordinates."""
    return FrameTransform(np.array([
        [-1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ]))


def LPS_from_RAS():
    """Transform from RAS to LPS coordinates."""
    return RAS_from_LPS()  # Same transform


# Type aliases for convenience
Point = Point3D
Vector = Vector3D
Transform = FrameTransform

# Export convenience shortcuts
p = point
v = vector
f = frame_transform
F = FrameTransform
