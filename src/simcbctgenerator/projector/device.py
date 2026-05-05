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

"""Simplified device classes for CBCT imaging.

This module contains minimal Device and CBCT implementations
focused on camera positioning and geometry for ray projection.
"""

from __future__ import annotations
from typing import Optional
import numpy as np
import math
from scipy.spatial.transform import Rotation

from .geometry import (
    Point3D, Vector3D, FrameTransform, CameraIntrinsicTransform,
    CameraProjection, point, vector
)


class Device:
    """Base class for X-ray imaging devices."""

    def __init__(
        self,
        sensor_height: int = 1536,
        sensor_width: int = 1536,
        pixel_size: float|tuple[float, float] = 0.194,
        source_to_detector_distance: float = 1020,
        world_from_device: Optional[FrameTransform] = None,
        detector_offset_x: float = 0.0,
        detector_offset_y: float = 0.0
    ):
        self.sensor_height = sensor_height
        self.sensor_width = sensor_width
        if isinstance(pixel_size, (int, float)):
            pixel_size = (pixel_size, pixel_size)
        self.pixel_size = pixel_size
        self.source_to_detector_distance = source_to_detector_distance
        self.detector_offset_x = detector_offset_x
        self.detector_offset_y = detector_offset_y

        if world_from_device is None:
            self.world_from_device = FrameTransform.identity()
        else:
            self.world_from_device = world_from_device

        # Create camera intrinsics
        self._rebuild_camera_intrinsics()

    def _rebuild_camera_intrinsics(self):
        """Rebuild camera intrinsics with current offset values."""
        self.camera_intrinsics = CameraIntrinsicTransform(
            sensor_height=self.sensor_height,
            sensor_width=self.sensor_width,
            pixel_size=self.pixel_size,
            source_to_detector_distance=self.source_to_detector_distance,
            detector_offset_x=self.detector_offset_x,
            detector_offset_y=self.detector_offset_y
        )

    @property
    def detector_height(self) -> float:
        """Height of detector in mm."""
        return self.sensor_height * self.pixel_size[1]

    @property
    def detector_width(self) -> float:
        """Width of detector in mm."""
        return self.sensor_width * self.pixel_size[0]

    @property
    def device_from_world(self) -> FrameTransform:
        """Transform from world to device coordinates."""
        return self.world_from_device.inv

    def get_camera_projection(self) -> CameraProjection:
        """Get current camera projection."""
        # Default implementation - override in subclasses
        return CameraProjection(self.camera_intrinsics, FrameTransform.identity())


class CBCT(Device):
    """Simplified CBCT device."""

    def __init__(
        self,
        isocenter: Point3D = None,
        alpha: float = 0,
        beta: float = 0,
        gamma: float = 0,
        degrees: bool = True,
        source_to_detector_distance: float = 1020,
        source_to_isocenter_vertical_distance: float = 530,
        source_to_isocenter_horizontal_offset: float = 0,
        sensor_height: int = 1536,
        sensor_width: int = 1536,
        pixel_size: float|tuple[float, float] = 0.194,
        world_from_device: Optional[FrameTransform] = None,
        rotate_camera_left: bool = True,
        rotation_direction_clockwise: bool = True,
        detector_offset_x: float = 0.0,
        detector_offset_y: float = 0.0
    ):
        """Initialize CBCT.

        Args:
            isocenter: Center of rotation in device coordinates
            alpha: Primary angulation in degrees/radians
            beta: Secondary angulation in degrees/radians
            gamma: Detector rotation in degrees/radians
            degrees: Whether angles are in degrees
            source_to_detector_distance: Distance from source to detector in mm
            source_to_isocenter_vertical_distance: Vertical offset of source from isocenter
            source_to_isocenter_horizontal_offset: Horizontal offset of source from isocenter
            detector_offset_x: Horizontal detector offset in mm
            detector_offset_y: Vertical detector offset in mm
        """
        super().__init__(
            sensor_height=sensor_height,
            sensor_width=sensor_width,
            pixel_size=pixel_size,
            source_to_detector_distance=source_to_detector_distance,
            world_from_device=world_from_device,
            detector_offset_x=detector_offset_x,
            detector_offset_y=detector_offset_y
        )

        # Set angles
        if degrees:
            self.alpha = math.radians(alpha)
            self.beta = math.radians(beta)
            self.gamma = math.radians(gamma)
        else:
            self.alpha = alpha
            self.beta = beta
            self.gamma = gamma

        # Set isocenter
        if isocenter is None:
            self.isocenter = point(0, 0, 0)
        else:
            self.isocenter = Point3D.from_any(isocenter)

        self.source_to_isocenter_vertical_distance = source_to_isocenter_vertical_distance
        self.source_to_isocenter_horizontal_offset = source_to_isocenter_horizontal_offset
        self.rotate_camera_left = rotate_camera_left
        self.rotation_direction_clockwise = rotation_direction_clockwise

    @property
    def principle_ray_in_world(self) -> Vector3D:
        """Principal ray direction in world coordinates."""
        # Default direction (detector direction)
        ray_device = vector(0, 0, 1)  # Pointing along +Z in device frame
        return self.world_from_device @ ray_device

    def move_to(
        self,
        isocenter: Optional[Point3D] = None,
        alpha: Optional[float] = None,
        beta: Optional[float] = None,
        gamma: Optional[float] = None,
        degrees: bool = True,
        detector_offset_x: Optional[float] = None,
        detector_offset_y: Optional[float] = None,
    ):
        """Move CBCT to specified position with optional detector offset update.

        Args:
            isocenter: New isocenter position
            alpha: Primary gantry angle
            beta: Secondary gantry angle
            gamma: Detector rotation angle
            degrees: Whether angles are in degrees
            detector_offset_x: New horizontal detector offset in mm
            detector_offset_y: New vertical detector offset in mm
        """
        if isocenter is not None:
            self.isocenter = Point3D.from_any(isocenter)

        if alpha is not None:
            self.alpha = math.radians(alpha) if degrees else alpha
            if self.rotation_direction_clockwise:
                self.alpha = -self.alpha
        if beta is not None:
            self.beta = math.radians(beta) if degrees else beta
        if gamma is not None:
            self.gamma = math.radians(gamma) if degrees else gamma

        # Handle offset updates
        offsets_changed = False
        if detector_offset_x is not None and detector_offset_x != self.detector_offset_x:
            self.detector_offset_x = detector_offset_x
            offsets_changed = True
        if detector_offset_y is not None and detector_offset_y != self.detector_offset_y:
            self.detector_offset_y = detector_offset_y
            offsets_changed = True

        if offsets_changed:
            self._rebuild_camera_intrinsics()


    @property
    def arm_from_device(self) -> FrameTransform:
        return self.device_from_arm.inv

    @property
    def device_from_arm(self) -> FrameTransform:
        rot = Rotation.from_euler("xy", [self.alpha, self.beta]).as_matrix()
        tans = np.array(self.isocenter)[:3]
        device_from_arm_data = np.eye(4)
        device_from_arm_data[:3, :3] = rot
        device_from_arm_data[:3, 3] = tans
        return FrameTransform(device_from_arm_data)

    @property
    def camera3d_from_device(self) -> FrameTransform:

        # Step 3: camera3d_from_arm (like original lines 231-237)
        camera3d_from_arm_data = np.eye(4)
        camera3d_from_arm_data[:3, 3] = np.array([
            0,
            -self.source_to_isocenter_horizontal_offset,
            self.source_to_isocenter_vertical_distance
        ])
        camera3d_from_arm = FrameTransform(camera3d_from_arm_data)

        # Step 4: Apply 90-degree rotation if rotate_camera_left (like original lines 238-242)
        if self.rotate_camera_left:
            z_rotation = Rotation.from_euler("z", 90, degrees=True).as_matrix()
            z_rot_transform = FrameTransform(np.eye(4))
            z_rot_transform.data[:3, :3] = z_rotation
            camera3d_from_arm = z_rot_transform @ camera3d_from_arm

        # Step 5: Apply gamma rotation (like original lines 247-249)
        gamma_rotation = Rotation.from_euler("z", self.gamma, degrees=False).as_matrix()
        gamma_transform = FrameTransform(np.eye(4))
        gamma_transform.data[:3, :3] = gamma_rotation

        # Step 6: camera3d_from_device (like original line 251)
        camera3d_from_device = gamma_transform @ camera3d_from_arm @ self.arm_from_device


        return camera3d_from_device

    def get_camera3d_from_world(self) -> FrameTransform:
        """Get camera3d_from_world transform following DeepDRR's exact approach.

        This replicates the original DeepDRR camera transform calculation.
        """

        return self.camera3d_from_device @ self.device_from_world

    def get_camera_projection(self) -> CameraProjection:
        """Get current camera projection based on CBCT pose."""
        # Calculate current camera transform
        extrinsic = self.get_camera3d_from_world()

        return CameraProjection(self.camera_intrinsics, extrinsic)

    @property
    def device_from_camera3d(self) -> FrameTransform:
        """Transform from camera3d to device coordinates."""
        # This is the inverse of the camera positioning
        # camera_transform = self.get_camera3d_from_world(
        #     self.isocenter,
        #     self.alpha,
        #     self.beta
        # )
        return self.camera3d_from_device.inv

    def move_by(
        self,
        delta_isocenter: Optional[Vector3D] = None,
        delta_alpha: Optional[float] = None,
        delta_beta: Optional[float] = None,
        delta_gamma: Optional[float] = None,
        degrees: bool = True,
    ):
        """Move CBCT by specified deltas."""
        if delta_isocenter is not None:
            self.isocenter = Point3D(np.array(self.isocenter) + np.array(delta_isocenter))

        if delta_alpha is not None:
            self.alpha += math.radians(delta_alpha) if degrees else delta_alpha
        if delta_beta is not None:
            self.beta += math.radians(delta_beta) if degrees else delta_beta
        if delta_gamma is not None:
            self.gamma += math.radians(delta_gamma) if degrees else delta_gamma
