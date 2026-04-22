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

"""Simplified Volume class for CT data handling.

This module contains a minimal Volume implementation that provides
only the essential functionality needed for ray projection.
"""

from __future__ import annotations
from typing import Optional, Tuple
import numpy as np
from .geometry import FrameTransform, frame_transform


class Volume:
    """Simplified volume class for CT data and materials."""

    def __init__(
        self,
        data: np.ndarray,
        spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        anatomical_from_IJK: Optional[FrameTransform] = None,
        world_from_anatomical: Optional[FrameTransform] = None,
    ):
        """Initialize a Volume.

        Args:
            data: 3D CT density data
            spacing: Voxel spacing in mm (x, y, z)
            anatomical_from_IJK: Transform from IJK indices to anatomical coordinates
            world_from_anatomical: Transform from anatomical to world coordinates
        """
        self.data = np.array(data).astype(np.float32)
        self.spacing = np.array(spacing, dtype=np.float32)

        # Set up coordinate transforms
        if anatomical_from_IJK is None:
            # Default: identity with scaling by spacing
            transform = np.eye(4)
            transform[0, 0] = spacing[0]
            transform[1, 1] = spacing[1]
            transform[2, 2] = spacing[2]
            self.anatomical_from_IJK = FrameTransform(transform)
        else:
            self.anatomical_from_IJK = frame_transform(anatomical_from_IJK)

        if world_from_anatomical is None:
            self.world_from_anatomical = FrameTransform.identity()
        else:
            self.world_from_anatomical = frame_transform(world_from_anatomical)


    @property
    def shape(self) -> Tuple[int, int, int]:
        """Get the shape of the volume."""
        return self.data.shape

    @property
    def world_from_ijk(self) -> FrameTransform:
        """Transform from IJK indices to world coordinates."""
        return self.world_from_anatomical @ self.anatomical_from_IJK

    @property
    def ijk_from_world(self) -> FrameTransform:
        """Transform from world coordinates to IJK indices."""
        return self.world_from_ijk.inv

    @property
    def IJK_from_world(self) -> FrameTransform:
        """Alias for ijk_from_world (for compatibility)."""
        return self.ijk_from_world

    def __array__(self) -> np.ndarray:
        """Return the volume data as numpy array."""
        return self.data



    @classmethod
    def from_data(
        cls,
        data: np.ndarray,
        spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        origin: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> "Volume":
        """Create volume from basic data arrays.

        Args:
            data: 3D CT density data
            spacing: Voxel spacing in mm
            origin: Volume origin in world coordinates

        Returns:
            Volume object
        """

        # Create anatomical_from_IJK transform with spacing and origin
        transform = np.eye(4)
        transform[0, 0] = spacing[0]
        transform[1, 1] = spacing[1]
        transform[2, 2] = spacing[2]
        transform[0, 3] = origin[0]
        transform[1, 3] = origin[1]
        transform[2, 3] = origin[2]

        anatomical_from_IJK = FrameTransform(transform)

        return cls(
            data=data,
            spacing=spacing,
            anatomical_from_IJK=anatomical_from_IJK
        )

    def resample_to_spacing(self, new_spacing: Tuple[float, float, float]) -> "Volume":
        """Resample volume to new spacing (simplified implementation)."""
        # For now, return self - full resampling would require scipy interpolation
        # This is a placeholder for future implementation if needed
        return self

    def copy(self) -> "Volume":
        """Create a copy of the volume."""
        return Volume(
            data=self.data.copy(),
            spacing=self.spacing.copy(),
            anatomical_from_IJK=FrameTransform(self.anatomical_from_IJK.data.copy()),
            world_from_anatomical=FrameTransform(self.world_from_anatomical.data.copy())
        )
