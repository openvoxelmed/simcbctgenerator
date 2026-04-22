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

"""RTK Geometry XML parsing utilities.

This module provides functionality to parse RTK XML geometry files
and create geometry data structures for CBCT projections.
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union
import numpy as np
import xml.etree.ElementTree as ET


@dataclass
class RTKGeometryData:
    """Per-projection geometry data from RTK XML file or config.

    Attributes:
        angles: Gantry angles in degrees for each projection
        offset_x: ProjectionOffsetX per angle in mm
        offset_y: ProjectionOffsetY per angle in mm
        source_to_isocenter: SourceToIsocenterDistance in mm
        source_to_detector: SourceToDetectorDistance in mm
    """
    angles: np.ndarray
    offset_x: np.ndarray
    offset_y: np.ndarray
    source_to_isocenter: float
    source_to_detector: float

    @property
    def num_projections(self) -> int:
        """Number of projections in the geometry."""
        return len(self.angles)

    def get_offsets_at_index(self, idx: int) -> tuple[float, float]:
        """Get (offset_x, offset_y) for projection index.

        Args:
            idx: Projection index

        Returns:
            Tuple of (offset_x, offset_y) in mm
        """
        if idx < 0 or idx >= len(self.offset_x):
            raise IndexError(f"Projection index {idx} out of range [0, {len(self.offset_x)})")
        return float(self.offset_x[idx]), float(self.offset_y[idx])

    def get_offsets_at_angle(self, angle: float, tolerance: float = 0.5) -> tuple[float, float]:
        """Get offsets for nearest angle within tolerance, or interpolate.

        Args:
            angle: Gantry angle in degrees
            tolerance: Maximum angle difference for exact match in degrees

        Returns:
            Tuple of (offset_x, offset_y) in mm
        """
        idx = np.argmin(np.abs(self.angles - angle))
        if np.abs(self.angles[idx] - angle) <= tolerance:
            return self.get_offsets_at_index(idx)
        return self._interpolate_offsets(angle)

    def _interpolate_offsets(self, angle: float) -> tuple[float, float]:
        """Linear interpolation of offsets for given angle.

        Args:
            angle: Gantry angle in degrees

        Returns:
            Tuple of interpolated (offset_x, offset_y) in mm
        """
        # Sort by angle for proper interpolation
        sort_idx = np.argsort(self.angles)
        sorted_angles = self.angles[sort_idx]
        sorted_offset_x = self.offset_x[sort_idx]
        sorted_offset_y = self.offset_y[sort_idx]

        offset_x = np.interp(angle, sorted_angles, sorted_offset_x)
        offset_y = np.interp(angle, sorted_angles, sorted_offset_y)
        return float(offset_x), float(offset_y)


def parse_rtk_geometry_xml(xml_path: Union[str, Path]) -> RTKGeometryData:
    """Parse RTK XML geometry file and extract all geometry parameters.

    Args:
        xml_path: Path to RTK geometry XML file

    Returns:
        RTKGeometryData with all geometry parameters

    Raises:
        FileNotFoundError: If XML file doesn't exist
        ValueError: If XML format is invalid or missing required elements
    """
    xml_path = Path(xml_path)
    if not xml_path.exists():
        raise FileNotFoundError(f"RTK geometry XML file not found: {xml_path}")

    try:
        tree = ET.parse(xml_path)
    except ET.ParseError as e:
        raise ValueError(f"Invalid XML format in {xml_path}: {e}")

    root = tree.getroot()

    # Validate root element
    if root.tag != "RTKThreeDCircularGeometry":
        raise ValueError(
            f"Invalid RTK geometry XML: expected root element 'RTKThreeDCircularGeometry', "
            f"got '{root.tag}'"
        )

    # Extract global parameters
    sid_elem = root.find("SourceToIsocenterDistance")
    sdd_elem = root.find("SourceToDetectorDistance")

    if sid_elem is None or sid_elem.text is None:
        raise ValueError("Missing SourceToIsocenterDistance in RTK geometry XML")
    if sdd_elem is None or sdd_elem.text is None:
        raise ValueError("Missing SourceToDetectorDistance in RTK geometry XML")

    source_to_isocenter = float(sid_elem.text)
    source_to_detector = float(sdd_elem.text)

    # Extract per-projection data
    projections = root.findall("Projection")
    if not projections:
        raise ValueError("No Projection elements found in RTK geometry XML")

    n_proj = len(projections)
    angles = np.zeros(n_proj, dtype=np.float64)
    offset_x = np.zeros(n_proj, dtype=np.float64)
    offset_y = np.zeros(n_proj, dtype=np.float64)

    for i, proj in enumerate(projections):
        # Gantry angle is required
        angle_elem = proj.find("GantryAngle")
        if angle_elem is None or angle_elem.text is None:
            raise ValueError(f"Missing GantryAngle in Projection {i}")
        angles[i] = float(angle_elem.text)

        # Offset X - required
        offset_x_elem = proj.find("ProjectionOffsetX")
        if offset_x_elem is None or offset_x_elem.text is None:
            raise ValueError(f"Missing ProjectionOffsetX in Projection {i}")
        offset_x[i] = float(offset_x_elem.text)

        # Offset Y - optional, defaults to 0.0
        offset_y_elem = proj.find("ProjectionOffsetY")
        if offset_y_elem is not None and offset_y_elem.text is not None:
            offset_y[i] = float(offset_y_elem.text)
        else:
            offset_y[i] = 0.0

    return RTKGeometryData(
        angles=angles,
        offset_x=offset_x,
        offset_y=offset_y,
        source_to_isocenter=source_to_isocenter,
        source_to_detector=source_to_detector
    )


def create_geometry_data(
    angles: np.ndarray,
    detector_offset: Union[float, np.ndarray, None] = None,
    detector_offset_x: Union[float, np.ndarray, None] = None,
    detector_offset_y: Union[float, np.ndarray, None] = None,
    source_to_isocenter: float = 1000.0,
    source_to_detector: float = 1536.0,
    xml_path: Optional[Union[str, Path]] = None
) -> RTKGeometryData:
    """Create geometry data from various input formats.

    Supports multiple input methods with the following priority:
    1. xml_path - load all parameters from RTK XML file
    2. Arrays - use provided per-angle arrays
    3. Single float - broadcast to all angles

    Args:
        angles: Array of gantry angles in degrees (ignored if xml_path provided)
        detector_offset: Legacy single offset value (X direction, Y=0)
        detector_offset_x: X offset - single float or per-angle array
        detector_offset_y: Y offset - single float or per-angle array
        source_to_isocenter: Source to isocenter distance in mm
        source_to_detector: Source to detector distance in mm
        xml_path: Path to RTK XML geometry file (overrides all other parameters)

    Returns:
        RTKGeometryData with geometry parameters
    """
    # If XML path provided, load everything from file
    if xml_path is not None:
        return parse_rtk_geometry_xml(xml_path)

    angles = np.asarray(angles, dtype=np.float64)
    n = len(angles)

    # Handle X offset
    if detector_offset_x is not None:
        if np.isscalar(detector_offset_x):
            ox = np.full(n, detector_offset_x, dtype=np.float64)
        else:
            ox = np.asarray(detector_offset_x, dtype=np.float64)
            if len(ox) != n:
                raise ValueError(
                    f"detector_offset_x length ({len(ox)}) must match angles length ({n})"
                )
    elif detector_offset is not None:
        # Legacy single offset maps to X
        if np.isscalar(detector_offset):
            ox = np.full(n, detector_offset, dtype=np.float64)
        else:
            ox = np.asarray(detector_offset, dtype=np.float64)
            if len(ox) != n:
                raise ValueError(
                    f"detector_offset length ({len(ox)}) must match angles length ({n})"
                )
    else:
        ox = np.zeros(n, dtype=np.float64)

    # Handle Y offset
    if detector_offset_y is not None:
        if np.isscalar(detector_offset_y):
            oy = np.full(n, detector_offset_y, dtype=np.float64)
        else:
            oy = np.asarray(detector_offset_y, dtype=np.float64)
            if len(oy) != n:
                raise ValueError(
                    f"detector_offset_y length ({len(oy)}) must match angles length ({n})"
                )
    else:
        oy = np.zeros(n, dtype=np.float64)

    return RTKGeometryData(
        angles=angles,
        offset_x=ox,
        offset_y=oy,
        source_to_isocenter=source_to_isocenter,
        source_to_detector=source_to_detector
    )
