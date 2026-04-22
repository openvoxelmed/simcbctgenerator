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

"""Field-of-view helpers shared by pipeline and API code."""

from __future__ import annotations

import logging

import numpy as np
import SimpleITK as sitk

logger = logging.getLogger(__name__)


def find_fov_center(x_coords: np.ndarray, y_coords: np.ndarray, image_shape: tuple, k: int = 100) -> tuple:
    """Find FOV center by minimizing variance of distances to k furthest points."""
    from scipy.optimize import minimize

    nx, ny = image_shape
    cx_init = np.mean(x_coords)
    cy_init = np.mean(y_coords)

    def objective(center):
        cx, cy = center
        distances = np.sqrt((x_coords - cx)**2 + (y_coords - cy)**2)
        top_k_distances = np.partition(distances, -k)[-k:]
        return np.var(top_k_distances)

    margin = 0.1
    bounds = [
        (nx * margin, nx * (1 - margin)),
        (ny * margin, ny * (1 - margin))
    ]

    result = minimize(objective, [cx_init, cy_init], method='L-BFGS-B', bounds=bounds)
    return result.x[0], result.x[1]


def create_circular_mask_per_slice(mask_image: sitk.Image, k: int = 100) -> sitk.Image:
    """Create circular mask per slice based on maximum radius with mask values."""
    mask_array = sitk.GetArrayFromImage(mask_image)
    circular_mask = np.zeros_like(mask_array, dtype=np.uint8)

    nz, ny, nx = mask_array.shape
    centers = []
    slice_data = []

    for z in range(nz):
        slice_mask = mask_array[z]
        y_coords, x_coords = np.where(slice_mask > 0)

        if len(x_coords) == 0:
            continue

        slice_data.append((z, x_coords, y_coords))
        k_use = min(k, len(x_coords))
        cx, cy = find_fov_center(x_coords, y_coords, image_shape=(nx, ny), k=k_use)
        centers.append([cx, cy])

    if len(centers) == 0:
        logger.warning("No valid slices found in mask")
        return mask_image

    centers = np.array(centers)
    cx_avg = np.mean(centers[:, 0])
    cy_avg = np.mean(centers[:, 1])

    logger.info(f"Computed FOV center: ({cx_avg:.2f}, {cy_avg:.2f}) from {len(centers)} slices")

    spacing = mask_image.GetSpacing()
    max_radius_mm = 205.0
    max_radius_pixels_x = max_radius_mm / spacing[0]
    max_radius_pixels_y = max_radius_mm / spacing[1]
    max_radius_constraint = min(max_radius_pixels_x, max_radius_pixels_y)

    logger.info(f"Maximum radius constraint: {max_radius_mm}mm = {max_radius_constraint:.2f} pixels")

    for z, x_coords, y_coords in slice_data:
        x_centered = x_coords - cx_avg
        y_centered = y_coords - cy_avg
        radii = np.sqrt(x_centered**2 + y_centered**2)
        max_radius = np.median(radii)
        max_radius = min(max_radius, max_radius_constraint)

        y_grid, x_grid = np.ogrid[:ny, :nx]
        x_grid_centered = x_grid - cx_avg
        y_grid_centered = y_grid - cy_avg
        distance_from_center = np.sqrt(x_grid_centered**2 + y_grid_centered**2)
        circular_mask[z] = (distance_from_center <= max_radius).astype(np.uint8)

    circular_mask_image = sitk.GetImageFromArray(circular_mask)
    circular_mask_image.CopyInformation(mask_image)
    circular_mask_image = sitk.Cast(circular_mask_image, sitk.sitkUInt8)

    return circular_mask_image
