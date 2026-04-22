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

"""Visualization utilities for registration results."""

import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger(__name__)


def save_multiplanar_comparison(ct_orig: Optional[sitk.Image], cbct: sitk.Image, ct_def: sitk.Image,
                                output_path: Path, patient_id: str,
                                region: Optional[str] = None):
    """Save multi-planar comparison images with equal padding for sagittal, coronal, and axial views.

    Args:
        ct_orig: Original CT image. If provided, an extra comparison column is added.
        cbct: CBCT image
        ct_def: Deformed/registered CT image
        output_path: Path to save visualization
        patient_id: Patient identifier
        region: Optional region name for title
    """
    try:

        # Convert to numpy arrays
        ct_orig_array = sitk.GetArrayFromImage(ct_orig) if ct_orig is not None else None
        cbct_array = sitk.GetArrayFromImage(cbct)
        ct_def_array = sitk.GetArrayFromImage(ct_def)

        # Get dimensions from CBCT
        nz, ny, nx = cbct_array.shape

        # Calculate middle slices
        mid_z = nz // 2  # Axial
        mid_y = ny // 2  # Coronal
        mid_x = nx // 2  # Sagittal

        # Extract slices for each orientation
        # Axial slices (Z direction)
        if ct_orig_array is not None:
            ct_orig_axial = ct_orig_array[mid_z, :, :]
        cbct_axial = cbct_array[mid_z, :, :]
        ct_def_axial = ct_def_array[mid_z, :, :]

        # Coronal slices (Y direction)
        if ct_orig_array is not None:
            ct_orig_coronal = ct_orig_array[:, mid_y, :].T
        cbct_coronal = cbct_array[:, mid_y, :].T
        ct_def_coronal = ct_def_array[:, mid_y, :].T

        # Sagittal slices (X direction)
        if ct_orig_array is not None:
            ct_orig_sagittal = ct_orig_array[:, :, mid_x].T
        cbct_sagittal = cbct_array[:, :, mid_x].T
        ct_def_sagittal = ct_def_array[:, :, mid_x].T

        # Find maximum dimensions for padding
        all_slices = []
        if ct_orig_array is not None:
            all_slices.extend([ct_orig_axial, ct_orig_coronal, ct_orig_sagittal])
        all_slices = [
            *all_slices,
            cbct_axial, ct_def_axial,
            cbct_coronal, ct_def_coronal,
            cbct_sagittal, ct_def_sagittal,
        ]

        max_h = max(slice_img.shape[0] for slice_img in all_slices)
        max_w = max(slice_img.shape[1] for slice_img in all_slices)

        def pad_to_size(img, target_h, target_w):
            """Pad image to target size with center alignment."""
            h, w = img.shape
            pad_h = target_h - h
            pad_w = target_w - w

            pad_top = pad_h // 2
            pad_bottom = pad_h - pad_top
            pad_left = pad_w // 2
            pad_right = pad_w - pad_left

            return np.pad(img, ((pad_top, pad_bottom), (pad_left, pad_right)),
                         mode='constant', constant_values=img.min())

        # Pad all slices to the same size
        slices_padded = {}
        orientations = ['axial', 'coronal', 'sagittal']
        images = ['cbct', 'ct_def']
        if ct_orig_array is not None:
            images.insert(0, 'ct_orig')

        for orient in orientations:
            slices_padded[orient] = {}
            for img_type in images:
                slice_data = locals()[f'{img_type}_{orient}']
                slices_padded[orient][img_type] = pad_to_size(slice_data, max_h, max_w)

        # Create the multi-planar comparison plot.
        include_orig = ct_orig_array is not None
        num_columns = 4 if include_orig else 3
        fig = plt.figure(figsize=(16 if include_orig else 12, 12))
        gs = gridspec.GridSpec(3, num_columns, figure=fig, hspace=0.3, wspace=0.2)

        orientation_labels = ['Axial', 'Coronal', 'Sagittal']

        for row, orient in enumerate(orientations):
            col = 0
            if include_orig:
                ax = fig.add_subplot(gs[row, col])
                ax.imshow(slices_padded[orient]['ct_orig'], cmap='gray', vmin=-1000, vmax=1000)
                if row == 0:
                    ax.set_title('Original CT', fontsize=12, fontweight='bold')
                ax.set_ylabel(orientation_labels[row], fontsize=12, fontweight='bold')
                ax.axis('off')
                col += 1

            # CBCT (target)
            ax = fig.add_subplot(gs[row, col])
            ax.imshow(slices_padded[orient]['cbct'], cmap='gray', vmin=-1000, vmax=1000)
            if row == 0:
                ax.set_title('CBCT (Target)', fontsize=12, fontweight='bold')
            if not include_orig:
                ax.set_ylabel(orientation_labels[row], fontsize=12, fontweight='bold')
            ax.axis('off')
            col += 1

            # Registered CT
            ax = fig.add_subplot(gs[row, col])
            ax.imshow(slices_padded[orient]['ct_def'], cmap='gray', vmin=-1000, vmax=1000)
            if row == 0:
                ax.set_title('Registered CT', fontsize=12, fontweight='bold')
            ax.axis('off')
            col += 1

            # Difference image
            ax = fig.add_subplot(gs[row, col])
            diff = slices_padded[orient]['ct_def'] - slices_padded[orient]['cbct']
            im = ax.imshow(diff, cmap='RdBu', vmin=-500, vmax=500)
            if row == 0:
                ax.set_title('Difference (CT - CBCT)', fontsize=12, fontweight='bold')
            ax.axis('off')

            # Add colorbar for difference images
            if row == len(orientations) - 1:  # Only for the last row
                cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                cbar.set_label('HU Difference', rotation=270, labelpad=15)

        # Set title
        title = f'Multi-Planar Registration Results - {patient_id}'
        if region:
            title = f'Multi-Planar Registration Results - {region} - {patient_id}'

        plt.suptitle(title, fontsize=16, fontweight='bold', y=0.95)

        # Save comparison image
        output_path.parent.mkdir(exist_ok=True, parents=True)
        plt.savefig(str(output_path), dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()

        logger.info(f"Saved multi-planar comparison: {output_path}")
        return True

    except Exception as e:
        logger.error(f"Error creating multi-planar comparison: {str(e)}")
        return False


def save_cbct_comparison(simulated_cbct: sitk.Image, real_cbct: sitk.Image,
                         output_path: Path, patient_id: str):
    """Save multi-planar comparison of simulated vs real CBCT.

    Creates a 3x3 figure with:
    - Columns: Simulated CBCT | Real CBCT | Difference
    - Rows: Axial | Coronal | Sagittal views

    Args:
        simulated_cbct: Simulated CBCT image from projection reconstruction
        real_cbct: Real clinical CBCT image
        output_path: Path to save visualization
        patient_id: Patient identifier for title
    """
    try:
        # Convert to numpy arrays
        sim_array = sitk.GetArrayFromImage(simulated_cbct)
        real_array = sitk.GetArrayFromImage(real_cbct)

        # Get dimensions from simulated CBCT
        nz, ny, nx = sim_array.shape

        # Calculate middle slices
        mid_z = nz // 2  # Axial
        mid_y = ny // 2  # Coronal
        mid_x = nx // 2  # Sagittal

        # Extract slices for each orientation
        sim_axial = sim_array[mid_z, :, :]
        real_axial = real_array[mid_z, :, :]

        sim_coronal = sim_array[:, mid_y, :].T
        real_coronal = real_array[:, mid_y, :].T

        sim_sagittal = sim_array[:, :, mid_x].T
        real_sagittal = real_array[:, :, mid_x].T

        # Find maximum dimensions for padding
        all_slices = [
            sim_axial, real_axial,
            sim_coronal, real_coronal,
            sim_sagittal, real_sagittal
        ]

        max_h = max(slice_img.shape[0] for slice_img in all_slices)
        max_w = max(slice_img.shape[1] for slice_img in all_slices)

        def pad_to_size(img, target_h, target_w):
            """Pad image to target size with center alignment."""
            h, w = img.shape
            pad_h = target_h - h
            pad_w = target_w - w

            pad_top = pad_h // 2
            pad_bottom = pad_h - pad_top
            pad_left = pad_w // 2
            pad_right = pad_w - pad_left

            return np.pad(img, ((pad_top, pad_bottom), (pad_left, pad_right)),
                         mode='constant', constant_values=img.min())

        # Pad all slices to the same size
        slices_padded = {}
        orientations = ['axial', 'coronal', 'sagittal']
        images = ['sim', 'real']

        for orient in orientations:
            slices_padded[orient] = {}
            for img_type in images:
                slice_data = locals()[f'{img_type}_{orient}']
                slices_padded[orient][img_type] = pad_to_size(slice_data, max_h, max_w)

        # Compute error metrics
        mask = (real_array > -900) & (sim_array > -900)
        if mask.sum() > 0:
            diff_masked = sim_array[mask] - real_array[mask]
            rmse = np.sqrt(np.mean(diff_masked**2))
            mae = np.mean(np.abs(diff_masked))
        else:
            rmse = 0.0
            mae = 0.0

        # Create the multi-planar comparison plot (3 columns: Simulated, Real, Difference)
        fig = plt.figure(figsize=(12, 12))
        gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.2)

        orientation_labels = ['Axial', 'Coronal', 'Sagittal']

        for row, orient in enumerate(orientations):
            # Simulated CBCT
            ax = fig.add_subplot(gs[row, 0])
            ax.imshow(slices_padded[orient]['sim'], cmap='gray', vmin=np.quantile(slices_padded[orient]['sim'], 0.01), vmax=np.quantile(slices_padded[orient]['sim'], 0.99))
            if row == 0:
                ax.set_title('Simulated CBCT', fontsize=12, fontweight='bold')
            ax.set_ylabel(orientation_labels[row], fontsize=12, fontweight='bold')
            ax.axis('off')

            # Real CBCT
            ax = fig.add_subplot(gs[row, 1])
            ax.imshow(slices_padded[orient]['real'], cmap='gray', vmin=np.quantile(slices_padded[orient]['real'], 0.01), vmax=np.quantile(slices_padded[orient]['real'], 0.99))
            if row == 0:
                ax.set_title('Real CBCT', fontsize=12, fontweight='bold')
            ax.axis('off')

            # Difference image
            ax = fig.add_subplot(gs[row, 2])
            diff = slices_padded[orient]['sim'] - slices_padded[orient]['real']
            im = ax.imshow(diff, cmap='RdBu', vmin=-300, vmax=300)
            if row == 0:
                ax.set_title('Difference (Sim - Real)', fontsize=12, fontweight='bold')
            ax.axis('off')

            # Add colorbar for difference images
            if row == len(orientations) - 1:
                cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                cbar.set_label('HU Difference', rotation=270, labelpad=15)

        # Set title with metrics
        title = f'CBCT Comparison - {patient_id}\nRMSE: {rmse:.1f} HU, MAE: {mae:.1f} HU'
        plt.suptitle(title, fontsize=14, fontweight='bold', y=0.98)

        # Save comparison image
        output_path.parent.mkdir(exist_ok=True, parents=True)
        plt.savefig(str(output_path), dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()

        logger.info(f"Saved CBCT comparison: {output_path}")
        logger.info(f"Metrics - RMSE: {rmse:.1f} HU, MAE: {mae:.1f} HU")
        return True

    except Exception as e:
        logger.error(f"Error creating CBCT comparison: {str(e)}")
        return False
