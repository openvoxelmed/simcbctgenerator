/******************************************************************************
 * simcbctgenerator
 *
 * Copyright 2025 Lukas Zimmermann and Michael Rauter
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *****************************************************************************/

/**
 * Polychromatic CUDA kernel for ray casting projection with spectral
 * beam-hardening simulation.
 *
 * Extends simple_project_kernel with:
 *   - Threshold-based fuzzy material decomposition (air / water / bone)
 *     following the SinoSynth approach.
 *   - Two-integral spectral decomposition: accumulate L_water and L_bone
 *     during a SINGLE ray traversal, then spectral integration after
 *     the loop in O(num_bins).
 *
 * Volume texture stores density values (linear attenuation in mm^-1),
 * identical to the format used by simple_project_kernel.  The HU -> mu
 * conversion is performed on the host BEFORE uploading the volume.
 *
 * Physics (per ray):
 *   L_water = sum( density * f_water ) * step
 *   L_bone  = sum( density * f_bone  ) * step
 *
 *   I = sum_k  spectrum[k] * exp( -L_water * ratio_water[k]
 *                                 -L_bone  * ratio_bone[k] )
 *
 *   proj = -ln(I)      (spectrum is pre-normalised to sum = 1)
 *
 * Self-consistency: at the reference energy E0, ratio = 1 for both
 * materials, so proj = L_water + L_bone = sum(density) * step, which
 * is identical to simple_project_kernel output.
 */

#include <stdio.h>

extern "C" __global__ void polychromatic_project_kernel(
    /* ---- geometry params (identical to simple_project_kernel) ---- */
    int out_width,                    // Output image width
    int out_height,                   // Output image height
    float step,                       // Ray marching step size (mm)
    float* minPointX,                 // Volume minimum bounds X (IJK)
    float* minPointY,
    float* minPointZ,
    float* maxPointX,                 // Volume maximum bounds X (IJK)
    float* maxPointY,
    float* maxPointZ,
    float* voxelSizeX,               // Voxel spacing X (mm)
    float* voxelSizeY,
    float* voxelSizeZ,
    float sx,                        // Source position X (world, mm)
    float sy,
    float sz,
    float* sx_ijk,                   // Source position X (IJK)
    float* sy_ijk,
    float* sz_ijk,
    float max_ray_length,            // Maximum ray length (-1 unlimited)
    float* world_from_index,         // 3x3 flattened (9 elements)
    float* ijk_from_world,           // 4x3 flattened (12 elements)
    cudaTextureObject_t volume_tex,  // Volume texture (density = mu in mm^-1)
    float* intensity,                // Output array (one float per pixel)
    /* ---- polychromatic parameters ---- */
    float* d_spectrum,               // [num_bins] normalised spectrum (sum=1)
    float* d_ratio_water,            // [num_bins] mu_water(Ek)/mu_water(E0)
    float* d_ratio_bone,             // [num_bins] mu_bone(Ek)/mu_bone(E0)
    int    num_bins,                 // Number of energy bins
    float  T1,                       // Lower threshold (density units)
    float  T2                        // Upper threshold (density units)
) {
    /* ================================================================
     * 1.  Pixel coordinates & bounds check
     * ================================================================ */
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= out_width || j >= out_height) return;

    int idx = i * out_height + j;

    /* ================================================================
     * 2.  Ray direction (cell-centred, world-normalised)
     * ================================================================ */
    float u = (float)i + 0.5f;
    float v = (float)j + 0.5f;

    float ray_x = u * world_from_index[0] + v * world_from_index[1] + world_from_index[2];
    float ray_y = u * world_from_index[3] + v * world_from_index[4] + world_from_index[5];
    float ray_z = u * world_from_index[6] + v * world_from_index[7] + world_from_index[8];

    float ray_len = sqrtf(ray_x * ray_x + ray_y * ray_y + ray_z * ray_z);
    if (ray_len > 0.0f) {
        ray_x /= ray_len;
        ray_y /= ray_len;
        ray_z /= ray_len;
    }

    /* ================================================================
     * 3.  Ray-volume intersection (slab test)
     * ================================================================ */
    float minAlpha = 0.0f;
    float maxAlpha = (max_ray_length > 0.0f) ? max_ray_length : 1000.0f;

    float rx_ijk = ijk_from_world[0]  * ray_x + ijk_from_world[1]  * ray_y + ijk_from_world[2]  * ray_z;
    float ry_ijk = ijk_from_world[4]  * ray_x + ijk_from_world[5]  * ray_y + ijk_from_world[6]  * ray_z;
    float rz_ijk = ijk_from_world[8]  * ray_x + ijk_from_world[9]  * ray_y + ijk_from_world[10] * ray_z;

    bool intersects_volume = true;

    if (rx_ijk != 0.0f) {
        float reci = 1.0f / rx_ijk;
        float a0 = (minPointX[0] - sx_ijk[0]) * reci;
        float a1 = (maxPointX[0] - sx_ijk[0]) * reci;
        minAlpha = fmaxf(minAlpha, fminf(a0, a1));
        maxAlpha = fminf(maxAlpha, fmaxf(a0, a1));
    } else if (sx_ijk[0] < minPointX[0] || sx_ijk[0] > maxPointX[0]) {
        intersects_volume = false;
    }

    if (ry_ijk != 0.0f && intersects_volume) {
        float reci = 1.0f / ry_ijk;
        float a0 = (minPointY[0] - sy_ijk[0]) * reci;
        float a1 = (maxPointY[0] - sy_ijk[0]) * reci;
        minAlpha = fmaxf(minAlpha, fminf(a0, a1));
        maxAlpha = fminf(maxAlpha, fmaxf(a0, a1));
    } else if (intersects_volume && (sy_ijk[0] < minPointY[0] || sy_ijk[0] > maxPointY[0])) {
        intersects_volume = false;
    }

    if (rz_ijk != 0.0f && intersects_volume) {
        float reci = 1.0f / rz_ijk;
        float a0 = (minPointZ[0] - sz_ijk[0]) * reci;
        float a1 = (maxPointZ[0] - sz_ijk[0]) * reci;
        minAlpha = fmaxf(minAlpha, fminf(a0, a1));
        maxAlpha = fminf(maxAlpha, fmaxf(a0, a1));
    } else if (intersects_volume && (sz_ijk[0] < minPointZ[0] || sz_ijk[0] > maxPointZ[0])) {
        intersects_volume = false;
    }

    if (!intersects_volume || minAlpha >= maxAlpha) {
        intensity[idx] = 0.0f;
        return;
    }

    /* ================================================================
     * 4.  Ray marching – two-integral material decomposition
     *
     *     density is in mm^-1 (same as simple_project_kernel).
     *     T1, T2 are in the same density units (converted on host).
     *     Air voxels (density <= 0) contribute nothing.
     * ================================================================ */
    float L_water = 0.0f;
    float L_bone  = 0.0f;

    const float inv_T_range = (T2 > T1) ? (1.0f / (T2 - T1)) : 0.0f;

    int   num_steps = (int)ceilf((maxAlpha - minAlpha) / step);
    float alpha     = minAlpha;

    for (int t = 0; t < num_steps; t++) {
        float ijk_x = sx_ijk[0] + alpha * rx_ijk + 0.5f;
        float ijk_y = sy_ijk[0] + alpha * ry_ijk + 0.5f;
        float ijk_z = sz_ijk[0] + alpha * rz_ijk + 0.5f;

        float density = tex3D<float>(volume_tex, ijk_x, ijk_y, ijk_z);
        density = fmaxf(density, 0.0f);   /* safety: air is zero */

        /* fuzzy threshold decomposition */
        float f_bone;
        if (density <= T1) {
            f_bone = 0.0f;
        } else if (density >= T2) {
            f_bone = 1.0f;
        } else {
            f_bone = (density - T1) * inv_T_range;
        }

        L_water += density * (1.0f - f_bone);
        L_bone  += density * f_bone;

        alpha += step;
    }

    /* step scaling (identical to simple_project_kernel) */
    L_water *= step;
    L_bone  *= step;

    /* ================================================================
     * 5.  Spectral integration (O(num_bins), once per pixel)
     *
     *     At E0 ratio=1 => proj = L_water+L_bone = monochromatic. ✓
     * ================================================================ */
    float poly_sum = 0.0f;

    for (int k = 0; k < num_bins; k++) {
        float atten = L_water * d_ratio_water[k]
                    + L_bone  * d_ratio_bone[k];
        poly_sum += d_spectrum[k] * expf(-atten);
    }

    intensity[idx] = -logf(fmaxf(poly_sum, 1e-10f));
}
