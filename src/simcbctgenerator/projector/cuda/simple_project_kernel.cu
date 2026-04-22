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
 * Simplified CUDA kernel for ray casting projection.
 *
 * This kernel performs basic ray casting without scatter simulation,
 * material-based attenuation, or spectral processing. It simply
 * integrates density values along rays.
 */

#include <stdio.h>

extern "C" __global__ void simple_project_kernel(
    int out_width,                    // Output image width
    int out_height,                   // Output image height
    float step,                       // Ray marching step size
    float* minPointX,                 // Volume minimum bounds X
    float* minPointY,                 // Volume minimum bounds Y
    float* minPointZ,                 // Volume minimum bounds Z
    float* maxPointX,                 // Volume maximum bounds X
    float* maxPointY,                 // Volume maximum bounds Y
    float* maxPointZ,                 // Volume maximum bounds Z
    float* voxelSizeX,               // Voxel spacing X
    float* voxelSizeY,               // Voxel spacing Y
    float* voxelSizeZ,               // Voxel spacing Z
    float sx,                        // Source position X (world)
    float sy,                        // Source position Y (world)
    float sz,                        // Source position Z (world)
    float* sx_ijk,                   // Source position X (IJK)
    float* sy_ijk,                   // Source position Y (IJK)
    float* sz_ijk,                   // Source position Z (IJK)
    float max_ray_length,            // Maximum ray length (-1 for unlimited)
    float* world_from_index,         // 3x3 transform matrix (flattened) - 9 elements
    float* ijk_from_world,           // 4x3 transform matrix (flattened) - 12 elements
    cudaTextureObject_t volume_tex,  // Volume texture
    float* intensity                 // Output intensity array
) {
    // Calculate pixel coordinates
    int i = blockIdx.x * blockDim.x + threadIdx.x;  // X coordinate
    int j = blockIdx.y * blockDim.y + threadIdx.y;  // Y coordinate

    // Check bounds
    if (i >= out_width || j >= out_height) return;

    // Calculate output index
    int idx = i * out_height + j;
    // int idx = j * out_width + i;

    // Calculate ray direction from world_from_index transform
    // Follow DeepDRR's exact approach: cell-centered sampling + 3x3 matrix
    // world_from_index is 3x3 matrix flattened: [3x3] = 9 elements
    float u = (float)i + 0.5f;  // cell-centered sampling like DeepDRR
    float v = (float)j + 0.5f;

    float ray_x = u * world_from_index[0] + v * world_from_index[1] + world_from_index[2];
    float ray_y = u * world_from_index[3] + v * world_from_index[4] + world_from_index[5];
    float ray_z = u * world_from_index[6] + v * world_from_index[7] + world_from_index[8];

    // Normalize ray direction
    float ray_len = sqrtf(ray_x * ray_x + ray_y * ray_y + ray_z * ray_z);
    if (ray_len > 0) {
        ray_x /= ray_len;
        ray_y /= ray_len;
        ray_z /= ray_len;
    }

    // Initialize ray marching
    float total_intensity = 0.0f;

    // Calculate proper volume entry/exit points like DeepDRR
    // This is critical - we need to start from the volume entry point
    float minAlpha = 0.0f;
    float maxAlpha = (max_ray_length > 0) ? max_ray_length : 1000.0f;  // reasonable default

    // Calculate ray-volume intersection (simplified for single volume)
    // Transform ray direction to IJK space
    float rx_ijk = ijk_from_world[0] * ray_x + ijk_from_world[1] * ray_y + ijk_from_world[2] * ray_z;
    float ry_ijk = ijk_from_world[4] * ray_x + ijk_from_world[5] * ray_y + ijk_from_world[6] * ray_z;
    float rz_ijk = ijk_from_world[8] * ray_x + ijk_from_world[9] * ray_y + ijk_from_world[10] * ray_z;

    // Calculate intersection with volume bounds (like DeepDRR lines 445-476)
    bool intersects_volume = true;
    if (rx_ijk != 0.0f) {
        float reci = 1.0f / rx_ijk;
        float alpha0 = (minPointX[0] - sx_ijk[0]) * reci;
        float alpha1 = (maxPointX[0] - sx_ijk[0]) * reci;
        minAlpha = fmaxf(minAlpha, fminf(alpha0, alpha1));
        maxAlpha = fminf(maxAlpha, fmaxf(alpha0, alpha1));
    } else if (sx_ijk[0] < minPointX[0] || sx_ijk[0] > maxPointX[0]) {
        intersects_volume = false;
    }

    if (ry_ijk != 0.0f && intersects_volume) {
        float reci = 1.0f / ry_ijk;
        float alpha0 = (minPointY[0] - sy_ijk[0]) * reci;
        float alpha1 = (maxPointY[0] - sy_ijk[0]) * reci;
        minAlpha = fmaxf(minAlpha, fminf(alpha0, alpha1));
        maxAlpha = fminf(maxAlpha, fmaxf(alpha0, alpha1));
    } else if (intersects_volume && (sy_ijk[0] < minPointY[0] || sy_ijk[0] > maxPointY[0])) {
        intersects_volume = false;
    }

    if (rz_ijk != 0.0f && intersects_volume) {
        float reci = 1.0f / rz_ijk;
        float alpha0 = (minPointZ[0] - sz_ijk[0]) * reci;
        float alpha1 = (maxPointZ[0] - sz_ijk[0]) * reci;
        minAlpha = fmaxf(minAlpha, fminf(alpha0, alpha1));
        maxAlpha = fminf(maxAlpha, fmaxf(alpha0, alpha1));
    } else if (intersects_volume && (sz_ijk[0] < minPointZ[0] || sz_ijk[0] > maxPointZ[0])) {
        intersects_volume = false;
    }

    // Only ray cast if the ray intersects the volume
    if (!intersects_volume || minAlpha >= maxAlpha) {
        intensity[idx] = 0.0f;
        return;
    }

    // Ray marching from entry to exit point
    int num_steps = (int)ceilf((maxAlpha - minAlpha) / step);
    float alpha = minAlpha;

    for (int t = 0; t < num_steps; t++) {
        // Calculate position in IJK coordinates with 0.5 offset (like DeepDRR)
        float ijk_x = sx_ijk[0] + alpha * rx_ijk + 0.5f;
        float ijk_y = sy_ijk[0] + alpha * ry_ijk + 0.5f;
        float ijk_z = sz_ijk[0] + alpha * rz_ijk + 0.5f;

        // Sample texture with correct axis ordering: texture uses [Z,Y,X] but we have [X,Y,Z]
        // So we need to swap: tex3D(tex, Z, Y, X) = tex3D(tex, ijk_z, ijk_y, ijk_x)
        float density = tex3D<float>(volume_tex, ijk_x, ijk_y, ijk_z);

        // Accumulate density (like DeepDRR line 583)
        total_intensity += density;

        alpha += step;
    }

    // Apply step scaling like DeepDRR (lines 619-621)
    total_intensity *= step;

    // Store the result
    intensity[idx] = total_intensity;
}
