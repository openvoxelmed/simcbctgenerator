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

extern "C" {

__global__ void binaryCuboidErosion(float *intVolMask, unsigned char *maskOut, int halfCuboidSizeX, int halfCuboidSizeY, int halfCuboidSizeZ, int sizeDimX, int sizeDimY, int sizeDimZ)
{
    int global_x = blockIdx.x * blockDim.x + threadIdx.x;
    int global_y = blockIdx.y * blockDim.y + threadIdx.y;
    int global_z = blockIdx.z * blockDim.z + threadIdx.z;

    if ((global_x >= sizeDimX) || (global_y >= sizeDimY) || (global_z >= sizeDimZ))
        return;

    int x1 = max(0, global_x - halfCuboidSizeX);
    int y1 = max(0, global_y - halfCuboidSizeY);
    int z1 = max(0, global_z - halfCuboidSizeZ);
    int x2 = min(sizeDimX-1, global_x + halfCuboidSizeX) + 1; // +1 for correct integral volume sampling
    int y2 = min(sizeDimY-1, global_y + halfCuboidSizeY) + 1; // +1 for correct integral volume sampling
    int z2 = min(sizeDimZ-1, global_z + halfCuboidSizeZ) + 1; // +1 for correct integral volume sampling

    int sizeIntDimY = sizeDimY + 1;
    int sizeIntDimZ = sizeDimZ + 1;

    float v1 = intVolMask[x1 * sizeIntDimY * sizeIntDimZ + y2 * sizeIntDimZ + z1];
    float v2 = intVolMask[x2 * sizeIntDimY * sizeIntDimZ + y2 * sizeIntDimZ + z1];
    float v3 = intVolMask[x1 * sizeIntDimY * sizeIntDimZ + y2 * sizeIntDimZ + z2];
    float v4 = intVolMask[x2 * sizeIntDimY * sizeIntDimZ + y2 * sizeIntDimZ + z2];
    float v5 = intVolMask[x1 * sizeIntDimY * sizeIntDimZ + y1 * sizeIntDimZ + z1];
    float v6 = intVolMask[x2 * sizeIntDimY * sizeIntDimZ + y1 * sizeIntDimZ + z1];
    float v7 = intVolMask[x1 * sizeIntDimY * sizeIntDimZ + y1 * sizeIntDimZ + z2];
    float v8 = intVolMask[x2 * sizeIntDimY * sizeIntDimZ + y1 * sizeIntDimZ + z2];

    float summedValues = v4 - v3 - v8 - v2 + v7 + v1 + v6 - v5;

    int numPixels = (x2-x1) * (y2-y1) * (z2-z1);

    int idx = (global_x * sizeDimY * sizeDimZ + global_y * sizeDimZ + global_z);
    maskOut[idx] = summedValues >= numPixels;
}

}
