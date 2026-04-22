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

__device__ float computeVoxelSum(float *intVol, int sizeIntDimY, int sizeIntDimZ, int x1, int y1, int z1, int x2, int y2, int z2)
{
    float v1 = intVol[x1 * sizeIntDimY * sizeIntDimZ + y2 * sizeIntDimZ + z1];
    float v2 = intVol[x2 * sizeIntDimY * sizeIntDimZ + y2 * sizeIntDimZ + z1];
    float v3 = intVol[x1 * sizeIntDimY * sizeIntDimZ + y2 * sizeIntDimZ + z2];
    float v4 = intVol[x2 * sizeIntDimY * sizeIntDimZ + y2 * sizeIntDimZ + z2];
    float v5 = intVol[x1 * sizeIntDimY * sizeIntDimZ + y1 * sizeIntDimZ + z1];
    float v6 = intVol[x2 * sizeIntDimY * sizeIntDimZ + y1 * sizeIntDimZ + z1];
    float v7 = intVol[x1 * sizeIntDimY * sizeIntDimZ + y1 * sizeIntDimZ + z2];
    float v8 = intVol[x2 * sizeIntDimY * sizeIntDimZ + y1 * sizeIntDimZ + z2];
    return(v4 - v3 - v8 - v2 + v7 + v1 + v6 - v5);
}

__global__ void motionFieldPropagation(float *intVolCompX,
                                       float *intVolCompY,
                                       float *intVolCompZ,
                                       float *intVolValidGradOcc,
                                       float *distance,
                                       float *outCompX,
                                       float *outCompY,
                                       float *outCompZ,
                                       int sizeDimX,
                                       int sizeDimY,
                                       int sizeDimZ,
                                       float spacingX,
                                       float spacingY,
                                       float spacingZ)
{
    int global_x = blockIdx.x * blockDim.x + threadIdx.x;
    int global_y = blockIdx.y * blockDim.y + threadIdx.y;
    int global_z = blockIdx.z * blockDim.z + threadIdx.z;

    if ((global_x >= sizeDimX) || (global_y >= sizeDimY) || (global_z >= sizeDimZ))
        return;

    int idx = (global_x * sizeDimY * sizeDimZ + global_y * sizeDimZ + global_z);
    int cuboidExtentX = round(distance[idx] / spacingX + 1);
    int cuboidExtentY = round(distance[idx] / spacingY + 1);
    int cuboidExtentZ = round(distance[idx] / spacingZ + 1);

    int x1 = max(0, global_x - cuboidExtentX);
    int y1 = max(0, global_y - cuboidExtentY);
    int z1 = max(0, global_z - cuboidExtentZ);
    int x2 = min(sizeDimX-1, global_x + cuboidExtentX) + 1; // +1 for correct integral volume sampling
    int y2 = min(sizeDimY-1, global_y + cuboidExtentY) + 1; // +1 for correct integral volume sampling
    int z2 = min(sizeDimZ-1, global_z + cuboidExtentZ) + 1; // +1 for correct integral volume sampling

    float gradXsumOverCuboid = computeVoxelSum(intVolCompX, sizeDimY + 1, sizeDimZ + 1, x1, y1, z1, x2, y2, z2);
    float gradYsumOverCuboid = computeVoxelSum(intVolCompY, sizeDimY + 1, sizeDimZ + 1, x1, y1, z1, x2, y2, z2);
    float gradZsumOverCuboid = computeVoxelSum(intVolCompZ, sizeDimY + 1, sizeDimZ + 1, x1, y1, z1, x2, y2, z2);
    float gradOccSumOverCuboid = computeVoxelSum(intVolValidGradOcc, sizeDimY + 1, sizeDimZ + 1, x1, y1, z1, x2, y2, z2);
    gradOccSumOverCuboid += 1e-7; // division by zero guard (should always only be 0/0.00001 anyway -> 0)

    outCompX[idx] = gradXsumOverCuboid / gradOccSumOverCuboid;
    outCompY[idx] = gradYsumOverCuboid / gradOccSumOverCuboid;
    outCompZ[idx] = gradZsumOverCuboid / gradOccSumOverCuboid;
}

}
