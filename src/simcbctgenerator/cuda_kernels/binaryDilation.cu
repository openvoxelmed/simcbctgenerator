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

__global__ void binaryDilation(unsigned char *mask, unsigned char *maskOut, unsigned char *kernel, int sizeDimX, int sizeDimY, int sizeDimZ)
{
    extern __shared__ unsigned char maskBlock[];
    __shared__ unsigned char kernelCoefficients[27];

    int processSizeX = blockDim.x - 2;
    int processSizeY = blockDim.y - 2;
    int processSizeZ = blockDim.z - 2;

    int xb = blockIdx.x * processSizeX + threadIdx.x;
    int yb = blockIdx.y * processSizeY + threadIdx.y;
    int zb = blockIdx.z * processSizeZ + threadIdx.z;
    int x = min(sizeDimX-1,max(0,xb-1));
    int y = min(sizeDimY-1,max(0,yb-1));
    int z = min(sizeDimZ-1,max(0,zb-1));
    int idxMask = (x * sizeDimY * sizeDimZ + y * sizeDimZ + z);
    int idxBlock = threadIdx.x * blockDim.y * blockDim.z + threadIdx.y * blockDim.z + threadIdx.z;
    maskBlock[idxBlock] = mask[idxMask];

    if ((threadIdx.x<3) && (threadIdx.y<3) && (threadIdx.z<3))
    {
        int idxKernel = threadIdx.x * 9 + threadIdx.y * 3 + threadIdx.z;
        kernelCoefficients[idxKernel] = kernel[idxKernel];
    }

    __syncthreads();

    if ((threadIdx.x <= 0) || (threadIdx.x>=blockDim.x-1) || (threadIdx.y <= 0) || (threadIdx.y>=blockDim.y-1) || (threadIdx.z <= 0) || (threadIdx.z>=blockDim.z-1))
        return;

     x = threadIdx.x-1;
     y = threadIdx.y-1;
     z = threadIdx.z-1;

    int global_x = blockIdx.x * processSizeX + x;
    int global_y = blockIdx.y * processSizeY + y;
    int global_z = blockIdx.z * processSizeZ + z;

    if ((global_x>=sizeDimX) ||
        (global_y>=sizeDimY) ||
        (global_z>=sizeDimZ))
        return;

    int idx = (global_x * sizeDimY * sizeDimZ + global_y * sizeDimZ + global_z);

    int count = 0;
    for (int ix = 0; ix<3; ix++)
    {
        for (int iy = 0; iy<3; iy++)
        {
            for (int iz = 0; iz<3; iz++)
            {
                int idxMask = ((x+ix) * blockDim.y * blockDim.z + (y+iy) * blockDim.z + (z+iz));
                count += maskBlock[idxMask] * kernelCoefficients[ix*9 + iy*3 + iz];
            }
        }
    }
    unsigned char binaryOutputValue = (unsigned char)(count>0);
    maskOut[idx] = binaryOutputValue;
}

}
