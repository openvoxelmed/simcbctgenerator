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

extern "C"{
__global__ void scaleMotionField(float *motionfield,
    float *motionfieldScaled,
    int sizeDimX,
    int sizeDimY,
    int sizeDimZ,
    float scaleFactor)
{
    int global_x = blockIdx.x * blockDim.x + threadIdx.x;
    int global_y = blockIdx.y * blockDim.y + threadIdx.y;
    int global_z = blockIdx.z * blockDim.z + threadIdx.z;

    if ((global_x >= sizeDimX) || (global_y >= sizeDimY) || (global_z >= sizeDimZ))
        return;

    //int idx = (x + y * sizeDimX + z  * sizeDimY * sizeDimX) * 3;
    int idx = (global_x  * sizeDimY * sizeDimZ + global_y * sizeDimZ + global_z) * 3;
    motionfieldScaled[idx] = motionfield[idx] * scaleFactor;
    motionfieldScaled[idx+1] = motionfield[idx+1] * scaleFactor;
    motionfieldScaled[idx+2] = motionfield[idx+2] * scaleFactor;
}
}
