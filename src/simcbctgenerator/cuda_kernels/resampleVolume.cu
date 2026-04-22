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

    __device__ float clip(float x, float a, float b)
    {
        return fmin(fmax(x,a),b);
    }

    //texture<fp_tex_float, 3> tex_volume;
    //texture<float, 3, cudaReadModeElementType> tex_volume2;

    __global__ void resampleVolume(float *volume,
        float *motionfield,
        int sizeDimX,
        int sizeDimY,
        int sizeDimZ,
        int normalizeValues,
        cudaTextureObject_t tex_volume)
    {
        int global_x = blockIdx.x * blockDim.x + threadIdx.x;
        int global_y = blockIdx.y * blockDim.y + threadIdx.y;
        int global_z = blockIdx.z * blockDim.z + threadIdx.z;

        if ((global_x >= sizeDimX) || (global_y >= sizeDimY) || (global_z >= sizeDimZ))
            return;

        int idx = (global_x * sizeDimY * sizeDimZ + global_y * sizeDimZ + global_z);

        int idx_mf = idx * 3; //(x + y * sizeDimX + z  * sizeDimY * sizeDimX) * 3;

        // bicubic interpolation
        //TODO: check if cubictex3d is still correct
        // float xpos = (float)(global_x) + 0.5 + motionfield[idx_mf+2]; //((float)sizeDimX); //motionfield[idx_mf];
        // float ypos = (float)(global_y) + 0.5 + motionfield[idx_mf+1]; //((float)sizeDimY); //motionfield[idx_mf+1];
        // float zpos = (float)(global_z) + 0.5 + motionfield[idx_mf+0]; //((float)sizeDimZ); //motionfield[idx_mf+2];
        // float density = cubicTex3D<float, float>(tex_volume, make_float3(zpos, ypos, xpos));

        float xpos = (float)(global_x) + 0.5 + motionfield[idx_mf+2]; //((float)sizeDimX); //motionfield[idx_mf];
        float ypos = (float)(global_y) + 0.5 + motionfield[idx_mf+1]; //((float)sizeDimY); //motionfield[idx_mf+1];
        float zpos = (float)(global_z) + 0.5 + motionfield[idx_mf+0]; //((float)sizeDimZ); //motionfield[idx_mf+2];
        float density = tex3D<float>(tex_volume, zpos, ypos, xpos);

        //float density = cubicTex3D(tex_volume, xpos, ypos, zpos);
        //float density = cubicTex3DSimple(tex_volume, make_float3(zpos, ypos, xpos));

        /*
        // nearest neighbor interpolation
        float xpos = (float)(x) + motionfield[idx_mf+2];
        float ypos = (float)(y) + motionfield[idx_mf+1];
        float zpos = (float)(z) + motionfield[idx_mf+0];
        float density = tex3D(tex_volume, zpos, ypos, xpos);
        */

        if (normalizeValues)
            density = (clip(density, -1000.f, 3071.f) + 1000.f) / 53220.f;//(clip(density, -1000.f, 3000.f) + 1000.f) / 4000.f;

        volume[idx] = density;
    }
}
