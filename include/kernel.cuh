#ifndef MAIN_KERNEL_CUH
#define MAIN_KERNEL_CUH 

#include "flux.cuh"
#include "types.h"
#include "movement.cuh"
#include "materials.cuh"
#include "prepNextIter.cuh"
#include "sampleFreePath.cuh"
#include <curand_kernel.h>



__global__ void mainKernel(float* posx, float* posy, float* posz, float* dirx, float* diry, float* dirz, float* step, const uint N, double* grid, const uint edgeN, const double voxelSize, curandState* randState) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= N) return;

    float3 pos = make_float3(posx[id], posy[id], posz[id]);
    float3 dir = make_float3(dirx[id], diry[id], dirz[id]);
    float s = step[id];
    float distance = getDistanceToNearestIntersection(pos, dir);
    float min_step = fminf(s, distance);
    flux(pos, pos + dir * min_step, grid, edgeN, voxelSize);
    u_int8_t matID = getMaterialID(pos);
    prepNextIter(pos + dir * distance, s, pos, dir, matID, &randState[id]);
    posx[id] = pos.x;
    posy[id] = pos.y;
    posz[id] = pos.z;
    dirx[id] = dir.x;
    diry[id] = dir.y;
    dirz[id] = dir.z;
    step[id] = s;
}

#endif // MAIN_KERNEL_CUH