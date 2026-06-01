#ifndef MAIN_KERNEL_CUH
#define MAIN_KERNEL_CUH 

#include "flux.cuh"
#include "types.h"
#include "movement.cuh"
#include "materials.cuh"
#include "prepNextIter.cuh"
#include "sampleFreePath.cuh"
#include <curand_kernel.h>



__global__ void mainKernel(float* posx, float* posy, float* posz, float* dirx, float* diry, float* dirz, float* step, double* particle_flux, const uint N, double* grid, const uint edgeN, const double voxelSize, curandState* randState, int* active_indices, const uint num_vive, int* alive) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= num_vive) return;

    int real_id = active_indices[id];
    float3 pos = make_float3(posx[real_id], posy[real_id], posz[real_id]);
    float3 dir = make_float3(dirx[real_id], diry[real_id], dirz[real_id]);
    float s = step[real_id];
    float distance = getDistanceToNearestIntersection(pos, dir);
    float min_step = fminf(s, distance);
    double flux_value = flux(pos, pos + dir * min_step, grid, edgeN, voxelSize);
    u_int8_t matID = getMaterialID(pos);


    alive[real_id] = prepNextIter(distance, s, pos, dir, matID, &randState[real_id]);
    posx[real_id] = pos.x;
    posy[real_id] = pos.y;
    posz[real_id] = pos.z;
    dirx[real_id] = dir.x;
    diry[real_id] = dir.y;
    dirz[real_id] = dir.z;
    step[real_id] = s;
    particle_flux[real_id] += flux_value;
}

#endif // MAIN_KERNEL_CUH

