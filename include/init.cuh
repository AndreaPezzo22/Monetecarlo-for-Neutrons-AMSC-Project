// This file defines the kernel for setting up the simulation system (read kernel actions for more informations)
//
// Parameters:
//  - posx, posy, posz: Arrays of length N containing the x, y, z coordinates of the N particles.
//  - dirx, diry, dirz: Arrays of length N containing the x, y, z components of the direction vectors for the N particles.
//  - randState: Array of length N containing the curandState for each particle, used for random number generation. 
//  - N: The number of particles in the simulation.
//  - seed: The seed for the random number generator.
//
// Kernel actions:
//  - Initializes the random state for each particle using curand_init.
//  - Sets the initial random positions of the particles inside the simulation domain.
//  - Sets the initial random direction vectors for the particles.
//  - TODO: restrain positions of particles according to given constraints passed as parameter.

#ifndef INIT_CUH
#define INIT_CUH

#include <curand_kernel.h>

#include "utils.h"
#include "types.h"
#include "materials.cuh"
#include "sampleFreePath.cuh"


extern __constant__ Material c_materials[10];

__global__ void init(float* posx, float* posy, float* posz, float* dirx, float* diry, float* dirz, float* step, curandState* randState, int N, unsigned long long seed, Region source) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= N) return;

    // Initialize the random state for each particle
    curand_init(seed, id, 0, &randState[id]);

    // Set local copy of the random state to avoid multiple accesses to global memory
    curandState localeState = randState[id];

    // Set initial random positions inside source bounds 
    float px = source.min_x + curand_uniform(&localeState) * (source.max_x - source.min_x);
    float py = source.min_y + curand_uniform(&localeState) * (source.max_y - source.min_y);
    float pz = source.min_z + curand_uniform(&localeState) * (source.max_z - source.min_z);

    posx[id] = px;
    posy[id] = py;
    posz[id] = pz;

    float3 pos = make_float3(px, py, pz);
    float3 dir = getRandomDirection(&localeState);

    dirx[id] = dir.x;
    diry[id] = dir.y;
    dirz[id] = dir.z;

    u_int8_t matID = getMaterialID(pos);
    step[id] = sampleFreePath(c_materials[matID].sigma_t, &localeState);
    randState[id] = localeState; 
}

#endif // INIT_CUH