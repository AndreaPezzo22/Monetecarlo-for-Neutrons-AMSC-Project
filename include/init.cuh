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

__global__ void init(float* posx, float* posy, float* posz, float* dirx, float* diry, float* dirz, float* step, curandState* randState, int N, unsigned long long seed) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= N) return;

    // Initialize the random state for each particle
    curand_init(seed, id, 0, &randState[id]);

    curandState *state = &randState[id];
    // Set initial random positions (for example, within a unit cube)
    float3 pos = make_float3(curand_uniform(state), curand_uniform(state), curand_uniform(state));
    posx[id] = pos.x;
    posy[id] = pos.y;
    posz[id] = pos.z;

    float3 dir = getRandomDirection(state);
    dirx[id] = dir.x;
    diry[id] = dir.y;
    dirz[id] = dir.z;

    u_int8_t matID = getMaterialID(pos);
    step[id] = sampleFreePath(c_materials[matID].sigma_t, &randState[id]);

}

#endif // INIT_CUH