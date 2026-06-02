/**
 * @brief Initializes particle state for the simulation.
 *
 * @param posx          Output array of particle x positions.
 * @param posy          Output array of particle y positions.
 * @param posz          Output array of particle z positions.
 * @param dirx          Output array of particle direction x components.
 * @param diry          Output array of particle direction y components.
 * @param dirz          Output array of particle direction z components.
 * @param step          Output array of distances to the next collision.
 * @param particle_flux Output array for per-particle flux accumulation.
 * @param randState     Input/output array of per-particle curandState.
 * @param N             Number of particles.
 * @param seed          Seed for random number generation.
 * @param source        Source region used to sample initial particle positions.
 *
 * Kernel actions:
 *  - Initializes the random state for each particle using curand_init.
 *  - Samples the initial position of each particle inside the source bounds.
 *  - Samples the initial direction of each particle using getRandomDirection.
 *  - Computes the initial material index at the particle position and samples the first free path.
 *  - Stores the updated curand state back to global memory.
 *
 * Note:
 *  - The source region is used directly for position sampling.
 *  - Particle positions are not additionally constrained beyond the source bounds in this kernel.
 */

#ifndef INIT_CUH
#define INIT_CUH

#include <curand_kernel.h>

#include "utils.h"
#include "types.h"
#include "materials.cuh"
#include "sampleFreePath.cuh"


extern __constant__ Material c_materials[10];

__global__ void init(float* posx, float* posy, float* posz, float* dirx, float* diry, float* dirz, float* step, double* particle_flux, curandState* randState, int N, unsigned long long seed, Region source) {
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
    particle_flux[id] = 0.0;
    u_int8_t matID = getMaterialID(pos);
    step[id] = sampleFreePath(c_materials[matID].sigma_t, &localeState);
    randState[id] = localeState; 
}

#endif // INIT_CUH