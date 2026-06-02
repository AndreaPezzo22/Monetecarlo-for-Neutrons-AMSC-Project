/**
 * @file sampleFreePath.cuh
 * @brief Samples a free path for a particle based on the material total cross-section.
 *
 * @param sigma_t Total cross-section of the current material.
 * @param state   CURAND state used for random number generation.
 *
 * @return Sampled free path distance.
 *
 * @pre sigma_t must be a valid float representing the total material cross-section.
 *
 * @details
 *  - If sigma_t <= 0, the function returns `CUDART_INF_F` to indicate an infinite path.
 *  - Otherwise, it samples a random distance from an exponential distribution using
 *    `-log(U) / sigma_t` where U is a uniform random number in (0,1).
 */

#ifndef STEP_H
#define STEP_H

#include "utils.h"

__device__ inline float sampleFreePath(float sigma_t, curandState *state) {
    if (sigma_t <= 0.0f) {
//	 printf("Sigma_t = 0");
	 return CUDART_INF_F; // Se sigma_t è zero o negativo, il passo è infinito (non avviene alcuna interazione)
    }
    return -logf(getRandomFloat(state, 0.0f, 1.0f)) / sigma_t;
}

#endif // STEP_H
