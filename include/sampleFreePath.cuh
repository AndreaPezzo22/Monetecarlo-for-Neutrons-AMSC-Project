#ifndef STEP_H
#define STEP_H

// Set the step size for a particle based on the total cross-section of the material it is currently in.
//
// Parameters:
// - sigma_t: the total cross-section of the material (sigma_t) where the particle is located.
// - state: the curandState used for random number generation.
// - CUDA_INF_F: a constant representing infinity in CUDA, used to indicate that if sigma_t is zero or negative, the step is infinite (no interaction occurs).
// 
// Included Functions:
// - getRandomFloat: a helper function to generate a random float in a specified range using curandState.
//
// Preconditions:
// - sigma_t must be a valid float value representing the total cross-section.
//
// Behavior:
// The function samples a random free path for the particle based on an exponential distribution, 
// which is commonly used in Monte Carlo simulations of particle transport. If sigma_t is zero 
// or negative, it returns infinity, indicating that the particle will not interact and can travel indefinitely.

// We must be sure that when we call the function we handle correctly the 
// CUDART_INF_F value

#include "utils.h"

__device__ inline float sampleFreePath(float sigma_t, curandState *state) {
    if (sigma_t <= 0.0f) {
//	 printf("Sigma_t = 0");
	 return CUDART_INF_F; // Se sigma_t è zero o negativo, il passo è infinito (non avviene alcuna interazione)
    }
    return -logf(getRandomFloat(state, 0.0f, 1.0f)) / sigma_t;
}

#endif // STEP_H
