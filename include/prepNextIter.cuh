// Prepare the next iteration of the particle transport simulation, computing new random
// direction if a reaction occurs, handling the particle's behaviour at boundaries, and
// updating the material in case of a change of material.
//
// Parameters:
//  - intersectionPoint: closest intersection point for the particle with the geometry
//  - s: random step size sampled previously for the particle
//  - r: current particle position
//  - d: particle direction (normalized vector)
//  - material: current material index of the particle
//  - state: random state for generating new random numbers
//
// Preconditions:
//  - r is inside the unit cube [0,1]^3.
//
// Behavior:
// If the closest intersection is closer than the sampled step size, we move the
// particle to the intersection point and either reflect at the boundary or update
// the material. Otherwise we advance by the sampled step and choose a new random
// direction.

#ifndef PREP_NEXT_ITER_CUH
#define PREP_NEXT_ITER_CUH

#include "helper_math.h"
#include "sampleFreePath.cuh"
#include "utils.h"
#include "materials.cuh"

extern __constant__ Material c_materials[10];

const int BOUNDARY_TYPE_REFLECTIVE = 0;

inline __device__ bool prepNextIter(const float distanceToIntersection,
                                   float &step,
                                   float3 &particlePos,
                                   float3 &dir,
                                   u_int8_t &material,
                                   curandState *state) {
    const float epsilon = 1e-6f;

    if (distanceToIntersection <= step) {
        particlePos = particlePos + dir * distanceToIntersection; // Move to the intersection point

        const float3 nextPos = particlePos + dir * epsilon;

        const bool hitX = nextPos.x < 0.0f || nextPos.x > 1.0f;
        const bool hitY = nextPos.y < 0.0f || nextPos.y > 1.0f;
        const bool hitZ = nextPos.z < 0.0f || nextPos.z > 1.0f;

        if (hitX || hitY || hitZ) {
            if(BOUNDARY_TYPE_REFLECTIVE){
                particlePos -= dir * epsilon; // Move back slightly to avoid sticking to the boundary
                if (hitX) dir.x = -dir.x;
                if (hitY) dir.y = -dir.y;
                if (hitZ) dir.z = -dir.z;
                step -= distanceToIntersection;
            } else return false; 

        } else {
            // Case 2: particle is not at a boundary but is changing material.
            // We update the material.
            u_int8_t next_mat = getMaterialID(nextPos);
            particlePos = nextPos; // To avoid calculating intersection while being exactly at the boundary between two materials
            if (material == next_mat) {
                step -= (distanceToIntersection + epsilon); 
                // printf("Material boundary at (%f, %f, %f), same material: %d, remaining step: %.50f \n", particlePos.x, particlePos.y, particlePos.z, material, step);
                // if(step <= 0){
                //     printf("Step < 0 : %f\n", step);
                // }; // To avoid negative step due to floating point precision issues
            } else {
                material = next_mat;
                step = sampleFreePath(c_materials[material].sigma_t, state);
                // printf("Material change at (%f, %f, %f), new material: %d\n", particlePos.x, particlePos.y, particlePos.z, material);
            }
        }

        return true;

        // TODO: handle absorption if boundaries are absorbing
    } else {
        float xi = curand_uniform(state);
        float p_abs = c_materials[material].sigma_a / c_materials[material].sigma_t;
        // printf("xi: %f, p_abs: %f\n", xi, p_abs);
        if (xi < p_abs) return false; // Particle is absorbed 
    
        particlePos = particlePos + dir * step;
        step = sampleFreePath(c_materials[material].sigma_t, state);
        dir = getRandomDirection(state);
        return true;
    }
}
#endif // PREP_NEXT_ITER_CUH
