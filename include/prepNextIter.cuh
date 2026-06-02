/**
 * @file prepNextIter.cuh
 * @brief Prepares the next transport iteration for a particle.
 *
 * This module updates particle position, direction, material state, and step size
 * after a movement step or geometry intersection.
 *
 * @details
 *  - If the particle intersects geometry before its sampled step, it is moved to the
 *    intersection and either reflected or transitioned into a new material.
 *  - If the particle travels the full sampled step without intersecting geometry, it
 *    advances and may be absorbed or continue with a new sampled path.
 */

#ifndef PREP_NEXT_ITER_CUH
#define PREP_NEXT_ITER_CUH

#include "helper_math.h"
#include "sampleFreePath.cuh"
#include "utils.h"
#include "materials.cuh"

extern __constant__ Material c_materials[10];

const int BOUNDARY_TYPE_REFLECTIVE = 0;

/**
 * @brief Advances particle state after a movement step or geometry intersection.
 *
 * @param distanceToIntersection Distance to the closest geometry intersection.
 * @param step                   Remaining distance to the next sampled collision.
 * @param particlePos            Current particle position (updated in place).
 * @param dir                    Current particle direction (updated in place).
 * @param material               Current material index (updated if material changes).
 * @param state                  CURAND state used for sampling new directions and free paths.
 *
 * @return true if the particle remains alive, false if it is absorbed or lost.
 */
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
