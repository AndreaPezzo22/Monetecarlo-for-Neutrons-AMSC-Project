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

inline __device__ void prepNextIter(const float3 intersectionPoint,
                                   float &s,
                                   float3 &r,
                                   float3 &d,
                                   u_int8_t &material,
                                   curandState *state) {
    const float epsilon = 1e-6f;
    const float distanceToIntersection = length(intersectionPoint - r);

    if (distanceToIntersection <= s) {
        r = intersectionPoint;
        const float3 nextPos = r + d * epsilon;

        const bool hitX = nextPos.x < 0.0f || nextPos.x > 1.0f;
        const bool hitY = nextPos.y < 0.0f || nextPos.y > 1.0f;
        const bool hitZ = nextPos.z < 0.0f || nextPos.z > 1.0f;

        if (hitX || hitY || hitZ) {
            r -= d * epsilon; // Move back slightly to avoid sticking to the boundary
            if (hitX) d.x = -d.x;
            if (hitY) d.y = -d.y;
            if (hitZ) d.z = -d.z;
            s -= distanceToIntersection;
        } else {
            // Case 2: particle is not at a boundary but is changing material.
            // We update the material.
            u_int8_t mat= getMaterialID(nextPos);
            r = nextPos;
            if (material == mat) {
                s -= distanceToIntersection;
            } else {
                material = mat;
                s = sampleFreePath(c_materials[material].sigma_t, state);
            }
        }

        // TODO: handle absorption if boundaries are absorbing
    } else {
        s = sampleFreePath(c_materials[material].sigma_t, state);
        r = r + d * s;
        d = getRandomDirection(state);
    }
}

#endif // PREP_NEXT_ITER_CUH