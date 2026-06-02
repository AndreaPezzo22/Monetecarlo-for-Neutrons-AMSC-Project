/**
 * @file movement.cuh
 * @brief Device helpers for particle movement and geometry intersection tests.
 *
 * This file provides ray/AABB intersection utilities and distance queries for
 * the global domain and material regions stored in device constant memory.
 */
#ifndef MOVEMENT_CUH
#define MOVEMENT_CUH

#include "types.h"
#include "materials.cuh"

extern __constant__ Region c_regions[]; 
extern __constant__ int c_num_regions; 

/**
 * @brief Computes the intersection distance between a ray and an axis-aligned bounding box.
 *
 * @param pos     Ray origin position.
 * @param inv_dir Reciprocal ray direction components (1/dir) for each axis.
 * @param box_min Minimum corner of the axis-aligned bounding box.
 * @param box_max Maximum corner of the axis-aligned bounding box.
 *
 * @return Distance along the ray to the nearest intersection with the box, or -1.0f if no intersection.
 */
inline __device__ float intersectAABB(float3 pos, float3 inv_dir, float3 box_min, float3 box_max) {
    float tx1 = (box_min.x - pos.x) * inv_dir.x;
    float tx2 = (box_max.x - pos.x) * inv_dir.x;
    float tmin = min(tx1, tx2);
    float tmax = max(tx1, tx2);

    float ty1 = (box_min.y - pos.y) * inv_dir.y;
    float ty2 = (box_max.y - pos.y) * inv_dir.y;
    tmin = max(tmin, min(ty1, ty2));
    tmax = min(tmax, max(ty1, ty2));

    float tz1 = (box_min.z - pos.z) * inv_dir.z;
    float tz2 = (box_max.z - pos.z) * inv_dir.z;
    tmin = max(tmin, min(tz1, tz2));
    tmax = min(tmax, max(tz1, tz2));

    // Check if ray intersects the box
    if (tmax >= max(0.0f, tmin)) {
        // If tmin > 0, we hit the box from the outside.
        // If tmin < 0, the ray origin is inside the box, so we return tmax (exit distance).
        return tmin > 0.0f ? tmin : tmax; 
    }
    
    return -1.0f; // Represents no intersection
}

/**
 * @brief Finds the nearest geometry intersection distance for a particle ray.
 *
 * @param pos Starting particle position.
 * @param dir Particle direction vector.
 *
 * @return Distance to the nearest domain or material-region intersection.
 */
inline __device__ float getDistanceToNearestIntersection(float3 pos, float3 dir){
    float closest_dist = 1.7320508f; // sqrt(3)
    
    // Precompute the inverse of direction, to minimize the number of divisions(multiplication is faster)
    float3 inv_dir = make_float3(1.0f / dir.x, 1.0f / dir.y, 1.0f / dir.z);

    // Check intersection with global domain (0.0 to 1.0)
    float dist_domain = intersectAABB(pos, inv_dir, make_float3(0.0f, 0.0f, 0.0f), make_float3(1.0f, 1.0f, 1.0f));
    
    if (dist_domain > 0.0f && dist_domain < closest_dist) {
        closest_dist = dist_domain;
    }

    // Check intersections with material regions
    for (int i = 0; i < c_num_regions; ++i) {
        Region r = c_regions[i];
        
        float3 box_min = make_float3((float)r.min_x, (float)r.min_y, (float)r.min_z);
        float3 box_max = make_float3((float)r.max_x, (float)r.max_y, (float)r.max_z);

        float dist = intersectAABB(pos, inv_dir, box_min, box_max);
        if (dist > 0.0f && dist < closest_dist) {
            closest_dist = dist;
        }
    }

    // Return the closest distance found for domain or material-region intersections.
    return closest_dist;
}

#endif