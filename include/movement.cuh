// Device functions used to handle particle movement in the domain
#ifndef MOVEMENT_CUH
#define MOVEMENT_CUH

#include "types.cuh"
#include "materials.cuh"

extern __constant__ Region c_regions[]; 
extern __constant__ int c_num_regions; 

// Slab method function to check for intersections with Axis Aligned Bounding Box (domain)
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

inline __device__ float getDistanceToNearestIntersection(float3 pos, float3 dir, float step){
    float closest_dist = step; 
    bool hit = false;
    
    // Precompute the inverse of direction, to minimize the number of divisions(multiplication is faster)
    float3 inv_dir = make_float3(1.0f / dir.x, 1.0f / dir.y, 1.0f / dir.z);

    // Check intersection with global domain (0.0 to 1.0)
    float dist_domain = intersectAABB(pos, inv_dir, make_float3(0.0f, 0.0f, 0.0f), make_float3(1.0f, 1.0f, 1.0f));
    
    if (dist_domain > 0.0f && dist_domain < closest_dist) {
        closest_dist = dist_domain;
        hit = true;
    }

    // Check intersections with material regions
    for (int i = 0; i < c_num_regions; ++i) {
        Region r = c_regions[i];
        
        float3 box_min = make_float3((float)r.min_ix, (float)r.min_iy, (float)r.min_iz);
        float3 box_max = make_float3((float)r.max_ix, (float)r.max_iy, (float)r.max_iz);

        float dist = intersectAABB(pos, inv_dir, box_min, box_max);
        if (dist > 0.0f && dist < closest_dist) {
            closest_dist = dist;
            hit = true;
        }
    }

    // Return the closest distance if an intersection occurred before 'step', otherwise return -1.0f
    return hit ? closest_dist : -1.0f;
}

#endif