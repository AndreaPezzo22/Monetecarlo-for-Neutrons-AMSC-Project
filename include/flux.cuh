// Compute and accumulate voxel flux contributions for a particle track
//
// Parameters:
//  - r0: starting particle position in normalized domain coordinates [0,1]^3
//  - rf: ending particle position in normalized domain coordinates [0,1]^3
//  - grid: device pointer to flattened 3D voxel flux tally array (size gridSize^3)
//  - gridSize: number of voxels along each axis (assuming cubic grid)
//  - voxelSize: physical size of each voxel (in domain units, e.g., 1.0/gridSize)
//
// Preconditions:
//  - Both r0 and rf are inside the unit cube [0,1]^3.
//  - grid has exactly gridSize * gridSize * gridSize entries.
//  - The direction from r0 to rf is nonzero (i.e., r0 != rf).
//  - gridSize > 0 and voxelSize > 0.
//
// Behavior:
//  - Traces the straight-line segment from r0 to rf through the voxel grid.
//  - For each voxel intersected by the segment, adds the length of the segment within that voxel to the corresponding grid entry.
//  - Uses atomic addition to handle concurrent writes in parallel CUDA kernels.
//  - Assumes row-major ordering for the flattened 3D array: idx = x + y*gridSize + z*gridSize*gridSize

#ifndef FLUX_CUH
#define FLUX_CUH

#include "helper_math.h"

inline __device__ int getsign(const float f) {
    return (int)copysignf(1.0f, f);
}

inline __device__ int3 signOfDir(const float3 dir) {
    return make_int3(getsign(dir.x), getsign(dir.y), getsign(dir.z));
}

inline __device__ void flux (float3 r0, float3 rf, double* grid, uint gridSize, double voxelSize) {
    // Normalize the direction vector from r0 to rf
    float3 dir = normalize(rf - r0);
    
    // Compute the distance along the ray to travel one voxel in each direction
    // delta = voxelSize / |dir| for each component
    float3 delta = voxelSize / fabs(dir);
    
    // Determine the starting voxel indices by flooring the position divided by voxelSize
    uint3 n = make_uint3(
        (uint)floorf(r0.x / voxelSize),
        (uint)floorf(r0.y / voxelSize),
        (uint)floorf(r0.z / voxelSize)
    );

    // Get the sign of each direction component (+1 or -1)
    int3 sign = signOfDir(dir);

    // Determine the step direction: 1 if positive, 0 if negative (for boundary calculation)
    int3 step = make_int3(max(sign.x, 0), max(sign.y, 0), max(sign.z, 0));
    
    // Compute the initial distances to the next voxel boundaries in each direction
    // t = (next_boundary - r0) / dir
    float3 t = (make_float3(n + make_uint3(step)) * voxelSize - r0) / dir;

    // Total length of the segment
    float tMax = length(rf - r0);
    // Current distance traveled along the ray
    float tCur = 0.0f;

    // Traverse the voxel grid until the entire segment is covered
    while (tCur < tMax) {
        // Find the smallest t value (closest intersection with voxel boundary)
        float tNext = fminf(t.x, fminf(t.y, t.z));
        
        // Length of the segment within the current voxel
        float segment = fminf(tNext, tMax) - tCur;

	// Check on the boundaries, if the particle is out of the grid we do not update the flux        
	if (n.x >= 0 && n.x < gridSize && 
            n.y >= 0 && n.y < gridSize && 
            n.z >= 0 && n.z < gridSize) {
	// Compute the flattened index for the current voxel
        int idx = n.x + n.y * gridSize + n.z * gridSize * gridSize;
        
        // Atomically add the segment length to the flux tally for this voxel
        atomicAdd(&grid[idx], (double)segment);
	}	

        // Advance to the next intersection point
        tCur = tNext;
        
        // Determine which axis we are stepping along (only one will be true)
        int stepX = (tNext == t.x);
        int stepY = (tNext == t.y);
        int stepZ = (tNext == t.z);

        // Update voxel index and t value for the stepped axis
        n.x += stepX * sign.x;
        t.x += stepX * delta.x;

        n.y += stepY * sign.y;
        t.y += stepY * delta.y;

        n.z += stepZ * sign.z;
        t.z += stepZ * delta.z;
    }
}

#endif // FLUX_CUH
