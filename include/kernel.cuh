#ifndef MAIN_KERNEL_CUH
#define MAIN_KERNEL_CUH 

#include "flux.cuh"
#include "types.h"
#include "movement.cuh"
#include "materials.cuh"
#include "prepNextIter.cuh"
#include "sampleFreePath.cuh"
#include <curand_kernel.h>



__global__ void mainKernel(float* posx, float* posy, float* posz, float* dirx, float* diry, float* dirz, float* step, const uint N, double* grid, const uint edgeN, const double voxelSize, curandState* randState, int* active_indices, const uint num_vive, int* alive) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= num_vive) return;

    int real_id = active_indices[id];
/*
    if (id % 10000 == 0) {
        printf("Processing particle %d / %d\n", real_id, N);
    }
*/
    float3 pos = make_float3(posx[real_id], posy[real_id], posz[real_id]);
    float3 dir = make_float3(dirx[real_id], diry[real_id], dirz[real_id]);
    float s = step[real_id];
    float distance = getDistanceToNearestIntersection(pos, dir);
    float min_step = fminf(s, distance);
    flux(pos, pos + dir * min_step, grid, edgeN, voxelSize);
    u_int8_t matID = getMaterialID(pos);

//    printf("\n--- ITERAZIONE ---\n");
//    printf("POS: (%.4f, %.4f, %.4f) | DIR: (%.2f, %.2f, %.2f) | MAT_ID: %d\n", pos.x, pos.y, pos.z, dir.x, dir.y, dir.z, matID);
//    printf("Step da fare: %.5f | Distanza ostacolo: %.5f\n", s, distance);

    bool particle_survives = true;

    if(s<distance) {
        float xi = curand_uniform(&randState[real_id]);
        float p_abs = c_materials[matID].sigma_a / c_materials[matID].sigma_t;    

        if (xi < p_abs) {
            // Absorption
            particle_survives = false;
            pos = pos + dir * s; // Move to the interaction point
//	    printf(">>> EVENTO: ASSORBIMENTO! (xi: %.3f < p_abs: %.3f) La particella muore.\n", xi, p_abs);
        } else {
//	    printf(">>> EVENTO: SCATTERING! (xi: %.3f >= p_abs: %.3f) La particella sopravvive.\n", xi, p_abs);	
	}
    } else {
//	printf(">>> EVENTO: CONFINE GEOMETRICO! La particella tocca il confine.\n");
    }

    if (particle_survives) {
        // Scattering
	alive[real_id] = 1;
        prepNextIter(pos + dir * distance, s, pos, dir, matID, &randState[real_id]);    
    } else {
        // Mark the particle as dead
        alive[real_id] = 0;
    }

 //   if (pos.x < 0.0f || pos.x > 1.0f || pos.y < 0.0f || pos.y > 1.0f || pos.z < 0.0f || pos.z > 1.0f) {
 //       alive[real_id] = 0;
 //       printf(">>> EVENTO: USCITA DAL DOMINIO! (Pos: %.2f, %.2f, %.2f). Particella terminata.\n", pos.x, pos.y, pos.z);
//    }

    posx[real_id] = pos.x;
    posy[real_id] = pos.y;
    posz[real_id] = pos.z;
    dirx[real_id] = dir.x;
    diry[real_id] = dir.y;
    dirz[real_id] = dir.z;
    step[real_id] = s;
}

#endif // MAIN_KERNEL_CUH

