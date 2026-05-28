#include "prepNextIter.cuh"
#include <vector>
#include <cassert>
#include <iostream>

__constant__ Material c_materials[10];
__constant__ Region c_regions[20]; 
__constant__ int c_num_regions;

__global__ void testPrepNextIterKernel(const float3 intersectionPoint,
                                   float &s,
                                   float3 *r,
                                   float3 *d,
                                   u_int8_t *material,
                                   unsigned int *iter,
                                   curandState *state) {

    if (threadIdx.x == 0 && blockIdx.x == 0) {
        curand_init(42ULL, 0, 0, state);
        prepNextIter(intersectionPoint, s, *r, *d, *material, *iter, state);
    }
}

std::vector<double> runPrepNextIterOnGPU(const float3 intersectionPoint,
                                   float &s,
                                   float3 &r,
                                   float3 &d,
                                   u_int8_t &material) {
                                        
    // ── Allocate unified memory (accessible on both CPU and GPU)
    float3*      u_r;
    float3*      u_d;
    u_int8_t*     u_material;
    unsigned int* u_iter;
    curandState* u_state;

    cudaMallocManaged(&u_r,        sizeof(float3));
    cudaMallocManaged(&u_d,        sizeof(float3));
    cudaMallocManaged(&u_material, sizeof(u_int8_t));
    cudaMallocManaged(&u_iter, sizeof(unsigned int));
    cudaMallocManaged(&u_state,    sizeof(curandState));

    // ── Initialize values directly from host
    *u_r        = r;
    *u_d        = d;
    *u_material = material;

    // Launch kernel with 1 block, 1 thread
    *u_iter = 0u;
    testPrepNextIterKernel<<<1, 1>>>(intersectionPoint, s, u_r, u_d, u_material, u_iter, u_state); 
    cudaDeviceSynchronize();

    // ── Read back results directly (no cudaMemcpy needed!)
    r        = *u_r;
    d        = *u_d;
    material = *u_material;

    cudaFree(u_r);
    cudaFree(u_d);
    cudaFree(u_material);
    cudaFree(u_iter);
    cudaFree(u_state);
}


int main() {
    return 0;
}