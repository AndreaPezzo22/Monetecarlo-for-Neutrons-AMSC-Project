#include <iostream>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "types.h"
#include "sampleFreePath.cuh"

__constant__ Material c_materials[10];
__constant__ Region c_regions[20];
__constant__ int c_num_regions;

__global__ void testSetStepKernel(u_int8_t *material, float *step) {
    curandState state;
    curand_init(42, 0, 0, &state);
    
    u_int8_t mat = *material;
    *step = sampleFreePath(c_materials[mat].sigma_t, &state);
}

int main() {
    const int numMaterials = 3;
    
    Material h_materials[numMaterials] = {
        {0.0f, 0.0f, 0.00f},   // Vuoto (ID 0): sigma_t = 0
        {0.1f, 0.05f, 0.15f},  // Acqua (ID 1): sigma_t = 0.15
        {0.2f, 0.8f, 1.00f}    // Uranio (ID 2): sigma_t = 1.00
    };
    
    cudaMemcpyToSymbol(c_materials, h_materials, numMaterials * sizeof(Material));
    
    for (int matID = 1; matID < numMaterials; matID++) {
        u_int8_t *d_material;
        float *d_step;
        cudaMalloc(&d_material, sizeof(u_int8_t));
        cudaMalloc(&d_step, sizeof(float));
        
        cudaMemcpy(d_material, &matID, sizeof(u_int8_t), cudaMemcpyHostToDevice);
        
        testSetStepKernel<<<1, 1>>>(d_material, d_step);
        cudaDeviceSynchronize();
        
        float h_step;
        cudaMemcpy(&h_step, d_step, sizeof(float), cudaMemcpyDeviceToHost);
        
        float sigma_t = h_materials[matID].sigma_t;
        
        std::cout << "Materiale " << (int)matID << ":\n";
        std::cout << "  sigma_t: " << sigma_t << "\n";
        std::cout << "  step calcolato: " << h_step << "\n";
        
        cudaFree(d_material);
        cudaFree(d_step);
    }
    
    return 0;
}
