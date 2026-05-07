#include <iostream>
#include <cuda_runtime.h>
#include "../../include/types.cuh"  // Per Region, Material
#include "../../include/materials.cuh"  // Per getMaterialID


__constant__ Material c_materials[10];
__constant__ Region c_regions[20]; 
__constant__ int c_num_regions;

__global__ void testGetMaterialID(float3* positions, u_int8_t* results, int numTests) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numTests) {
        results[idx] = getMaterialID(positions[idx]);
    }
}

int main() {
    // 1. Definisci dati di test fissi (come nel main.cu, ma semplificati)
    const int numRegions = 2;
    
	Material h_materials[3];
    Region h_regions[numRegions];

    h_regions[0] = {-50.0, 50.0, -50.0, 50.0, -50.0, 50.0, 1}; // L'Acqua riempie la zona da -5 a +5 (ID 1)
    h_regions[1] = {-10.0, 10.0, -10.0, 10.0, -10.0, 10.0, 2}; // L'Uranio sta al centro da -1 a +1 (ID 2)

    h_materials[0] = {0.0f, 0.0f, 0.00f};  // Vuoto (ID 0)
    h_materials[1] = {0.1f, 0.05f, 0.15f}; // Acqua (ID 1)
    h_materials[2] = {0.2f, 0.8f, 1.00f};  // Uranio (ID 2)

    // Copia su GPU
    cudaMemcpyToSymbol(c_regions, h_regions, numRegions * sizeof(Region));
    cudaMemcpyToSymbol(c_materials, h_materials, 3 * sizeof(Material));
    cudaMemcpyToSymbol(c_num_regions, &numRegions, sizeof(int));

    // 2. Definisci posizioni di test controllate e risultati attesi
    const int numTests = 5;
    float3 h_positions[numTests] = {
        {0, 0, 0},      // Dentro Uranio (ID 2)
        {30, 30, 30},      // Dentro Acqua (ID 1)
        {60, 60, 60},   // Fuori tutto → Vuoto (ID 0, default)
		{-20, -20, -20}, // Dentro Acqua (ID 1)
		{5, 5, 5}       // Dentro Uranio (ID 2)
    };
    u_int8_t expected[numTests] = {2, 1, 0, 1, 2};

    // 3. Alloca memoria GPU
    float3* d_positions;
    u_int8_t* d_results;
    cudaMalloc(&d_positions, numTests * sizeof(float3));
    cudaMalloc(&d_results, numTests * sizeof(u_int8_t));

    // Copia posizioni su GPU
    cudaMemcpy(d_positions, h_positions, numTests * sizeof(float3), cudaMemcpyHostToDevice);

    // 4. Lancia kernel di test
    testGetMaterialID<<<1, numTests>>>(d_positions, d_results, numTests);
    cudaDeviceSynchronize();

    // 5. Recupera risultati e verifica
    u_int8_t h_results[numTests];
    cudaMemcpy(h_results, d_results, numTests * sizeof(u_int8_t), cudaMemcpyDeviceToHost);

    bool allPassed = true;
    for (int i = 0; i < numTests; i++) {
        std::cout << "Test " << i << ": Pos (" << h_positions[i].x << ", " << h_positions[i].y << ", " << h_positions[i].z 
                  << ") → Mat ID " << (int)h_results[i] << " (atteso " << (int)expected[i] << ")";
        if (h_results[i] == expected[i]) {
            std::cout << " ✓\n";
        } else {
            std::cout << " ✗\n";
            allPassed = false;
        }
    }

    if (allPassed) {
        std::cout << "\nTutti i test passati!\n";
    } else {
        std::cout << "\nAlcuni test falliti.\n";
    }

    // Pulizia
    cudaFree(d_positions);
    cudaFree(d_results);

    return allPassed ? 0 : 1;
}