#include "init.cuh"
#include "types.h"
#include <cassert>
#include <iostream>
#include <cmath>

__constant__ Material c_materials[10];
__constant__ Region c_regions[20]; 
__constant__ int c_num_regions;

bool isFiniteFloat(float f) {
    return std::isfinite(f);
}

int main() {
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

    int N = 10000;
    const int blockSize = 256;
    const int numBlocks = (N + blockSize - 1) / blockSize;

    // ── Allocate unified memory (accessible on both CPU and GPU)
    float *posx, *posy, *posz, *dirx, *diry, *dirz, *step;
    curandState* state;
    cudaMallocManaged(&posx, sizeof(float) * N);
    cudaMallocManaged(&posy, sizeof(float) * N);
    cudaMallocManaged(&posz, sizeof(float) * N);
    cudaMallocManaged(&dirx, sizeof(float) * N);
    cudaMallocManaged(&diry, sizeof(float) * N);
    cudaMallocManaged(&dirz, sizeof(float) * N);
    cudaMallocManaged(&step, sizeof(float) * N);
    cudaMallocManaged(&state, sizeof(curandState) * N);

    // Launch kernel with a valid block size and enough blocks for N threads
    init<<<numBlocks, blockSize>>>(posx, posy, posz, dirx, diry, dirz, step, state, N, 42ULL);
    cudaError_t err = cudaGetLastError();
    assert(err == cudaSuccess);
    err = cudaDeviceSynchronize();
    assert(err == cudaSuccess);

    double pos_sum_x = 0.0;
    double pos_sum_y = 0.0;
    double pos_sum_z = 0.0;
    double sum_x = 0.0;
    double sum_y = 0.0;
    double sum_z = 0.0;
    double sum_x2 = 0.0;
    double sum_y2 = 0.0;
    double sum_z2 = 0.0;

    for (int i = 0; i < N; ++i) {
        float x = posx[i];
        float y = posy[i];
        float z = posz[i];

        assert(isFiniteFloat(x));
        assert(isFiniteFloat(y));
        assert(isFiniteFloat(z));
        assert(x >= 0.0f && x <= 1.0f);
        assert(y >= 0.0f && y <= 1.0f);
        assert(z >= 0.0f && z <= 1.0f);

        pos_sum_x += x;
        pos_sum_y += y;
        pos_sum_z += z;

        float dx = dirx[i];
        float dy = diry[i];
        float dz = dirz[i];
        assert(isFiniteFloat(dx));
        assert(isFiniteFloat(dy));
        assert(isFiniteFloat(dz));

        float norm = std::sqrt(dx * dx + dy * dy + dz * dz);
        assert(std::fabs(norm - 1.0f) < 1e-3f);

        sum_x += dx;
        sum_y += dy;
        sum_z += dz;
        sum_x2 += dx * dx;
        sum_y2 += dy * dy;
        sum_z2 += dz * dz;
    }

    const double mean_px = pos_sum_x / N;
    const double mean_py = pos_sum_y / N;
    const double mean_pz = pos_sum_z / N;
    const double mean_x = sum_x / N;
    const double mean_y = sum_y / N;
    const double mean_z = sum_z / N;
    const double mean_x2 = sum_x2 / N;
    const double mean_y2 = sum_y2 / N;
    const double mean_z2 = sum_z2 / N;

    assert(std::fabs(mean_px - 0.5) < 0.02);
    assert(std::fabs(mean_py - 0.5) < 0.02);
    assert(std::fabs(mean_pz - 0.5) < 0.02);
    assert(std::fabs(mean_x) < 0.02);
    assert(std::fabs(mean_y) < 0.02);
    assert(std::fabs(mean_z) < 0.02);
    assert(std::fabs(mean_x2 - (1.0 / 3.0)) < 0.02);
    assert(std::fabs(mean_y2 - (1.0 / 3.0)) < 0.02);
    assert(std::fabs(mean_z2 - (1.0 / 3.0)) < 0.02);

    std::cout << "init kernel test passed: positions in [0,1], uniformly distribured in the unit cube, directions unit length, sphere stats OK\n";

    cudaFree(posx);
    cudaFree(posy);
    cudaFree(posz);
    cudaFree(dirx);
    cudaFree(diry);
    cudaFree(dirz);
    cudaFree(state);

    return 0;
}