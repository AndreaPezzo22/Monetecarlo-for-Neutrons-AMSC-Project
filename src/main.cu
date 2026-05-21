#include "types.h"
#include "init.cuh"
#include "kernel.cuh"
#include <cassert>

__constant__ Material c_materials[10];
__constant__ Region c_regions[20]; 
__constant__ int c_num_regions; // num regions

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

    uint edgeN = 100;
    uint *gridSize;
    double *voxelSize;

    cudaMallocManaged(&gridSize, sizeof(uint));
    cudaMallocManaged(&voxelSize, sizeof(double));
    *gridSize = edgeN * edgeN * edgeN;
    *voxelSize = 1.0 / edgeN;

    double *grid;
    cudaMallocManaged(&grid, sizeof(double) * *gridSize);

    mainKernel<<<numBlocks, blockSize>>>(posx, posy, posz, dirx, diry, dirz, step, grid, gridSize, voxelSize, state);

    cudaFree(posx);
    cudaFree(posy);
    cudaFree(posz);
    cudaFree(dirx);
    cudaFree(diry);
    cudaFree(dirz);
    cudaFree(state);
    cudaFree(gridSize);
    cudaFree(voxelSize);
    cudaFree(grid);
}
