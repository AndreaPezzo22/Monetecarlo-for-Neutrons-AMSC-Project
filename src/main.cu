#include "types.h"
#include "init.cuh"
#include "utils.h"
#include "kernel.cuh"
#include <cassert>

#include <thrust/device_ptr.h>
#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#include <thrust/sequence.h>
#include <thrust/fill.h>

struct is_alive {

    int* alive_array;
    __device__ bool operator()(const int& indice_particella) const {
        return alive_array[indice_particella] == 1;
    }
};

__constant__ Material c_materials[10];
__constant__ Region c_regions[20]; 
__constant__ int c_num_regions; // num regions

int main() {
    std::cout << "Starting simulation..." << std::endl;
    const int numRegions = 3;
    
	Material h_materials[4];
    Region h_regions[numRegions];

    h_regions[0] = {0.2f, 0.8f, 0.2f, 0.8f, 0.2f, 0.8f, 1};
    h_regions[1] = {0.3f, 0.7f, 0.3f, 0.7f, 0.3f, 0.7f, 2};
    h_regions[2] = {0.4f, 0.5f, 0.4f, 0.5f, 0.4f, 0.5f, 1};

    h_materials[0] = {0.1f, 0.1f, 0.20f};
    h_materials[1] = {0.1f, 0.05f, 0.15f};
    h_materials[2] = {0.2f, 0.8f, 1.00f};

    // Copia su GPU
    std::cout << "Copying data to GPU..." << std::endl;
    cudaMemcpyToSymbol(c_regions, h_regions, numRegions * sizeof(Region));
    cudaMemcpyToSymbol(c_materials, h_materials, 3 * sizeof(Material));
    cudaMemcpyToSymbol(c_num_regions, &numRegions, sizeof(int));
    int N = 1;
    const int blockSize = 512;
    const int numBlocks = (N + blockSize - 1) / blockSize;

    // ── Allocate unified memory (accessible on both CPU and GPU)
    std::cout << "Allocating memory..." << std::endl;
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

    // -- Allocate compaction arrays
    int* active_indices;
    int* alive;
    int *compacted_indices;
    cudaMallocManaged(&active_indices, sizeof(int) * N);
    cudaMallocManaged(&alive, sizeof(int) * N);
    cudaMallocManaged(&compacted_indices, sizeof(int) * N);

    thrust::sequence(thrust::device, active_indices, active_indices + N); // Inizializza active_indices con 0, 1, 2, ..., N-1
    thrust::fill(thrust::device, alive, alive + N, 1); // Inizializza alive con 1 (tutti vivi)

    // Launch kernel with a valid block size and enough blocks for N threads
    init<<<numBlocks, blockSize>>>(posx, posy, posz, dirx, diry, dirz, step, state, N, 42ULL);
    CUDA_CHECK(cudaGetLastError());       // Catches launch errors (e.g., invalid grid size)
    CUDA_CHECK(cudaDeviceSynchronize());  // Catches execution errors (e.g., memory violation inside the kernel)

    uint edgeN = 10;
    uint gridSize = edgeN * edgeN * edgeN;
    double voxelSize = 1.0 / edgeN;


    double *grid;
    cudaMallocManaged(&grid, sizeof(double) * gridSize);
    cudaMemset(grid, 0, sizeof(double) * gridSize);

    // -- Transport Cycle

    int particelle_vive = N;
    int iterazione = 0;
    const int max_iter = 100;

    std::cout << "Starting transport cycle..." << std::endl;
    while (particelle_vive > 0 && iterazione < max_iter) {
        int currentNumBlocks = (particelle_vive + blockSize - 1) / blockSize;

        mainKernel<<<currentNumBlocks, blockSize>>>(posx, posy, posz, dirx, diry, dirz, step, N, grid, edgeN, voxelSize, state, active_indices, particelle_vive, alive);
        CUDA_CHECK(cudaGetLastError());       // Catches launch errors (e.g., invalid grid size)
        CUDA_CHECK(cudaDeviceSynchronize());  // Catches execution errors (e.g., memory violation inside the kernel)
    
        // Thrust copies in compacted_indices only the indices of particles that are still alive (alive == 1)
        auto end_ptr = thrust::copy_if(
            thrust::device, 
            active_indices, 
            active_indices + particelle_vive, 
            compacted_indices, 
            is_alive{alive}
        );

        int nuove_vive = end_ptr - compacted_indices; // Numero di particelle ancora vive dopo la compattazione
        
        std::cout << "Iterazione " << iterazione << ": " << nuove_vive << " particelle vive." << std::endl;

        int* temp = active_indices;
        active_indices = compacted_indices;
        compacted_indices = temp;

        particelle_vive = nuove_vive;
        iterazione++;
    }

    if (particelle_vive > 0) {
        std::cout << "Simulation ended after reaching the maximum number of iterations (" << max_iter << ")." << std::endl;
    } else {
        std::cout << "Simulation ended with all particles absorbed after " << iterazione << " iterations." << std::endl;
    }

    save_to_vtk("output.vtk", grid, edgeN, voxelSize);

    cudaFree(posx);
    cudaFree(posy);
    cudaFree(posz);
    cudaFree(dirx);
    cudaFree(diry);
    cudaFree(dirz);
    cudaFree(state);
    cudaFree(grid);
}

