#include "types.h"
#include "init.cuh"
#include "utils.h"
#include "kernel.cuh"
#include "Domain.hpp"

#include <cassert>
#include <thrust/device_ptr.h>
#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#include <thrust/sequence.h>
#include <thrust/fill.h>
#include <chrono>


struct is_alive {
    int* alive_array;
    __device__ bool operator()(const int& indice_particella) const {
        return alive_array[indice_particella] == 1;
    }
};

// ── Functor: computes (x − μ)² for each element ───────────────────────────────
struct SquaredDeviation {
    double mean;
    explicit SquaredDeviation(double m) : mean(m) {}

    __host__ __device__
    double operator()(double x) const {
        double d = x - mean;
        return d * d;
    }
};

__constant__ Material c_materials[10];
__constant__ Region c_regions[20]; 
__constant__ int c_num_regions; // num regions

int main(int argc, char* argv[]) {
    std::cout << "Starting simulation..." << std::endl;
    auto t0 = std::chrono::steady_clock::now();

    // --------------------------------------------------------------------
    // SET DOMAIN from  file or default
    std::string configFile = (argc > 1) ? argv[1] : "default";
    Domain domain(configFile);
    domain.printSummaryNormalized();

    int numRegions = domain.getRegions().size();
    int N = domain.getNumParticles();
    Region source = domain.getSource();

    // Copy materials and regions to constant memory on the GPU
    std::cout << "Copying data to GPU..." << std::endl;
    cudaMemcpyToSymbol(c_regions, domain.getRegions().data(), domain.getRegionsBytes());
    cudaMemcpyToSymbol(c_materials, domain.getMaterials().data(), domain.getMaterialsBytes());
    cudaMemcpyToSymbol(c_num_regions, &numRegions, sizeof(int));

    // Kernel launch parameters
    const int blockSize = 512;
    const int numBlocks = (N + blockSize - 1) / blockSize;

    // --------------------------------------------------------------------
    // ── Allocate unified memory (accessible on both CPU and GPU)
    std::cout << "Allocating memory..." << std::endl;
    float *posx, *posy, *posz, *dirx, *diry, *dirz, *step;
    double* particle_flux;
    curandState* state;
    cudaMallocManaged(&posx, sizeof(float) * N);
    cudaMallocManaged(&posy, sizeof(float) * N);
    cudaMallocManaged(&posz, sizeof(float) * N);
    cudaMallocManaged(&dirx, sizeof(float) * N);
    cudaMallocManaged(&diry, sizeof(float) * N);
    cudaMallocManaged(&dirz, sizeof(float) * N);
    cudaMallocManaged(&step, sizeof(float) * N);
    cudaMallocManaged(&particle_flux, sizeof(double) * N);
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
    init<<<numBlocks, blockSize>>>(posx, posy, posz, dirx, diry, dirz, step, particle_flux, state, N, 42ULL, source);
    CUDA_CHECK(cudaGetLastError());       // Catches launch errors (e.g., invalid grid size)
    CUDA_CHECK(cudaDeviceSynchronize());  // Catches execution errors (e.g., memory violation inside the kernel)

    uint edgeN = 100;
    uint gridSize = edgeN * edgeN * edgeN;
    double voxelSize = 1.0 / edgeN;


    double *grid;
    cudaMallocManaged(&grid, sizeof(double) * gridSize);
    cudaMemset(grid, 0, sizeof(double) * gridSize);

    // -- Transport Cycle

    int particelle_vive = N;
    int iterazione = 0;
    const int max_iter = 1000000;

    std::cout << "Starting transport cycle..." << std::endl;
    while (particelle_vive > 0 && iterazione < max_iter) {
        int currentNumBlocks = (particelle_vive + blockSize - 1) / blockSize;

        mainKernel<<<currentNumBlocks, blockSize>>>(posx, posy, posz, dirx, diry, dirz, step, particle_flux, N, grid, edgeN, voxelSize, state, active_indices, particelle_vive, alive);
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
        
        // std::cout << "Iterazione " << iterazione << ": " << nuove_vive << " particelle vive." << std::endl;

        int* temp = active_indices;
        active_indices = compacted_indices;
        compacted_indices = temp;

        particelle_vive = nuove_vive;
        iterazione++;
    }

    CUDA_CHECK(cudaDeviceSynchronize());  // Catches execution errors (e.g., memory violation inside the kernel)
    auto t1 = std::chrono::steady_clock::now();
    double seconds = std::chrono::duration<double, std::milli>(t1 - t0).count() * 1e-3;

    // --------------------------------------------------------------------
    // ── Statistics: Compute mean and variance of particle flux

    // Reduce sums all elements. The '0.0f' is the initial value.
    thrust::device_ptr<double> begin(particle_flux);
    thrust::device_ptr<double> end = begin + N;

    double sum = thrust::reduce(begin, end, 0.0, thrust::plus<double>());
    double mean = sum / N;

    // --- COMPUTE VARIANCE ---
    // transform_reduce applies the functor, then sums the results
    double variance_sum = thrust::transform_reduce(
        begin, 
        end, 
        SquaredDeviation(mean), 
        0.0, 
        thrust::plus<double>()
    );
    

    // Divide by N for population variance (or N-1 for sample variance)
    double variance = variance_sum / N;
    double std_dev = std::sqrt(variance);
    double relative_error = (std_dev / std::sqrt(N)) / mean;
    std::cout << "Wall time: " << seconds << " s\n";

    // Output results
    std::cout << "Mean:             " << mean << "\n";
    std::cout << "Std Dev:          " << std_dev << "\n";
    std::cout << "Relative Error:   " << relative_error * 100.0 << " %\n"; // Assuming expected mean is 1.0 for demonstration
    std::cout << "FOM:              " << 1 / (relative_error * seconds) << "\n";

    if (particelle_vive > 0) {
        std::cout << "Simulation ended after reaching the maximum number of iterations (" << max_iter << ")." << std::endl;
    } else {
        std::cout << "Simulation ended with all particles absorbed after " << iterazione << " iterations." << std::endl;
    }

    save_to_vtk("output.vtk", grid, domain.getRegions(), edgeN, voxelSize);

    cudaFree(posx);
    cudaFree(posy);
    cudaFree(posz);
    cudaFree(dirx);
    cudaFree(diry);
    cudaFree(dirz);
    cudaFree(state);
    cudaFree(grid);
}

