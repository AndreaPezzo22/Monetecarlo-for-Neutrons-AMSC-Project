#include "flux.cuh"
#include <vector>
#include <cassert>
#include <iostream>

void __global__ testFluxKernel(float3 r0, float3 rf, double* grid, uint gridSize, double voxelSize) {
    // We only need one thread to test a single ray's logic
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        flux(r0, rf, grid, gridSize, voxelSize);
    }
}

// This handles all memory allocation, zeroing, kernel execution, and cleanup.
std::vector<double> runFluxOnGPU(float3 r0, float3 rf, uint gridSize, double voxelSize) {
    size_t num_voxels = gridSize * gridSize * gridSize;
    size_t byte_size = num_voxels * sizeof(double);
    
    double* d_grid;
    cudaMalloc(&d_grid, byte_size);
    // CRITICAL: Zero out the memory before atomicAdd!
    cudaMemset(d_grid, 0, byte_size); 

    // Launch kernel with 1 block, 1 thread
    testFluxKernel<<<1, 1>>>(r0, rf, d_grid, gridSize, voxelSize);
    cudaDeviceSynchronize();

    // Fetch results
    std::vector<double> h_grid(num_voxels);
    cudaMemcpy(h_grid.data(), d_grid, byte_size, cudaMemcpyDeviceToHost);

    cudaFree(d_grid);
    return h_grid;
}

void testSingleVoxelContainment() {
    float3 r0 = make_float3(0.2f, 0.5f, 0.5f);
    float3 rf = make_float3(0.5f, 0.5f, 0.5f);
    
    std::vector<double> grid = runFluxOnGPU(r0, rf, 1, 1.0);

    assert(grid[0] - length(rf - r0) < 1e-6);
    std::cout << "PASSED: testSingleVoxelContainment" << std::endl;
}

void testMultipleVoxelsHorizontal() {
    // Ray moving horizontally across 4 voxels in x-direction
    // gridSize=4, voxelSize=0.25, y=0.5 (voxel 2), z=0.5 (voxel 2)
    float3 r0 = make_float3(0.1f, 0.5f, 0.5f);
    float3 rf = make_float3(0.9f, 0.5f, 0.5f);
    uint gridSize = 4;
    double voxelSize = 0.25;
    
    std::vector<double> grid = runFluxOnGPU(r0, rf, gridSize, voxelSize);
    
    // Expected: voxel (0,2,2): 0.15, (1,2,2):0.25, (2,2,2):0.25, (3,2,2):0.15
    int idx0 = 0 + 2*4 + 2*4*4; // 0 + 8 + 32 = 40
    int idx1 = 1 + 2*4 + 2*4*4; // 1 + 8 + 32 = 41
    int idx2 = 2 + 2*4 + 2*4*4; // 2 + 8 + 32 = 42
    int idx3 = 3 + 2*4 + 2*4*4; // 3 + 8 + 32 = 43
    
    assert(fabs(grid[idx0] - 0.15) < 1e-6);
    assert(fabs(grid[idx1] - 0.25) < 1e-6);
    assert(fabs(grid[idx2] - 0.25) < 1e-6);
    assert(fabs(grid[idx3] - 0.15) < 1e-6);
    
    // Check other voxels are zero
    for (size_t i = 0; i < grid.size(); ++i) {
        if (i != idx0 && i != idx1 && i != idx2 && i != idx3) {
            assert(fabs(grid[i]) < 1e-6);
        }
    }
    std::cout << "PASSED: testMultipleVoxelsHorizontal" << std::endl;
}

void testDiagonalRay() {
    // Diagonal ray crossing multiple voxels
    // gridSize=2, voxelSize=0.5
    float3 r0 = make_float3(0.1f, 0.1f, 0.1f);
    float3 rf = make_float3(0.9f, 0.9f, 0.9f);
    uint gridSize = 2;
    double voxelSize = 0.5;
    
    std::vector<double> grid = runFluxOnGPU(r0, rf, gridSize, voxelSize);
    
    // For simplicity, just check total sum equals total length
    double totalLength = length(rf - r0);
    double sum = 0.0;
    for (double val : grid) sum += val;
    assert(fabs(sum - totalLength) < 1e-6);
    std::cout << "PASSED: testDiagonalRay" << std::endl;
}

void testNegativeDirection() {
    // Ray in negative x direction
    float3 r0 = make_float3(0.9f, 0.5f, 0.5f);
    float3 rf = make_float3(0.1f, 0.5f, 0.5f);
    uint gridSize = 4;
    double voxelSize = 0.25;
    
    std::vector<double> grid = runFluxOnGPU(r0, rf, gridSize, voxelSize);
    
    // Similar to horizontal but in reverse
    int idx3 = 3 + 2*4 + 2*4*4; // 43
    int idx2 = 2 + 2*4 + 2*4*4; // 42
    int idx1 = 1 + 2*4 + 2*4*4; // 41
    int idx0 = 0 + 2*4 + 2*4*4; // 40
    
    assert(fabs(grid[idx3] - 0.15) < 1e-6);
    assert(fabs(grid[idx2] - 0.25) < 1e-6);
    assert(fabs(grid[idx1] - 0.25) < 1e-6);
    assert(fabs(grid[idx0] - 0.15) < 1e-6);
    std::cout << "PASSED: testNegativeDirection" << std::endl;
}

void testBoundaryStart() {
    // Ray starting exactly on voxel boundary
    float3 r0 = make_float3(0.5f, 0.5f, 0.5f);
    float3 rf = make_float3(0.75f, 0.5f, 0.5f);
    uint gridSize = 2;
    double voxelSize = 0.5;
    
    std::vector<double> grid = runFluxOnGPU(r0, rf, gridSize, voxelSize);
    
    // r0 at boundary between voxel 1 and 2? n.x = floor(0.5/0.5)=1
    // So starts in voxel 1 (x=1, y=1, z=1)
    // rf at 0.75, still in voxel 1
    int idx = 1 + 1*2 + 1*2*2; // 1 + 2 + 4 = 7
    double expected = length(rf - r0); // 0.25
    assert(fabs(grid[idx] - expected) < 1e-6);
    std::cout << "PASSED: testBoundaryStart" << std::endl;
}

int main() {
    testSingleVoxelContainment();
    testMultipleVoxelsHorizontal();
    testDiagonalRay();
    testNegativeDirection();
    testBoundaryStart();
    std::cout << "All flux tests passed!" << std::endl;
    return 0;
}