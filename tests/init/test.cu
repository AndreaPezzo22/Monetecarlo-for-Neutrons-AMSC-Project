#include "init.cuh"
#include <cassert>
#include <iostream>
#include <cmath>

bool isFiniteFloat(float f) {
    return std::isfinite(f);
}

int main() {
    int N = 10000;
    const int blockSize = 256;
    const int numBlocks = (N + blockSize - 1) / blockSize;

    // ── Allocate unified memory (accessible on both CPU and GPU)
    float *posx, *posy, *posz, *dirx, *diry, *dirz;
    curandState* state;
    cudaMallocManaged(&posx, sizeof(float) * N);
    cudaMallocManaged(&posy, sizeof(float) * N);
    cudaMallocManaged(&posz, sizeof(float) * N);
    cudaMallocManaged(&dirx, sizeof(float) * N);
    cudaMallocManaged(&diry, sizeof(float) * N);
    cudaMallocManaged(&dirz, sizeof(float) * N);
    cudaMallocManaged(&state, sizeof(curandState) * N);

    // Launch kernel with a valid block size and enough blocks for N threads
    init<<<numBlocks, blockSize>>>(posx, posy, posz, dirx, diry, dirz, state, N, 42ULL);
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