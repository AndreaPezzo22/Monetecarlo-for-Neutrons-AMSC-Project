#ifndef UTILS_H 
#define UTILS_H

// Firme delle funzioni di utilità C++ (lettura file, statistiche)
#include <curand_kernel.h>
#include "math_constants.h"
#include <fstream>
#include <iostream>

inline __device__ float3 getRandomDirection(curandState *state) {
    // curand_uniform returns a random float in the range (0.0, 1.0]
    float u1 = curand_uniform(state);
    float u2 = curand_uniform(state);

    // 1. Uniformly sample z from [-1.0, 1.0]
    float z = 2.0f * u1 - 1.0f;

    // 2. Compute the radius in the x-y plane
    // We use fmaxf to prevent NaN in case z exactly equals 1.0 or -1.0 
    // due to floating-point precision quirks.
    float r = sqrtf(fmaxf(0.0f, 1.0f - z * z));

    // 3. Uniformly sample theta from [0.0, 2*PI]
    float theta = 2.0f * CUDART_PI_F * u2;

    // 4. Compute x and y
    // sincosf is a special CUDA math intrinsic that computes sine and cosine 
    // simultaneously, which is much faster than calling them separately.
    float sin_theta, cos_theta;
    sincosf(theta, &sin_theta, &cos_theta);

    return make_float3(r * cos_theta, r * sin_theta, z);
}

inline __device__ float getRandomFloat(curandState *state, float min, float max) {
    // curand_uniform returns a random float in the range (0.0, 1.0]
    float u = curand_uniform(state);
    return min + u * (max - min);
}

#define CUDA_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      std::cerr << "CUDA Error: " << cudaGetErrorString(code) 
                << " at " << file << ":" << line << std::endl;
      if (abort) exit(code);
   }
}

void save_to_vtk(const char *filename, const double* grid, const uint edgeN, const double voxelSize);

#endif // UTILS_H