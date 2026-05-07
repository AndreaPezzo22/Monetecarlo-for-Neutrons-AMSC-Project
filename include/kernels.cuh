#ifndef KERNELS_H
#define KERNELS_H

#include "types.cuh"
#include <cuda_runtime.h>

__global__ void traverse(float* posX, float* posY, float* posZ, float* dirX, float* dirY, float* dirZ, int* outMatID, int numParticles);

#endif