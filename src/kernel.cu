# include "types.cuh"
# include "materials.cuh"
# include <curand_kernel.h>

__global__ void traverse (float* d_posX, float* d_posY, float* d_posZ,
			  float* d_dirX, float* d_dirY, float* d_dirZ,
			  int* d_outMatID,
			  int numParticles){

	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id >= numParticles) return;

	int prec_matID; // Materiale precedente della particella

	// Leggiamo la posizione della particella
	float x = d_posX[id];
	float y = d_posY[id];
	float z = d_posZ[id];
	
	// Calcolo ID materiale, make float3 ? per trasformare in un unico argomento
	prec_matID = getMaterialID(x, y, z);


	d_outMatID[id] = prec_matID;
}
