#include <iostream>
#include <random>
#include <cuda_runtime.h>
#include "types.cuh"
#include "kernels.cuh"

__constant__ Material c_materials[10];
__constant__ Region c_regions[20]; 
__constant__ int c_num_regions; // Quante scataole stiamo definendo

int main() {
    int numParticles = 1;
    int numRegions = 2;

    // 1. Prepariamo e carichiamo i materiali sulla GPU tramite la funzione ponte
    Material h_materials[3];
    Region h_regions[numRegions];
    Particle h_particle;

    h_regions[1] = {-50.0, 50.0, -50.0, 50.0, -50.0, 50.0, 0}; // L'Acqua riempie la zona da -5 a +5 (ID 0)
    h_regions[2] = {-10.0, 10.0, -10.0, 10.0, -10.0, 10.0, 1}; // L'Uranio sta al centro da -1 a +1 (ID 1)

    h_materials[0] = {0.0f, 0.0f, 0.00f};  // Vuoto (ID 0)
    h_materials[1] = {0.1f, 0.05f, 0.15f}; // Acqua (ID 1)
    h_materials[2] = {0.2f, 0.8f, 1.00f};  // Uranio (ID 2)

    cudaMemcpyToSymbol(c_materials, h_materials, 3 * sizeof(Material)); // number hard coded for now
	cudaMemcpyToSymbol(c_regions, h_regions, numRegions * sizeof(Region));
	cudaMemcpyToSymbol(c_num_regions, &numRegions, sizeof(int));

    // 2. Creiamo i dati sulla CPU (Host)
    int h_outMatID = -1;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-20.0f, 20.0f);

    h_particle.x = dist(gen);
    h_particle.y = dist(gen);
    h_particle.z = dist(gen);

    std::cout << "Coordinate generate: X=" << h_particle.x << ", Y=" << h_particle.y << ", Z=" << h_particle.z << "\n";

    // 3. Allochiamo memoria fisica REALE sulla GPU
    float *d_x, *d_y, *d_z, *d_dx, *d_dy, *d_dz;
    int *d_outMatID;
    cudaMalloc(&d_x, sizeof(float));
    cudaMalloc(&d_y, sizeof(float));
    cudaMalloc(&d_z, sizeof(float));
    cudaMalloc(&d_dx, sizeof(float));
    cudaMalloc(&d_dy, sizeof(float));
    cudaMalloc(&d_dz, sizeof(float));
    cudaMalloc(&d_outMatID, sizeof(int));

    // 4. Copiamo la posizione dalla CPU alla GPU
    cudaMemcpy(d_x, &h_particle.x, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, &h_particle.y, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_z, &h_particle.z, sizeof(float), cudaMemcpyHostToDevice);

    // 5. Lanciamo il kernel
    traverse<<<1, 1>>>(d_x, d_y, d_z, d_dx, d_dy, d_dz, d_outMatID, numParticles);
    cudaDeviceSynchronize();

    // 6. Riportiamo il risultato dalla GPU alla CPU
    cudaMemcpy(&h_outMatID, d_outMatID, sizeof(int), cudaMemcpyDeviceToHost);

    // 7. Stampiamo il risultato
    std::cout << "Materiale rilevato dalla GPU (ID): " << h_outMatID << "\n";
    if (h_outMatID == 0) std::cout << "-> Siamo nell'Acqua!\n";
    else if (h_outMatID == 1) std::cout << "-> Siamo nell'Uranio!\n";
    else std::cout << "-> Siamo nel Vuoto!\n";

    // Pulizia
    cudaFree(d_x); cudaFree(d_y); cudaFree(d_z); cudaFree(d_dx); cudaFree(d_dy); cudaFree(d_dz); cudaFree(d_outMatID);

    return 0;
}
