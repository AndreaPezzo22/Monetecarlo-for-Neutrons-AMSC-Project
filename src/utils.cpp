#include <fstream>
#include <cstddef> // For size_t

#include "utils.h"

void save_to_vtk(const char *filename, const double* grid, const std::vector<Region> regions, const uint edgeN, const double voxelSize) {
  std::ofstream file(filename);

  if(!file.is_open())
    return;

  // Calculate total cells ONCE using size_t to prevent 32-bit integer overflow
  const size_t totalCells = edgeN * edgeN * edgeN;

  // 1. HEADER & GEOMETRY
  // -------------------------------------
  file << "# vtk DataFile Version 3.0\n";
  file << "Neutron Transport Results\n";
  file << "ASCII\n";
  file << "DATASET STRUCTURED_POINTS\n";
  file << "DIMENSIONS " << (edgeN + 1) << " " << (edgeN + 1) << " " << (edgeN + 1) << "\n";
  file << "SPACING " << voxelSize << " " << voxelSize << " " << voxelSize << "\n";
  file << "ORIGIN 0.0 0.0 0.0\n";

  // 2. DATA SECTION
  // -------------------------------------
  file << "CELL_DATA " << totalCells << "\n";

  // --- VARIABLE 1: Track Length ---
  file << "SCALARS track_length double 1\n"; 
  file << "LOOKUP_TABLE default\n";

  // Pass 1: Write all lengths
  // Using size_t for 'i' prevents signed int wrap-around on massive grids
  for(size_t i = 0; i < totalCells; ++i)
    {
      file << grid[i] << "\n";
    }

  
  // --- VARIABLE 2: Material / Region ID ---
  file << "SCALARS material_id int 1\n"; 
  file << "LOOKUP_TABLE default\n";
  
  for(uint z = 0; z < edgeN; ++z) {
      double cz = (z + 0.5) * voxelSize;
      for(uint y = 0; y < edgeN; ++y) {
          double cy = (y + 0.5) * voxelSize;
          for(uint x = 0; x < edgeN; ++x) {
              double cx = (x + 0.5) * voxelSize;
              
              int current_mat_id = 0; // Fallback to 0 if nothing matches

              // Iterate in reverse: last added region has the highest priority
              for (int r = regions.size() - 1; r >= 0; --r) {
                  const auto& reg = regions[r];
                  if (cx >= reg.min_x && cx <= reg.max_x &&
                      cy >= reg.min_y && cy <= reg.max_y &&
                      cz >= reg.min_z && cz <= reg.max_z) {
                      current_mat_id = reg.mat_id; 
                      break; // Match found, skip remaining checks for this voxel
                  }
              }
              file << current_mat_id << "\n";
          }
      }
  }
  file.close(); // Optional, as the std::ofstream destructor handles this
}