#include "utils.h"
#include <fstream>
#include <cstddef> // For size_t

void save_to_vtk(const char *filename, const double* grid, const uint edgeN, const double voxelSize) {
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

  file.close(); // Optional, as the std::ofstream destructor handles this
}