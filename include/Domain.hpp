#ifndef DOMAIN_HPP
#define DOMAIN_HPP

#include <string>
#include <vector>

#include "types.h"

class Domain {
private:
    int numParticles;
    int numGridIntervals;
    Region source; 
    std::vector<Material> materials; 
    std::vector<Region> regions; 

    // [0,1]^3 normalization parameters
    float max_span; 
    float offset_x; 
    float offset_y;
    float offset_z;

public:
    Domain();
    Domain(const std::string& filepath);

    // getters
    int getNumParticles() const { return numParticles; }
    int getNumGridIntervals() const { return numGridIntervals; }
    float getMaxSpan() const { return max_span; }
    Region getSourceRegion() const { return source; }

    const std::vector<Material>& getMaterials() const { return materials; }
    size_t getMaterialsBytes() const { return materials.size()*sizeof(Material); } // Needed for MemCopy

    const std::vector<Region>& getRegions() const { return regions; }
    size_t getRegionsBytes() const { return regions.size()*sizeof(Region); }

    const Region getSource() const { return source; }

    void printSummaryNormalized() const;
    void printSummary() const;
};

#endif