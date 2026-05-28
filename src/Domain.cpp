#include <fstream>
#include <stdexcept>
#include <limits>
#include <algorithm>
#include <iostream>

#include "Domain.hpp" 
#include "json.hpp"

using json = nlohmann::json;
const int MAX_MATERIALS = 10;
const int MAX_REGIONS = 20;

// Default constructor
Domain::Domain() {
    // Initialize defaults if needed
    numParticles = 10'000;
    numGridIntervals = 100;
}

// File parsing constructor
Domain::Domain(const std::string& filepath) {
    // Open and parse the main configuration file and materials file
    std::string full_path = filepath;

    // Append .json if it doesn't already end with it
    std::string ext = ".json";
    if (full_path.length() < ext.length() || 
        full_path.compare(full_path.length() - ext.length(), ext.length(), ext) != 0) {
        full_path += ext;
    }
    // Prepend configurations/ if it doesn't already start with it
    std::string dir = "configurations/";
    if (full_path.find(dir) != 0) {
        full_path = dir + full_path;
    }

    std::ifstream file(full_path);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open domain file: " + full_path);
    }
    json config = json::parse(file);

    std::ifstream mat_file("configurations/materials.json");
    if (!mat_file.is_open()) {
        throw std::runtime_error("Failed to open configuration/materials.json");
    }
    json mat_config = json::parse(mat_file);

    // ------------------------------------------------------------------------------
    // Extract integer parameters
    numParticles = config["numParticles"].get<int>();
    numGridIntervals = config["numGridIntervals"].get<int>();

    // Extract Source
    json& source_json = config["source"];
    source.min_x = source_json["min_x"].get<float>(); 
    source.max_x = source_json["max_x"].get<float>();
    source.min_y = source_json["min_y"].get<float>();
    source.max_y = source_json["max_y"].get<float>();
    source.min_z = source_json["min_z"].get<float>();
    source.max_z = source_json["max_z"].get<float>(); 

    // Extract Default material and add it to the materials list 
    std::string default_material = config["default"].get<std::string>();

    // ------------------------------------------------------------------------------
    // Find default material in materials.json and add it to the materials list, also keep track of material names to be added from regions
    std::vector<std::string> material_names;

    int mat_index = 0;
    if(!mat_config.contains(default_material)) {
        throw std::runtime_error("Default material '" + default_material + "' not found in configurations/materials.json");
    } else {
        json& mat_json = mat_config[default_material];
        Material m;
        m.sigma_s = mat_json["sigma_s"].get<float>();
        m.sigma_a = mat_json["sigma_a"].get<float>();
        m.sigma_t = mat_json["sigma_t"].get<float>();
        materials.push_back(m);
        material_names.push_back(default_material);
        ++mat_index;
    }

    // ------------------------------------------------------------------------------
    // Extract Regions
    json& regions_json = config["regions"];
    // Track global min/max for each axis to determine the domain size
    float global_min_x = std::numeric_limits<float>::infinity();
    float global_max_x = -std::numeric_limits<float>::infinity();
    float global_min_y = std::numeric_limits<float>::infinity();
    float global_max_y = -std::numeric_limits<float>::infinity();
    float global_min_z = std::numeric_limits<float>::infinity();
    float global_max_z = -std::numeric_limits<float>::infinity();

    int reg_index = 0;
    for(const auto& reg_json : regions_json) {
        if (reg_index >= MAX_REGIONS-1) {
            throw std::runtime_error("Exceeded maximum number of regions (20)");
        }
        // Extract region bounds and material name
        float min_x = reg_json["min_x"].get<float>();
        float max_x = reg_json["max_x"].get<float>();
        float min_y = reg_json["min_y"].get<float>();
        float max_y = reg_json["max_y"].get<float>();
        float min_z = reg_json["min_z"].get<float>();
        float max_z = reg_json["max_z"].get<float>();
        std::string mat_name = reg_json["material_name"].get<std::string>();

        // Search for the material vector, if not found returns material_names.end()
        auto it = std::find(material_names.begin(), material_names.end(), mat_name);

        Region r;
        r.min_x = min_x;
        r.max_x = max_x;
        r.min_y = min_y;
        r.max_y = max_y;
        r.min_z = min_z;
        r.max_z = max_z;
        if(!mat_config.contains(mat_name)) {
            throw std::runtime_error("Material '" + mat_name + "' not found in configurations/materials.json");
        } else if(it != material_names.end()) {
            // Material already exists, use its index
            r.mat_id = std::distance(material_names.begin(), it);
        } else {
            if(mat_index >= MAX_MATERIALS) {
                throw std::runtime_error("Exceeded maximum number of materials (10)");
            }
            json& mat_json = mat_config[mat_name];
            Material m;
            m.sigma_s = mat_json["sigma_s"].get<float>();
            m.sigma_a = mat_json["sigma_a"].get<float>();
            m.sigma_t = mat_json["sigma_t"].get<float>();
            materials.push_back(m);
            material_names.push_back(mat_name);
            r.mat_id = mat_index; // Store the index of the material in current region
            ++mat_index;
        }
        regions.push_back(r);
        
        // Check if the region represent an extreme of the domain and update global min/max accordingly
        global_min_x = std::min(global_min_x, min_x);
        global_max_x = std::max(global_max_x, max_x);
        global_min_y = std::min(global_min_y, min_y);
        global_max_y = std::max(global_max_y, max_y);
        global_min_z = std::min(global_min_z, min_z);
        global_max_z = std::max(global_max_z, max_z);

        ++reg_index;
    }
    // Calculate the spans for each axis
    float span_x = global_max_x - global_min_x;
    float span_y = global_max_y - global_min_y;
    float span_z = global_max_z - global_min_z;
    // Find the maximum span across all axes
    max_span = std::max({span_x, span_y, span_z});
    // Save the offsets for all axis to map the regions to the normalized unit cube later
    offset_x = global_min_x;
    offset_y = global_min_y;
    offset_z = global_min_z;

    // ------------------------------------------------------------------------------
    // Normalize region bounds to fit within a unit cube [0,1]^3
    // Source
    source.min_x = (source.min_x - offset_x) / max_span;
    source.max_x = (source.max_x - offset_x) / max_span;
    source.min_y = (source.min_y - offset_y) / max_span;
    source.max_y = (source.max_y - offset_y) / max_span;
    source.min_z = (source.min_z - offset_z) / max_span;
    source.max_z = (source.max_z - offset_z) / max_span;

    // Regions
    for (auto& r : regions) {
        r.min_x = (r.min_x - offset_x) / max_span;
        r.max_x = (r.max_x - offset_x) / max_span;
        r.min_y = (r.min_y - offset_y) / max_span;
        r.max_y = (r.max_y - offset_y) / max_span;
        r.min_z = (r.min_z - offset_z) / max_span;
        r.max_z = (r.max_z - offset_z) / max_span;
    }
    // Cross Sections
    for (auto& m : materials) {
        m.sigma_s *= max_span;
        m.sigma_a *= max_span;
        m.sigma_t *= max_span;
    }

    // Add the default materials as a 1x1x1 region covering the entire domain
    Region default_region;
    default_region.min_x = 0.0f;
    default_region.max_x = 1.0f;
    default_region.min_y = 0.0f;
    default_region.max_y = 1.0f;
    default_region.min_z = 0.0f;
    default_region.max_z = 1.0f;
    default_region.mat_id = 0; // The default material is always at index 0
    regions.insert(regions.begin(), default_region); // Insert at the beginning to ensure it has the lowest priority
}

void Domain::printSummaryNormalized() const {
    std::cout << "\n=== Domain Summary ===\n";
    std::cout << "Particles: " << numParticles << "\n";
    std::cout << "Grid Intervals: " << numGridIntervals << "\n";
    std::cout << "Max Span: " << max_span << "\n\n";

    std::cout << "--- Source ---\n";
    std::cout << "  X: [" << source.min_x << ", " << source.max_x << "] | "
              << "Y: [" << source.min_y << ", " << source.max_y << "] | "
              << "Z: [" << source.min_z << ", " << source.max_z << "]\n\n";

    std::cout << "--- Materials (" << materials.size() << ") ---\n";
    for (size_t i = 0; i < materials.size(); ++i) {
        std::cout << "  [" << i << "] sigma_s: " << materials[i].sigma_s 
                << " | sigma_a: " << materials[i].sigma_a 
                << " | sigma_t: " << materials[i].sigma_t << "\n";
    }

    std::cout << "\n--- Regions (" << regions.size() << ") ---\n";
    for (size_t i = 0; i < regions.size(); ++i) {
        const auto& r = regions[i];
        std::cout << "  [" << i << "] Mat ID: " << static_cast<int>(r.mat_id) 
                << " | X: [" << r.min_x << ", " << r.max_x << "]"
                << " | Y: [" << r.min_y << ", " << r.max_y << "]"
                << " | Z: [" << r.min_z << ", " << r.max_z << "]\n";
    }
    std::cout << "======================\n\n";
}

void Domain::printSummary() const {
    std::cout << "\n=== Domain Summary (Original Configuration) ===\n";
    std::cout << "Particles: " << numParticles << "\n";
    std::cout << "Grid Intervals: " << numGridIntervals << "\n";
    std::cout << "Max Span: " << max_span << "\n";
    std::cout << "Offsets (X, Y, Z): " << offset_x << ", " << offset_y << ", " << offset_z << "\n\n";

    std::cout << "--- Source ---\n";
    std::cout << "  X: [" << (source.min_x * max_span) + offset_x << ", " << (source.max_x * max_span) + offset_x << "] | "
              << "Y: [" << (source.min_y * max_span) + offset_y << ", " << (source.max_y * max_span) + offset_y << "] | "
              << "Z: [" << (source.min_z * max_span) + offset_z << ", " << (source.max_z * max_span) + offset_z << "]\n\n";

    std::cout << "--- Materials (" << materials.size() << ") ---\n";
    for (size_t i = 0; i < materials.size(); ++i) {
        std::cout << "  [" << i << "] sigma_s: " << materials[i].sigma_s / max_span 
                << " | sigma_a: " << materials[i].sigma_a / max_span 
                << " | sigma_t: " << materials[i].sigma_t / max_span << "\n";
    }

    std::cout << "\n--- Regions (" << regions.size() << ") ---\n";
    for (size_t i = 0; i < regions.size(); ++i) {
        const auto& r = regions[i];
        std::cout << "  [" << i << "] Mat ID: " << static_cast<int>(r.mat_id) 
                << " | X: [" << (r.min_x * max_span) + offset_x << ", " << (r.max_x * max_span) + offset_x << "]"
                << " | Y: [" << (r.min_y * max_span) + offset_y << ", " << (r.max_y * max_span) + offset_y << "]"
                << " | Z: [" << (r.min_z * max_span) + offset_z << ", " << (r.max_z * max_span) + offset_z << "]\n";
    }
    std::cout << "===============================================\n\n";
}