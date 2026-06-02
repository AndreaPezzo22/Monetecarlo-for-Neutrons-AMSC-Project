# Montecarlo-for-Neutrons-AMSC-Project
This project implements a high-performance 3D neutron transport simulator based on the Monte Carlo method and fully parallelized on GPUs using CUDA. 

The program tracks the trajectory of a large number of particles within a customizable domain, handling interactions (scattering, absorption), passage through different materials, and calculating the neutron flux within a three-dimensional grid that can be exported in VTK format (viewable with software such as ParaView).

### Table of Contents
- [🔬 State of the Art](#-state-of-the-art)
- [💻 Architecture and GPU Parallelization (CUDA)](#-architecture-and-gpu-parallelization-cuda)
- [🛠️ Details and Description of Functions](#️-details-and-description-of-functions)
- [⚙️ Hardware Resources](#-hardware-resources)
- [🚀 How to compile and execute](#-how-to-compile-and-execute)
- [📐 How to define the Simulation Parameters & Domain](#-how-to-define-the-simulation-parameters--domain)
- [👁️ Visualize the Results](#️-visualize-the-results)
- [📁 Repository Structure](#-repository-structure)
- [Authors](#authors)

## 🔬 State of the Art

Monte Carlo simulations are used to numerically approximate quantities of interest by simulating a high number of independent and identically distributed (i.i.d.) events and averaging the outcomes. Particle transport processes are highly amenable to this approach, as individual particle histories can often be considered independent. The idea is to trace out a number of neutron histories, simulating the interactions that occur along a neutron's flight path, while keeping track of the flux contribution of individual particles.

This project simulates particle transport within a three-dimensional domain containing regions of different materials.

Transport physics is governed by the macroscopic cross sections (Σt​, Σa​) of the materials. The code iteratively performs the following steps to simulate the particle lifecycle:
1.  Initialization: Particles are injected into the domain with a random initial position (within the bounds of the source) and an isotropic starting direction. Furthermore, a free path length $s$ is sampled using the exponential attenuation formula:
$$s = -\frac{\ln(\xi)}{\Sigma_t}$$
where $\sigma_t$​ depends on the material into which the particle is injected.

2. Tracking & Flux Estimation: The particle advances by the sampled path length or stops at the nearest geometric boundary. During this displacement, the scalar flux is estimated on the 3D grid using the Track Length Estimator method, which accumulates the length of the track segment left by the particle in each voxel.

3. Collision & Boundary Handling: If a collision occurs during step 2, the particle either scatters or is absorbed based on the absorption probability $\frac{\Sigma_a}{\Sigma_t}$​​. In the event of scattering, a new free path is sampled. If the particle hits a region boundary, the remaining path sampled in the previous step is reduced by the distance already traveled.

---

## 💻 Architecture and GPU Parallelization (CUDA)
The code was written in C++/CUDA to take advantage of the high degree of parallelism in GPU architectures, assigning the processing of **a single particle to a single thread**.

**Initialization and Management Domain**

Key technical features and optimizations include:

* **Memory Management:** Use of *Unified Memory* (`cudaMallocManaged`) to simplify data transfer between the Host (CPU) and Device (GPU), combined with `__constant__` memory for ultra-fast access to material and region properties in read-only mode.
* **Launch Strategy (Grid/Block):** The kernel is optimized to launch in blocks of **256 threads**, ensuring perfect Warp utilization (a multiple of 32) and leaving sufficient physical registers for complex transport calculations without bottlenecks.
* **Stream Compaction (Thrust):** Since particles die (due to absorption or leakage) at different rates, the GPU’s Warps would gradually empty, causing inefficiency (divergence). To solve this problem, the **Thrust** library (`thrust::copy_if`) intervenes at the end of each iteration. Dead particles are discarded, and the indices of live particles are recompressed into a dense array, allowing the GPU to launch only the exact number of blocks needed in the next cycle at 100% efficiency.
* **Thread-Safe Tallying:** The accumulation of flow in the global grid is performed using `atomicAdd` operations, ensuring that no *race conditions* occur when thousands of threads attempt to write to the same voxel simultaneously.
* **Numerical Robustness:** Implementation of advanced protections for floating-point arithmetic (FP32), including *Early-Exit* to prevent division by zero (NaN) and *Anti-Sticky Boundary* routines based on the memory-free nature of the exponential distribution to prevent particles from getting stuck on fictitious boundaries.

---

## 🛠️ Details and Description of Functions

The simulator’s operation is driven by a series of specialized functions distributed across CUDA header files (`.cuh`), C++ source files (`.cpp`), and the main kernel. The responsibilities and behavior of each are described below:

#### 📂 Domain Initialization and Management (`init.cuh` & `Domain.cpp`)
* **`Domain::Domain(const std::string& filepath)` (C++)**
  * **What it does:** This is the constructor of the Domain class. It parses the global configuration and materials JSON files (`materials.json`). It extracts the fundamental parameters and calculates the scaling factors to **normalize the entire simulation space into the unit cube $[0,1]^3$**. It also proportionally scales the macroscopic collision sections to maintain consistent physics.
* **`init(...)` (CUDA Kernel)**
  * **What it does:** Initializes the initial state of all $N$ particles on the GPU. Configures the `curand_init` random number generator independently for each thread, samples a uniform random position confined within the source, and sets an isotropic initial direction. Finally, it determines the initial material and samples the first mean free path.

#### 📂 Monte Carlo Physics (`sampleFreePath.cuh` & `utils.h`)
* **`sampleFreePath(float sigma_t, curandState *state)`**
  * **What it does:** Models the stochastic interaction probability of a neutron. It samples the distance the particle will travel before a collision by calculating the inverse of the exponential distribution: $s = -\ln(\xi)/\Sigma_t$. If the total cross section $\Sigma_t$ is zero or negative (e.g., perfect vacuum), it returns an infinite value (`CUDART_INF_F`), indicating that the particle will travel without colliding.
* **`getRandomDirection(curandState *state)`**
  * **What it does:** Generates a normalized three-dimensional vector representing an isotropic direction in space. It uses the CUDA intrinsic function `sincosf` to simultaneously compute the sine and cosine of the angle, ensuring significantly higher mathematical performance.

#### 📂 Geometry and Material Recognition (`materials.cuh` & `movement.cuh`)
* **`getMaterialID(float3 pos)`**
  * **What it does:** Determines the material index associated with a 3D coordinate in space by scanning the constant array of geometric regions. By evaluating the regions in sequential order, **the last defined region containing the point has the highest priority**. If the point does not fall within any custom structure, it returns the default material.
* **`intersectAABB(float3 pos, float3 inv_dir, float3 box_min, float3 box_max)`**
  * **What it does:** Implements the Ray-AABB intersection algorithm (Slab Method). Calculates the entry and exit points of the particle’s ray relative to an axis-aligned box (region or global domain). 
* **`getDistanceToNearestIntersection(float3 pos, float3 dir)`**
  * **What it does:** Calculates the minimum distance between the particle’s current position and the nearest geometric boundary along its trajectory. Optimizes the calculation by pre-computing the reciprocal of the direction (`1.0f / dir`) to use fast multiplications instead of divisions.

#### 📂 Flux Calculation and Transport Cycle (`flux.cuh` & `prepNextIter.cuh`)
* **`flux(float3 r0, float3 rf, double* grid, uint gridSize, double voxelSize)`**
  * **What it does:** Tracks the linear path of the particle from one point to another through the 3D voxel grid. It analytically calculates the length of the track segment deposited within each intersected voxel and accumulates it in the global grid using the safe `atomicAdd` function.
* **`prepNextIter(...)`**
  * **What it does:** It is the decision engine that determines the neutron’s fate at the end of its path:
    1. **Geometric Boundary:** If the neutron hits a wall before the step is exhausted, it is moved to the boundary. The material is updated or the direction is reversed (if the boundary is reflective). A tiny advance $\epsilon$ (epsilon) is applied to prevent permanent numerical sticking at the boundaries (*Anti-Sticky Boundary*).
    2. **Physical Interaction:** If the step ends within a material, evaluate the cross-section ratio $\Sigma_a/\Sigma_t$. If the stochastic probability dictates absorption, return `false` (neutron death); otherwise, simulate isotropic scattering by resetting the direction and step.

#### 📂 Coordination and Output (`kernel.cuh`, `main.cu`, and `utils.cpp`)
* **`mainKernel(...)` (CUDA Kernel)**
  * **What it does:** Represents the execution core launched in parallel on the GPU. For each active thread (retrieved via the compacted array), it calculates the closest geometric distance, determines the actual displacement (`min_step`), invokes `flux` to record the flux in the voxels, and calls `prepNextIter` to update the particle’s state.
* **`save_to_vtk(...)` (C++)**
  * **What it does:** Executed on the CPU at the end of the simulation. Writes the structured file `output.vtk` containing the scalar values of the accumulated flow per voxel (`track_length`) and the IDs of the voxelized materials. The file is ready to be opened with software such as ParaView.
 
## ⚙️ Hardware Resources

The experiment ran on a single node of the Politecnico of Milan cluster. The simulation took place with a single GPU for the particle parallelization. In particular, the characteristics of the node that has been used are:
* CPU
  * Model: 2x Intel(R) Xeon(R) Gold 5520+ configured in a double socket architecture (NUMA).
  * Core and Thread: a total of 56 physical cores (28 per processor) and 112 logical threads.
  * Frequency: dynamic clock speed capable of reaching a peak of 4.0 GHz.
  * Cache: 105 MiB of L3 cache, which is essential for reducing data access times during intensive processing.
* GPU
  * Model: 2x NVIDIA L4.
  * VRAM: 24 GB of dedicated memory for each graphics card.
  * Software ecosystem: CUDA 13.0.

## 🚀 How to compile and execute

1. Clone the repository in your system:
   ```bash
   git clone https://github.com/AndreaPezzo22/Montecarlo-for-Neutrons-AMSC-Project.git
   cd Montecarlo-for-Neutrons-AMSC-Project
2. Within the new directory compile using the command
   ```bash
   make
   ```
3. Run the executable
   ```bash
   ./sim [config_name]
   ```
   The program expects the corresponding configuration file to be located at `configurations/[config_name].json`, relative to your current working directory.
   More information on how to define the simulation parameters and Domain in the next 

   If `[config_name]` is omitted, the simulation defaults to `configurations/default.json`, a basic domain that serves as a proof of concept.

> **Note:** Examples of possible configurations are already present in the `configurations/` directory, and are explained in `configurations/examples_guide.json`

## 📐 How to define the Simulation Parameters & Domain

The utility of such a simulator relies on the efficient configuration of diverse simulation scenarios.

To achieve this we opted for JSON-based Parameters definition, the domain geometry is constructed using Axis-Aligned Bounding Boxes (AABBs) to define discrete regions within the simulation space.

### Configuration File Structure

To configure a simulation, create a new JSON file within the configurations directory. Below, an example from `configurations/default.json`:

```json
{
    "numParticles": 10000000,
    "numGridIntervals": 100,
    "source": {
        "min_x": -0.5, "max_x": 0.5,
        "min_y": -0.5, "max_y": 0.5,
        "min_z": -0.5, "max_z": 0.5
    },
    "default": "vacuum",
    "regions": [
        {
            "min_x": -3.0, "max_x": 0.0,
            "min_y": -3.0, "max_y": 3.0,
            "min_z": -3.0, "max_z": 3.0,
            "material_name": "boron"
        },
        {
            "min_x": 0.0, "max_x": 3.0,
            "min_y": -3.0, "max_y": 3.0,
            "min_z": -3.0, "max_z": 3.0,
            "material_name": "graphite"
        }
    ]
}
```

Parameter Specifications:

* `numParticles` (integer): The total number of independent Monte Carlo particle histories to simulate.
* `numGridIntervals` (integer): Specifies the spatial granularity. The global domain will be divided by this number along each axis to construct the 3D voxel grid used to tally the scalar flux.
* `source`: The bounding box defining the spatial limits within which initial particle positions are uniformly sampled.
* `default` (string): The background material assigned to any space within the domain that is not explicitly covered by a defined region.
* `regions`: An array of objects defining discrete AABBs and their associated materials. You can define an arbitrary number of regions, and the material_name can be completely distinct from other regions and the default material.
* **Overlap Priority**: Regions are evaluated sequentially. If spatial boundaries overlap, the region defined last in the array has the highest priority and will overwrite preceding material assignments in the intersected volume.
* **Units**: All spatial coordinates (min_x, max_x, etc.) must be expressed in $cm$.

### Material Definitions (materials.json)

Any `material_name` (and `default`) referenced in a configuration file must correspond to an explicit entry in `configurations/materials.json`. Cross-sections ($\Sigma_s$, $\Sigma_a$, $\Sigma_t$) must be expressed in $cm^{-1}$.

```json
{
  "vacuum":   { "sigma_s": 0.00,  "sigma_a": 0.00,      "sigma_t": 0.00 },
  "water":    { "sigma_s": 3.45,  "sigma_a": 0.022,     "sigma_t": 3.472 },
  "graphite": { "sigma_s": 0.38,  "sigma_a": 0.0003,    "sigma_t": 0.3803 },
  "concrete": { "sigma_s": 0.85,  "sigma_a": 0.08,      "sigma_t": 0.93 },
  "iron":     { "sigma_s": 1.10,  "sigma_a": 0.22,      "sigma_t": 1.32 },
  "boron":    { "sigma_s": 0.40,  "sigma_a": 20.00,     "sigma_t": 20.40 },
  "cadmium":  { "sigma_s": 0.32,  "sigma_a": 116.7,     "sigma_t": 117 },
  "uranium":  { "sigma_s": 0.43,  "sigma_a": 0.37,      "sigma_t": 0.8 }
}
```
### Normalization

To standardize the simulation logic across the different possible domains, the program automatically normalizes the configured physical domain into a dimensionless unit cube $[0, 1]^3$. 

This process consists of four main steps:

1. **Offsetting:** The program calculates the global bounding box encompassing all defined regions and the source. The global minimums ($x_{min}, y_{min}, z_{min}$) are used as offsets to shift the entire geometry so that its bottom-left-rear corner rests at the origin $(0,0,0)$.

2. **Scaling Lengths:** The maximum physical span across all three axes ($L_{max}$) is determined. Every spatial coordinate is then shifted and scaled to fit within the unit cube:
$$x_{scaled} = \frac{x - x_{min}}{L_{max}}$$

3. **Scaling Physics (Cross-Sections):** Because the physical free path is calculated as $l = \frac{-\ln(\xi)}{\Sigma_{t}}$, dividing the physical length by $L_{max}$ requires multiplying the macroscopic cross-section ($\Sigma_t$, measured in inverse length $cm^{-1}$) by the same factor to maintain identical transport behavior. The code automatically applies:
$$\Sigma_{scaled} = \Sigma_t \cdot L_{max}$$

4. **Default Background Initialization:** After scaling, the simulator automatically creates a foundational region that perfectly spans the entire normalized $[0, 1]^3$ domain. This region is assigned the `default` material and is processed with the lowest priority. This guarantees that any empty space not explicitly covered by a custom region is filled with the background material.

> **Practical Takeaway for Configuration:** You do not need to manually center your domain at the origin, restrict your coordinates to a specific range, or define a bounding box for the default background. Define the geometry using exact physical dimensions; the simulator handles all spatial mapping, cross-section adjustments, and background initialization internally.

## 👁️ Visualize the Results   

The results can be displayed using softwares as ParaView. We suggest to import the file output.vtk with the following settings:
* Volume view of the object.
* Filter the values using the threshold filter and adjusting the minimum and maximum values.
> **Note:** Dimensions in the visualization will be normalized on a $[0,1]^3$ unit cube.

## 📁 Repository Structure

```text
.
├── Makefile
├── README.md
├── configurations
│   ├── conf.md
│   ├── default.json
│   ├── kernel_pattern.json
│   ├── materials.json
│   ├── matrioska_cube.json
│   ├── maze.json
│   ├── multilayered_wall.json
│   └── slit_collimator.json
├── include
│   ├── Domain.hpp
│   ├── flux.cuh
│   ├── helper_math.h
│   ├── init.cuh
│   ├── json.hpp
│   ├── kernel.cuh
│   ├── materials.cuh
│   ├── movement.cuh
│   ├── prepNextIter.cuh
│   ├── sampleFreePath.cuh
│   ├── types.h
│   └── utils.h
├── output.vtk
├── sim
├── src
│   ├── Domain.cpp
│   ├── main.cu
│   ├── main.o
│   ├── utils.cpp
└── └── utils.o

```
## Authors

* Sergio Enrico Pisoni sergioenrico.pisoni@mail.polimi.it
* Andrea Pezzo andrea.pezzo@mail.polimi.it
* Leonardo Stefanelli leonardo.stefanelli@mail.polimi.it
