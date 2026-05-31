# Montecarlo-for-Neutrons-AMSC-Project
This project implements a high-performance 3D neutron transport simulator based on the Monte Carlo method and fully parallelized on GPUs using CUDA. 

The program tracks the trajectory of a large number of particles within a customizable domain, handling interactions (scattering, absorption), passage through different materials, and calculating the neutron flux within a three-dimensional grid that can be exported in VTK format (viewable with software such as ParaView).

## 🔬 State of the Art

The Monte Carlo simulation is used in those practical applications, where it is too complex to allow direct numerical solutions of the transport equations. In this cases, one develops a statistical analogue description of a particle's life history on the computer, using random sampling methods. Then by running off a large number of such case histories, these results can be averaged to obtain estimates of the expected behavior of the particle population. Particle transport processes are quite amenable to such treatment, since the individual interaction events are usually described in terms of statistical characteristics. The application of Monte carlo methods to the direct simulation of particle requires to model the relevant physics of each particle interaction event as closely as possible. The essential idea is to trace out a number of neutron histories, using a table of random numbers to determine whether and what type of interactions occur along a neutron's flight.

This project implements a particle transport code based on the Monte Carlo method, designed to simulate the path of neutrons within a three-dimensional domain containing regions of different materials.  

Transport physics is governed by the macroscopic cross sections ($\Sigma_t$, $\Sigma_a$) of materials. The code iteratively performs the following steps to simulate the particle lifecycle:
1. Initialization: Particles are injected into the domain with a random initial position (within the bounds of the source) and an isotropic starting direction.
2. Material Detection: Each particle identifies the material of the region it is currently situated in.
3. Path Sampling: A new free path length ($s$) is sampled using the exponential attenuation formula:
$$s = -\frac{\ln(\xi)}{\Sigma_t}$$
4. Tracking & Flux Estimation: The particle advances by the sampled path length or stops at the nearest geometric boundary. During this displacement, the scalar flux ($\Phi$) is estimated on the 3D grid using the Track Length Estimator method, which atomically accumulates the length of the track segment left by the particle in each voxel.
5. Collision & Boundary Handling: Once the particle finishes its step, the code decides its fate. If a collision occurs, it evaluates whether the particle scatters or gets absorbed based on the absorption probability $p_{abs} = \frac{\Sigma_a}{\Sigma_t}$. If it hits a region boundary, the material is updated for the next iteration.


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

## 🚀 How to compile and execute

### Prerequisiti
* 

### Compilazione
1. Clone the repository in your system:
   ```bash
   git clone [https://github.com/](https://github.com/)[TuoUsername]/Montecarlo-for-Neutrons-AMSC-Project.git
   cd Montecarlo-for-Neutrons-AMSC-Project
2. Within the new directory compile using the command
   ```bash
   make
   ```
3. To execute the simulation, you can either use one of the ready-to-use experiments within the `configurations` directory or create a custom one using the same JSON format. Alternatively, if you just want to try the code, execute it without specifying a filename, since it calls a default configuration automatically.

* **Custom Experiment**
  Within the `configurations` directory, create a new JSON file substituting the correct values:
  ```json
  {
      "numParticles": number_of_particles,
      "numGridIntervals": number_of_intervals, 
      "source": { "min_x": min_x, "max_x": max_x, "min_y": min_y, "max_y": max_y, "min_z": min_z, "max_z": max_z },
      "default": "default_material",
      "regions": [
          { "min_x": min_x, "max_x": max_x, "min_y": min_y, "max_y": max_y, "min_z": min_z, "max_z": max_z, "material_name": "name_of_the_material" }
      ]
  }
  ```
  In the main directory of the project run the execution command as
  ```bash 
  ./sim configurations/name_of_the_configuration   
  ```
* **Experiment of the library**
  Choose between one of the available experiments, a clear explanation is available [here](https://www.markdownguide.org).
  ```bash 
  ./sim configurations/name_of_the_configuration
  ```

* **Default**
  ``` bash
  ./sim
  ```
    
## 👁️ Visualize the Results   

The results can be displayed using softwares as ParaView. We suggest to import the file output.vtk with the following settings:
* Volume view of the object.
* Filter the values using the threshold filter and adjusting the minimum and maximum values.

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
│   └── utils.o
└── tests
    ├── flux
    │   ├── Makefile
    │   └── test.cu
    ├── getMaterial
    │   ├── Makefile
    │   ├── test.cu
    │   └── test.o
    ├── init
    │   ├── Makefile
    │   ├── test
    │   └── test.cu
    ├── prepNextIter
    │   ├── Makefile
    │   └── test.cu
    └── step
        ├── Makefile
        ├── test
        ├── test.cu
        └── test.o