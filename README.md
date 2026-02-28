# Anonymous Artifact for IROS 2026 Submission (Paper #922)

This repository provides an implementation of the system described in the submitted manuscript, designed to be easily integrated into trajectory analysis pipelines.

## Environment

### Hardware
- NVIDIA GPU with CUDA support
- GPU memory: >= 16 GB recommended

### Software
- OS: Linux (Ubuntu 20.04/22.04 recommended)
- CUDA Toolkit: 12.3
- C++ compiler: g++ >= 9.4.0

### Trajectory format
Plain text, one point per line:
- 2D: `x,y`
- 3D: `x,y,z`
---

## How to Call the Function

This project can be used either via a **command-line interface (CLI)** or as a **C++ function**.

### Option A: Call via CLI

To build:
```bash
make
```
To run:
```bash
./fd <trajectory 1 file> <trajectory 2 file> <number of dimensions>
``` 

### Option B: Call via C++ function

To use the code with function call, the following headers should be included:
```cpp
#include "include/txtReader.hpp" // to read trajectory files
#include "include/wrapper.hpp" // to compute FD
```

To compute FD between two trajectories, the following code should be used:
```cpp
Curve c1(txtLoaderX("trajectory 1 filename"), X); // X is the number of dimensions in the trajectory
Curve c2(txtLoaderX("trajectroy 2 filename"), X);
compute_distance_parallel(c1, c2);
```

