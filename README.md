# RobotFD: GPU-Accelerated Fr´echet Distance for Robot Path Analysis

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
Delimiter: comma
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

# Correctness Proof of the GPU-Parallelized VE-FD Computation

This document proves that the proposed dynamic programming algorithm computes the same value as the vertex-edge Fréchet distance (VE-FD) between two polygonal paths.

Let

$$P=(p_0,\dots,p_{m-1})$$

and

$$Q=(q_0,\dots,q_{n-1})$$

be two polygonal paths. Since $P$ has $m-1$ line segments and $Q$ has $n-1$ line segments, the free-space diagram contains $(m-1)\times(n-1)$ cells. Cell $(i,j)$ corresponds to the segment pair

$$(p_i,p_{i+1}) \quad \text{and} \quad (q_j,q_{j+1}),$$

where

$$0\le i\le m-2,\qquad 0\le j\le n-2.$$

## VE-FD as a Bottleneck Path Problem

In the VE-FD formulation, a feasible VE path can be viewed as a monotone path through the cell grid. Each traversed cell edge corresponds to a vertex-edge event, that is, matching a vertex of one curve to a segment of the other curve.

For cell $(i,j)$, define the two local edge weights:

$$w^{\mathrm{top}}_{i,j}=dist(q_{j+1},(p_i,p_{i+1})),$$

and

$$w^{\mathrm{right}}_{i,j}=dist(p_{i+1},(q_j,q_{j+1})).$$

Here $w^{\mathrm{top}}\_{i,j}$ is the weight of leaving cell $(i,j)$ through its top edge, and $w^{\mathrm{right}}\_{i,j}$ is the weight of leaving cell $(i,j)$ through its right edge.

The cost of a VE path is the maximum weight of all vertex-edge events visited by the path, together with the two endpoint distances

$$dist(p_0,q_0)$$

and

$$dist(p_{m-1},q_{n-1}).$$

Therefore, VE-FD is the minimum possible bottleneck value among all monotone VE paths from the start to the end of the grid.

The algorithm incorporates the two endpoint distances by initializing

$$R_{-1,0}=\max(dist(p_0,q_0),dist(p_{m-1},q_{n-1})).$$

All other out-of-range values of $T_{i,j}$ and $R_{i,j}$ are set to $\infty$.

## Dynamic Programming States

For each valid cell $(i,j)$, define $T_{i,j}$ as the minimum possible bottleneck value among all monotone VE paths that reach the top edge of cell $(i,j)$, and define $R_{i,j}$ as the minimum possible bottleneck value among all monotone VE paths that reach the right edge of cell $(i,j)$.

Thus, $T_{i,j}$ and $R_{i,j}$ represent optimal bottleneck values for two possible ways of leaving cell $(i,j)$.

## Recurrence

To reach an outgoing edge of cell $(i,j)$, a monotone path must first enter cell $(i,j)$. There are only two possible predecessors:

1. It can enter from below, through the top edge of cell $(i,j-1)$. The best bottleneck value for this case is $T_{i,j-1}$.
2. It can enter from the left, through the right edge of cell $(i-1,j)$. The best bottleneck value for this case is $R_{i-1,j}$.

Therefore, the best bottleneck value upon entering cell $(i,j)$ is

$$\min(T_{i,j-1},R_{i-1,j}).$$

After entering cell $(i,j)$, if the path leaves through the top edge, it additionally visits the top-edge vertex-edge event with weight $w^{\mathrm{top}}_{i,j}$. Since the objective is a bottleneck objective, the resulting value is

$$T_{i,j}=\max\left(w^{\mathrm{top}}_{i,j},\min(T_{i,j-1},R_{i-1,j})\right).$$

Similarly, if the path leaves through the right edge, it additionally visits the right-edge vertex-edge event with weight $w^{\mathrm{right}}_{i,j}$, giving

$$R_{i,j}=\max\left(w^{\mathrm{right}}_{i,j},\min(T_{i,j-1},R_{i-1,j})\right).$$

These are exactly the recurrences computed by the algorithm.

## Proof by Induction over Anti-Diagonals

We prove by induction over the anti-diagonal index

$$k=i+j$$

that after processing anti-diagonal $k$, the algorithm has computed the correct values of $T_{i,j}$ and $R_{i,j}$ for every valid cell $(i,j)$ with $i+j\le k$.

### Base Case

The first valid cell is $(0,0)$. Its incoming value is

$$\min(T_{0,-1},R_{-1,0}).$$

By the boundary condition, $T_{0,-1}=\infty$, while

$$R_{-1,0}=\max(dist(p_0,q_0),dist(p_{m-1},q_{n-1})).$$

Therefore,

$$\min(T_{0,-1},R_{-1,0})=R_{-1,0}.$$

This value correctly includes the two endpoint distances, which are visited by every complete VE path. The algorithm then computes

$$T_{0,0}=\max(w^{\mathrm{top}}_{0,0},R_{-1,0})$$

and

$$R_{0,0}=\max(w^{\mathrm{right}}_{0,0},R_{-1,0}).$$

These are exactly the optimal bottleneck values for paths leaving the first cell through its top and right edges. Hence, the claim holds for $k=0$.

### Induction Step

Assume that after processing all anti-diagonals smaller than $k$, all values $T_{a,b}$ and $R_{a,b}$ with $a+b<k$ are correct.

Consider any valid cell $(i,j)$ with

$$i+j=k.$$

The only possible predecessors of cell $(i,j)$ are $T_{i,j-1}$ and $R_{i-1,j}$. Both predecessor cells, if valid, lie on anti-diagonal $k-1$, because

$$i+(j-1)=k-1$$

and

$$(i-1)+j=k-1.$$

Therefore, by the induction hypothesis, the values $T_{i,j-1}$ and $R_{i-1,j}$ used by the algorithm are already correct. If a predecessor is outside the grid, its value is $\infty$, which correctly prevents invalid paths from being selected.

Thus, the best possible bottleneck value before leaving cell $(i,j)$ is

$$\min(T_{i,j-1},R_{i-1,j}).$$

Taking the top edge adds the local event weight $w^{\mathrm{top}}_{i,j}$ under the bottleneck objective, so the optimal value for the top edge is

$$T_{i,j}=\max\left(w^{\mathrm{top}}_{i,j},\min(T_{i,j-1},R_{i-1,j})\right).$$

Taking the right edge similarly gives

$$R_{i,j}=\max\left(w^{\mathrm{right}}_{i,j},\min(T_{i,j-1},R_{i-1,j})\right).$$

These are exactly the values computed by the algorithm. Hence, $T_{i,j}$ and $R_{i,j}$ are correct for every cell on anti-diagonal $k$.

By induction, the algorithm computes correct $T_{i,j}$ and $R_{i,j}$ values for all valid cells.

## Terminal Value

The terminal cell of the free-space diagram is

$$(m-2,n-2).$$

Any complete monotone VE path must leave this terminal cell either through its top edge or through its right edge. Therefore, the optimal bottleneck value of a complete VE path is

$$\min(T_{m-2,n-2},R_{m-2,n-2}).$$

This is exactly the value returned by the algorithm.

Therefore, the algorithm computes the VE-FD value between $P$ and $Q$.

## Conclusion

The algorithm evaluates the same bottleneck dynamic program induced by the VE-FD grid formulation. It processes cells in anti-diagonal order, so all dependencies are satisfied before each cell is computed. The boundary initialization incorporates the endpoint vertex-vertex distances, and the final minimum over the two outgoing edges of the terminal cell gives the optimal complete VE path value. Hence, the returned value is identical to VE-FD.
