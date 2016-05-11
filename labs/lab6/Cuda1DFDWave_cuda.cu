/* CUDA finite difference wave equation solver, written by
 * Jeff Amelang, 2012
 *
 * Modified by Kevin Yuh, 2013-14 */

#include <cstdio>

#include <cuda_runtime.h>

#include "Cuda1DFDWave_cuda.cuh"


/*
 * Wave solver kernel.
 */
__global__
void
cudaWaveSolverKernel(const float *old_displacements,
                     const float *current_displacements,
                     float *new_displacements,
                     const unsigned int numberOfNodes,
                     const float courant,
                     const float left_boundary_value) {
    // Get current thread's index.
    unsigned int thread_index = blockIdx.x * blockDim.x + threadIdx.x;
    const float courantSquared = courant * courant;

    // Skip the first element (left boundary condition) and set the
    // left boundary value.
    if (thread_index == 0) {
        new_displacements[thread_index] = left_boundary_value;
        thread_index += blockDim.x * gridDim.x;
    }

    // Skip the last element (right boundary condition).
    while (thread_index <= numberOfNodes - 2) {
        new_displacements[thread_index] =
            2 * current_displacements[thread_index] - old_displacements[thread_index]
            + courantSquared * (current_displacements[thread_index + 1]
                    - 2 * current_displacements[thread_index]
                    + current_displacements[thread_index - 1]);

        // Update thread_index.
        thread_index += blockDim.x * gridDim.x;
    }

    // Set right boundary value.
    if (thread_index == numberOfNodes - 1) {
        new_displacements[thread_index] = 0;
    }
}

/*
 * Helper function to call the kernel.
 */
void cudaCallWaveSolverKernel(const unsigned int blocks,
                              const unsigned int threadsPerBlock,
                              const float *old_displacements,
                              const float *current_displacements,
                              float *new_displacements,
                              const unsigned int numberOfNodes,
                              const float courant,
                              const float left_boundary_value) {
    cudaWaveSolverKernel<<<blocks, threadsPerBlock>>>(old_displacements,
            current_displacements, new_displacements, numberOfNodes, courant,
            left_boundary_value);
}
