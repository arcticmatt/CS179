/* CUDA finite difference wave equation solver, written by
 * Jeff Amelang, 2012
 *
 * Modified by Kevin Yuh, 2013-14 */

#ifndef CUDA_1D_FD_WAVE_CUDA_CUH
#define CUDA_1D_FD_WAVE_CUDA_CUH

void cudaCallWaveSolverKernel(const unsigned int blocks,
                              const unsigned int threadsPerBlock,
                              const float *old_displacements,
                              const float *current_displacements,
                              float *new_displacements,
                              const unsigned int numberOfNodes,
                              const float courant,
                              const float left_boundary_value);

#endif // CUDA_1D_FD_WAVE_CUDA_CUH
