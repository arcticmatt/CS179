/*
 * CUDA blur
 * Kevin Yuh, 2014
 * Revised by Nailen Matschke, 2016
 */

#include <cstdio>

#include <cuda_runtime.h>

#include "blur_device.cuh"


__global__
void cudaBlurKernel(const float *raw_data, const float *blur_v, float *out_data,
    int n_frames, int blur_v_size) {

    /* GPU-accelerated convolution. */
    /* Get current thread's id. */
    uint thread_index = blockIdx.x * blockDim.x + threadIdx.x;

    /* While this thread is dealing with a valid index... */
    while (thread_index < n_frames) {
        // Zero out data to begin with
        out_data[thread_index] = 0;

        // Perform data calculation
        int min = thread_index < blur_v_size ? thread_index + 1 : blur_v_size;
        for (int j = 0; j < min; j++)
            out_data[thread_index] += raw_data[thread_index - j] * blur_v[j];

        // Update thread_index
        thread_index += blockDim.x * gridDim.x;
    }
}


void cudaCallBlurKernel(const unsigned int blocks,
        const unsigned int threadsPerBlock,
        const float *raw_data,
        const float *blur_v,
        float *out_data,
        const unsigned int n_frames,
        const unsigned int blur_v_size) {

    /* Call the kernel above this function. */
    cudaBlurKernel<<<blocks, threadsPerBlock>>>(raw_data, blur_v, out_data,
            n_frames, blur_v_size);
}
