/* CUDA blur
 * Kevin Yuh, 2014 */

#include <cstdio>
#include <cmath>
#include <climits>
#include <algorithm>

#include <cuda_runtime.h>
#include <cufft.h>

#include "fft_convolve_cuda.cuh"


/*
Atomic-max function. You may find it useful for normalization.

We haven't really talked about this yet, but __device__ functions not
only are run on the GPU, but are called from within a kernel.

Source:
http://stackoverflow.com/questions/17399119/
cant-we-use-atomic-operations-for-floating-point-variables-in-cuda
*/
__device__ static float atomicMax(float* address, float val)
{
    int* address_as_i = (int*) address;
    int old = *address_as_i, assumed;
    do {
        assumed = old;
        old = ::atomicCAS(address_as_i, assumed,
            __float_as_int(::fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}



__global__
void
cudaProdScaleKernel(const cufftComplex *raw_data, const cufftComplex *impulse_v,
    cufftComplex *out_data,
    int padded_length) {


    /* DONE: Implement the point-wise multiplication and scaling for the
    FFT'd input and impulse response.

    Recall that these are complex numbers, so you'll need to use the
    appropriate rule for multiplying them.

    Also remember to scale by the padded length of the signal
    (see the notes for Question 1).

    As in Assignment 1 and Week 1, remember to make your implementation
    resilient to varying numbers of threads.

    */

    // Get current thread's index.
    uint thread_index = blockIdx.x * blockDim.x + threadIdx.x;

    // For complex number multiplication.
    float a, b, c, d;

    while (thread_index < padded_length) {
        // Point-wise multiplication and scale with real components.
         a = raw_data[thread_index].x;
         b = raw_data[thread_index].y;
         c = impulse_v[thread_index].x;
         d = impulse_v[thread_index].y;
         out_data[thread_index].x = (a * c - b * d) / padded_length;
         out_data[thread_index].y = (a * d + b * c) / padded_length;

        // Update thread_index.
        thread_index += blockDim.x * gridDim.x;
    }
}

__global__
void
cudaMaximumKernel(cufftComplex *out_data, float *max_abs_val,
    int padded_length) {

    /* TODO 2: Implement the maximum-finding and subsequent
    normalization (dividing by maximum).

    There are many ways to do this reduction, and some methods
    have much better performance than others.

    For this section: Please explain your approach to the reduction,
    including why you chose the optimizations you did
    (especially as they relate to GPU hardware).

    You'll likely find the above atomicMax function helpful.
    (CUDA's atomicMax function doesn't work for floating-point values.)
    It's based on two principles:
        1) From Week 2, any atomic function can be implemented using
        atomic compare-and-swap.
        2) One can "represent" floating-point values as integers in
        a way that preserves comparison, if the sign of the two
        values is the same. (see http://stackoverflow.com/questions/
        29596797/can-the-return-value-of-float-as-int-be-used-to-
        compare-float-in-cuda)

    */

    // Size determined from third parameter in cudaCallMaximumKernel.
    extern __shared__ float partial_outputs[];

    // Get current thread's index.
    uint thread_index = blockIdx.x * blockDim.x + threadIdx.x;

    float thread_max = INT_MIN;
    while (thread_index < padded_length) {
        // Find the maximum MAGNITUDE (take abs value) for this thread.
        thread_max = max(thread_max, fabs(out_data[thread_index].x));

        thread_index += blockDim.x * gridDim.x;
    }

    partial_outputs[threadIdx.x] = thread_max;

    // Make sure all threads in block finish before continuing.
    __syncthreads();

    // Use the first thread in the block to calculate the block's
    // max.
    if (threadIdx.x == 0) {
        float block_max = INT_MIN;

        for (uint thread_idx = 0; thread_idx < blockDim.x; ++thread_idx) {
            block_max = max(block_max, partial_outputs[thread_idx]);
        }

        // Now we take the max with the output.
        atomicMax(max_abs_val, block_max);
    }
}

__global__
void
cudaDivideKernel(cufftComplex *out_data, float *max_abs_val,
    int padded_length) {

    /* DONE: Implement the division kernel. Divide all
    data by the value pointed to by max_abs_val.

    This kernel should be quite short.
    */

    // Get current thread's index.
    uint thread_index = blockIdx.x * blockDim.x + threadIdx.x;

    while (thread_index < padded_length) {
        // Perform the division.
        out_data[thread_index].x /= *max_abs_val;
        out_data[thread_index].y /= *max_abs_val;

        // Update thread_index.
        thread_index += blockDim.x * gridDim.x;
    }
}


void cudaCallProdScaleKernel(const unsigned int blocks,
        const unsigned int threadsPerBlock,
        const cufftComplex *raw_data,
        const cufftComplex *impulse_v,
        cufftComplex *out_data,
        const unsigned int padded_length) {

    /* DONE: Call the element-wise product and scaling kernel. */
    cudaProdScaleKernel<<<blocks, threadsPerBlock>>>(raw_data, impulse_v,
            out_data, padded_length);
}

void cudaCallMaximumKernel(const unsigned int blocks,
        const unsigned int threadsPerBlock,
        cufftComplex *out_data,
        float *max_abs_val,
        const unsigned int padded_length) {


    /* DONE: Call the max-finding kernel. */
    cudaMaximumKernel<<<blocks, threadsPerBlock,
        threadsPerBlock * sizeof(float)>>>(out_data, max_abs_val, padded_length);
}


void cudaCallDivideKernel(const unsigned int blocks,
        const unsigned int threadsPerBlock,
        cufftComplex *out_data,
        float *max_abs_val,
        const unsigned int padded_length) {

    /* DONE: Call the division kernel. */
    cudaDivideKernel<<<blocks, threadsPerBlock>>>(out_data, max_abs_val,
            padded_length);
}
