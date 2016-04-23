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



/*
 * Performs point-wise multiplication and scaling of raw_data and impulse_v.
 * Complex multiplication is used... fancy.
 */
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

/*
 * Optimization methods:
 * In general, we use a naive reduction, followed by a binary reduction of the
 * resulting shared memory array.
 * - Find max of each thread in block (parallelized) and store these maximums
 *   into a shared memory array of size threadsPerBlock.
 * - To find the max of the elements we stored in the shared memory array,
 *   use a binary reduction with sequential addressing. The details of this
 *   can be found in lecture/Mark Harris reduction slides. Basically, we
 *   calculate the max of elements 2^0 steps apart and write these maximums back
 *   to shared memory. Then, we calculate the max of elements 2^1 steps apart
 *   and write these maximums back to shared memory. This iterative process
 *   continues until the max is stored at sdata[0].
 * - We use threadIdx.x = 0 to store take the max of the block's absolute max
 *   and max_abs_val (ATOMICALLY!).
 * - etc. We also try to minimize reads and writes to shared/global memory
*    by using local variables where possible (e.g. by not writing the max
*    to shared memory each time, and instead keeping track of it with a
*    local variable).
 */
__global__
void
cudaMaximumKernel(cufftComplex *out_data, float *max_abs_val,
    int padded_length) {

    /* DONE: Implement the maximum-finding and subsequent
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
    extern __shared__ float sdata[];

    // Local index (for shared memory).
    uint tid = threadIdx.x;
    // Get current thread's index (global index).
    uint thread_index = blockIdx.x * blockDim.x + threadIdx.x;

    while (thread_index < padded_length) {
        // Find the maximum MAGNITUDE (take abs value) for this thread.
        thread_max = max(thread_max, fabs(out_data[thread_index].x));

        thread_index += blockDim.x * gridDim.x;
    }

    // Store thread max into shared memory.
    sdata[tid] = thread_max;

    // Make sure all threads in block finish before continuing.
    __syncthreads();

    // Run a binary reduction to find the maximum value in the shared memory.
    // Use sequential addressing to avoid bank conflicts.
    // Source: Mark Harrisris reduction slides.
    for (uint s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = max(sdata[tid], sdata[tid + s]);
        }

        // Make sure all threads have finished coalescing the shared data.
        // This is necessary because the next iteration of the for-loop
        // relies on this calculation.
        __syncthreads();
    }

    // Use the first thread in the block to calculate the block's
    // max.
    if (threadIdx.x == 0) {
        atomicMax(max_abs_val, sdata[0]);
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
