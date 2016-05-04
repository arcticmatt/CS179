#include <cassert>
#include <cmath>
#include <cstdio>
#include <cuda_runtime.h>
#include "classify_cuda.cuh"

/*
 * Arguments:
 * data: Memory that contains both the review LSA coefficients and the labels.
 *       Format decided by implementation of classify.
 * batch_size: Size of mini-batch, how many elements to process at once
 * step_size: Step size for gradient descent. Tune this as needed. 1.0 is sane
 *            default.
 * weights: Pointer to weights vector of length REVIEW_DIM.
 * errors: Pointer to a single float used to describe the error for the batch.
 *         An output variable for the kernel. The kernel can either write the
 *         value of loss function over the batch or the misclassification rate
 *         in the batch to errors.
 */
__global__
void trainLogRegKernel(
    float *data,
    int batch_size,
    float step_size,
	float *weights,
    float *errors)
{
    // Shared memory whose size is determined by cudaClassify.
    // Shared memory holds the weights and the gradient.
    extern __shared__ float sdata[];
    float *sdata_weights = sdata;
    float *sdata_grad = sdata + REVIEW_DIM;

    // Num misclassified, per block.
    __shared__ float num_misclassified[1];
    if (threadIdx.x == 0) {
        num_misclassified[0] = 0;
    }

    // Use index within the block to populate shared memory.
    unsigned int block_index = threadIdx.x;

    while (block_index < REVIEW_DIM) {
        // Populate shared memory.
        sdata_weights[block_index] = weights[block_index];
        sdata_grad[block_index] = 0;
        block_index += blockDim.x;
    }

    // Make sure shared data has been fully populated.
    __syncthreads();

    // Now use the global thread index to calculate the gradient.
    unsigned int original_thread_index = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int thread_index = original_thread_index;

    while (thread_index < batch_size) {
        // For every point in data, calculate the contribution to the gradient.
        // This is given by
        // (y_n * x_n) / (1 + exp(y_n w^T x_n)

        // Pointer to the features of the current point.
        int curr_index = thread_index * (REVIEW_DIM + 1);
        float dot_prod = 0;
        float y_val = data[curr_index + REVIEW_DIM];

        // Calculate dot product.
        for (int i = 0; i < REVIEW_DIM; i++) {
            dot_prod += sdata_weights[i] * data[curr_index + i];
        }

        // Check for misclassified points.
        bool sign_dot_prod = dot_prod > 0 ? true : false;
        bool sign_y_val = y_val > 0 ? true : false;
        if (sign_dot_prod != sign_y_val) {
            atomicAdd(num_misclassified, 1.0);
        }

        // Update gradient.
        for (int i = 0; i < REVIEW_DIM; i++) {
            float grad_update = (y_val * data[curr_index + i]) / (1.0 + exp(y_val * dot_prod));
            atomicAdd(&sdata_grad[i], grad_update);
        }

        thread_index += blockDim.x * gridDim.x;
    }

    // Make sure all threads in the block have updated the gradient.
    __syncthreads();

    if (threadIdx.x == 0) {
        // Update global weights, taking into account all blocks.
        for (int i = 0; i < REVIEW_DIM; i++) {
            // Multiply gradient by (-1 / N) here, which flips the sign of
            // the update to an addition.
            atomicAdd(&weights[i], step_size * sdata_grad[i] * (1.0 / batch_size));
        }

        // Add each block's error contribution to the global error.
        atomicAdd(errors, *num_misclassified / (float) batch_size);
    }
}

/*
 * All parameters have the same meaning as in docstring for trainLogRegKernel.
 * Notably, cudaClassify returns a float that quantifies the error in the
 * minibatch. This error should go down as more training occurs.
 */
float cudaClassify(
    float *data,
    int batch_size,
    float step_size,
    float *weights,
    cudaStream_t stream)
{
    int block_size = (batch_size < 1024) ? batch_size : 1024;
    int grid_size = (batch_size + block_size - 1) / block_size;
    int shmem_bytes = (REVIEW_DIM * 2) * sizeof(float);

    float *d_errors;
    cudaMalloc(&d_errors, sizeof(float));
    cudaMemset(d_errors, 0, sizeof(float));

    trainLogRegKernel<<<grid_size, block_size, shmem_bytes, stream>>>(
        data,
        batch_size,
        step_size,
        weights,
        d_errors);

    float h_errors = -1.0;
    cudaMemcpy(&h_errors, d_errors, sizeof(float), cudaMemcpyDefault);
    cudaFree(d_errors);
    return h_errors;
}
