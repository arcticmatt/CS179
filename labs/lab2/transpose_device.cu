#include <cassert>
#include <stdio.h>
#include <cuda_runtime.h>
#include "transpose_device.cuh"

/*
 * TODO for all kernels (including naive):
 * Leave a comment above all non-coalesced memory accesses and bank conflicts.
 * Make it clear if the suboptimal access is a read or write. If an access is
 * non-coalesced, specify how many cache lines it touches, and if an access
 * causes bank conflicts, say if its a 2-way bank conflict, 4-way bank
 * conflict, etc.
 *
 * Comment all of your kernels.
 */


/*
 * Each block of the naive transpose handles a 64x64 block of the input matrix,
 * with each thread of the block handling a 1x4 section and each warp handling
 * a 32x4 section.
 *
 * If we split the 64x64 matrix into 32 blocks of shape (32, 4), then we have
 * a block matrix of shape (2 blocks, 16 blocks).
 * Warp 0 handles block (0, 0), warp 1 handles (1, 0), warp 2 handles (0, 1),
 * warp n handles (n % 2, n / 2).
 *
 * This kernel is launched with block shape (64, 16) and grid shape
 * (n / 64, n / 64) where n is the size of the square matrix.
 *
 * You may notice that we suggested in lecture that threads should be able to
 * handle an arbitrary number of elements and that this kernel handles exactly
 * 4 elements per thread. This is OK here because to overwhelm this kernel
 * it would take a 4194304 x 4194304    matrix, which would take ~17.6TB of
 * memory (well beyond what I expect GPUs to have in the next few years).
 */
__global__
void naiveTransposeKernel(const float *input, float *output, int n) {
    // DONE: do not modify code, just comment on suboptimal accesses

    const int i = threadIdx.x + 64 * blockIdx.x;
    int j = 4 * threadIdx.y + 64 * blockIdx.y;
    const int end_j = j + 4;

    /*
     * The write here (output[j + n * i]) is non-coalesced, and touches
     * 32 cache lines. This is because it accesses the output array
     * in a non-contiguous manner. That is, the output array is stored in
     * in column major order. However, within a single warp, we access via
     * rows. These row accesses do not hit the memory in sequential
     * order, causing multiple memory accesses to occur.
     */
    for (; j < end_j; j++)
        output[j + n * i] = input[i + n * j];
}

__global__
void shmemTransposeKernel(const float *input, float *output, int n) {
    // TODO: Modify transpose kernel to use shared memory. All global memory
    // reads and writes should be coalesced. Minimize the number of shared
    // memory bank conflicts (0 bank conflicts should be possible using
    // padding). Again, comment on all sub-optimal accesses.

    __shared__ float sh_in[64 * 64];

    const int i0 = threadIdx.x + 64 * blockIdx.x;
    int j0 = 4 * threadIdx.y + 64 * blockIdx.y;
    const int i1 = threadIdx.x;
    int j1 = 4 * threadIdx.y;
    const int end_j0 = j0 + 4;

    // Read in a 64 x 64 block from global memory into the 64 x 64 sized
    // shared memory array.
    for (; j0 < end_j0; j0++, j1++) {
        assert (i0 % 64 == i1);
        assert (j0 % 64 == j1);
        sh_in[i1 + 64 * j1] = input[i0 + n * j0];
    }

    __syncthreads();

    // Reassign variables, b/c we just incremented them.
    j0 = 4 * threadIdx.y + 64 * blockIdx.y;
    j1 = 4 * threadIdx.y;

    // Write the memory in shared memory to the global output array.
    // We access the shared memory in non-sequential order - this is not
    // a problem for shared memory when trying to achieve optimal performance.
    // Note that we access the output array in sequential order.
    for (; j0 < end_j0; j0++, j1++) {
        assert (i0 % 64 == i1);
        assert (j0 % 64 == j1);
        if (i1 == 0 && j1 == 63) {
            printf("===== threadIdx.x = %d, threadIdx.y = %d =====\n", threadIdx.x, threadIdx.y);
            printf("i0 = %d\n", i0);
            printf("j0 = %d\n", j0);
            printf("i1 = %d\n", i1);
            printf("j1 = %d\n", j1);
            printf("sh_in[%d] = %f, output[%d] = %f\n", j1 + 64 * i1, sh_in[j1 + 64 * i1], i0 + n * j0, output[i0 + n * j0]);
        }
        output[i0 + n * j0] = sh_in[j1 + 64 * i1];
    }
}

__global__
void optimalTransposeKernel(const float *input, float *output, int n) {
    // TODO: This should be based off of your shmemTransposeKernel.
    // Use any optimization tricks discussed so far to improve performance.
    // Consider ILP and loop unrolling.

    const int i = threadIdx.x + 64 * blockIdx.x;
    int j = 4 * threadIdx.y + 64 * blockIdx.y;
    const int end_j = j + 4;

    for (; j < end_j; j++)
        output[j + n * i] = input[i + n * j];
}

void cudaTranspose(
    const float *d_input,
    float *d_output,
    int n,
    TransposeImplementation type)
{
    if (type == NAIVE) {
        dim3 blockSize(64, 16);
        dim3 gridSize(n / 64, n / 64);
        naiveTransposeKernel<<<gridSize, blockSize>>>(d_input, d_output, n);
    }
    else if (type == SHMEM) {
        dim3 blockSize(64, 16);
        dim3 gridSize(n / 64, n / 64);
        shmemTransposeKernel<<<gridSize, blockSize>>>(d_input, d_output, n);
    }
    else if (type == OPTIMAL) {
        dim3 blockSize(64, 16);
        dim3 gridSize(n / 64, n / 64);
        optimalTransposeKernel<<<gridSize, blockSize>>>(d_input, d_output, n);
    }
    // Unknown type
    else
        assert(false);
}
