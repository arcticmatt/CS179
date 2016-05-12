#include <cstdio>
#include <cmath>
#include <algorithm>
#include <climits>

#include <cuda_runtime.h>

#include "CudaGillespie_cuda.cuh"

/*
Atomic-max function. You may find it useful for normalization.

We haven't really talked about this yet, but __device__ functions not
only are run on the GPU, but are called from within a kernel.

Source:
http://stackoverflow.com/questions/17399119/
cant-we-use-atomic-operations-for-floating-point-variables-in-cuda
*/
__device__ static float atomicMin(float* address, float val)
{
    int* address_as_i = (int*) address;
    int old = *address_as_i, assumed;
    do {
        assumed = old;
        old = ::atomicCAS(address_as_i, assumed,
            __float_as_int(::fminf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}


/*
 * Gillespie kernel. Each call to the kernel will advance each simulation
 * by one step.
 *
 * We use the random numbers in rand_reactions to decide which transition
 * occurs, and the random numbers in rand_times to decide on a dt.
 *
 * The times in simulation_times get updated with the calculated dt,
 * and the concentrations/states may or may not get updated depending on the
 * transition.
 */
__global__
void
cudaGillespieTimestepKernel(const float *rand_reactions,
                            const float *rand_times,
                            float *simulation_times,
                            float *simulation_concentrations,
                            State *simulation_states,
                            const unsigned int b,
                            const unsigned int g,
                            const float k_on,
                            const float k_off,
                            const unsigned int num_simulations) {
    // Get current thread's index.
    unsigned int thread_index = blockIdx.x * blockDim.x + threadIdx.x;

    // Go through every simulation.
    while (thread_index < num_simulations) {
        float rand_reaction = rand_reactions[thread_index];
        float curr_conc = simulation_concentrations[thread_index];
        State curr_state = simulation_states[thread_index];
        float lambda = 0;
        if (curr_state == OFF) {
            // lambda = sum of rate parameters.
            lambda = k_on + (curr_conc * g);
            float cutoff = k_on / lambda;

            if (rand_reaction < cutoff) {
                // Flip to on
                simulation_states[thread_index] = ON;
            } else {
                // Decay
                simulation_concentrations[thread_index]--;
            }
        } else {
            // lambda = sum of rate parameters.
            lambda = k_off + b + (curr_conc * g);
            float cutoff1 = k_off  / lambda;
            float cutoff2 = cutoff1 + (b / lambda);

            if (rand_reaction < cutoff1) {
                // Flip to off
                simulation_states[thread_index] = OFF;
            } else if (rand_reaction < cutoff2) {
                // Grow
                simulation_concentrations[thread_index]++;
            } else {
                // Decay
                simulation_concentrations[thread_index]--;
            }
        }

        // Update time by calculated lambda.
        float rand_time = rand_times[thread_index];
        simulation_times[thread_index] += -log(rand_time) / lambda;

        // Update thread_index.
        thread_index += blockDim.x * gridDim.x;
    }
}

/*
 * Helper function to call Gillespie kernel.
 */
void cudaCallGillespieTimestepKernel(const unsigned int blocks,
                                     const unsigned int threads_per_block,
                                     const float *rand_reactions,
                                     const float *rand_times,
                                     float *simulation_times,
                                     float *simulation_concentrations,
                                     State *simulation_states,
                                     const unsigned int b,
                                     const unsigned int g,
                                     const float k_on,
                                     const float k_off,
                                     const unsigned int num_simulations) {
    cudaGillespieTimestepKernel<<<blocks, threads_per_block>>>(rand_reactions,
            rand_times, simulation_times, simulation_concentrations,
            simulation_states, b, g, k_on, k_off, num_simulations);
}


/*
 * Resampling kernel. After each iteration of the Gillespie algorithm, update
 * the values in an array of uniformly spaced samples. We use 1000 points
 * "evenly" spaced from 0 to 100.
 *
 * For each simulation, check its current time. If the index corresponding
 * to this time exceeds the last filled index, fill up to the current index.
 * Then, update the last filled index.
 */
__global__
void
cudaResamplingKernel(float *simulation_samples,
                     int *last_sample_indices,
                     const float *simulation_times,
                     const float *simulation_concentrations,
                     const unsigned int num_simulations) {
    // Get current thread's index.
    unsigned int thread_index = blockIdx.x * blockDim.x + threadIdx.x;

    // Go through every simulation.
    while (thread_index < num_simulations) {
        float *curr_sample = simulation_samples + (thread_index * SAMPLE_SIZE);
        float curr_time = simulation_times[thread_index];
        int last_sample_index = last_sample_indices[thread_index];
        int curr_index = curr_time / ((float) SAMPLE_TIME / SAMPLE_SIZE);

        // If the index corresponding to the current simulation time is
        // beyond the last sample index, populate the array up to the
        // curr_index.
        if (curr_index > last_sample_index
                && last_sample_index < SAMPLE_SIZE) {
            float curr_conc = simulation_concentrations[thread_index];

            while (last_sample_index <= curr_index
                    && last_sample_index < SAMPLE_SIZE) {
                curr_sample[last_sample_index++] = curr_conc;
            }
        }

        // Update last_sample_indices in GPU memory.
        last_sample_indices[thread_index] = last_sample_index;

        // Update thread_index.
        thread_index += blockDim.x * gridDim.x;
    }
}

/*
 * Helper function to call Gillespie kernel.
 */
void cudaCallResamplingKernel(const unsigned int blocks,
                              const unsigned int threads_per_block,
                              float *simulation_samples,
                              int *last_sample_indices,
                              const float *simulation_times,
                              const float *simulation_concentrations,
                              const unsigned int num_simulations) {
    cudaResamplingKernel<<<blocks, threads_per_block>>>(simulation_samples,
            last_sample_indices, simulation_times, simulation_concentrations,
            num_simulations);
}

/*
 * Minimum kernel. Used to find the minimum in an array of floats. Mainly
 * copied from the "maximum kernel" from lab 3.
 */
__global__
void
cudaMinimumKernel(const float *simulation_times,
                  float *min_val,
                  const unsigned int num_simulations) {
    extern __shared__ float partial_outputs[];

    // Get current thread's index.
    unsigned int thread_index = blockIdx.x * blockDim.x + threadIdx.x;
    float thread_min = INT_MAX;

    while (thread_index < num_simulations) {
        // Find the maximum MAGNITUDE (take abs value) for this thread.
        thread_min = min(thread_min, simulation_times[thread_index]);

        thread_index += blockDim.x * gridDim.x;
    }

    partial_outputs[threadIdx.x] = thread_min;

    // Make sure all threads in block finish before continuing.
    __syncthreads();

    // Use the first thread in the block to calculate the block's
    // max.
    if (threadIdx.x == 0) {
        float block_min = INT_MAX;

        for (uint thread_idx = 0; thread_idx < blockDim.x; ++thread_idx) {
            block_min = min(block_min, partial_outputs[thread_idx]);
        }

        // Now we take the max with the output.
        atomicMin(min_val, block_min);
    }
}

/*
 * Helper function to call minimum kernel.
 */
void cudaCallMinimumKernel(const unsigned int blocks,
                           const unsigned int threads_per_block,
                           const float *simulation_times,
                           float *min_val,
                           const unsigned int num_simulations) {
    cudaMinimumKernel<<<blocks, threads_per_block, threads_per_block * sizeof(float)>>>(
            simulation_times, min_val, num_simulations);
}

/*
 * Mean kernel. For each timepoint, we want to get the mean value for all the
 * simulations. This means we must sum the values of all the simulations at
 * that timepoint, then divide by the total number of simulations.
 */
__global__
void
cudaMeanKernel(float *simulation_samples,
               float *sample_means,
               const unsigned int sample_index,
               const unsigned int num_simulations) {
    extern __shared__ float sdata[];

    // Get current thread's index.
    unsigned int thread_index = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[threadIdx.x] = 0;

    // Go through every simulation.
    while (thread_index < num_simulations) {
        float *curr_sample = simulation_samples + (thread_index * SAMPLE_SIZE);
        float sample_conc = curr_sample[sample_index];
        sdata[threadIdx.x] += sample_conc;

        // Update thread_index.
        thread_index += blockDim.x * gridDim.x;
    }

    __syncthreads();

    // Use the first thread in the block to calculate the block's sum
    if (threadIdx.x == 0) {
        float block_sum = 0;

        for (uint thread_idx = 0; thread_idx < blockDim.x; ++thread_idx) {
            block_sum += sdata[thread_idx];
        }

        block_sum /= (float) num_simulations;
        atomicAdd(sample_means + sample_index, block_sum);
    }
}

/*
 * Helper function to call mean kernel.
 */
void cudaCallMeanKernel(const unsigned int blocks,
                        const unsigned int threads_per_block,
                        float *simulation_samples,
                        float *sample_means,
                        const unsigned int sample_index,
                        const unsigned int num_simulations) {
    cudaMeanKernel<<<blocks, threads_per_block, threads_per_block * sizeof(float)>>>(
            simulation_samples, sample_means, sample_index,
            num_simulations);
}

/*
 * Variance kernel. For each timepoint, we want to get the variance for all the
 * simulations. This means we must take sum the squared differences from the mean
 * at that timepoint, then divide by the total number of simulations. We rely
 * on the fact that sample_means has been populated at sample_index
 * (i.e. sample_means[sample_index] holds the correct average) for this
 * kernel to work.
 */
__global__
void
cudaVarianceKernel(float *simulation_samples,
                   float *sample_vars,
                   float *sample_means,
                   const unsigned int sample_index,
                   const unsigned int num_simulations) {
    extern __shared__ float sdata[];

    // Get current thread's index.
    unsigned int thread_index = blockIdx.x * blockDim.x + threadIdx.x;
    float average = sample_means[sample_index];
    sdata[threadIdx.x] = 0;

    // Go through every simulation.
    while (thread_index < num_simulations) {
        float *curr_sample = simulation_samples + (thread_index * SAMPLE_SIZE);
        float sample_conc = curr_sample[sample_index];
        float sq_diff = powf(average - sample_conc, 2);
        sdata[threadIdx.x] += sq_diff;

        // Update thread_index.
        thread_index += blockDim.x * gridDim.x;
    }

    __syncthreads();

    // Use the first thread in the block to calculate the block's sum
    if (threadIdx.x == 0) {
        float block_sum = 0;

        for (uint thread_idx = 0; thread_idx < blockDim.x; ++thread_idx) {
            block_sum += sdata[thread_idx];
        }

        block_sum /= (float) num_simulations;
        atomicAdd(sample_vars + sample_index, block_sum);
    }
}

/*
 * Helper function to call variance kernel.
 */
void cudaCallVarianceKernel(const unsigned int blocks,
                            const unsigned int threads_per_block,
                            float *simulation_samples,
                            float *sample_vars,
                            float *sample_means,
                            const unsigned int sample_index,
                            const unsigned int num_simulations) {
    cudaVarianceKernel<<<blocks, threads_per_block, threads_per_block * sizeof(float)>>>(
            simulation_samples, sample_vars, sample_means,
            sample_index, num_simulations);
}
