#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <fstream>
#include <algorithm>

#include <cuda_runtime.h>
#include <curand.h>

#include "CudaGillespie_cuda.cuh"
#include "ta_utilities.hpp"

/* Check errors on CUDA runtime functions */
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        fprintf(stderr, "code = %d\n", code);
        exit(code);
    }
}

int main(int argc, char* argv[]) {
    if (argc != 4) {
        printf("Usage: (threads per block) (max number of blocks) (num simulations)\n");
        exit(-1);
    }

    const unsigned int threads_per_block = atoi(argv[1]);
    const unsigned int blocks = atoi(argv[2]);
    const unsigned int num_simulations = atoi(argv[3]);
    // Rate constants
    const unsigned int b = 10;
    const unsigned int g = 1;
    const float k_on = 0.1;
    const float k_off = 0.9;
    // Rand number generator
    curandGenerator_t gen;
    // Create pseudo-random number generator
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    // Set seed
    curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);

    // === Host memory ===
    // Mean and variance.
    float *sample_means_host = new float[SAMPLE_SIZE];
    float *sample_vars_host = new float[SAMPLE_SIZE];
    float min_time_host = 0;
    float big_float = INT_MAX;

    // === Device memory ===
    size_t sim_size = num_simulations * sizeof(float);
    // Arrays that contain random floats (0 - 1) for choosing transitions and
    // timesteps.
    float *rand_reactions_dev;
    float *rand_times_dev;
    // Simulation values.
    float *simulation_times_dev;
    float *simulation_concentrations_dev;
    State *simulation_states_dev;
    float *min_time_dev;
    // Sample values.
    int *last_sample_indices_dev;
    float *simulation_samples_dev;
    float *sample_means_dev;
    float *sample_vars_dev;

    // Allocate device memory.
    gpuErrchk(cudaMalloc((void **) &rand_reactions_dev, sim_size));
    gpuErrchk(cudaMalloc((void **) &rand_times_dev, sim_size));
    gpuErrchk(cudaMalloc((void **) &simulation_times_dev, sim_size));
    gpuErrchk(cudaMalloc((void **) &simulation_concentrations_dev, sim_size));
    gpuErrchk(cudaMalloc((void **) &simulation_states_dev, sim_size));
    gpuErrchk(cudaMalloc((void **) &min_time_dev, sizeof(float)));
    gpuErrchk(cudaMalloc((void **) &last_sample_indices_dev, sim_size));
    gpuErrchk(cudaMalloc((void **) &simulation_samples_dev, sim_size * SAMPLE_SIZE));
    gpuErrchk(cudaMalloc((void **) &sample_means_dev, SAMPLE_SIZE * sizeof(float)));
    gpuErrchk(cudaMalloc((void **) &sample_vars_dev, SAMPLE_SIZE * sizeof(float)));

    // Memset device memory.
    gpuErrchk(cudaMemset(simulation_times_dev, 0, sim_size));
    gpuErrchk(cudaMemset(simulation_concentrations_dev, 0, sim_size));
    gpuErrchk(cudaMemset(simulation_states_dev, 0, sim_size));
    gpuErrchk(cudaMemset(last_sample_indices_dev, 0, sim_size));
    gpuErrchk(cudaMemset(simulation_samples_dev, 0, sim_size * SAMPLE_SIZE));
    gpuErrchk(cudaMemset(sample_means_dev, 0, SAMPLE_SIZE * sizeof(float)));
    gpuErrchk(cudaMemset(sample_vars_dev, 0, SAMPLE_SIZE * sizeof(float)));

    int iter = 0;
    while (true) {
        // Fill random arrays.
        curandGenerateUniform(gen, rand_reactions_dev, num_simulations);
        curandGenerateUniform(gen, rand_times_dev, num_simulations);

        // Gillespie algorithm to update states, concentrations.
        cudaCallGillespieTimestepKernel(blocks, threads_per_block, rand_reactions_dev,
                rand_times_dev, simulation_times_dev, simulation_concentrations_dev,
                simulation_states_dev, b, g, k_on, k_off, num_simulations);

        // Resampling kernel to populate simulation_samples_dev.
        cudaCallResamplingKernel(blocks, threads_per_block, simulation_samples_dev,
                last_sample_indices_dev, simulation_times_dev,
                simulation_concentrations_dev, num_simulations);

        // Set min_time_dev to be a very large number.
        gpuErrchk(cudaMemcpy(min_time_dev, &big_float, sizeof(float),
                    cudaMemcpyHostToDevice));
        // Get min of simulation_times.
        cudaCallMinimumKernel(blocks, threads_per_block, simulation_times_dev,
                min_time_dev, num_simulations);
        // Copy min_time_dev back to host.
        gpuErrchk(cudaMemcpy(&min_time_host, min_time_dev, sizeof(float),
                    cudaMemcpyDeviceToHost));

        // If all simulations have finished, break.
        if (min_time_host >= SAMPLE_TIME) {
            break;
        }

        iter++;
    }

    for (int i = 0; i < SAMPLE_SIZE; i++) {
        // Calculate mean of simulation_samples.
        cudaCallMeanKernel(blocks, threads_per_block, simulation_samples_dev,
                sample_means_dev, i, num_simulations);

        // Calculate variance of simulation_samples.
        cudaCallVarianceKernel(blocks, threads_per_block, simulation_samples_dev,
                sample_vars_dev, sample_means_dev, i, num_simulations);
    }

    // Copy means and vars back to host.
    gpuErrchk(cudaMemcpy(sample_means_host, sample_means_dev, SAMPLE_SIZE * sizeof(float),
                cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(sample_vars_host, sample_vars_dev, SAMPLE_SIZE * sizeof(float),
                cudaMemcpyDeviceToHost));

    // Print means and variances.
    printf("=== means ===\n");
    for (int i = 0; i < SAMPLE_SIZE; i++) {
        printf("%f\n", sample_means_host[i]);
    }
    printf("\n\n\n=== variances ===\n");
    for (int i = 0; i < SAMPLE_SIZE; i++) {
        printf("%f\n", sample_vars_host[i]);
    }

    // Free host memory.
    delete[] sample_means_host;
    delete[] sample_vars_host;

    // Free device memory.
    cudaFree(rand_reactions_dev);
    cudaFree(rand_times_dev);
    cudaFree(simulation_times_dev);
    cudaFree(simulation_concentrations_dev);
    cudaFree(simulation_states_dev);
    cudaFree(min_time_dev);
    cudaFree(last_sample_indices_dev);
    cudaFree(simulation_samples_dev);
    cudaFree(sample_means_dev);
    cudaFree(sample_vars_dev);
}
