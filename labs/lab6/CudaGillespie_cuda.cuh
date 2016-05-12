#ifndef CUDA_GILLESPIE_CUDA_CUH
#define CUDA_GILLESPIE_CUDA_CUH

#define SAMPLE_TIME 100
#define SAMPLE_SIZE 1000

enum State { OFF, ON };

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
                                     const unsigned int num_simulations);

void cudaCallResamplingKernel(const unsigned int blocks,
                              const unsigned int threads_per_block,
                              float *simulation_samples,
                              int *last_sample_indices,
                              const float *simulation_times,
                              const float *simulation_concentrations,
                              const unsigned int num_simulations);

void cudaCallMinimumKernel(const unsigned int blocks,
                           const unsigned int threads_per_block,
                           const float *simulation_times,
                           float *min_val,
                           const unsigned int num_simulations);

void cudaCallMeanKernel(const unsigned int blocks,
                        const unsigned int threads_per_block,
                        float *simulation_samples,
                        float *sample_means,
                        const unsigned int sample_index,
                        const unsigned int num_simulations);

void cudaCallVarianceKernel(const unsigned int blocks,
                            const unsigned int threads_per_block,
                            float *simulation_samples,
                            float *sample_vars,
                            float *sample_means,
                            const unsigned int sample_index,
                            const unsigned int num_simulations);

#endif // CUDA_GILLESPIE_CUDA_CUH
