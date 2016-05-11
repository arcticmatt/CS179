#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <sstream>

#include <cuda_runtime.h>

#include "classify_cuda.cuh"
#include "ta_utilities.hpp"

using namespace std;

static const float STEP_SIZE = 0.1;

/*
NOTE: You can use this macro to easily check cuda error codes
and get more information.

Modified from:
http://stackoverflow.com/questions/14038589/
what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api
*/
#define gpuErrChk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(
    cudaError_t code,
    const char *file,
    int line,
    bool abort=true)
{
    if (code != cudaSuccess) {
        fprintf(stderr,"GPUassert: %s %s %d\n",
            cudaGetErrorString(code), file, line);
        exit(code);
    }
}

// timing setup code
cudaEvent_t start;
cudaEvent_t stop;

#define START_TIMER() {                         \
    gpuErrChk(cudaEventCreate(&start));         \
    gpuErrChk(cudaEventCreate(&stop));          \
    gpuErrChk(cudaEventRecord(start));          \
}

#define STOP_RECORD_TIMER(name) {                           \
    gpuErrChk(cudaEventRecord(stop));                       \
    gpuErrChk(cudaEventSynchronize(stop));                  \
    gpuErrChk(cudaEventElapsedTime(&name, start, stop));    \
    gpuErrChk(cudaEventDestroy(start));                     \
    gpuErrChk(cudaEventDestroy(stop));                      \
}

////////////////////////////////////////////////////////////////////////////////
// Start non boilerplate code

// Fills output with standard normal data
void gaussianFill(float *output, int size) {
    // seed generator to 2015
    std::default_random_engine generator(2015);
    std::normal_distribution<float> distribution(0.0, 0.1);
    for (int i=0; i < size; i++) {
        output[i] = distribution(generator);
    }
}

// Takes a string of comma seperated floats and stores the float values into
// output. Each string should consist of REVIEW_DIM + 1 floats.
void readLSAReview(string review_str, float *output, int stride) {
    stringstream stream(review_str);
    int component_idx = 0;

    for (string component; getline(stream, component, ','); component_idx++) {
        output[stride * component_idx] = atof(component.c_str());
    }
    assert(component_idx == REVIEW_DIM + 1);
}

void classify(istream& in_stream, int batch_size) {
    // Allocate host memory, initialize device memory.
    size_t size_weights = REVIEW_DIM * sizeof(float);
    size_t size_data = batch_size * (REVIEW_DIM + 1) * sizeof(float);
    float *weights_host = (float *) malloc(size_weights);
    float *weights_dev;
    float *data_host1 = (float *) malloc(size_data);
    float *data_dev1;
    float *data_host2 = (float *) malloc(size_data);
    float *data_dev2;

    // Allocate device memory.
    gpuErrChk(cudaMalloc((void **) &weights_dev, size_weights));
    gpuErrChk(cudaMalloc((void **) &data_dev1, size_data));
    gpuErrChk(cudaMalloc((void **) &data_dev2, size_data));

    // Fill the host weights with standard normal data.
    gaussianFill(weights_host, REVIEW_DIM);

    // Copy host memory to device memory.
    gpuErrChk(cudaMemcpy(weights_dev, weights_host, size_weights,
                cudaMemcpyHostToDevice));

    // Create the stream.
    cudaStream_t s[2];
    cudaStreamCreate(&s[0]);
    cudaStreamCreate(&s[1]);

    // Create data in/out arrays
    float *data_devs[2];
    float *data_hosts[2];
    data_devs[0] = data_dev1;
    data_devs[1] = data_dev2;
    data_hosts[0] = data_host1;
    data_hosts[1] = data_host2;

    // Main loop to process input lines (each line corresponds to a review).
    int review_idx = 0;
    // Flag that determines which stream we will use.
    int stream_flag = 0;
    int lsa_offset = REVIEW_DIM + 1;
    for (string review_str; getline(in_stream, review_str); review_idx++) {
        float *data_dev = data_devs[stream_flag];
        float *data_host = data_hosts[stream_flag];

        readLSAReview(review_str,
                data_host + review_idx * lsa_offset, 1);

        // Check if full batch. If so, run kernel.
        if (review_idx == batch_size - 1) {
            review_idx = -1;
            // Host to device.
            gpuErrChk(cudaMemcpyAsync(data_dev, data_host, size_data,
                        cudaMemcpyHostToDevice, s[stream_flag]));
            // Synchronize streams, so that kernels do not run in parallel.
            // Needed so that weights get updated batch-wise.
            gpuErrChk(cudaStreamSynchronize(s[stream_flag ^ 1]));
            // Run kernel.
            //float error = cudaClassify(data_dev, batch_size, STEP_SIZE,
                    //weights_dev, s[stream_flag]);
            //printf("Error = %f\n", error);
            printf("Error = %f\n", 0.0);
            // Change streams.
            stream_flag ^= 1;
        }
    }

    // Synchronize and destroy streams.
    for (int i = 0; i < 2; i++) {
        cudaStreamSynchronize(s[i]);
        cudaStreamDestroy(s[i]);
    }

    // Copy device weights to host.
    gpuErrChk(cudaMemcpy(weights_host, weights_dev, size_weights,
                cudaMemcpyDeviceToHost));
    for (int i = 0; i < REVIEW_DIM; i++) {
        printf("Weight #%d = %f\n", i, weights_host[i]);
    }

    // Free memory on host and device.
    free(weights_host);
    free(data_host1);
    free(data_host2);

    gpuErrChk(cudaFree(weights_dev));
    gpuErrChk(cudaFree(data_dev1));
    gpuErrChk(cudaFree(data_dev2));
}

int main(int argc, char** argv) {
    if (argc != 2) {
		printf("./classify <path to datafile>\n");
		return -1;
    }
    // These functions allow you to select the least utilized GPU
    // on your system as well as enforce a time limit on program execution.
    // Please leave these enabled as a courtesy to your fellow classmates
    // if you are using a shared computer. You may ignore or remove these
    // functions if you are running on your local machine.
    TA_Utilities::select_least_utilized_GPU();
    int max_time_allowed_in_seconds = 100;
    TA_Utilities::enforce_time_limit(max_time_allowed_in_seconds);

    // Init timing
	float time_initial, time_final;

    int batch_size = 1;

	// begin timer
	time_initial = clock();

    ifstream ifs(argv[1]);
    stringstream buffer;
    buffer << ifs.rdbuf();
    classify(buffer, batch_size);

	// End timer
	time_final = clock();
	printf("Total time to run classify: %f (s)\n", (time_final - time_initial) / CLOCKS_PER_SEC);


}
