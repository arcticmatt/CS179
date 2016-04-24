
/*
Based off work by Nelson, et al.
Brigham Young University (2010)

Adapted by Kevin Yuh (2015)

Modified by Jordan Bonilla and Matthew Cedeno (2016)
*/


#include <stdio.h>
#include <cuda.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <cufft.h>
#include <time.h>
#include <limits.h>
#include <math.h>
#include "ta_utilities.hpp"

#define PI 3.14159265358979
#define EPSILON .0001


/* Check errors on CUDA runtime functions */
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        exit(code);
    }
}



/* Check errors on cuFFT functions */
void gpuFFTchk(int errval){
    if (errval != CUFFT_SUCCESS){
        printf("Failed FFT call, error code %d\n", errval);
    }
}


/* Check errors on CUDA kernel calls */
void checkCUDAKernelError()
{
    cudaError_t err = cudaGetLastError();
    if  (cudaSuccess != err){
        fprintf(stderr, "Error %s\n", cudaGetErrorString(err));
    } else {
        fprintf(stderr, "No kernel error detected\n");
    }

}

__global__
void
cudaFreqScaleKernel(cufftComplex *sinogram_data, const float ramp_slope,
        const float ramp_intercept, const unsigned int sinogram_length) {
    // Get current thread's index.
    unsigned int thread_index = blockIdx.x * blockDim.x + threadIdx.x;

    while (thread_index < sinogram_length) {
        // Perform the frequency scaling. Only scale the real component, since
        // that is what we extract later.
        sinogram_data[thread_index].x *=
            ramp_slope * sinogram_data[thread_index].x + ramp_intercept;

        // Update thread_index.
        thread_index += blockDim.x * gridDim.x;
    }
}

void cudaCallFrequencyScaleKernel(const unsigned int blocks,
        const unsigned int threadsPerBlock,
        cufftComplex *sinogram_data,
        const float ramp_slope,
        const float ramp_intercept,
        const unsigned int sinogram_length) {
    /* Call the frequency scaling kernel. */
    cudaFreqScaleKernel<<<blocks, threadsPerBlock>>>(sinogram_data, ramp_slope,
            ramp_intercept, sinogram_length);
}

__global__
void
cudaBackprojectKernel(float *output_data, const float *sinogram_data,
        const unsigned int width, const unsigned int height,
        const unsigned int nAngles, const unsigned int sinogram_width) {
    // Calculate output length.
    const unsigned int output_length = width * height;

    // Get current thread's index.
    unsigned int thread_index = blockIdx.x * blockDim.x + threadIdx.x;

    // Absolute coordinates.
    unsigned int x;
    unsigned int y;
    // Centered coordinates.
    int x_cent;
    int y_cent;
    // Calculated coordinates.
    float xi;
    float yi;
    // Helper variables for calculations.
    float m, q, d, theta;
    int sinogram_index;

    while (thread_index < output_length) {
        x = thread_index % width;
        y = thread_index / width;
        // Calculate centered coordinates.
        x_cent = x - width / 2;
        y_cent = -y + height / 2;

        // Don't parallelize sinogram iteration.
        for (int i = 0; i < nAngles; i++) {
            theta = 2 * PI * i / nAngles;
            // Handle edge cases. Handle float imprecision.
            if (fabs(theta) < EPSILON) {
                d = x_cent;
                sinogram_index = d;
            } else if (fabs(theta - (PI / 2.0)) < EPSILON) {
                d = y_cent;
                sinogram_index = d;
            } else {
                m = -cosf(theta) / sinf(theta);
                q = -1 / m;
                xi = (y_cent - m * x_cent) / (q - m);
                yi = q * xi;
                d = sqrt(xi * xi + yi * yi);
                // Use absolute coordinates when indexing into output.
                if ((q > 0 && xi < 0) || (q < 0 && xi > 0))
                    sinogram_index = -d;
                else
                    sinogram_index = d;
            }
            sinogram_index += sinogram_width / 2;
            output_data[y * width + x] +=
                sinogram_data[i * sinogram_width + sinogram_index];
        }

        // Update thread_index.
        thread_index += blockDim.x * gridDim.x;
    }
}

void cudaCallBackprojectKernel(const unsigned int blocks,
        const unsigned int threadsPerBlock,
        float *output_data,
        const float *sinogram_data,
        const unsigned int width,
        const unsigned int height,
        const unsigned int nAngles,
        const unsigned int sinogram_width) {
    /* Call the backprojection kernel. */
    cudaBackprojectKernel<<<blocks, threadsPerBlock>>>(output_data, sinogram_data,
            width, height, nAngles, sinogram_width);
}


int main(int argc, char** argv){
    // These functions allow you to select the least utilized GPU
    // on your system as well as enforce a time limit on program execution.
    // Please leave these enabled as a courtesy to your fellow classmates
    // if you are using a shared computer. You may ignore or remove these
    // functions if you are running on your local machine.
    TA_Utilities::select_least_utilized_GPU();
    int max_time_allowed_in_seconds = 10;
    TA_Utilities::enforce_time_limit(max_time_allowed_in_seconds);

    // Begin timer and check for the correct number of inputs
    time_t start = clock();
    if (argc != 7){
        fprintf(stderr, "Incorrect number of arguments.\n\n");
        fprintf(stderr, "\nArguments: \n \
        < Input sinogram text file's name > \n \
        < Width or height of original image, whichever is larger > \n \
        < Number of angles in sinogram >\n \
        < threads per block >\n \
        < number of blocks >\n \
        < output text file's name >\n");
        exit(EXIT_FAILURE);
    }






    /********** Parameters **********/

    int width = atoi(argv[2]);
    int height = width;
    int sinogram_width = (int)ceilf( height * sqrt(2) );

    int nAngles = atoi(argv[3]);


    int threadsPerBlock = atoi(argv[4]);
    int nBlocks = atoi(argv[5]);


    /********** Data storage *********/


    // GPU DATA STORAGE
    cufftComplex *dev_sinogram_cmplx;
    cufftReal *dev_sinogram_float;
    float* output_dev;  // Image storage


    cufftComplex *sinogram_host;

    size_t size_result = width*height*sizeof(float);
    float *output_host = (float *)malloc(size_result);

    /*********** Set up IO, Read in data ************/

    sinogram_host = (cufftComplex *)malloc(  sinogram_width*nAngles*sizeof(cufftComplex) );

    FILE *dataFile = fopen(argv[1],"r");
    if (dataFile == NULL){
        fprintf(stderr, "Sinogram file missing\n");
        exit(EXIT_FAILURE);
    }

    FILE *outputFile = fopen(argv[6], "w");
    if (outputFile == NULL){
        fprintf(stderr, "Output file cannot be written\n");
        exit(EXIT_FAILURE);
    }

    int j, i;

    for(i = 0; i < nAngles * sinogram_width; i++){
        fscanf(dataFile,"%f",&sinogram_host[i].x);
        sinogram_host[i].y = 0;
    }

    fclose(dataFile);


    /*********** Assignment starts here *********/

    /* TODO: Allocate memory for all GPU storage above, copy input sinogram
    over to dev_sinogram_cmplx. */
    const unsigned int sinogram_length = sinogram_width * nAngles;
    gpuErrchk(cudaMalloc((void **) &dev_sinogram_cmplx,
                sinogram_length * sizeof(cufftComplex)));
    gpuErrchk(cudaMalloc((void **) &dev_sinogram_float,
                sinogram_length * sizeof(cufftReal)));

    gpuErrchk(cudaMemcpy(dev_sinogram_cmplx, sinogram_host,
                sinogram_length * sizeof(cufftComplex),
                cudaMemcpyHostToDevice));


    /* TODO 1: Implement the high-pass filter:
        - Use cuFFT for the forward FFT
        - Create your own kernel for the frequency scaling.
        - Use cuFFT for the inverse FFT
        - extract real components to floats
        - Free the original sinogram (dev_sinogram_cmplx)

        Note: If you want to deal with real-to-complex and complex-to-real
        transforms in cuFFT, you'll have to slightly change our code above.
    */
    cufftHandle plan;
    int batch = 1;
    cufftPlan1d(&plan, sinogram_length, CUFFT_C2C, batch);

    // Run forward DFT on the sinogram data (do this in-place).
    // TODO: Should this be RDC?
    cufftExecC2C(plan, dev_sinogram_cmplx, dev_sinogram_cmplx, CUFFT_FORWARD);

    // Find min/max freq on CPU. TODO: complex part?
    float min_freq = INT_MAX;
    float max_freq = INT_MIN;
    float freq;
    for(int i = 0; i < nAngles * sinogram_width; i++){
        freq = sinogram_host[i].x;
        if (freq < min_freq)
            min_freq = freq;
        if (freq > max_freq)
            max_freq = freq;
    }

    // Find slope/intercept of ramp scale.
    const float ramp_slope = max_freq - min_freq;
    const float ramp_intercept = ramp_slope * -min_freq;

    // Frequency scaling.
    cudaCallFrequencyScaleKernel(nBlocks, threadsPerBlock, dev_sinogram_cmplx,
            ramp_slope, ramp_intercept, sinogram_length);

    // Run inverse DFT on the sinogram data (do this in-place). Use C2R
    // in order to extract floats to real components.
    // TODO: symmetric input?
    cufftExecC2R(plan, dev_sinogram_cmplx, dev_sinogram_float);


    /* TODO 2: Implement backprojection.
        - Allocate memory for the output image.
        - Create your own kernel to accelerate backprojection.
        - Copy the reconstructed image back to output_host.
        - Free all remaining memory on the GPU.
    */
    printf("About to do backprojection...\n");
    gpuErrchk(cudaMalloc((void **) &output_dev, size_result));
    // TODO: initialize output_dev to all zeros?
    printf("Memset device...\n");
    gpuErrchk(cudaMemset(output_dev, 0, size_result));
    printf("Running backprojection kernel...\n");
    cudaCallBackprojectKernel(nBlocks, threadsPerBlock, output_dev,
            dev_sinogram_float, width, height, nAngles, sinogram_width);
    printf("Copying device to host...\n");
    gpuErrchk(cudaMemcpy(output_host, output_dev, size_result,
                cudaMemcpyDeviceToHost));

    /* Export image data. */

    printf("Exporting image data...\n");
    for(j = 0; j < width; j++){
        for(i = 0; i < height; i++){
            fprintf(outputFile, "%e ",output_host[j*width + i]);
        }
        fprintf(outputFile, "\n");
    }


    /* Cleanup: Free host memory, close files. */
    cudaFree(dev_sinogram_cmplx);
    cudaFree(dev_sinogram_float);
    cudaFree(output_dev);

    free(sinogram_host);
    free(output_host);

    fclose(outputFile);
    printf("CT reconstruction complete. Total run time: %f seconds\n", (float) (clock() - start) / 1000.0);
    return 0;
}



