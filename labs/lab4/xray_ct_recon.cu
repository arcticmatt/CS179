
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

/*
 * Hi-pass filter for sinogram frequencies. Highest frequencies are in the
 * middle, lowest are at the sides.
 */
__global__
void
cudaFreqScaleKernel(cufftComplex *sinogram_data, const unsigned int sinogram_width,
        const unsigned int sinogram_length) {
    // Get current thread's index.
    unsigned int thread_index = blockIdx.x * blockDim.x + threadIdx.x;

    // Relative distance w/in sinogram.
    unsigned int relative;
    // Scaling factor.
    float sfactor;
    const unsigned int half_width = sinogram_width / 2;

    while (thread_index < sinogram_length) {
        // Calculate scaling factor. Scale by distance from the center, where
        // the highest frequencies (in the center) are scaled by 1 and the
        // lowest frequencies (at the sides) are scaled by 0.
        relative = thread_index % sinogram_width;
        if (relative < half_width) {
            sfactor = (float) (relative) / half_width;
        } else {
            sfactor = (float) (sinogram_width - relative) / half_width;
        }

        // Perform the frequency scaling.
        sinogram_data[thread_index].x *= sfactor;
        sinogram_data[thread_index].y *= sfactor;

        // Update thread_index.
        thread_index += blockDim.x * gridDim.x;
    }
}

void cudaCallFrequencyScaleKernel(const unsigned int blocks,
        const unsigned int threadsPerBlock,
        cufftComplex *sinogram_data,
        const unsigned int sinogram_width,
        const unsigned int sinogram_length) {
    /* Call the frequency scaling kernel. */
    cudaFreqScaleKernel<<<blocks, threadsPerBlock>>>(sinogram_data,
            sinogram_width, sinogram_length);
}

/*
 * Extract real components of complex data into an array of floats.
 */
__global__
void
cudaExtractRealKernel(cufftComplex *sinogram_cmplx, float *sinogram_real,
        const unsigned int sinogram_length) {
    // Get current thread's index.
    unsigned int thread_index = blockIdx.x * blockDim.x + threadIdx.x;

    while (thread_index < sinogram_length) {
        // Copy over real component.
        sinogram_real[thread_index] = sinogram_cmplx[thread_index].x;

        // Update thread_index.
        thread_index += blockDim.x * gridDim.x;
    }
}

void cudaCallExtractRealKernel(const unsigned int blocks,
        const unsigned int threadsPerBlock,
        cufftComplex *sinogram_cmplx,
        float *sinogram_real,
        const unsigned int sinogram_length) {
    /* Call the extract real kernel. */
    cudaExtractRealKernel<<<blocks, threadsPerBlock>>>(sinogram_cmplx,
            sinogram_real, sinogram_length);
}


/*
 * Backprojection kernel. Reconstruct the image given the passed-in
 * sinogram data.
 */
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

        assert(output_data[thread_index] == 0);
        // Don't parallelize sinogram iteration.
        for (int i = 0; i < nAngles; i++) {
            theta = PI * ((float) i / nAngles);
            // Handle edge cases. Handle float imprecision.
            if (fabs(theta) < EPSILON) {
                d = x_cent;
            } else if (fabs(theta - (PI / 2.0)) < EPSILON) {
                d = y_cent;
            } else {
                m = -cosf(theta) / sinf(theta);
                q = -1.0 / m;
                xi = (y_cent - m * x_cent) / (q - m);
                yi = q * xi;
                d = sqrtf(xi * xi + yi * yi);
                // Check if the distance should be negative.
                if ((q > 0 && xi < 0) || (q < 0 && xi > 0))
                    d *= -1.0;
            }
            sinogram_index = d;
            assert(abs(sinogram_index) <= sinogram_width / 2);
            // Adjust index, needs to be btwn 0 and sinogram_width.
            sinogram_index += sinogram_width / 2;
            output_data[thread_index] +=
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

    /* DONE: Allocate memory for all GPU storage above, copy input sinogram
    over to dev_sinogram_cmplx. */
    const unsigned int sinogram_length = sinogram_width * nAngles;
    gpuErrchk(cudaMalloc((void **) &dev_sinogram_cmplx,
                sinogram_length * sizeof(cufftComplex)));
    gpuErrchk(cudaMalloc((void **) &dev_sinogram_float,
                sinogram_length * sizeof(cufftReal)));

    gpuErrchk(cudaMemcpy(dev_sinogram_cmplx, sinogram_host,
                sinogram_length * sizeof(cufftComplex),
                cudaMemcpyHostToDevice));


    /* DONE 1: Implement the high-pass filter:
        - Use cuFFT for the forward FFT
        - Create your own kernel for the frequency scaling.
        - Use cuFFT for the inverse FFT
        - extract real components to floats
        - Free the original sinogram (dev_sinogram_cmplx)

        Note: If you want to deal with real-to-complex and complex-to-real
        transforms in cuFFT, you'll have to slightly change our code above.
    */
    // Make a plan for C2C.
    cufftHandle plan;
    cufftPlan2d(&plan, nAngles, sinogram_width, CUFFT_C2C);

    // Run forward DFT on the sinogram data (do this in-place). Use C2C.
    assert(cufftExecC2C(plan, dev_sinogram_cmplx, dev_sinogram_cmplx, CUFFT_FORWARD) == CUFFT_SUCCESS);

    // Frequency scaling.
    cudaCallFrequencyScaleKernel(nBlocks, threadsPerBlock, dev_sinogram_cmplx,
            sinogram_width, sinogram_length);

    // Run inverse DFT on the sinogram data (do this in-place). Use C2C.
    assert(cufftExecC2C(plan, dev_sinogram_cmplx, dev_sinogram_cmplx, CUFFT_INVERSE) == CUFFT_SUCCESS);

    // Extract the real components to floats.
    cudaCallExtractRealKernel(nBlocks, threadsPerBlock, dev_sinogram_cmplx,
            dev_sinogram_float, sinogram_length);


    /* DONE 2: Implement backprojection.
        - Allocate memory for the output image.
        - Create your own kernel to accelerate backprojection.
        - Copy the reconstructed image back to output_host.
        - Free all remaining memory on the GPU.
    */
    // Allocate and clear memory.
    gpuErrchk(cudaMalloc((void **) &output_dev, size_result));
    gpuErrchk(cudaMemset(output_dev, 0, size_result));

    // Backprojection.
    cudaCallBackprojectKernel(nBlocks, threadsPerBlock, output_dev,
            dev_sinogram_float, width, height, nAngles, sinogram_width);

    // Copy reconstructed image back to output_host.
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



