/* CUDA finite difference wave equation solver, written by
 * Jeff Amelang, 2012
 *
 * Modified by Kevin Yuh, 2013-14 */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <fstream>
#include <cassert>


#include <cuda_runtime.h>
#include <algorithm>

#include "Cuda1DFDWave_cuda.cuh"
#include "ta_utilities.hpp"

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


int main(int argc, char* argv[]) {
  if (argc < 3){
      printf("Usage: (threads per block) (max number of blocks)\n");
      exit(-1);
  }


  // make sure output directory exists
  std::ifstream test("output");
  if ((bool)test == false) {
    printf("Cannot find output directory, please make it (\"mkdir output\")\n");
    exit(1);
  }

  /* Additional parameters for the assignment */

  const bool CUDATEST_WRITE_ENABLED = true;   //enable writing files
  const unsigned int threadsPerBlock = atoi(argv[1]);
  const unsigned int maxBlocks = atoi(argv[2]);



  // Parameters regarding our simulation
  const size_t numberOfIntervals = 1e5;
  const size_t numberOfTimesteps = 1e5;
  const size_t numberOfOutputFiles = 3;

  //Parameters regarding our initial wave
  const float courant = 1.0;
  const float omega0 = 10;
  const float omega1 = 100;

  // derived
  const size_t numberOfNodes = numberOfIntervals + 1;
  const float courantSquared = courant * courant;
  const float dx = 1./numberOfIntervals;
  const float dt = courant * dx;




  /************************* CPU Implementation *****************************/


  // make 3 copies of the domain for old, current, and new displacements
  float ** data = new float*[3];
  for (unsigned int i = 0; i < 3; ++i) {
    // make a copy
    data[i] = new float[numberOfNodes];
    // fill it with zeros
    std::fill(&data[i][0], &data[i][numberOfNodes], 0);
  }

  for (size_t timestepIndex = 0; timestepIndex < numberOfTimesteps;
       ++timestepIndex) {
    if (timestepIndex % (numberOfTimesteps / 10) == 0) {
      printf("CPU: Processing timestep %8zu (%5.1f%%)\n",
             timestepIndex, 100 * timestepIndex / float(numberOfTimesteps));
    }

    // nickname displacements
    const float * oldDisplacements =     data[(timestepIndex - 1) % 3];
    const float * currentDisplacements = data[(timestepIndex + 0) % 3];
    float * newDisplacements =           data[(timestepIndex + 1) % 3];

    for (unsigned int a = 1; a <= numberOfNodes - 2; ++a){
        newDisplacements[a] =
                2*currentDisplacements[a] - oldDisplacements[a]
                + courantSquared * (currentDisplacements[a+1]
                        - 2*currentDisplacements[a]
                        + currentDisplacements[a-1]);
    }


    // apply wave boundary condition on the left side, specified above
    const float t = timestepIndex * dt;
    if (omega0 * t < 2 * M_PI) {
      newDisplacements[0] = 0.8 * sin(omega0 * t) + 0.1 * sin(omega1 * t);
    } else {
      newDisplacements[0] = 0;
    }

    // apply y(t) = 0 at the rightmost position
    newDisplacements[numberOfNodes - 1] = 0;


    // enable this is you're having troubles with instabilities
#if 0
    // check health of the new displacements
    for (size_t nodeIndex = 0; nodeIndex < numberOfNodes; ++nodeIndex) {
      if (std::isfinite(newDisplacements[nodeIndex]) == false ||
          std::abs(newDisplacements[nodeIndex]) > 2) {
        printf("Error: bad displacement on timestep %zu, node index %zu: "
               "%10.4lf\n", timestepIndex, nodeIndex,
               newDisplacements[nodeIndex]);
      }
    }
#endif

    // if we should write an output file
    if (numberOfOutputFiles > 0 &&
        (timestepIndex+1) % (numberOfTimesteps / numberOfOutputFiles) == 0) {
      printf("writing an output file\n");
      // make a filename
      char filename[500];
      sprintf(filename, "output/CPU_data_%08zu.dat", timestepIndex);
      // write output file
      FILE* file = fopen(filename, "w");
      for (size_t nodeIndex = 0; nodeIndex < numberOfNodes; ++nodeIndex) {
        fprintf(file, "%e,%e\n", nodeIndex * dx,
                newDisplacements[nodeIndex]);
      }
      fclose(file);
    }
  }



  /************************* GPU Implementation *****************************/

  int num_devices = 0;
  gpuErrchk(cudaGetDeviceCount(&num_devices));
  assert(num_devices > 0);
  const unsigned int comm_interval = 3;

  {


    const unsigned int blocks = std::min(maxBlocks, (unsigned int) ceil(
                numberOfNodes/float(threadsPerBlock)));

    //Space on the CPU to copy file data back from GPU
    float *file_output = new float[numberOfNodes];

    /* DONE: Create GPU memory for your calculations.
    As an initial condition at time 0, zero out your memory as well. */
    int orig_data_length = numberOfNodes * 3;
    // There are two garbage regions per device, each with a size of 3.
    int redundant_data_length = comm_interval * num_devices * 2;
    int data_length = orig_data_length + redundant_data_length;
    int reg_dev_data_length = (numberOfNodes / num_devices) + comm_interval * 2;
    float *data_dev;
    printf("data_dev = %p\n", data_dev);
    gpuErrchk(cudaMalloc((void **) &data_dev, data_length * sizeof(float)));
    printf("data_dev = %p\n", data_dev);
    gpuErrchk(cudaMemset(data_dev, 0, data_length * sizeof(float)));

    // Looping through all times t = 0, ..., t_max
    for (size_t timestepIndex = 0; timestepIndex < numberOfTimesteps;
            ++timestepIndex) {

        if (timestepIndex % (numberOfTimesteps / 10) == 0) {
            printf("GPU: Processing timestep %8zu (%5.1f%%)\n",
                 timestepIndex, 100 * timestepIndex / float(numberOfTimesteps));
        }

        //Left boundary condition on the CPU - a sum of sine waves
        const float t = timestepIndex * dt;
        float left_boundary_value;
        if (omega0 * t < 2 * M_PI) {
            left_boundary_value = 0.8 * sin(omega0 * t) + 0.1 * sin(omega1 * t);
        } else {
            left_boundary_value = 0;
        }

        // Create pointers to old, current, and new displacements.
        float *old_displacements_dev = data_dev +
            ((timestepIndex - 1) % 3) * numberOfNodes;
        float *current_displacements_dev = data_dev +
            ((timestepIndex + 0) % 3) * numberOfNodes;
        float *new_displacements_dev = data_dev +
            ((timestepIndex + 1) % 3) * numberOfNodes;

        // If we're at the correct timestep, perform data exchanges. This
        // ensures that multiple gpus can run on the data.
        float *curr_old_displacements[2];
        curr_old_displacements[0] = old_displacements_dev;
        curr_old_displacements[1] = current_displacements_dev;
        if (timestepIndex % comm_interval == 0) {
            for (int i = 0; i < 2; i++) {
                float *displacements = curr_old_displacements[i];
                for (int dev_number = 0; dev_number < num_devices - 1; dev_number++) {
                    // dev_number = left, dev_number + 1 = right.
                    // Num bytes before left block.
                    int left_prev = reg_dev_data_length * dev_number;
                    float *left_redundant = displacements +
                        (left_prev + comm_interval + numberOfNodes);
                    float *left_good = left_redundant - comm_interval;
                    // Num bytes before right block.
                    int right_prev = left_prev + reg_dev_data_length;
                    float *right_redundant = displacements + right_prev;
                    float *right_good = right_redundant + comm_interval;
                    // Copy right-to-left.
                    gpuErrchk(cudaMemcpyPeer(left_redundant, dev_number, right_good,
                            dev_number + 1, comm_interval * sizeof(float)));
                    // Copy left-to-right.
                    gpuErrchk(cudaMemcpyPeer(right_redundant, dev_number + 1,
                            left_good, dev_number, comm_interval * sizeof(float)));
                }
            }
        }

        // Multi-gpu implementation.
        // Iterate through each device and call the kernel on a different gpu.
        int dev_number_of_nodes = 0;
        int dev_length = numberOfNodes + (comm_interval * 2 * num_devices);
        for (int dev_number = 0; dev_number < num_devices; dev_number++) {
            // Set device for the kernel to run on.
            gpuErrchk(cudaSetDevice(dev_number));

            if (dev_number < num_devices - 1) {
                // If we're not on the last device...
                dev_number_of_nodes = reg_dev_data_length;
            } else {
                dev_number_of_nodes = dev_length -
                    (reg_dev_data_length * (num_devices - 1));
            }

            // Call the kernel, which sets the boundary values.
            cudaCallWaveSolverKernel(blocks, threadsPerBlock, old_displacements_dev,
                    current_displacements_dev, new_displacements_dev,
                    dev_number_of_nodes, courant, left_boundary_value);

            // Increment pointers in preparation for next kernel call.
            old_displacements_dev += dev_number_of_nodes;
            current_displacements_dev += dev_number_of_nodes;
            new_displacements_dev += dev_number_of_nodes;
        }

        // Reset pointers
        old_displacements_dev = data_dev +
            ((timestepIndex - 1) % 3) * numberOfNodes;
        current_displacements_dev = data_dev +
            ((timestepIndex + 0) % 3) * numberOfNodes;
        new_displacements_dev = data_dev +
            ((timestepIndex + 1) % 3) * numberOfNodes;


        // Check if we need to write a file
        if (CUDATEST_WRITE_ENABLED == true && numberOfOutputFiles > 0 &&
                (timestepIndex+1) % (numberOfTimesteps / numberOfOutputFiles)
                == 0) {


            /* DONE: Copy data from GPU back to the CPU in file_output */
            gpuErrchk(cudaMemcpy(file_output, current_displacements_dev,
                        numberOfNodes * sizeof(float), cudaMemcpyDeviceToHost));

            printf("writing an output file\n");
            // make a filename
            char filename[500];
            sprintf(filename, "output/GPU_data_%08zu.dat", timestepIndex);
            // write output file
            FILE* file = fopen(filename, "w");
            for (size_t nodeIndex = 0; nodeIndex < numberOfNodes; ++nodeIndex) {
                fprintf(file, "%e,%e\n", nodeIndex * dx,
                        file_output[nodeIndex]);
            }
            fclose(file);
        }

    }


    /* DONE: Clean up GPU memory */
    cudaFree(data_dev);
}

  printf("You can now turn the output files into pictures by running "
         "\"python makePlots.py\". It should produce png files in the output "
         "directory. (You'll need to have gnuplot in order to produce the "
         "files - either install it, or run on the remote machines.)\n");

  return 0;
}
