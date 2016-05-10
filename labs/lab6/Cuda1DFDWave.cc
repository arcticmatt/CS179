/* CUDA finite difference wave equation solver, written by
 * Jeff Amelang, 2012
 *
 * Modified by Kevin Yuh, 2013-14 */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <fstream>


#include <cuda_runtime.h>
#include <algorithm>

#include "Cuda1DFDWave_cuda.cuh"
#include "ta_utilities.hpp"





int main(int argc, char* argv[]) {
  // These functions allow you to select the least utilized GPU
  // on your system as well as enforce a time limit on program execution.
  // Please leave these enabled as a courtesy to your fellow classmates
  // if you are using a shared computer. You may ignore or remove these
  // functions if you are running on your local machine.
  TA_Utilities::select_least_utilized_GPU();
  int max_time_allowed_in_seconds = 40;
  TA_Utilities::enforce_time_limit(max_time_allowed_in_seconds);

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
      printf("Processing timestep %8zu (%5.1f%%)\n",
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

  {
  
  
    const unsigned int blocks = std::min(maxBlocks, (unsigned int) ceil(
                numberOfNodes/float(threadsPerBlock)));
  
    //Space on the CPU to copy file data back from GPU
    float *file_output = new float[numberOfNodes];

    /* TODO: Create GPU memory for your calculations. 
    As an initial condition at time 0, zero out your memory as well. */
    
    
    // Looping through all times t = 0, ..., t_max
    for (size_t timestepIndex = 0; timestepIndex < numberOfTimesteps;
            ++timestepIndex) {
        
        if (timestepIndex % (numberOfTimesteps / 10) == 0) {
            printf("Processing timestep %8zu (%5.1f%%)\n",
                 timestepIndex, 100 * timestepIndex / float(numberOfTimesteps));
        }
        
        
        /* TODO: Call a kernel to solve the problem (you'll need to make
        the kernel in the .cu file) */

        
        //Left boundary condition on the CPU - a sum of sine waves
        const float t = timestepIndex * dt;
        float left_boundary_value;
        if (omega0 * t < 2 * M_PI) {
            left_boundary_value = 0.8 * sin(omega0 * t) + 0.1 * sin(omega1 * t);
        } else {
            left_boundary_value = 0;
        }
        
        
        /* TODO: Apply left and right boundary conditions on the GPU. 
        The right boundary conditon will be 0 at the last position
        for all times t */
        
        // Check if we need to write a file
        if (CUDATEST_WRITE_ENABLED == true && numberOfOutputFiles > 0 &&
                (timestepIndex+1) % (numberOfTimesteps / numberOfOutputFiles) 
                == 0) {
            
            
            /* TODO: Copy data from GPU back to the CPU in file_output */
            
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
    
    
    /* TODO: Clean up GPU memory */
  
  
}
  
  printf("You can now turn the output files into pictures by running "
         "\"python makePlots.py\". It should produce png files in the output "
         "directory. (You'll need to have gnuplot in order to produce the "
         "files - either install it, or run on the remote machines.)\n");

  return 0;
}
