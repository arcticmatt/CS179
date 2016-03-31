/* 
 * CUDA blur
 * Kevin Yuh, 2014 
 * Revised by Nailen Matschke, 2016
 */

#ifndef BLUR_DEVICE_CUH
#define BLUR_DEVICE_CUH

void cudaCallBlurKernel(const unsigned int blocks,
        const unsigned int threadsPerBlock,
        const float *raw_data,
        const float *blur_v,
        float *out_data,
        const unsigned int n_frames,
        const unsigned int blur_v_size);

#endif
