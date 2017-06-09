/* Udacity Homework 3
   HDR Tone-mapping

  Background HDR
  ==============

  A High Dynamic Range (HDR) image contains a wider variation of intensity
  and color than is allowed by the RGB format with 1 byte per channel that we
  have used in the previous assignment.  

  To store this extra information we use single precision floating point for
  each channel.  This allows for an extremely wide range of intensity values.

  In the image for this assignment, the inside of church with light coming in
  through stained glass windows, the raw input floating point values for the
  channels range from 0 to 275.  But the mean is .41 and 98% of the values are
  less than 3!  This means that certain areas (the windows) are extremely bright
  compared to everywhere else.  If we linearly map this [0-275] range into the
  [0-255] range that we have been using then most values will be mapped to zero!
  The only thing we will be able to see are the very brightest areas - the
  windows - everything else will appear pitch black.

  The problem is that although we have cameras capable of recording the wide
  range of intensity that exists in the real world our monitors are not capable
  of displaying them.  Our eyes are also quite capable of observing a much wider
  range of intensities than our image formats / monitors are capable of
  displaying.

  Tone-mapping is a process that transforms the intensities in the image so that
  the brightest values aren't nearly so far away from the mean.  That way when
  we transform the values into [0-255] we can actually see the entire image.
  There are many ways to perform this process and it is as much an art as a
  science - there is no single "right" answer.  In this homework we will
  implement one possible technique.

  Background Chrominance-Luminance
  ================================

  The RGB space that we have been using to represent images can be thought of as
  one possible set of axes spanning a three dimensional space of color.  We
  sometimes choose other axes to represent this space because they make certain
  operations more convenient.

  Another possible way of representing a color image is to separate the color
  information (chromaticity) from the brightness information.  There are
  multiple different methods for doing this - a common one during the analog
  television days was known as Chrominance-Luminance or YUV.

  We choose to represent the image in this way so that we can remap only the
  intensity channel and then recombine the new intensity values with the color
  information to form the final image.

  Old TV signals used to be transmitted in this way so that black & white
  televisions could display the luminance channel while color televisions would
  display all three of the channels.
  

  Tone-mapping
  ============

  In this assignment we are going to transform the luminance channel (actually
  the log of the luminance, but this is unimportant for the parts of the
  algorithm that you will be implementing) by compressing its range to [0, 1].
  To do this we need the cumulative distribution of the luminance values.

  Example
  -------

  input : [2 4 3 3 1 7 4 5 7 0 9 4 3 2]
  min / max / range: 0 / 9 / 9

  histo with 3 bins: [4 7 3]

  cdf : [4 11 14]


  Your task is to calculate this cumulative distribution by following these
  steps.

*/
#include "utils.h"
#include <stdio.h>

void checkKernelErrors()
{
    checkCudaErrors(cudaPeekAtLastError());
    checkCudaErrors(cudaDeviceSynchronize());
}

__global__
void min_max_reduce_kernel(
    const float* min_in, float* min_out, 
    const float* max_in, float* max_out, 
    const size_t n, const size_t shmOffset)
{
    extern __shared__ float s_data[];

    float* s_min_data = s_data;
    float* s_max_data = &s_data[shmOffset];

    int threadId = threadIdx.x;
    int left = blockIdx.x*(blockDim.x * 2) + threadId;
    int right = left + blockDim.x;

    s_min_data[threadId] = min(min_in[left], min_in[right]);
    s_max_data[threadId] = max(max_in[left], max_in[right]);
    __syncthreads();

    for(unsigned int step = blockDim.x / 2; step > 0; step >>= 1)
    {
        if(threadId < step)
        {
            left = threadId;
            right = threadId + step;

            s_min_data[threadId] = min(s_min_data[left], s_min_data[right]);
            s_max_data[threadId] = max(s_max_data[left], s_max_data[right]);
        }
        __syncthreads();
    }

    if(threadId == 0)
    {
        min_out[blockIdx.x] = s_min_data[threadId]; 
        max_out[blockIdx.x] = s_max_data[threadId];
    }
}

void reduce_min_max(const float* const d_logLuminance, 
            const size_t numRows, const size_t numCols,
            float& min_out, float& max_out)
{
    size_t threads = 256;
    size_t maxBlocks = threads * 2;
    size_t length = numRows * numCols;
    size_t blocks = (length + (maxBlocks - 1))/maxBlocks;
    size_t shmBytes = 2 * threads * sizeof(float);

    float* d_min_in = NULL;
    float* d_min_out = NULL;
    checkCudaErrors(cudaMalloc(&d_min_in, length * sizeof(float)));
    checkCudaErrors(cudaMemcpy(d_min_in, d_logLuminance, length * sizeof(float), cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaMalloc(&d_min_out, blocks * sizeof(float)));
    checkCudaErrors(cudaMemset(d_min_out, 0.0f, blocks * sizeof(float)));

    float* d_max_in = NULL;
    float* d_max_out = NULL;
    checkCudaErrors(cudaMalloc(&d_max_in, length * sizeof(float)));
    checkCudaErrors(cudaMemcpy(d_max_in, d_logLuminance, length * sizeof(float), cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaMalloc(&d_max_out, blocks * sizeof(float)));
    checkCudaErrors(cudaMemset(d_max_out, 0.0f, blocks * sizeof(float)));

    min_max_reduce_kernel<<<blocks, threads, shmBytes>>>(
        d_min_in, d_min_out, 
        d_max_in, d_max_out, 
        length, threads);
    checkKernelErrors();

    while(blocks >= 1)
    {
        length = blocks;
        blocks = (length + ((threads*2) - 1))/(threads*2);
        checkCudaErrors(cudaMemcpy(d_min_in, d_min_out, length * sizeof(float), cudaMemcpyDeviceToDevice));
        checkCudaErrors(cudaMemcpy(d_max_in, d_max_out, length * sizeof(float), cudaMemcpyDeviceToDevice));

        min_max_reduce_kernel<<<blocks, threads, shmBytes>>>(
            d_min_in, d_min_out, 
            d_max_in, d_max_out, 
            length, threads);
        checkKernelErrors();

        if(blocks == 1)
        {
            break;
        }
    }

    float min, max;
    checkCudaErrors(cudaMemcpy(&min, d_min_out, 1*sizeof(float), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(&max, d_max_out, 1*sizeof(float), cudaMemcpyDeviceToHost));

    checkCudaErrors(cudaFree(d_min_in));
    checkCudaErrors(cudaFree(d_min_out));
    checkCudaErrors(cudaFree(d_max_in));
    checkCudaErrors(cudaFree(d_max_out));

    min_out = min;
    max_out = max;
}

// #include <thrust/device_vector.h>
// #include <thrust/transform_reduce.h>
// #include <thrust/extrema.h>
// float thrust_min(const float* const d_in, const size_t length)
// {
//     thrust::device_vector<float> d_x(d_in, d_in + length);
//     return *thrust::min_element(d_x.begin(), d_x.end());
// }
// float thrust_max(const float* const d_in, const size_t length)
// {
//     thrust::device_vector<float> d_x(d_in, d_in + length);
//     return *thrust::max_element(d_x.begin(), d_x.end());
// }

void your_histogram_and_prefixsum(const float* const d_logLuminance,
                                  unsigned int* const d_cdf,
                                  float &min_logLum,
                                  float &max_logLum,
                                  const size_t numRows,
                                  const size_t numCols,
                                  const size_t numBins)
{
  //TODO
  /*Here are the steps you need to implement
    1) find the minimum and maximum value in the input logLuminance channel
       store in min_logLum and max_logLum
    2) subtract them to find the range
    3) generate a histogram of all the values in the logLuminance channel using
       the formula: bin = (lum[i] - lumMin) / lumRange * numBins
    4) Perform an exclusive scan (prefix sum) on the histogram to get
       the cumulative distribution of luminance values (this should go in the
       incoming d_cdf pointer which already has been allocated for you)       */

    reduce_min_max(d_logLuminance, numRows, numCols, min_logLum, max_logLum);
    printf("min: %f max: %f\n", min_logLum, max_logLum);
    // printf("max: %f max: %f\n", thrust_min(d_logLuminance, numRows * numCols), thrust_max(d_logLuminance, numRows * numCols)); 
}
