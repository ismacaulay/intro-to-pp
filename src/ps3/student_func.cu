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

void checkKernelErrors()
{
    checkCudaErrors(cudaPeekAtLastError());
    checkCudaErrors(cudaDeviceSynchronize());
}

__global__
void min_max_reduce_kernel(
    const float* min_in, float* min_out,
    const float* max_in, float* max_out,
    const size_t n)
{
    extern __shared__ float s_data[];

    /*
    shared memory size: 2 * #threads

    [------------------shared memory---------------------]
    [--------min data---------][--------max data---------]
    */
    float* s_min_data = s_data;
    float* s_max_data = &s_data[blockDim.x];

    /*
    Allocate the shared memory using 2 * block size. This way we do the first reduction step
    right now and process double the values per step.

    Step 1:
    [---------------------------------------in--------------------------------------]
    [--------blockDim---------][--------blockDim---------]
    [--------reduced shm------]
    */
    int threadId = threadIdx.x;
    int left = blockIdx.x*(blockDim.x * 2) + threadId;
    int right = left + blockDim.x;

    // if right < n, then left < n so we should compute the min/max values
    if(right < n)
    {
        s_min_data[threadId] = min(min_in[left], min_in[right]);
        s_max_data[threadId] = max(max_in[left], max_in[right]);
    }
    // if left < n but right > n, then the left must be the best value
    else if(left < n)
    {
        s_min_data[threadId] = min_in[left];
        s_max_data[threadId] = max_in[left];

    }
    __syncthreads();

    /*
    Now we start with the shared memory data we just allocated, split it in half
    and compare the values. Each iteration we cut the step in half.

    Step 2: Repeat
    [--------values ------]
    [---step--][---step---]
    [-reduced-]
    */
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

    /*
    We reduced down to a single min/max value, so write that value to out.
    We only need to do this once though
    */
    if(threadId == 0)
    {
        min_out[blockIdx.x] = s_min_data[0];
        max_out[blockIdx.x] = s_max_data[0];
    }
}

void reduce_min_max(
    const float* const d_logLuminance, size_t length,
    float& min_out, float& max_out)
{
    size_t threads = 256;
    size_t maxBlocks = threads * 2;
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
        length);
    checkKernelErrors();

    while(blocks > 1)
    {
        length = blocks;
        blocks = (length + ((threads*2) - 1))/(threads*2);
        checkCudaErrors(cudaMemcpy(d_min_in, d_min_out, length * sizeof(float), cudaMemcpyDeviceToDevice));
        checkCudaErrors(cudaMemcpy(d_max_in, d_max_out, length * sizeof(float), cudaMemcpyDeviceToDevice));

        min_max_reduce_kernel<<<blocks, threads, shmBytes>>>(
            d_min_in, d_min_out,
            d_max_in, d_max_out,
            length);
        checkKernelErrors();
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

__global__
void histogram_kernel(
    const float* in, size_t n,
    unsigned int* const out, size_t bins,
    float lumMin, float lumRange)
{
    int index = blockIdx.x*(blockDim.x) + threadIdx.x;
    int bin = ((in[index] - lumMin) / lumRange) * bins;
    atomicAdd(out+bin, 1);
}

void generate_histogram(
    const float* const d_in, size_t length,
    unsigned int * const d_histogram, size_t numBins,
    float lumMin, float lumMax)
{
    size_t threads = 256;
    size_t blocks = (length + (threads - 1))/threads;

    histogram_kernel<<<blocks, threads>>>(d_in, length, d_histogram, numBins, lumMin, lumMax-lumMin);
    checkKernelErrors();
}

__global__
void exclusive_sum_scan_kernel(unsigned int* const in, unsigned int* const sums, size_t length)
{
    extern __shared__ unsigned int temp[];

    /*
    shared memory size: 2 * #threads
    */
    int localIndex = 2 * threadIdx.x;
    int globalIndex = 2 * blockDim.x * blockIdx.x + localIndex;

    temp[localIndex] = in[globalIndex];
    temp[localIndex+1] = in[globalIndex+1];

    int offset = 1;
    for(int d = blockDim.x; d > 0; d = d/2)
    {
        __syncthreads();
        if(threadIdx.x < d)
        {
            int index1 = offset * (localIndex+1) - 1;
            int index2 = offset * (localIndex+2) - 1;

            temp[index2] = temp[index2] + temp[index1];
        }
        offset *= 2;
    }

    if(threadIdx.x == 0)
    {
        if(sums)
            sums[blockIdx.x] = temp[length-1];
        temp[length - 1] = 0;
    }

    for(int d = 1; d < blockDim.x*2; d*=2)
    {
        offset /= 2;
        __syncthreads();
        if(threadIdx.x < d)
        {
            int index1 = offset * (localIndex+1) - 1;
            int index2 = offset * (localIndex+2) - 1;

            unsigned int t = temp[index1];
            temp[index1] = temp[index2];
            temp[index2] = temp[index2] + t;
        }
    }

    __syncthreads();

    in[globalIndex] = temp[localIndex];
    in[globalIndex+1] = temp[localIndex+1];
}

__global__
void sum_scan_addition_kernel(unsigned int* const in, unsigned int* sums, size_t length)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    int sumIndex = blockIdx.x;

    if(index < length)
    {
        in[index] = in[index] + sums[sumIndex];
    }
}

void exclusive_sum_scan(unsigned int* const d_in, size_t length)
{
    size_t threads = 1024;
    size_t blocks = (length + (threads - 1))/threads;
    size_t shared = threads * sizeof(unsigned int);

    unsigned int* d_sums = NULL;
    checkCudaErrors(cudaMalloc(&d_sums, blocks * sizeof(unsigned int)));
    checkCudaErrors(cudaMemset(d_sums, 0, blocks * sizeof(unsigned int)));

    exclusive_sum_scan_kernel<<<blocks, threads/2, shared>>>(d_in, d_sums, threads);
    checkKernelErrors();

    exclusive_sum_scan_kernel<<<1, threads/2, shared>>>(d_sums, NULL, threads);
    checkKernelErrors();

    sum_scan_addition_kernel<<<blocks, threads>>>(d_in, d_sums, length);
    checkKernelErrors();

    checkCudaErrors(cudaFree(d_sums));
}

// #include <thrust/scan.h>
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

// void thrust_exclusive_scan(
//     unsigned int* const d_in, int length)
// {
//     thrust::device_vector<unsigned int> d_data(d_in, d_in + length);
//     thrust::exclusive_scan(d_data.begin(), d_data.end(), thrust::device_ptr<unsigned int>(d_in));
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

    reduce_min_max(d_logLuminance, numRows*numCols, min_logLum, max_logLum);
    // min_logLum = thrust_min(d_logLuminance, numRows*numCols);
    // max_logLum = thrust_max(d_logLuminance, numRows*numCols);

    generate_histogram(d_logLuminance, numRows*numCols, d_cdf, numBins, min_logLum, max_logLum);

    exclusive_sum_scan(d_cdf, numBins);
    // thrust_exclusive_scan(d_cdf, numBins);
}
