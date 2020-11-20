
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "histogram.h"

#include <stdio.h>

__global__ void histo_kernal(int *buffer,int *histo, unsigned int size)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	
	histo[i] = 0;
	__syncthreads();

	if(i < size)
		atomicAdd(&histo[buffer[i]],1);
	
}



int* calculate_histogram(int *arr, unsigned int size)
{
	int *d_arr,*d_histogram;
	int* histogram = new int[ARR_SIZE];
	int num_blocks = (size + ARR_SIZE)/ARR_SIZE;

	reset_array(histogram, ARR_SIZE);

	cudaMalloc((void**)&d_arr, size * sizeof(int));
	cudaMalloc((void**)&d_histogram, ARR_SIZE * sizeof(int));

	cudaMemcpy(d_arr, arr, size * sizeof(int), cudaMemcpyHostToDevice);
	
	histo_kernal<<<num_blocks,ARR_SIZE>>>(d_arr, d_histogram, size);

	cudaMemcpy(histogram, d_histogram, ARR_SIZE * sizeof(int), cudaMemcpyDeviceToHost);

	cudaFree(d_arr);
	cudaFree(d_histogram);

	return histogram;
}
