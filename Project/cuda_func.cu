
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include "func.h"


__global__ void xor_kernal(char* data ,char* key,char* xorstring,int size_data,int size_key)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if(i<size_data)
		xorstring[i] =  (char) data[i] ^ key[i%size_key];
	if(i == size_data-1)
		xorstring[size_data-1] = '\0';
}

void checkSuccess(cudaError_t err ,char* str){
	if (err != cudaSuccess) {
	        fprintf(stderr, "%s  - %s\n",str, cudaGetErrorString(err));
	        exit(EXIT_FAILURE);
	 }

}

void myXor(char* data ,char* key,char* xorstring,int size_data,int size_key)
{
	char* d_data;
	char* d_key;
	char* d_xorstring;

	int num_blocks = (size_data + SIZE)/SIZE;

	// allocation memory to cuda
	checkSuccess(cudaMalloc((void**)&d_data, size_data * sizeof(char)),(char*) "Failed to allocate device memory");
	checkSuccess(cudaMalloc((void**)&d_key, size_key * sizeof(char)),(char*) "Failed to allocate device memory");
	checkSuccess(cudaMalloc((void**)&d_xorstring, size_data * sizeof(char)),(char*) "Failed to allocate device memory");

	// copy memory to cuda
	checkSuccess(cudaMemcpy(d_data, data, size_data*sizeof(char), cudaMemcpyHostToDevice),(char*) "Failed to copy data from host to device");
	checkSuccess(cudaMemcpy(d_key, key, size_key*sizeof(char), cudaMemcpyHostToDevice),(char*) "Failed to copy key from host to device");
	checkSuccess(cudaMemcpy(d_xorstring, xorstring, size_data*sizeof(char), cudaMemcpyHostToDevice),(char*) "Failed to copy xorString from host to device");

	xor_kernal<<<num_blocks,SIZE>>>(d_data, d_key, d_xorstring,size_data,size_key);

	// copy memory to host
	checkSuccess(cudaMemcpy((void*) xorstring, d_xorstring, size_data*sizeof(char), cudaMemcpyDeviceToHost),(char*) "Failed to copy xorString from device to host");

	// free memory
	cudaFree(d_data);
	cudaFree(d_key);
	cudaFree(d_xorstring);

}


