#include <stdio.h>
#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <vector>

using std::cout;
using std::generate;
using std::vector;

#define SIZE 64
#define THREADS 32 

#define CUDA_CALL(x) do { if((x)!=cudaSuccess) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    return EXIT_FAILURE;}} while(0)
#define CURAND_CALL(x) do { if((x)!=CURAND_STATUS_SUCCESS) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    return EXIT_FAILURE;}} while(0)

__global__ void MatrixMul(const int *Md, const int *Nd, int *Pd, int Width) {

	// Calculate the row index of the Pd element and M
	int row = (blockIdx.y * blockDim.y) + threadIdx.y;

	// Calculate the column idenx of Pd and N
	int col = (blockIdx.x * blockDim.x) + threadIdx.x;

	if((row < Width) && (col < Width)) {
	// each thread computes one element of the block sub-matrix
		
		for (int k = 0; k < Width; ++k) {
		 	 // dot product or corresponding row and column. 
		 	 Pd[(row * Width) + col] += Md[(row * Width) + k] * Nd[(k * Width) + col];
		}
	}
}

// Check result on the CPU (single threaded)

void verify_result(vector<int> &a, vector<int> &b, vector<int> &c, int N) {
  	// Loop over every row...
  	float time;
	cudaEvent_t start, stop;

	// start tracking the time
    cudaEventCreate(&start) ;
	cudaEventCreate(&stop) ;
	cudaEventRecord(start, 0) ;

  	for (int i = 0; i < N; i++) {
    	// Loop every column...
    	for (int j = 0; j < N; j++) {
      		// For every element in the row-column pair
      		
      		int tmp = 0;
      		for (int k = 0; k < N; k++) {
        		// Accumulate the partial results
        	tmp += a[i * N + k] * b[k * N + j];
      		}

      		// Check against the CPU result
      		assert(tmp == c[i * N + j]);
    	}
  	}

    cudaEventRecord(stop, 0) ;
	cudaEventSynchronize(stop) ;
	cudaEventElapsedTime(&time, start, stop) ;

	printf("Compute time on CPU:  %3.6f ms \n", time);
}

int main(void){
	
	// initialize event creation for time tracking
	float time;
	cudaEvent_t start, stop;


	// Matrix size of 32 x 32; 
	int N = SIZE; 

	printf("Matrix Size: %d x %d\n", N, N); 

	// size (in bytes) of matrix
	size_t size = N * N * sizeof(int); 

	vector<int> host_a(N * N);
	vector<int> host_b(N * N); 
	vector<int> host_c(N * N); 

	// generate random indices between 0 and 1. 
	generate(host_a.begin(), host_a.end(), []() {return rand() % 2; });  
	generate(host_b.begin(), host_b.end(), []() {return rand() % 2; });

	// device memory allocation
	int *dev_a, *dev_b, *dev_c;

    cudaMalloc(&dev_a, size);
    cudaMalloc(&dev_b, size);
    cudaMalloc(&dev_c, size);

    //copy data from host to device
    cudaMemcpy(dev_a, host_a.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, host_b.data(), size, cudaMemcpyHostToDevice);

    int BLOCKS = N/THREADS; 

    dim3 threads(THREADS, THREADS);
    dim3 blocks(BLOCKS, BLOCKS); 

    // start tracking the time
    cudaEventCreate(&start) ;
	cudaEventCreate(&stop) ;
	cudaEventRecord(start, 0) ;

    // launch kernal

    printf("No. of blocks: %d x %d\n", BLOCKS, BLOCKS); 
    printf("No. of therads: %d x %d\n", THREADS, THREADS); 
    MatrixMul<<<blocks, threads>>> (dev_a, dev_b, dev_c, N); 

    // stop tracking the time
    cudaEventRecord(stop, 0) ;
	cudaEventSynchronize(stop) ;
	cudaEventElapsedTime(&time, start, stop) ;


    cudaMemcpy(host_c.data(), dev_c, size, cudaMemcpyDeviceToHost);

	printf("Compute time on GPU:  %3.6f ms \n", time);

	// verify result on CPU
    verify_result(host_a, host_b, host_c, N);

    //free device memory
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);

	return 0; 
}

