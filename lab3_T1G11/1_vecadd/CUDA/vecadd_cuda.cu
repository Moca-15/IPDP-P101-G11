#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "cuda.h"

#define BLOCKSIZE 128
#define CUDA_CHECK(func)                                                                              \
	do {                                                                                              \
		cudaError_t err = (func);                                                                     \
		if(err != cudaSuccess) {                                                                      \
			fprintf(stderr, "CUDA error at %s:%d %s\n", __FILE__,__LINE__,cudaGetErrorString(err));   \
			return 1;                                                                                 \
   		}                                                                                             \
	}while(0);



__global__ void vecadd_cuda(double *A, double *B, double *C, const int N){
		int idx = threadIdx.x + blockIdx.x*blockDim.x;
		if(idx >= N) return;
		C[idx] = A[idx] + B[idx];

}

int main(int argc, char* argv[]){

	int N;

	if(argc < 2){
		printf("Usage: %s <vector size N>\n", argv[0]);
		return 1;
	}

	N = atoi(argv[1]);
	int size = N * sizeof(double);

	double *A = (double*)malloc(size);
	double *B = (double*)malloc(size);
	double *C = (double*)malloc(size);

	for(int i=0; i<N; i++){
		A[i] = (double)i;
		B[i] = (double)(2*(N-i));
	}

	double *d_A, *d_B, *d_C;

	CUDA_CHECK(cudaMalloc((void**)&d_A, size));
	CUDA_CHECK(cudaMalloc((void**)&d_B, size));
	CUDA_CHECK(cudaMalloc((void**)&d_C, size));

	cudaEvent_t start, stop;
	CUDA_CHECK(cudaEventCreate(&start));
	CUDA_CHECK(cudaEventCreate(&stop));

	//LOAD DATA ON DEVICE

	float hostToDeviceTime;

	CUDA_CHECK(cudaEventRecord(start));
	CUDA_CHECK(cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaEventRecord(stop));
	CUDA_CHECK(cudaEventSynchronize(stop));
	CUDA_CHECK(cudaEventElapsedTime(&hostToDeviceTime, start, stop));

	//VECTOR ADDITION

	int n_blocks = (N + BLOCKSIZE-1)/BLOCKSIZE;

	float vecAddTime;

	CUDA_CHECK(cudaEventRecord(start));
	vecadd_cuda<<<n_blocks,BLOCKSIZE>>>(d_A, d_B, d_C,N);
	CUDA_CHECK(cudaEventRecord(stop));
	CUDA_CHECK(cudaEventSynchronize(stop));
	CUDA_CHECK(cudaEventElapsedTime(&vecAddTime, start, stop));

	//COPY BACK TO HOST

	float deviceToHostTime;

	CUDA_CHECK(cudaEventRecord(start));
	CUDA_CHECK(cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost));
	CUDA_CHECK(cudaEventRecord(stop));
	CUDA_CHECK(cudaEventSynchronize(stop));
	CUDA_CHECK(cudaEventElapsedTime(&deviceToHostTime, start, stop));

	for(int i=0; i<N; i++){
		double diff = C[i] - (double)(2*N-i);
		if(fabs(diff) > 1E-6){
			printf("Value exceeding tolerance at i=%d : %lf; diff=%lf\n",i,C[i],diff);
			break;
		}
	}

	//CLEANUP
	free(A); free(B); free(C);
	CUDA_CHECK(cudaFree(d_A)); CUDA_CHECK(cudaFree(d_B)); CUDA_CHECK(cudaFree(d_C));
	CUDA_CHECK(cudaEventDestroy(start)); CUDA_CHECK(cudaEventDestroy(stop));

	float totalTime = hostToDeviceTime + vecAddTime + deviceToHostTime;

	printf("Vector size: %d\n",N);
	printf("Copy A and B Host to Device elapsed time: %f seconds\n", hostToDeviceTime/1000.0f);
	printf("Kernel elapsed time: %f seconds\n", vecAddTime/1000.0f);
	printf("Copy C Device to Host elapsed time: %f seconds\n", deviceToHostTime/1000.0f);
	printf("Total elapsed time: %f seconds\n", totalTime/1000.0f);
}
