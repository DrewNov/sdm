#include <stdio.h>
#include <stdlib.h>
#include "sdm_jaekel.h"

__global__ void main_cuda(float *d_v1, float *d_v2, float *d_result) {
	int i = threadIdx.x;
	d_result[i] = d_v1[i] + d_v2[i];
}

int main(int argc, char *argv[]) {
	int n = 1024;
	int n_bytes = sizeof(float) * n;

	dim3 grid = dim3(1, 1, 1);
	dim3 block = dim3(n, 1, 1);

	float *h_v1 = (float *) malloc(n_bytes);
	float *h_v2 = (float *) malloc(n_bytes);
	float *h_result = (float *) malloc(n_bytes);

	float *d_v1;
	float *d_v2;
	float *d_result;

	for (int i = 0; i < n; ++i) {
		h_v1[i] = i;
		h_v2[i] = i;
	}

	cudaMalloc((void **) &d_v1, n_bytes);
	cudaMalloc((void **) &d_v2, n_bytes);
	cudaMalloc((void **) &d_result, n_bytes);

	cudaMemcpy(d_v1, h_v1, n_bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_v2, h_v2, n_bytes, cudaMemcpyHostToDevice);

	main_cuda << < grid, block >> > (d_v1, d_v2, d_result);

	cudaEvent_t event;

	cudaEventCreate(&event);
	cudaEventRecord(event, 0);
	cudaEventSynchronize(event);

	cudaMemcpy(h_result, d_result, n_bytes, cudaMemcpyDeviceToHost);

	for (int i = 0; i < n; ++i) {
		//printf("%4d: %4.1f\n", i, h_result[i]);
	}

	cudaEventDestroy(event);

	cudaFree(d_v1);
	cudaFree(d_v2);
	cudaFree(d_result);

	free(h_v1);
	free(h_v2);
	free(h_result);

	//sdm part
	sdm_jaekel_t sdm;

	sdm_init(&sdm, 100, 10, 3);

	printf("SDM size: %lu\n\n", sizeof(sdm));

	for (int i = 0; i < sdm.nidx * sdm.nloc; i += sdm.nidx) {
		printf("SDM indexes for location #%3d:", i / sdm.nidx + 1);
		for (int j = 0; j < sdm.nidx; ++j) {
			printf("%5d", sdm.idxs[i + j]);
		}
		printf("\n");
	}

	sdm_free(&sdm);

	return 0;
}