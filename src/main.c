#include <stdio.h>
#include "sdm_jaekel.h"

/*__global__ void main_cuda(float *d_v1, float *d_v2, float *d_result) {
	int i = threadIdx.x;
	d_result[i] = d_v1[i] + d_v2[i];
}*/

int main(int argc, char *argv[]) {
	/*int n = 1024;
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
	free(h_result);*/

	//SDM part
	sdm_jaekel_t sdm;

	if (argc == 1) {
		printf("Example of use: sdm 100 10 3, where nloc=100; ndim=10; nidx=3\n");
		//return 1;
	}

	int nloc = 100; //atoi(argv[1]); //100
	int ndim = 10; //atoi(argv[2]); //10
	int nidx = 3; //atoi(argv[3]); //3

	int *addr1 = STR2BIN("0101010101", ndim);
	int *addr2 = STR2BIN("0101000101", ndim);
	int *v_in  = STR2BIN("0100101010", ndim);
	int *v_out = STR2BIN("0000000000", ndim);

	sdm_init(&sdm, nloc, ndim, nidx);

	printf("SDM size: %lu\n", sizeof(sdm));
	sdm_print(&sdm);

	printf("addr1: %s\n", BIN2STR(addr1, ndim));
	printf(" v_in: %s\n", BIN2STR(v_in, ndim));
	printf("Write: %d locations activated\n", sdm_write(&sdm, addr1, v_in));
	printf("\n");

	printf("addr1: %s\n", BIN2STR(addr1, ndim));
	printf("Read1: %d locations activated\n", sdm_read(&sdm, addr1, v_out));
	printf("v_out: %s\n", BIN2STR(v_out, ndim));
	printf("\n");

	printf("addr2: %s\n", BIN2STR(addr2, ndim));
	printf("Read2: %d locations activated\n", sdm_read(&sdm, addr2, v_out));
	printf("v_out: %s\n", BIN2STR(v_out, ndim));

	sdm_print(&sdm);

	sdm_free(&sdm);

	return 0;
}
