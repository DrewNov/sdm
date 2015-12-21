/*
 * This is implementation of Jaekel's version of SDM
 * by DrewNov, December 2015.
 * C Binary Vector Symbols (CBVS) was taken as a basis. (http://pendicular.net/cbvs.php)
 */

#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdio.h>
#include "sdm_jaekel.h"


//CUDA functions:
__global__ void d_sdm_activate(int *idxs, int *addr, int *nidx, int *actl, int *nact) {
	int i = threadIdx.x, j = 0, flag = 1;

	for (j = (*nidx) * i; j < (*nidx) * (i + 1); ++j) {
		int idx = idxs[j];

		if (addr[abs(idx)] != (idx > 0)) { //todo: clarify what to do if 0
			flag = 0;
			break;
		}
	}

	if (flag) {
		actl[(*nact)++] = i;
	}
}

void sdm_activate_cuda(sdm_jaekel_t *sdm, int *addr) {
	dim3 grid = dim3(1, 1, 1);
	dim3 block = dim3(sdm->nloc, 1, 1);

	cudaEvent_t start, stop;
	float time;

	int *d_idxs;
	int *d_addr;
	int *d_nidx;
	int *d_actl;
	int *d_nact;

	sdm->nact = 0;

	cudaMalloc((void **) &d_idxs, sdm->nidx * sizeof(int));
	cudaMalloc((void **) &d_addr, sdm->ndim * sizeof(int));
	cudaMalloc((void **) &d_nidx, sizeof(int));
	cudaMalloc((void **) &d_actl, sdm->nloc * sizeof(int));
	cudaMalloc((void **) &d_nact, sizeof(int));

	cudaMemcpy(d_idxs, sdm->idxs, sdm->nidx * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_addr, addr, sdm->ndim * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_nidx, &(sdm->nidx), sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_nact, &(sdm->nact), sizeof(int), cudaMemcpyHostToDevice);

	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start, 0);
	d_sdm_activate << < grid, block >> > (d_idxs, d_addr, d_nidx, d_actl, d_nact);
	cudaEventRecord(stop, 0);

	cudaEventSynchronize(stop);

	cudaEventElapsedTime(&time, start, stop);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	cudaMemcpy(&(sdm->nact), d_nact, sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(&(sdm->actl), d_actl, sdm->nact * sizeof(int), cudaMemcpyDeviceToHost);

	cudaFree(d_idxs);
	cudaFree(d_addr);
	cudaFree(d_nidx);
	cudaFree(d_actl);
	cudaFree(d_nact);

	printf("Time: %f\n", time);
}


//Helper functions:
void sdm_activate(sdm_jaekel_t *sdm, int *addr) {
	int i, j, k = 0;

	for (i = 0; i < sdm->nloc; ++i) {
		int flag = 1;

		for (j = sdm->nidx * i; j < sdm->nidx * (i + 1); ++j) {
			int idx = sdm->idxs[j];

			if (addr[abs(idx)] != (idx > 0)) { //todo: clarify what to do if 0
				flag = 0;
				break;
			}
		}

		if (flag) {
			sdm->actl[k] = i;
			k++;
		}
	}

	sdm->nact = k;
}

void sdm_cntrvary(sdm_jaekel_t *sdm, int *p_cntr, int *v_in) {
	int i;

	for (i = 0; i < sdm->ndim; i++) {
		v_in[i] ? p_cntr[i]++ : p_cntr[i]--;
	}
}

void sdm_cntradd(sdm_jaekel_t *sdm, int *p_cntr) {
	int i;
	long *p_sum = sdm->sumc;

	for (i = 0; i < sdm->ndim; i++) {
		(*p_sum++) += (*p_cntr++);
	}
}

void sdm_cntrsum(sdm_jaekel_t *sdm) {
	int i;

	memset(sdm->sumc, 0, sdm->ndim * sizeof(long));

	for (i = 0; i < sdm->nact; i++) {
		sdm_cntradd(sdm, &(sdm->cntr[sdm->actl[i] * sdm->ndim]));
	}
}

int sdm_sum2bit(long cntrsum) {
	//return (cntrsum > 0) ? 1 : ((cntrsum < 0) ? 0 : (int) (random() % 2));
	return cntrsum > 0; //todo: clarify what to do if 0
}

void sdm_sum2bin(sdm_jaekel_t *sdm, int *vect) {
	int i;

	for (i = 0; i < sdm->ndim; ++i) {
		vect[i] = sdm_sum2bit(sdm->sumc[i]);
	}
}


//Main functions:
void sdm_init(sdm_jaekel_t *sdm, int nloc, int ndim, int nidx) {
	sdm->cntr = (int *) malloc(nloc * ndim * sizeof(int));
	sdm->idxs = (int *) malloc(nloc * nidx * sizeof(int));
	sdm->nloc = nloc;
	sdm->ndim = ndim;
	sdm->nidx = nidx;

	sdm->actl = (int *) malloc(nloc * sizeof(int));
	sdm->sumc = (long *) malloc(ndim * sizeof(long));
	sdm->nact = 0;

	memset(sdm->cntr, 0, nloc * ndim * sizeof(int));
	memset(sdm->actl, 0, nloc * sizeof(int));
	memset(sdm->sumc, 0, ndim * sizeof(long));

	/* Initialize indexes randomly */
	srandom((unsigned int) time(NULL));

	int i = nloc * nidx;
	int *p_idxs = sdm->idxs;

	while (i--) {
		int rnd = (int) (random() % ndim); //todo: clarify what to do if several identical indexes for location
		*p_idxs++ = random() % 2 ? rnd : -rnd;
	}
}

void sdm_free(sdm_jaekel_t *sdm) {
	if (sdm->cntr != 0) free(sdm->cntr);
	if (sdm->idxs != 0) free(sdm->idxs);
	if (sdm->actl != 0) free(sdm->actl);
	if (sdm->sumc != 0) free(sdm->sumc);
	memset(sdm, 0, sizeof(sdm_jaekel_t));
}

void sdm_print(sdm_jaekel_t *sdm) {
	int i, j;

	for (i = 0; i < sdm->nloc; ++i) {
		printf("#%3d:", i);

		//Run through Counters:
		for (j = sdm->ndim * i; j < sdm->ndim * (i + 1); ++j) {
			int cntr = sdm->cntr[j];
			printf("%4d", cntr);
		}
		printf("\t\t");

		//Run through Indexes:
		for (j = sdm->nidx * i; j < sdm->nidx * (i + 1); ++j) {
			int idx = sdm->idxs[j];
			printf("%5d", idx);
		}
		printf("\n");
	}

	printf("\n");
}


int sdm_write(sdm_jaekel_t *sdm, int *addr, int *v_in) {
	int i;

	sdm_activate(sdm, addr);
	//sdm_activate_cuda(sdm, addr);

	for (i = 0; i < sdm->nact; i++) {
		sdm_cntrvary(sdm, &(sdm->cntr[sdm->actl[i] * sdm->ndim]), v_in);
	}

	return sdm->nact;
}

int sdm_read(sdm_jaekel_t *sdm, int *addr, int *v_out) {
	sdm_activate(sdm, addr);
	//sdm_activate_cuda(sdm, addr);
	sdm_cntrsum(sdm);
	sdm_sum2bin(sdm, v_out);

	return sdm->nact;
}
