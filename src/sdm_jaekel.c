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
__global__ void sdm_write_cuda(sdm_jaekel_t sdm, unsigned long addr, unsigned long v_in, int *nact) {
	int i = threadIdx.x;
	int j;

	if (!(addr & sdm.mask[2 * i] ^ sdm.mask[2 * i + 1])) {
		short *p_cntr = sdm.cntr + (i + 1) * sdm.d - 1; //pointer to last counter in location

		for (j = 0; j < sdm.d; ++j) {
			1L << j & v_in ? (*p_cntr--)++ : (*p_cntr--)--;
		}

		(*nact)++;
	}
}

__global__ void sdm_read_cuda(sdm_jaekel_t sdm, unsigned long addr, int *nact) {
	int i = threadIdx.x;
	int j;

	if (!(addr & sdm.mask[2 * i] ^ sdm.mask[2 * i + 1])) {
		short *p_cntr = sdm.cntr + i * sdm.d;

		for (j = 0; j < sdm.d; j++) {
			sdm.sumc[j] += p_cntr[j];
		}

		(*nact)++;
	}
}


//Main functions:
void sdm_init(sdm_jaekel_t *sdm, unsigned short n, unsigned short d, unsigned short k) {
	int i;
	unsigned long *h_mask;
	size_t size_short = sizeof(short);

	cudaMalloc((void **) &(sdm->cntr), n * d * size_short);
	cudaMalloc((void **) &(sdm->mask), n * 2 * sizeof(long));
	cudaMalloc((void **) &(sdm->sumc), d * size_short);

	sdm->n = n;
	sdm->d = d;
	sdm->k = k;

	cudaMemset(sdm->cntr, 0, n * d * size_short);
	cudaMemset(sdm->sumc, 0, d * size_short);

	/* Initialize mask randomly */
	srandom((unsigned int) time(NULL));

	h_mask = (unsigned long *) malloc(n * 2 * sizeof(long));

	for (i = 0; i < n; ++i) {
		int j = k;
		unsigned long selection_mask = 0;
		unsigned long value_mask = 0;

		while (j) {
			long rnd_digit = random() % d;

			if (!(selection_mask >> rnd_digit & 1)) {
				selection_mask |= 1L << rnd_digit;
				value_mask |= random() % 2 << rnd_digit;
				j--;
			}
		}

		h_mask[2 * i] = selection_mask;
		h_mask[2 * i + 1] = value_mask;
	}

	cudaMemcpy(sdm->mask, h_mask, n * 2 * sizeof(long), cudaMemcpyHostToDevice);
}

void sdm_free(sdm_jaekel_t *sdm) {
	if (sdm->cntr != 0) cudaFree(sdm->cntr);
	if (sdm->mask != 0) cudaFree(sdm->mask);
	if (sdm->sumc != 0) cudaFree(sdm->sumc);
	memset(sdm, 0, sizeof(sdm_jaekel_t));
}

void sdm_print(sdm_jaekel_t *sdm) {
	int i, j;

	short *cntr = (short *) malloc(sdm->n * sdm->d * sizeof(short));
	unsigned long *mask = (unsigned long *) malloc(sdm->n * 2 * sizeof(long));

	cudaMemcpy(cntr, sdm->cntr, sdm->n * sdm->d * sizeof(short), cudaMemcpyDeviceToHost);
	cudaMemcpy(mask, sdm->mask, sdm->n * 2 * sizeof(long), cudaMemcpyDeviceToHost);

	for (i = 0; i < sdm->n; ++i) {
		if (*(cntr + i * sdm->d)) {
			printf("#%4d:", i);

			//Run through Counters:
			for (j = sdm->d * i; j < sdm->d * (i + 1); ++j) {
				printf("%2d", cntr[j]);
			}

			//Run through Indexes:
			printf("\nmask1: %s\nmask2: %s\n\n", BIN2STR(*mask, sdm->d), BIN2STR(*(mask + 1), sdm->d));
			mask += 2;
		}
	}

	printf("\n");
}


int sdm_write(sdm_jaekel_t *sdm, unsigned long addr, unsigned long v_in) {
	int h_nact = 0;
	int *d_nact;

	cudaMalloc((void **) &d_nact, sizeof(int));
	cudaMemset(d_nact, 0, sizeof(int));

	sdm_write_cuda <<< 1, sdm->n >>> (*sdm, addr, v_in, d_nact);

	cudaMemcpy(&h_nact, d_nact, sizeof(int), cudaMemcpyDeviceToHost);

	return h_nact;
}

int sdm_read(sdm_jaekel_t *sdm, unsigned long addr, unsigned long *v_out) {
	int i;
	int h_nact = 0;
	int *d_nact;
	short *h_sumc = (short *) malloc(sdm->d * sizeof(short));

	cudaMalloc((void **) &d_nact, sizeof(int));
	cudaMemset(d_nact, 0, sizeof(int));

	cudaMemset(sdm->sumc, 0, sdm->d * sizeof(short));

	sdm_read_cuda <<< 1, sdm->n >>> (*sdm, addr, d_nact);

	cudaMemcpy(&h_nact, d_nact, sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_sumc, sdm->sumc, sdm->d * sizeof(short), cudaMemcpyDeviceToHost);

	*v_out = 0;

	for (i = 0; i < sdm->d; ++i) {
		*v_out <<= 1;
		*v_out |= h_sumc[i] > 0;
	}

	return h_nact;
}
