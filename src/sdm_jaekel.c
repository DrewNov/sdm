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


//Cuda functions:
__global__ void sdm_write_cuda(sdm_jaekel_t sdm, unsigned long addr, unsigned long v_in, int *nact) {
	int i = threadIdx.x, j;

	if (!(addr & sdm.mask[2 * i] ^ sdm.mask[2 * i + 1])) {
		short *p_cntr = sdm.cntr + (i + 1) * sdm.d - 1; //pointer to last counter in location

		for (j = 0; j < sdm.d; ++j) {
			1L << j & v_in ? (*p_cntr--)++ : (*p_cntr--)--;
			//(*p_cntr--) = (short) i;
		}

		(*nact)++;
	}
}


//Helper functions:
void sdm_activate(sdm_jaekel_t *sdm, unsigned long addr) {
	unsigned short i, j = 0;
	unsigned long *mask = sdm->mask;

	for (i = 0; i < sdm->n; ++i) {
		if (!(addr & *mask ^ *(mask + 1))) {
			sdm->actl[j++] = i;
		}
		mask += 2;
	}

	sdm->nact = j;
}

void sdm_cntrvary(sdm_jaekel_t *sdm, short *p_cntr, unsigned long v_in) {
	int i;

	p_cntr += sdm->d - 1; //pointer to last counter in location

	for (i = 0; i < sdm->d; ++i) {
		1L << i & v_in ? (*p_cntr--)++ : (*p_cntr--)--;
	}
}

void sdm_cntradd(sdm_jaekel_t *sdm, short *p_cntr) {
	int i;
	short *p_sum = sdm->sumc;

	for (i = 0; i < sdm->d; i++) {
		(*p_sum++) += (*p_cntr++);
	}
}

void sdm_cntrsum(sdm_jaekel_t *sdm) {
	int i;

	memset(sdm->sumc, 0, sdm->d * sizeof(short));

	for (i = 0; i < sdm->nact; i++) {
		sdm_cntradd(sdm, sdm->cntr + sdm->actl[i] * sdm->d);
	}
}

void sdm_sum2bin(sdm_jaekel_t *sdm, unsigned long *vect) {
	int i;

	*vect = 0;

	for (i = 0; i < sdm->d; ++i) {
		*vect <<= 1;
		*vect |= sdm->sumc[i] > 0;
	}
}


//Main functions:
void sdm_init(sdm_jaekel_t *sdm, unsigned short n, unsigned short d, unsigned short k) {
	int i;
	unsigned long *h_mask;
	size_t size_short = sizeof(short);

	cudaMalloc((void **) &(sdm->cntr), n * d * size_short);
	cudaMalloc((void **) &(sdm->mask), n * 2 * sizeof(long));
	sdm->n = n;
	sdm->d = d;
	sdm->k = k;

	cudaMalloc((void **) &(sdm->actl), n * size_short);
	cudaMalloc((void **) &(sdm->sumc), d * size_short);
	sdm->nact = 0;

	cudaMemset(sdm->cntr, 0, n * d * size_short);
	cudaMemset(sdm->actl, 0, n * size_short);
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
	if (sdm->cntr != 0) free(sdm->cntr);
	if (sdm->mask != 0) free(sdm->mask);
	if (sdm->actl != 0) free(sdm->actl);
	if (sdm->sumc != 0) free(sdm->sumc);
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

	sdm_write_cuda <<<1, sdm->n>>> (*sdm, addr, v_in, d_nact);

	cudaMemcpy(&h_nact, d_nact, sizeof(int), cudaMemcpyDeviceToHost);

	return h_nact;
}

int sdm_read(sdm_jaekel_t *sdm, unsigned long addr, unsigned long *v_out) {
	sdm_activate(sdm, addr);
	sdm_cntrsum(sdm);
	sdm_sum2bin(sdm, v_out);

	return sdm->nact;
}
