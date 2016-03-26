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


//Helper functions:
int comparator(const void *a, const void *b) {
	return *(unsigned short *) b - *(unsigned short *) a;
}


//CUDA functions:
__global__ void sdm_write_cuda(sdm_jaekel_t sdm, unsigned *addr, unsigned *v_in, int *nact) {
	int i = blockDim.x * blockIdx.x + threadIdx.x, j, k, part_bits = sizeof(unsigned) * 8;
	short *p_cntr;
	unsigned short *p_idx = sdm.idxs + sdm.k * i;
	unsigned location_num = 0;

	//calculating location number from mask todo: make a separate function
	for (k = 0; k < sdm.k; ++k) {
		unsigned short idx = *p_idx++;
		unsigned part_num = idx / part_bits;
		unsigned num_in_part = idx % part_bits;
		unsigned digit_mask = 1U << num_in_part;

		location_num <<= 1;
		location_num |= (addr[part_num] & digit_mask) > 0;
	}

	//modifying counters of activated location
	p_cntr = sdm.cntr + sdm.d * ((1 << sdm.k) * i + location_num) + part_bits - 1; //last counter in 1 part of location

	for (j = 0; j < sdm.d; ++j) {
		1U << (j % part_bits) & *v_in ? (*p_cntr--)++ : (*p_cntr--)--;

		if ((j + 1) % part_bits == 0) {
			v_in++;
			p_cntr += part_bits * 2 - 1;
		}
	}

	(*nact)++;
}

__global__ void sdm_read_cuda(sdm_jaekel_t sdm, unsigned *addr, int *nact) {
	int i = blockDim.x * blockIdx.x + threadIdx.x, j, k, part_bits = sizeof(unsigned) * 8;
	short *p_cntr;
	unsigned short *p_idx = sdm.idxs + sdm.k * i;
	unsigned location_num = 0;

	//calculating location number from mask todo: make a separate function
	for (k = 0; k < sdm.k; ++k) {
		unsigned short idx = *p_idx++;
		unsigned part_num = idx / part_bits;
		unsigned num_in_part = idx % part_bits;
		unsigned digit_mask = 1U << num_in_part;

		location_num <<= 1;
		location_num |= (addr[part_num] & digit_mask) > 0;
	}

	//reading counters of activated location
	p_cntr = sdm.cntr + sdm.d * ((1 << sdm.k) * i + location_num);

	for (j = 0; j < sdm.d; j++) {
		sdm.sumc[j] += p_cntr[j];
	}

	(*nact)++;
}


//Main functions:
void sdm_init(sdm_jaekel_t *sdm, unsigned n, unsigned d, unsigned k) {
	int i, pow2_k = 1 << k;
	unsigned short *h_idxs;
	size_t size_short = sizeof(short);

	cudaMalloc((void **) &(sdm->cntr), n * d * pow2_k * size_short);
	cudaMalloc((void **) &(sdm->idxs), n * k * size_short);
	cudaMalloc((void **) &(sdm->sumc), d * size_short);

	sdm->n = n;
	sdm->d = d;
	sdm->k = k;

	cudaMemset(sdm->cntr, 0, n * d * pow2_k * size_short);
	cudaMemset(sdm->sumc, 0, d * size_short);

	/* Initialize idxs randomly */
	srandom((unsigned) time(NULL));

	h_idxs = (unsigned short *) malloc(n * k * size_short);

	for (i = 0; i < n * k; ++i) {
		h_idxs[i] = (unsigned short) (random() % d);

		if ((i + 1) % k == 0) {
			qsort(h_idxs + i - k + 1, k, size_short, comparator);
		}
	}

	cudaMemcpy(sdm->idxs, h_idxs, n * k * size_short, cudaMemcpyHostToDevice);
}

void sdm_free(sdm_jaekel_t *sdm) {
	if (sdm->cntr != 0) cudaFree(sdm->cntr);
	if (sdm->idxs != 0) cudaFree(sdm->idxs);
	if (sdm->sumc != 0) cudaFree(sdm->sumc);
	memset(sdm, 0, sizeof(sdm_jaekel_t));
}

void sdm_print(sdm_jaekel_t *sdm) {
	int i, j;

	short *cntr = (short *) malloc(sdm->n * sdm->d * sizeof(short));
	unsigned long *mask = (unsigned long *) malloc(sdm->n * 2 * sizeof(long));

	cudaMemcpy(cntr, sdm->cntr, sdm->n * sdm->d * sizeof(short), cudaMemcpyDeviceToHost);
	cudaMemcpy(mask, sdm->idxs, sdm->n * 2 * sizeof(long), cudaMemcpyDeviceToHost);

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


int sdm_write(sdm_jaekel_t *sdm, unsigned *addr, unsigned *v_in) {
	int h_nact = 0, vect_size = sdm->d / 8;
	int *d_nact;
	unsigned *d_addr, *d_v_in;

	cudaMalloc((void **) &d_nact, sizeof(int));
	cudaMemset(d_nact, 0, sizeof(int));

	cudaMalloc((void **) &d_addr, vect_size);
	cudaMemcpy(&d_addr, addr, vect_size, cudaMemcpyHostToDevice);

//	cudaMalloc((void **) &d_v_in, vect_size);
//	cudaMemcpy(&d_v_in, v_in, vect_size, cudaMemcpyHostToDevice);
	d_v_in = d_addr;

	sdm_write_cuda <<< sdm->n / 1024, 1024 >>> (*sdm, d_addr, d_v_in, d_nact);

	cudaMemcpy(&h_nact, d_nact, sizeof(int), cudaMemcpyDeviceToHost);

	return h_nact;
}

int sdm_read(sdm_jaekel_t *sdm, unsigned *addr, unsigned *v_out) {
	int i, vect_size = sdm->d / 8;
	unsigned *p_v_out = v_out;

	int h_nact = 0;
	short *h_sumc = (short *) malloc(sdm->d * sizeof(short));

	int *d_nact;
	unsigned *d_addr;

	cudaMalloc((void **) &d_nact, sizeof(int));
	cudaMemset(d_nact, 0, sizeof(int));

	cudaMemset(sdm->sumc, 0, sdm->d * sizeof(short));

	cudaMalloc((void **) &d_addr, vect_size);
	cudaMemcpy(&d_addr, addr, vect_size, cudaMemcpyHostToDevice);

	sdm_read_cuda <<< sdm->n / 1024, 1024 >>> (*sdm, d_addr, d_nact);

	cudaMemcpy(&h_nact, d_nact, sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_sumc, sdm->sumc, sdm->d * sizeof(short), cudaMemcpyDeviceToHost);

	memset(v_out, 0, vect_size);

	for (i = 0; i < sdm->d; ++i) {
		*p_v_out <<= 1;
		*p_v_out |= h_sumc[i] > 0;

		if ((i + 1) % (sizeof(unsigned) * 8) == 0) {
			p_v_out++;
		}
	}

	return h_nact;
}
