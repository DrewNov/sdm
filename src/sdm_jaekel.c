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
__global__ void sdm_write_cuda(sdm_jaekel_t *sdm, unsigned long addr, unsigned long v_in) {
	unsigned short i, j, k = 0;
	unsigned long *mask = sdm->mask;

	for (i = 0; i < sdm->n; ++i) {
		if (!(addr & *mask ^ *(mask + 1))) {
			short *p_cntr = sdm->cntr + (i + 1) * sdm->d - 1; //pointer to last counter in location

			for (j = 0; j < sdm->d; ++j) {
				1L << j & v_in ? (*p_cntr--)++ : (*p_cntr--)--;
			}

			k++;
		}
		mask += 2;
	}

	sdm->nact = k;
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
	unsigned long size_short = sizeof(short);

	sdm->cntr = (short *) malloc(n * d * size_short);
	sdm->mask = (unsigned long *) malloc(n * 2 * sizeof(long));
	sdm->n = n;
	sdm->d = d;
	sdm->k = k;

	sdm->actl = (unsigned short *) malloc(n * size_short);
	sdm->sumc = (short *) malloc(d * size_short);
	sdm->nact = 0;

	memset(sdm->cntr, 0, n * d * size_short);
	memset(sdm->actl, 0, n * size_short);
	memset(sdm->sumc, 0, d * size_short);

	/* Initialize mask randomly */
	srandom((unsigned int) time(NULL));

	int i = 0;

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

		sdm->mask[2 * i] = selection_mask;
		sdm->mask[2 * i + 1] = value_mask;
	}
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
	unsigned long *mask = sdm->mask;

	for (i = 0; i < sdm->n; ++i) {
		printf("#%4d:", i);

		//Run through Counters:
		for (j = sdm->d * i; j < sdm->d * (i + 1); ++j) {
			printf("%2d", sdm->cntr[j]);
		}

		//Run through Indexes:
		printf("\nmask1: %s\nmask2: %s\n\n", BIN2STR(*mask, sdm->d), BIN2STR(*(mask + 1), sdm->d));
		mask += 2;
	}

	printf("\n");
}


int sdm_write(sdm_jaekel_t *sdm, unsigned long addr, unsigned long v_in) {
	int i;

	sdm_activate(sdm, addr);

	for (i = 0; i < sdm->nact; i++) {
		sdm_cntrvary(sdm, sdm->cntr + sdm->actl[i] * sdm->d, v_in);
	}

	return sdm->nact;
}

int sdm_read(sdm_jaekel_t *sdm, unsigned long addr, unsigned long *v_out) {
	sdm_activate(sdm, addr);
	sdm_cntrsum(sdm);
	sdm_sum2bin(sdm, v_out);

	return sdm->nact;
}
