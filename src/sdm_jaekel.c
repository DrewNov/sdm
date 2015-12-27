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
void sdm_activate(sdm_jaekel_t *sdm, int *addr) {
	int i, j, k = 0;

	for (i = 0; i < sdm->n; ++i) {
		int flag = 1;

		for (j = sdm->k * i; j < sdm->k * (i + 1); ++j) {
			int idx = sdm->mask[j];

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

void sdm_cntrvary(sdm_jaekel_t *sdm, short *p_cntr, int *v_in) {
	int i;

	for (i = 0; i < sdm->d; i++) {
		v_in[i] ? p_cntr[i]++ : p_cntr[i]--;
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
		sdm_cntradd(sdm, &(sdm->cntr[sdm->actl[i] * sdm->d]));
	}
}

int sdm_sum2bit(short cntrsum) {
	//return (cntrsum > 0) ? 1 : ((cntrsum < 0) ? 0 : (int) (random() % 2));
	return cntrsum > 0; //todo: clarify what to do if 0
}

void sdm_sum2bin(sdm_jaekel_t *sdm, int *vect) {
	int i;

	for (i = 0; i < sdm->d; ++i) {
		vect[i] = sdm_sum2bit(sdm->sumc[i]);
	}
}


//Main functions:
void sdm_init(sdm_jaekel_t *sdm, unsigned short n, unsigned short d, unsigned short k) {
	sdm->cntr = (short *) malloc(n * d * sizeof(short));
	sdm->mask = (unsigned long *) malloc(n * 2 * sizeof(long));
	sdm->n = n;
	sdm->d = d;
	sdm->k = k;

	sdm->actl = (unsigned short *) malloc(n * sizeof(short));
	sdm->sumc = (short *) malloc(d * sizeof(short));
	sdm->nact = 0;

	memset(sdm->cntr, 0, n * d * sizeof(short));
	memset(sdm->actl, 0, n * sizeof(short));
	memset(sdm->sumc, 0, d * sizeof(short));

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
				selection_mask += 1 << rnd_digit;
				value_mask += (random() % 2) << rnd_digit;
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

	for (i = 0; i < sdm->n; ++i) {
		printf("#%3d:", i);

		//Run through Counters:
		for (j = sdm->d * i; j < sdm->d * (i + 1); ++j) {
			short cntr = sdm->cntr[j];
			printf("%4d", cntr);
		}
		printf("\t\t");

		//Run through Indexes:
		for (j = sdm->k * i; j < sdm->k * (i + 1); ++j) {
			int idx = sdm->mask[j];
			printf("%5d", idx);
		}
		printf("\n");
	}

	printf("\n");
}


int sdm_write(sdm_jaekel_t *sdm, long addr, long v_in) {
	int i;

	sdm_activate(sdm, addr);

	for (i = 0; i < sdm->nact; i++) {
		sdm_cntrvary(sdm, &(sdm->cntr[sdm->actl[i] * sdm->d]), v_in);
	}

	return sdm->nact;
}

int sdm_read(sdm_jaekel_t *sdm, long addr, long v_out) {
	sdm_activate(sdm, addr);
	sdm_cntrsum(sdm);
	sdm_sum2bin(sdm, v_out);

	return sdm->nact;
}
