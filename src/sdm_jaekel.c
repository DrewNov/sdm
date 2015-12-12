#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include "sdm_jaekel.h"

int sdm_init(sdm_jaekel_t *sdm, unsigned int n, unsigned int s) {
	sdm->nbit = n;
	sdm->nlng = (unsigned int) (n / (8 * sizeof(unsigned int)));
	sdm->nloc = s;
	sdm->nact = 0;
	sdm->nsat = 0;
	sdm->addr = (unsigned int *) malloc(s * n / 8);
	sdm->cntr = (int *) malloc(s * n * sizeof(int));
	sdm->actl = (unsigned int *) malloc(s * sizeof(unsigned int));
	sdm->dist = (unsigned int *) malloc(2 * s * sizeof(unsigned int));
	sdm->sumc = (long *) malloc(n * sizeof(signed long));

	if (sdm->addr == 0 || sdm->cntr == 0 || sdm->actl == 0 ||
		sdm->dist == 0 || sdm->sumc == 0 || n != sdm->nlng * 8 * sizeof(unsigned int)) {
		sdm_free(sdm);
		return 1;
	}

	/* Reset arrays to zero */
	memset(sdm->cntr, 0, s * n * sizeof(int));
	memset(sdm->actl, 0, s * sizeof(unsigned int));
	memset(sdm->dist, 0, 2 * s * sizeof(unsigned int));
	memset(sdm->sumc, 0, n * sizeof(signed long));

	/* Initialize addresses randomly */
	srandom((unsigned int) time(NULL));
	unsigned int i = (unsigned int) (s * n / (8 * sizeof(unsigned int)));
	unsigned int *a = sdm->addr;
	while (i--) (*a++) = (unsigned int) (random() % (long) pow(2, sizeof(int) * 8));

	return 0;
}

void sdm_free(sdm_jaekel_t *sdm) {
	if (sdm->addr != 0) free(sdm->addr);
	if (sdm->cntr != 0) free(sdm->cntr);
	if (sdm->actl != 0) free(sdm->actl);
	if (sdm->dist != 0) free(sdm->dist);
	if (sdm->sumc != 0) free(sdm->sumc);
	memset(sdm, 0, sizeof(sdm_jaekel_t));
}