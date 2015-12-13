#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "sdm_jaekel.h"

void sdm_init(sdm_jaekel_t *sdm, int nloc, int ndim, int nidx) {
	sdm->cntr = (int *) malloc(nloc * ndim * sizeof(int));
	sdm->idxs = (int *) malloc(nloc * nidx * sizeof(int));
	sdm->nloc = nloc;
	sdm->ndim = nloc;
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
		int rnd = (int) (random() % ndim);
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