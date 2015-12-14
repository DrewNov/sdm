/*
 * This is implementation of Jaekel's version of SDM
 * by DrewNov, December 2015.
 * C Binary Vector Symbols (CBVS) was taken as a basis. (http://pendicular.net/cbvs.php)
 */

typedef struct {
	/* SDM state variables and parameters */
	int *cntr;		/* Counters */
	int *idxs;		/* Indexes */
	int nloc;		/* Number of locations */
	int ndim;		/* Number of dimensions in location */
	int nidx;		/* Number of indexes in location */

	/* SDM read/write state variables */
	int *actl;		/* Active locations */
	long *sumc;		/* Sum of counter vectors */
	int nact;		/* Number of active locations */
} sdm_jaekel_t;


void sdm_init(sdm_jaekel_t *sdm, int nloc, int ndim, int nidx);
void sdm_free(sdm_jaekel_t *sdm);

int sdm_write(sdm_jaekel_t *sdm, int *addr, int *v_in);
int sdm_read(sdm_jaekel_t *sdm, int *addr, int *v_out);
