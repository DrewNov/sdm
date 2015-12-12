typedef struct {
	/* SDM state variables and parameters */
	unsigned int *addr;		/* Addresses */
	         int *cntr;		/* Counters */
	unsigned int nloc;		/* Number of locations (S) */
	unsigned int nbit;		/* Number of bits in vectors (N) */
	unsigned int nlng;		/* Number of (unsigned) long words in vectors */
	unsigned int nsat;		/* Number of updates of saturated counters */

	/* SDM read/write state variables */
	unsigned int *actl;		/* Active locations */
	unsigned int *dist;		/* Distances of active locations */
	        long *sumc;		/* Sum of counter vectors */
	unsigned int nact;		/* Number of active locations */
} sdm_jaekel_t;


int sdm_init(sdm_jaekel_t *sdm, unsigned int n, unsigned int s);
void sdm_free(sdm_jaekel_t *sdm);