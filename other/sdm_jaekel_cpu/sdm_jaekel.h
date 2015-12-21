/*
 * This is implementation of Jaekel's version of SDM
 * by DrewNov, December 2015.
 * C Binary Vector Symbols (CBVS) was taken as a basis. (http://pendicular.net/cbvs.php)
 */

#define STR2BIN(x, n) str2bin(x, n)
#define BIN2STR(x, n) bin2str(x, n)


inline int *str2bin(const char *str, const int n) {
	int i = 0;
	int *bin = (int *) malloc(n * sizeof(int));

	for (i = 0; i < n; ++i) {
		bin[i] = str[i] - '0';
	}

	return bin;
}

inline char *bin2str(const int *bin, const int n) {
	int i = 0;
	char *str = (char *) malloc(n * sizeof(char) + 1);

	for (i = 0; i < n; ++i) {
		str[i] = (char) (bin[i] + '0');
	}

	str[i] = '\0';

	return str;
}

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
void sdm_print(sdm_jaekel_t *sdm);

int sdm_write(sdm_jaekel_t *sdm, int *addr, int *v_in);
int sdm_read(sdm_jaekel_t *sdm, int *addr, int *v_out);
