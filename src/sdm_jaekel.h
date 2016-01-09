/*
 * This is implementation of Jaekel's version of SDM
 * by DrewNov, December 2015.
 * C Binary Vector Symbols (CBVS) was taken as a basis. (http://pendicular.net/cbvs.php)
 */

#define STR2BIN(x, n) str2bin(x, n)
#define BIN2STR(x, n) bin2str(x, n)


inline unsigned long str2bin(char *str, int n) {
	int i = 0;
	unsigned long bin = 0;

	for (i = 0; i < n; ++i) {
		bin <<= 1;
		bin |= str[i] - '0';
	}

	return bin;
}

inline char *bin2str(unsigned long bin, int n) {
	int i;
	char *str = (char *) malloc(n * sizeof(char) + 1);

	for (i = n - 1; i >= 0; --i) {
		str[i] = (char) ((bin & 1) + '0');
		bin >>= 1;
	}

	str[n] = '\0';

	return str;
}

typedef struct {
	/* SDM state variables and parameters */
	short *cntr;			/* Counters */
	unsigned long *mask;	/* Indexes */
	unsigned short n;		/* Number of locations */
	unsigned short d;		/* Number of dimensions in location */
	unsigned short k;		/* Number of selection-bits in mask */

	/* SDM read/write state variables */
	unsigned short *actl;	/* Active locations */
	short *sumc;			/* Sum of counter vectors */
	unsigned short nact;	/* Number of active locations */
} sdm_jaekel_t;


void sdm_init(sdm_jaekel_t *sdm, unsigned short n, unsigned short d, unsigned short k);
void sdm_free(sdm_jaekel_t *sdm);
void sdm_print(sdm_jaekel_t *sdm);

int sdm_write(sdm_jaekel_t *sdm, unsigned long addr, unsigned long v_in);
int sdm_read(sdm_jaekel_t *sdm, unsigned long addr, unsigned long *v_out);
