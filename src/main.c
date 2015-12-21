#include <stdio.h>
#include "sdm_jaekel.h"


int main(int argc, char *argv[]) {
	sdm_jaekel_t sdm;

//	if (argc == 1) {
//		printf("Example of use: sdm 100 10 3, where nloc=100; ndim=10; nidx=3\n");
//		return 1;
//	}

//	int nloc = 100; //atoi(argv[1]); //100
//	int ndim = 10; //atoi(argv[2]); //10
//	int nidx = 3; //atoi(argv[3]); //3
//
//	int *addr1 = STR2BIN("0101010101", ndim);
//	int *addr2 = STR2BIN("0101000101", ndim);
//	int *v_in  = STR2BIN("0100101010", ndim);
//	int *v_out = STR2BIN("0000000000", ndim);

	int nloc = 1024; //atoi(argv[1]); //100
	int ndim = 64; //atoi(argv[2]); //10
	int nidx = 8; //atoi(argv[3]); //3

	int *addr1 = STR2BIN("1110101010100100111010101010010001010010011101010101001000101001", ndim);
	int *addr2 = STR2BIN("1110101010100100111011101010010001010010010101010101001000101001", ndim);
	int *v_in  = STR2BIN("0000001000000000100000001000100000000000000100000000001000000100", ndim);
	int *v_out = STR2BIN("0000000000000000000000000000000000000000000000000000000000000000", ndim);

	sdm_init(&sdm, nloc, ndim, nidx);

//	sdm_print(&sdm);

	printf("addr1: %s\n", BIN2STR(addr1, ndim));
	printf(" v_in: %s\n", BIN2STR(v_in, ndim));
	printf("Write: %d locations activated\n", sdm_write(&sdm, addr1, v_in));
	printf("\n");

	printf("addr1: %s\n", BIN2STR(addr1, ndim));
	printf("Read1: %d locations activated\n", sdm_read(&sdm, addr1, v_out));
	printf("v_out: %s\n", BIN2STR(v_out, ndim));
	printf("\n");

	printf("addr2: %s\n", BIN2STR(addr2, ndim));
	printf("Read2: %d locations activated\n", sdm_read(&sdm, addr2, v_out));
	printf("v_out: %s\n", BIN2STR(v_out, ndim));

//	sdm_print(&sdm);

	sdm_free(&sdm);

	return 0;
}
