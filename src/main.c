#include <stdio.h>
#include "sdm_jaekel.h"


int main(int argc, char *argv[]) {
	sdm_jaekel_t sdm;

//	if (argc == 1) {
//		printf("Example of use: sdm 100 10 3, where n=100; d=10; k=3\n");
//		return 1;
//	}

	unsigned short n = 1024; //atoi(argv[1]);
	unsigned short d = 64; //atoi(argv[2]);
	unsigned short k = 8; //atoi(argv[3]);

	//difference in 2 digits:                           |                    |
	unsigned long addr1 = STR2BIN("1110101010100100111010101010010001010010011101010101001000101001", d);
	unsigned long addr2 = STR2BIN("1110101010100100111011101010010001010010010101010101001000101001", d);
	unsigned long v_in  = STR2BIN("0000001000000000100000001000100000000000000100000000001000000100", d);
	unsigned long v_out;

	sdm_init(&sdm, n, d, k);

	sdm_print(&sdm);

	printf("addr1: %s\n", BIN2STR(addr1, d));
	printf(" v_in: %s\n", BIN2STR(v_in, d));
	printf("Write: %d locations activated\n\n", sdm_write(&sdm, addr1, v_in));

	sdm_print(&sdm);

	printf("addr1: %s\n", BIN2STR(addr1, d));
	printf("Read1: %d locations activated\n", sdm_read(&sdm, addr1, &v_out));
	printf("v_out: %s\n\n", BIN2STR(v_out, d));

	printf("addr2: %s\n", BIN2STR(addr2, d));
	printf("Read2: %d locations activated\n", sdm_read(&sdm, addr2, &v_out));
	printf("v_out: %s\n\n", BIN2STR(v_out, d));

	sdm_print(&sdm);

	sdm_free(&sdm);

	return 0;
}
