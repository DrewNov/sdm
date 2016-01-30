#include <stdio.h>
#include "sdm_jaekel.h"
#include "bmp.h"


int main(int argc, char *argv[]) {
	sdm_jaekel_t sdm;

//	if (argc == 1) {
//		printf("Example of use: sdm 100 10 3, where n=100; d=10; k=3\n");
//		return 1;
//	}

	unsigned short n = 1024; //atoi(argv[1]);
	unsigned short d = 64; //atoi(argv[2]);
	unsigned short k = 8; //atoi(argv[3]);
//
//	//difference in 2 digits:                           |                    |
//	unsigned long addr1 = STR2BIN("1110101010100100111010101010010001010010011101010101001000101001", d);
//	unsigned long addr2 = STR2BIN("1110101010100100111011101010010001010010010101010101001000101001", d);
//	unsigned long v_in  = STR2BIN("0000001000000000100000001000100000000000000100000000001000000100", d);
//	unsigned long v_out;
//
	sdm_init(&sdm, n, d, k);
//
//	sdm_print(&sdm);
//
//	printf("addr1: %s\n", BIN2STR(addr1, d));
//	printf(" v_in: %s\n", BIN2STR(v_in, d));
//	printf("Write: %d locations activated\n\n", sdm_write(&sdm, addr1, v_in));
//
//	sdm_print(&sdm);
//
//	printf("addr1: %s\n", BIN2STR(addr1, d));
//	printf("Read1: %d locations activated\n", sdm_read(&sdm, addr1, &v_out));
//	printf("v_out: %s\n\n", BIN2STR(v_out, d));
//
//	printf("addr2: %s\n", BIN2STR(addr2, d));
//	printf("Read2: %d locations activated\n", sdm_read(&sdm, addr2, &v_out));
//	printf("v_out: %s\n\n", BIN2STR(v_out, d));

	bmp8_t bmp;
	int vectors_in_layer;
	unsigned long *vectors;
	unsigned long *p_vectors;
	unsigned char *pixels_out;
	unsigned char *p_pixels_out;
	char *path_in = "img/lena32.bmp";
	char *path_out = "img/lena32_out.bmp";

	//Reading bmp from file to array of pixels
	bmp_read(&bmp, path_in);

	vectors_in_layer = bmp.infoHeader.biSizeImage / d;

	vectors = (unsigned long *) malloc(bmp.infoHeader.biBitCount * vectors_in_layer * sizeof(long));
	p_vectors = vectors;

	pixels_out = (unsigned char *) malloc(bmp.infoHeader.biSizeImage * sizeof(char));
	p_pixels_out = pixels_out;

	//Writing pixels by layers into SDM
	for (int j = 0; j < bmp.infoHeader.biBitCount; ++j) {
		unsigned long vector = 0;
		unsigned char layer_mask = 1L << j;

		for (int i = 0; i < bmp.infoHeader.biSizeImage; ++i) {
			vector <<= 1;
			vector |= (bmp.pixels[i] & layer_mask) >> j;

			if ((i + 1) % d == 0) {
				sdm_write(&sdm, vector, vector);
				*p_vectors++ = vector;
				vector = 0;
			}
		}
	}

	//Reading from SDM and writing to new bmp file with the same headers
	FILE *file_in = fopen(path_in, "rb");
	FILE *file_out = fopen(path_out, "wb");

	for (int i = 0; i < bmp.header.bfOffBits; ++i) { //copying headers
		fputc(fgetc(file_in), file_out);
	}

	fclose(file_in);

	for (int i = 0; i < bmp.infoHeader.biBitCount * vectors_in_layer; ++i) { //reading vectors from SDM
		unsigned long vector_out;
		sdm_read(&sdm, vectors[i], &vector_out);
		vectors[i] = vector_out;
	}

	for (int k = 0; k < vectors_in_layer; ++k) { //converting vectors to pixels
		for (int j = d - 1; j >= 0; --j) {
			unsigned char pixel = 0;
			unsigned long digit_mask = 1L << j;

			for (int i = bmp.infoHeader.biBitCount - 1; i >= 0; --i) {
				unsigned long vector = vectors[i * vectors_in_layer + k];
				pixel <<= 1;
				pixel |= (vector & digit_mask) >> j;
			}

			*p_pixels_out++ = pixel;
		}
	}

	for (int i = 0; i < bmp.infoHeader.biSizeImage; ++i) { //writing pixels to new bmp file
		fputc(pixels_out[i], file_out);
	}

	fclose(file_out);

	sdm_free(&sdm);

	return 0;
}
