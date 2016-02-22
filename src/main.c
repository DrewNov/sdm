#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "sdm_jaekel.h"
#include "bmp.h"


int main(int argc, char *argv[]) {
	sdm_jaekel_t sdm;
	bmp8_t bmp;

	int i, j, l, vectors_in_layer;
	unsigned short n, d, k;
	unsigned long *vectors, *p_vectors;
	unsigned char *pixels_out, *p_pixels_out;
	char *path_in = "img/lena512.bmp", path_out[36];

	if (argc == 1) {
		printf("Example of use: sdm 1024 64 8, where n=1024; d=64; k=8\n");
		n = 1024;
		d = 64;
		k = 8;
	} else {
		n = (unsigned short) atoi(argv[1]);
		d = (unsigned short) atoi(argv[2]);
		k = (unsigned short) atoi(argv[3]);
	}

	sprintf(path_out, "img/lena512_%d_%d_%d_%d.bmp", n, d, k, (unsigned int) time(NULL));

	sdm_init(&sdm, n, d, k);

	//Reading bmp from file to array of pixels
	bmp_read(&bmp, path_in);

	vectors_in_layer = bmp.infoHeader.biSizeImage / d;

	vectors = (unsigned long *) malloc(bmp.infoHeader.biBitCount * vectors_in_layer * sizeof(long));
	p_vectors = vectors;

	pixels_out = (unsigned char *) malloc(bmp.infoHeader.biSizeImage * sizeof(char));
	p_pixels_out = pixels_out;

	//Writing pixels by layers into SDM
	for (j = 0; j < bmp.infoHeader.biBitCount; ++j) {
		unsigned long vector = 0;
		unsigned char layer_mask = (unsigned char) 1 << j;

		for (i = 0; i < bmp.infoHeader.biSizeImage; ++i) {
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

	for (i = 0; i < bmp.header.bfOffBits; ++i) { //copying headers
		fputc(fgetc(file_in), file_out);
	}

	fclose(file_in);

	for (i = 0; i < bmp.infoHeader.biBitCount * vectors_in_layer; ++i) { //reading vectors from SDM
		unsigned long vector_out;
		sdm_read(&sdm, vectors[i], &vector_out);
		vectors[i] = vector_out;
	}

	for (l = 0; l < vectors_in_layer; ++l) { //converting vectors to pixels
		for (j = d - 1; j >= 0; --j) {
			unsigned char pixel = 0;
			unsigned long digit_mask = 1UL << j;

			for (i = bmp.infoHeader.biBitCount - 1; i >= 0; --i) {
				unsigned long vector = vectors[i * vectors_in_layer + l];
				pixel <<= 1;
				pixel |= (vector & digit_mask) >> j;
			}

			*p_pixels_out++ = pixel;
		}
	}

	printf("\n");
	for (i = 0; i < bmp.infoHeader.biSizeImage; ++i) { //writing pixels to new bmp file
		fputc(pixels_out[i], file_out);

//		printf("%4d", pixels_out[i]);
//
//		if ((i + 1) % bmp.infoHeader.biWidth == 0) {
//			printf("\n");
//		}
	}

	fclose(file_out);

	sdm_free(&sdm);

	return 0;
}
