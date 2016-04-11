#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include "sdm_jaekel.h"
#include "bmp.h"


int main(int argc, char *argv[]) {
	sdm_jaekel_t sdm;
	bmp8_t bmp;

	int i, j, l;
	unsigned n, d, k, vectors_in_layer, part_bits = sizeof(unsigned) * 8;
	unsigned **vectors, **p_vectors, *vector, *p_vector;
	unsigned char *pixels_out, *p_pixels_out;
	char *path_in = "img/lena512.bmp", path_out[36];

	if (argc < 4) {
		printf("Example of use:\n\n"
				       "sdm [n d k]\n\n"
				       "n - number of locations, default value is 1024 (multiple of 1024)\n"
				       "d - number of digits, default value is 8192 (multiple of 32)\n"
				       "k - number of selection-bits in mask, default value is 3 \n\n");

		if (argc > 1) {
			return 0;
		}

		n = 1024;
		d = 8192;
		k = 3;
	} else {
		n = (unsigned) atoi(argv[1]);
		d = (unsigned) atoi(argv[2]);
		k = (unsigned) atoi(argv[3]);
	}

	sprintf(path_out, "img/lena512_%d_%d_%d_%d.bmp", n, d, k, (unsigned) time(NULL));

	sdm_init(&sdm, n, d, k);

	//Reading bmp from file to array of pixels
	bmp_read(&bmp, path_in);

	vectors_in_layer = bmp.infoHeader.biSizeImage / d;

	vectors = (unsigned **) malloc(bmp.infoHeader.biBitCount * vectors_in_layer * sizeof(unsigned *));
	p_vectors = vectors;

	pixels_out = (unsigned char *) malloc(bmp.infoHeader.biSizeImage * sizeof(char));
	p_pixels_out = pixels_out;

	//Writing pixels by layers into SDM
	vector = (unsigned *) malloc(d / 8);
	p_vector = vector;
	memset(vector, 0, d / 8);

	for (j = 0; j < bmp.infoHeader.biBitCount; ++j) {
		unsigned char layer_mask = (unsigned char) 1 << j;

		printf("writing   to layer: #%d\n", j + 1);

		for (i = 0; i < bmp.infoHeader.biSizeImage; ++i) {
			*p_vector <<= 1;
			*p_vector |= (bmp.pixels[i] & layer_mask) > 0;

			if ((i + 1) % part_bits == 0) {
				p_vector++;
			}

			if ((i + 1) % d == 0) {
				sdm_write(&sdm, vector, vector);

				*p_vectors++ = vector;

				vector = (unsigned *) malloc(d / 8);
				p_vector = vector;
				memset(vector, 0, d / 8);
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
		unsigned *vector_out = (unsigned *) malloc(d / 8);

		if (i % vectors_in_layer == 0) {
			printf("reading from layer: #%d\n", i / vectors_in_layer + 1);
		}

		sdm_read(&sdm, vectors[i], vector_out);
		vectors[i] = vector_out;
	}

	for (l = 0; l < vectors_in_layer; ++l) { //converting vectors to pixels
		for (j = 0; j < d; ++j) {
			unsigned char pixel = 0;
			unsigned part_num = j / part_bits;
			unsigned num_in_part = j % part_bits;
			unsigned digit_mask = 1U << (part_bits - num_in_part - 1);

			for (i = bmp.infoHeader.biBitCount - 1; i >= 0; --i) {
				unsigned vector = vectors[i * vectors_in_layer + l][part_num];

				pixel <<= 1;
				pixel |= (vector & digit_mask) > 0;
			}

			*p_pixels_out++ = pixel;
		}
	}

	printf("\n");
	for (i = 0; i < bmp.infoHeader.biSizeImage; ++i) { //writing pixels to new bmp file
		fputc(pixels_out[i], file_out);
	}

	fclose(file_out);

	sdm_free(&sdm);

	return 0;
}
