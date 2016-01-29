#include <stdio.h>
#include "bmp.h"

unsigned short read_u16(FILE *fp) {
	unsigned char b0, b1;

	b0 = fgetc(fp);
	b1 = fgetc(fp);

	return ((b1 << 8) | b0);
}

unsigned int read_u32(FILE *fp) {
	unsigned char b0, b1, b2, b3;

	b0 = fgetc(fp);
	b1 = fgetc(fp);
	b2 = fgetc(fp);
	b3 = fgetc(fp);

	return ((((((b3 << 8) | b2) << 8) | b1) << 8) | b0);
}

int bmp_read(bmp8_t *bmp, char *path) {
	FILE *file_in = fopen(path, "rb");

	if (!file_in) {
		printf("File '%s' not found!\n", path);
		return 1;
	}

	bmp->header.bfType = read_u16(file_in);
	bmp->header.bfSize = read_u32(file_in);
	bmp->header.bfReserved1 = read_u16(file_in);
	bmp->header.bfReserved2 = read_u16(file_in);
	bmp->header.bfOffBits = read_u32(file_in);

	bmp->infoHeader.biSize = read_u32(file_in);
	bmp->infoHeader.biWidth = read_u32(file_in);
	bmp->infoHeader.biHeight = read_u32(file_in);
	bmp->infoHeader.biPlanes = read_u16(file_in);
	bmp->infoHeader.biBitCount = read_u16(file_in);
	bmp->infoHeader.biCompression = read_u32(file_in);
	bmp->infoHeader.biSizeImage = read_u32(file_in);
	bmp->infoHeader.biXPelsPerMeter = read_u32(file_in);
	bmp->infoHeader.biYPelsPerMeter = read_u32(file_in);
	bmp->infoHeader.biClrUsed = read_u32(file_in);
	bmp->infoHeader.biClrImportant = read_u32(file_in);

	printf("bfType: %d\n", bmp->header.bfType);
	printf("bfSize: %d\n", bmp->header.bfSize);
	printf("bfOffBits: %d\n\n", bmp->header.bfOffBits);

	printf("biSize: %d\n", bmp->infoHeader.biSize);
	printf("biSizeImage: %d\n", bmp->infoHeader.biSizeImage);
	printf("biWidth: %d\n", bmp->infoHeader.biWidth);
	printf("biHeight: %d\n", bmp->infoHeader.biHeight);
	printf("biBitCount: %d\n", bmp->infoHeader.biBitCount);
	printf("biPlanes: %d\n", bmp->infoHeader.biPlanes);
	printf("biCompression: %d\n", bmp->infoHeader.biCompression);
	printf("biClrUsed: %d\n", bmp->infoHeader.biClrUsed);
	printf("biClrImportant: %d\n\n", bmp->infoHeader.biClrImportant);

	bmp->pixels = (unsigned char *) malloc(sizeof(unsigned char) * bmp->infoHeader.biSizeImage);
	unsigned char *p_pixels = bmp->pixels;

	fseek(file_in, bmp->header.bfOffBits, SEEK_SET);

	for (int i = 0; i < bmp->infoHeader.biHeight; ++i) {
		for (int j = 0; j < bmp->infoHeader.biWidth; ++j) {
			*p_pixels = fgetc(file_in);
			printf("%4d", *p_pixels);
			p_pixels++;
		}
		printf("\n");
	}

	fclose(file_in);

	return 0;
}
