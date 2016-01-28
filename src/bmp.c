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

int bmp() {
	char *path = "girl.bmp";
	FILE *file_in = fopen(path, "rb");

	if (!file_in) {
		printf("File '%s' not found!\n", path);
		return 1;
	}

	BITMAPFILEHEADER header;

	header.bfType = read_u16(file_in);
	header.bfSize = read_u32(file_in);
	header.bfReserved1 = read_u16(file_in);
	header.bfReserved2 = read_u16(file_in);
	header.bfOffBits = read_u32(file_in);

	BITMAPINFOHEADER infoHeader;

	infoHeader.biSize = read_u32(file_in);
	infoHeader.biWidth = read_u32(file_in);
	infoHeader.biHeight = read_u32(file_in);
	infoHeader.biPlanes = read_u16(file_in);
	infoHeader.biBitCount = read_u16(file_in);
	infoHeader.biCompression = read_u32(file_in);
	infoHeader.biSizeImage = read_u32(file_in);
	infoHeader.biXPelsPerMeter = read_u32(file_in);
	infoHeader.biYPelsPerMeter = read_u32(file_in);
	infoHeader.biClrUsed = read_u32(file_in);
	infoHeader.biClrImportant = read_u32(file_in);

	printf("bfType: %d\n", header.bfType);
	printf("bfSize: %d\n", header.bfSize);
	printf("bfOffBits: %d\n\n", header.bfOffBits);

	printf("biSize: %d\n", infoHeader.biSize);
	printf("biSizeImage: %d\n", infoHeader.biSizeImage);
	printf("biWidth: %d\n", infoHeader.biWidth);
	printf("biHeight: %d\n", infoHeader.biHeight);
	printf("biBitCount: %d\n", infoHeader.biBitCount);
	printf("biPlanes: %d\n", infoHeader.biPlanes);
	printf("biCompression: %d\n", infoHeader.biCompression);

	fclose(file_in);

	return 0;
}
