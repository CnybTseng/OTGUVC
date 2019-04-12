#include <stdio.h>
#include <stdlib.h>
#include "bitmap.h"

#define FOUR_BYTES_ALIGN(width_in_bit)  ((((width_in_bit)+31)>>5)<<2)

#pragma pack(1)
typedef struct {
	unsigned short type;
	unsigned int   size;
	unsigned short reserved1;
	unsigned short reserved2;
	unsigned int   off_bits;
} bit_map_file_header;

typedef struct {
	unsigned int   size;
	unsigned int   width;
	unsigned int   height;
	unsigned short planes;
	unsigned short bit_count;
	unsigned int   compression;
	unsigned int   size_image;
	unsigned int   x_pels_per_meter;
	unsigned int   y_pels_per_meter;
	unsigned int   clr_used;
	unsigned int   clr_important;
} bit_map_info_header;

typedef struct {
	unsigned char blue;
	unsigned char green;
	unsigned char red;
	unsigned char reserved;
} bgra;

struct bitmap {
	bit_map_file_header file_header;
	bit_map_info_header info_header;
	bgra *palette;
	unsigned char *data;
};
#pragma pack()

bitmap *read_bmp(const char *filename)
{	
	bitmap *bmp = calloc(1, sizeof(bitmap));
	if (!bmp) {
		fprintf(stderr, "calloc[%s:%d].\n", __FILE__, __LINE__);
		return bmp;
	}
	
	FILE *fp = fopen(filename, "rb");
	if (!fp) {
		fprintf(stderr, "fopen[%s:%d].\n", __FILE__, __LINE__);
		goto cleanup;
	}
	
	fread(&bmp->file_header, sizeof(bit_map_file_header), 1, fp);
	if (bmp->file_header.type != 0x4D42) {
		fprintf(stderr, "bitmap type error!\n");
		goto cleanup;
	}
	
	fread(&bmp->info_header, sizeof(bit_map_info_header), 1, fp);
	if (bmp->info_header.bit_count < 24) {
		int ncolors = 1 << bmp->info_header.bit_count;
		bmp->palette = calloc(ncolors, sizeof(bgra));
		if (!bmp->palette) {
			fprintf(stderr, "calloc[%s:%d].\n", __FILE__, __LINE__);
			goto cleanup;
		}
		
		fread(bmp->palette, sizeof(bgra), ncolors, fp);
	}

	int pitch = FOUR_BYTES_ALIGN(bmp->info_header.bit_count * bmp->info_header.width);
	bmp->data = calloc(bmp->info_header.height * pitch, sizeof(unsigned char));
	if (!bmp->data) {
		fprintf(stderr, "calloc[%s:%d].\n", __FILE__, __LINE__);
		cleanup:delete_bmp(bmp);
		if (fp) fclose(fp);
		return NULL;
	}
	
	fread(bmp->data, sizeof(unsigned char), bmp->info_header.height * pitch, fp);
	fclose(fp);
		
	return bmp;
}

bitmap *create_bmp(const char *const data, int width, int height, int bits_per_pixel)
{
	bitmap *bmp = calloc(1, sizeof(bitmap));
	if (!bmp) {
		fprintf(stderr, "calloc[%s:%d].\n", __FILE__, __LINE__);
		return bmp;
	}
	
	int ncolors = 1 << bits_per_pixel;
	int not_color = bits_per_pixel < 24;
	int pitch = FOUR_BYTES_ALIGN(bits_per_pixel * width);
		
	bmp->file_header.type = 0x4D42;
	bmp->file_header.size = sizeof(bit_map_file_header) + sizeof(bit_map_info_header) +
		not_color * ncolors * sizeof(bgra) + pitch * height;
	bmp->file_header.reserved1 = 0;
	bmp->file_header.reserved2 = 0;
	bmp->file_header.off_bits = bmp->file_header.size - pitch * height;
	
	bmp->info_header.size = sizeof(bit_map_info_header);
	bmp->info_header.width = width;
	bmp->info_header.height = height;
	bmp->info_header.planes = 1;
	bmp->info_header.bit_count = bits_per_pixel;
	bmp->info_header.compression = 0;
	bmp->info_header.size_image = width * height * (bits_per_pixel >> 3);
	bmp->info_header.x_pels_per_meter = 0;
	bmp->info_header.y_pels_per_meter = 0;
	bmp->info_header.clr_used = 0;
	bmp->info_header.clr_important = 0;
	
	if (not_color) {
		bmp->palette = calloc(ncolors, sizeof(bgra));
		if (!bmp->palette) {
			fprintf(stderr, "calloc[%s:%d].\n", __FILE__, __LINE__);
			goto cleanup;
		}
		
		for (int i = 0; i < ncolors; ++i) {
			bmp->palette[i].blue = i;
			bmp->palette[i].green = i;
			bmp->palette[i].red = i;
			bmp->palette[i].reserved = i;
		}
	}
	
	bmp->data = calloc(height * pitch, sizeof(unsigned char));
	if (!bmp->data) {
		fprintf(stderr, "calloc[%s:%d].\n", __FILE__, __LINE__);
		cleanup:delete_bmp(bmp);
		return NULL;
	}
	
	int line_width = (bits_per_pixel >> 3) * width;
	for (int y = 0; y < height; ++y) {
		for (int x = 0; x < line_width; ++x) {
			bmp->data[y * pitch + x] = data[y * line_width + x];
		}
	}
	
	return bmp;
}

unsigned char *get_bmp_data(bitmap *bmp)
{
	return bmp->data;
}

int get_bmp_width(bitmap *bmp)
{
	return (int)bmp->info_header.width;
}

int get_bmp_height(bitmap *bmp)
{
	return (int)bmp->info_header.height;
}

int get_bmp_bit_count(bitmap *bmp)
{
	return (int)bmp->info_header.bit_count;
}

int get_bmp_pitch(bitmap *bmp)
{
	return FOUR_BYTES_ALIGN(bmp->info_header.bit_count * bmp->info_header.width);
}

void save_bmp(bitmap *bmp, const char *filename)
{
	FILE *fp = fopen(filename, "wb");
	if (!fp) {
		fprintf(stderr, "fopen[%s:%d].\n", __FILE__, __LINE__);
		return;
	}
	
	fwrite(&bmp->file_header, sizeof(bit_map_file_header), 1, fp);
	fwrite(&bmp->info_header, sizeof(bit_map_info_header), 1, fp);
	
	if (bmp->info_header.bit_count < 24) {
		int ncolors = 1 << bmp->info_header.bit_count;
		fwrite(bmp->palette, sizeof(bgra), ncolors, fp);
	}
	
	int pitch = FOUR_BYTES_ALIGN(bmp->info_header.bit_count * bmp->info_header.width);
	fwrite(bmp->data, sizeof(unsigned char), bmp->info_header.height * pitch, fp);
	fclose(fp);
}

void delete_bmp(bitmap *bmp)
{
	if (!bmp) return;
	
	if (bmp->palette) {
		free(bmp->palette);
		bmp->palette = NULL;
	}
	
	if (bmp->data) {
		free(bmp->data);
		bmp->data = NULL;
	}
	
	free(bmp);
	bmp = NULL;
}