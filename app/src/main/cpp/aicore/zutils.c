#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include "zutils.h"
#include "half.h"

void mmfree(int n, ...)
{
	va_list ap;
	va_start(ap, n);
	
	for (int i = 0; i < n; ++i) {
		void *p = va_arg(ap, void *);
		if (p) {
			free(p);
			p = NULL;
		}
	}
	
	va_end(ap);
}

void mset(char *const X, size_t size, const char *const val, int nvals)
{
	for (int i = 0; i < nvals; ++i) {
		for (size_t j = 0; j < size; j += nvals) {
			X[j + i] = val[i];
		}
	}
}

void mcopy(const char *const X, char *const Y, size_t size)
{
	for (size_t i = 0; i < size; ++i) {
		Y[i] = X[i];
	}
}

void save_volume(float *data, int width, int height, int nchannels, const char *path)
{
	FILE *fp = fopen(path, "w");
	if (!fp) {
		fprintf(stderr, "fopen[%s:%d].\n", __FILE__, __LINE__);
		return;
	}
	
	for (int c = 0; c < nchannels; ++c) {
		fprintf(fp, "channel=%d\n", c);
		float *at = data + c * width * height;
		for (int y = 0; y < height; ++y) {
			for (int x = 0; x < width; ++x) {
				fprintf(fp, "%.7f ", at[y * width + x]);
			}
			fputs("\n", fp);
		}
		fputs("\n\n\n", fp);
	}

	fclose(fp);
}

#ifdef OPENCL
int nchw_to_nhwc(const float *const input, MEM_MAP_PTR_TYPE *const output, int width, int height,
	int channels, int batch, int input_row_pitch, int output_row_pitch, int channel_block_size)
{
	int shifts = 0;
	if (4 == channel_block_size) shifts = 2;
	else if (8 == channel_block_size) shifts = 3;
	else {
		fprintf(stderr, "only support 4 or 8 channel_block_size!\n");
		return -1;
	}

	const int channel_blocks = (channels + channel_block_size - 1) >> shifts;
	const int input_slice_pitch = input_row_pitch * height;
	const int input_batch_size = input_row_pitch * height * channels;
	for (int b = 0; b < batch; ++b) {
		const float *src_batch = input + b * input_batch_size;
		MEM_MAP_PTR_TYPE *dst_batch = output + b * ((width << shifts) * channel_blocks);
		for (int k = 0; k < channel_blocks; ++k) {
			const float *src = src_batch + k * (input_slice_pitch << shifts);
			MEM_MAP_PTR_TYPE *dst = dst_batch + k * (width << shifts);
			int channel_remainder = channels - (k << shifts);
			channel_remainder = channel_remainder < channel_block_size ? channel_remainder : channel_block_size;
			for (int c = 0; c < channel_remainder; ++c) {
				for (int y = 0; y < height; ++y) {
					for (int x = 0; x < width; ++x) {
						dst[y * output_row_pitch + (x << shifts) + c] = 
							HOST_TO_DEVICE(src[c * input_slice_pitch + y * input_row_pitch + x]);
					}
				}
			}
			for (int c = channel_remainder; c < channel_block_size; ++c) {
				for (int y = 0; y < height; ++y) {
					for (int x = 0; x < width; ++x) {
						dst[y * output_row_pitch + (x << shifts) + c] = 0;
					}
				}
			}
		}
	}
	
	return 0;
}

int nhwc_to_nchw(const MEM_MAP_PTR_TYPE *const input, float *const output, int width, int height,
	int channels, int batch, int input_row_pitch, int output_row_pitch, int channel_block_size)
{
	int shifts = 0;
	if (4 == channel_block_size) shifts = 2;
	else if (8 == channel_block_size) shifts = 3;
	else {
		fprintf(stderr, "only support 4 or 8 channel_block_size!\n");
		return -1;
	}

	const int channel_blocks = (channels + channel_block_size - 1) >> shifts;
	const int output_slice_pitch = output_row_pitch * height;
	const int output_batch_size = output_row_pitch * height * channels;
	for (int b = 0; b < batch; ++b) {
		const MEM_MAP_PTR_TYPE *src_batch = input + b * ((width << shifts) * channel_blocks);
		float *dst_batch = output + b * output_batch_size;
		for (int k = 0; k < channel_blocks; ++k) {
			const MEM_MAP_PTR_TYPE *src = src_batch + k * (width << shifts);
			float *dst = dst_batch + k * (output_slice_pitch << shifts);
			int channel_remainder = channels - (k << shifts);
			channel_remainder = channel_remainder < channel_block_size ? channel_remainder : channel_block_size;
			for (int c = 0; c < channel_remainder; ++c) {
				for (int y = 0; y < height; ++y) {
					for (int x = 0; x < width; ++x) {
						dst[c * output_slice_pitch + y * output_row_pitch + x] =
							DEVICE_TO_HOST(src[y * input_row_pitch + (x << shifts) + c]);
					}
				}
			}
		}
	}
	
	return 0;
}
#endif

int round_up_division_2(int x)
{
	return (x + 1) >> 1;
}

int round_up_division_4(int x)
{
	return (x + 3) >> 2;
}

unsigned int roundup_power_of_2(unsigned int a)
{
	unsigned int position;
	int i;
	
	if (a == 0) {
		return 0;
	}

	position = 0;
	for (i = a; i != 0; i >>= 1) {
		position++;
	}

	return (unsigned int)(1 << position);
}

unsigned int round_up_multiple_of_8(unsigned int x)
{
	return ((x + 7) >> 3) << 3;
}