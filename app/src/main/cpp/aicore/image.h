#ifndef _IMAGE_H_
#define _IMAGE_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include "znet.h"
#include "bitmap.h"
#include "zutils.h"

struct image_standardizer;
typedef struct image_standardizer image_standardizer;

AICORE_LOCAL image *create_image(int width, int height, int nchannels);
AICORE_LOCAL void free_image(image *img);
#ifdef __INTEL_SSE__
AICORE_LOCAL void split_channel_sse(unsigned char *src, unsigned char *dst, int src_pitch, int w, int h);
#endif
#ifdef __ARM_NEON__
AICORE_LOCAL void split_channel_neon(unsigned char *src, unsigned char *dst, int src_pitch, int w, int h);
#endif
AICORE_LOCAL void split_channel(unsigned char *src, unsigned char *dst, int src_pitch, int w, int h);
#ifdef __ARM_NEON__
AICORE_LOCAL void resize_image_neon(unsigned char *src, unsigned char *dst, int src_w, int src_h,
	int dst_w, int dst_h, int nchannels);
#endif
AICORE_LOCAL void package_neighbor_pixle(int src_w, int src_h, int dst_w, int dst_h, short *x_tab,
	short *y_tab, unsigned char *src, unsigned char *pack);
#ifdef __ARM_NEON__	
AICORE_LOCAL void resize_image_neon_faster(unsigned char *pack, unsigned char *dst, int dst_w, int dst_h,
	int nchannels, unsigned short *dx_tab, unsigned short *dy_tab);
#endif
AICORE_LOCAL void resize_image_hv(unsigned char *src, unsigned char *dst, int src_w, int src_h,
	int dst_w, int dst_h, int nchannels);
AICORE_LOCAL void resize_image(unsigned char *src, unsigned char *dst, int src_w, int src_h,
                  int dst_w, int dst_h, int nchannels);
#ifdef __ARM_NEON__
AICORE_LOCAL void embed_image_neon(unsigned char *src, image *dst, int src_w, int src_h);
AICORE_LOCAL void make_bilinear_interp_table(int src_w, int src_h, int dst_w, int dst_h, short *x_tab,
	short *y_tab, unsigned short *dx_tab, unsigned short *dy_tab);
#endif
AICORE_LOCAL void embed_image(unsigned char *src, image *dst, int src_w, int src_h);
AICORE_LOCAL void set_image(image *img, float val);
AICORE_LOCAL void vertical_mirror(image *img);
AICORE_LOCAL void swap_channel(image *img);
AICORE_LOCAL image_standardizer *create_image_standardizer(int width, int height, int standard_width, int standard_height, int nchannels);
AICORE_LOCAL void *get_standardizer_output_ptr(image_standardizer *standardizer);
AICORE_LOCAL void standardize_image(image_standardizer *standardizer, const unsigned char *const rgb24,
	unsigned int width, unsigned int height, int roix, int roiy, int roiw, int roih, void *output);
AICORE_LOCAL void standardize_ion_image(image_standardizer *standardizer, void *input, unsigned int width,
	unsigned int height, int roix, int roiy, int roiw, int roih, void *output);
AICORE_LOCAL void free_image_standardizer(image_standardizer *standardizer);

#ifdef __cplusplus
}
#endif

#endif