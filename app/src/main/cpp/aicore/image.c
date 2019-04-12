#include <omp.h>
#include <string.h>
#ifdef __INTEL_SSE__
#	include <emmintrin.h>
#	include <tmmintrin.h>
#elif __ARM_NEON__
#	include <arm_neon.h>
#endif
#include "image.h"
#ifdef OPENCL
#	include "cl_wrapper.h"
#endif
#include "half.h"

#ifdef OPENCL
extern cl_wrapper wrapper;
extern char BINARY_FILENAME_TO_START(cl_common, h);
extern char BINARY_FILENAME_TO_END(cl_common, h);
extern char BINARY_FILENAME_TO_START(utils, cl);
extern char BINARY_FILENAME_TO_END(utils, cl);
#endif

struct image_standardizer {
	int width;
	int height;
	int resized_width;
	int resized_height;
	int standard_width;
	int standard_height;
#if defined(OPENCL) && defined(WINOGRAD_CONVOLUTION)
	char *program_buffer;
	cl_program program;
	cl_kernel kernel;
	cl_mem d_input;
#else
	unsigned char *chw_image;
	unsigned char *resized_chw_image;
#endif
	image *output;
	float scale;
};

static void get_standardizer_parameter(int width, int height, int standard_width, int standard_height,
	int *resized_width, int *resized_height, float *scale);
static void set_fill_color(float (*color)[4], int nchannels);

image *create_image(int width, int height, int nchannels)
{
	image *img = calloc(1, sizeof(image));
	if (!img) {
		fprintf(stderr, "calloc[%s:%d].\n", __FILE__, __LINE__);
		return img;
	}
	
	img->w = width;
	img->h = height;
	img->c = nchannels;
	
	img->data = calloc(roundup_power_of_2(width * height * nchannels), sizeof(float));
	if (!img->data) {
		fprintf(stderr, "calloc[%s:%d].\n", __FILE__, __LINE__);
#ifdef OPENCL
		goto cleanup;
#else
		free_image(img);
		return 0;
#endif
	}
#ifdef OPENCL
	cl_mem_flags mem_flags = CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR;
	cl_image_format image_format = {
		.image_channel_order = CL_RGBA,
		.image_channel_data_type = IMAGE_CHANNEL_DATA_TYPE
	};
	
	cl_image_desc image_desc;
	memset(&image_desc, 0, sizeof(cl_image_desc));
	image_desc.image_type = CL_MEM_OBJECT_IMAGE2D;
	image_desc.image_width = width * round_up_division_4(nchannels);
	image_desc.image_height = height;
	
	cl_int errcode;
	img->d_data = clCreateImage(wrapper.context, mem_flags, &image_format, &image_desc, NULL, &errcode);
	if (CL_SUCCESS != errcode) {
		fprintf(stderr, "clCreateImage fail[%s:%d:%d].\n", __FILE__, __LINE__, errcode);
		cleanup:free_image(img);
		return 0;
	}
#endif
	return img;
}

void free_image(image *img)
{
	if (!img) return;
	if (img->data) {
		free(img->data);
		img->data = NULL;
	}
#ifdef OPENCL	
	clReleaseMemObject(img->d_data);
#endif	
	free(img);
	img = NULL;
}

#ifdef __INTEL_SSE__
void split_channel_sse(unsigned char *src, unsigned char *dst, int src_pitch, int w, int h)
{
	int pixels_per_load = 16;
	int excess = w - w % pixels_per_load;
	#pragma omp parallel for
	for (int y = 0; y < h; ++y) {
		unsigned char *psrc = src + y * src_pitch;
		unsigned char *pred = dst + y * w;
		unsigned char *pgrn = dst + w * (h + y);
		unsigned char *pblu = dst + w * ((h << 1) + y);
		for (int x = 0; x < excess; x += pixels_per_load) {
			__m128i BGR1 = _mm_loadu_si128((__m128i *)(psrc));
			__m128i BGR2 = _mm_loadu_si128((__m128i *)(psrc + 16));
			__m128i BGR3 = _mm_loadu_si128((__m128i *)(psrc + 32));
			
			__m128i B = _mm_shuffle_epi8(BGR1, _mm_setr_epi8(
				0, 3, 6, 9, 12, 15, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1));
			B = _mm_or_si128(B, _mm_shuffle_epi8(BGR2, _mm_setr_epi8(
				-1, -1, -1, -1, -1, -1, 2, 5, 8, 11, 14, -1, -1, -1, -1, -1)));
			B = _mm_or_si128(B, _mm_shuffle_epi8(BGR3, _mm_setr_epi8(
				-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 4, 7, 10, 13)));
			
			__m128i G = _mm_shuffle_epi8(BGR1, _mm_setr_epi8(
				1, 4, 7, 10, 13, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1));
			G = _mm_or_si128(G, _mm_shuffle_epi8(BGR2, _mm_setr_epi8(
				-1, -1, -1, -1, -1, 0, 3, 6, 9, 12, 15, -1, -1, -1, -1, -1)));
			G = _mm_or_si128(G, _mm_shuffle_epi8(BGR3, _mm_setr_epi8(
				-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 2, 5, 8, 11, 14)));
			
			__m128i R = _mm_shuffle_epi8(BGR1, _mm_setr_epi8(
				2, 5, 8, 11, 14, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1));
			R = _mm_or_si128(R, _mm_shuffle_epi8(BGR2, _mm_setr_epi8(
				-1, -1, -1, -1, -1, 1, 4, 7, 10, 13, -1, -1, -1, -1, -1, -1)));
			R = _mm_or_si128(R, _mm_shuffle_epi8(BGR3, _mm_setr_epi8(
				-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 3, 6, 9, 12, 15)));
			
			_mm_storeu_si128((__m128i *)pred, R);
			_mm_storeu_si128((__m128i *)pgrn, G);
			_mm_storeu_si128((__m128i *)pblu, B);
			
			psrc += 48;
			pred += 16;
			pgrn += 16;
			pblu += 16;
		}
	}
	
	if (excess == w) return;
	
	int swap[3] = {2, 1, 0};
	for (int c = 0; c < 3; ++c) {
		unsigned char *at = dst + swap[c] * w * h;
		for (int y = 0; y < h; ++y) {
			for (int x = excess; x < w; ++x) {
				at[y * w + x] = src[y * src_pitch + 3 * x + c];
			}
		}
	}
}
#endif

#ifdef __ARM_NEON__
void split_channel_neon(unsigned char *src, unsigned char *dst, int src_pitch, int w, int h)
{
	int pixels_per_load = 16;
	int excess = w - w % pixels_per_load;
	#pragma omp parallel for
	for (int y = 0; y < h; ++y) {
		unsigned char *psrc = src + y * src_pitch;
		unsigned char *pred = dst + y * w;
		unsigned char *pgrn = dst + w * (h + y);
		unsigned char *pblu = dst + w * ((h << 1) + y);
		for (int x = 0; x < excess; x += pixels_per_load) {
			uint8x16x3_t BGR16 = vld3q_u8(psrc);
			vst1q_u8(pred, BGR16.val[2]);
			vst1q_u8(pgrn, BGR16.val[1]);
			vst1q_u8(pblu, BGR16.val[0]);
			psrc += 48;
			pred += 16;
			pgrn += 16;
			pblu += 16;
		}
	}
	
	if (excess == w) return;
	
	int swap[3] = {2, 1, 0};
	for (int c = 0; c < 3; ++c) {
		unsigned char *at = dst + swap[c] * w * h;
		for (int y = 0; y < h; ++y) {
			for (int x = excess; x < w; ++x) {
				at[y * w + x] = src[y * src_pitch + 3 * x + c];
			}
		}
	}
}
#endif

void split_channel(unsigned char *src, unsigned char *dst, int src_pitch, int w, int h)
{
	int swap[3] = {2, 1, 0};
	for (int c = 0; c < 3; ++c) {
		unsigned char *at = dst + swap[c] * w * h;
		#pragma omp parallel for
		for (int y = 0; y < h; ++y) {
			for (int x = 0; x < w; ++x) {
				at[(h - 1 - y) * w + x] = src[y * src_pitch + 3 * x + c];
			}
		}
	}
}

#ifdef __ARM_NEON__
static inline uint8x8_t interpolate(uint16x8_t Dx, uint16x8_t Dy, uint8x8_t V1,
                                    uint8x8_t V2, uint8x8_t V3, uint8x8_t V4)
{
	uint16x4_t Dx_u16_l4 = vget_low_u16(Dx);
	uint16x4_t Dx_u16_h4 = vget_high_u16(Dx);
	uint32x4_t Dx_u32_l4 = vmovl_u16(Dx_u16_l4);
	uint32x4_t Dx_u32_h4 = vmovl_u16(Dx_u16_h4);
		
	uint16x4_t Dy_u16_l4 = vget_low_u16(Dy);
	uint16x4_t Dy_u16_h4 = vget_high_u16(Dy);
	uint32x4_t Dy_u32_l4 = vmovl_u16(Dy_u16_l4);
	uint32x4_t Dy_u32_h4 = vmovl_u16(Dy_u16_h4);
	
	uint32x4_t _Dx_u32_l4 = vsubq_u32(vdupq_n_u32(4096), Dx_u32_l4);
	uint32x4_t _Dx_u32_h4 = vsubq_u32(vdupq_n_u32(4096), Dx_u32_h4);
	uint32x4_t _Dy_u32_l4 = vsubq_u32(vdupq_n_u32(4096), Dy_u32_l4);
	uint32x4_t _Dy_u32_h4 = vsubq_u32(vdupq_n_u32(4096), Dy_u32_h4);
	
	uint16x8_t V1_u16 = vmovl_u8(V1);
	uint16x8_t V2_u16 = vmovl_u8(V2);
	uint16x8_t V3_u16 = vmovl_u8(V3);
	uint16x8_t V4_u16 = vmovl_u8(V4);
	
	uint16x4_t V1_u16_l4 = vget_low_u16(V1_u16);
	uint16x4_t V1_u16_h4 = vget_high_u16(V1_u16);
	uint32x4_t V1_u32_l4 = vmovl_u16(V1_u16_l4);
	uint32x4_t V1_u32_h4 = vmovl_u16(V1_u16_h4);
	
	uint16x4_t V2_u16_l4 = vget_low_u16(V2_u16);
	uint16x4_t V2_u16_h4 = vget_high_u16(V2_u16);
	uint32x4_t V2_u32_l4 = vmovl_u16(V2_u16_l4);
	uint32x4_t V2_u32_h4 = vmovl_u16(V2_u16_h4);
	
	uint16x4_t V3_u16_l4 = vget_low_u16(V3_u16);
	uint16x4_t V3_u16_h4 = vget_high_u16(V3_u16);
	uint32x4_t V3_u32_l4 = vmovl_u16(V3_u16_l4);
	uint32x4_t V3_u32_h4 = vmovl_u16(V3_u16_h4);
	
	uint16x4_t V4_u16_l4 = vget_low_u16(V4_u16);
	uint16x4_t V4_u16_h4 = vget_high_u16(V4_u16);
	uint32x4_t V4_u32_l4 = vmovl_u16(V4_u16_l4);
	uint32x4_t V4_u32_h4 = vmovl_u16(V4_u16_h4);
	
	uint32x4_t I1_u32_l4 = vaddq_u32(vmulq_u32(Dx_u32_l4, V2_u32_l4), vmulq_u32(_Dx_u32_l4, V1_u32_l4));
	uint32x4_t I1_u32_h4 = vaddq_u32(vmulq_u32(Dx_u32_h4, V2_u32_h4), vmulq_u32(_Dx_u32_h4, V1_u32_h4));
	
	uint32x4_t I2_u32_l4 = vaddq_u32(vmulq_u32(Dx_u32_l4, V4_u32_l4), vmulq_u32(_Dx_u32_l4, V3_u32_l4));
	uint32x4_t I2_u32_h4 = vaddq_u32(vmulq_u32(Dx_u32_h4, V4_u32_h4), vmulq_u32(_Dx_u32_h4, V3_u32_h4));
	
	uint32x4_t II_u32_l4 = vaddq_u32(vmulq_u32(Dy_u32_l4, I2_u32_l4), vmulq_u32(_Dy_u32_l4, I1_u32_l4));
	uint32x4_t II_u32_h4 = vaddq_u32(vmulq_u32(Dy_u32_h4, I2_u32_h4), vmulq_u32(_Dy_u32_h4, I1_u32_h4));
	
	uint16x4_t II_u16_l4 = vshrn_n_u32(II_u32_l4, 16);
	uint16x4_t II_u16_h4 = vshrn_n_u32(II_u32_h4, 16);
	
	II_u16_l4 = vrshr_n_u16(II_u16_l4, 8);
	II_u16_h4 = vrshr_n_u16(II_u16_h4, 8);
	
	uint16x8_t II_u16 = vcombine_u16(II_u16_l4, II_u16_h4);
	
	return vqmovn_u16(II_u16);
}
#endif

#ifdef __ARM_NEON__
static inline uint8x8_t batch_read_pixel(unsigned char *ptr, int pitch, int16x8_t x, short y)
{
	uint8x8_t Pix = {
		ptr[y * pitch + x[0]],
		ptr[y * pitch + x[1]],
		ptr[y * pitch + x[2]],
		ptr[y * pitch + x[3]],
		ptr[y * pitch + x[4]],
		ptr[y * pitch + x[5]],
		ptr[y * pitch + x[6]],
		ptr[y * pitch + x[7]]
	};
	
	return Pix;
}
#endif

#ifdef __ARM_NEON__
void resize_image_neon(unsigned char *src, unsigned char *dst, int src_w, int src_h,
                       int dst_w, int dst_h, int nchannels)
{
	float s = (float)src_w / dst_w;
	float32x4_t delta_l4 = {0, 1, 2, 3};
	float32x4_t delta_h4 = {4, 5, 6, 7};
	int16x8_t minx = vdupq_n_s16(0);
	int16x8_t maxx = vdupq_n_s16(src_w - 2);
	for (int c = 0; c < nchannels; ++c) {
		unsigned char *src_at = src + c * src_w * src_h;
		unsigned char *dst_at = dst + c * dst_w * dst_h;
		#pragma omp parallel for
		for (int y = 0; y < dst_h; ++y) {
			float sy = s * (y + 0.5) - 0.5;
			short top = (short)sy;
			uint16x8_t Dy = vdupq_n_u16((unsigned short)((sy -top) * 4096));
			if (top < 0) top = 0;
			if (top > src_h - 2) top = src_h - 2;
			for (int x = 0; x < dst_w; x += 8) {
				float32x4_t X_f32_l4 = vaddq_f32(vdupq_n_f32(x + 0.5), delta_l4);
				float32x4_t X_f32_h4 = vaddq_f32(vdupq_n_f32(x + 0.5), delta_h4);
				
				X_f32_l4 = vsubq_f32(vmulq_n_f32(X_f32_l4, s), vdupq_n_f32(0.5));
				X_f32_h4 = vsubq_f32(vmulq_n_f32(X_f32_h4, s), vdupq_n_f32(0.5));
				
				int32x4_t X_s32_l4 = vcvtq_s32_f32(X_f32_l4);
				int32x4_t X_s32_h4 = vcvtq_s32_f32(X_f32_h4);
				
				int16x4_t X_s16_l4 = vmovn_s32(X_s32_l4);
				int16x4_t X_s16_h4 = vmovn_s32(X_s32_h4);
				
				float32x4_t Dx_f32_l4 = vsubq_f32(X_f32_l4, vcvtq_f32_s32(X_s32_l4));
				float32x4_t Dx_f32_h4 = vsubq_f32(X_f32_h4, vcvtq_f32_s32(X_s32_h4));
				
				Dx_f32_l4 = vmulq_n_f32(Dx_f32_l4, 4096);
				Dx_f32_h4 = vmulq_n_f32(Dx_f32_h4, 4096);
				
				uint32x4_t Dx_u32_l4 = vcvtq_u32_f32(Dx_f32_l4);
				uint32x4_t Dx_u32_h4 = vcvtq_u32_f32(Dx_f32_h4);
				
				uint16x4_t Dx_u16_l4 = vmovn_u32(Dx_u32_l4);
				uint16x4_t Dx_u16_h4 = vmovn_u32(Dx_u32_h4);

				uint16x8_t Dx = vcombine_u16(Dx_u16_l4, Dx_u16_h4);
				
				int16x8_t left = vcombine_s16(X_s16_l4, X_s16_h4);
				left = vminq_s16(vmaxq_s16(left, minx), maxx);
				
				uint8x8_t V1 = batch_read_pixel(src_at, src_w, left, top);
				uint8x8_t V2 = batch_read_pixel(src_at, src_w, vaddq_s16(left, vdupq_n_s16(1)), top);
				uint8x8_t V3 = batch_read_pixel(src_at, src_w, left, top + 1);
				uint8x8_t V4 = batch_read_pixel(src_at, src_w, vaddq_s16(left, vdupq_n_s16(1)), top + 1);

				uint8x8_t J8 = interpolate(Dx, Dy, V1, V2, V3, V4);
				
				vst1_u8(dst_at + y * dst_w + x, J8);
			}
		}
	}
}
#endif

#ifdef __ARM_NEON__
void make_bilinear_interp_table(int src_w, int src_h, int dst_w, int dst_h, short *x_tab,
                                short *y_tab, unsigned short *dx_tab, unsigned short *dy_tab)
{
	float s = (float)src_w / dst_w;
	float32x4_t delta_l4 = {0, 1, 2, 3};
	float32x4_t delta_h4 = {4, 5, 6, 7};
	int16x8_t minx = vdupq_n_s16(0);
	int16x8_t maxx = vdupq_n_s16(src_w - 2);
	
	#pragma omp parallel for
	for (int y = 0; y < dst_h; ++y) {
		float sy = s * (y + 0.5) - 0.5;
		short top = (short)sy;
		uint16x8_t Dy = vdupq_n_u16((unsigned short)((sy -top) * 4096));
		if (top < 0) top = 0;
		if (top > src_h - 2) top = src_h - 2;
		short *px = x_tab + y * dst_w;
		short *py = y_tab + y * dst_w;
		unsigned short *pdx = dx_tab + y * dst_w;
		unsigned short *pdy = dy_tab + y * dst_w;
		for (int x = 0; x < dst_w; x += 8) {
			float32x4_t X_f32_l4 = vaddq_f32(vdupq_n_f32(x + 0.5), delta_l4);
			float32x4_t X_f32_h4 = vaddq_f32(vdupq_n_f32(x + 0.5), delta_h4);
			
			X_f32_l4 = vsubq_f32(vmulq_n_f32(X_f32_l4, s), vdupq_n_f32(0.5));
			X_f32_h4 = vsubq_f32(vmulq_n_f32(X_f32_h4, s), vdupq_n_f32(0.5));
			
			int32x4_t X_s32_l4 = vcvtq_s32_f32(X_f32_l4);
			int32x4_t X_s32_h4 = vcvtq_s32_f32(X_f32_h4);
			
			int16x4_t X_s16_l4 = vmovn_s32(X_s32_l4);
			int16x4_t X_s16_h4 = vmovn_s32(X_s32_h4);
			
			int16x8_t left = vcombine_s16(X_s16_l4, X_s16_h4);
			left = vminq_s16(vmaxq_s16(left, minx), maxx);
			
			float32x4_t Dx_f32_l4 = vsubq_f32(X_f32_l4, vcvtq_f32_s32(X_s32_l4));
			float32x4_t Dx_f32_h4 = vsubq_f32(X_f32_h4, vcvtq_f32_s32(X_s32_h4));
			
			Dx_f32_l4 = vmulq_n_f32(Dx_f32_l4, 4096);
			Dx_f32_h4 = vmulq_n_f32(Dx_f32_h4, 4096);
			
			uint32x4_t Dx_u32_l4 = vcvtq_u32_f32(Dx_f32_l4);
			uint32x4_t Dx_u32_h4 = vcvtq_u32_f32(Dx_f32_h4);
			
			uint16x4_t Dx_u16_l4 = vmovn_u32(Dx_u32_l4);
			uint16x4_t Dx_u16_h4 = vmovn_u32(Dx_u32_h4);

			uint16x8_t Dx = vcombine_u16(Dx_u16_l4, Dx_u16_h4);
			
			vst1q_s16(px, left);
			vst1q_s16(py, vdupq_n_s16(top));
			vst1q_u16(pdx, Dx);
			vst1q_u16(pdy, Dy);
			
			px += 8;
			py += 8;
			pdx += 8;
			pdy += 8;
		}
	}
}

void package_neighbor_pixle(int src_w, int src_h, int dst_w, int dst_h, short *x_tab,
                            short *y_tab, unsigned char *src, unsigned char *pack)
{
	for (int c = 0; c < 3; ++c) {
		unsigned char *src_at = src + c * src_w * src_h;
		unsigned char *pack_at = pack + c * dst_w * dst_h * 4;
		#pragma omp parallel for
		for (int y = 0; y < dst_h; ++y) {
			for (int x = 0; x < dst_w; ++x) {
				int id = y * dst_w + x;
				pack_at[y * dst_w * 4 + 4 * x]     = src_at[y_tab[id] * src_w + x_tab[id]];
				pack_at[y * dst_w * 4 + 4 * x + 1] = src_at[y_tab[id] * src_w + x_tab[id] + 1];
				pack_at[y * dst_w * 4 + 4 * x + 2] = src_at[(y_tab[id] + 1) * src_w + x_tab[id]];
				pack_at[y * dst_w * 4 + 4 * x + 3] = src_at[(y_tab[id] + 1) * src_w + x_tab[id] + 1];
			}
		}
	}
}

void resize_image_neon_faster(unsigned char *pack, unsigned char *dst, int dst_w, int dst_h,
                              int nchannels, unsigned short *dx_tab, unsigned short *dy_tab)
{
	for (int c = 0; c < nchannels; ++c) {
		unsigned char *pack_at = pack + c * dst_w * dst_h * 4;
		unsigned char *dst_at = dst + c * dst_w * dst_h;
		#pragma omp parallel for num_threads(4)
		for (int y = 0; y < dst_h; ++y) {
			unsigned char *ppk = pack_at + y * dst_w * 4;
			unsigned char *pdst = dst_at + y * dst_w;
			unsigned short *pdx = dx_tab + y * dst_w;
			unsigned short *pdy = dy_tab + y * dst_w;
			for (int x = 0; x < dst_w; x += 8) {
				uint8x8x4_t V = vld4_u8(ppk);
				uint16x8_t Dx = vld1q_u16(pdx);
				uint16x8_t Dy = vld1q_u16(pdy);
				uint8x8_t IV = interpolate(Dx, Dy, V.val[0], V.val[1], V.val[2], V.val[3]);
				vst1_u8(pdst, IV);
				ppk += 32;
				pdst += 8;
				pdx += 8;
				pdy += 8;
			}
		}
	}
}
#endif

void resize_image_hv(unsigned char *src, unsigned char *dst, int src_w, int src_h,
                     int dst_w, int dst_h, int nchannels)
{
	float sh = (float)(src_w - 1) / (dst_w - 1);
	float sv = (float)(src_h - 1) / (dst_h - 1);
	unsigned char *resized = calloc(dst_w * src_h * nchannels, sizeof(unsigned char));
	if (!resized) {
		fprintf(stderr, "calloc[%s:%d].\n", __FILE__, __LINE__);
		return;
	}
	
	for (int c = 0; c < nchannels; ++c) {
		unsigned char *src_at = src + c * src_w * src_h;
		unsigned char *rsz_at = resized + c * dst_w * src_h;
		unsigned char *dst_at = dst + c * dst_w * dst_h;
		
		// 水平缩放
		#pragma omp parallel for
		for (int y = 0; y < src_h; ++y) {
			for (int x = 0; x < dst_w; ++x) {
				float sx = sh * (x + 0.5) - 0.5;
				int left = (int)sx;
				int dx = (int)((sx - left) * 4096);
				if (left < 0) left = 0;
				if (left > src_w - 2) left = src_w - 2;
				int val = dx * src_at[y * src_w + left + 1] + (4096 - dx) * src_at[y * src_w + left];
				rsz_at[y * dst_w + x] = val >> 12;
			}
		}
		
		// 垂直缩放
		#pragma omp parallel for
		for (int y = 0; y < dst_h; ++y) {
			float sy = sv * (y + 0.5) - 0.5;
			int top = (int)sy;
			int dy = (int)((sy - top) * 4096);
			if (top < 0) top = 0;
			if (top > src_h - 2) top = src_h - 2;
			for (int x = 0; x < dst_w; ++x) {
				int val = dy * rsz_at[(top + 1) * dst_w + x] + (4096 - dy) * rsz_at[top * dst_w + x];
				dst_at[y * dst_w + x] = val >> 12;
			}
		}
	}
	
	free(resized);
}

void resize_image(unsigned char *src, unsigned char *dst, int src_w, int src_h,
                  int dst_w, int dst_h, int nchannels)
{
	float s = (float)src_w / dst_w;
	for (int c = 0; c < nchannels; ++c) {
		unsigned char *src_at = src + c * src_w * src_h;
		unsigned char *dst_at = dst + c * dst_w * dst_h;
		#pragma omp parallel for
		for (int y = 0; y < dst_h; ++y) {
			float sy = s * (y + 0.5) - 0.5;
			int top = (int)sy;
			int dy = (int)((sy -top) * 4096);
			if (top < 0) top = 0;
			if (top > src_h - 2) top = src_h - 2;
			for (int x = 0; x < dst_w; ++x) {
				float sx = s * (x + 0.5) - 0.5;
				int left = (int)sx;
				int dx = (int)((sx - left) * 4096);
				if (left < 0) left = 0;
				if (left > src_w - 2) left = src_w - 2;
				int v1 = dx * src_at[top * src_w + left + 1] + (4096 - dx) * src_at[top * src_w + left];
				int v2 = dx * src_at[(top + 1) * src_w + left + 1] + (4096 - dx) * src_at[(top + 1) * src_w + left];
				dst_at[y * dst_w + x] = (dy * v2 + (4096 - dy) * v1) >> 24;
			}
		}
	}
}

#ifdef __ARM_NEON__
void embed_image_neon(unsigned char *src, image *dst, int src_w, int src_h)
{
	float norm = 1.0 / 255;
	int dx = (dst->w - src_w) / 2;
	int dy = (dst->h - src_h) / 2;
	int pixels_per_load = 16;
	int excess = src_w - src_w % pixels_per_load;
	
	for (int c = 0; c < dst->c; ++c) {
		unsigned char *src_at = src + c * src_w * src_h;
		float *dst_at = dst->data + c * dst->w * dst->h;
		#pragma omp parallel for
		for (int y = 0; y < src_h; ++y) {
			unsigned char *psrc = src_at + y * src_w;
			float *pdst = dst_at + (y + dy) * dst->w + dx;
			for (int x = 0; x < excess; x += pixels_per_load) {
				uint8x16_t V_u8 = vld1q_u8(psrc);
				
				uint8x8_t V_u8_l8 = vget_low_u8(V_u8);
				uint8x8_t V_u8_h8 = vget_high_u8(V_u8);
				
				uint16x8_t V_u16_l8 = vmovl_u8(V_u8_l8);
				uint16x8_t V_u16_h8 = vmovl_u8(V_u8_h8);
				
				uint16x4_t V_u16_ll4 = vget_low_u16(V_u16_l8);
				uint16x4_t V_u16_lh4 = vget_high_u16(V_u16_l8);
				
				uint16x4_t V_u16_hl4 = vget_low_u16(V_u16_h8);
				uint16x4_t V_u16_hh4 = vget_high_u16(V_u16_h8);
				
				uint32x4_t V_u32_ll4 = vmovl_u16(V_u16_ll4);
				uint32x4_t V_u32_lh4 = vmovl_u16(V_u16_lh4);
				
				uint32x4_t V_u32_hl4 = vmovl_u16(V_u16_hl4);
				uint32x4_t V_u32_hh4 = vmovl_u16(V_u16_hh4);
				
				float32x4_t V_f32_ll4 = vcvtq_f32_u32(V_u32_ll4);
				float32x4_t V_f32_lh4 = vcvtq_f32_u32(V_u32_lh4);
				
				float32x4_t V_f32_hl4 = vcvtq_f32_u32(V_u32_hl4);
				float32x4_t V_f32_hh4 = vcvtq_f32_u32(V_u32_hh4);
				
				V_f32_ll4 = vmulq_n_f32(V_f32_ll4, norm);
				V_f32_lh4 = vmulq_n_f32(V_f32_lh4, norm);
				
				V_f32_hl4 = vmulq_n_f32(V_f32_hl4, norm);
				V_f32_hh4 = vmulq_n_f32(V_f32_hh4, norm);
				
				vst1q_f32(pdst,      V_f32_ll4);
				vst1q_f32(pdst +  4, V_f32_lh4);
				vst1q_f32(pdst +  8, V_f32_hl4);
				vst1q_f32(pdst + 12, V_f32_hh4);
				
				psrc += pixels_per_load;
				pdst += pixels_per_load;
			}
		}
		
		if (excess == src_w) continue;
		
		for (int y = 0; y < src_h; ++y) {
			for (int x = excess; x < src_w; ++x) {
				dst_at[(y + dy) * dst->w + x + dx] = src_at[y * src_w + x] * norm;
			}
		}
	}
}
#endif

void embed_image(unsigned char *src, image *dst, int src_w, int src_h)
{
	int dx = (dst->w - src_w) / 2;
	int dy = (dst->h - src_h) / 2;
	for (int c = 0; c < dst->c; ++c) {
		unsigned char *src_at = src + c * src_w * src_h;
		float *dst_at = dst->data + c * dst->w * dst->h;
		#pragma omp parallel for
		for (int y = 0; y < src_h; ++y) {
			for (int x = 0; x < src_w; ++x) {
				dst_at[(y + dy) * dst->w + x + dx] = src_at[y * src_w + x] / 255.0f;
			}
		}
	}
}

void set_image(image *img, float val)
{
	size_t size = img->w * img->h * img->c * sizeof(float);
	mset((char *const)img->data, size, (const char *const)&val, sizeof(float));
}

void vertical_mirror(image *img)
{
	int hh = img->h >> 1;
	for (int c = 0; c < img->c; ++c) {
		float *at = img->data + c * img->w * img->h;
		for (int y = 0; y < hh; ++y) {
			for (int x = 0; x < img->w; ++x) {
				float swap = at[y * img->w + x];
				at[y * img->w + x] = at[(img->h - 1 - y) * img->w + x];
				at[(img->h - 1 - y) * img->w + x] = swap;
			}
		}
	}
}

void swap_channel(image *img)
{
	int offset = img->w * img->h * 2;
	for (int y = 0; y < img->h; ++y) {
		for (int x = 0; x < img->w; ++x) {
			float swap = img->data[y * img->w + x];
			img->data[y * img->w + x] = img->data[y * img->w + x + offset];
			img->data[y * img->w + x + offset] = swap;
		}
	}
}

void get_standardizer_parameter(int width, int height, int standard_width, int standard_height,
	int *resized_width, int *resized_height, float *scale)
{
	if (standard_width / (float)width < standard_height / (float)height) {
		*resized_width = standard_width;
		*resized_height = (int)(height * standard_width / (float)width);
		*scale = width / (float)standard_width;
	} else {
		*resized_width = (int)(width * standard_height / (float)height);
		*resized_height = standard_height;
		*scale = height / (float)standard_height;
	}
}

void set_fill_color(float (*color)[4], int nchannels)
{
#ifndef CHANNEL_BLOCK_SIZE8
	for (int i = 0; i < 3; ++i) (*color)[i] = 0.5f;
	for (int i = 3; i < nchannels; ++i) (*color)[i] = 0;
#else
	const int half_channels = nchannels << 1;
	cl_half *ptr = (cl_half *)(*color);
	for (int i = 0; i < 3; ++i) ptr[i] = to_half(0.5);
	for (int i = 3; i < half_channels; ++i) ptr[i] = 0;
#endif
}

image_standardizer *create_image_standardizer(int width, int height, int standard_width, int standard_height, int nchannels)
{
	image_standardizer *standardizer = calloc(1, sizeof(image_standardizer));
	if (!standardizer) {
		fprintf(stderr, "calloc fail[%s:%d].\n", __FILE__, __LINE__);
		return standardizer;
	}
	
	standardizer->width = width;
	standardizer->height = height;
	standardizer->standard_width = standard_width;
	standardizer->standard_height = standard_height;

	get_standardizer_parameter(width, height, standard_width, standard_height, &standardizer->resized_width,
		&standardizer->resized_height, &standardizer->scale);

#if defined(OPENCL) && defined(WINOGRAD_CONVOLUTION)
	cl_mem_flags mem_flags = CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR;
	cl_image_format image_format = {
		.image_channel_order = CL_RGBA,
		.image_channel_data_type = CL_UNORM_INT8
	};
	
	cl_image_desc image_desc;
	memset(&image_desc, 0, sizeof(cl_image_desc));
	image_desc.image_type = CL_MEM_OBJECT_IMAGE2D;
	image_desc.image_width = standardizer->width;
	image_desc.image_height = standardizer->height;
	
	cl_int errcode;
	standardizer->d_input = clCreateImage(wrapper.context, mem_flags, &image_format, &image_desc, NULL, &errcode);
	if (CL_SUCCESS != errcode) {
		fprintf(stderr, "clCreateImage fail[%s:%d:%d].\n", __FILE__, __LINE__, errcode);
		goto cleanup;
	}

	size_t header_size = (size_t)(&BINARY_FILENAME_TO_END(cl_common, h) - &BINARY_FILENAME_TO_START(cl_common, h));
	size_t size = (size_t)(&BINARY_FILENAME_TO_END(utils, cl) - &BINARY_FILENAME_TO_START(utils, cl));
	standardizer->program_buffer = calloc(header_size + size + 1, sizeof(char));
	if (!standardizer->program_buffer) {
		fprintf(stderr, "calloc fail[%s:%d].\n", __FILE__, __LINE__);
		goto cleanup;
	}
	
	memcpy(standardizer->program_buffer, &BINARY_FILENAME_TO_START(cl_common, h), header_size);
	memcpy(standardizer->program_buffer + header_size, &BINARY_FILENAME_TO_START(utils, cl), size);
	standardizer->program_buffer[header_size + size] = '\0';
	
	char options[256] = "-I.";
	PARSE_PRECISION;
	standardizer->program = cl_make_wrapper_program(wrapper, "utils.cl", standardizer->program_buffer, options, &errcode);
	if (CL_SUCCESS != errcode) {
		fprintf(stderr, "cl_make_wrapper_program[%s:%d:%d].\n", __FILE__, __LINE__, errcode);
		goto cleanup;
	}
	
	standardizer->kernel = cl_make_wrapper_kernel(wrapper, standardizer->program, "normalize_image", &errcode);
	if (CL_SUCCESS != errcode) {
		fprintf(stderr, "cl_make_wrapper_kernel[%s:%d:%d].\n", __FILE__, __LINE__, errcode);
		goto cleanup;
	}
#else
	standardizer->chw_image = calloc(width * height * nchannels, sizeof(unsigned char));
	if (!standardizer->chw_image) {
		fprintf(stderr, "calloc fail[%s:%d].\n", __FILE__, __LINE__);
		goto cleanup;
	}
	
	standardizer->resized_chw_image = calloc(standardizer->resized_width * standardizer->resized_height * nchannels,
		sizeof(float));
	if (!standardizer->resized_chw_image) {
		fprintf(stderr, "calloc fail[%s:%d].\n", __FILE__, __LINE__);
		goto cleanup;
	}
#endif	
	standardizer->output = create_image(standardizer->standard_width, standardizer->standard_height, nchannels);
	if (!standardizer->output) {
		cleanup:free_image_standardizer(standardizer);
		return 0;
	}
#if defined(OPENCL) && defined(WINOGRAD_CONVOLUTION)
	size_t origin[] = {0, 0, 0};
	size_t region[] = {standardizer->standard_width, standardizer->standard_height, 1};
	float fill_color[4];
	set_fill_color(&fill_color, 4);
	cl_mem standard_image = standardizer->output->d_data;
	errcode = clEnqueueFillImage(wrapper.command_queue, standard_image, fill_color, origin, region, 0, NULL, NULL);
	if (CL_SUCCESS != errcode) {
		fprintf(stderr, "clEnqueueFillImage fail[%s:%d:%d].\n", __FILE__, __LINE__, errcode);
	}
#else
	set_image(standardizer->output, 0.5);
#endif
	return standardizer;
}

void *get_standardizer_output_ptr(image_standardizer *standardizer)
{
#if defined(OPENCL) && defined(WINOGRAD_CONVOLUTION)
	return standardizer->output->d_data;
#else
	return standardizer->output->data;
#endif	
}

void standardize_image(image_standardizer *standardizer, const unsigned char *const rgb24,
	unsigned int width, unsigned int height, int roix, int roiy, int roiw, int roih, void *output)
{
#if defined(OPENCL) && defined(WINOGRAD_CONVOLUTION)
	cl_int errcode;
	size_t origin[] = {0, 0, 0};
	size_t region[] = {standardizer->width, standardizer->height, 1};
	size_t row_pitch, slice_pitch;
	unsigned char *h_input = clEnqueueMapImage(wrapper.command_queue, standardizer->d_input, CL_TRUE, CL_MAP_WRITE,
		origin, region, &row_pitch, &slice_pitch, 0, NULL, NULL, &errcode);

	for (int y = 0; y < standardizer->height; ++y) {
		for (int x = 0; x < standardizer->width; ++x) {
			h_input[(standardizer->height - 1 - y) * row_pitch + (x << 2)]     = rgb24[y * width * 3 + x * 3 + 2];
			h_input[(standardizer->height - 1 - y) * row_pitch + (x << 2) + 1] = rgb24[y * width * 3 + x * 3 + 1];
			h_input[(standardizer->height - 1 - y) * row_pitch + (x << 2) + 2] = rgb24[y * width * 3 + x * 3 + 0];
			h_input[(standardizer->height - 1 - y) * row_pitch + (x << 2) + 3] = 0;
		}
	}
	
	clEnqueueUnmapMemObject(wrapper.command_queue, standardizer->d_input, h_input, 0, NULL, NULL);

	get_standardizer_parameter(roiw, roih, standardizer->standard_width, standardizer->standard_height,
		&standardizer->resized_width, &standardizer->resized_height, &standardizer->scale);
	
	cl_mem standard_image = output;
	errcode  = clSetKernelArg(standardizer->kernel, 0, sizeof(cl_mem), &standardizer->d_input);
	errcode |= clSetKernelArg(standardizer->kernel, 1, sizeof(cl_mem), &standard_image);
	errcode |= clSetKernelArg(standardizer->kernel, 2, sizeof(int), &standardizer->resized_width);
	errcode |= clSetKernelArg(standardizer->kernel, 3, sizeof(int), &standardizer->resized_height);
	errcode |= clSetKernelArg(standardizer->kernel, 4, sizeof(float), &standardizer->scale);
	errcode |= clSetKernelArg(standardizer->kernel, 5, sizeof(int), &roix);
	errcode |= clSetKernelArg(standardizer->kernel, 6, sizeof(int), &roiy);
	if (CL_SUCCESS != errcode) {
		fprintf(stderr, "clSetKernelArg fail[%s:%d:%d].\n", __FILE__, __LINE__, errcode);
	}

	cl_event event;
	cl_uint work_dim = 2;
	size_t global_work_size[] = {standardizer->resized_width, standardizer->resized_height};
	clEnqueueNDRangeKernel(wrapper.command_queue, standardizer->kernel, work_dim, NULL, global_work_size,
		NULL, 0, NULL, &event);

#ifdef NDEBUG	
	cl_ulong start, end;
	clFinish(wrapper.command_queue);
	errcode  = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
	errcode |= clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
	float duration = (end - start) * 1e-6f;
	LOGD("GPU, normalize_image: %fms.\n", duration);
#endif
	clReleaseEvent(event);
#else
	split_channel((unsigned char *)rgb24, standardizer->chw_image, width * 3, standardizer->width, standardizer->height);
	resize_image(standardizer->chw_image, standardizer->resized_chw_image, standardizer->width,
		standardizer->height, standardizer->resized_width, standardizer->resized_height, 3);
	image img = {standardizer->standard_width, standardizer->standard_height, 3, output};
	embed_image(standardizer->resized_chw_image, &img, standardizer->resized_width, standardizer->resized_height);
#endif		
}

void standardize_ion_image(image_standardizer *standardizer, void *input, unsigned int width,
	unsigned int height, int roix, int roiy, int roiw, int roih, void *output)
{
#if defined(OPENCL) && defined(WINOGRAD_CONVOLUTION)
	get_standardizer_parameter(roiw, roih, standardizer->standard_width, standardizer->standard_height,
		&standardizer->resized_width, &standardizer->resized_height, &standardizer->scale);

	cl_int errcode;
	cl_mem d_input = input;
	cl_mem standard_image = output;
	errcode  = clSetKernelArg(standardizer->kernel, 0, sizeof(cl_mem), &d_input);
	errcode |= clSetKernelArg(standardizer->kernel, 1, sizeof(cl_mem), &standard_image);
	errcode |= clSetKernelArg(standardizer->kernel, 2, sizeof(int), &standardizer->resized_width);
	errcode |= clSetKernelArg(standardizer->kernel, 3, sizeof(int), &standardizer->resized_height);
	errcode |= clSetKernelArg(standardizer->kernel, 4, sizeof(float), &standardizer->scale);
	errcode |= clSetKernelArg(standardizer->kernel, 5, sizeof(int), &roix);
	errcode |= clSetKernelArg(standardizer->kernel, 6, sizeof(int), &roiy);
	if (CL_SUCCESS != errcode) {
		fprintf(stderr, "clSetKernelArg fail[%s:%d:%d].\n", __FILE__, __LINE__, errcode);
	}

	cl_event event;
	cl_uint work_dim = 2;
	size_t global_work_size[] = {standardizer->resized_width, standardizer->resized_height};
	clEnqueueNDRangeKernel(wrapper.command_queue, standardizer->kernel, work_dim, NULL, global_work_size,
		NULL, 0, NULL, &event);

#ifdef NDEBUG	
	cl_ulong start, end;
	clFinish(wrapper.command_queue);
	errcode  = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
	errcode |= clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
	float duration = (end - start) * 1e-6f;
	LOGD("GPU, normalize_image: %fms.\n", duration);
#endif
	clReleaseEvent(event);
#endif	
}

void free_image_standardizer(image_standardizer *standardizer)
{
	if (!standardizer) return;
#if defined(OPENCL) && defined(WINOGRAD_CONVOLUTION)
	free(standardizer->program_buffer);
	clReleaseMemObject(standardizer->d_input);
	clReleaseProgram(standardizer->program);
	clReleaseKernel(standardizer->kernel);
	free_image(standardizer->output);
#else
	if (standardizer->chw_image) free(standardizer->chw_image);
	if (standardizer->resized_chw_image) free(standardizer->resized_chw_image);
#endif
	free(standardizer);
}