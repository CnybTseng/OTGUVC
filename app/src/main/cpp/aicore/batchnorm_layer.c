#include <math.h>
#include <omp.h>
#ifdef __INTEL_SSE__
#	include <emmintrin.h>
#	include <tmmintrin.h>
#elif __ARM_NEON__
#	include <arm_neon.h>
#	include "neon_math.h"
#endif
#include "batchnorm_layer.h"
#include "convolutional_layer.h"

#ifdef __INTEL_SSE__
static void normalize_sse(float *X, float *mean, float *variance, int batch_size, int nchannels, int size);
#elif __ARM_NEON__
static void normalize_neon(float *X, float *mean, float *variance, int batch_size, int nchannels, int size);
#endif

void normalize(float *X, float *mean, float *variance, int batch_size, int nchannels, int size)
{
#ifdef __INTEL_SSE__
	return normalize_sse(X, mean, variance, batch_size, nchannels, size);
#elif __ARM_NEON__
	return normalize_neon(X, mean, variance, batch_size, nchannels, size);
#endif
	for (int i = 0; i < batch_size; ++i) {
		for (int j = 0; j < nchannels; ++j) {
			float *at = X + (i * nchannels + j) * size;
			for (int k = 0; k < size; ++k) {
				at[k] = (at[k] - mean[j]) / (sqrt(variance[j]) + 1e-6);
			}
		}
	}
}

void forward_batchnorm_layer(void *layer, znet *net)
{
	LAYER_TYPE type = *(LAYER_TYPE *)layer;
	if (type == CONVOLUTIONAL) {
		convolutional_layer *l = (convolutional_layer *)layer;
		int size = l->output_size.w * l->output_size.h;
		if (znet_workmode(net) == INFERENCE) {
			normalize(l->output, l->rolling_mean, l->rolling_variance, l->batch_size, l->nfilters, size);
		} else {
			fprintf(stderr, "Not implemented[%s:%d].\n", __FILE__, __LINE__);
			return;
		}
		
		mul_bias(l->output, l->scales, l->batch_size, l->nfilters, size);
		add_bias(l->output, l->biases, l->batch_size, l->nfilters, size);
	} else {
		fprintf(stderr, "Not implemented[%s:%d].\n", __FILE__, __LINE__);
	}
}

void backward_batchnorm_layer(void *layer, znet *net)
{
	fprintf(stderr, "Not implemented[%s:%d].\n", __FILE__, __LINE__);
}

#ifdef __INTEL_SSE__
void normalize_sse(float *X, float *mean, float *variance, int batch_size, int nchannels, int size)
{
	for (int i = 0; i < batch_size; ++i) {
		#pragma omp parallel for num_threads(8)
		for (int j = 0; j < nchannels; ++j) {
			float *at = X + (i * nchannels + j) * size;
			int batches = 4;
			int excess = size - size % batches;
			__m128 scales = _mm_set_ps1(1 / (sqrt(variance[j]) + 1e-6));
			__m128 biases = _mm_set_ps1(-mean[j] / (sqrt(variance[j]) + 1e-6));
			for (int k = 0; k < excess; k += batches) {
				__m128 xs = _mm_loadu_ps(at + k);
				xs = _mm_add_ps(_mm_mul_ps(xs, scales), biases);
				_mm_storeu_ps(at + k, xs);
			}
			for (int k = excess; k < size; ++k) {
				at[k] = (at[k] - mean[j]) / (sqrt(variance[j]) + 1e-6);
			}
		}
	}
}

#elif __ARM_NEON__
void normalize_neon(float *X, float *mean, float *variance, int batch_size, int nchannels, int size)
{
	for (int i = 0; i < batch_size; ++i) {
		#pragma omp parallel for num_threads(4)
		for (int j = 0; j < nchannels; ++j) {
			float *at = X + (i * nchannels + j) * size;
			int batches = 4;
			int excess = size - size % batches;
			float scale = 1 / (sqrt(variance[j]) + 1e-6);
			float32x4_t biases = vdupq_n_f32(-mean[j] / (sqrt(variance[j]) + 1e-6));
			for (int k = 0; k < excess; k += batches) {
				float32x4_t xs = vld1q_f32(at + k);
				xs = vaddq_f32(vmulq_n_f32(xs, scale), biases);
				vst1q_f32(at + k, xs);
			}
			for (int k = excess; k < size; ++k) {
				at[k] = (at[k] - mean[j]) / (sqrt(variance[j]) + 1e-6);
			}
		}
	}
}
#endif