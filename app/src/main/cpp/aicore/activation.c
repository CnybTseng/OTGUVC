#include <math.h>
#include <omp.h>
#ifdef __INTEL_SSE__
#	include <emmintrin.h>
#	include <tmmintrin.h>
#	include "sse_math.h"
#elif __ARM_NEON__
#	include <arm_neon.h>
#	include "neon_math.h"
#endif
#include "activation.h"

static void relu_activate(float *X, int n);
static void leaky_activate(float *X, int n);
static void linear_activate(float *X, int n);
static void logistic_active(float *X, int n);
#ifdef __INTEL_SSE__
static void relu_activate_sse(float *X, int n);
static void leaky_activate_sse(float *X, int n);
static void logistic_active_sse(float *X, int n);
#elif __ARM_NEON__
static void relu_activate_neon(float *X, int n);
static void leaky_activate_neon(float *X, int n);
static void logistic_active_neon(float *X, int n);
#endif

void activate(float *X, int n, ACTIVATION activation)
{
	if (activation == RELU) {
		relu_activate(X, n);
	} else if (activation == LEAKY) {
		leaky_activate(X, n);
	} else if (activation == LINEAR){
		linear_activate(X, n);
	} else if (activation == LOGISTIC) {
		logistic_active(X, n);
	} else {
		fprintf(stderr, "Not implemented[%s:%d].\n", __FILE__, __LINE__);
	}
}

void relu_activate(float *X, int n)
{
#ifdef __INTEL_SSE__
	return relu_activate_sse(X, n);
#elif __ARM_NEON__	
	return relu_activate_neon(X, n);
#endif
	#pragma omp parallel for num_threads(8)
	for (int i = 0; i < n; ++i) {
		X[i] = (X[i] > 0) * X[i];
	}
}

void leaky_activate(float *X, int n)
{
#ifdef __INTEL_SSE__
	return leaky_activate_sse(X, n);
#elif __ARM_NEON__
	return leaky_activate_neon(X, n);
#endif
	#pragma omp parallel for num_threads(8)
	for (int i = 0; i < n; ++i) {
		X[i] = (X[i] > 0) ? X[i] : 0.1 * X[i];
	}
}

void linear_activate(float *X, int n)
{
	return;
}

void logistic_active(float *X, int n)
{
#ifdef __INTEL_SSE__
	return logistic_active_sse(X, n);
#elif __ARM_NEON__
	return logistic_active_neon(X, n);
#endif
	#pragma omp parallel for num_threads(8)
	for (int i = 0; i < n; ++i) {
		X[i] = 1 / (1 + exp(-X[i]));
	}
}

#ifdef __INTEL_SSE__
void relu_activate_sse(float *X, int n)
{
	int batches = 4;
	int excess = n - n % batches;
	__m128 zeros = _mm_set1_ps(0);
	#pragma omp parallel for num_threads(8)
	for (int i = 0; i < excess; i += batches) {
		__m128 xs = _mm_loadu_ps(X + i);
		__m128 mask = _mm_cmpgt_ps(xs, zeros);
		xs = _mm_and_ps(xs, mask);
		_mm_storeu_ps(X + i, xs);
	}
	
	for (int i = excess; i < n; ++i) {
		X[i] = (X[i] > 0) * X[i];
	}
}

void leaky_activate_sse(float *X, int n)
{
	int batches = 4;
	int excess = n - n % batches;
	__m128 zeros = _mm_set1_ps(0);
	__m128 zero_point_one = _mm_set1_ps(0.1);
	#pragma omp parallel for num_threads(8)
	for (int i = 0; i < excess; i += batches) {
		__m128 xs = _mm_loadu_ps(X + i);
		__m128 mask = _mm_cmpgt_ps(xs, zeros);
		__m128 ys = _mm_mul_ps(zero_point_one, xs);
		xs = _mm_and_ps(mask, xs);
		ys = _mm_andnot_ps(mask, ys); 
		__m128 zs = _mm_or_ps(xs, ys);
		_mm_storeu_ps(X + i, zs);
	}
	
	for (int i = excess; i < n; ++i) {
		X[i] = (X[i] > 0) ? X[i] : 0.1 * X[i];
	}
}

void logistic_active_sse(float *X, int n)
{
	int batches = 4;
	int excess = n - n % batches;
	__m128 zeros = _mm_set1_ps(0);
	__m128 ones  = _mm_set1_ps(1);
	#pragma omp parallel for num_threads(8)
	for (int i = 0; i < excess; i += batches) {
		__m128 xs = _mm_loadu_ps(X + i);
		__m128 ys = exp_ps(_mm_sub_ps(zeros, xs));
		__m128 zs = _mm_div_ps(ones, _mm_add_ps(ones, ys));
		_mm_storeu_ps(X + i, zs);
	}
	
	for (int i = excess; i < n; ++i) {
		X[i] = 1 / (1 + exp(-X[i]));
	}
}

#elif __ARM_NEON__
void relu_activate_neon(float *X, int n)
{
	int batches = 4;
	int excess = n - n % batches;
	float32x4_t zeros = vdupq_n_f32(0);
	#pragma omp parallel for num_threads(4)
	for (int i = 0; i < excess; i += batches) {
		float32x4_t xs = vld1q_f32(X + i);
		uint32x4_t mask = vcgtq_f32(xs, zeros);
		float32x4_t zs = vbslq_f32(mask, xs, zeros);
		vst1q_f32(X + i, zs);
	}
	
	for (int i = excess; i < n; ++i) {
		X[i] = (X[i] > 0) * X[i];
	}
}

void leaky_activate_neon(float *X, int n)
{
	int batches = 4;
	int excess = n - n % batches;
	float32x4_t zeros = vdupq_n_f32(0);
	#pragma omp parallel for num_threads(4)
	for (int i = 0; i < excess; i += batches) {
		float32x4_t xs = vld1q_f32(X + i);
		uint32x4_t mask = vcgtq_f32(xs, zeros);
		float32x4_t ys = vmulq_n_f32(xs, 0.1);
		float32x4_t zs = vbslq_f32(mask, xs, ys);
		vst1q_f32(X + i, zs);
	}
	
	for (int i = excess; i < n; ++i) {
		X[i] = (X[i] > 0) ? X[i] : 0.1 * X[i];
	}
}

void logistic_active_neon(float *X, int n)
{
	int batches = 4;
	int excess = n - n % batches;
	float32x4_t ones = vdupq_n_f32(1);
	#pragma omp parallel for num_threads(4)
	for (int i = 0; i < excess; i += batches) {
		float32x4_t xs = vld1q_f32(X + i);
		float32x4_t ys = vaddq_f32(ones, exp_ps(vnegq_f32(xs)));
		float32x4_t zs = vrecpeq_f32(ys);
		vst1q_f32(X + i, zs);
	}
	
	for (int i = excess; i < n; ++i) {
		X[i] = 1 / (1 + exp(-X[i]));
	}
}
#endif