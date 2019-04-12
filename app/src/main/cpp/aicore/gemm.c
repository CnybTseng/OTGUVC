#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <sys/time.h>
#ifdef __INTEL_SSE__
#	include <emmintrin.h>
#	include <tmmintrin.h>
#elif __ARM_NEON__
#	include <arm_neon.h>
#	include "neon_math.h"
#endif
#ifdef OPENCL
#	include "cl_wrapper.h"
#endif
#include "gemm.h"

struct gemm_context {
#ifdef OPENCL
	char *program_buffer;
	cl_program program;
	cl_kernel kernel;
	cl_mem d_A;
	cl_mem d_B;
	cl_mem d_C;
#endif
	int transa;
	int transb;
	int m;
	int n;
	int k;
	int round_up_m;
	int round_up_n;
	int round_up_k;
};

static void gemm_nn(gemm_context *context, int m, int n, int k, float alpha, float *A, int lda,
                    float *B, int ldb, float beta, float *C, int ldc);
static void gemm_tn(gemm_context *context, int m, int n, int k, float alpha, float *A, int lda,
                    float *B, int ldb, float beta, float *C, int ldc);
static void gemm_nt(gemm_context *context, int m, int n, int k, float alpha, float *A, int lda,
                    float *B, int ldb, float beta, float *C, int ldc);
static void gemm_tt(gemm_context *context, int m, int n, int k, float alpha, float *A, int lda,
                    float *B, int ldb, float beta, float *C, int ldc);

#ifdef __INTEL_SSE__
static void gemm_nn_sse(int m, int n, int k, float alpha, float *A, int lda,
                        float *B, int ldb, float *C, int ldc) __attribute__((used));
#elif __ARM_NEON__
static void gemm_nn_neon(int m, int n, int k, float alpha, float *A, int lda,
                         float *B, int ldb, float *C, int ldc) __attribute__((used));
#endif
#ifdef OPENCL
extern cl_wrapper wrapper;
extern char BINARY_FILENAME_TO_START(cl_common, h);
extern char BINARY_FILENAME_TO_END(cl_common, h);
extern char BINARY_FILENAME_TO_START(blas, cl);
extern char BINARY_FILENAME_TO_END(blas, cl);

static void gemm_nn_cl(gemm_context *context, int m, int n, int k, float alpha, float *A, int lda,
                float *B, int ldb, float beta, float *C, int ldc);
static void __attribute__((unused)) gemm_nn_cl_sm(gemm_context *context, int m, int n, int k, float alpha,
                   float *A, int lda, float *B, int ldb, float beta, float *C, int ldc);
static void __attribute__((unused)) gemm_nn_cl_tp(gemm_context *context, int m, int n, int k, float alpha,
                   float *A, int lda, float *B, int ldb, float beta, float *C, int ldc);
#ifdef __linux__
static void __attribute__((unused)) gemm_nn_cl_ion(gemm_context *context, int m, int n, int k, float alpha,
                    float *A, int lda, float *B, int ldb, float beta, float *C, int ldc);
#endif
static void gemm_nt_cl(gemm_context *context, int m, int n, int k, float alpha, float *A, int lda,				
                float *B, int ldb, float beta, float *C, int ldc);				
#endif

gemm_context *create_gemm_context(int transa, int transb, int m, int n, int k)
{
	gemm_context *context = calloc(1, sizeof(gemm_context));
	if (!context) {
		fprintf(stderr, "calloc fail[%s:%d].\n", __FILE__, __LINE__);
		return context;
	}
	
	context->transa = transa;
	context->transb = transb;
	context->m = m;
	context->n = n;
	context->k = k;
#if defined(OPENCL) && !defined(WINOGRAD_CONVOLUTION)
	size_t header_size = (size_t)(&BINARY_FILENAME_TO_END(cl_common, h) - &BINARY_FILENAME_TO_START(cl_common, h));
	size_t size = (size_t)(&BINARY_FILENAME_TO_END(blas, cl) - &BINARY_FILENAME_TO_START(blas, cl));
	context->program_buffer = calloc(header_size + size + 1, sizeof(char));
	if (!context->program_buffer) {
		fprintf(stderr, "calloc fail[%s:%d].\n", __FILE__, __LINE__);
		goto cleanup;
	}
	
	memcpy(context->program_buffer, &BINARY_FILENAME_TO_START(cl_common, h), header_size);
	memcpy(context->program_buffer + header_size, &BINARY_FILENAME_TO_START(blas, cl), size);
	context->program_buffer[header_size + size] = '\0';
	
	cl_int errcode;
	char options[] = "-cl-fast-relaxed-math -I.";
	context->program = cl_make_wrapper_program(wrapper, "blas.cl", context->program_buffer, options, &errcode);
	if (CL_SUCCESS != errcode) {
		fprintf(stderr, "cl_make_wrapper_program[%s:%d:%d].\n", __FILE__, __LINE__, errcode);
		goto cleanup;
	}
	
	context->kernel = cl_make_wrapper_kernel(wrapper, context->program, "sgemm_nn_8x8", &errcode);
	if (CL_SUCCESS != errcode) {
		fprintf(stderr, "cl_make_wrapper_kernel[%s:%d:%d].\n", __FILE__, __LINE__, errcode);
		goto cleanup;
	}

	context->round_up_m = round_up_multiple_of_8(m);
	context->round_up_n = round_up_multiple_of_8(n);
	context->round_up_k = round_up_multiple_of_8(k);
	
	context->d_A = clCreateBuffer(wrapper.context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR,
		context->round_up_m * context->round_up_k * sizeof(float), NULL, &errcode);
	if (CL_SUCCESS != errcode) {
		fprintf(stderr, "clCreateBuffer fail[%s:%d:%d].\n", __FILE__, __LINE__, errcode);
		goto cleanup;
	}

	context->d_B = clCreateBuffer(wrapper.context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR,
		context->round_up_k * context->round_up_n * sizeof(float), NULL, &errcode);
	if (CL_SUCCESS != errcode) {
		fprintf(stderr, "clCreateBuffer fail[%s:%d:%d].\n", __FILE__, __LINE__, errcode);
		goto cleanup;
	}
	
	context->d_C = clCreateBuffer(wrapper.context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
		context->round_up_m * context->round_up_n * sizeof(float), NULL, &errcode);
	if (CL_SUCCESS != errcode) {
		fprintf(stderr, "clCreateBuffer fail[%s:%d:%d].\n", __FILE__, __LINE__, errcode);
		cleanup:free_gemm_context(context);
		return 0;
	}
#endif
	return context;
}					
					
/** @brief 通用的矩阵乘法,C=alhpa*A*B+beta*C.相乘前对A或B转置或共轭转置暂不支持.
 ** @param transa 转置(transa=1)或不转置(transa=0)矩阵A.
 ** @param transb 转置(transa=1)或不转置(transa=0)矩阵B.
 ** @param m 矩阵C的行数.
 ** @param n 矩阵C的列数.
 ** @param k 如果transa=0,k为矩阵A的列数.如果transa=1,k为矩阵A的行数.
 **          如果transb=0,k为矩阵B的行数.如果transb=1,k为矩阵B的列数.
 ** @param alpha 矩阵A和矩阵B乘积的标量乘子.
 ** @param A 矩阵A.
 ** @param lda 矩阵A或其转置的行步长.
 ** @param B 矩阵B.
 ** @param ldb 矩阵B或其转置的行步长.
 ** @param beta 矩阵C的标量乘子.
 ** @param C 矩阵C.
 ** @param ldc 矩阵C的行步长.
 **/
void gemm(gemm_context *context, int transa, int transb, int m, int n, int k, float alpha,
          float *A, int lda, float *B, int ldb, float beta, float *C, int ldc)
{
#if !defined OPENCL
	const int mn = m * n;
#ifdef __ARM_NEON__
	const int batches = 4;
	const int excess = mn - mn % batches;
	#pragma omp parallel for num_threads(4)
	for (int i = 0; i < excess; i += batches) {
		float32x4_t cs = vld1q_f32(C + i);
		cs = vmulq_n_f32(cs, beta);
		vst1q_f32(C + i, cs);
	}

	for (int i = excess; i < mn; ++i) {
		C[i] *= beta;
	}
#else
	#pragma omp parallel for
	for (int i = 0; i < mn; ++i) {
		C[i] *= beta;
	}
#endif
#endif	
	if (!transa && !transb) {
		gemm_nn(context, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
	} else if (transa && !transb) {
		gemm_tn(context, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
	} else if (!transa && transb) {
		gemm_nt(context, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
	} else {
		gemm_tt(context, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
	}
}

void free_gemm_context(gemm_context *context)
{
	if (!context) return;
#if defined(OPENCL) && !defined(WINOGRAD_CONVOLUTION)
	free(context->program_buffer);
	clReleaseMemObject(context->d_A);
	clReleaseMemObject(context->d_B);
	clReleaseMemObject(context->d_C);
	clReleaseProgram(context->program);
	clReleaseKernel(context->kernel);
#endif
	free(context);
	context = NULL;
}

void gemm_nn(gemm_context *context, int m, int n, int k, float alpha, float *A, int lda,
             float *B, int ldb, float beta, float *C, int ldc)
{
#if !defined OPENCL
#ifdef __ARM_NEON__
	gemm_nn_neon(m, n, k, alpha, A, lda, B, ldb, C, ldc);
#elif __INTEL_SSE__
	gemm_nn_sse(m, n, k, alpha, A, lda, B, ldb, C, ldc);
#endif
	#pragma omp parallel for
	for (int i = 0; i < m; ++i) {
		for (int j = 0; j < n; ++j) {
			float sum = 0;
			for (int l = 0; l < k; ++l) {
				sum += alpha * A[i * lda + l] * B[l * ldb + j];
			}
			C[i * ldc + j] += sum;
		}
	}
#else
	gemm_nn_cl(context, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
#endif
}
			 
void gemm_tn(gemm_context *context, int m, int n, int k, float alpha, float *A, int lda,
             float *B, int ldb, float beta, float *C, int ldc)
{
	fprintf(stderr, "Not implemented[%s:%d].\n", __FILE__, __LINE__);
}		 
			 
void gemm_nt(gemm_context *context, int m, int n, int k, float alpha, float *A, int lda,
             float *B, int ldb, float beta, float *C, int ldc)
{
#if !defined OPENCL
	#pragma omp parallel for
	for (int i = 0; i < m; ++i) {
		for (int j = 0; j < n; ++j) {
			float sum = 0;
			for (int l = 0; l < k; ++l) {
				sum += alpha * A[i * lda + l] * B[j * ldb + l];
			}
			C[i * ldc + j] += sum;
		}
	}
#else
	gemm_nt_cl(context, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
#endif
}		 
			 
void gemm_tt(gemm_context *context, int m, int n, int k, float alpha, float *A, int lda,
             float *B, int ldb, float beta, float *C, int ldc)
{
	fprintf(stderr, "Not implemented[%s:%d].\n", __FILE__, __LINE__);
}

#ifdef __INTEL_SSE__
void gemm_nn_sse(int m, int n, int k, float alpha, float *A, int lda,
                 float *B, int ldb, float *C, int ldc)
{
	
}

#elif __ARM_NEON__
void gemm_nn_neon(int m, int n, int k, float alpha, float *A, int lda,
                  float *B, int ldb, float *C, int ldc)
{
	
}
#endif	

#ifdef OPENCL
void gemm_nn_cl(gemm_context *context, int m, int n, int k, float alpha, float *A, int lda,
                float *B, int ldb, float beta, float *C, int ldc)
{	
	cl_int errcode;
	const int _m = context->round_up_m;
	const int _n = context->round_up_n;
	const int _k = context->round_up_k;
	const int _lda = _k;
	const int _ldb = _n;
	const int _ldc = _n;

	float *h_A = clEnqueueMapBuffer(wrapper.command_queue, context->d_A, CL_TRUE, CL_MAP_WRITE,
		0, _m * _k * sizeof(float), 0, NULL, NULL, &errcode);
	for (int y = 0; y < m; ++y) {
		for (int x = 0; x < k; ++x) {
			h_A[y * _k + x] = A[y * k + x];
		}
		for (int x = k; x < _k; ++x) {
			h_A[y * _k + x] = 0;
		}
	}
	for (int y = m; y < _m; ++y) {
		for (int x = 0; x < _k; ++x) {
			h_A[y * _k + x] = 0;
		}
	}
	clEnqueueUnmapMemObject(wrapper.command_queue, context->d_A, h_A, 0, NULL, NULL);
	
	float *h_B = clEnqueueMapBuffer(wrapper.command_queue, context->d_B, CL_TRUE, CL_MAP_WRITE,
		0, _k * _n * sizeof(float), 0, NULL, NULL, &errcode);
	for (int y = 0; y < k; ++y) {
		for (int x = 0; x < n; ++x) {
			h_B[y * _n + x] = B[y * n + x];
		}
		for (int x = n; x < _n; ++x) {
			h_B[y * _n + x] = 0;
		}
	}
	for (int y = k; y < _k; ++y) {
		for (int x = 0; x < _n; ++x) {
			h_B[y * _n + x] = 0;
		}
	}
	clEnqueueUnmapMemObject(wrapper.command_queue, context->d_B, h_B, 0, NULL, NULL);
	
	float *h_C = clEnqueueMapBuffer(wrapper.command_queue, context->d_C, CL_TRUE, CL_MAP_WRITE,
		0, _m * _n * sizeof(float), 0, NULL, NULL, &errcode);
	for (int y = 0; y < m; ++y) {
		for (int x = 0; x < n; ++x) {
			h_C[y * _n + x] = C[y * n + x];
		}
		for (int x = n; x < _n; ++x) {
			h_C[y * _n + x] = 0;
		}
	}
	for (int y = m; y < _m; ++y) {
		for (int x = 0; x < _n; ++x) {
			h_C[y * _n + x] = 0;
		}
	}
	clEnqueueUnmapMemObject(wrapper.command_queue, context->d_C, h_C, 0, NULL, NULL);
	
	errcode  = clSetKernelArg(context->kernel, 0, sizeof(int), &_m);
	errcode |= clSetKernelArg(context->kernel, 1, sizeof(int), &_n); 
	errcode |= clSetKernelArg(context->kernel, 2, sizeof(int), &_k); 
	errcode |= clSetKernelArg(context->kernel, 3, sizeof(float), &alpha);
	errcode |= clSetKernelArg(context->kernel, 4, sizeof(cl_mem), &context->d_A); 
	errcode |= clSetKernelArg(context->kernel, 5, sizeof(int), &_lda); 
	errcode |= clSetKernelArg(context->kernel, 6, sizeof(cl_mem), &context->d_B); 
	errcode |= clSetKernelArg(context->kernel, 7, sizeof(int), &_ldb);
	errcode |= clSetKernelArg(context->kernel, 8, sizeof(float), &beta);
	errcode |= clSetKernelArg(context->kernel, 9, sizeof(cl_mem), &context->d_C); 
	errcode |= clSetKernelArg(context->kernel, 10, sizeof(int), &_ldc); 
	
	if (CL_SUCCESS != errcode) {
		fprintf(stderr, "clSetKernelArg fail!\n");
		return;
	}
	
	cl_event event;
	cl_uint work_dim = 2;
	size_t global_work_size[] = {_n >> 3, _m >> 3};
	errcode = clEnqueueNDRangeKernel(wrapper.command_queue, context->kernel, work_dim, NULL, global_work_size,
		NULL, 0, NULL, &event);

#ifdef NDEBUG	
	cl_ulong start, end;
	clFinish(wrapper.command_queue);
	errcode  = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
	errcode |= clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
	LOGD("gemm_nn_cl(|%dx%d|*|%dx%d|=>|%dx%d|*|%dx%d|): %f ms.\n", m, k, k, n, _m, _k, _k, _n, (end - start) * 1e-6f);
#endif
	clReleaseEvent(event);

	h_C = clEnqueueMapBuffer(wrapper.command_queue, context->d_C, CL_TRUE, CL_MAP_READ,
		0, _m * _n * sizeof(float), 0, NULL, NULL, &errcode);
	for (int y = 0; y < m; ++y) {
		for (int x = 0; x < n; ++x) {
			C[y * n + x] = h_C[y * _n + x];
		}
	}
	clEnqueueUnmapMemObject(wrapper.command_queue, context->d_C, h_C, 0, NULL, NULL);
}

void gemm_nn_cl_sm(gemm_context *context, int m, int n, int k, float alpha, float *A, int lda,
                   float *B, int ldb, float beta, float *C, int ldc)
{	
	cl_int errcode;
	char options[] = "-cl-fast-relaxed-math";
	cl_program program = cl_make_wrapper_program(wrapper, "blas.cl", context->program_buffer, options, &errcode);
	if (CL_SUCCESS != errcode) {
		fprintf(stderr, "cl_make_wrapper_program[%s:%d:%d].\n", __FILE__, __LINE__, errcode);
		return;
	}
	
	cl_kernel kernel = cl_make_wrapper_kernel(wrapper, program, "sgemm_nn_sm", &errcode);
	if (CL_SUCCESS != errcode) {
		fprintf(stderr, "cl_make_wrapper_kernel[%s:%d:%d].\n", __FILE__, __LINE__, errcode);
		return;
	}
	
	cl_kernel common_kernel = cl_make_wrapper_kernel(wrapper, program, "sgemm_nn_common", &errcode);
	if (CL_SUCCESS != errcode) {
		fprintf(stderr, "cl_make_wrapper_kernel[%s:%d:%d].\n", __FILE__, __LINE__, errcode);
		return;
	}
	
	cl_mem d_A = clCreateBuffer(wrapper.context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR,
		m * k * sizeof(float), NULL, &errcode);

	cl_mem d_B = clCreateBuffer(wrapper.context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR,
		k * n * sizeof(float), NULL, &errcode);

	cl_mem d_C = clCreateBuffer(wrapper.context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
		m * n * sizeof(float), NULL, &errcode);

	float *h_A = clEnqueueMapBuffer(wrapper.command_queue, d_A, CL_TRUE, CL_MAP_WRITE,
		0, m * k * sizeof(float), 0, NULL, NULL, &errcode);
	memcpy(h_A, A, m * k * sizeof(float));
	clEnqueueUnmapMemObject(wrapper.command_queue, d_A, h_A, 0, NULL, NULL);
	
	float *h_B = clEnqueueMapBuffer(wrapper.command_queue, d_B, CL_TRUE, CL_MAP_WRITE,
		0, k * n * sizeof(float), 0, NULL, NULL, &errcode);
	memcpy(h_B, B, k * n * sizeof(float));
	clEnqueueUnmapMemObject(wrapper.command_queue, d_B, h_B, 0, NULL, NULL);
	
	float *h_C = clEnqueueMapBuffer(wrapper.command_queue, d_C, CL_TRUE, CL_MAP_WRITE,
		0, m * n * sizeof(float), 0, NULL, NULL, &errcode);
	memcpy(h_C, C, m * n * sizeof(float));
	clEnqueueUnmapMemObject(wrapper.command_queue, d_C, h_C, 0, NULL, NULL);
	
	errcode  = clSetKernelArg(kernel, 0, sizeof(int), &m);
	errcode |= clSetKernelArg(kernel, 1, sizeof(int), &n); 
	errcode |= clSetKernelArg(kernel, 2, sizeof(int), &k); 
	errcode |= clSetKernelArg(kernel, 3, sizeof(float), &alpha);
	errcode |= clSetKernelArg(kernel, 4, sizeof(cl_mem), &d_A); 
	errcode |= clSetKernelArg(kernel, 5, sizeof(int), &lda); 
	errcode |= clSetKernelArg(kernel, 6, sizeof(cl_mem), &d_B); 
	errcode |= clSetKernelArg(kernel, 7, sizeof(int), &ldb);
	errcode |= clSetKernelArg(kernel, 8, sizeof(float), &beta);
	errcode |= clSetKernelArg(kernel, 9, sizeof(cl_mem), &d_C); 
	errcode |= clSetKernelArg(kernel, 10, sizeof(int), &ldc); 
	
	if (CL_SUCCESS != errcode) {
		fprintf(stderr, "clSetKernelArg fail!\n");
		return;
	}
	
	errcode  = clSetKernelArg(common_kernel, 0, sizeof(int), &m);
	errcode |= clSetKernelArg(common_kernel, 1, sizeof(int), &n); 
	errcode |= clSetKernelArg(common_kernel, 2, sizeof(int), &k); 
	errcode |= clSetKernelArg(common_kernel, 3, sizeof(float), &alpha);
	errcode |= clSetKernelArg(common_kernel, 4, sizeof(cl_mem), &d_A); 
	errcode |= clSetKernelArg(common_kernel, 5, sizeof(int), &lda); 
	errcode |= clSetKernelArg(common_kernel, 6, sizeof(cl_mem), &d_B); 
	errcode |= clSetKernelArg(common_kernel, 7, sizeof(int), &ldb);
	errcode |= clSetKernelArg(common_kernel, 8, sizeof(float), &beta);
	errcode |= clSetKernelArg(common_kernel, 9, sizeof(cl_mem), &d_C); 
	errcode |= clSetKernelArg(common_kernel, 10, sizeof(int), &ldc); 
	
	if (CL_SUCCESS != errcode) {
		fprintf(stderr, "clSetKernelArg fail!\n");
		return;
	}
	
	const int tile_rows = 8;
	const int tile_cols = 8;
	const int _m = (m / tile_rows) * tile_rows;
	const int _n = (n / tile_cols) * tile_cols;
	
	cl_event event;
	cl_uint work_dim = 2;
	
	if (_m && _n) {
		size_t global_work_size[] = {_n, _m};
		size_t local_work_size[]  = {16, 16};
		errcode = clEnqueueNDRangeKernel(wrapper.command_queue, kernel, work_dim, NULL, global_work_size,
			local_work_size, 0, NULL, &event);
		
		if (n != _n) {
			size_t global_work_offset[] = {_n, 0};
			size_t global_work_size[] = {n - _n, _m};
			errcode = clEnqueueNDRangeKernel(wrapper.command_queue, common_kernel, work_dim, global_work_offset,
				global_work_size, NULL, 0, NULL, NULL);
		}
		
		if (m != _m) {
			size_t global_work_offset[] = {0, _m};
			size_t global_work_size[] = {n, m - _m};
			errcode = clEnqueueNDRangeKernel(wrapper.command_queue, common_kernel, work_dim, global_work_offset,
				global_work_size, NULL, 0, NULL, NULL);
		}
	} else {
		size_t global_work_size[] = {n, m};
		errcode = clEnqueueNDRangeKernel(wrapper.command_queue, common_kernel, work_dim, NULL,
			global_work_size, NULL, 0, NULL, &event);
	}

#ifdef NDEBUG	
	cl_ulong start, end;
	clFinish(wrapper.command_queue);
	errcode  = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
	errcode |= clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
	LOGD("gemm_nn_cl_sm: %f ms.\n", (end - start) * 1e-6f);
#endif
	clReleaseEvent(event);

	h_C = clEnqueueMapBuffer(wrapper.command_queue, d_C, CL_TRUE, CL_MAP_READ,
		0, m * n * sizeof(float), 0, NULL, NULL, &errcode);
	memcpy(C, h_C, m * n * sizeof(float));
	clEnqueueUnmapMemObject(wrapper.command_queue, d_C, h_C, 0, NULL, NULL);

	clReleaseMemObject(d_A);
	clReleaseMemObject(d_B);
	clReleaseMemObject(d_C);

	clReleaseProgram(program);
	clReleaseKernel(kernel);
	clReleaseKernel(common_kernel);
}

void gemm_nn_cl_tp(gemm_context *context, int m, int n, int k, float alpha, float *A, int lda,
                   float *B, int ldb, float beta, float *C, int ldc)
{
	cl_int errcode;
	char options[] = "-cl-fast-relaxed-math";
	cl_program program = cl_make_wrapper_program(wrapper, "blas.cl", context->program_buffer, options, &errcode);
	if (CL_SUCCESS != errcode) {
		fprintf(stderr, "cl_make_wrapper_program[%s:%d:%d].\n", __FILE__, __LINE__, errcode);
		return;
	}
	
	cl_kernel kernel = cl_make_wrapper_kernel(wrapper, program, "sgemm_nn_8x4_tp", &errcode);
	if (CL_SUCCESS != errcode) {
		fprintf(stderr, "cl_make_wrapper_kernel[%s:%d:%d].\n", __FILE__, __LINE__, errcode);
		return;
	}
	
	const int _m = ((m + 7) / 8) * 8;
	const int _n = ((n + 3) / 4) * 4;
	const int _k = ((k + 3) / 4) * 4;
	const int _lda = _k;
	const int _ldc = _n;

	cl_mem d_A = clCreateBuffer(wrapper.context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR,
		_m * _k * sizeof(float), NULL, &errcode);
	float *h_A = clEnqueueMapBuffer(wrapper.command_queue, d_A, CL_TRUE, CL_MAP_WRITE, 0,
		_m * _k * sizeof(float), 0, NULL, NULL, &errcode);

	for (int y = 0; y < m; ++y) {
		for (int x = 0; x < k; ++x) {
			h_A[y * _k + x] = A[y * k + x];
		}
		for (int x = k; x < _k; ++x) {
			h_A[y * _k + x] = 0;
		}
	}
	for (int y = m; y < _m; ++y) {
		for (int x = 0; x < _k; ++x) {
			h_A[y * _k + x] = 0;
		}
	}
	clEnqueueUnmapMemObject(wrapper.command_queue, d_A, h_A, 0, NULL, NULL);

	cl_mem d_C = clCreateBuffer(wrapper.context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
		_m * _n * sizeof(float), NULL, &errcode);
	float *h_C = clEnqueueMapBuffer(wrapper.command_queue, d_C, CL_TRUE, CL_MAP_WRITE, 0,
		_m * _n * sizeof(float), 0, NULL, NULL, &errcode);

	for (int y = 0; y < m; ++y) {
		for (int x = 0; x < n; ++x) {
			h_C[y * _n + x] = C[y * n + x];
		}
		for (int x = n; x < _n; ++x) {
			h_C[y * _n + x] = 0;
		}
	}
	for (int y = m; y < _m; ++y) {
		for (int x = 0; x < _n; ++x) {
			h_C[y * _n + x] = 0;
		}
	}
	clEnqueueUnmapMemObject(wrapper.command_queue, d_C, h_C, 0, NULL, NULL);
		
	cl_mem_flags flags = CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR;	
	cl_image_format image_format = {CL_RGBA, CL_FLOAT};
	cl_image_desc image_desc = {
		.image_type = CL_MEM_OBJECT_IMAGE2D,
		.image_width = _n >> 2,
		.image_height = _k,
		.image_row_pitch = 0};
	cl_mem d_B = clCreateImage(wrapper.context, flags, &image_format, &image_desc, NULL, &errcode);
	
	size_t origin[] = {0, 0, 0};
	size_t region[] = {_n >> 2, _k, 1};
	size_t image_row_pitch, image_slice_pitch;
	float *h_B = clEnqueueMapImage(wrapper.command_queue, d_B, CL_TRUE, CL_MAP_WRITE, origin,
		region, &image_row_pitch, &image_slice_pitch, 0, NULL, NULL, &errcode);

	image_row_pitch = image_row_pitch >> 2;
	for (int y = 0; y < k; ++y) {
		for (int x = 0; x < n; ++x) {
			h_B[y * image_row_pitch + x] = B[y * n + x];
		}
		for (int x = n; x < _n; ++x) {
			h_B[y * image_row_pitch + x] = 0;
		}
	}
	for (int y = k; y < _k; ++y) {
		for (int x = 0; x < _n; ++x) {
			h_B[y * image_row_pitch + x] = 0;
		}
	}

	clEnqueueUnmapMemObject(wrapper.command_queue, d_B, h_B, 0, NULL, NULL);
	
	errcode  = clSetKernelArg(kernel, 0, sizeof(int), &_m);
	errcode |= clSetKernelArg(kernel, 1, sizeof(int), &_n); 
	errcode |= clSetKernelArg(kernel, 2, sizeof(int), &_k); 
	errcode |= clSetKernelArg(kernel, 3, sizeof(float), &alpha);
	errcode |= clSetKernelArg(kernel, 4, sizeof(cl_mem), &d_A); 
	errcode |= clSetKernelArg(kernel, 5, sizeof(int), &_lda); 
	errcode |= clSetKernelArg(kernel, 6, sizeof(cl_mem), &d_B); 
	errcode |= clSetKernelArg(kernel, 7, sizeof(float), &beta);
	errcode |= clSetKernelArg(kernel, 8, sizeof(cl_mem), &d_C); 
	errcode |= clSetKernelArg(kernel, 9, sizeof(int), &_ldc); 
	if (CL_SUCCESS != errcode) {
		fprintf(stderr, "clSetKernelArg fail!\n");
		return;
	}
	
	cl_event event;
	cl_uint work_dim = 2;
	size_t global_work_size[] = {_n >> 2, _m >> 3};
	size_t local_work_size[] = {128, 8};
	clEnqueueNDRangeKernel(wrapper.command_queue, kernel, work_dim, NULL, global_work_size,
		local_work_size, 0, NULL, &event);

#ifdef NDEBUG	
	cl_ulong start, end;
	clFinish(wrapper.command_queue);
	errcode  = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
	errcode |= clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
	LOGD("gemm_nn_cl_tp(|%dx%d|*|%dx%d|=>|%dx%d|*|%dx%d|): %f ms.\n", m, k, k, n, _m, _k, _k, _n, (end - start) * 1e-6f);
#endif
	clReleaseEvent(event);
	
	h_C = clEnqueueMapBuffer(wrapper.command_queue, d_C, CL_TRUE, CL_MAP_WRITE, 0,
		_m * _n * sizeof(float), 0, NULL, NULL, &errcode);	

	for (int y = 0; y < m; ++y) {
		for (int x = 0; x < n; ++x) {
			C[y * n + x] = h_C[y * _n + x];
		}
	}
	clEnqueueUnmapMemObject(wrapper.command_queue, d_C, h_C, 0, NULL, NULL);
		
	clReleaseMemObject(d_A);
	clReleaseMemObject(d_B);
	clReleaseMemObject(d_C);

	clReleaseProgram(program);
	clReleaseKernel(kernel);
}

#ifdef __linux__
void gemm_nn_cl_ion(gemm_context *context, int m, int n, int k, float alpha, float *A, int lda,
                    float *B, int ldb, float beta, float *C, int ldc)
{
	cl_int errcode;
	char options[] = "-cl-fast-relaxed-math";
	cl_program program = cl_make_wrapper_program(wrapper, "blas.cl", context->program_buffer, options, &errcode);
	if (CL_SUCCESS != errcode) {
		fprintf(stderr, "cl_make_wrapper_program[%s:%d:%d].\n", __FILE__, __LINE__, errcode);
		return;
	}
	
	cl_kernel kernel = cl_make_wrapper_kernel(wrapper, program, "matmul_8x4_blocks", &errcode);
	if (CL_SUCCESS != errcode) {
		fprintf(stderr, "cl_make_wrapper_kernel[%s:%d:%d].\n", __FILE__, __LINE__, errcode);
		return;
	}
	
	cl_image_format image_a_format = {
		.image_channel_order = CL_RGBA,
		.image_channel_data_type = CL_FLOAT
	};
	
	cl_image_desc image_a_desc;
	memset(&image_a_desc, 0, sizeof(cl_image_desc));
	image_a_desc.image_type = CL_MEM_OBJECT_IMAGE2D,
	image_a_desc.image_width = (k + 3) / 4,
	image_a_desc.image_height = ((m + 7) / 8) * 8,	
	image_a_desc.image_row_pitch = cl_get_ion_image_row_pitch(wrapper, image_a_format, image_a_desc);
	
	cl_ion_context ion_context_a = cl_make_ion_buffer_for_nonplanar_image(wrapper, image_a_desc);
	
	cl_mem_flags mem_flags = CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR | CL_MEM_EXT_HOST_PTR_QCOM;	
	cl_mem image_a = clCreateImage(wrapper.context, mem_flags, &image_a_format, &image_a_desc, &ion_context_a.ion_mem, &errcode);
		
	size_t image_a_origin[] = {0, 0, 0};
	size_t image_a_region[] = {image_a_desc.image_width, image_a_desc.image_height, 1};
	size_t image_a_row_pitch, image_a_slice_pitch;
	float *image_a_ptr = clEnqueueMapImage(wrapper.command_queue, image_a, CL_TRUE, CL_MAP_WRITE, image_a_origin,
		image_a_region, &image_a_row_pitch, &image_a_slice_pitch, 0, NULL, NULL, &errcode);

	image_a_row_pitch = image_a_row_pitch >> 2;
	for (int i = 0; i < m; ++i) {
		for (int j = 0; j < k; ++j) {
			*(image_a_ptr + i * image_a_row_pitch + j) = *(A + i * k + j);
		}
		for (int j = k; j < image_a_desc.image_width; ++j) {
			*(image_a_ptr + i * image_a_row_pitch + j) = 0;
		}
	}
	for (int i = m; i < image_a_desc.image_height; ++i) {
		for (int j = 0; j < image_a_desc.image_width; ++j) {
			*(image_a_ptr + i * image_a_row_pitch + j) = 0;
		}
	}
	
	clEnqueueUnmapMemObject(wrapper.command_queue, image_a, image_a_ptr, 0, NULL, NULL);
	
	cl_image_format image_b_format = {
		.image_channel_order = CL_RGBA,
		.image_channel_data_type = CL_FLOAT};
	
	cl_image_desc image_b_desc;
	memset(&image_b_desc, 0, sizeof(cl_image_desc));
	image_b_desc.image_type = CL_MEM_OBJECT_IMAGE2D,
	image_b_desc.image_width = (n + 3) / 4,
	image_b_desc.image_height = ((k + 7) / 8) * 8,	
	image_b_desc.image_row_pitch = cl_get_ion_image_row_pitch(wrapper, image_b_format, image_b_desc);
	
	cl_ion_context ion_context_b = cl_make_ion_buffer_for_nonplanar_image(wrapper, image_b_desc);
	
	mem_flags = CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR | CL_MEM_EXT_HOST_PTR_QCOM;	
	cl_mem image_b = clCreateImage(wrapper.context, mem_flags, &image_b_format, &image_b_desc, &ion_context_b.ion_mem, &errcode);
	
	size_t image_b_origin[] = {0, 0, 0};
	size_t image_b_region[] = {image_b_desc.image_width, image_b_desc.image_height, 1};
	size_t image_b_row_pitch, image_b_slice_pitch;
	float *image_b_ptr = clEnqueueMapImage(wrapper.command_queue, image_b, CL_TRUE, CL_MAP_WRITE, image_b_origin,
		image_b_region, &image_b_row_pitch, &image_b_slice_pitch, 0, NULL, NULL, &errcode);

	image_b_row_pitch = image_b_row_pitch >> 2;
	for (int i = 0; i < k; ++i) {
		for (int j = 0; j < n; ++j) {
			*(image_b_ptr + i * image_b_row_pitch + j) = *(B + i * n + j);
		}
		for (int j = n; j < image_b_desc.image_width; ++j) {
			*(image_b_ptr + i * image_b_row_pitch + j) = 0;
		}
	}
	for (int i = k; i < image_b_desc.image_height; ++i) {
		for (int j = 0; j < image_b_desc.image_width; ++j) {
			*(image_b_ptr + i * image_b_row_pitch + j) = 0;
		}
	}
	
	clEnqueueUnmapMemObject(wrapper.command_queue, image_b, image_b_ptr, 0, NULL, NULL);
	
	cl_image_format image_c_format = {
		.image_channel_order = CL_RGBA,
		.image_channel_data_type = CL_FLOAT};
	
	cl_image_desc image_c_desc;
	memset(&image_c_desc, 0, sizeof(cl_image_desc));
	image_c_desc.image_type = CL_MEM_OBJECT_IMAGE2D,
	image_c_desc.image_width = (n + 3) / 4,
	image_c_desc.image_height = ((m + 7) / 8) * 8,	
	image_c_desc.image_row_pitch = cl_get_ion_image_row_pitch(wrapper, image_c_format, image_c_desc);
	
	cl_ion_context ion_context_c = cl_make_ion_buffer_for_nonplanar_image(wrapper, image_c_desc);
	
	mem_flags = CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR | CL_MEM_EXT_HOST_PTR_QCOM;	
	cl_mem image_c = clCreateImage(wrapper.context, mem_flags, &image_c_format, &image_c_desc, &ion_context_c.ion_mem, &errcode);
	
	errcode  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &image_a); 
	errcode |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &image_b); 
	errcode |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &image_c); 
	errcode |= clSetKernelArg(kernel, 3, sizeof(cl_int), &k); 
	
	cl_event event;
	cl_uint work_dim = 2;
	size_t global_work_size[] = {image_b_desc.image_width, image_a_desc.image_height / 8};
	clEnqueueNDRangeKernel(wrapper.command_queue, kernel, work_dim, NULL, global_work_size,
		NULL, 0, NULL, &event);
	
#ifdef NDEBUG	
	cl_ulong start, end;
	clFinish(wrapper.command_queue);
	errcode  = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
	errcode |= clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
	LOGD("gemm_nn_cl_ion: %f ms.\n", (end - start) * 1e-6f);
#endif
	clReleaseEvent(event);
	
	size_t image_c_origin[] = {0, 0, 0};
	size_t image_c_region[] = {image_c_desc.image_width, image_c_desc.image_height, 1};
	size_t image_c_row_pitch, image_c_slice_pitch;
	float *image_c_ptr = clEnqueueMapImage(wrapper.command_queue, image_c, CL_TRUE, CL_MAP_READ, image_c_origin,
		image_c_region, &image_c_row_pitch, &image_c_slice_pitch, 0, NULL, NULL, &errcode);

	image_c_row_pitch = image_c_row_pitch >> 2;
	for (int i = 0; i < m; ++i) {
		for (int j = 0; j < n; ++j) {
			*(C + i * n + j) = *(image_c_ptr + i * image_c_row_pitch + j);
		}
	}
	
	clEnqueueUnmapMemObject(wrapper.command_queue, image_a, image_a_ptr, 0, NULL, NULL);
	
	cl_free_ion_context(wrapper, ion_context_a);
	cl_free_ion_context(wrapper, ion_context_b);
	cl_free_ion_context(wrapper, ion_context_c);
	clReleaseMemObject(image_a);
	clReleaseMemObject(image_b);
	clReleaseMemObject(image_c);
	clReleaseProgram(program);
	clReleaseKernel(kernel);
}
#endif

void gemm_nt_cl(gemm_context *context, int m, int n, int k, float alpha, float *A, int lda,				
                float *B, int ldb, float beta, float *C, int ldc)
{
	cl_int errcode;
	char options[] = "-cl-fast-relaxed-math";
	cl_program program = cl_make_wrapper_program(wrapper, "blas.cl", context->program_buffer, options, &errcode);
	if (CL_SUCCESS != errcode) {
		fprintf(stderr, "cl_make_wrapper_program[%s:%d:%d].\n", __FILE__, __LINE__, errcode);
		return;
	}
	
	cl_kernel kernel = cl_make_wrapper_kernel(wrapper, program, "sgemm_nt_8x4", &errcode);
	if (CL_SUCCESS != errcode) {
		fprintf(stderr, "cl_make_wrapper_kernel[%s:%d:%d].\n", __FILE__, __LINE__, errcode);
		return;
	}
	
	cl_kernel common_kernel = cl_make_wrapper_kernel(wrapper, program, "sgemm_nt_common", &errcode);
	if (CL_SUCCESS != errcode) {
		fprintf(stderr, "cl_make_wrapper_kernel[%s:%d:%d].\n", __FILE__, __LINE__, errcode);
		return;
	}
	
	cl_mem d_A = clCreateBuffer(wrapper.context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR,
		m * k * sizeof(float), NULL, &errcode);

	cl_mem d_B = clCreateBuffer(wrapper.context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR,
		k * n * sizeof(float), NULL, &errcode);

	cl_mem d_C = clCreateBuffer(wrapper.context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
		m * n * sizeof(float), NULL, &errcode);

	float *h_A = clEnqueueMapBuffer(wrapper.command_queue, d_A, CL_TRUE, CL_MAP_WRITE,
		0, m * k * sizeof(float), 0, NULL, NULL, &errcode);
	memcpy(h_A, A, m * k * sizeof(float));
	clEnqueueUnmapMemObject(wrapper.command_queue, d_A, h_A, 0, NULL, NULL);
	
	float *h_B = clEnqueueMapBuffer(wrapper.command_queue, d_B, CL_TRUE, CL_MAP_WRITE,
		0, k * n * sizeof(float), 0, NULL, NULL, &errcode);
	memcpy(h_B, B, k * n * sizeof(float));
	clEnqueueUnmapMemObject(wrapper.command_queue, d_B, h_B, 0, NULL, NULL);
	
	float *h_C = clEnqueueMapBuffer(wrapper.command_queue, d_C, CL_TRUE, CL_MAP_WRITE,
		0, m * n * sizeof(float), 0, NULL, NULL, &errcode);
	memcpy(h_C, C, m * n * sizeof(float));
	clEnqueueUnmapMemObject(wrapper.command_queue, d_C, h_C, 0, NULL, NULL);
	
	errcode  = clSetKernelArg(kernel, 0, sizeof(int), &m);
	errcode |= clSetKernelArg(kernel, 1, sizeof(int), &n); 
	errcode |= clSetKernelArg(kernel, 2, sizeof(int), &k); 
	errcode |= clSetKernelArg(kernel, 3, sizeof(float), &alpha);
	errcode |= clSetKernelArg(kernel, 4, sizeof(cl_mem), &d_A); 
	errcode |= clSetKernelArg(kernel, 5, sizeof(int), &lda); 
	errcode |= clSetKernelArg(kernel, 6, sizeof(cl_mem), &d_B); 
	errcode |= clSetKernelArg(kernel, 7, sizeof(int), &ldb);
	errcode |= clSetKernelArg(kernel, 8, sizeof(float), &beta);
	errcode |= clSetKernelArg(kernel, 9, sizeof(cl_mem), &d_C); 
	errcode |= clSetKernelArg(kernel, 10, sizeof(int), &ldc); 
	
	if (CL_SUCCESS != errcode) {
		fprintf(stderr, "clSetKernelArg fail!\n");
		return;
	}
	
	errcode  = clSetKernelArg(common_kernel, 0, sizeof(int), &m);
	errcode |= clSetKernelArg(common_kernel, 1, sizeof(int), &n); 
	errcode |= clSetKernelArg(common_kernel, 2, sizeof(int), &k); 
	errcode |= clSetKernelArg(common_kernel, 3, sizeof(float), &alpha);
	errcode |= clSetKernelArg(common_kernel, 4, sizeof(cl_mem), &d_A); 
	errcode |= clSetKernelArg(common_kernel, 5, sizeof(int), &lda); 
	errcode |= clSetKernelArg(common_kernel, 6, sizeof(cl_mem), &d_B); 
	errcode |= clSetKernelArg(common_kernel, 7, sizeof(int), &ldb);
	errcode |= clSetKernelArg(common_kernel, 8, sizeof(float), &beta);
	errcode |= clSetKernelArg(common_kernel, 9, sizeof(cl_mem), &d_C); 
	errcode |= clSetKernelArg(common_kernel, 10, sizeof(int), &ldc); 
	
	if (CL_SUCCESS != errcode) {
		fprintf(stderr, "clSetKernelArg fail!\n");
		return;
	}
	
	const int tile_rows = 8;
	const int tile_cols = 4;
	const int _m = (m / tile_rows) * tile_rows;
	const int _n = (n / tile_cols) * tile_cols;

	cl_event event;
	cl_uint work_dim = 2;
	size_t global_work_size[] = {_n >> 2, _m >> 3};
	errcode = clEnqueueNDRangeKernel(wrapper.command_queue, kernel, work_dim, NULL,
		global_work_size, NULL, 0, NULL, &event);
	
	if (n != _n) {
		size_t global_work_offset[] = {_n, 0};
		size_t global_work_size[] = {n - _n, _m};
		errcode = clEnqueueNDRangeKernel(wrapper.command_queue, common_kernel, work_dim, global_work_offset,
			global_work_size, NULL, 0, NULL, NULL);
	}
	
	if (m != _m) {
		size_t global_work_offset[] = {0, _m};
		size_t global_work_size[] = {n, m - _m};
		errcode = clEnqueueNDRangeKernel(wrapper.command_queue, common_kernel, work_dim, global_work_offset,
			global_work_size, NULL, 0, NULL, NULL);
	}

#ifdef NDEBUG	
	cl_ulong start, end;
	clFinish(wrapper.command_queue);
	errcode  = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
	errcode |= clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
	LOGD("gemm_nt_cl: %f ms.\n", (end - start) * 1e-6f);
#endif
	clReleaseEvent(event);
	
	h_C = clEnqueueMapBuffer(wrapper.command_queue, d_C, CL_TRUE, CL_MAP_READ,
		0, m * n * sizeof(float), 0, NULL, NULL, &errcode);
	memcpy(C, h_C, m * n * sizeof(float));
	clEnqueueUnmapMemObject(wrapper.command_queue, d_C, h_C, 0, NULL, NULL);

	clReleaseMemObject(d_A);
	clReleaseMemObject(d_B);
	clReleaseMemObject(d_C);

	clReleaseProgram(program);
	clReleaseKernel(kernel);
	clReleaseKernel(common_kernel);
}
#endif	 