#include <omp.h>
#include <math.h>
#include <sys/time.h>
#include <string.h>
#ifdef __INTEL_SSE__
#	include <emmintrin.h>
#	include <tmmintrin.h>
#elif __ARM_NEON__
#	include <arm_neon.h>
#	include "neon_math.h"
#endif
#include "convolutional_layer.h"
#include "batchnorm_layer.h"
#include "activation.h"
#include "im2col.h"
#include "gemm.h"
#include "half.h"

#ifdef MERGE_BATCHNORM_TO_CONV
static void merge_batchnorm_params(convolutional_layer *layer);
#endif

#ifdef NNPACK
static void forward_convolutional_layer_nnp(void *_layer, znet *net);
#endif

#if defined(OPENCL) && defined(WINOGRAD_CONVOLUTION)
static void forward_convolutional_layer_gpu(void *_layer, znet *net);
#endif

#ifdef OPENCL
extern cl_wrapper wrapper;
extern char BINARY_FILENAME_TO_START(cl_common, h);
extern char BINARY_FILENAME_TO_END(cl_common, h);
extern char BINARY_FILENAME_TO_START(convolution, cl);
extern char BINARY_FILENAME_TO_END(convolution, cl);

struct direct_convolution_context {
	char *program_buffer;
	cl_program program;
	cl_kernel kernel;
	cl_mem d_input;
	cl_mem d_weights;
	cl_mem d_biases;
	cl_mem d_output;
	int nfilters;
	int input_channel_blocks;
	int output_channel_blocks;
	int weight_image_width;
	int weight_image_height;
	int bias_image_width;
	int input_image_width;
	int input_image_height;
	int output_image_width;
	int output_image_height;
};

static void load_direct_convolution_weight(direct_convolution_context *context, float *weights, float *biases);
static direct_convolution_context *create_direct_convolution_context(convolutional_layer *layer);
static void forward_convolutional_layer_1x1(convolutional_layer *layer);
static void free_direct_convolution_context(direct_convolution_context *context);
#endif

#ifdef __INTEL_SSE__
static void add_bias_sse(float *output, float *biases, int batch_size, int nchannels, int size);
static void mul_bias_sse(float *output, float *scales, int batch_size, int nchannels, int size);
#elif __ARM_NEON__
static void add_bias_neon(float *output, float *biases, int batch_size, int nchannels, int size);
static void mul_bias_neon(float *output, float *scales, int batch_size, int nchannels, int size);
#endif

void *make_convolutional_layer(ACTIVATION activation, dim3 input_size, int filter_size, int nfilters,
                               int stride, int padding, int batch_size, int batch_norm, dim3 *output_size)
{
	convolutional_layer *layer = calloc(1, sizeof(convolutional_layer));
	if (!layer) {
		fprintf(stderr, "calloc[%s:%d].\n", __FILE__, __LINE__);
		return layer;
	}

	layer->type = CONVOLUTIONAL;
	layer->activation = activation;
	layer->input_size = input_size;
	layer->filter_size = filter_size;
	layer->nfilters = nfilters;
	layer->stride = stride;
	layer->padding = padding;
	layer->output_size.w = convolutional_output_width(layer);
	layer->output_size.h = convolutional_output_height(layer);
	layer->output_size.c = nfilters;
	layer->batch_size = batch_size;
	layer->batch_norm = batch_norm;
	layer->nweights = filter_size * filter_size * input_size.c * nfilters;
	layer->nbiases = nfilters;
	layer->ninputs = input_size.w * input_size.h * input_size.c;
	layer->vmsize = input_size.c * filter_size * filter_size * layer->output_size.w * layer->output_size.h;
	layer->noutputs = layer->output_size.w * layer->output_size.h * nfilters;
	layer->weights = NULL;
	layer->scales = NULL;
	layer->biases = NULL;
	layer->rolling_mean = NULL;
	layer->rolling_variance = NULL;
	layer->input = NULL;
	layer->vecmat = NULL;
	layer->output = NULL;
#ifdef NNPACK
	layer->algorithm = nnp_convolution_algorithm_wt8x8_fp16;
	layer->transformed_kernel_size = 0;
	layer->transformed_kernel = NULL;
#endif
#if defined(OPENCL)
#ifdef WINOGRAD_CONVOLUTION
	layer->wtc = NULL;
	layer->itc = NULL;
	layer->mmc = NULL;
	layer->oitc = NULL;
#endif
	layer->dcc = NULL;
#endif
	layer->gc = NULL;
	
	if (output_size) {
		*output_size = layer->output_size;
	}

	layer->weights = calloc(layer->nweights, sizeof(float));
	if (!layer->weights) {
		fprintf(stderr, "calloc[%s:%d].\n", __FILE__, __LINE__);
		goto cleanup;
	}

#if defined(OPENCL)
#ifdef 	WINOGRAD_CONVOLUTION
	if (3 == filter_size) {
		layer->wtc = create_weight_transform_context(F4x4_3x3, input_size.c, nfilters);
		if (!layer->wtc) {
			goto cleanup;
		}
		
		layer->itc = create_input_transform_context(F4x4_3x3, input_size.w, input_size.h, input_size.c, stride, padding);
		if (!layer->itc) {
			goto cleanup;
		}
		
		layer->mmc =create_matrix_multiplication_context(layer->wtc, layer->itc);
		if (!layer->mmc) {
			goto cleanup;
		}
		
		layer->oitc = create_output_inverse_transform_context(layer->mmc, activation);
		if (!layer->oitc) {
			goto cleanup;
		}
	}
#endif
	if (1 == filter_size) {
		layer->dcc = create_direct_convolution_context(layer);
		if (!layer->dcc) {
			goto cleanup;
		}
	}
#endif

	const int m = layer->nfilters;
	const int n = layer->output_size.w * layer->output_size.h;
	const int k = layer->filter_size * layer->filter_size * layer->input_size.c;
	layer->gc = create_gemm_context(0, 0, m, n, k);

	layer->scales = calloc(nfilters, sizeof(float));
	if (!layer->scales) {
		fprintf(stderr, "calloc[%s:%d].\n", __FILE__, __LINE__);
		goto cleanup;
	}
	
	layer->biases = calloc(layer->nbiases, sizeof(float));
	if (!layer->biases) {
		fprintf(stderr, "calloc[%s:%d].\n", __FILE__, __LINE__);
		goto cleanup;
	}
	
	layer->rolling_mean = calloc(nfilters, sizeof(float));
	if (!layer->rolling_mean) {
		fprintf(stderr, "calloc[%s:%d].\n", __FILE__, __LINE__);
		goto cleanup;
	}
	
	layer->rolling_variance = calloc(nfilters, sizeof(float));
	if (!layer->rolling_variance) {
		fprintf(stderr, "calloc[%s:%d].\n", __FILE__, __LINE__);
		goto cleanup;
	}
	
	layer->vecmat = calloc(layer->vmsize, sizeof(float));
	if (!layer->vecmat) {
		fprintf(stderr, "calloc[%s:%d].\n", __FILE__, __LINE__);
		goto cleanup;
	}
	
	layer->output = calloc(layer->noutputs * batch_size, sizeof(float));
	if (!layer->output) {
		fprintf(stderr, "calloc[%s:%d].\n", __FILE__, __LINE__);
		cleanup:free_convolution_layer(layer);
		return 0;
	}

	return (void *)layer;
}

void free_convolution_layer(void *_layer)
{
	convolutional_layer *layer = (convolutional_layer *)_layer;
	if (!layer) return;
	
	if (layer->weights) {
		free(layer->weights);
		layer->weights = NULL;
	}
	
	if (layer->scales) {
		free(layer->scales);
		layer->scales = NULL;
	}
	
	if (layer->biases) {
		free(layer->biases);
		layer->biases = NULL;
	}
	
	if (layer->rolling_mean) {
		free(layer->rolling_mean);
		layer->rolling_mean = NULL;
	}
	
	if (layer->rolling_variance) {
		free(layer->rolling_variance);
		layer->rolling_variance = NULL;
	}
	
	if (layer->vecmat) {
		free(layer->vecmat);
		layer->vecmat = NULL;
	}
	
	if (layer->output) {
		free(layer->output);
		layer->output = NULL;
	}
	
#ifdef NNPACK
	if (layer->transformed_kernel) {
		free(layer->transformed_kernel);
		layer->transformed_kernel = NULL;
	}
#endif
#if defined(OPENCL)
#ifdef WINOGRAD_CONVOLUTION
	if (3 == layer->filter_size) {
		free_weight_transform_context(layer->wtc);
		free_input_transform_context(layer->itc);
		free_matrix_multiplication_context(layer->mmc);
		free_output_inverse_transform_context(layer->oitc);
	}
#endif
	if (1 == layer->filter_size) {
		free_direct_convolution_context(layer->dcc);
	}
#endif
	free_gemm_context(layer->gc);
	
	free(layer);
	layer = NULL;
}

void print_convolutional_layer_info(void *_layer, int id)
{
	convolutional_layer *layer = (convolutional_layer*)_layer;
	static double total_bflop = 0;
	double bflop = layer->filter_size * layer->filter_size * layer->input_size.c * layer->output_size.w *
		layer->output_size.h * layer->output_size.c * 2 / 1000000000.0;
	total_bflop += bflop;
	printf("%2d\tconv\t\t%4d x%4d x%4d\t\t%dx%d/%d\t\t%4d\t\t%4d x%4d x%4d\t%.9f->%.9f BFLOPs\n",
		id,
		layer->input_size.w,
		layer->input_size.h,
		layer->input_size.c,
		layer->filter_size,
		layer->filter_size,
		layer->stride,
		layer->nfilters,
		layer->output_size.w,
		layer->output_size.h,
		layer->output_size.c,
		bflop,
		total_bflop);
}

void set_convolutional_layer_input(void *_layer, void *input)
{
	convolutional_layer *layer = (convolutional_layer *)_layer;
#if !defined(OPENCL) || !defined(WINOGRAD_CONVOLUTION)
	layer->input = input;
#else
	if (3 == layer->filter_size) {
		set_winograd_convolution_input(layer->itc, input);
	} else if (1 == layer->filter_size) {
		layer->dcc->d_input = input;
	}
#endif
}

void *get_convolutional_layer_output(void *_layer)
{
	convolutional_layer *layer = (convolutional_layer *)_layer;
#if !defined(OPENCL) || !defined(WINOGRAD_CONVOLUTION)
	return layer->output;
#else
	if (3 == layer->filter_size) {
		return get_winograd_convolution_output(layer->oitc);
	} else if (1 == layer->filter_size) {
		return layer->dcc->d_output;
	}
	return NULL;
#endif
}

void forward_convolutional_layer(void *_layer, znet *net)
{
#ifdef NNPACK
	return forward_convolutional_layer_nnp(_layer, net); 
#endif
	convolutional_layer *layer = (convolutional_layer *)_layer;
	if (3 == layer->filter_size) {
#if defined(OPENCL) && defined(WINOGRAD_CONVOLUTION)
		return forward_convolutional_layer_gpu(_layer, net);
#endif		
	} else if (1 == layer->filter_size) {
#if defined(OPENCL) && defined(WINOGRAD_CONVOLUTION)
		return forward_convolutional_layer_1x1(layer);
#endif
	} else {;}
	
	float alpha = 0;
	size_t size = layer->noutputs * layer->batch_size * sizeof(float);
	mset((char *const)layer->output, size, (const char *const)&alpha, sizeof(float));
	
	int m = layer->nfilters;
	int n = layer->output_size.w * layer->output_size.h;
	int k = layer->filter_size * layer->filter_size * layer->input_size.c;
	for (int i = 0; i < layer->batch_size; ++i) {
		float *image = layer->input + i * layer->ninputs;
		float *A = layer->weights;
		float *B = layer->vecmat;
		float *C = layer->output + i * layer->noutputs;
		static double total = 0;
		struct timeval t1, t2; 
		gettimeofday(&t1, NULL);
		if (1 == layer->filter_size) {
			B = image;
		} else {
			im2col_cpu(image, layer->input_size.w, layer->input_size.h, layer->input_size.c,
				layer->filter_size, layer->stride, layer->padding, B);
		}
		gettimeofday(&t2, NULL);
		double duration = ((double)t2.tv_sec - t1.tv_sec) * 1000 + (t2.tv_usec - t1.tv_usec) / 1000.0;
		total += duration;
		printf("im2col_cpu: %f ms, total %f ms.\n", duration, total);
		
		gemm(layer->gc, 0, 0, m, n, k, 1, A, k, B, n, 1, C, n);
	}

#ifndef MERGE_BATCHNORM_TO_CONV	
	if (layer->batch_norm) {
		forward_batchnorm_layer(layer, net);
	} else {
#endif
		add_bias(layer->output, layer->biases, layer->batch_size, layer->nfilters, n);
#ifndef MERGE_BATCHNORM_TO_CONV
	}
#endif
	
	activate(layer->output, layer->noutputs * layer->batch_size, layer->activation);
}

void backward_convolutional_layer(convolutional_layer *layer, znet *net)
{
	fprintf(stderr, "Not implemented[%s:%d].\n", __FILE__, __LINE__);
}

void load_convolutional_layer_weights(convolutional_layer *layer, FILE *fp)
{
	fread(layer->biases, sizeof(float), layer->nbiases, fp);
	if (layer->batch_norm) {
		fread(layer->scales, sizeof(float), layer->nfilters, fp);
		fread(layer->rolling_mean, sizeof(float), layer->nfilters, fp);
		fread(layer->rolling_variance, sizeof(float), layer->nfilters, fp);
	}

	fread(layer->weights, sizeof(float), layer->nweights, fp);
#ifdef MERGE_BATCHNORM_TO_CONV
	if (layer->batch_norm) merge_batchnorm_params(layer);
#endif
	if (3 == layer->filter_size) {
#if defined(OPENCL) && defined(WINOGRAD_CONVOLUTION)
		transform_weight(layer->wtc, layer->weights, layer->biases, NULL);
#endif
	} else if (1 == layer->filter_size) {
#ifdef OPENCL
		load_direct_convolution_weight(layer->dcc, layer->weights, layer->biases);
#endif
	} else {;}
}

void load_convolutional_layer_weights_from_buffer(convolutional_layer *layer, char **buffer)
{
	char *ptr = *buffer;
	for (int i = 0; i < layer->nbiases; ++i) {
		layer->biases[i] = *((float *)ptr);
		ptr += sizeof(float);
	}

	if (layer->batch_norm) {
		for (int i = 0; i < layer->nfilters; ++i) {
			layer->scales[i] = *((float *)ptr);
			ptr += sizeof(float);
		}
		
		for (int i = 0; i < layer->nfilters; ++i) {
			layer->rolling_mean[i] = *((float *)ptr);
			ptr += sizeof(float);
		}
		
		for (int i = 0; i < layer->nfilters; ++i) {
			layer->rolling_variance[i] = *((float *)ptr);
			ptr += sizeof(float);
		}
	}
	
	for (int i = 0; i < layer->nweights; ++i) {
		layer->weights[i] = *((float *)ptr);
		ptr += sizeof(float);
	}

	*buffer = ptr;
#ifdef MERGE_BATCHNORM_TO_CONV
	if (layer->batch_norm) merge_batchnorm_params(layer);
#endif
	if (3 == layer->filter_size) {
#if defined(OPENCL) && defined(WINOGRAD_CONVOLUTION)
		transform_weight(layer->wtc, layer->weights, layer->biases, NULL);
#endif
	} else if (1 == layer->filter_size) {
#ifdef OPENCL
		load_direct_convolution_weight(layer->dcc, layer->weights, layer->biases);
#endif
	} else {;}	
}

int convolutional_output_width(convolutional_layer *layer)
{
	return (layer->input_size.w + 2 * layer->padding - layer->filter_size) / layer->stride + 1;
}

int convolutional_output_height(convolutional_layer *layer)
{
	return (layer->input_size.h + 2 * layer->padding - layer->filter_size) / layer->stride + 1;
}

/** @brief 添加加性偏置到输入的卷积输出,或已添加乘性偏置的输入的卷积输出上.
 ** @param output 输入的卷积输出,或已添加乘性偏置的输入的卷积输出.
 ** @param biases 神经元加性偏置.
 ** @param batch_size 批量大小.
 ** @param nchannels 卷积输出的通道数.
 ** @param size 卷积输出的大小.
 **/
void add_bias(float *output, float *biases, int batch_size, int nchannels, int size)
{
#ifdef __INTEL_SSE__
	return add_bias_sse(output, biases, batch_size, nchannels, size);
#elif __ARM_NEON__
	return add_bias_neon(output, biases, batch_size, nchannels, size);
#endif
	for (int i = 0; i < batch_size; ++i) {
		for (int j = 0; j < nchannels; ++j) {
			float *at = output + (i * nchannels + j) * size;
			for (int k = 0; k < size; ++k) {
				at[k] += biases[j];
			}
		}
	}
}

/** @brief 添加乘性偏置到输入的卷积输出上.
 ** @param output 输入的卷积输出.
 ** @param scales 神经元乘性偏置.
 ** @param batch_size 批量大小.
 ** @param nchannels 卷积输出的通道数.
 ** @param size 卷积输出的大小.
 **/
void mul_bias(float *output, float *scales, int batch_size, int nchannels, int size)
{
#ifdef __INTEL_SSE__
	return mul_bias_sse(output, scales, batch_size, nchannels, size);
#elif __ARM_NEON__
	return mul_bias_neon(output, scales, batch_size, nchannels, size);
#endif
	for (int i = 0; i < batch_size; ++i) {
		for (int j = 0; j < nchannels; ++j) {
			float *at = output + (i * nchannels + j) * size;
			for (int k = 0; k < size; ++k) {
				at[k] *= scales[j];
			}
		}
	}
}

#ifdef MERGE_BATCHNORM_TO_CONV
void merge_batchnorm_params(convolutional_layer *layer)
{
	int num_weis = layer->filter_size * layer->filter_size * layer->input_size.c;
	for (int i = 0; i < layer->nfilters; ++i) {
		float alpha = layer->scales[i] / sqrt(layer->rolling_variance[i] + 1e-6);
		float *at = layer->weights + i * num_weis;
		for (int j = 0; j < num_weis; ++j) {
			at[j] *= alpha;
		}
		
		layer->biases[i] = layer->biases[i] - layer->rolling_mean[i] * alpha;
	}
}
#endif

#ifdef NNPACK
void forward_convolutional_layer_nnp(void *_layer, znet *net)
{	
	convolutional_layer *layer = (convolutional_layer *)_layer;
	int n = layer->output_size.w * layer->output_size.h;	
	struct nnp_size input_size = {layer->input_size.w, layer->input_size.h};
	struct nnp_padding input_padding = {layer->padding, layer->padding, layer->padding, layer->padding};
	struct nnp_size kernel_size = {layer->filter_size, layer->filter_size};
	struct nnp_size stride = {layer->stride, layer->stride};
	
	float zeros[2048];
	for (int i = 0; i < 2048; ++i) zeros[i] = 0;

	if (3 != layer->filter_size) {
#ifdef NNPACK_PROFILING_ENABLE
		static double total = 0;
		struct timeval t1, t2; 
		gettimeofday(&t1, NULL);
#endif
		nnp_convolution_inference(
			nnp_convolution_algorithm_direct,
			nnp_convolution_transform_strategy_tuple_based,
			layer->input_size.c,
			layer->nfilters,
			input_size,
			input_padding,
			kernel_size,
			stride,
			layer->input,
			layer->weights,
			zeros,
			layer->output,
			NULL,
			NULL,
			nnp_activation_identity,
			NULL,
			znet_threadpool(net),
			NULL
		);
#ifdef NNPACK_PROFILING_ENABLE
		gettimeofday(&t2, NULL);
		double duration = ((double)t2.tv_sec - t1.tv_sec) * 1000 + (t2.tv_usec - t1.tv_usec) / 1000.0;
		total += duration;
		printf("nnp_convolution_algorithm_direct: %f ms, total %f ms.\n", duration, total);
#endif
	} else {
		if (NULL == layer->transformed_kernel) {
			nnp_convolution_inference(
				layer->algorithm,
				nnp_convolution_transform_strategy_precompute,
				layer->input_size.c,
				layer->nfilters,
				input_size,
				input_padding,
				kernel_size,
				stride,
				NULL,
				NULL,
				NULL,
				NULL,
				NULL,
				&layer->transformed_kernel_size,
				nnp_activation_identity,
				NULL,
				znet_threadpool(net),
				NULL
			);
			
			layer->transformed_kernel = calloc(layer->transformed_kernel_size, 1);
			
			nnp_convolution_inference(
				layer->algorithm,
				nnp_convolution_transform_strategy_precompute,
				layer->input_size.c,
				layer->nfilters,
				input_size,
				input_padding,
				kernel_size,
				stride,
				layer->input,
				layer->weights,
				NULL,
				layer->output,
				layer->transformed_kernel,
				&layer->transformed_kernel_size,
				nnp_activation_identity,
				NULL,
				znet_threadpool(net),
				NULL
			);
		}
#ifdef NNPACK_PROFILING_ENABLE		
		static double total = 0;
		struct timeval t1, t2; 
		gettimeofday(&t1, NULL);
#endif
		nnp_convolution_inference(
			layer->algorithm,
			nnp_convolution_transform_strategy_reuse,
			layer->input_size.c,
			layer->nfilters,
			input_size,
			input_padding,
			kernel_size,
			stride,
			layer->input,
			layer->transformed_kernel,
			zeros,
			layer->output,
			NULL,
			NULL,
			nnp_activation_identity,
			NULL,
			znet_threadpool(net),
			NULL
		);
#ifdef NNPACK_PROFILING_ENABLE
		gettimeofday(&t2, NULL);
		double duration = ((double)t2.tv_sec - t1.tv_sec) * 1000 + (t2.tv_usec - t1.tv_usec) / 1000.0;
		total += duration;
		printf("nnp_convolution_algorithm_wt8x8_fp16: %f ms, total %f ms.\n", duration, total);
#endif
	}

#ifndef MERGE_BATCHNORM_TO_CONV	
	if (layer->batch_norm) {
		forward_batchnorm_layer(layer, net);
	} else {
#endif
		add_bias(layer->output, layer->biases, layer->batch_size, layer->nfilters, n);
#ifndef MERGE_BATCHNORM_TO_CONV
	}
#endif
	
	activate(layer->output, layer->noutputs * layer->batch_size, layer->activation);
}
#endif

#if defined(OPENCL) && defined(WINOGRAD_CONVOLUTION)
void forward_convolutional_layer_gpu(void *_layer, znet *net)
{
	convolutional_layer *layer = (convolutional_layer *)_layer;
	float *transformed_input = NULL;
	float *inverse_transformed_output = NULL;	
	transform_input(layer->itc, transformed_input);	
	multiply_transformed_matrix(layer->mmc, NULL);
	inverse_transform_output(layer->oitc, inverse_transformed_output);
}
#endif

#ifdef OPENCL
void get_direct_convolution_output_image_size(convolutional_layer *layer, int *width, int *height)
{
	if (layer) {
		*width = layer->dcc->output_image_width;
		*height = layer->dcc->output_image_height;
	} else {
		*width = 0;
		*height = 0;
	}
}

void load_direct_convolution_weight(direct_convolution_context *context, float *weights, float *biases)
{	
	cl_int errcode;
	size_t origin[] = {0, 0, 0};
	size_t region[] = {context->weight_image_width, context->weight_image_height, 1};
	size_t image_row_pitch, image_slice_pitch;
	MEM_MAP_PTR_TYPE *h_weights = clEnqueueMapImage(wrapper.command_queue, context->d_weights, CL_TRUE, CL_MAP_WRITE,
		origin, region, &image_row_pitch, &image_slice_pitch, 0, NULL, NULL, &errcode);
	image_row_pitch = image_row_pitch / sizeof(MEM_MAP_PTR_TYPE);
	const int buffer_row_pitch = context->input_channel_blocks << 2;
	for (int y = 0; y < context->weight_image_height; ++y) {
		for (int x = 0; x < context->input_channel_blocks; ++x) {
			MEM_MAP_PTR_TYPE *ptr = h_weights + y * image_row_pitch + (x << 4);
			for (int z = 0; z < 4; ++z) {
#if 0
				ptr[(z << 2) + 0] = weights[((y << 2) + z) * buffer_row_pitch + (x << 2) + 0];
				ptr[(z << 2) + 1] = weights[((y << 2) + z) * buffer_row_pitch + (x << 2) + 1];
				ptr[(z << 2) + 2] = weights[((y << 2) + z) * buffer_row_pitch + (x << 2) + 2];
				ptr[(z << 2) + 3] = weights[((y << 2) + z) * buffer_row_pitch + (x << 2) + 3];
#else
				ptr[(z << 2) + 0] = HOST_TO_DEVICE(weights[((y << 2) + 0) * buffer_row_pitch + (x << 2) + z]);
				ptr[(z << 2) + 1] = HOST_TO_DEVICE(weights[((y << 2) + 1) * buffer_row_pitch + (x << 2) + z]);
				ptr[(z << 2) + 2] = HOST_TO_DEVICE(weights[((y << 2) + 2) * buffer_row_pitch + (x << 2) + z]);
				ptr[(z << 2) + 3] = HOST_TO_DEVICE(weights[((y << 2) + 3) * buffer_row_pitch + (x << 2) + z]);
#endif
			}
		}
	}
	clEnqueueUnmapMemObject(wrapper.command_queue, context->d_weights, h_weights, 0, NULL, NULL);
	
	region[0] = context->bias_image_width;
	region[1] = 1;
	MEM_MAP_PTR_TYPE *h_biases = clEnqueueMapImage(wrapper.command_queue, context->d_biases, CL_TRUE, CL_MAP_WRITE, origin,
		region, &image_row_pitch, &image_slice_pitch, 0, NULL, NULL, &errcode);
	for (int i = 0; i < context->nfilters; ++i) h_biases[i] = HOST_TO_DEVICE(biases[i]);
	clEnqueueUnmapMemObject(wrapper.command_queue, context->d_biases, h_biases, 0, NULL, NULL);
}

direct_convolution_context *create_direct_convolution_context(convolutional_layer *layer)
{
	direct_convolution_context *context = calloc(1, sizeof(direct_convolution_context));
	if (!context) {
		fprintf(stderr, "calloc fail[%s:%d].\n", __FILE__, __LINE__);
		return context;
	}
	
	context->program = 0;
	context->kernel = 0;
	context->d_input = 0;
	context->d_weights = 0;
	context->d_biases = 0;
	context->d_output = 0;
	context->nfilters = layer->nfilters;
	context->input_channel_blocks = round_up_division_4(layer->input_size.c);
	context->weight_image_width = context->input_channel_blocks << 2;
	context->weight_image_height = round_up_division_4(layer->nfilters);
	context->bias_image_width = layer->nfilters;
	context->input_image_width = layer->input_size.w * context->input_channel_blocks;
	context->input_image_height = layer->input_size.h;
	context->output_channel_blocks = round_up_division_4(layer->output_size.c);
	context->output_image_width = layer->output_size.w * context->output_channel_blocks;
	context->output_image_height = layer->output_size.h;
	
	char options[256] = "-I. -cl-fast-relaxed-math";
	switch (layer->activation) {
	case RELU:
		strcat(options, " -DRELU");
		break;
	case LEAKY:
		strcat(options, " -DLEAKY");
		break;
	case LINEAR:
		strcat(options, " -DLINEAR");
		break;
	case LOGISTIC:
		strcat(options, " -DLOGISTIC");
		break;
	default:
		strcat(options, " -DLINEAR");
		break;
	}
	PARSE_PRECISION;

	size_t header_size = (size_t)(&BINARY_FILENAME_TO_END(cl_common, h) - &BINARY_FILENAME_TO_START(cl_common, h));
	size_t size = (size_t)(&BINARY_FILENAME_TO_END(convolution, cl) - &BINARY_FILENAME_TO_START(convolution, cl));
	context->program_buffer = calloc(header_size + size + 1, sizeof(char));
	if (!context->program_buffer) {
		fprintf(stderr, "calloc fail[%s:%d].\n", __FILE__, __LINE__);
		goto cleanup;
	}
	
	memcpy(context->program_buffer, &BINARY_FILENAME_TO_START(cl_common, h), header_size);
	memcpy(context->program_buffer + header_size, &BINARY_FILENAME_TO_START(convolution, cl), size);
	context->program_buffer[header_size + size] = '\0';

	cl_int errcode;
	context->program = cl_make_wrapper_program(wrapper, "convolution.cl", context->program_buffer, options, &errcode);
	if (CL_SUCCESS != errcode) {
		fprintf(stderr, "cl_make_wrapper_program[%s:%d:%d].\n", __FILE__, __LINE__, errcode);
		goto cleanup;
	}
	
	context->kernel = cl_make_wrapper_kernel(wrapper, context->program, "direct_convolution_2d_1x1", &errcode);
	if (CL_SUCCESS != errcode) {
		fprintf(stderr, "cl_make_wrapper_kernel[%s:%d:%d].\n", __FILE__, __LINE__, errcode);
		goto cleanup;
	}
	
	cl_mem_flags mem_flags = CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR;
	cl_image_format image_format = {
		.image_channel_order = CL_RGBA,
		.image_channel_data_type = IMAGE_CHANNEL_DATA_TYPE
	};
	
	cl_image_desc image_desc;
	memset(&image_desc, 0, sizeof(cl_image_desc));
	image_desc.image_type = CL_MEM_OBJECT_IMAGE2D;
	image_desc.image_width = context->weight_image_width;
	image_desc.image_height = context->weight_image_height;
	
	context->d_weights = clCreateImage(wrapper.context, mem_flags, &image_format, &image_desc, NULL, &errcode);
	if (CL_SUCCESS != errcode) {
		fprintf(stderr, "clCreateImage fail[%s:%d:%d].\n", __FILE__, __LINE__, errcode);
		goto cleanup;
	}

	memset(&image_desc, 0, sizeof(cl_image_desc));
	image_desc.image_type = CL_MEM_OBJECT_IMAGE1D;
	image_desc.image_width = context->bias_image_width;
	
	context->d_biases = clCreateImage(wrapper.context, mem_flags, &image_format, &image_desc, NULL, &errcode);
	if (CL_SUCCESS != errcode) {
		fprintf(stderr, "clCreateImage[%s:%d:%d].\n", __FILE__, __LINE__, errcode);
		goto cleanup;
	}
	
	mem_flags = CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR;
	memset(&image_desc, 0, sizeof(cl_image_desc));
	image_desc.image_type = CL_MEM_OBJECT_IMAGE2D;
	image_desc.image_width = context->output_image_width;
	image_desc.image_height = context->output_image_height;
	
	context->d_output = clCreateImage(wrapper.context, mem_flags, &image_format, &image_desc, NULL, &errcode);
	if (CL_SUCCESS != errcode) {
		fprintf(stderr, "clCreateImage fail[%s:%d:%d].\n", __FILE__, __LINE__, errcode);
		cleanup:free_direct_convolution_context(context);
		return 0;
	}
	
	return context;
}

void forward_convolutional_layer_1x1(convolutional_layer *layer)
{
	cl_int errcode;
	int flag = 1;
	if (layer->activation == LINEAR) flag = 0;
	direct_convolution_context *context = layer->dcc;
	errcode  = clSetKernelArg(context->kernel, 0, sizeof(cl_mem), &context->d_weights);
	errcode |= clSetKernelArg(context->kernel, 1, sizeof(cl_mem), &context->d_input);
	errcode |= clSetKernelArg(context->kernel, 2, sizeof(cl_mem), &context->d_biases);
	errcode |= clSetKernelArg(context->kernel, 3, sizeof(cl_mem), &context->d_output);
	errcode |= clSetKernelArg(context->kernel, 4, sizeof(int), &layer->output_size.w);
	errcode |= clSetKernelArg(context->kernel, 5, sizeof(int), &context->input_channel_blocks);
	errcode |= clSetKernelArg(context->kernel, 6, sizeof(int), &flag);
	if (CL_SUCCESS != errcode) {
		fprintf(stderr, "clSetKernelArg fail[%s:%d:%d].\n", __FILE__, __LINE__, errcode);
		return;
	}
	
	cl_event event;
	cl_uint work_dim = 3;
	size_t global_work_size[] = {context->output_channel_blocks, round_up_division_4(layer->output_size.w),
		layer->output_size.h};
	clEnqueueNDRangeKernel(wrapper.command_queue, context->kernel, work_dim, NULL, global_work_size,
		NULL, 0, NULL, &event);

#ifdef NDEBUG	
	cl_ulong start, end;
	clFinish(wrapper.command_queue);
	errcode  = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
	errcode |= clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
	float duration = (end - start) * 1e-6f;
	LOGD("GPU, direct_convolution_1x1: %fms.\n", duration);
#endif
	clReleaseEvent(event);
}

void free_direct_convolution_context(direct_convolution_context *context)
{
	if (context) {
		free(context->program_buffer);
		clReleaseMemObject(context->d_biases);
		clReleaseMemObject(context->d_weights);
		clReleaseMemObject(context->d_output);
		clReleaseProgram(context->program);
		clReleaseKernel(context->kernel);
		free(context);
	}
}
#endif

#ifdef __INTEL_SSE__
void add_bias_sse(float *output, float *biases, int batch_size, int nchannels, int size)
{
	for (int i = 0; i < batch_size; ++i) {
		#pragma omp parallel for num_threads(8)
		for (int j = 0; j < nchannels; ++j) {
			float *at = output + (i * nchannels + j) * size;
			int batches = 4;
			int excess = size - size % batches;
			__m128 bs = _mm_set1_ps(biases[j]);
			for (int k = 0; k < excess; k += batches) {
				__m128 os = _mm_loadu_ps(at + k);
				os = _mm_add_ps(os, bs);
				_mm_storeu_ps(at + k, os);
			}
			for (int k = excess; k < size; ++k) {
				at[k] += biases[j];
			}
		}
	}
}

void mul_bias_sse(float *output, float *scales, int batch_size, int nchannels, int size)
{
	for (int i = 0; i < batch_size; ++i) {
		#pragma omp parallel for num_threads(8)
		for (int j = 0; j < nchannels; ++j) {
			float *at = output + (i * nchannels + j) * size;
			int batches = 4;
			int excess = size - size % batches;
			__m128 ss = _mm_set1_ps(scales[j]);
			for (int k = 0; k < excess; k += batches) {
				__m128 os = _mm_loadu_ps(at + k);
				os = _mm_mul_ps(os, ss);
				_mm_storeu_ps(at + k, os);
			}
			for (int k = excess; k < size; ++k) {
				at[k] *= scales[j];
			}
		}
	}
}

#elif __ARM_NEON__
void add_bias_neon(float *output, float *biases, int batch_size, int nchannels, int size)
{
	for (int i = 0; i < batch_size; ++i) {
		#pragma omp parallel for num_threads(4)
		for (int j = 0; j < nchannels; ++j) {
			float *at = output + (i * nchannels + j) * size;
			int batches = 4;
			int excess = size - size % batches;
			float32x4_t bs = vdupq_n_f32(biases[j]);
			for (int k = 0; k < excess; k += batches) {
				float32x4_t os = vld1q_f32(at + k);
				os = vaddq_f32(os, bs);
				vst1q_f32(at + k, os);
			}
			for (int k = excess; k < size; ++k) {
				at[k] += biases[j];
			}
		}
	}
}

void mul_bias_neon(float *output, float *scales, int batch_size, int nchannels, int size)
{
	for (int i = 0; i < batch_size; ++i) {
		#pragma omp parallel for num_threads(4)
		for (int j = 0; j < nchannels; ++j) {
			float *at = output + (i * nchannels + j) * size;
			int batches = 4;
			int excess = size - size % batches;
			for (int k = 0; k < excess; k += batches) {
				float32x4_t os = vld1q_f32(at + k);
				os = vmulq_n_f32(os, scales[j]);
				vst1q_f32(at + k, os);
			}
			for (int k = excess; k < size; ++k) {
				at[k] *= scales[j];
			}
		}
	}
}
#endif