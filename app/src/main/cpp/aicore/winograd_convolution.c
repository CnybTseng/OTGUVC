#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "winograd_convolution.h"
#include "gemm.h"
#ifdef OPENCL
#	include "cl_wrapper.h"
#endif
#include "half.h"

#ifdef OPENCL
extern cl_wrapper wrapper;
extern char BINARY_FILENAME_TO_START(cl_common, h);
extern char BINARY_FILENAME_TO_END(cl_common, h);
extern char BINARY_FILENAME_TO_START(convolution, cl);
extern char BINARY_FILENAME_TO_END(convolution, cl);

struct weight_transform_context {
	char *program_buffer;
	cl_program program;
	cl_kernel kernel;
	cl_mem d_weight;
	cl_mem d_transformed_weight;
	cl_mem d_biases;
	int filter_size;
	int filter_channels;
	int nfilters;
	int tile_input_size;
	int weight_image_width;
	int weight_image_height;
	int transformed_weight_image_width;
	int transformed_weight_image_height;
	int biases_image_width;
	int input_channel_blocks;
};

struct input_transform_context {
	char *program_buffer;
	cl_program program;
	cl_kernel kernel;
	cl_mem d_input;
	cl_mem d_transformed_input;
	int input_width;
	int input_height;
	int input_channels;
	int stride;
	int padding;
	int end_of_line;
	int ntilesX;
	int ntilesY;
	int transformed_input_image_width;
	int transformed_input_image_height;
	int input_image_width;
	int input_image_height;
	int input_channel_blocks;
};

struct matrix_multiplication_context {
	char *program_buffer;
	cl_program program;
	cl_kernel kernel;
	weight_transform_context *wtc;
	input_transform_context *itc;
	cl_mem d_output;
	int output_image_width;
	int output_image_height;
	int output_channel_blocks;
};

struct output_inverse_transform_context {
	char *program_buffer;
	cl_program program;
	cl_kernel kernel;
	matrix_multiplication_context *mmc;
	cl_mem d_inverse_transformed_output;
	int tile_output_size;
	int inverse_transformed_output_image_width;
	int inverse_transformed_output_image_height;
};
#endif

int get_image_tile_size(WINOGRAD_CONV_TYPE conv)
{
	const int filter_size = 3;
	int tile_output_size = get_tile_output_size(conv);	
	return filter_size + tile_output_size - 1;
}

int get_tile_output_size(WINOGRAD_CONV_TYPE conv)
{
	if (conv == F6x6_3x3) {
		return 6;
	} else if (conv == F4x4_3x3) {
		return 4;
	} else if (conv == F2x2_3x3) {
		return 2;
	}
	
	fprintf(stderr, "Not implemented[%s:%d].\n", __FILE__, __LINE__);
	return 0;
}

#ifdef OPENCL
weight_transform_context *create_weight_transform_context(WINOGRAD_CONV_TYPE conv, int filter_channels, int nfilters)
{
	if (conv != F4x4_3x3) {
		fprintf(stderr, "Winograd convolution type %d isn't supported now[%s:%d]!\n", conv, __FILE__, __LINE__);
		return 0;
	}
	
	weight_transform_context *context = calloc(1, sizeof(weight_transform_context));
	if (!context) {
		fprintf(stderr, "calloc fail[%s:%d].\n", __FILE__, __LINE__);
		return context;
	}
	
	context->filter_size = 3;
	context->filter_channels = filter_channels;
	context->nfilters = nfilters;
	
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
	char options[256] = "-cl-fast-relaxed-math -I.";
	PARSE_PRECISION;
	context->program = cl_make_wrapper_program(wrapper, "convolution.cl", context->program_buffer, options, &errcode);
	if (CL_SUCCESS != errcode) {
		fprintf(stderr, "cl_make_wrapper_program[%s:%d:%d].\n", __FILE__, __LINE__, errcode);
		goto cleanup;
	}
	
	context->kernel = cl_make_wrapper_kernel(wrapper, context->program, "weight_transform_f4x4_3x3", &errcode);
	if (CL_SUCCESS != errcode) {
		fprintf(stderr, "cl_make_wrapper_kernel[%s:%d:%d].\n", __FILE__, __LINE__, errcode);
		goto cleanup;
	}
	
	context->tile_input_size = get_image_tile_size(conv);
	context->weight_image_width = (context->filter_size * context->filter_size) * ((filter_channels + 3) >> 2);
	context->weight_image_height = nfilters;
	context->input_channel_blocks = (context->filter_channels + 3) >> 2;
	context->transformed_weight_image_width = (context->tile_input_size * context->tile_input_size) * context->input_channel_blocks;
	context->transformed_weight_image_height = nfilters;
	context->biases_image_width = (nfilters + 3) >> 2;
	
	cl_mem_flags mem_flags = CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR;
	cl_image_format image_format = {
		.image_channel_order = CL_RGBA,
		.image_channel_data_type = IMAGE_CHANNEL_DATA_TYPE
	};
	
	cl_image_desc weight_image_desc;
	memset(&weight_image_desc, 0, sizeof(cl_image_desc));
	weight_image_desc.image_type = CL_MEM_OBJECT_IMAGE2D;
	weight_image_desc.image_width = context->weight_image_width;
	weight_image_desc.image_height = context->weight_image_height;
	
	context->d_weight = clCreateImage(wrapper.context, mem_flags, &image_format, &weight_image_desc, NULL, &errcode);
	if (CL_SUCCESS != errcode) {
		fprintf(stderr, "clCreateImage fail[%s:%d:%d].\n", __FILE__, __LINE__, errcode);
		goto cleanup;
	}

	mem_flags = CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR;
	cl_image_desc transformed_weight_image_desc;
	memset(&transformed_weight_image_desc, 0, sizeof(cl_image_desc));
	transformed_weight_image_desc.image_type = CL_MEM_OBJECT_IMAGE2D;
	transformed_weight_image_desc.image_width = context->transformed_weight_image_width;
	transformed_weight_image_desc.image_height = context->transformed_weight_image_height;
	
	context->d_transformed_weight = clCreateImage(wrapper.context, mem_flags, &image_format, &transformed_weight_image_desc,
		NULL, &errcode);
	if (CL_SUCCESS != errcode) {
		fprintf(stderr, "clCreateImage[%s:%d:%d].\n", __FILE__, __LINE__, errcode);
		goto cleanup;
	}
	
	mem_flags = CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR;
	cl_image_desc biases_image_desc;
	memset(&biases_image_desc, 0, sizeof(cl_image_desc));
	biases_image_desc.image_type = CL_MEM_OBJECT_IMAGE1D;
	biases_image_desc.image_width = context->biases_image_width;
	
	context->d_biases = clCreateImage(wrapper.context, mem_flags, &image_format, &biases_image_desc, NULL, &errcode);
	if (CL_SUCCESS != errcode) {
		fprintf(stderr, "clCreateImage[%s:%d:%d].\n", __FILE__, __LINE__, errcode);
		cleanup:free_weight_transform_context(context);
		return 0;
	}
	
	return context;
}

void get_transformed_weight_image_size(weight_transform_context *context, int *width, int *height)
{
	if (context) {
		*width = context->transformed_weight_image_width;
		*height = context->transformed_weight_image_height;
	} else {
		fprintf(stderr, "invalid weight transform context[%s:%d]!\n", __FILE__, __LINE__);
		*width = 0;
		*height = 0;
	}
}

void transform_weight(weight_transform_context *context, float *weights, float *biases, float *transformed_weights)
{
	cl_int errcode;
	size_t weight_image_origin[] = {0, 0, 0};
	size_t weight_image_region[] = {context->weight_image_width, context->weight_image_height, 1};
	size_t image_row_pitch, image_slice_pitch;
	MEM_MAP_PTR_TYPE *h_weight = clEnqueueMapImage(wrapper.command_queue, context->d_weight, CL_TRUE, CL_MAP_WRITE, weight_image_origin,
		weight_image_region, &image_row_pitch, &image_slice_pitch, 0, NULL, NULL, &errcode);

	image_row_pitch = image_row_pitch / sizeof(MEM_MAP_PTR_TYPE);
	const int filter_slice_pitch = context->filter_size * context->filter_size;
	for (int y = 0; y < context->nfilters; ++y) {
		for (int g = 0; g < context->input_channel_blocks; ++g) {
			const float *ptr = weights + (y * context->filter_channels + (g << 2)) * filter_slice_pitch;
			const int filter_channel_remainder = context->filter_channels - (g << 2);
			const int filter_channel_in_group = filter_channel_remainder < 4 ? filter_channel_remainder : 4;
			for (int i = 0; i < context->filter_size * context->filter_size; ++i) {
				for (int x = 0; x < filter_channel_in_group; ++x) {
					h_weight[y * image_row_pitch + ((g * context->filter_size * context->filter_size + i) << 2) + x] =
						HOST_TO_DEVICE(ptr[x * filter_slice_pitch + i]);
				}
				for (int x = filter_channel_in_group; x < 4; ++x) {
					h_weight[y * image_row_pitch + ((g * context->filter_size * context->filter_size + i) << 2) + x] = 0;
				}
			}
		}
	}	
	
	clEnqueueUnmapMemObject(wrapper.command_queue, context->d_weight, h_weight, 0, NULL, NULL);

	errcode  = clSetKernelArg(context->kernel, 0, sizeof(cl_mem), &context->d_weight);
	errcode |= clSetKernelArg(context->kernel, 1, sizeof(cl_mem), &context->d_transformed_weight);
	if (CL_SUCCESS != errcode) {
		fprintf(stderr, "clSetKernelArg fail[%s:%d:%d].\n", __FILE__, __LINE__, errcode);
	}
	
	cl_event event;
	cl_uint work_dim = 2;
	size_t work_group_size;
	clGetKernelWorkGroupInfo(context->kernel, wrapper.device, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &work_group_size, NULL);
	size_t global_work_size[] = {(context->filter_channels + 3) >> 2, context->nfilters};
	clEnqueueNDRangeKernel(wrapper.command_queue, context->kernel, work_dim, NULL, global_work_size,
		NULL, 0, NULL, &event);

#ifdef NDEBUG
	static float total = 0;
	cl_ulong start, end;
	clFinish(wrapper.command_queue);
	errcode  = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
	errcode |= clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
	float duration = (end - start) * 1e-6f;
	total += duration;
	LOGD("GPU, weight_transform_f4x4_3x3: %fms, total %fms\n", duration, total);
#endif
	clReleaseEvent(event);
	
	size_t biases_image_origin[] = {0, 0, 0};
	size_t biases_image_region[] = {context->biases_image_width, 1, 1};
	MEM_MAP_PTR_TYPE *h_biases = clEnqueueMapImage(wrapper.command_queue, context->d_biases, CL_TRUE, CL_MAP_WRITE, biases_image_origin,
		biases_image_region, &image_row_pitch, &image_slice_pitch, 0, NULL, NULL, &errcode);
	for (int i = 0; i < context->nfilters; ++i) h_biases[i] = HOST_TO_DEVICE(biases[i]);
	clEnqueueUnmapMemObject(wrapper.command_queue, context->d_biases, h_biases, 0, NULL, NULL);
	
	if (!transformed_weights) return;
	size_t transformed_weight_image_origin[] = {0, 0, 0};
	size_t transformed_weight_image_region[] = {context->transformed_weight_image_width, context->transformed_weight_image_height, 1};
	MEM_MAP_PTR_TYPE *h_transformed_weight = clEnqueueMapImage(wrapper.command_queue, context->d_transformed_weight, CL_TRUE, CL_MAP_READ,
		transformed_weight_image_origin, transformed_weight_image_region, &image_row_pitch, &image_slice_pitch, 0, NULL, NULL, &errcode);
	image_row_pitch = image_row_pitch / sizeof(MEM_MAP_PTR_TYPE);
	const int dst_row_pitch = (context->tile_input_size * context->tile_input_size) * (((context->filter_channels + 3) / 4) * 4);
	for (int i = 0; i < context->nfilters; ++i) {
		for (int j = 0; j < dst_row_pitch; ++j) {
			transformed_weights[i * dst_row_pitch + j] = DEVICE_TO_HOST(h_transformed_weight[i * image_row_pitch + j]);
		}
	}
	clEnqueueUnmapMemObject(wrapper.command_queue, context->d_transformed_weight, h_transformed_weight, 0, NULL, NULL);
}

void free_weight_transform_context(weight_transform_context *context)
{
	if (context) {
		free(context->program_buffer);
		clReleaseMemObject(context->d_weight);
		clReleaseMemObject(context->d_transformed_weight);
		clReleaseMemObject(context->d_biases);
		clReleaseProgram(context->program);
		clReleaseKernel(context->kernel);
		free(context);
	}
}

input_transform_context *create_input_transform_context(WINOGRAD_CONV_TYPE conv, int input_width,
	int input_height, int input_channels, int stride, int padding)
{
	if (conv != F4x4_3x3) {
		fprintf(stderr, "Winograd convolution type %d isn't supported now[%s:%d]!\n", conv, __FILE__, __LINE__);
		return 0;
	}
	
	input_transform_context *context = calloc(1, sizeof(input_transform_context));
	if (!context) {
		fprintf(stderr, "calloc fail[%s:%d].\n", __FILE__, __LINE__);
		return context;
	}
	
	context->input_width = input_width;
	context->input_height = input_height;
	context->input_channels = input_channels;
	context->stride = stride;
	context->padding = padding;
	
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
	char options[256] = "-cl-fast-relaxed-math -I.";
	PARSE_PRECISION;
	context->program = cl_make_wrapper_program(wrapper, "convolution.cl", context->program_buffer, options, &errcode);
	if (CL_SUCCESS != errcode) {
		fprintf(stderr, "cl_make_wrapper_program[%s:%d:%d].\n", __FILE__, __LINE__, errcode);
		goto cleanup;
	}
	
	context->kernel = cl_make_wrapper_kernel(wrapper, context->program, "input_transform_f4x4_3x3", &errcode);
	if (CL_SUCCESS != errcode) {
		fprintf(stderr, "cl_make_wrapper_kernel[%s:%d:%d].\n", __FILE__, __LINE__, errcode);
		goto cleanup;
	}

	context->input_channel_blocks = (input_channels + 3) >> 2;
	context->input_image_width = input_width * context->input_channel_blocks;
	context->input_image_height = input_height;
	
	const int input_size = get_image_tile_size(conv);
	const int output_size = get_tile_output_size(conv);
	context->ntilesX = (input_width + (output_size - 1)) / output_size;
	context->ntilesY = (input_height + (output_size - 1)) / output_size;
	context->transformed_input_image_width = context->ntilesX * context->ntilesY;
	context->transformed_input_image_height = context->input_channel_blocks * input_size * input_size;
	
	cl_mem_flags mem_flags = CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR;
	cl_image_format image_format = {
		.image_channel_order = CL_RGBA,
		.image_channel_data_type = IMAGE_CHANNEL_DATA_TYPE
	};
	
	cl_image_desc transformed_input_image_desc;
	memset(&transformed_input_image_desc, 0, sizeof(cl_image_desc));
	transformed_input_image_desc.image_type = CL_MEM_OBJECT_IMAGE2D,
	transformed_input_image_desc.image_width = context->transformed_input_image_width;
	transformed_input_image_desc.image_height = context->transformed_input_image_height;

	context->d_transformed_input = clCreateImage(wrapper.context, mem_flags, &image_format, &transformed_input_image_desc, NULL, &errcode);
	if (CL_SUCCESS != errcode) {
		fprintf(stderr, "clCreateImage fail[%s:%d:%d].\n", __FILE__, __LINE__, errcode);
		cleanup:free_input_transform_context(context);
		return 0;
	}
	
	return context;
}

void get_input_image_size(input_transform_context *context, int *width, int *height)
{
	if (context) {
		*width = context->input_image_width;
		*height = context->input_image_height;
	} else {
		fprintf(stderr, "invalid input transform context[%s:%d]!\n", __FILE__, __LINE__);
		*width = 0;
		*height = 0;
	}
}	

void get_transformed_input_image_size(input_transform_context *context, int *width, int *height)
{
	if (context) {
		*width = context->transformed_input_image_width;
		*height = context->transformed_input_image_height;
	} else {
		fprintf(stderr, "invalid input transform context[%s:%d]!\n", __FILE__, __LINE__);
		*width = 0;
		*height = 0;
	}
}
	
void transform_input(input_transform_context *context, float *transformed_input)
{	
	cl_int errcode;
	errcode  = clSetKernelArg(context->kernel, 0, sizeof(cl_mem), &context->d_input);
	errcode |= clSetKernelArg(context->kernel, 1, sizeof(cl_mem), &context->d_transformed_input);
	errcode |= clSetKernelArg(context->kernel, 2, sizeof(int), &context->input_width);
	errcode |= clSetKernelArg(context->kernel, 3, sizeof(int), &context->input_height);
	errcode |= clSetKernelArg(context->kernel, 4, sizeof(int), &context->input_channels);
	errcode |= clSetKernelArg(context->kernel, 5, sizeof(int), &context->ntilesX);
	if (CL_SUCCESS != errcode) {
		fprintf(stderr, "clSetKernelArg fail[%s:%d:%d].\n", __FILE__, __LINE__, errcode);
	}

	cl_event event;
	cl_uint work_dim = 2;
	size_t global_work_size[] = {context->ntilesX * context->ntilesY, context->input_channel_blocks};
	clEnqueueNDRangeKernel(wrapper.command_queue, context->kernel, work_dim, NULL, global_work_size,
		NULL, 0, NULL, &event);

#ifdef NDEBUG	
	static float total = 0;
	cl_ulong start, end;
	clFinish(wrapper.command_queue);
	errcode  = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
	errcode |= clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
	float duration = (end - start) * 1e-6f;
	total += duration;
	LOGD("GPU, input_transform_f4x4_3x3: %fms, total %fms\n", duration, total);
#endif
	clReleaseEvent(event);
	
	if (!transformed_input) return;
	size_t image_row_pitch, image_slice_pitch;
	size_t transformed_input_image_origin[] = {0, 0, 0};
	size_t transformed_input_image_region[] = {context->transformed_input_image_width, context->transformed_input_image_height, 1};
	MEM_MAP_PTR_TYPE *h_transformed_input = clEnqueueMapImage(wrapper.command_queue, context->d_transformed_input, CL_TRUE, CL_MAP_READ,
		transformed_input_image_origin, transformed_input_image_region, &image_row_pitch, &image_slice_pitch, 0, NULL, NULL, &errcode);
	image_row_pitch = image_row_pitch / sizeof(MEM_MAP_PTR_TYPE);
	const int dst_row_pitch = context->transformed_input_image_width << 2;
	for (int y = 0; y < context->transformed_input_image_height; ++y) {
		for (int x = 0; x < dst_row_pitch; ++x) {
			transformed_input[y * dst_row_pitch + x] = DEVICE_TO_HOST(h_transformed_input[y * image_row_pitch + x]);
		}
	}
	clEnqueueUnmapMemObject(wrapper.command_queue, context->d_transformed_input, h_transformed_input, 0, NULL, NULL);
}

void free_input_transform_context(input_transform_context *context)
{
	if (context) {
		free(context->program_buffer);
		clReleaseMemObject(context->d_transformed_input);
		clReleaseProgram(context->program);
		clReleaseKernel(context->kernel);
		free(context);
	}
}

matrix_multiplication_context *create_matrix_multiplication_context(weight_transform_context *wtc, input_transform_context *itc)
{	
	matrix_multiplication_context *context = calloc(1, sizeof(matrix_multiplication_context));
	if (!context) {
		fprintf(stderr, "calloc fail[%s:%d].\n", __FILE__, __LINE__);
		return context;
	}
	
	context->wtc = wtc;
	context->itc = itc;
	
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
	char options[256] = "-cl-fast-relaxed-math -I.";
	PARSE_PRECISION;
	context->program = cl_make_wrapper_program(wrapper, "convolution.cl", context->program_buffer, options, &errcode);
	if (CL_SUCCESS != errcode) {
		fprintf(stderr, "cl_make_wrapper_program[%s:%d:%d].\n", __FILE__, __LINE__, errcode);
		goto cleanup;
	}
	
	context->kernel = cl_make_wrapper_kernel(wrapper, context->program, "matrix_multiply", &errcode);
	if (CL_SUCCESS != errcode) {
		fprintf(stderr, "cl_make_wrapper_kernel[%s:%d:%d].\n", __FILE__, __LINE__, errcode);
		goto cleanup;
	}

	context->output_channel_blocks = (wtc->nfilters + 3) >> 2;
	context->output_image_width = itc->transformed_input_image_width;
	context->output_image_height = (wtc->tile_input_size * wtc->tile_input_size) * context->output_channel_blocks;
	
	cl_mem_flags mem_flags = CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR;
	cl_image_format image_format = {
		.image_channel_order = CL_RGBA,
		.image_channel_data_type = IMAGE_CHANNEL_DATA_TYPE
	};
	
	cl_image_desc output_image_desc;
	memset(&output_image_desc, 0, sizeof(cl_image_desc));
	output_image_desc.image_type = CL_MEM_OBJECT_IMAGE2D,
	output_image_desc.image_width = context->output_image_width;
	output_image_desc.image_height = context->output_image_height;

	context->d_output = clCreateImage(wrapper.context, mem_flags, &image_format, &output_image_desc, NULL, &errcode);
	if (CL_SUCCESS != errcode) {
		fprintf(stderr, "clCreateImage fail[%s:%d:%d].\n", __FILE__, __LINE__, errcode);
		cleanup:free_matrix_multiplication_context(context);
		return 0;
	}
	
	return context;
}

void get_transformed_output_image_size(matrix_multiplication_context *context, int *width, int *height)
{
	if (context) {
		*width = context->output_image_width;
		*height = context->output_image_height;
	} else {
		fprintf(stderr, "invalid matrix multiply context[%s:%d]!\n", __FILE__, __LINE__);
		*width = 0;
		*height = 0;
	}
}

void multiply_transformed_matrix(matrix_multiplication_context *context, float *output)
{
	cl_int errcode;
	const int ntiles = context->itc->ntilesX * context->itc->ntilesY;
	errcode  = clSetKernelArg(context->kernel, 0, sizeof(cl_mem), &context->wtc->d_transformed_weight);
	errcode |= clSetKernelArg(context->kernel, 1, sizeof(cl_mem), &context->itc->d_transformed_input);
	errcode |= clSetKernelArg(context->kernel, 2, sizeof(cl_mem), &context->d_output);
	errcode |= clSetKernelArg(context->kernel, 3, sizeof(int), &context->itc->input_channel_blocks);
	errcode |= clSetKernelArg(context->kernel, 4, sizeof(int), &context->output_channel_blocks);
	errcode |= clSetKernelArg(context->kernel, 5, sizeof(int), &ntiles);
	if (CL_SUCCESS != errcode) {
		fprintf(stderr, "clSetKernelArg fail[%s:%d:%d].\n", __FILE__, __LINE__, errcode);
	}
	
	cl_event event;
	cl_uint work_dim = 2;
	const int global_work_size_x = (ntiles + 3) >> 2;
	const int global_work_size_y = context->output_channel_blocks * (context->wtc->tile_input_size * context->wtc->tile_input_size);
	size_t global_work_size[] = {global_work_size_x, global_work_size_y};
	clEnqueueNDRangeKernel(wrapper.command_queue, context->kernel, work_dim, NULL, global_work_size,
		NULL, 0, NULL, &event);

#ifdef NDEBUG	
	static float total = 0;
	cl_ulong start, end;
	clFinish(wrapper.command_queue);
	errcode  = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
	errcode |= clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
	float duration = (end - start) * 1e-6f;
	total += duration;
	LOGD("36x([%dx%d]x[%dx%d])\n", context->wtc->transformed_weight_image_height, context->wtc->transformed_weight_image_width / 36,
		context->itc->transformed_input_image_height / 36, context->itc->transformed_input_image_width);
	LOGD("GPU, matrix_multiply[%dx%d]: %fms, total %fms\n", global_work_size[0], global_work_size[1], duration, total);
#endif
	clReleaseEvent(event);
	
	if (!output) return;
	size_t output_image_origin[] = {0, 0, 0};
	size_t output_image_region[] = {context->output_image_width, context->output_image_height, 1};
	size_t image_row_pitch, image_slice_pitch;
	MEM_MAP_PTR_TYPE *h_output = clEnqueueMapImage(wrapper.command_queue, context->d_output, CL_TRUE, CL_MAP_READ,
		output_image_origin, output_image_region, &image_row_pitch, &image_slice_pitch, 0, NULL, NULL, &errcode);
	image_row_pitch = image_row_pitch / sizeof(MEM_MAP_PTR_TYPE);
	const int dst_row_pitch = context->output_image_width << 2;
	for (int y = 0; y < context->output_image_height; ++y) {
		for (int x = 0; x < dst_row_pitch; ++x) {
			output[y * dst_row_pitch + x] = DEVICE_TO_HOST(h_output[y * image_row_pitch + x]);
		}
	}
	clEnqueueUnmapMemObject(wrapper.command_queue, context->d_output, h_output, 0, NULL, NULL);
}

void free_matrix_multiplication_context(matrix_multiplication_context *context)
{
	if (context) {
		free(context->program_buffer);
		clReleaseMemObject(context->d_output);
		clReleaseProgram(context->program);
		clReleaseKernel(context->kernel);
		free(context);
	}
}

output_inverse_transform_context *create_output_inverse_transform_context(matrix_multiplication_context *mmc, ACTIVATION act)
{	
	output_inverse_transform_context *context = calloc(1, sizeof(output_inverse_transform_context));
	if (!context) {
		fprintf(stderr, "calloc fail[%s:%d].\n", __FILE__, __LINE__);
		return context;
	}
	
	context->mmc = mmc;
	char options[256] = "-cl-fast-relaxed-math -I.";
	switch (act) {
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
	
	context->kernel = cl_make_wrapper_kernel(wrapper, context->program, "inverse_output_transform_f4x4_3x3", &errcode);
	if (CL_SUCCESS != errcode) {
		fprintf(stderr, "cl_make_wrapper_kernel[%s:%d:%d].\n", __FILE__, __LINE__, errcode);
		goto cleanup;
	}

	context->inverse_transformed_output_image_width = mmc->output_channel_blocks * mmc->itc->input_width;
	context->inverse_transformed_output_image_height = mmc->itc->input_height;
	
	cl_mem_flags mem_flags = CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR;
	cl_image_format image_format = {
		.image_channel_order = CL_RGBA,
		.image_channel_data_type = IMAGE_CHANNEL_DATA_TYPE
	};

	cl_image_desc image_desc;
	memset(&image_desc, 0, sizeof(cl_image_desc));
	image_desc.image_type = CL_MEM_OBJECT_IMAGE2D,
	image_desc.image_width = context->inverse_transformed_output_image_width;
	image_desc.image_height = context->inverse_transformed_output_image_height;

	context->d_inverse_transformed_output = clCreateImage(wrapper.context, mem_flags, &image_format, &image_desc, NULL, &errcode);
	if (CL_SUCCESS != errcode) {
		fprintf(stderr, "clCreateBuffer fail[%s:%d:%d].\n", __FILE__, __LINE__, errcode);
		cleanup:free_output_inverse_transform_context(context);
		return 0;
	}
	
	return context;
}	

void get_inverse_transformed_output_image_size(output_inverse_transform_context *context, int *width, int *height)
{
	if (context) {
		*width = context->inverse_transformed_output_image_width;
		*height = context->inverse_transformed_output_image_height;
	} else {
		fprintf(stderr, "invalid inverse output transform context[%s:%d]!\n", __FILE__, __LINE__);
		*width = 0;
		*height = 0;
	}
}

void inverse_transform_output(output_inverse_transform_context *context, float *inverse_transformed_output)
{
	cl_int errcode;
	errcode  = clSetKernelArg(context->kernel, 0, sizeof(cl_mem), &context->mmc->d_output);
	errcode |= clSetKernelArg(context->kernel, 1, sizeof(cl_mem), &context->mmc->wtc->d_biases);
	errcode |= clSetKernelArg(context->kernel, 2, sizeof(cl_mem), &context->d_inverse_transformed_output);
	errcode |= clSetKernelArg(context->kernel, 3, sizeof(int), &context->mmc->itc->ntilesX);
	errcode |= clSetKernelArg(context->kernel, 4, sizeof(int), &context->mmc->itc->input_width);
	errcode |= clSetKernelArg(context->kernel, 5, sizeof(int), &context->mmc->itc->input_height);
	if (CL_SUCCESS != errcode) {
		fprintf(stderr, "clSetKernelArg fail[%s:%d:%d].\n", __FILE__, __LINE__, errcode);
	}
	
	cl_event event;
	cl_uint work_dim = 2;
	const int ntiles = context->mmc->itc->ntilesX * context->mmc->itc->ntilesY;
	const int output_channel_blocks = context->mmc->output_channel_blocks;
	size_t global_work_size[] = {ntiles, output_channel_blocks};
	clEnqueueNDRangeKernel(wrapper.command_queue, context->kernel, work_dim, NULL, global_work_size,
		NULL, 0, NULL, &event);
	
#ifdef NDEBUG	
	static float total = 0;
	cl_ulong start, end;
	clFinish(wrapper.command_queue);
	errcode  = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
	errcode |= clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
	float duration = (end - start) * 1e-6f;
	total += duration;
	LOGD("GPU, inverse_output_transform_f4x4_3x3[%dx%d]: %fms, total %fms\n", global_work_size[0], global_work_size[1], duration, total);
#endif
	clReleaseEvent(event);	
	
	if (!inverse_transformed_output) return;
	size_t origin[] = {0, 0, 0};
	size_t region[] = {context->inverse_transformed_output_image_width, context->inverse_transformed_output_image_height, 1};
	size_t image_row_pitch, image_slice_pitch;
	MEM_MAP_PTR_TYPE *h_inverse_transformed_output = clEnqueueMapImage(wrapper.command_queue, context->d_inverse_transformed_output,
		CL_TRUE, CL_MAP_READ, origin, region, &image_row_pitch, &image_slice_pitch, 0, NULL, NULL, &errcode);
	image_row_pitch = image_row_pitch / sizeof(MEM_MAP_PTR_TYPE);
	const int dst_row_pitch = context->inverse_transformed_output_image_width << 2;
	for (int y = 0; y < context->inverse_transformed_output_image_height; ++y) {
		for (int x = 0; x < dst_row_pitch; ++x) {
			inverse_transformed_output[y * dst_row_pitch + x] = DEVICE_TO_HOST(h_inverse_transformed_output[y * image_row_pitch + x]);
		}
	}
	clEnqueueUnmapMemObject(wrapper.command_queue, context->d_inverse_transformed_output, h_inverse_transformed_output, 0, NULL, NULL);
}

void free_output_inverse_transform_context(output_inverse_transform_context *context)
{
	if (context) {
		free(context->program_buffer);
		clReleaseMemObject(context->d_inverse_transformed_output);
		clReleaseProgram(context->program);
		clReleaseKernel(context->kernel);
		free(context);
	}
}

void set_winograd_convolution_input(input_transform_context *context, void *input)
{
	context->d_input = input;
}

void *get_winograd_convolution_output(output_inverse_transform_context *context)
{
	return context->d_inverse_transformed_output;
}
#endif