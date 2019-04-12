#include <string.h>
#include "resample_layer.h"

#ifdef OPENCL
extern cl_wrapper wrapper;
extern char BINARY_FILENAME_TO_START(cl_common, h);
extern char BINARY_FILENAME_TO_END(cl_common, h);
extern char BINARY_FILENAME_TO_START(resample, cl);
extern char BINARY_FILENAME_TO_END(resample, cl);

struct resample_context {
	char *program_buffer;
	cl_program program;
	cl_kernel kernel;
	cl_mem d_input;
	cl_mem d_output;
	cl_int stride;
	cl_int channel_blocks;
	cl_int input_image_width;
	cl_int input_image_height;
	cl_int output_image_width;
	cl_int output_image_height;
};

static resample_context *create_resample_context(resample_layer *layer);
static void forward_resample_layer_gpu(resample_layer *layer);
static void free_resample_context(resample_context *context);
#endif

void *make_resample_layer(dim3 input_size, int batch_size, int stride, dim3 *output_size)
{
	resample_layer *layer = calloc(1, sizeof(resample_layer));
	if (!layer) {
		fprintf(stderr, "calloc[%s:%d].\n", __FILE__, __LINE__);
		return layer;
	}
	
	layer->type = RESAMPLE;
	layer->input_size = input_size;
	if (stride > 0) {
		layer->output_size.w = input_size.w * stride;
		layer->output_size.h = input_size.h * stride;
		layer->stride = stride;
		layer->upsample = 1;
	} else {
		layer->output_size.w = input_size.w / stride;
		layer->output_size.h = input_size.h / stride;
		layer->stride = -stride;
		layer->upsample = 0;
	}
	
	layer->output_size.c = input_size.c;
	layer->batch_size = batch_size;
	layer->ninputs = input_size.w * input_size.h * input_size.c;
	layer->noutputs = layer->output_size.w * layer->output_size.h * layer->output_size.c;
	layer->input = NULL;
	layer->output = NULL;
#ifdef OPENCL
	layer->rc = create_resample_context(layer);
	if (!layer->rc) {
		goto cleanup;
	}
#endif
	
	layer->output = calloc(layer->noutputs * batch_size, sizeof(float));
	if (!layer->output) {
		fprintf(stderr, "calloc[%s:%d].\n", __FILE__, __LINE__);
#ifdef OPENCL
		cleanup:
#endif
		free_resample_layer(layer);
		return 0;
	}
	
	if (output_size) {
		output_size->w = layer->output_size.w;
		output_size->h = layer->output_size.h;
		output_size->c = layer->output_size.c;
	}
	
	return layer;
}

void free_resample_layer(void *_layer)
{
	resample_layer *layer = (resample_layer *)_layer;
	if (!layer) return;
	
	if (layer->output) {
		free(layer->output);
		layer->output = NULL;
	}

#ifdef OPENCL
	free_resample_context(layer->rc);
#endif
	
	free(layer);
	layer = NULL;
}

void print_resample_layer_info(void *_layer, int id)
{
	resample_layer *layer = (resample_layer *)_layer;
	printf("%2d\tresample\t%4d x%4d x%4d\t\t%d\t\t\t\t%4d x%4d x%4d\n",
		id,
		layer->input_size.w,
		layer->input_size.h,
		layer->input_size.c,
		layer->stride,
		layer->output_size.w,
		layer->output_size.h,
		layer->output_size.c);
}

void set_resample_layer_input(void *_layer, void *input)
{
	resample_layer *layer = (resample_layer *)_layer;
#if !defined(OPENCL) || !defined(WINOGRAD_CONVOLUTION)
	layer->input = input;
#else
	layer->rc->d_input = input;
#endif
}

void *get_resample_layer_output(void *_layer)
{
	resample_layer *layer = (resample_layer *)_layer;
#if !defined(OPENCL) || !defined(WINOGRAD_CONVOLUTION)
	return layer->output;
#else
	return layer->rc->d_output;
#endif
}

void forward_resample_layer(void *_layer, znet *net)
{
	resample_layer *layer = (resample_layer *)_layer;
#if defined(OPENCL) && defined(WINOGRAD_CONVOLUTION)
	return forward_resample_layer_gpu(layer);
#endif
	float alpha = 0;
	size_t size = layer->noutputs * layer->batch_size * sizeof(float);
	mset((char *const)layer->output, size, (const char *const)&alpha, sizeof(float));
	
	for (int b = 0; b < layer->batch_size; ++b) {
		float *in = layer->input + b * layer->ninputs;
		float *out = layer->output + b * layer->noutputs;
		if (layer->upsample) {
			upsample(in, layer->input_size.w, layer->input_size.h,
				layer->input_size.c, layer->stride, out);
		} else {
			fprintf(stderr, "Not implemented[%s:%d].\n", __FILE__, __LINE__);
		}
	}
}

void backward_resample_layer(resample_layer *layer, znet *net)
{
	fprintf(stderr, "Not implemented[%s:%d].\n", __FILE__, __LINE__);
}

void upsample(float *in, int width, int height, int nchannels, int stride, float *out)
{
	int us_width = width * stride;
	int us_height = height * stride;
	for (int c = 0; c < nchannels; ++c) {
		for (int y = 0; y < us_height; ++y) {
			for (int x = 0; x < us_width; ++x) {
				int y0 = y / stride;
				int x0 = x / stride;
				float val = in[c * width * height + y0 * width + x0];
				out[c * us_width * us_height + y * us_width + x] = val;
			}
		}
	}
}

#ifdef OPENCL
void get_resample_output_image_size(resample_layer *layer, int *width, int *height)
{
	if (layer) {
		*width = layer->rc->output_image_width;
		*height = layer->rc->output_image_height;
	} else {
		*width = 0;
		*height = 0;
	}
}

resample_context *create_resample_context(resample_layer *layer)
{
	resample_context *context = calloc(1, sizeof(resample_context));
	if (!context) {
		fprintf(stderr, "calloc fail[%s:%d].\n", __FILE__, __LINE__);
		return context;
	}
	
	context->stride = layer->stride;
	context->channel_blocks = round_up_division_4(layer->input_size.c);
	context->input_image_width = context->channel_blocks * layer->input_size.w;
	context->input_image_height = layer->input_size.h;
	context->output_image_width = context->channel_blocks * layer->output_size.w;
	context->output_image_height = layer->output_size.h;

	size_t header_size = (size_t)(&BINARY_FILENAME_TO_END(cl_common, h) - &BINARY_FILENAME_TO_START(cl_common, h));
	size_t size = (size_t)(&BINARY_FILENAME_TO_END(resample, cl) - &BINARY_FILENAME_TO_START(resample, cl));
	context->program_buffer = calloc(header_size + size + 1, sizeof(char));
	if (!context->program_buffer) {
		fprintf(stderr, "calloc fail[%s:%d].\n", __FILE__, __LINE__);
		goto cleanup;
	}
	
	memcpy(context->program_buffer, &BINARY_FILENAME_TO_START(cl_common, h), header_size);
	memcpy(context->program_buffer + header_size, &BINARY_FILENAME_TO_START(resample, cl), size);
	context->program_buffer[header_size + size] = '\0';
	
	cl_int errcode;
	char options[256] = "-cl-fast-relaxed-math -I.";
	PARSE_PRECISION;
	context->program = cl_make_wrapper_program(wrapper, "resample.cl", context->program_buffer, options, &errcode);
	if (CL_SUCCESS != errcode) {
		fprintf(stderr, "cl_make_wrapper_program[%s:%d:%d].\n", __FILE__, __LINE__, errcode);
		goto cleanup;
	}
	
	if (layer->upsample) {
		context->kernel = cl_make_wrapper_kernel(wrapper, context->program, "upsampleB1", &errcode);
	} else {
		fprintf(stderr, "Not implemented[%s:%d]!\n", __FILE__, __LINE__);
		goto cleanup;
	}
	if (CL_SUCCESS != errcode) {
		fprintf(stderr, "cl_make_wrapper_kernel[%s:%d:%d].\n", __FILE__, __LINE__, errcode);
		goto cleanup;
	}
	
	cl_mem_flags mem_flags = CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR;
	cl_image_format image_format = {
		.image_channel_order = CL_RGBA,
		.image_channel_data_type = IMAGE_CHANNEL_DATA_TYPE
	};
	
	cl_image_desc image_desc;
	memset(&image_desc, 0, sizeof(cl_image_desc));
	image_desc.image_type = CL_MEM_OBJECT_IMAGE2D;
	image_desc.image_width = context->output_image_width;
	image_desc.image_height = context->output_image_height;
	
	context->d_output = clCreateImage(wrapper.context, mem_flags, &image_format, &image_desc, NULL, &errcode);
	if (CL_SUCCESS != errcode) {
		fprintf(stderr, "clCreateImage fail[%s:%d:%d].\n", __FILE__, __LINE__, errcode);
		cleanup:free_resample_context(context);
		return 0;
	}
	
	return context;
}

void forward_resample_layer_gpu(resample_layer *layer)
{
	cl_int errcode;
	resample_context *context = layer->rc;
	errcode  = clSetKernelArg(context->kernel, 0, sizeof(cl_mem), &context->d_input);
	errcode |= clSetKernelArg(context->kernel, 1, sizeof(cl_mem), &context->d_output);
	errcode |= clSetKernelArg(context->kernel, 2, sizeof(int), &context->stride);
	errcode |= clSetKernelArg(context->kernel, 3, sizeof(int), &layer->input_size.w);
	errcode |= clSetKernelArg(context->kernel, 4, sizeof(int), &layer->output_size.w);
	if (CL_SUCCESS != errcode) {
		fprintf(stderr, "clSetKernelArg fail[%s:%d:%d].\n", __FILE__, __LINE__, errcode);
		return;
	}
	
	cl_event event;
	cl_uint work_dim = 2;
	size_t global_work_size[] = {context->output_image_width, context->output_image_height};
	clEnqueueNDRangeKernel(wrapper.command_queue, context->kernel, work_dim, NULL, global_work_size,
		NULL, 0, NULL, &event);

#ifdef NDEBUG	
	cl_ulong start, end;
	clFinish(wrapper.command_queue);
	errcode  = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
	errcode |= clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
	float duration = (end - start) * 1e-6f;
	LOGD("GPU, upsample: %fms.\n", duration);
#endif
	clReleaseEvent(event);
}

void free_resample_context(resample_context *context)
{
	if (context) {
		free(context->program_buffer);
		clReleaseMemObject(context->d_output);
		clReleaseProgram(context->program);
		clReleaseKernel(context->kernel);
		free(context);
	}
}
#endif