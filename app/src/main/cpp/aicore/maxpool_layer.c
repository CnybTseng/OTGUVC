#include <omp.h>
#include <float.h>
#include <string.h>
#include "maxpool_layer.h"
#ifdef OPENCL
#	include "cl_wrapper.h"
#endif

#ifdef NNPACK
struct maxpool_thread_param {
	void *layer;
	znet *net;
};

static void maxpool_thread(struct maxpool_thread_param *param, size_t batch_size, size_t nchannels);
#endif

#ifdef OPENCL
extern cl_wrapper wrapper;
extern char BINARY_FILENAME_TO_START(cl_common, h);
extern char BINARY_FILENAME_TO_END(cl_common, h);
extern char BINARY_FILENAME_TO_START(maxpool, cl);
extern char BINARY_FILENAME_TO_END(maxpool, cl);

struct maxpool_gpu_context {
	char *program_buffer;
	cl_program program;
	cl_kernel kernel;
	cl_mem d_input;
	cl_mem d_output;
	int channel_blocks;
	int input_image_width;
	int input_image_height;
	int output_image_width;
	int output_image_height;
};

static maxpool_gpu_context *create_maxpool_gpu_context(maxpool_layer *layer);
static void forward_maxpool_layer_gpu(maxpool_layer *layer);
static void free_maxpool_gpu_context(maxpool_gpu_context *context);
#endif

void *make_maxpool_layer(dim3 input_size, int filter_size, int stride, int padding, int batch_size,
                         dim3 *output_size)
{
	maxpool_layer *layer = calloc(1, sizeof(maxpool_layer));
	if (!layer) {
		fprintf(stderr, "calloc[%s:%d].\n", __FILE__, __LINE__);
		return layer;
	}
	
	layer->type = MAXPOOL;
	layer->input_size = input_size;
	layer->filter_size = filter_size;
	layer->stride = stride;
	layer->padding = padding;
	layer->output_size.w = maxpool_output_width(layer);
	layer->output_size.h = maxpool_output_height(layer);
	layer->output_size.c = input_size.c;
	layer->batch_size = batch_size;
	layer->ninputs = input_size.w * input_size.h * input_size.c;
	layer->noutputs = layer->output_size.w * layer->output_size.h * layer->output_size.c;
	layer->input = NULL;
	layer->output = NULL;
#ifdef OPENCL
	layer->mpgc = NULL;
#endif
	
	if (output_size) {
		*output_size = layer->output_size;
	}
	
#ifdef OPENCL
	layer->mpgc = create_maxpool_gpu_context(layer);
	if (!layer->mpgc) {
		goto cleanup;
	}
#endif	
	
	layer->output = calloc(layer->noutputs * batch_size, sizeof(float));
	if (!layer->output) {
		fprintf(stderr, "calloc[%s:%d].\n", __FILE__, __LINE__);
#ifdef OPENCL
		cleanup:
#endif
		free_maxpool_layer(layer);
	}
	
	return (void *)layer;
}					 
						 
void free_maxpool_layer(void *_layer)
{
	maxpool_layer *layer = (maxpool_layer *)_layer;
	if (!layer) return;
	
	if (layer->output) {
		free(layer->output);
		layer->output = NULL;
	}
	
#ifdef OPENCL
	free_maxpool_gpu_context(layer->mpgc);
#endif	
	
	free(layer);
	layer = NULL;
}

void print_maxpool_layer_info(void *_layer, int id)
{
	maxpool_layer *layer = (maxpool_layer *)_layer;
	printf("%2d\tmaxpool\t\t%4d x%4d x%4d\t\t%dx%d/%d\t\t%4d\t\t%4d x%4d x%4d\n",
		id,
		layer->input_size.w,
		layer->input_size.h,
		layer->input_size.c,
		layer->filter_size,
		layer->filter_size,
		layer->stride,
		layer->input_size.c,
		layer->output_size.w,
		layer->output_size.h,
		layer->output_size.c);
}

void set_maxpool_layer_input(void *_layer, void *input)
{
	maxpool_layer *layer = (maxpool_layer *)_layer;
#if !defined(OPENCL) || !defined(WINOGRAD_CONVOLUTION)
	layer->input = input;
#else
	layer->mpgc->d_input = input;
#endif
}

void *get_maxpool_layer_output(void *_layer)
{
	maxpool_layer *layer = (maxpool_layer *)_layer;
#if !defined(OPENCL) || !defined(WINOGRAD_CONVOLUTION)
	return layer->output;
#else
	return layer->mpgc->d_output;
#endif
}

void forward_maxpool_layer(void *_layer, znet *net)
{	
	maxpool_layer *layer = (maxpool_layer *)_layer;
#ifdef NNPACK
	struct maxpool_thread_param param = {_layer, net};
	return pthreadpool_compute_2d(znet_threadpool(net), (pthreadpool_function_2d_t)maxpool_thread,
		&param, layer->batch_size, layer->output_size.c);
#endif
#if defined(OPENCL) && defined(WINOGRAD_CONVOLUTION)
	return forward_maxpool_layer_gpu(layer);
#endif
	int offsetx = -layer->padding / 2;
	int offsety = -layer->padding / 2;
	int inwh = layer->input_size.w * layer->input_size.h;
	int outwh = layer->output_size.w * layer->output_size.h;
	
	for (int b = 0; b < layer->batch_size; ++b) {
		#pragma omp parallel for
		for (int c = 0; c < layer->output_size.c; ++c) {
			int dslice = (b * layer->output_size.c + c) * outwh;
			int dslice0 = (b * layer->input_size.c + c) * inwh;
			for (int y = 0; y < layer->output_size.h; ++y) {
				for (int x = 0; x < layer->output_size.w; ++x) {
					int maxidx = -1;
					float maxval = -FLT_MAX;
					for (int dy = 0; dy < layer->filter_size; ++dy) {
						for (int dx = 0; dx < layer->filter_size; ++dx) {
							int x0 = x * layer->stride + dx + offsetx;
							int y0 = y * layer->stride + dy + offsety;
							int idx0 = dslice0 + y0 * layer->input_size.w + x0;
							int valid = x0 > -1 && x0 < layer->input_size.w &&
								y0 > -1 && y0 < layer->input_size.h;
							float val = valid ? layer->input[idx0] : -FLT_MAX;
							int bigger = val > maxval;
							maxidx = bigger ? idx0 : maxidx;
							maxval = bigger ? val : maxval;
						}
					}
					
					int idx = dslice + y * layer->output_size.w + x;
					layer->output[idx] = maxval;
				}
			}
		}
	}
}

void backward_maxpool_layer(maxpool_layer *layer, znet *net)
{
	fprintf(stderr, "Not implemented[%s:%d].\n", __FILE__, __LINE__);
}

int maxpool_output_width(maxpool_layer *layer)
{
	return (layer->input_size.w - layer->filter_size + layer->padding) / layer->stride + 1;
}

int maxpool_output_height(maxpool_layer *layer)
{
	return (layer->input_size.h - layer->filter_size + layer->padding) / layer->stride + 1;
}

#ifdef NNPACK
void maxpool_thread(struct maxpool_thread_param *param, size_t batch_size, size_t nchannels)
{
	maxpool_layer *layer = (maxpool_layer *)param->layer;
	int offsetx = -layer->padding / 2;
	int offsety = -layer->padding / 2;
	int inwh = layer->input_size.w * layer->input_size.h;
	int outwh = layer->output_size.w * layer->output_size.h;
	
	int dslice = (batch_size * layer->output_size.c + nchannels) * outwh;
	int dslice0 = (batch_size * layer->input_size.c + nchannels) * inwh;
	for (int y = 0; y < layer->output_size.h; ++y) {
		for (int x = 0; x < layer->output_size.w; ++x) {
			int maxidx = -1;
			float maxval = -FLT_MAX;
			for (int dy = 0; dy < layer->filter_size; ++dy) {
				for (int dx = 0; dx < layer->filter_size; ++dx) {
					int x0 = x * layer->stride + dx + offsetx;
					int y0 = y * layer->stride + dy + offsety;
					int idx0 = dslice0 + y0 * layer->input_size.w + x0;
					int valid = x0 > -1 && x0 < layer->input_size.w &&
						y0 > -1 && y0 < layer->input_size.h;
					float val = valid ? layer->input[idx0] : -FLT_MAX;
					int bigger = val > maxval;
					maxidx = bigger ? idx0 : maxidx;
					maxval = bigger ? val : maxval;
				}
			}
			
			int idx = dslice + y * layer->output_size.w + x;
			layer->output[idx] = maxval;
		}
	}
}
#endif

#ifdef OPENCL
maxpool_gpu_context *create_maxpool_gpu_context(maxpool_layer *layer)
{
	maxpool_gpu_context *context = calloc(1, sizeof(maxpool_gpu_context));
	if (!context) {
		fprintf(stderr, "calloc fail[%s:%d].\n", __FILE__, __LINE__);
		return context;
	}
	
	context->program = 0;
	context->kernel = 0;
	context->d_input = 0;
	context->d_output = 0;
	context->channel_blocks = (layer->input_size.c + 3) >> 2;
	context->input_image_width = layer->input_size.w * context->channel_blocks;
	context->input_image_height = layer->input_size.h;
	context->output_image_width = layer->output_size.w * context->channel_blocks;
	context->output_image_height = layer->output_size.h;

	size_t header_size = (size_t)(&BINARY_FILENAME_TO_END(cl_common, h) - &BINARY_FILENAME_TO_START(cl_common, h));
	size_t size = (size_t)(&BINARY_FILENAME_TO_END(maxpool, cl) - &BINARY_FILENAME_TO_START(maxpool, cl));
	context->program_buffer = calloc(header_size + size + 1, sizeof(char));
	if (!context->program_buffer) {
		fprintf(stderr, "calloc fail[%s:%d].\n", __FILE__, __LINE__);
		goto cleanup;
	}
	
	memcpy(context->program_buffer, &BINARY_FILENAME_TO_START(cl_common, h), header_size);
	memcpy(context->program_buffer + header_size, &BINARY_FILENAME_TO_START(maxpool, cl), size);
	context->program_buffer[header_size + size] = '\0';
	
	cl_int errcode;
	char options[256] = "-I.";
	PARSE_PRECISION;
	context->program = cl_make_wrapper_program(wrapper, "maxpool.cl", context->program_buffer, options, &errcode);
	if (CL_SUCCESS != errcode) {
		fprintf(stderr, "cl_make_wrapper_program[%s:%d:%d].\n", __FILE__, __LINE__, errcode);
		goto cleanup;
	}
	
	context->kernel = cl_make_wrapper_kernel(wrapper, context->program, "maxpool_2x2", &errcode);
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
		cleanup:free_maxpool_gpu_context(context);
		return 0;
	}
	
	return context;
}

void forward_maxpool_layer_gpu(maxpool_layer *layer)
{
	cl_int errcode;
	maxpool_gpu_context *context = layer->mpgc;
	errcode  = clSetKernelArg(context->kernel, 0, sizeof(cl_mem), &context->d_input);
	errcode |= clSetKernelArg(context->kernel, 1, sizeof(cl_mem), &context->d_output);
	errcode |= clSetKernelArg(context->kernel, 2, sizeof(int), &layer->input_size.w);
	errcode |= clSetKernelArg(context->kernel, 3, sizeof(int), &layer->input_size.h);
	errcode |= clSetKernelArg(context->kernel, 4, sizeof(int), &layer->output_size.w);
	errcode |= clSetKernelArg(context->kernel, 5, sizeof(int), &layer->output_size.h);
	errcode |= clSetKernelArg(context->kernel, 6, sizeof(int), &layer->padding);
	errcode |= clSetKernelArg(context->kernel, 7, sizeof(int), &layer->stride);
	if (CL_SUCCESS != errcode) {
		fprintf(stderr, "clSetKernelArg fail[%s:%d].\n", __FILE__, __LINE__);
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
	LOGD("GPU, maxpool_2x2: %fms.\n", duration);
#endif
	clReleaseEvent(event);
}

void free_maxpool_gpu_context(maxpool_gpu_context *context)
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