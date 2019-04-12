#include <stdarg.h>
#include <string.h>
#include "route_layer.h"
#include "convolutional_layer.h"
#include "resample_layer.h"

static int parse_input_layer(void *layer, dim3 *output_size);

#ifdef OPENCL
extern cl_wrapper wrapper;
#endif
#if defined(OPENCL) && defined(WINOGRAD_CONVOLUTION)
static void forward_route_layer_gpu(route_layer *layer, znet *net);
#endif

void *make_route_layer(int batch_size, int nroutes, void *layers[], int *layer_id, dim3 *output_size)
{
	route_layer *layer = calloc(1, sizeof(route_layer));
	if (!layer) {
		fprintf(stderr, "calloc[%s:%d].\n", __FILE__, __LINE__);
		return layer;
	}
	
	layer->type = ROUTE;
	layer->output_size.w = 0;
	layer->output_size.h = 0;
	layer->output_size.c = 0;
	layer->batch_size = batch_size;
	layer->ninputs = 0;
	layer->noutputs = 0;
	layer->nroutes = nroutes;
	layer->input_layers = NULL;
	layer->input_sizes = NULL;
	layer->output = NULL;
#ifdef OPENCL
	layer->d_output = 0;
#endif
	
	layer->input_layers = calloc(nroutes, sizeof(int));
	if (!layer->input_layers) {
		fprintf(stderr, "calloc[%s:%d].\n", __FILE__, __LINE__);
		goto cleanup;
	}
	
	layer->input_sizes = calloc(nroutes, sizeof(int));
	if (!layer->input_sizes) {
		fprintf(stderr, "calloc[%s:%d].\n", __FILE__, __LINE__);
		goto cleanup;
	}
	
	for (int i = 0; i < nroutes; ++i) {
		layer->input_layers[i] = layer_id[i];
		layer->input_sizes[i] = parse_input_layer(layers[i], &layer->output_size);
		layer->ninputs += layer->input_sizes[i];
	}
	
	if (output_size) {
		*output_size = layer->output_size;
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
	image_desc.image_width = layer->output_size.w * round_up_division_4(layer->output_size.c);
	image_desc.image_height = layer->output_size.h;
	
	cl_int errcode;
	layer->d_output = clCreateImage(wrapper.context, mem_flags, &image_format, &image_desc, NULL, &errcode);
	if (CL_SUCCESS != errcode) {
		fprintf(stderr, "clCreateImage fail[%s:%d:%d].\n", __FILE__, __LINE__, errcode);
		goto cleanup;
	}
#endif

	layer->noutputs = layer->ninputs;
	layer->output = calloc(layer->noutputs * batch_size, sizeof(float));
	if (!layer->output) {
		fprintf(stderr, "calloc[%s:%d].\n", __FILE__, __LINE__);
		cleanup:free_route_layer(layer);
	}
	
	return layer;
}

void free_route_layer(void *_layer)
{
	route_layer *layer = (route_layer *)_layer;
	if (!layer) return;
	
	if (layer->input_layers) {
		free(layer->input_layers);
		layer->input_layers = NULL;
	}
	
	if (layer->input_sizes) {
		free(layer->input_sizes);
		layer->input_sizes = NULL;
	}
	
	if (layer->output) {
		free(layer->output);
		layer->output = NULL;
	}
	
#ifdef OPENCL
	clReleaseMemObject(layer->d_output);
#endif
	
	free(layer);
	layer = NULL;
}

void print_route_layer_info(void *_layer, int id)
{
	route_layer *layer = (route_layer *)_layer;
	printf("%02d\troute ", id);
	for (int i = 0; i < layer->nroutes; ++i) {
		printf("%d", layer->input_layers[i] + 1);
		if (i < layer->nroutes - 1) printf(",");
	}
	printf("\n");
}

void set_route_layer_input(void *_layer, void *input)
{
	;
}

void *get_route_layer_output(void *_layer)
{
	route_layer *layer = (route_layer *)_layer;
#if !defined(OPENCL) || !defined(WINOGRAD_CONVOLUTION)
	return layer->output;
#else
	return layer->d_output;
#endif
}

void forward_route_layer(void *_layer, znet *net)
{
	route_layer *layer = (route_layer *)_layer;
#if defined(OPENCL) && defined(WINOGRAD_CONVOLUTION)
	return forward_route_layer_gpu(layer, net);
#endif	
	int offset = 0;
	void **layers = znet_layers(net);
	for (int r = 0; r < layer->nroutes; ++r) {
		LAYER_TYPE type = *(LAYER_TYPE *)(layers[layer->input_layers[r]]);		
		if (type == CONVOLUTIONAL) {
			convolutional_layer *input_layer = (convolutional_layer *)layers[layer->input_layers[r]];
			for (int b = 0; b < layer->batch_size; ++b) {
				float *X = input_layer->output + b * layer->input_sizes[r];
				float *Y = layer->output + b * layer->noutputs + offset;
				mcopy((const char *const)X, (char *const)Y, layer->input_sizes[r] * sizeof(float));
			}
			offset += layer->input_sizes[r];
		} else if (type == RESAMPLE) {
			resample_layer *input_layer = (resample_layer *)layers[layer->input_layers[r]];
			for (int b = 0; b < layer->batch_size; ++b) {
				float *X = input_layer->output + b * layer->input_sizes[r];
				float *Y = layer->output + b * layer->noutputs + offset;
				mcopy((const char *const)X, (char *const)Y, layer->input_sizes[r] * sizeof(float));
			}
			offset += layer->input_sizes[r];
		} else {
			fprintf(stderr, "Not implemented[%s:%d].\n", __FILE__, __LINE__);
		}
	}
}

void backward_route_layer(route_layer *layer, znet *net)
{
	fprintf(stderr, "Not implemented[%s:%d].\n", __FILE__, __LINE__);
}

int parse_input_layer(void *layer, dim3 *output_size)
{
	LAYER_TYPE type = *(LAYER_TYPE *)layer;
	if (type == CONVOLUTIONAL) {
		convolutional_layer *l = (convolutional_layer *)layer;
		output_size->w  = l->output_size.w;
		output_size->h  = l->output_size.h;
		output_size->c += l->output_size.c;
		return l->noutputs;
	} else if (type == RESAMPLE) {
		resample_layer *l = (resample_layer *)layer;
		output_size->w  = l->output_size.w;
		output_size->h  = l->output_size.h;
		output_size->c += l->output_size.c;
		return l->noutputs;
	} else {
		fprintf(stderr, "Not implemented[%s:%d].\n", __FILE__, __LINE__);
		return 0;
	}
}

#if defined(OPENCL) && defined(WINOGRAD_CONVOLUTION)
void forward_route_layer_gpu(route_layer *layer, znet *net)
{
	void **layers = znet_layers(net);
	int row_start = 0;
	for (int r = 0; r < layer->nroutes; ++r) {
		LAYER_TYPE type = *(LAYER_TYPE *)(layers[layer->input_layers[r]]);		
		if (type == CONVOLUTIONAL) {
			convolutional_layer *input_layer = (convolutional_layer *)layers[layer->input_layers[r]];
			cl_mem src_image = get_convolutional_layer_output(input_layer);
			int src_image_width, src_image_height;
			if (3 == input_layer->filter_size) {
				get_inverse_transformed_output_image_size(input_layer->oitc, &src_image_width, &src_image_height);
			} else if (1 == input_layer->filter_size) {
				get_direct_convolution_output_image_size(input_layer, &src_image_width, &src_image_height);
			}
			size_t src_origion[] = {0, 0, 0};
			size_t dst_origion[] = {row_start, 0, 0};
			size_t region[] = {src_image_width, src_image_height, 1};
			clEnqueueCopyImage(wrapper.command_queue, src_image, layer->d_output, src_origion, dst_origion,
				region, 0, NULL, NULL);
			row_start += src_image_width;
		} else if (type == RESAMPLE) {
			resample_layer *input_layer = (resample_layer *)layers[layer->input_layers[r]];
			cl_mem src_image = get_resample_layer_output(input_layer);
			int src_image_width, src_image_height;
			get_resample_output_image_size(input_layer, &src_image_width, &src_image_height);
			size_t src_origion[] = {0, 0, 0};
			size_t dst_origion[] = {row_start, 0, 0};
			size_t region[] = {src_image_width, src_image_height, 1};
			clEnqueueCopyImage(wrapper.command_queue, src_image, layer->d_output, src_origion, dst_origion,
				region, 0, NULL, NULL);
			row_start += src_image_width;
		} else {
			fprintf(stderr, "Not implemented[%s:%d].\n", __FILE__, __LINE__);
		}
	}
}
#endif