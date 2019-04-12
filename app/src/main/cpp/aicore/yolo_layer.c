#include <omp.h>
#include <math.h>
#include <string.h>
#include "yolo_layer.h"
#include "activation.h"
#include "list.h"

#ifdef OPENCL
extern cl_wrapper wrapper;
#endif

static box get_yolo_box(float *box_volume, int id, int layer_width, int layer_height,
                        int net_width, int net_height, int *anchor_box,
						int image_width, int image_height);
static float *get_yolo_prob(float *prob_volume, int id, int layer_width, int layer_height,
                            int classes, float objectness);

void *make_yolo_layer(dim3 input_size, int batch_size, int nscales, int total_scales, int classes, int *mask,
                      int *anchor_boxes)
{
	yolo_layer *layer = calloc(1, sizeof(yolo_layer));
	if (!layer) {
		fprintf(stderr, "calloc[%s:%d].\n", __FILE__, __LINE__);
		return layer;
	}
	
	layer->type = YOLO;
	layer->input_size = input_size;
	layer->output_size = layer->input_size;
	layer->batch_size = batch_size;
	layer->ninputs = input_size.w * input_size.h * input_size.c;
	layer->noutputs = layer->ninputs;
	layer->nscales = nscales;
	layer->total_scales = total_scales;
	layer->classes = classes;
	layer->mask = NULL;
	layer->anchor_boxes = NULL;
	layer->input = NULL;
	layer->output = NULL;
	
	layer->mask = calloc(nscales, sizeof(int));
	if (!layer->mask) {
		fprintf(stderr, "calloc[%s:%d].\n", __FILE__, __LINE__);
		goto cleanup;
	}
	
	for (int i = 0; i < nscales; ++i) {
		layer->mask[i] = mask ? mask[i] : i;
	}
	
	layer->anchor_boxes = calloc(total_scales * 2, sizeof(int));
	if (!layer->anchor_boxes) {
		fprintf(stderr, "calloc[%s:%d].\n", __FILE__, __LINE__);
		goto cleanup;
	}
		
	for (int i = 0; i < total_scales; ++i) {
		layer->anchor_boxes[2 * i] = anchor_boxes[2 * i];
		layer->anchor_boxes[2 * i + 1] = anchor_boxes[2 * i + 1];
	}
	
	layer->output = calloc(layer->ninputs * batch_size, sizeof(float));
	if (!layer->output) {
		fprintf(stderr, "calloc[%s:%d].\n", __FILE__, __LINE__);
		cleanup:free_yolo_layer(layer);
	}
	
	return (void *)layer;
}

void free_yolo_layer(void *_layer)
{
	yolo_layer *layer = (yolo_layer *)_layer;
	if (!layer) return;
	
	if (layer->mask) {
		free(layer->mask);
		layer->mask = NULL;
	}
	
	if (layer->anchor_boxes) {
		free(layer->anchor_boxes);
		layer->anchor_boxes = NULL;
	}
	
	if (layer->output) {
		free(layer->output);
		layer->output = NULL;
	}
}

void print_yolo_layer_info(void *_layer, int id)
{
	printf("%02d\tyolo\n", id);
}

void set_yolo_layer_input(void *_layer, void *input)
{
	yolo_layer *layer = (yolo_layer *)_layer;
#if !defined(OPENCL) || !defined(WINOGRAD_CONVOLUTION)
	layer->input = input;
#else
	layer->d_input = input;
#endif
}

void *get_yolo_layer_output(void *_layer)
{
	yolo_layer *layer = (yolo_layer *)_layer;
	return layer->output;
}

void forward_yolo_layer(void *_layer, znet *net)
{
	yolo_layer *layer = (yolo_layer *)_layer;
#if !defined(OPENCL) || !defined(WINOGRAD_CONVOLUTION)
	int total = layer->ninputs * layer->batch_size;
	memcpy(layer->output, layer->input, total * sizeof(float));
#else
	cl_int errcode;
	size_t origin[] = {0, 0, 0};
	size_t input_image_width, input_image_height;
	clGetImageInfo(layer->d_input, CL_IMAGE_WIDTH, sizeof(size_t), &input_image_width, NULL);
	clGetImageInfo(layer->d_input, CL_IMAGE_HEIGHT, sizeof(size_t), &input_image_height, NULL);
	size_t region[] = {input_image_width, input_image_height, 1};
	size_t image_row_pitch, image_slice_pitch;
	MEM_MAP_PTR_TYPE *h_input = clEnqueueMapImage(wrapper.command_queue, layer->d_input, CL_TRUE, CL_MAP_WRITE,
		origin, region, &image_row_pitch, &image_slice_pitch, 0, NULL, NULL, &errcode);
	image_row_pitch = image_row_pitch / sizeof(MEM_MAP_PTR_TYPE);
	nhwc_to_nchw(h_input, layer->output, layer->output_size.w, layer->output_size.h,
		layer->output_size.c, 1, image_row_pitch, layer->output_size.w, 4);
	clEnqueueUnmapMemObject(wrapper.command_queue, layer->d_input, h_input, 0, NULL, NULL);
#endif
	int volume_per_scale = layer->output_size.w * layer->output_size.h * (4 + 1 + layer->classes);
	if (znet_workmode(net) == INFERENCE) {
		for (int b = 0; b < layer->batch_size; ++b) {
			#pragma omp parallel for
			for (int s = 0; s < layer->nscales; ++s) {
				float *at = layer->output + b * layer->noutputs + s * volume_per_scale;
				activate(at, 2 * layer->output_size.w * layer->output_size.h, LOGISTIC);
				at += 4 * layer->output_size.w * layer->output_size.h;
				activate(at, (1 + layer->classes) * layer->output_size.w * layer->output_size.h, LOGISTIC);
			}
		}
	} else {
		fprintf(stderr, "Not implemented[%s:%d].\n", __FILE__, __LINE__);
	}
}

void get_yolo_layer_detections(yolo_layer *layer, znet *net, int imgw, int imgh, float thresh, list *l)
{
	int size = layer->output_size.w * layer->output_size.h;
	int volume_per_scale = size * (4 + 1 + layer->classes);
	int width = znet_input_width(net);
	int height = znet_input_height(net);
	for (int s = 0; s < layer->nscales; ++s) {
		float *box_vol = layer->output + s * volume_per_scale;
		float *obj_slc = layer->output + s * volume_per_scale + 4 * size;
		float *prob_vol = layer->output + s * volume_per_scale + 5 * size;
		for (int i = 0; i < size; ++i) {
			if (obj_slc[i] < thresh) continue;
			detection *det = list_alloc_mem(sizeof(detection));
			if (!det) continue;
			det->bbox = get_yolo_box(box_vol, i, layer->output_size.w, layer->output_size.h,
				width, height, &layer->anchor_boxes[2 * layer->mask[s]], imgw, imgh);
			det->classes = layer->classes;
			det->probabilities = get_yolo_prob(prob_vol, i, layer->output_size.w, layer->output_size.h,
				layer->classes, obj_slc[i]);
			det->objectness = obj_slc[i];
			list_add_tail(l, det);
		}
	}
}

void free_yolo_layer_detections(list *l)
{
	if (!l) return;
	node *nd = l->head;
	while (nd) {
		detection *det = (detection *)nd->val;
		if (det->probabilities) {
			list_free_mem(det->probabilities);
			det->probabilities = NULL;
		}
		nd = nd->next;
	}
	
	list_clear(l);
}

void backward_yolo_layer(yolo_layer *layer, znet *net)
{
	fprintf(stderr, "Not implemented[%s:%d].\n", __FILE__, __LINE__);
}

box get_yolo_box(float *box_volume, int id, int layer_width, int layer_height,
                 int net_width, int net_height, int *anchor_box,
				 int image_width, int image_height)
{
	box b;
	int slice_size = layer_width * layer_height;
	b.x = (id % layer_width + box_volume[id]) / layer_width;
	b.y = (id / layer_width + box_volume[id + slice_size]) / layer_height;
	b.w = exp(box_volume[id + 2 * slice_size]) * anchor_box[0] / net_width;
	b.h = exp(box_volume[id + 3 * slice_size]) * anchor_box[1] / net_height;
	
	float sx = net_width / (float)image_width;
	float sy = net_height / (float)image_height;
	float s = sx < sy ? sx : sy;
	
	int rsz_width = (int)(s * image_width);
	int rsz_height = (int)(s * image_height);

	b.x = (b.x - (net_width - rsz_width) / 2.0f / net_width) * net_width / rsz_width;
	b.y = (b.y - (net_height - rsz_height) / 2.0f / net_height) * net_height / rsz_height;
	b.w = b.w * net_width / rsz_width;
	b.h = b.h * net_height / rsz_height;
	
	return b;
}

float *get_yolo_prob(float *prob_volume, int id, int layer_width, int layer_height,
                     int classes, float objectness)
{
	float *probabilities = list_alloc_mem(classes * sizeof(float));
	if (!probabilities) {
		fprintf(stderr, "calloc[%s:%d].\n", __FILE__, __LINE__);
		return probabilities;
	}
	
	int slice_size = layer_width * layer_height;
	for (int i = 0; i < classes; ++i) {
		probabilities[i] = objectness * prob_volume[id + slice_size * i];
	}
	
	return probabilities;
}