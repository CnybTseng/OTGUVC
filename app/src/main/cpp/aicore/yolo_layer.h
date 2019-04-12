#ifndef _YOLO_LAYER_H_
#define _YOLO_LAYER_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include "znet.h"
#ifdef OPENCL
#	include "cl_wrapper.h"
#endif
#include "zutils.h"

typedef struct {
	LAYER_TYPE type;
	dim3 input_size;
	dim3 output_size;
	int batch_size;
	int ninputs;
	int noutputs;
	int nscales;
	int total_scales;
	int classes;
	int *mask;
	int *anchor_boxes;
	float *input;
	float *output;
#ifdef OPENCL
	cl_mem d_input;
#endif
} yolo_layer;

AICORE_LOCAL void free_yolo_layer(void *_layer);
AICORE_LOCAL void print_yolo_layer_info(void *_layer, int id);
AICORE_LOCAL void set_yolo_layer_input(void *_layer, void *input);
AICORE_LOCAL void *get_yolo_layer_output(void *_layer);
AICORE_LOCAL void forward_yolo_layer(void *_layer, znet *net);
AICORE_LOCAL void get_yolo_layer_detections(yolo_layer *layer, znet *net, int imgw, int imgh, float thresh, list *l);
AICORE_LOCAL void free_yolo_layer_detections(list *l);
AICORE_LOCAL void backward_yolo_layer(yolo_layer *layer, znet *net);

#ifdef __cplusplus
}
#endif

#endif