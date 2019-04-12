#ifndef _MAXPOOL_LAYER_H_
#define _MAXPOOL_LAYER_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include "znet.h"
#include "zutils.h"

#ifdef OPENCL
struct maxpool_gpu_context;
typedef struct maxpool_gpu_context maxpool_gpu_context;
#endif

typedef struct {
	LAYER_TYPE type;
	dim3 input_size;
	dim3 output_size;
	int filter_size;
	int stride;
	int padding;
	int batch_size;
	int ninputs;
	int noutputs;
	float *input;
	float *output;
#ifdef OPENCL
	maxpool_gpu_context *mpgc;
#endif
} maxpool_layer;

AICORE_LOCAL void free_maxpool_layer(void *_layer);
AICORE_LOCAL void print_maxpool_layer_info(void *_layer, int id);
AICORE_LOCAL void set_maxpool_layer_input(void *_layer, void *input);
AICORE_LOCAL void *get_maxpool_layer_output(void *_layer);
AICORE_LOCAL void forward_maxpool_layer(void *_layer, znet *net);
AICORE_LOCAL void backward_maxpool_layer(maxpool_layer *layer, znet *net);
AICORE_LOCAL int maxpool_output_width(maxpool_layer *layer);
AICORE_LOCAL int maxpool_output_height(maxpool_layer *layer);

#ifdef __cplusplus
}
#endif

#endif