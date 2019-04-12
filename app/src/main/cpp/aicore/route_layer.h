#ifndef _ROUTE_LAYER_H_
#define _ROUTE_LAYER_H_

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
	dim3 output_size;
	int batch_size;
	int ninputs;
	int noutputs;
	int nroutes;
	int *input_layers;
	int *input_sizes;
	float *output;
#ifdef OPENCL
	cl_mem d_output;
#endif
} route_layer;

AICORE_LOCAL void free_route_layer(void *_layer);
AICORE_LOCAL void print_route_layer_info(void *_layer, int id);
AICORE_LOCAL void set_route_layer_input(void *_layer, void *input);
AICORE_LOCAL void *get_route_layer_output(void *_layer);
AICORE_LOCAL void forward_route_layer(void *_layer, znet *net);
AICORE_LOCAL void backward_route_layer(route_layer *layer, znet *net);

#ifdef __cplusplus
}
#endif

#endif