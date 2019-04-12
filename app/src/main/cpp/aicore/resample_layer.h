#ifndef _RESAMPLE_H_
#define _RESAMPLE_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include "znet.h"
#ifdef OPENCL
#	include "cl_wrapper.h"
#endif
#include "zutils.h"

#ifdef OPENCL
struct resample_context;
typedef struct resample_context resample_context;
#endif

typedef struct {
	LAYER_TYPE type;
	dim3 input_size;
	dim3 output_size;
	int stride;
	int batch_size;
	int ninputs;
	int noutputs;
	int upsample;
	float *input;
	float *output;
#ifdef OPENCL
	resample_context *rc;
#endif
} resample_layer;

AICORE_LOCAL void free_resample_layer(void *_layer);
AICORE_LOCAL void print_resample_layer_info(void *_layer, int id);
AICORE_LOCAL void set_resample_layer_input(void *_layer, void *input);
AICORE_LOCAL void *get_resample_layer_output(void *_layer);
AICORE_LOCAL void forward_resample_layer(void *_layer, znet *net);
AICORE_LOCAL void backward_resample_layer(resample_layer *layer, znet *net);
AICORE_LOCAL void upsample(float *in, int width, int height, int nchannels, int stride, float *out);
#ifdef OPENCL
AICORE_LOCAL void get_resample_output_image_size(resample_layer *layer, int *width, int *height);
#endif

#ifdef __cplusplus
}
#endif

#endif