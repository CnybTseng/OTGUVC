#ifndef _BATCHNORM_LAYER_H_
#define _BATCHNORM_LAYER_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include "znet.h"
#include "zutils.h"

AICORE_LOCAL void normalize(float *X, float *mean, float *variance, int batch_size, int nchannels, int size);
AICORE_LOCAL void forward_batchnorm_layer(void *layer, znet *net);
AICORE_LOCAL void backward_batchnorm_layer(void *layer, znet *net);

#ifdef __cplusplus
}
#endif

#endif