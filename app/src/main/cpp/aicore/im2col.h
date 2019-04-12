#ifndef _IM2COL_H_
#define _IM2COL_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include "zutils.h"

AICORE_LOCAL void im2col_cpu(float *image, int width, int height, int channels, int fsize,
                int stride, int padding, float *matrix);

#ifdef __cplusplus
}
#endif

#endif