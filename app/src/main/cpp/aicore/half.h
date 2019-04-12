#ifndef _HALF_H_
#define _HALF_H_

#ifdef __cplusplus
extern "C"
{
#endif

#ifdef OPENCL
#include "CL/opencl.h"
#include "zutils.h"

#ifdef USE_FLOAT
#	define HOST_TO_DEVICE(val) val
#	define DEVICE_TO_HOST(val) val
#else
#	define HOST_TO_DEVICE(val) to_half(val)
#	define DEVICE_TO_HOST(val) to_float(val)
#endif

AICORE_LOCAL cl_half to_half(float f);
AICORE_LOCAL float to_float(cl_half h);

#endif

#ifdef __cplusplus
}
#endif

#endif
