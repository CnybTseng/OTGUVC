#ifdef __ARM_NEON__
#ifndef _NEON_MATH_H_
#define _NEON_MATH_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <arm_neon.h>
#include "zutils.h"

typedef float32x4_t v4sf;

AICORE_LOCAL v4sf log_ps(v4sf x);
AICORE_LOCAL v4sf exp_ps(v4sf x);
AICORE_LOCAL v4sf sin_ps(v4sf x);
AICORE_LOCAL v4sf cos_ps(v4sf x);

#ifdef __cplusplus
}
#endif

#endif
#endif